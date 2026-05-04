from __future__ import annotations

import csv
import time
from typing import List

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import Libraries.data_scraping as ds
import Libraries.data_uploading as du

# ---------------------------------------------------------------------------
# Grade config
# ---------------------------------------------------------------------------
GRADE_IDS = {
    "1st": "4069a25a-cf4d-4ebb-8190-5f7a4cabd4d7",
    "2nd": "2226702c-82de-456a-b0ad-17e31e94a291",
    "3rd": "d9da39c6-9180-4257-8a00-a65450a548fb",
    "4th": "31787ef0-778b-47d7-8126-68dae30baa0f",
}

BASE_URL = "https://play.cricket.com.au/grade/{grade_id}?tab=stats&category={category}"

SA_JSON = "credentials.json"
SPREADSHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1gTbhS70aDUPKOhNJvlEM5TogjcGmvUb4-iSrMV_yhAY/edit"
)

# The site only exposes 2 stat <td>s per row regardless of category.
# Batting:  runs | avg
# Bowling:  wkts | avg
BATTING_COLS = ["rank", "player", "club", "grade", "runs", "avg"]
BOWLING_COLS = ["rank", "player", "club", "grade", "wkts", "avg"]


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

def _wait_for_table(driver, timeout: int = 15):
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "tr.w-play-competition-stats__table-row")
        )
    )
    time.sleep(1)


def _scrape_stats_page(driver, url: str, grade_label: str) -> List[List[str]]:
    """
    Scrape a play.cricket.com.au stats leaderboard page.

    Each row's DOM:
      <tr class="w-play-competition-stats__table-row ...">
        <th class="w-play-competition-stats__player-cell ...">
          <div class="...player-index">1</div>
          <span class="...player-name--name">Hennig, Matthew</span>
          <span class="...player-name--club">St Kilda Cricket Club</span>
        </th>
        <td class="w-play-competition-stats__table-cell ...">41</td>   <!-- runs or wkts -->
        <td class="w-play-competition-stats__table-cell ...">20.15</td> <!-- avg -->
      </tr>
    """
    driver.get(url)
    _wait_for_table(driver)

    rows: List[List[str]] = []

    table_rows = driver.find_elements(
        By.CSS_SELECTOR, "tr.w-play-competition-stats__table-row"
    )

    for tr in table_rows:
        # --- rank / player / club from <th> ---
        try:
            th = tr.find_element(By.CSS_SELECTOR, "th.w-play-competition-stats__player-cell")
        except Exception:
            continue

        rank = ""
        try:
            rank = th.find_element(
                By.CSS_SELECTOR, ".w-play-competition-stats__player-index"
            ).text.strip()
        except Exception:
            pass

        player = ""
        try:
            player = th.find_element(
                By.CSS_SELECTOR, ".w-play-competition-stats__player-name--name"
            ).text.strip()
        except Exception:
            pass

        club = ""
        try:
            club = th.find_element(
                By.CSS_SELECTOR, ".w-play-competition-stats__player-name--club"
            ).text.strip()
        except Exception:
            pass

        # --- stat values from <td>s ---
        tds = tr.find_elements(By.CSS_SELECTOR, "td.w-play-competition-stats__table-cell")
        stat_values = [td.text.strip() for td in tds]

        rows.append([rank, player, club, grade_label] + stat_values)

    return rows


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def scrape_batting_stats(driver, grade_label: str, grade_id: str) -> List[List[str]]:
    url = BASE_URL.format(grade_id=grade_id, category="batting")
    print(f"  GET {url}")
    return _scrape_stats_page(driver, url, grade_label)


def scrape_bowling_stats(driver, grade_label: str, grade_id: str) -> List[List[str]]:
    url = BASE_URL.format(grade_id=grade_id, category="bowling")
    print(f"  GET {url}")
    return _scrape_stats_page(driver, url, grade_label)


# ---------------------------------------------------------------------------
# Upload helper — always clears then rewrites (season totals, not appended)
# ---------------------------------------------------------------------------

def _upload_champion_tab(csv_path: str, sheet_name: str):
    service = du.build_service(SA_JSON)
    spreadsheet_id = du.extract_spreadsheet_id(SPREADSHEET_URL)
    du.ensure_sheet(service, spreadsheet_id, sheet_name)

    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"'{sheet_name}'!A1:Z5000",
    ).execute()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        print(f"No data to upload for '{sheet_name}'.")
        return

    CHUNK = 1000
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i : i + CHUNK]
        start_a1 = du.a1(i + 1, 1)
        du.values_update(service, spreadsheet_id, sheet_name, start_a1, chunk)

    print(f"Uploaded {len(rows)} rows to '{sheet_name}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_batting: List[List[str]] = [BATTING_COLS]
    all_bowling: List[List[str]] = [BOWLING_COLS]

    driver = ds.build_driver()
    try:
        for grade_label, grade_id in GRADE_IDS.items():
            print(f"Scraping batting — {grade_label}s ...")
            rows = scrape_batting_stats(driver, grade_label, grade_id)
            print(f"  → {len(rows)} rows")
            all_batting.extend(rows)

            print(f"Scraping bowling — {grade_label}s ...")
            rows = scrape_bowling_stats(driver, grade_label, grade_id)
            print(f"  → {len(rows)} rows")
            all_bowling.extend(rows)
    finally:
        driver.quit()

    batting_csv = "batting_champions.csv"
    bowling_csv = "bowling_champions.csv"

    with open(batting_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(all_batting)
    print(f"Saved {batting_csv}")

    with open(bowling_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(all_bowling)
    print(f"Saved {bowling_csv}")

    print("Uploading to Google Sheets...")
    _upload_champion_tab(batting_csv, "batting champions")
    _upload_champion_tab(bowling_csv, "bowling champions")

    print("Done.")


if __name__ == "__main__":
    main()
