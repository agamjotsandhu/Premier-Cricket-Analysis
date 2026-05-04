from __future__ import annotations

import csv
import time
from typing import List

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import data_scraping as ds
import data_uploading as du

# ---------------------------------------------------------------------------
# Grade config
# ---------------------------------------------------------------------------
GRADE_IDS = {
    "1st": "4069a25a-cf4d-4ebb-8190-5f7a4cabd4d7",
    "2nd": "2226702c-82de-456a-b0ad-17e31e94a291",
    "3rd": "d9da39c6-9180-4257-8a00-a65450a548fb",
    "4th": "31787ef0-778b-47d7-8126-68dae30baa0f",
}

# ladder=0 is the main ladder; multi-day grades may have ladder=1 etc.
STANDINGS_URL = "https://play.cricket.com.au/grade/{grade_id}?tab=standings&ladder=0"

SA_JSON = "credentials.json"
SPREADSHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1gTbhS70aDUPKOhNJvlEM5TogjcGmvUb4-iSrMV_yhAY/edit"
)

# Columns match the <td> order visible in the DOM:
# position | team | grade | played | points | nrr | wins | losses | draws
LADDER_COLS = ["position", "team", "grade", "played", "points", "nrr", "wins", "losses", "draws"]


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def _wait_for_table(driver, timeout: int = 15):
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "tr.w-play-competition-standings__table-row")
        )
    )
    time.sleep(1)


def scrape_ladder(driver, grade_label: str, grade_id: str) -> List[List[str]]:
    """
    Scrape the standings table for one grade.

    Row DOM:
      <tr class="w-play-competition-standings__table-row ...">
        <th class="w-play-competition-standings__team-cell ...">
          <div class="...team-index">1</div>
          <span class="...team-name">St Kilda 1st XI</span>
        </th>
        <td>17</td>   <!-- played -->
        <td>71</td>   <!-- points -->
        <td>0.693</td> <!-- NRR -->
        <td>11</td>   <!-- wins -->
        <td>2</td>    <!-- losses  (hide-mobile) -->
        <td>2</td>    <!-- draws   (hide-mobile) -->
      </tr>
    """
    url = STANDINGS_URL.format(grade_id=grade_id)
    print(f"  GET {url}")
    driver.get(url)
    _wait_for_table(driver)

    rows: List[List[str]] = []

    table_rows = driver.find_elements(
        By.CSS_SELECTOR, "tr.w-play-competition-standings__table-row"
    )

    for tr in table_rows:
        # --- position and team name from <th> ---
        try:
            th = tr.find_element(By.CSS_SELECTOR, "th.w-play-competition-standings__team-cell")
        except Exception:
            continue

        position = ""
        try:
            position = th.find_element(
                By.CSS_SELECTOR, ".w-play-competition-standings__team-index"
            ).text.strip()
        except Exception:
            pass

        team = ""
        try:
            team = th.find_element(
                By.CSS_SELECTOR, ".w-play-competition-standings__team-name"
            ).text.strip()
        except Exception:
            pass

        # --- stat columns from <td>s ---
        tds = tr.find_elements(By.CSS_SELECTOR, "td.w-play-competition-standings__table-cell")
        stat_values = [td.text.strip() for td in tds]

        # Pad to 6 values (played, points, nrr, wins, losses, draws)
        # in case hide-mobile cols aren't rendered in headless
        while len(stat_values) < 6:
            stat_values.append("")

        rows.append([position, team, grade_label] + stat_values[:6])

    return rows


# ---------------------------------------------------------------------------
# Upload helper — clear and rewrite (standings change each round)
# ---------------------------------------------------------------------------

def _upload_ladder_tab(csv_path: str, sheet_name: str):
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
    all_rows: List[List[str]] = [LADDER_COLS]

    driver = ds.build_driver()
    try:
        for grade_label, grade_id in GRADE_IDS.items():
            print(f"Scraping ladder — {grade_label}s ...")
            rows = scrape_ladder(driver, grade_label, grade_id)
            print(f"  → {len(rows)} teams")
            all_rows.extend(rows)
    finally:
        driver.quit()

    csv_path = "ladder.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(all_rows)
    print(f"Saved {csv_path}")

    print("Uploading to Google Sheets...")
    _upload_ladder_tab(csv_path, "ladder")

    print("Done.")


if __name__ == "__main__":
    main()
