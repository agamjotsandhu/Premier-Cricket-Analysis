import scrape_individual_performances as sip
import data_scraping as ds
import data_uploading as du
import pandas as pd
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------
# Constants  (mirror main.py — edit as needed)
# ------------------------------------------------------------------
SA_JSON          = "credentials.json"
SPREADSHEET_URL  = "https://docs.google.com/spreadsheets/d/1gTbhS70aDUPKOhNJvlEM5TogjcGmvUb4-iSrMV_yhAY/edit"
CACHE_FILE       = "scraped_individual_matches.json"   # separate cache from overs scrape
MAX_WORKERS      = 4
FORCE_RESCRAPE   = False

BATTING_CSV  = "batting_data.csv"
BOWLING_CSV  = "bowling_data.csv"

grade_links = [
    "https://play.cricket.com.au/grade/4069a25a-cf4d-4ebb-8190-5f7a4cabd4d7?tab=matches",  # Firsts
    "https://play.cricket.com.au/grade/2226702c-82de-456a-b0ad-17e31e94a291?tab=matches",  # Seconds
    "https://play.cricket.com.au/grade/d9da39c6-9180-4257-8a00-a65450a548fb?tab=matches",  # Thirds
    "https://play.cricket.com.au/grade/31787ef0-778b-47d7-8126-68dae30baa0f?tab=matches",  # Fourths
]

# ------------------------------------------------------------------
# Column schemas
# ------------------------------------------------------------------
BATTING_COLS = [
    "round", "grade", "matchup", "opposition", "batting_team",
    "batsman_name", "runs", "balls", "fours", "sixes", "how_out"
]

BOWLING_COLS = [
    "round", "grade", "matchup", "opposition", "bowling_team",
    "bowler_name", "overs", "maidens", "runs", "wickets", "wides", "no_balls"
]

# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------
def load_cache() -> set:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_cache(cache: set):
    with open(CACHE_FILE, "w") as f:
        json.dump(list(cache), f)

# ------------------------------------------------------------------
# Upload helpers  (mirrors data_uploading.py pattern)
# ------------------------------------------------------------------
def upload_batting(csv_path: str):
    du.upload_csv_append(SA_JSON, SPREADSHEET_URL, csv_path, sheet_name="batting_data")

def upload_bowling(csv_path: str):
    du.upload_csv_append(SA_JSON, SPREADSHEET_URL, csv_path, sheet_name="bowling_data")


# data_uploading.py doesn't have a generic function yet, so we define one inline
# that mirrors upload_csv_append_overs exactly.
import re as _re
import csv as _csv
from google.oauth2.service_account import Credentials as _Creds
from googleapiclient.discovery import build as _build

def _upload_csv_to_sheet(sa_json_path: str, spreadsheet_url: str,
                          csv_path: str, sheet_name: str):
    """Generic CSV → Google Sheets append, used for batting_data and bowling_data."""
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds  = _Creds.from_service_account_file(sa_json_path, scopes=scopes)
    service = _build("sheets", "v4", credentials=creds)

    m = _re.search(r"/d/([a-zA-Z0-9-_]+)", spreadsheet_url)
    if not m:
        raise ValueError("Could not parse spreadsheet id from URL")
    spreadsheet_id = m.group(1)

    # Ensure sheet exists
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_ids = {s["properties"]["title"] for s in meta.get("sheets", [])}
    if sheet_name not in sheet_ids:
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": [{"addSheet": {"properties": {
                "title": sheet_name,
                "gridProperties": {"rowCount": 20000, "columnCount": 30}
            }}}]}
        ).execute()

    # Find last filled row
    def _a1(row: int, col: int) -> str:
        letters = ""
        c = col
        while c > 0:
            c, rem = divmod(c - 1, 26)
            letters = chr(65 + rem) + letters
        return f"{letters}{row}"

    resp = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"'{sheet_name}'!A1:{_a1(20000, 30)}"
    ).execute()
    last_row = len(resp.get("values", []))
    start_row = last_row + 1 if last_row > 0 else 1

    # Load and upload CSV in chunks
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(_csv.reader(f))

    if not rows:
        print(f"CSV '{csv_path}' is empty — nothing to upload.")
        return

    CHUNK = 1000
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        start_a1 = _a1(start_row + i, 1)
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"'{sheet_name}'!{start_a1}",
            valueInputOption="USER_ENTERED",
            body={"values": chunk},
        ).execute()

    print(f"Uploaded {len(rows)} rows to sheet '{sheet_name}'.")


# ------------------------------------------------------------------
# Per-match worker  (runs in a subprocess)
# ------------------------------------------------------------------
def scrape_one(job):
    round_num, _grade_link, match_url = job
    driver = ds.build_driver()
    try:
        batting_rows, bowling_rows = sip.scrape_scorecard_data(driver, match_url)

        # Prepend round number
        batting_rows = [[round_num] + row for row in batting_rows]
        bowling_rows = [[round_num] + row for row in bowling_rows]

        return batting_rows, bowling_rows, match_url
    finally:
        driver.quit()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    start_time = time.time()

    cache = load_cache()
    if FORCE_RESCRAPE:
        cache = set()

    # --- Scan all grade pages for round/match index ---
    grade_round_index = {}
    print("Scanning grades for rounds and matches...")
    scout = ds.build_driver()
    try:
        for grade_link in grade_links:
            if not grade_link.startswith("http"):
                continue
            rounds = ds.scan_grade_all_rounds(scout, grade_link)
            grade_round_index[grade_link] = rounds
            print(f"  Found {len(rounds)} rounds in {grade_link}")
    finally:
        scout.quit()

    # --- Build job list (skip cached) ---
    jobs = []
    skipped_count = 0
    for grade_link, rounds in grade_round_index.items():
        for round_num, matches in rounds.items():
            for _match_name, match_url in matches:
                if match_url in cache:
                    skipped_count += 1
                    continue
                jobs.append((round_num, grade_link, match_url))

    print(f"Jobs to process: {len(jobs)}  |  Skipped (cached): {skipped_count}")

    # --- Parallel scrape ---
    master_batting: list = []
    master_bowling: list = []
    scraped_count = 0
    error_count   = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(scrape_one, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            match_url = job[2]
            try:
                batting, bowling, url = fut.result()
                if batting or bowling:
                    master_batting.extend(batting)
                    master_bowling.extend(bowling)
                    scraped_count += 1
                # Always cache so we don't retry on next run
                cache.add(url)
                save_cache(cache)
                print(f"  Done: {url}")
            except Exception as e:
                error_count += 1
                print(f"  Error ({match_url}): {e}")

    # --- Export CSVs ---
    if not master_batting and not master_bowling:
        print("No new data scraped.")
    else:
        df_batting = pd.DataFrame(master_batting, columns=BATTING_COLS)
        df_bowling = pd.DataFrame(master_bowling, columns=BOWLING_COLS)

        df_batting.to_csv(BATTING_CSV, index=False)
        df_bowling.to_csv(BOWLING_CSV, index=False)
        print(f"Exported {len(df_batting)} batting rows  → {BATTING_CSV}")
        print(f"Exported {len(df_bowling)} bowling rows  → {BOWLING_CSV}")

        # --- Upload to Google Sheets ---
        print("Uploading to Google Sheets...")
        _upload_csv_to_sheet(SA_JSON, SPREADSHEET_URL, BATTING_CSV, "batting_data")
        _upload_csv_to_sheet(SA_JSON, SPREADSHEET_URL, BOWLING_CSV, "bowling_data")

    duration = time.time() - start_time
    print(f"\nSummary:")
    print(f"  Matches scraped : {scraped_count}")
    print(f"  Skipped (cache) : {skipped_count}")
    print(f"  Errors          : {error_count}")
    print(f"  Wall-clock time : {duration:.2f}s")


if __name__ == "__main__":
    main()
