import data_scraping as ds
import data_uploading as du
import pandas as pd
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
SA_JSON = "/Users/agamjotsandhu/Desktop/Random projects/Premier Cricket Analysis/credentials.json"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1gTbhS70aDUPKOhNJvlEM5TogjcGmvUb4-iSrMV_yhAY/edit"
CACHE_FILE = "scraped_matches.json"
MAX_WORKERS = 4
FORCE_RESCRAPE = True

grade_links = [
    "https://play.cricket.com.au/grade/4069a25a-cf4d-4ebb-8190-5f7a4cabd4d7?tab=matches",  # Firsts
    "https://play.cricket.com.au/grade/2226702c-82de-456a-b0ad-17e31e94a291?tab=matches",  # Seconds
    "https://play.cricket.com.au/grade/d9da39c6-9180-4257-8a00-a65450a548fb?tab=matches",  # Thirds
    "https://play.cricket.com.au/grade/31787ef0-778b-47d7-8126-68dae30baa0f?tab=matches",  # Fourths
]

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(list(cache), f)

def scrape_one(job):
    round_num, grade_link, match_url = job
    driver = ds.build_driver()
    try:
        results = ds.scrape_match_data(driver, match_url)
        if not results:
            return [], [], match_url
        
        over_rows, score_rows = results
        
        # Prepend round number to every row
        over_rows = [[round_num] + row for row in over_rows]
        score_rows = [[round_num] + row for row in score_rows]
        
        return over_rows, score_rows, match_url
    finally:
        driver.quit()

def main():
    start_time = time.time()
    
    cache = load_cache()
    if FORCE_RESCRAPE:
        cache = set()
    
    grade_round_index = {}
    print("Scanning grades for rounds and matches...")
    with ds.build_driver() as scout:
        for grade_link in grade_links:
            if not grade_link.startswith("http"):
                continue
            rounds = ds.scan_grade_all_rounds(scout, grade_link)
            grade_round_index[grade_link] = rounds
            print(f"Found {len(rounds)} rounds in {grade_link}")

    jobs = []
    skipped_count = 0
    for grade_link, rounds in grade_round_index.items():
        for round_num, matches in rounds.items():
            for _match_name, match_url in matches:
                if match_url in cache:
                    skipped_count += 1
                    continue
                jobs.append((round_num, grade_link, match_url))

    print(f"Total jobs to process: {len(jobs)} (Skipped {skipped_count} cached matches)")

    master_rows_overs = []
    master_rows_scores = []
    
    scraped_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(scrape_one, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                overs, scores, match_url = fut.result()
                if overs or scores:
                    master_rows_overs.extend(overs)
                    master_rows_scores.extend(scores)
                    scraped_count += 1
                
                # Update cache
                cache.add(match_url)
                save_cache(cache)
                print(f"Done: {match_url}")
            except Exception as e:
                print(f"Error processing job: {e}")

    if not master_rows_overs and not master_rows_scores:
        print("No new data scraped.")
    else:
        # Build DataFrames
        cols_over = ["round", "grade", "matchup", "opposition", "current_batting", "over_num", "bowler_for_row", "runs", "wickets"]
        master_df_overs = pd.DataFrame(master_rows_overs, columns=cols_over)

        cols_score = ["round", "grade", "matchup", "opposition", "batting_team", "total_runs", "total_wickets"]
        master_df_scores = pd.DataFrame(master_rows_scores, columns=cols_score)

        # Export
        master_df_overs.to_csv("overs.csv", index=False)
        master_df_scores.to_csv("scores.csv", index=False)

        # Upload
        print("Uploading to Google Sheets...")
        du.upload_csv_append_overs(SA_JSON, SPREADSHEET_URL, "overs.csv")
        du.upload_csv_append_scores(SA_JSON, SPREADSHEET_URL, "scores.csv")

    end_time = time.time()
    duration = end_time - start_time
    print(f"\nSummary:")
    print(f"Total matches scraped: {scraped_count}")
    print(f"Total skipped (cache hits): {skipped_count}")
    print(f"Wall-clock time: {duration:.2f} seconds")

if __name__ == "__main__":
    main()
