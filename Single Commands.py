import data_uploading as du
# Constants
SA_JSON = "credentials.json"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1gTbhS70aDUPKOhNJvlEM5TogjcGmvUb4-iSrMV_yhAY/edit"
CACHE_FILE = "scraped_matches.json"
MAX_WORKERS = 4
FORCE_RESCRAPE = False

print("Uploading to Google Sheets...")
du.upload_csv_append_overs(SA_JSON, SPREADSHEET_URL, "overs.csv")
du.upload_csv_append_scores(SA_JSON, SPREADSHEET_URL, "scores.csv")