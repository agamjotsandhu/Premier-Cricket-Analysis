# dataUploading.py
import re
import csv
from typing import List
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


def build_service(sa_json_path: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(sa_json_path, scopes=scopes)
    return build("sheets", "v4", credentials=creds)


def extract_spreadsheet_id(spreadsheet_url: str) -> str:
    """
    Extract the spreadsheet ID from a full Google Sheets URL.
    """
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", spreadsheet_url)
    if not m:
        raise ValueError("Could not parse spreadsheet id from URL")
    return m.group(1)


def a1(row: int, col: int) -> str:
    """
    Convert 1-based (row, col) to an A1-style cell reference.
    """
    letters = ""
    c = col
    while c > 0:
        c, rem = divmod(c - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"


def ensure_sheet(service, spreadsheet_id: str, title: str) -> int:
    """
    Ensure a sheet with the given title exists.
    Return its sheetId (creating it if needed).
    """
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for s in meta.get("sheets", []):
        if s["properties"]["title"] == title:
            return s["properties"]["sheetId"]

    body = {
        "requests": [
            {
                "addSheet": {
                    "properties": {
                        "title": title,
                        "gridProperties": {
                            "rowCount": 2000,
                            "columnCount": 50,
                        },
                    }
                }
            }
        ]
    }
    resp = service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id, body=body
    ).execute()
    return resp["replies"][0]["addSheet"]["properties"]["sheetId"]


def find_last_filled_row(service, spreadsheet_id: str, sheet_name: str,
                         max_cols: int = 50, max_rows: int = 20000) -> int:
    """
    Returns the last 1-based row index that has any value in A:Z (or up to max_cols),
    or 0 if the sheet is empty.
    """
    end_a1 = a1(max_rows, max_cols)
    resp = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"'{sheet_name}'!A1:{end_a1}"
    ).execute()
    grid = resp.get("values", [])
    return len(grid)


def values_update(service, spreadsheet_id: str, sheet_name: str,
                  start_a1: str, values: List[List[str]]):
    """
    Write `values` to the sheet starting at `start_a1`.
    """
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{sheet_name}'!{start_a1}",
        valueInputOption="USER_ENTERED",
        body={"values": values},
    ).execute()


def upload_csv_append_ballbyball(sa_json_path: str, spreadsheet_url: str, csv_path: str):
    """
    Append all rows from CSV into a sheet called 'ball-by-ball'.
    Creates the sheet if missing. Does NOT clear existing data.
    """
    service = build_service(sa_json_path)
    spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)

    sheet_name = "ball-by-ball"
    ensure_sheet(service, spreadsheet_id, sheet_name)

    # Find the last filled row to know where to append
    last_row = find_last_filled_row(service, spreadsheet_id, sheet_name, max_cols=50)
    start_row = last_row + 1 if last_row > 0 else 1

    # Load CSV
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("CSV is empty; nothing to upload.")
        return

    # Write all rows starting at the first empty line
    CHUNK = 1000
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        start_a1 = a1(start_row + i, 1)
        values_update(service, spreadsheet_id, sheet_name, start_a1, chunk)

    print(f"Appended {len(rows)} rows to sheet '{sheet_name}' in spreadsheet {spreadsheet_id}.")
    


def upload_csv_append_overs(sa_json_path: str, spreadsheet_url: str, csv_path: str):
    service = build_service(sa_json_path)
    spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)

    sheet_name = "overs"
    ensure_sheet(service, spreadsheet_id, sheet_name)

    # Find the last filled row to know where to append
    last_row = find_last_filled_row(service, spreadsheet_id, sheet_name, max_cols=50)
    start_row = last_row + 1 if last_row > 0 else 1

    # Load CSV
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("CSV is empty; nothing to upload.")
        return

    # Write all rows starting at the first empty line
    CHUNK = 1000
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        start_a1 = a1(start_row + i, 1)
        values_update(service, spreadsheet_id, sheet_name, start_a1, chunk)

    print(f"Appended {len(rows)} rows to sheet '{sheet_name}' in spreadsheet {spreadsheet_id}.")



def upload_csv_append_scores(sa_json_path: str, spreadsheet_url: str, csv_path: str):
    service = build_service(sa_json_path)
    spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)

    sheet_name = "scores"
    ensure_sheet(service, spreadsheet_id, sheet_name)

    # Find the last filled row to know where to append
    last_row = find_last_filled_row(service, spreadsheet_id, sheet_name, max_cols=50)
    start_row = last_row + 1 if last_row > 0 else 1

    # Load CSV
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("CSV is empty; nothing to upload.")
        return

    # Write all rows starting at the first empty line
    CHUNK = 1000
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        start_a1 = a1(start_row + i, 1)
        values_update(service, spreadsheet_id, sheet_name, start_a1, chunk)

    print(f"Appended {len(rows)} rows to sheet '{sheet_name}' in spreadsheet {spreadsheet_id}.")