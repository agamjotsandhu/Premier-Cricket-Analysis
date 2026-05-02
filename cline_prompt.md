# Cline task: scrape every match in the configured grades, and make it fast

## Context
This is a Selenium-based cricket scraper. Today it only collects matches involving Essendon (because the grade URLs include a `teamId` filter and the parsing code hardcodes `"ESS"` as "us"). Three CSVs are produced per round and appended to a Google Sheet: `ball_by_ball.csv`, `overs.csv`, `scores.csv`.

**Files to touch:**
- `main.py` — orchestrator
- `data_scraping.py` — Selenium scraping logic
- *(do not change)* `data_augmentation.py`, `data_uploading.py`

**Goal:** scrape every match in the configured grades for the whole season, and cut total runtime substantially.

---

## Part 1 — Functional changes (scrape every team)

### 1.1 Remove the team filter from grade URLs
In `main.py`, replace the current `grade_links` with the bare grade URLs (drop the `&teamId=...` query param). Keep the same set of grades the user already had, including the commented-out ones — leave those commented:

```python
grade_links = [
    "https://play.cricket.com.au/grade/4069a25a-cf4d-4ebb-8190-5f7a4cabd4d7?tab=matches",  # Firsts  (~21 rounds)
    "https://play.cricket.com.au/grade/2226702c-82de-456a-b0ad-17e31e94a291?tab=matches",  # Seconds (~21 rounds)
    "https://play.cricket.com.au/grade/d9da39c6-9180-4257-8a00-a65450a548fb?tab=matches",  # Thirds  (fewer rounds)
    "https://play.cricket.com.au/grade/31787ef0-778b-47d7-8126-68dae30baa0f?tab=matches",  # Fourths (fewer rounds)
]
```

Round counts differ across these grades — Firsts and Seconds run a full home-and-away + finals series (~21 rounds), while Thirds and Fourths have fewer. Do **not** treat the round count as fixed; see §1.3.

### 1.2 Replace the hardcoded `"ESS"` opposition logic
In `data_scraping.py`, `scrape_ball_by_ball` currently does:

```python
opposition = next(team for team in re.findall(r"\b[A-Z]{2,3}\b", matchup) if team != "ESS")
```

That assumes Essendon is always one of the teams. Remove it. Compute opposition **per innings** instead: when `team_a` is batting, opposition is `team_b`, and vice versa.

Concretely:
- Add an `opposition: str` parameter to the inner `scrape_innings` function.
- Use that parameter wherever `opposition` is referenced inside the innings loop (ball rows and over rows).
- At the two call sites:
  - Innings A: `scrape_innings(team_a, team_b)`
  - Innings B: `scrape_innings(team_b, team_a)`
- For the score rows, use `team_b` as opposition for innings A's score row, and `team_a` for innings B's.

### 1.3 Discover rounds dynamically per grade
**Do not** hardcode a round range. The grades have different numbers of rounds — Firsts and Seconds have 21 rounds each (home-and-away + finals), while Thirds and Fourths have fewer because lower grades typically don't play the same finals series. Hardcoding `range(1, 21)` either misses rounds in the senior grades or wastes time looking for non-existent rounds in the junior grades.

The fix is structural: don't iterate round numbers at all from `main.py`. Instead, have the grade-page scan (see §2.2) return whatever rounds it finds on each grade page, and build the job list from that. Each grade contributes its own set of rounds.

Side effect of this change: the existing `for ROUND in range(12, 14):` outer loop in `main.py` and the `master_df_X.insert(0, "round", ROUND)` calls inside it must go. Round becomes a *per-job* value rather than a loop variable. The simplest pattern is:
1. Worker function receives `(round_num, grade, match_url)` as its job tuple.
2. Worker prepends `round_num` to every row it returns, so each ball/over/score row already carries its own round.
3. DataFrames are built once at the end with column lists that include `"round"` as the first column.

This naturally keeps the existing CSV column order (`round, grade, matchup, opposition, ...`).

Also, the line `print(f"Round {ROUND}, {match_names[0]}")` was crashing on empty rounds. Once round discovery is dynamic, this whole branch goes away — empty rounds simply never enter the job list. But during scanning, log a one-liner per grade like `Found N rounds in {grade_url}` so progress is visible.

---

## Part 2 — Performance changes

### 2.1 Headless Chrome with resource blocking
The module-level `driver = webdriver.Chrome()` at the top of `data_scraping.py` must go. Replace it with a factory function `build_driver()` that returns a configured driver:

- `--headless=new`
- `--disable-gpu`
- `--no-sandbox`
- `--disable-dev-shm-usage`
- `pageLoadStrategy = "eager"` (set on `Options`)
- Block images, fonts, and stylesheets via `prefs`:
  ```python
  options.add_experimental_option("prefs", {
      "profile.managed_default_content_settings.images": 2,
      "profile.managed_default_content_settings.stylesheets": 2,
      "profile.managed_default_content_settings.fonts": 2,
  })
  ```
- A reasonable `--window-size=1280,1024` (some sites collapse layout below a width threshold; if the existing CSS selectors stop matching after blocking stylesheets, fall back to leaving stylesheets enabled).

Every existing function that uses the global `driver` must be refactored to take a `driver` parameter.

### 2.2 Scan each grade page exactly once
`round_match_links_getter` currently does `driver.get(LINK)` on every call. With multiple rounds × multiple grades that's many redundant page loads.

Replace it with a function `scan_grade_all_rounds(driver, grade_url) -> dict[int, list[tuple[str, str]]]` that:
- Calls `driver.get(grade_url)` once.
- Iterates every `li.w-fixtures-listing__item` and reads the header text from `h3.w-fixtures-listing__section-header span`.
- **Extracts the round number with a regex** (`re.search(r"Round\s+(\d+)", header, flags=re.I)`) rather than literal string equality. Sections whose header doesn't match (e.g. unrelated headings) are skipped silently.
- Collects all `(match_name, match_url)` pairs under each matched round.
- Returns a dict keyed by round number. The dict size will naturally differ per grade (Firsts/Seconds ≈ 21 entries, Thirds/Fourths fewer), and that's fine — downstream code consumes whatever rounds are present.

In `main.py`, scan all grades up front:

```python
grade_round_index = {}  # grade_url -> {round_num: [(name, url), ...]}
with build_driver() as scout:
    for grade_link in grade_links:
        rounds = ds.scan_grade_all_rounds(scout, grade_link)
        grade_round_index[grade_link] = rounds
        print(f"Found {len(rounds)} rounds in {grade_link}")
```

Then flatten into jobs (round numbers come from whatever was discovered, not a fixed range):

```python
jobs = []
for grade_link, rounds in grade_round_index.items():
    for round_num, matches in rounds.items():
        for _match_name, match_url in matches:
            jobs.append((round_num, grade_link, match_url))
```

### 2.3 Parallel match scraping
This is the biggest single win. Use `concurrent.futures.ProcessPoolExecutor` (**not** `ThreadPoolExecutor` — a single Selenium driver is not safe to share across threads, and the GIL would limit gains anyway).

- Add a top-level constant `MAX_WORKERS = 4` in `main.py` (configurable, but default 4 — be polite to the site).
- Use the flat `jobs` list built in §2.2.
- Worker function: `def scrape_one(job) -> tuple[list, list, list, str]` where the first three are ball_rows, over_rows, score_rows and the fourth is the `match_url` (used for cache writeback, see §2.4). Each worker:
  1. Unpacks `round_num, grade, match_url = job`.
  2. Creates its own driver via `build_driver()`.
  3. Calls `scrape_ball_by_ball(driver, match_url)`.
  4. **Prepends `round_num` to every row** in each of the three returned lists, so `round` ends up as the first column of every CSV row.
  5. Quits the driver in a `finally` block.
- Aggregate all rows across all jobs in the parent process. Build the three DataFrames with column lists starting `["round", ...]` (matching the existing CSV schema). Write CSVs and upload to Sheets **once, at the end of the run** — not per round.

Pattern:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = [ex.submit(scrape_one, job) for job in jobs]
    for fut in as_completed(futures):
        balls, overs, scores, match_url = fut.result()
        master_rows_ball_by_ball.extend(balls)
        master_rows_overs.extend(overs)
        master_rows_scores.extend(scores)
        # cache writeback (see §2.4)
```

### 2.4 Skip already-scraped matches (cache)
Completed matches don't change. Persist a cache of match URLs already scraped so reruns don't redo them.

- File: `scraped_matches.json` next to `main.py`. Format: a JSON list of strings (match URLs).
- Load at the start of `main.py`. Filter the jobs list to exclude URLs already in the cache.
- After each successful future result, append its URL to the cache and write the file. Use a small lock or a simple "rewrite the file each time" approach — the cache is tiny.
- Add a `FORCE_RESCRAPE = False` constant near the top of `main.py` so the user can flip it to `True` to ignore the cache.

---

## Constraints (do not change these)

- **CSV schemas** for `ball_by_ball.csv`, `overs.csv`, `scores.csv` must stay identical (same column names, same order). `data_augmentation.py` and `data_uploading.py` depend on them.
- Keep all the existing helpers intact: `extract_runs_to_batsman`, `extract_wides`, `extract_noballs`, `extract_byes`, `extract_leg_byes`, `extract_dismissal_type`, `extract_bowler`.
- Keep `data_augmentation.add_ball_by_ball_metrics` and `add_wickets_down` unchanged. Continue calling `add_ball_by_ball_metrics` after assembling the per-round (or per-run) DataFrame.
- Keep the Google Sheets append flow (`upload_csv_append_ballbyball`, `upload_csv_append_overs`, `upload_csv_append_scores`) intact. If you batch the upload to once-per-run instead of once-per-round, that's fine, but the sheets receiving the data must still be `ball-by-ball`, `overs`, `scores`.

---

## Verification (do this before declaring done)

1. Run `python -m py_compile main.py data_scraping.py data_augmentation.py data_uploading.py` — must pass.
2. Grep the codebase for `"ESS"` and the `teamId=` substring — must return zero hits.
3. Grep for `webdriver.Chrome(` — must appear only inside `build_driver()`.
4. Grep for `driver.get(` — count the call sites and confirm there are exactly:
   - one inside `scan_grade_all_rounds`
   - one inside `scrape_ball_by_ball`
   (no others.)
5. Smoke test: in `main.py`, temporarily slice the job list to a handful of matches (e.g. `jobs = jobs[:4]`) and set `MAX_WORKERS = 2`. Run end-to-end and confirm:
   - `ball_by_ball.csv`, `overs.csv`, `scores.csv` are written with the original column headers (starting with `round`).
   - The Google Sheet receives appended rows.
   - Each match URL was hit exactly once and is now present in `scraped_matches.json`.
6. Remove the slice and the `MAX_WORKERS` override before handing back. Also confirm that the per-grade scan logged round counts that look right — Firsts and Seconds should report ~21 rounds, Thirds and Fourths fewer.

Print a short summary at the end of the run: total matches scraped, total skipped (cache hits), and wall-clock time.
