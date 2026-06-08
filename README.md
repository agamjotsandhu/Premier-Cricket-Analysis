# Premier Cricket Match Analysis

A data pipeline and analytics project for scraping, storing, and analysing grade cricket match data from [play.cricket.com.au](https://play.cricket.com.au). Covers **1st through 4th grade** competitions across all rounds of the season.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Scrapeable Data Types](#scrapeable-data-types)
- [Data Dictionary](#data-dictionary)
- [Setup](#setup)
- [Running the Scrapers](#running-the-scrapers)
- [Analytics Notebook](#analytics-notebook)
- [Google Sheets Integration](#google-sheets-integration)

---

## Overview

The pipeline uses headless Chrome (via Selenium) to navigate match pages, extract structured data, and export it to CSV files and Google Sheets. Parallel scraping (up to 4 workers) and a JSON-based cache prevent redundant work across runs.

**Data source:** play.cricket.com.au  
**Grades covered:** 1st, 2nd, 3rd, 4th  
**Match types:** 1-day and 2-day formats (up to 4 innings)

---

## Project Structure

```
.
├── main.py                          # Scrapes over-by-over data + match scores
├── execute_individual_scraping.py   # Scrapes individual batting/bowling scorecards
├── scrape_champion_players.py       # Scrapes season batting/bowling leaderboards
├── scrape_ladder_position.py        # Scrapes grade ladder standings
│
├── Libraries/
│   ├── data_scraping.py             # Core Selenium helpers (driver, round index, ball-by-ball)
│   ├── data_uploading.py            # Google Sheets upload helpers
│   └── data_augmentation.py        # Feature engineering utilities
│
├── data/                            # Scraped CSV outputs
│   ├── overs.csv
│   ├── scores.csv
│   ├── batting_data.csv
│   ├── bowling_data.csv
│   ├── batting_champions.csv
│   └── bowling_champions.csv
│
├── notebooks/
│   └── cricket_analytics.ipynb     # Analysis and modelling notebook
│
├── outputs/                         # Generated charts and figures
├── data_dictionary.md               # Detailed column-level documentation
├── scraped_matches.json             # Cache: overs/scores scrape
├── scraped_individual_matches.json  # Cache: batting/bowling scorecard scrape
└── credentials.json                 # Google service account key (git-ignored)
```

---

## Scrapeable Data Types

The site exposes data at **five distinct granularities**. Each is scraped by a different script.

### 1. Over-by-Over Data
**Script:** `main.py`  
**Output:** `data/overs.csv`

One row per over per innings per match. Captures the scoring progression of an innings at the over level — useful for analysing run rates, bowling spells, and momentum shifts.

| Field | Description |
|-------|-------------|
| Round | Competition round number |
| Grade | 1st / 2nd / 3rd / 4th |
| Matchup | Team A vs Team B |
| Batting team / Opposition | Which side is batting/bowling |
| Over number | Over within the innings (1-indexed) |
| Bowler | Name of the bowler for that over |
| Runs scored | Total runs in that over (including extras) |
| Wickets taken | Wickets in that over |

---

### 2. Match-Level Scores
**Script:** `main.py`  
**Output:** `data/scores.csv`

One row per innings per match (two rows for a 1-day game, up to four for a 2-day game). The top-level summary of each innings.

| Field | Description |
|-------|-------------|
| Round | Competition round number |
| Grade | 1st / 2nd / 3rd / 4th |
| Matchup | Team A vs Team B |
| Batting team / Opposition | Which side batted |
| Total runs | Final innings total |
| Total wickets | Wickets lost (with `d` suffix if declared, e.g. `9d`) |

---

### 3. Individual Batting Scorecards
**Script:** `execute_individual_scraping.py`  
**Output:** `data/batting_data.csv`

One row per batter per innings per match. Scraped from the Scorecard tab. Includes players who did not bat.

| Field | Description |
|-------|-------------|
| Round, Grade, Matchup | Context identifiers |
| Batting team / Opposition | Team context |
| Batsman name | Full player name |
| Runs | Runs scored |
| Balls faced | Deliveries received |
| Fours | Boundary fours |
| Sixes | Sixes hit |
| How out | Dismissal type and fielder/bowler (e.g. `c: D Parikh b: J Sawrey`, `lbw: S Mills`, `not out`, `did not bat`) |

---

### 4. Individual Bowling Scorecards
**Script:** `execute_individual_scraping.py`  
**Output:** `data/bowling_data.csv`

One row per bowler per innings per match. Scraped from the Scorecard tab alongside batting data.

| Field | Description |
|-------|-------------|
| Round, Grade, Matchup | Context identifiers |
| Bowling team / Opposition | Team context |
| Bowler name | Full player name |
| Overs | Overs bowled (e.g. `3.2` = 3 overs and 2 balls) |
| Maidens | Maiden overs |
| Runs conceded | Total runs given away |
| Wickets | Wickets taken |
| Wides | Wide deliveries |
| No balls | No ball deliveries |

---

### 5. Season Leaderboards (Champions)
**Script:** `scrape_champion_players.py`  
**Output:** `data/batting_champions.csv`, `data/bowling_champions.csv`

Season-aggregate leaderboards from the Stats tab. These are cumulative totals across the entire season, not per-match.

**Batting leaderboard:**

| Field | Description |
|-------|-------------|
| Rank | Position on the leaderboard within grade |
| Player | Name in "Surname, First" format |
| Club | Club name |
| Grade | 1st / 2nd / 3rd / 4th |
| Runs | Season total runs |
| Average | Runs per dismissal |

**Bowling leaderboard:**

| Field | Description |
|-------|-------------|
| Rank | Position on the leaderboard within grade |
| Player | Name in "Surname, First" format |
| Club | Club name |
| Grade | 1st / 2nd / 3rd / 4th |
| Wickets | Season total wickets |
| Average | Runs conceded per wicket |

---

### 6. Ladder / Standings
**Script:** `scrape_ladder_position.py`  
**Output:** `ladder.csv`

Current competition standings snapshot for each grade. Cleared and rewritten each run since standings change every round.

| Field | Description |
|-------|-------------|
| Position | Ladder position |
| Team | Team name |
| Grade | 1st / 2nd / 3rd / 4th |
| Played | Matches played |
| Points | Competition points |
| NRR | Net run rate |
| Wins | Total wins |
| Losses | Total losses |
| Draws | Total draws |

---

## Data Dictionary

See [data_dictionary.md](data_dictionary.md) for full column-level documentation including notes on edge cases (declared innings, 2-day matches, abbreviated bowler names, etc.).

---

## Setup

### Prerequisites

- Python 3.10+
- Google Chrome installed
- ChromeDriver matching your Chrome version (or managed via `webdriver-manager`)

### Install dependencies

```bash
pip install selenium pandas google-auth google-api-python-client
```

### Google Sheets credentials

Place a service account key file at `credentials.json` in the project root. The service account must have editor access to the target spreadsheet. This file is git-ignored.

---

## Running the Scrapers

Each script is independent and can be run on its own schedule.

```bash
# Scrape over-by-over data and match scores (incremental, uses cache)
python main.py

# Scrape individual batting and bowling scorecards (incremental, uses cache)
python execute_individual_scraping.py

# Scrape season batting/bowling leaderboards (full rewrite each run)
python scrape_champion_players.py

# Scrape current ladder standings (full rewrite each run)
python scrape_ladder_position.py
```

**Caching:** `main.py` and `execute_individual_scraping.py` maintain separate JSON cache files (`scraped_matches.json` and `scraped_individual_matches.json`) so previously scraped match URLs are skipped. Set `FORCE_RESCRAPE = True` in either script to bypass the cache and re-scrape everything.

**Parallelism:** Both incremental scrapers use `ProcessPoolExecutor` with up to 4 parallel Chrome workers. Adjust `MAX_WORKERS` in the script constants if needed.

---

## Analytics Notebook

`notebooks/cricket_analytics.ipynb` contains exploratory analysis and modelling built on top of the scraped data, including:

- Strike rate vs average scatter analysis
- Dismissal pattern breakdown
- Bowling phase analysis (powerplay, middle overs, death)
- Extras and no-ball/wide profiling
- Win/loss outcome modelling (logistic regression, decision trees, random forests)
- ROC curve and permutation importance for match outcome prediction
- Random effects modelling for player consistency
- PCA batter and bowler clustering
- Spaghetti plots and slope histograms for longitudinal player performance
- Tactical decision tree analysis

---

## Google Sheets Integration

All scrapers upload their output to a shared Google Sheets workbook via the Sheets v4 API. Each data type lands in its own named tab:

| Tab name | Script | Behaviour |
|----------|--------|-----------|
| `overs` | `main.py` | Append-only (incremental) |
| `scores` | `main.py` | Append-only (incremental) |
| `batting_data` | `execute_individual_scraping.py` | Append-only (incremental) |
| `bowling_data` | `execute_individual_scraping.py` | Append-only (incremental) |
| `batting champions` | `scrape_champion_players.py` | Clear and rewrite each run |
| `bowling champions` | `scrape_champion_players.py` | Clear and rewrite each run |
| `ladder` | `scrape_ladder_position.py` | Clear and rewrite each run |
