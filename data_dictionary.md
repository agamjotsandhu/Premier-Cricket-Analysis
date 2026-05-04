# Cricket Analytics — Data Dictionary

## Overview

Six CSV files are produced by this pipeline, covering three layers of data: season-level player leaderboards, match-level scorecards, and ball/over-level play data. All data is sourced from play.cricket.com.au and covers grades 1st through 4th.

---

## `batting_champions.csv`

Season-long batting leaderboard aggregated across all grades.

| Column | Type | Description |
|--------|------|-------------|
| `rank` | integer | Player's rank within their grade leaderboard |
| `player` | string | Player name in "Surname, First" format |
| `club` | string | Player's club name |
| `grade` | string | Grade the player competed in (`1st`, `2nd`, `3rd`, `4th`) |
| `runs` | integer | Total runs scored for the season |
| `avg` | float | Batting average (runs per dismissal) for the season |

**One row per player per grade.** A player appearing in multiple grades will have multiple rows.

---

## `bowling_champions.csv`

Season-long bowling leaderboard aggregated across all grades.

| Column | Type | Description |
|--------|------|-------------|
| `rank` | integer | Player's rank within their grade leaderboard |
| `player` | string | Player name in "Surname, First" format |
| `club` | string | Player's club name |
| `grade` | string | Grade the player competed in (`1st`, `2nd`, `3rd`, `4th`) |
| `wkts` | integer | Total wickets taken for the season |
| `avg` | float | Bowling average (runs conceded per wicket) for the season |

**One row per player per grade.**

---

## `batting_data.csv`

Ball-level batting scorecard data. One row per batting innings per player per match.

| Column | Type | Description |
|--------|------|-------------|
| `round` | integer | Competition round number |
| `grade` | string | Grade (`1st`, `2nd`, etc.) |
| `matchup` | string | Short match identifier (e.g. `CAS vs RIC`) |
| `opposition` | string | The fielding/bowling team in this row |
| `batting_team` | string | The batting team in this row |
| `batsman_name` | string | Full name of the batter |
| `runs` | integer | Runs scored (blank if did not bat or no data) |
| `balls` | integer | Balls faced (blank if did not bat or no data) |
| `fours` | integer | Number of fours hit |
| `sixes` | integer | Number of sixes hit |
| `how_out` | string | Dismissal description (e.g. `c: D Parikh b: J Sawrey`, `lbw: S Mills`, `not out`, `did not bat`) |

**Notes:**
- `runs` and `balls` are blank for players who did not bat or whose data was unavailable.
- `how_out` uses the format `c: [fielder] b: [bowler]` for caught dismissals, `b: [bowler]` for bowled, `lbw: [bowler]` for lbw, `c&b: [bowler]` for caught and bowled.

---

## `bowling_data.csv`

Spell-level bowling data. One row per bowler per innings per match.

| Column | Type | Description |
|--------|------|-------------|
| `round` | integer | Competition round number |
| `grade` | string | Grade (`1st`, `2nd`, etc.) |
| `matchup` | string | Short match identifier |
| `opposition` | string | The batting team faced in this row |
| `bowling_team` | string | The bowling team in this row |
| `bowler_name` | string | Full name of the bowler |
| `overs` | float | Overs bowled (e.g. `3.2` = 3 overs and 2 balls) |
| `maidens` | integer | Maiden overs bowled (blank if unavailable) |
| `runs` | integer | Runs conceded |
| `wickets` | integer | Wickets taken |
| `wides` | float | Wides bowled |
| `no_balls` | float | No balls bowled |

---

## `overs.csv`

Over-by-over summary data. One row per over per innings per match.

| Column | Type | Description |
|--------|------|-------------|
| `round` | integer | Competition round number |
| `grade` | string | Grade (`1st`, `2nd`, etc.) |
| `matchup` | string | Short match identifier |
| `opposition` | string | The bowling team for this over |
| `current_batting` | string | The batting team for this over |
| `over_num` | integer | Over number within the innings (1-indexed) |
| `bowler_for_row` | string | Bowler who bowled this over (name may be abbreviated) |
| `runs` | integer | Runs scored in this over (including extras) |
| `wickets` | integer | Wickets taken in this over |

**Notes:**
- Rows are ordered from the last over to the first within each innings (descending `over_num`) as scraped from the site.
- Bowler names are sometimes abbreviated (e.g. `D Parikh` vs `Dhruval Parikh`) depending on how the source page rendered them.

---

## `scores.csv`

Match-level scorecard totals. One row per innings per match.

| Column | Type | Description |
|--------|------|-------------|
| `round` | integer | Competition round number |
| `grade` | string | Grade (`1st`, `2nd`, etc.) |
| `matchup` | string | Short match identifier |
| `opposition` | string | The bowling team for this innings |
| `batting_team` | string | The batting team for this innings |
| `total_runs` | integer | Total runs scored in the innings |
| `total_wickets` | string | Wickets lost (may include `d` suffix for declared innings, e.g. `9d`) |

**Notes:**
- Each match produces exactly two rows (one per innings), except 2-day matches which may produce up to four.
- Declared innings are indicated by a `d` appended to `total_wickets` (e.g. `311,9d`).
- Round 5 is absent from the sample data, likely due to a bye round or incomplete scraping.
