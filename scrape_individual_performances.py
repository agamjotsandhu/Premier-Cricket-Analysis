from __future__ import annotations
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from typing import List, Tuple
import re


def scrape_scorecard_data(driver, match_url: str) -> Tuple[List, List]:
    """
    Scrapes the SCORECARD tab for a given match URL.

    Returns:
        batting_rows: list of
            [grade, matchup, opposition, batting_team,
             batsman_name, runs, balls, fours, sixes, how_out]
        bowling_rows: list of
            [grade, matchup, opposition, bowling_team,
             bowler_name, overs, maidens, runs, wickets, wides, no_balls]

    opposition  = the team NOT batting in that innings (i.e. the bowling side)
    bowling_team = same as opposition (kept explicit for clarity)
    """
    wait = WebDriverWait(driver, 10)
    driver.get(match_url)

    # ------------------------------------------------------------------
    # Guard: only scrape completed matches
    # ------------------------------------------------------------------
    try:
        status_el = wait.until(
            EC.visibility_of_element_located(
                (By.CSS_SELECTOR, "span.o-content-meta__tag-match-status--completed")
            )
        )
        status_text = status_el.text.strip().upper()
        if "ABANDONED" in status_text or "UPCOMING" in status_text:
            return [], []
    except TimeoutException:
        # Not completed / element not found — skip
        return [], []

    # ------------------------------------------------------------------
    # Grade detection  (1st / 2nd / 3rd / 4th)
    # ------------------------------------------------------------------
    grade = ""
    try:
        grade_elems = driver.find_elements(By.CSS_SELECTOR, ".o-play-match-card__team-name")
        grade_text = " ".join([e.text for e in grade_elems if e.text]).strip()
        m_grade = re.search(r"\b(1st|2nd|3rd|4th)\b", grade_text, flags=re.I)
        grade = m_grade.group(1).lower() if m_grade else ""
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Click SCORECARD tab (it may already be active, ignore failure)
    # ------------------------------------------------------------------
    try:
        scorecard_tab = wait.until(
            EC.element_to_be_clickable((By.ID, "tab-scorecard"))
        )
        scorecard_tab.click()
    except Exception:
        pass  # already on scorecard, or tab id differs — continue anyway

    # ------------------------------------------------------------------
    # Read innings toggle to get team names
    # ------------------------------------------------------------------
    try:
        toggle_group = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "fieldset.o-toggle.js-toggle[name='innings']")
            )
        )
    except TimeoutException:
        return [], []

    toggle_options = toggle_group.find_elements(By.CSS_SELECTOR, ".o-toggle__option")
    if len(toggle_options) < 2:
        return [], []

    innings_meta = []
    for opt in toggle_options:
        label = opt.find_element(By.CSS_SELECTOR, "label.o-toggle__label")
        inp   = opt.find_element(By.CSS_SELECTOR, "input.o-toggle__input")
        spans = label.find_elements(By.TAG_NAME, "span")
        team_name = spans[0].text.strip() if spans else label.text.strip().split("\n")[0]
        innings_meta.append({"label": label, "input": inp, "team_name": team_name})

    # The two competing teams are always the first two toggle entries
    team_a = innings_meta[0]["team_name"]
    team_b = innings_meta[1]["team_name"]
    matchup = f"{team_a} vs {team_b}"

    def resolve_opposition(batting_team: str) -> str:
        return team_b if batting_team == team_a else team_a

    # ------------------------------------------------------------------
    # Per-innings scrape helpers
    # ------------------------------------------------------------------

    def _parse_batting_rows(batting_team: str, opposition: str) -> list:
        """
        Scrape the batting table for the currently displayed innings.
        Rows look like standard <tr> inside a batting section.
        """
        rows = []
        try:
            # Wait for the batting section to be present
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "table tbody tr")
                )
            )
            # Find all tables on the page; batting is first, bowling second
            tables = driver.find_elements(By.CSS_SELECTOR, "table")
            if not tables:
                return rows

            batting_table = tables[0]
            tr_els = batting_table.find_elements(By.CSS_SELECTOR, "tbody tr")

            for tr in tr_els:
                tds = tr.find_elements(By.TAG_NAME, "td")
                if len(tds) < 2:
                    continue

                name = tds[0].text.strip()
                if not name or name.upper() in ("EXTRAS", "TOTAL", "DNB"):
                    continue
                # Skip summary/total rows that start with numbers or are blank
                if re.match(r"^\d", name):
                    continue

                how_out = tds[1].text.strip() if len(tds) > 1 else ""

                # "did not bat" rows — include with empty numerics
                if "did not bat" in how_out.lower():
                    rows.append([
                        grade, matchup, opposition, batting_team,
                        name, "", "", "", "", how_out
                    ])
                    continue

                # Numeric columns: runs, balls, 4s, 6s
                # Column layout (0-indexed): name | how_out | R | B | 4s | 6s | SR | wicket_icon
                runs    = tds[2].text.strip() if len(tds) > 2 else ""
                balls   = tds[3].text.strip() if len(tds) > 3 else ""
                fours   = tds[4].text.strip() if len(tds) > 4 else ""
                sixes   = tds[5].text.strip() if len(tds) > 5 else ""

                # Strip asterisk from not-out scores
                runs = re.sub(r"\*", "", runs).strip()

                rows.append([
                    grade, matchup, opposition, batting_team,
                    name, runs, balls, fours, sixes, how_out
                ])
        except Exception:
            pass
        return rows

    def _parse_bowling_rows(batting_team: str, bowling_team: str) -> list:
        """
        Scrape the bowling table for the currently displayed innings.
        opposition (for column consistency) = batting_team.
        bowling_team = the side doing the bowling.
        """
        rows = []
        try:
            tables = driver.find_elements(By.CSS_SELECTOR, "table")
            if len(tables) < 2:
                return rows

            bowling_table = tables[1]
            tr_els = bowling_table.find_elements(By.CSS_SELECTOR, "tbody tr")

            for tr in tr_els:
                tds = tr.find_elements(By.TAG_NAME, "td")
                if len(tds) < 2:
                    continue

                name = tds[0].text.strip()
                if not name or re.match(r"^\d", name):
                    continue

                # Column layout: name | O | M | R | W | Econ | Wd | NB
                overs   = tds[1].text.strip() if len(tds) > 1 else ""
                maidens = tds[2].text.strip() if len(tds) > 2 else ""
                runs    = tds[3].text.strip() if len(tds) > 3 else ""
                wickets = tds[4].text.strip() if len(tds) > 4 else ""
                # Skip Econ (index 5)
                wides   = tds[6].text.strip() if len(tds) > 6 else ""
                no_balls = tds[7].text.strip() if len(tds) > 7 else ""

                rows.append([
                    grade, matchup, batting_team, bowling_team,
                    name, overs, maidens, runs, wickets, wides, no_balls
                ])
        except Exception:
            pass
        return rows

    # ------------------------------------------------------------------
    # Iterate innings
    # ------------------------------------------------------------------
    all_batting: List[List] = []
    all_bowling: List[List] = []

    for meta in innings_meta:
        batting_team = meta["team_name"]
        opposition   = resolve_opposition(batting_team)

        try:
            wait.until(EC.element_to_be_clickable(meta["label"])).click()
            wait.until(
                lambda d, inp=meta["input"]: inp.get_attribute("aria-checked") in ("true", "True")
            )
            # Small pause for table re-render
            import time
            time.sleep(0.5)

            batting_rows = _parse_batting_rows(batting_team, opposition)
            bowling_rows = _parse_bowling_rows(batting_team, opposition)

            all_batting.extend(batting_rows)
            all_bowling.extend(bowling_rows)

        except Exception as e:
            # Log but don't abort other innings
            print(f"  [warn] innings '{batting_team}' scrape failed: {e}")
            continue

    return all_batting, all_bowling
