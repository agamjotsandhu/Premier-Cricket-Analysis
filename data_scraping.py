from __future__ import annotations
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from selenium.webdriver.chrome.options import Options
from typing import List, Tuple
import re
import time

def build_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,1024")
    options.page_load_strategy = "eager"
    
    options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.fonts": 2,
    })
    
    driver = webdriver.Chrome(options=options)
    return driver

def scan_grade_all_rounds(driver, grade_url) -> dict[int, list[tuple[str, str]]]:
    driver.get(grade_url)
    
    # Wait until the fixture sections load
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "li.w-fixtures-listing__item")
        )
    )

    # Scroll to bottom to ensure all lazy-loaded rounds are present
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # Give it a moment to load any additional sections
    
    fixture_sections = driver.find_elements(By.CSS_SELECTOR, "li.w-fixtures-listing__item")
    round_index = {}
    
    for section in fixture_sections:
        try:
            header_text = section.find_element(By.CSS_SELECTOR, "h3.w-fixtures-listing__section-header span").text
            match_round = re.search(r"Round\s+(\d+)", header_text, flags=re.I)
            if not match_round:
                continue
            
            round_num = int(match_round.group(1))
            match_cards = section.find_elements(By.CSS_SELECTOR, "a.o-play-match-card__link")
            
            matches = []
            for a in match_cards:
                home_team = a.find_element(By.CSS_SELECTOR, ".o-play-match-card__team--home .o-play-match-card__team-name").text
                away_team = a.find_element(By.CSS_SELECTOR, ".o-play-match-card__team--away .o-play-match-card__team-name").text
                match_name = f"{home_team} V {away_team}"
                match_url = a.get_attribute("href")
                matches.append((match_name, match_url))
            
            if matches:
                round_index[round_num] = matches
        except Exception:
            continue
            
    return round_index

def scrape_match_data(driver, match_url: str) -> Tuple[List, List]:
    wait = WebDriverWait(driver, 10)
    driver.get(match_url)

    # Match status: skip if ABANDONED or UPCOMING
    status_el = wait.until(
        EC.visibility_of_element_located(
            (By.CSS_SELECTOR, "span.o-content-meta__tag-match-status--completed")
        )
    )

    status_text = status_el.text.strip()
    if ("ABANDONED" in status_text) or ("UPCOMING" in status_text): 
        return [], []
    
    # Grade: extract 1st/2nd/3rd/4th
    grade_elems = wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".o-play-match-card__team-name"))
    )
    grade_text = " ".join([e.text for e in grade_elems if e.text]).strip()
    m_grade = re.search(r"\b(1st|2nd|3rd|4th)\b", grade_text, flags=re.I)
    grade = m_grade.group(1).lower().replace("xi", "").strip() if m_grade else ""

    # Ball-by-ball tab (overs data is here)
    try: 
        wait.until(EC.element_to_be_clickable((By.ID, "tab-ball-by-ball"))).click()
    except:
        return [], []

    # Innings toggles
    toggle_group = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "fieldset.o-toggle.js-toggle[name='innings']"))
    )

    toggle_labels = toggle_group.find_elements(By.CSS_SELECTOR, ".o-toggle__option label.o-toggle__label")
    toggle_inputs = toggle_group.find_elements(By.CSS_SELECTOR, ".o-toggle__option input.o-toggle__input")

    if len(toggle_labels) < 2 or len(toggle_inputs) < 2:
        raise RuntimeError("Expected two innings toggles.")

    team_a = toggle_labels[0].find_elements(By.TAG_NAME, "span")[0].text.strip()
    team_b = toggle_labels[1].find_elements(By.TAG_NAME, "span")[0].text.strip()

    matchup = f"{team_a} vs {team_b}"

    over_rows: List[List] = []
    all_scores: List[List] = []

    def scrape_innings(current_batting: str, opposition: str) -> list[list]:
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article.w-play-match-centre-ballbyball__over"))
        )

        over_blocks = driver.find_elements(By.CSS_SELECTOR, "article.w-play-match-centre-ballbyball__over")
        rows: list[list] = []

        for over in over_blocks:
            # --- over number ---
            over_h3 = over.find_element(By.CSS_SELECTOR, "h3.w-play-match-centre-ballbyball__over-title").text
            m_over = re.search(r"Over\s+(\d+)", over_h3, flags=re.I)
            over_num = int(m_over.group(1)) if m_over else -1

            # --- runs / wickets (default to "0" if missing) ---
            runs_els = over.find_elements(
                By.CSS_SELECTOR,
                "p.w-play-match-centre-ballbyball__runs-wickets span.w-play-match-centre-ballbyball__runs"
            )
            runs_over = (runs_els[0].text.strip() if runs_els and runs_els[0].text else "0")

            wickets_els = over.find_elements(
                By.CSS_SELECTOR,
                "p.w-play-match-centre-ballbyball__runs-wickets span.w-play-match-centre-ballbyball__wickets"
            )
            wickets_over = (wickets_els[0].text.strip() if wickets_els and wickets_els[0].text else "0")

            # --- optional bowler from over-summary if present ---
            bowler_summary = ""
            try:
                summary_text = over.find_element(
                    By.CSS_SELECTOR,
                    "li.w-play-match-centre-ballbyball__entry--over-summary .w-play-match-centre-ballbyball__content",
                ).text.strip()
                m_bowler = re.search(r"Bowler:\s*([^\.]+)\.", summary_text, flags=re.I)
                if m_bowler:
                    bowler_summary = m_bowler.group(1).strip()
            except Exception:
                pass

            if not bowler_summary:
                # Try to get bowler from the first ball message in this over block
                try:
                    msg_els = over.find_elements(By.CSS_SELECTOR, "p.w-play-match-centre-ballbyball__entry--message")
                    if not msg_els:
                        msg_els = over.find_elements(By.CSS_SELECTOR, ".w-play-match-centre-ballbyball__content")
                    
                    if msg_els:
                        first_msg = msg_els[0].text.strip()
                        bowler_summary = extract_bowler(first_msg)
                except:
                    pass

            rows.append(
                [
                    grade,
                    matchup,
                    opposition,
                    current_batting,
                    over_num,
                    bowler_summary,
                    runs_over,
                    wickets_over
                ]
            )
        return rows

    # innings A
    try: 
        team_name_a, scores_a_text = toggle_labels[0].text.strip().split("\n")
        total_wickets_a, total_runs_a = (scores_a_text.split("-") if "-" in scores_a_text else ("10", scores_a_text))
    except:
        team_name_a = team_a
        total_runs_a = "0"
        total_wickets_a = "0"

    scores_a = [grade, matchup, team_b, team_name_a, total_runs_a, total_wickets_a]
    all_scores.append(scores_a)

    try: 
        wait.until(EC.element_to_be_clickable(toggle_labels[0])).click()
        wait.until(lambda d: toggle_inputs[0].get_attribute("aria-checked") in ("true", "True"))
        over_rows.extend(scrape_innings(team_a, team_b))
    except:
        pass

    # innings B
    try: 
        team_name_b, scores_b_text = toggle_labels[1].text.strip().split("\n")
        total_wickets_b, total_runs_b = (scores_b_text.split("-") if "-" in scores_b_text else ("10", scores_b_text))
    except:
        team_name_b = team_b
        total_runs_b = "0"
        total_wickets_b = "0"
    
    scores_b = [grade, matchup, team_a, team_name_b, total_runs_b, total_wickets_b]
    all_scores.append(scores_b)

    try: 
        wait.until(EC.element_to_be_clickable(toggle_labels[1])).click()
        wait.until(lambda d: toggle_inputs[1].get_attribute("aria-checked") in ("true", "True"))
        over_rows.extend(scrape_innings(team_b, team_a))
    except: 
        pass

    return over_rows, all_scores

def extract_bowler(msg: str) -> str:
    s = (msg or "").strip()
    if s.startswith("Out!"):
        m = re.search(r"Out!\s*[^.]*\.\s*(.+)", s, flags=re.I)
        if not m:
            return ""
        tail = m.group(1).strip()
        m_bowler = re.search(r"^(.+?)\s+to\s+", tail, flags=re.I)
        if m_bowler:
            return m_bowler.group(1).strip()
        return ""
    else:
        m = re.search(r"^\s*(.+?)\s+to\s+.+?(?:\s*:|$)", s, flags=re.I)
        if m:
            return m.group(1).strip()
        return ""
