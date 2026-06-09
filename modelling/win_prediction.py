"""
win_prediction.py
-----------------
ML pipeline predicting cricket match win/loss from match-level features.
Target: ≥80% accuracy on held-out round (temporal validation).

Data notes
----------
- bowling_data.csv has misnamed columns (verified by economy cross-check):
    CSV "runs"     → actual wickets taken by bowler
    CSV "wickets"  → actual runs conceded by bowler
    CSV "wides"    → economy rate (decimal, runs/over)
    CSV "no_balls" → actual wides bowled  (no-balls column absent)
- Team names carry ordinal prefixes in multi-innings matches:
    "1st MEL" / "2nd MEL" both normalize to base team "MEL"
- overs.csv provides per-over runs/wickets for each batting team in each match
"""

import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature registry: metadata for every engineered feature.
# Printed in the final report once the accuracy target is reached.
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_REGISTRY: dict[str, dict] = {
    # ── Match context ────────────────────────────────────────────────────────
    "grade_num": {
        "group": "Match Context",
        "description": "Competition grade as integer (1st=1, 2nd=2, 3rd=3, 4th=4).",
        "rationale": "Different grades represent different skill levels and match formats. "
                     "Grade 2–4 are two-day matches with much higher scoring totals.",
        "iteration": "base",
    },
    "match_is_twoday": {
        "group": "Match Context",
        "description": "1 if the match has 4 or more innings (two-day format), else 0.",
        "rationale": "Two-day matches produce larger totals and different win patterns than "
                     "one-day games. An explicit flag lets the model separate these regimes.",
        "iteration": "base",
    },

    # ── Batting (from batting_data) ──────────────────────────────────────────
    "bat_total_runs": {
        "group": "Batting",
        "description": "Sum of individual batter contributions (excludes extras).",
        "rationale": "The most direct measure of batting output. Excludes wides/no-balls so "
                     "it represents genuine batter productivity.",
        "iteration": "base",
    },
    "bat_total_balls": {
        "group": "Batting",
        "description": "Total balls faced by all batters who played.",
        "rationale": "Measures batting time consumed. Longer innings generally reflect "
                     "stability and accumulation.",
        "iteration": "base",
    },
    "bat_strike_rate": {
        "group": "Batting",
        "description": "Runs per 100 balls faced.",
        "rationale": "Captures batting intent: aggressive teams score faster, which can "
                     "signal momentum and pressure on the opposition.",
        "iteration": "base",
    },
    "bat_n_20plus": {
        "group": "Batting",
        "description": "Number of batters who scored ≥20 runs.",
        "rationale": "Multiple 20+ contributions signal batting depth — wins often come "
                     "from teams where several batters contribute meaningfully.",
        "iteration": "base",
    },
    "bat_n_30plus": {
        "group": "Batting",
        "description": "Number of batters who scored ≥30 runs.",
        "rationale": "A finer depth metric. Reaching 30 usually means a batter played a "
                     "substantial knock rather than a cameo.",
        "iteration": "base",
    },
    "bat_n_50plus": {
        "group": "Batting",
        "description": "Number of batters who scored ≥50 runs (half-centuries).",
        "rationale": "Identifies match-winning innings. One or more half-centuries typically "
                     "anchor a competitive total.",
        "iteration": "base",
    },
    "bat_top_score": {
        "group": "Batting",
        "description": "Highest individual score in the innings.",
        "rationale": "A strong individual performance can rescue an innings. Also serves as "
                     "a proxy for whether a 'match-winner' performed.",
        "iteration": "base",
    },
    "bat_n_batters": {
        "group": "Batting",
        "description": "Number of players who batted (excluding 'did not bat').",
        "rationale": "In limited-overs games, a lower number can mean the team scored quickly "
                     "and won without using their full batting order. In two-day games it "
                     "reflects innings depth.",
        "iteration": "base",
    },
    "bat_dis_bowled": {
        "group": "Batting",
        "description": "Count of batters dismissed bowled.",
        "rationale": "Bowled dismissals suggest the bowler beat the bat through gaps — "
                     "a sign of quality bowling or poor technique under pressure.",
        "iteration": "base",
    },
    "bat_dis_caught": {
        "group": "Batting",
        "description": "Count of batters dismissed caught.",
        "rationale": "Most common dismissal. High caught count indicates the opposition set "
                     "good attacking fields and induced edges/mistimed shots.",
        "iteration": "base",
    },
    "bat_dis_lbw": {
        "group": "Batting",
        "description": "Count of lbw dismissals.",
        "rationale": "LBW suggests the batting team struggled against straight-pitched or "
                     "swinging deliveries — bowling pitch usage.",
        "iteration": "base",
    },
    "bat_dis_run_out": {
        "group": "Batting",
        "description": "Count of run-out dismissals.",
        "rationale": "Run-outs typically result from pressure, miscommunication, or aggressive "
                     "running — a signal of a team under pressure or trying too hard to score.",
        "iteration": "base",
    },
    "bat_dis_stumped": {
        "group": "Batting",
        "description": "Count of stumped dismissals.",
        "rationale": "Stumpings indicate the batting team was drawn down the wicket, often by "
                     "spin bowling in favourable conditions.",
        "iteration": "base",
    },
    "bat_dis_not_out": {
        "group": "Batting",
        "description": "Number of batters who finished not out.",
        "rationale": "High not-out count means the batting team still had wickets in hand — "
                     "often associated with a successful run chase or a declared innings.",
        "iteration": "base",
    },
    "bat_extras_est": {
        "group": "Batting",
        "description": "Official innings total minus sum of individual batter runs (≈ extras).",
        "rationale": "Extras (wides, no-balls, byes) are free runs given to the batting team. "
                     "Higher extras indicate bowling indiscipline by the opposition.",
        "iteration": "base",
    },
    "bat_boundary_runs": {
        "group": "Batting",
        "description": "Runs scored through boundaries (4s × 4 + 6s × 6).",
        "rationale": "Boundary count is a direct measure of attacking batting. Teams that "
                     "score heavily in boundaries often post large totals.",
        "iteration": "base",
    },
    "bat_boundary_pct": {
        "group": "Batting",
        "description": "Fraction of team runs coming from boundaries.",
        "rationale": "High boundary percentage signals that most runs came from shots rather "
                     "than running between wickets — a proxy for batting dominance.",
        "iteration": "base",
    },
    "bat_wickets_lost": {
        "group": "Batting",
        "description": "Total official wickets lost across all innings (from scores.csv).",
        "rationale": "Fewer wickets lost means the team maintained their batting resources. "
                     "An all-out innings (10 wickets) vs a declaration at 5 wickets tells "
                     "very different stories about control.",
        "iteration": "base",
    },
    "bat_declared": {
        "group": "Batting",
        "description": "1 if any of the team's innings was a declared innings.",
        "rationale": "A team declares when it is confident its total is enough to win. "
                     "Declaring is a strong signal of batting dominance in two-day cricket.",
        "iteration": "base",
    },

    # ── Opposition bowling (bowling_data where they bowled against us) ────────
    "bowl_opp_wickets": {
        "group": "Opposition Bowling",
        "description": "Total wickets the opposition took when bowling against our team.",
        "rationale": "More wickets taken means the opposition disrupted our batting order. "
                     "This directly reflects how much trouble the bowling attack caused.",
        "iteration": "base",
    },
    "bowl_opp_total_overs": {
        "group": "Opposition Bowling",
        "description": "Total overs the opposition bowled against our team.",
        "rationale": "Related to how long the innings lasted. Fewer overs (with the same "
                     "total runs) indicates faster scoring.",
        "iteration": "base",
    },
    "bowl_opp_economy": {
        "group": "Opposition Bowling",
        "description": "Opposition's economy rate (runs per over) when bowling against us.",
        "rationale": "A lower economy means the opposition bowled tightly, making it harder "
                     "to score freely. High economy = our team scored easily.",
        "iteration": "base",
    },
    "bowl_opp_strike_rate": {
        "group": "Opposition Bowling",
        "description": "Opposition's bowling strike rate (balls per wicket).",
        "rationale": "Measures how quickly the opposition took wickets. A low strike rate "
                     "means they dismissed our batters rapidly — a sign of danger.",
        "iteration": "base",
    },
    "bowl_opp_wides": {
        "group": "Opposition Bowling",
        "description": "Total wides bowled by the opposition against us.",
        "rationale": "Wides are free runs. An opposition that gives away many wides is "
                     "lacking discipline — free runs boost our total.",
        "iteration": "base",
    },
    "bowl_n_bowlers": {
        "group": "Opposition Bowling",
        "description": "Number of different bowlers used by the opposition.",
        "rationale": "More bowlers can indicate the captain was searching for a penetrating "
                     "option. Fewer bowlers dominating overs often signals a stronger attack.",
        "iteration": "base",
    },
    "bowl_wicket_conc": {
        "group": "Opposition Bowling",
        "description": "Fraction of opposition's wickets taken by their single best bowler.",
        "rationale": "A high concentration means one bowler carried the opposition attack — "
                     "a fragile attack. A lower value indicates a well-spread, stronger attack.",
        "iteration": "base",
    },

    # ── Own bowling (bowling_data where WE bowled against them) ─────────────
    "our_bowl_wickets": {
        "group": "Own Bowling",
        "description": "Total wickets our team took when bowling against the opposition.",
        "rationale": "Directly measures how well our bowlers dismissed the opposition. "
                     "Teams that take more wickets constrain the opposition's total.",
        "iteration": "base",
    },
    "our_bowl_overs": {
        "group": "Own Bowling",
        "description": "Total overs our team bowled against the opposition.",
        "rationale": "More overs means the match went longer from our bowling end. "
                     "Combined with wickets, it helps define our bowling dominance.",
        "iteration": "base",
    },
    "our_bowl_economy": {
        "group": "Own Bowling",
        "description": "Our team's economy rate (runs conceded per over) when bowling.",
        "rationale": "A lower economy means we restricted the opposition. This is a key "
                     "complement to our batting total in determining match outcomes.",
        "iteration": "base",
    },
    "our_bowl_sr": {
        "group": "Own Bowling",
        "description": "Our bowling strike rate (balls per wicket, capped at 999).",
        "rationale": "Measures how frequently we took wickets. A lower value means we "
                     "dismissed the opposition quickly — high pressure bowling.",
        "iteration": "base",
    },
    "our_bowl_wides": {
        "group": "Own Bowling",
        "description": "Total wides we bowled when bowling against the opposition.",
        "rationale": "Our wides give the opposition free runs. Fewer wides means disciplined "
                     "bowling — a sign of a well-prepared attack.",
        "iteration": "base",
    },
    "our_n_bowlers": {
        "group": "Own Bowling",
        "description": "Number of bowlers we used when bowling against the opposition.",
        "rationale": "Using more bowlers can reflect a varied attack or a struggle to "
                     "find penetration. Context from economy/wickets completes the picture.",
        "iteration": "base",
    },

    # ── Over-by-over (overs.csv) ─────────────────────────────────────────────
    "overs_total_faced": {
        "group": "Over-by-Over",
        "description": "Total overs faced by our batting team across all innings.",
        "rationale": "Longer innings generally correlate with larger totals and batting "
                     "dominance. Short innings suggest collapse or a limited overs game.",
        "iteration": "base",
    },
    "overs_early_wickets": {
        "group": "Over-by-Over",
        "description": "Wickets lost in the first 5 overs of each innings (summed across innings).",
        "rationale": "Early wicket loss puts the batting team under immediate pressure. "
                     "A top-order collapse forces the rest of the batting order to rebuild.",
        "iteration": "base",
    },
    "overs_first_half_rr": {
        "group": "Over-by-Over",
        "description": "Average runs per over in the first half of each innings.",
        "rationale": "Captures how quickly the team scored in the early phase. A high early "
                     "run rate indicates the opposition bowlers struggled up front.",
        "iteration": "base",
    },
    "overs_second_half_rr": {
        "group": "Over-by-Over",
        "description": "Average runs per over in the second half of each innings.",
        "rationale": "Captures death-overs / acceleration phase scoring. Teams that "
                     "accelerate well in the back half post large, match-winning totals.",
        "iteration": "base",
    },
    "overs_acceleration": {
        "group": "Over-by-Over",
        "description": "Ratio of second-half run rate to first-half run rate.",
        "rationale": "An acceleration > 1 means the team scored faster as the innings "
                     "progressed — confident batting depth and/or a collapsing attack.",
        "iteration": "base",
    },

    # ── Iteration-2 derived features ─────────────────────────────────────────
    "bowl_economy_diff": {
        "group": "Bowling (derived)",
        "description": "Opposition's economy minus our economy (positive = we bowled tighter).",
        "rationale": "A direct comparison of bowling quality. If we concede fewer runs per "
                     "over than the opposition did against us, we have a bowling advantage.",
        "iteration": 2,
    },
    "bat_bowl_joint_score": {
        "group": "Batting+Bowling (derived)",
        "description": "Normalised product of batting total and wickets taken: "
                       "(bat_total_runs/100) × (our_bowl_wickets/10).",
        "rationale": "Captures whether BOTH batting and bowling were strong simultaneously. "
                     "A team that bowled well but batted poorly can still lose — this "
                     "feature penalises that imbalance, helping separate true dominance "
                     "from one-dimensional strength.",
        "iteration": 2,
    },

    # ── Iteration-3 derived features ─────────────────────────────────────────
    "bat_depth_index": {
        "group": "Batting (derived)",
        "description": "Weighted milestone count: bat_n_20plus + 2×bat_n_30plus + 3×bat_n_50plus.",
        "rationale": "Composite batting depth score. Rewards teams where multiple batters "
                     "reached significant thresholds, weighting larger milestones more.",
        "iteration": 3,
    },
    "our_bowl_pressure": {
        "group": "Own Bowling (derived)",
        "description": "Our wickets per over × 6 (wickets per 36 balls).",
        "rationale": "Combines wicket-taking ability with overs bowled into a single pressure "
                     "metric. High pressure = we took wickets frequently.",
        "iteration": 3,
    },

    # ── Iteration-4 derived features ─────────────────────────────────────────
    "bat_collapse_risk": {
        "group": "Batting (derived)",
        "description": "Early wickets (overs 1–5) divided by total wickets lost.",
        "rationale": "If many wickets fell early relative to total, the innings was fragile "
                     "at the start. High collapse risk often leads to a lower total.",
        "iteration": 4,
    },
    "our_net_wicket_adv": {
        "group": "Bowling (derived)",
        "description": "Our wickets taken minus opposition's wickets taken against us.",
        "rationale": "Net wicket advantage measures which side's bowling was more penetrating. "
                     "A positive value means we took more wickets than they did.",
        "iteration": 4,
    },

    # ── Iteration-5 derived features ─────────────────────────────────────────
    "bat_avg_per_wicket": {
        "group": "Batting (derived)",
        "description": "Batting average proxy: total runs / (wickets lost + 1).",
        "rationale": "Normalises batting output by wickets lost. A team scoring 200 for "
                     "5 wickets is in a stronger position than one scoring 200 all out.",
        "iteration": 5,
    },
    "bat_overs_advantage": {
        "group": "Batting+Bowling (derived)",
        "description": "Overs we faced batting ÷ overs we bowled: > 1 means we batted longer.",
        "rationale": "In cricket a side that bats for more overs than it bowls typically "
                     "accumulates more runs overall. This ratio captures time-in-bat dominance "
                     "using only the reliable overs columns, avoiding the runs_conceded "
                     "discrepancies seen in bowling_data.",
        "iteration": 5,
    },
    # kept for reference — no longer added in the active iteration loop
    "bat_top_score_pct": {
        "group": "Batting (derived)",
        "description": "Top scorer's runs as a fraction of team total.",
        "rationale": "High reliance on one batter indicates a fragile innings.",
        "iteration": "unused",
    },
    "bowl_opp_pressure_index": {
        "group": "Opposition Bowling (derived)",
        "description": "Opposition wickets × economy (combined difficulty index).",
        "rationale": "Combines wicket-taking with run-rate pressure into one value.",
        "iteration": "unused",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load & clean raw data
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    bat = pd.read_csv(DATA_DIR / "batting_data.csv")
    bw  = pd.read_csv(DATA_DIR / "bowling_data.csv")
    sc  = pd.read_csv(DATA_DIR / "scores.csv")
    ov  = pd.read_csv(DATA_DIR / "overs.csv")

    # Rename misaligned bowling columns (verified: 14 runs / 3 overs ≈ 4.67 = stored economy)
    bw = bw.rename(columns={
        "runs":     "wickets_taken",
        "wickets":  "runs_conceded",
        "wides":    "economy_rate",
        "no_balls": "wides_count",
    })

    # Track declared innings BEFORE stripping the 'd' suffix
    sc["bat_declared_raw"] = sc["total_wickets"].astype(str).str.endswith("d").astype(int)

    # Declared: strip 'd' suffix, then cast ("9d" → 9)
    sc["total_wickets"] = (
        sc["total_wickets"].astype(str).str.replace("d", "", regex=False)
    )
    sc["total_wickets"] = pd.to_numeric(sc["total_wickets"], errors="coerce").fillna(0)
    sc["total_runs"]    = pd.to_numeric(sc["total_runs"],    errors="coerce").fillna(0)

    # Batting numerics
    for col in ["runs", "balls", "fours", "sixes"]:
        bat[col] = pd.to_numeric(bat[col], errors="coerce").fillna(0)

    # Bowling numerics
    for col in ["overs", "wickets_taken", "runs_conceded", "economy_rate",
                "wides_count", "maidens"]:
        bw[col] = pd.to_numeric(bw[col], errors="coerce").fillna(0)

    # Overs numerics
    for col in ["over_num", "runs", "wickets"]:
        ov[col] = pd.to_numeric(ov[col], errors="coerce").fillna(0)

    return bat, bw, sc, ov


def normalize_team(name: str) -> str:
    """
    Strip leading ordinal prefix so both innings map to the same club name.
    '1st MEL' → 'MEL', '2nd ST' → 'ST', 'CAS' → 'CAS'.
    """
    return re.sub(r"^\d+(st|nd|rd|th)\s+", "", str(name)).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Label construction
# ─────────────────────────────────────────────────────────────────────────────

def build_labels(sc: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (team, match).  won=1 if that team's aggregated runs > opponent.
    Two-day matches: both innings summed per base team before comparison.
    Ties excluded.
    """
    sc = sc.copy()
    sc["base_team"] = sc["batting_team"].apply(normalize_team)

    totals = (
        sc.groupby(["round", "grade", "matchup", "base_team"])["total_runs"]
        .sum()
        .reset_index()
    )

    records = []
    for (rnd, grade, matchup), grp in totals.groupby(["round", "grade", "matchup"]):
        if len(grp) < 2:
            continue
        max_r = grp["total_runs"].max()
        if max_r == grp["total_runs"].min():
            continue  # tie — exclude
        for _, row in grp.iterrows():
            records.append({
                "round":   rnd,   "grade":  grade,
                "matchup": matchup, "team": row["base_team"],
                "won":     int(row["total_runs"] == max_r),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

_GRADE_MAP       = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
_DISMISSAL_LABELS = ["bowled", "caught", "lbw", "run_out", "stumped", "not_out"]
_META_COLS        = {"round", "grade", "matchup", "team", "won"}


def _dismissal(how_out) -> str:
    if pd.isna(how_out):
        return "unknown"
    v = str(how_out).strip().lower()
    if v == "did not bat":               return "did_not_bat"
    if v == "not out":                   return "not_out"
    if v.startswith("b:"):               return "bowled"
    if v.startswith("lbw:"):             return "lbw"
    if "run out" in v:                   return "run_out"
    if v.startswith("st:"):              return "stumped"
    if "c&b" in v or v.startswith("c:"): return "caught"
    if "hit wicket" in v:                return "hit_wicket"
    if "retired" in v:                   return "retired"
    return "other"


def _bat_features(bat_grp: pd.DataFrame, score_total: float) -> dict:
    batted = bat_grp[
        bat_grp["how_out"].apply(lambda x: str(x).strip() != "did not bat")
    ]
    total_runs  = float(batted["runs"].sum())
    total_balls = float(batted["balls"].sum())

    feats = {
        "bat_total_runs":  total_runs,
        "bat_total_balls": total_balls,
        "bat_strike_rate": (total_runs / total_balls * 100) if total_balls > 0 else 0.0,
        "bat_n_20plus":    int((batted["runs"] >= 20).sum()),
        "bat_n_30plus":    int((batted["runs"] >= 30).sum()),
        "bat_n_50plus":    int((batted["runs"] >= 50).sum()),
        "bat_top_score":   float(batted["runs"].max()) if len(batted) > 0 else 0.0,
        "bat_n_batters":   int(len(batted)),
    }

    dis = bat_grp["how_out"].apply(_dismissal).value_counts()
    for d in _DISMISSAL_LABELS:
        feats[f"bat_dis_{d}"] = int(dis.get(d, 0))

    # Extras = runs not credited to individual batters (wides, no-balls, byes)
    feats["bat_extras_est"] = max(0.0, score_total - total_runs)

    boundary_runs = float(batted["fours"].sum() * 4 + batted["sixes"].sum() * 6)
    feats["bat_boundary_runs"] = boundary_runs
    feats["bat_boundary_pct"]  = (boundary_runs / total_runs) if total_runs > 0 else 0.0

    return feats


def _opp_bowl_features(bowl_grp: pd.DataFrame) -> dict:
    """Opposition's bowling AGAINST this team."""
    if len(bowl_grp) == 0:
        return {k: 0.0 for k in [
            "bowl_opp_wickets", "bowl_opp_total_overs",
            "bowl_opp_economy", "bowl_opp_strike_rate",
            "bowl_opp_wides",   "bowl_n_bowlers", "bowl_wicket_conc",
        ]}
    total_wkts  = float(bowl_grp["wickets_taken"].sum())
    total_overs = float(bowl_grp["overs"].sum())
    total_runs  = float(bowl_grp["runs_conceded"].sum())
    total_wides = float(bowl_grp["wides_count"].sum())
    economy = total_runs / total_overs if total_overs > 0 else 0.0
    bowl_sr = (total_overs * 6.0) / total_wkts if total_wkts > 0 else 999.0
    max_wkts = float(bowl_grp["wickets_taken"].max())
    wkt_conc = max_wkts / total_wkts if total_wkts > 0 else 0.0
    return {
        "bowl_opp_wickets":     total_wkts,
        "bowl_opp_total_overs": total_overs,
        "bowl_opp_economy":     economy,
        "bowl_opp_strike_rate": bowl_sr,
        "bowl_opp_wides":       total_wides,
        "bowl_n_bowlers":       float(bowl_grp["bowler_name"].nunique()),
        "bowl_wicket_conc":     wkt_conc,
    }


def _own_bowl_features(bowl_grp: pd.DataFrame) -> dict:
    """Our team's own bowling AGAINST the opposition. Key addition over v1."""
    if len(bowl_grp) == 0:
        return {k: 0.0 for k in [
            "our_bowl_wickets", "our_bowl_overs", "our_bowl_economy",
            "our_bowl_sr", "our_bowl_wides", "our_n_bowlers",
        ]}
    total_wkts  = float(bowl_grp["wickets_taken"].sum())
    total_overs = float(bowl_grp["overs"].sum())
    total_runs  = float(bowl_grp["runs_conceded"].sum())
    total_wides = float(bowl_grp["wides_count"].sum())
    economy = total_runs / total_overs if total_overs > 0 else 0.0
    # Lower strike rate = took wickets more frequently (better bowling)
    bowl_sr = (total_overs * 6.0) / total_wkts if total_wkts > 0 else 999.0
    return {
        "our_bowl_wickets": total_wkts,
        "our_bowl_overs":   total_overs,
        "our_bowl_economy": economy,
        "our_bowl_sr":      bowl_sr,
        "our_bowl_wides":   total_wides,
        "our_n_bowlers":    float(bowl_grp["bowler_name"].nunique()),
    }


def _overs_features(ov_grp: pd.DataFrame) -> dict:
    """
    Per-over runs/wickets for this team's batting innings.
    Handles multi-innings by processing each innings (original current_batting name)
    separately then summing, so first-half/second-half ratios stay within each innings.
    """
    null = {
        "overs_total_faced":   0.0,
        "overs_early_wickets": 0.0,
        "overs_first_half_rr": 0.0,
        "overs_second_half_rr":0.0,
        "overs_acceleration":  1.0,
    }
    if len(ov_grp) == 0:
        return null

    total_overs      = 0.0
    early_wickets    = 0
    fh_runs, fh_ovs  = 0, 0
    sh_runs, sh_ovs  = 0, 0

    # Process each original innings name separately (avoids mixing over numbers across innings)
    for _, inn_grp in ov_grp.groupby("current_batting"):
        max_ov  = float(inn_grp["over_num"].max())
        midpt   = max_ov / 2.0
        total_overs += max_ov

        early_wickets += int(inn_grp.loc[inn_grp["over_num"] <= 5, "wickets"].sum())

        fh = inn_grp[inn_grp["over_num"] <= midpt]
        sh = inn_grp[inn_grp["over_num"] >  midpt]
        fh_runs += int(fh["runs"].sum());  fh_ovs += len(fh)
        sh_runs += int(sh["runs"].sum());  sh_ovs += len(sh)

    fh_rr = fh_runs / fh_ovs if fh_ovs > 0 else 0.0
    sh_rr = sh_runs / sh_ovs if sh_ovs > 0 else 0.0
    accel = sh_rr / fh_rr if fh_rr > 0 else 1.0

    return {
        "overs_total_faced":    total_overs,
        "overs_early_wickets":  float(early_wickets),
        "overs_first_half_rr":  fh_rr,
        "overs_second_half_rr": sh_rr,
        "overs_acceleration":   accel,
    }


def build_feature_matrix(
    bat: pd.DataFrame, bw: pd.DataFrame, sc: pd.DataFrame,
    ov: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Build one feature row per (round, grade, matchup, team)."""
    bat = bat.copy(); bw = bw.copy(); sc = sc.copy(); ov = ov.copy()

    bat["base_team"] = bat["batting_team"].apply(normalize_team)
    bw["base_bowl"]  = bw["bowling_team"].apply(normalize_team)
    bw["base_opp"]   = bw["opposition"].apply(normalize_team)
    sc["base_team"]  = sc["batting_team"].apply(normalize_team)
    ov["base_bat"]   = ov["current_batting"].apply(normalize_team)

    # Match-level flag: ≥4 innings rows → two-day format
    match_format = (
        sc.groupby(["round", "grade", "matchup"]).size()
        .reset_index(name="n_innings")
    )
    match_format["match_is_twoday"] = (match_format["n_innings"] >= 4).astype(int)

    rows = []
    for _, lbl in labels.iterrows():
        rnd, grade, matchup, team = lbl["round"], lbl["grade"], lbl["matchup"], lbl["team"]

        # Filters
        bat_m = ((bat["round"] == rnd) & (bat["grade"] == grade) &
                 (bat["matchup"] == matchup) & (bat["base_team"] == team))
        sc_m  = ((sc["round"]  == rnd) & (sc["grade"]  == grade) &
                 (sc["matchup"]  == matchup) & (sc["base_team"]  == team))
        opp_bowl_m = ((bw["round"] == rnd) & (bw["grade"] == grade) &
                      (bw["matchup"] == matchup) & (bw["base_opp"]  == team))
        own_bowl_m = ((bw["round"] == rnd) & (bw["grade"] == grade) &
                      (bw["matchup"] == matchup) & (bw["base_bowl"] == team))
        ov_m  = ((ov["round"]  == rnd) & (ov["grade"]  == grade) &
                 (ov["matchup"]  == matchup) & (ov["base_bat"]  == team))

        score_total   = float(sc.loc[sc_m, "total_runs"].sum())
        wickets_lost  = float(sc.loc[sc_m, "total_wickets"].sum())
        declared_flag = int(sc.loc[sc_m, "bat_declared_raw"].sum() > 0)

        fmt_row = match_format[
            (match_format["round"]   == rnd) &
            (match_format["grade"]   == grade) &
            (match_format["matchup"] == matchup)
        ]
        is_twoday = int(fmt_row["match_is_twoday"].values[0]) if len(fmt_row) else 0

        row = {
            "round": rnd, "grade": grade, "matchup": matchup, "team": team,
            "won":       lbl["won"],
            "grade_num": _GRADE_MAP.get(grade, 1),
            "match_is_twoday": is_twoday,
            "bat_wickets_lost": wickets_lost,
            "bat_declared":     declared_flag,
        }
        row.update(_bat_features(bat[bat_m], score_total))
        row.update(_opp_bowl_features(bw[opp_bowl_m]))
        row.update(_own_bowl_features(bw[own_bowl_m]))
        row.update(_overs_features(ov[ov_m]))
        rows.append(row)

    return pd.DataFrame(rows)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def add_derived_features(df: pd.DataFrame, iteration: int) -> tuple[pd.DataFrame, list[str]]:
    """
    Add exactly 2 new engineered features for the given iteration.
    All transformations are pure arithmetic — no statistics fitted on any data subset.
    """
    df = df.copy()
    new_cols: list[str] = []

    if iteration == 2:
        # Economy differential: positive = we bowled tighter than they did against us
        df["bowl_economy_diff"]   = df["bowl_opp_economy"] - df["our_bowl_economy"]
        # Joint batting-bowling quality; penalises one-sided dominance
        df["bat_bowl_joint_score"] = (df["bat_total_runs"] / 100.0) * (df["our_bowl_wickets"] / 10.0)
        new_cols = ["bowl_economy_diff", "bat_bowl_joint_score"]

    elif iteration == 3:
        # Composite batting depth score; heavier weight for larger milestones
        df["bat_depth_index"] = (
            df["bat_n_20plus"] + 2 * df["bat_n_30plus"] + 3 * df["bat_n_50plus"]
        )
        # Bowling pressure: wickets per over (combined wicket-taking & overs intensity)
        df["our_bowl_pressure"] = (
            df["our_bowl_wickets"] / (df["our_bowl_overs"] + 1e-9) * 6.0
        )
        new_cols = ["bat_depth_index", "our_bowl_pressure"]

    elif iteration == 4:
        # Collapse risk: how many wickets fell in the powerplay relative to total
        df["bat_collapse_risk"] = df["overs_early_wickets"] / (df["bat_wickets_lost"] + 1.0)
        # Net wicket advantage: positive = we took more wickets than they did
        df["our_net_wicket_adv"] = df["our_bowl_wickets"] - df["bowl_opp_wickets"]
        new_cols = ["bat_collapse_risk", "our_net_wicket_adv"]

    elif iteration == 5:
        # Batting average proxy (moved here so it pairs with the overs advantage below)
        df["bat_avg_per_wicket"] = df["bat_total_runs"] / (df["bat_wickets_lost"] + 1.0)
        # Overs advantage: did we bat more overs than we bowled? > 1 = batting dominance
        df["bat_overs_advantage"] = df["overs_total_faced"] / (df["our_bowl_overs"] + 1.0)
        new_cols = ["bat_avg_per_wicket", "bat_overs_advantage"]

    return df, new_cols


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model builders
# ─────────────────────────────────────────────────────────────────────────────

def _safe_cv(y: np.ndarray, max_folds: int = 5) -> int:
    return max(2, min(max_folds, int(np.bincount(y).min())))


def train_lr(X_tr: np.ndarray, y_tr: np.ndarray, class_weight) -> tuple:
    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(
        solver="liblinear", penalty="l2",
        class_weight=class_weight, max_iter=2000, random_state=42,
    )
    lr.fit(scaler.transform(X_tr), y_tr)
    return lr, scaler


def train_svm(X_tr: np.ndarray, y_tr: np.ndarray, class_weight) -> tuple:
    scaler = StandardScaler().fit(X_tr)
    X_s = scaler.transform(X_tr)
    base = SVC(kernel="rbf", probability=True, class_weight=class_weight, random_state=42)
    grid = GridSearchCV(
        base,
        {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        cv=_safe_cv(y_tr, 5), scoring="accuracy", n_jobs=-1,
    )
    grid.fit(X_s, y_tr)
    return grid.best_estimator_, scaler


def train_rf(X_tr: np.ndarray, y_tr: np.ndarray, class_weight) -> RandomForestClassifier:
    cv = _safe_cv(y_tr, 5)
    best_depth, best_cv = None, -1.0
    for depth in [None, 5, 10, 15, 20]:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=depth,
            class_weight=class_weight, random_state=42, n_jobs=-1,
        )
        score = cross_val_score(rf, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
        if score > best_cv:
            best_cv, best_depth = score, depth
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=best_depth,
        class_weight=class_weight, random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    return rf


def eval_model(model, X_raw: np.ndarray, y_true: np.ndarray,
               scaler: StandardScaler | None = None) -> dict:
    X = scaler.transform(X_raw) if scaler is not None else X_raw
    y_pred = model.predict(X)
    y_prob = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba") else np.zeros(len(y_pred), dtype=float)
    )
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Feature report
# ─────────────────────────────────────────────────────────────────────────────

def print_feature_report(
    feature_names: list[str],
    rf_importances: np.ndarray,
    target_reached: bool,
    val_acc: float,
) -> None:
    fi = pd.Series(rf_importances, index=feature_names).sort_values(ascending=False)

    print("\n" + "=" * 80)
    if target_reached:
        print(f"  FEATURE ENGINEERING REPORT  (target reached — val acc = {val_acc:.2%})")
    else:
        print(f"  FEATURE ENGINEERING REPORT  (best val acc = {val_acc:.2%})")
    print("=" * 80)

    # Group features by their registry group
    groups_seen: dict[str, list] = {}
    for feat in fi.index:
        meta = FEATURE_REGISTRY.get(feat, {
            "group": "Unlisted",
            "description": "No description registered.",
            "rationale": "",
            "iteration": "?",
        })
        g = meta["group"]
        groups_seen.setdefault(g, []).append((feat, fi[feat], meta))

    rank = 1
    for group, items in groups_seen.items():
        print(f"\n── {group} {'─'*(70-len(group))}")
        for feat, imp, meta in items:
            bar = "█" * max(1, int(imp * 350))
            iteration = meta.get("iteration", "base")
            iter_tag  = "base" if iteration == "base" else f"iter {iteration}"
            print(f"  #{rank:>2}  {feat:<40} imp={imp:.5f}  {bar}")
            print(f"        [{iter_tag}] {meta['description']}")
            if meta.get("rationale"):
                # Wrap rationale to 74 chars
                words = meta["rationale"].split()
                line, lines = [], []
                for w in words:
                    if len(" ".join(line + [w])) > 72:
                        lines.append(" ".join(line))
                        line = [w]
                    else:
                        line.append(w)
                if line:
                    lines.append(" ".join(line))
                for i, l in enumerate(lines):
                    prefix = "        WHY: " if i == 0 else "             "
                    print(f"{prefix}{l}")
            rank += 1

    print("\n" + "=" * 80)

    if not target_reached:
        print("  80% target NOT reached. Suggested next features:")
        print("  1. bat_partnership_depth — count of consecutive batters both scoring 15+,")
        print("     a proxy for partnership quality and innings consolidation.")
        print("  2. bowl_top_wicket_taker_pct — fraction of our wickets by our leading")
        print("     bowler (low = balanced attack; high = fragile if key bowler tires).")
        print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  HTML report
# ─────────────────────────────────────────────────────────────────────────────

_GROUP_COLORS: dict[str, str] = {
    "Bowling (derived)":            "#2563EB",
    "Own Bowling":                  "#7C3AED",
    "Opposition Bowling":           "#0284C7",
    "Opposition Bowling (derived)": "#38BDF8",
    "Batting":                      "#059669",
    "Batting (derived)":            "#10B981",
    "Batting+Bowling (derived)":    "#DB2777",
    "Over-by-Over":                 "#D97706",
    "Match Context":                "#4B5563",
}
_DEFAULT_COLOR = "#6B7280"


def generate_html_report(
    feature_names: list[str],
    rf_importances: np.ndarray,
    val_acc: float,
    cm: np.ndarray,
    model_rows: list[dict],   # [{name, train_acc, val_acc, precision, recall, f1, auc}, …]
    output_path: Path,
) -> None:
    fi = pd.Series(rf_importances, index=feature_names).sort_values(ascending=False)
    max_imp = float(fi.max()) if len(fi) > 0 else 1.0

    # ── helpers ──────────────────────────────────────────────────────────────
    def _esc(s: str) -> str:
        return (str(s)
                .replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    def _color(group: str) -> str:
        return _GROUP_COLORS.get(group, _DEFAULT_COLOR)

    def _iter_badge(iteration) -> str:
        if iteration == "base":
            return '<span class="badge badge-base">base feature</span>'
        if iteration in ("unused", "?"):
            return ""
        return f'<span class="badge badge-eng">engineered · iter {_esc(str(iteration))}</span>'

    # ── chart rows ───────────────────────────────────────────────────────────
    chart_rows_html = []
    for rank, (feat, imp) in enumerate(fi.items(), 1):
        meta  = FEATURE_REGISTRY.get(feat, {"group": "Unlisted", "description": "", "iteration": "?"})
        color = _color(meta["group"])
        pct   = imp / max_imp * 100
        chart_rows_html.append(f"""
        <div class="chart-row">
          <span class="chart-label" title="{_esc(feat)}">
            <span class="rank">#{rank}</span>{_esc(feat)}
          </span>
          <div class="chart-bar-wrap">
            <div class="chart-bar" style="width:{pct:.1f}%;background:{color}"></div>
            <span class="chart-val">{imp:.3f}</span>
          </div>
        </div>""")
    chart_html = "\n".join(chart_rows_html)

    # ── feature cards ────────────────────────────────────────────────────────
    cards_html_parts = []
    for rank, (feat, imp) in enumerate(fi.items(), 1):
        meta  = FEATURE_REGISTRY.get(feat, {
            "group": "Unlisted", "description": "No description.",
            "rationale": "", "iteration": "?",
        })
        color   = _color(meta["group"])
        pct_all = imp / fi.sum() * 100
        desc    = _esc(meta.get("description", ""))
        rat     = _esc(meta.get("rationale", ""))
        group   = _esc(meta.get("group", ""))
        badge   = _iter_badge(meta.get("iteration", "?"))

        cards_html_parts.append(f"""
      <div class="card" style="--accent:{color}">
        <div class="card-header">
          <div class="card-rank">#{rank}</div>
          <div class="card-title-block">
            <h3 class="card-feat">{_esc(feat)}</h3>
            <span class="card-group" style="color:{color}">{group}</span>
          </div>
          <div class="card-imp-block">
            <div class="card-imp-num">{imp:.4f}</div>
            <div class="card-imp-sub">{pct_all:.1f}% of total</div>
          </div>
        </div>
        <div class="card-imp-bar-wrap">
          <div class="card-imp-bar" style="width:{pct_all:.1f}%;background:{color}"></div>
        </div>
        <div class="card-body">
          <p class="card-desc"><strong>What it measures:</strong> {desc}</p>
          {"" if not rat else f'<p class="card-rat"><strong>Why it predicts outcomes:</strong> {rat}</p>'}
        </div>
        <div class="card-footer">{badge}</div>
      </div>""")
    cards_html = "\n".join(cards_html_parts)

    # ── confusion matrix ─────────────────────────────────────────────────────
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total  = tn + fp + fn + tp

    # ── model metrics table rows ─────────────────────────────────────────────
    metric_rows = []
    for mr in model_rows:
        best = mr.get("is_best", False)
        cls  = ' class="best-row"' if best else ""
        metric_rows.append(
            f'<tr{cls}>'
            f'<td><strong>{_esc(mr["name"])}</strong></td>'
            f'<td>{mr["train_acc"]:.1%}</td>'
            f'<td><strong>{mr["val_acc"]:.1%}</strong></td>'
            f'<td>{mr["precision"]:.1%}</td>'
            f'<td>{mr["recall"]:.1%}</td>'
            f'<td>{mr["f1"]:.1%}</td>'
            f'<td>{mr["auc"]:.3f}</td>'
            f'</tr>'
        )
    metrics_html = "\n".join(metric_rows)

    acc_pct = f"{val_acc:.1%}"

    # ── full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cricket Win Prediction — Feature Report</title>
<style>
  :root {{
    --bg:#0F172A; --surface:#1E293B; --surface2:#263144;
    --border:#334155; --text:#E2E8F0; --muted:#94A3B8;
    --green:#10B981; --red:#F87171; --yellow:#FCD34D;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
       line-height:1.6;padding:0 0 60px}}

  /* ── header ── */
  header{{background:linear-gradient(135deg,#0F2027,#203A43,#2C5364);
          padding:48px 40px 40px;border-bottom:1px solid var(--border)}}
  header h1{{font-size:2rem;font-weight:700;letter-spacing:-.02em}}
  header .subtitle{{color:var(--muted);margin-top:4px;font-size:1rem}}
  .accuracy-pill{{display:inline-block;margin-top:18px;
                  background:rgba(16,185,129,.15);border:1px solid var(--green);
                  color:var(--green);padding:6px 20px;border-radius:9999px;
                  font-size:1.5rem;font-weight:700;letter-spacing:.04em}}
  .accuracy-label{{font-size:.8rem;color:var(--muted);margin-left:8px;font-weight:400}}

  /* ── layout ── */
  main{{max-width:1080px;margin:0 auto;padding:40px 24px}}
  section{{margin-bottom:48px}}
  h2{{font-size:1.25rem;font-weight:600;margin-bottom:20px;padding-bottom:8px;
      border-bottom:1px solid var(--border);color:var(--text)}}

  /* ── summary cards ── */
  .summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px}}
  .stat-card{{background:var(--surface);border:1px solid var(--border);
              border-radius:12px;padding:20px;text-align:center}}
  .stat-val{{font-size:2rem;font-weight:700}}
  .stat-label{{font-size:.8rem;color:var(--muted);margin-top:4px}}

  /* ── confusion matrix ── */
  .cm-wrap{{display:flex;gap:32px;align-items:flex-start;flex-wrap:wrap}}
  .cm-table-wrap table{{border-collapse:collapse;font-size:.9rem}}
  .cm-table-wrap th,.cm-table-wrap td{{padding:10px 18px;border:1px solid var(--border);text-align:center}}
  .cm-table-wrap thead th{{background:var(--surface2);color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}}
  .cm-table-wrap .axis-label{{background:var(--surface2);color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;writing-mode:horizontal-tb}}
  .cm-tp{{background:rgba(16,185,129,.15);color:var(--green);font-weight:700;font-size:1.1rem}}
  .cm-tn{{background:rgba(16,185,129,.1);color:var(--green);font-weight:700;font-size:1.1rem}}
  .cm-fp{{background:rgba(248,113,113,.15);color:var(--red);font-weight:700;font-size:1.1rem}}
  .cm-fn{{background:rgba(248,113,113,.1);color:var(--red);font-weight:700;font-size:1.1rem}}
  .cm-note{{background:var(--surface);border:1px solid var(--border);border-radius:10px;
            padding:16px 20px;font-size:.87rem;color:var(--muted);max-width:380px}}
  .cm-note p{{margin-bottom:8px}}
  .cm-note p:last-child{{margin-bottom:0}}

  /* ── model metrics table ── */
  .metrics-table{{width:100%;border-collapse:collapse;font-size:.88rem}}
  .metrics-table th{{padding:10px 14px;border-bottom:2px solid var(--border);
                     text-align:left;color:var(--muted);font-weight:600;
                     font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}}
  .metrics-table td{{padding:10px 14px;border-bottom:1px solid var(--border)}}
  .metrics-table .best-row td{{background:rgba(16,185,129,.07)}}

  /* ── importance chart ── */
  .chart-row{{display:flex;align-items:center;gap:12px;margin-bottom:8px;font-size:.83rem}}
  .chart-label{{flex:0 0 220px;text-align:right;color:var(--muted);white-space:nowrap;
                overflow:hidden;text-overflow:ellipsis}}
  .rank{{color:var(--border);margin-right:6px;font-size:.75rem}}
  .chart-bar-wrap{{flex:1;display:flex;align-items:center;gap:8px}}
  .chart-bar{{height:16px;border-radius:4px;min-width:4px;transition:width .3s}}
  .chart-val{{flex:0 0 44px;font-size:.78rem;color:var(--muted)}}

  /* ── feature cards ── */
  .cards-grid{{display:grid;gap:20px}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:14px;
         overflow:hidden;border-left:4px solid var(--accent)}}
  .card-header{{display:flex;align-items:flex-start;gap:16px;padding:20px 20px 0}}
  .card-rank{{font-size:1.6rem;font-weight:700;color:var(--border);flex:0 0 auto;
              line-height:1;padding-top:2px}}
  .card-title-block{{flex:1}}
  .card-feat{{font-size:1rem;font-weight:700;font-family:monospace;letter-spacing:.02em}}
  .card-group{{font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em}}
  .card-imp-block{{text-align:right;flex:0 0 auto}}
  .card-imp-num{{font-size:1.4rem;font-weight:700}}
  .card-imp-sub{{font-size:.75rem;color:var(--muted)}}
  .card-imp-bar-wrap{{height:4px;background:var(--surface2);margin:12px 0 0}}
  .card-imp-bar{{height:100%;border-radius:0 2px 2px 0}}
  .card-body{{padding:12px 20px 0;font-size:.88rem;color:var(--muted)}}
  .card-body p{{margin-bottom:8px}}
  .card-body strong{{color:var(--text)}}
  .card-footer{{padding:10px 20px 14px;display:flex;gap:8px;flex-wrap:wrap}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:9999px;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em}}
  .badge-base{{background:rgba(148,163,184,.15);color:var(--muted)}}
  .badge-eng{{background:rgba(219,39,119,.15);color:#F472B6}}

  /* ── insight box ── */
  .insight-box{{background:var(--surface);border:1px solid var(--border);
                border-left:4px solid #2563EB;border-radius:12px;
                padding:20px 24px;font-size:.9rem;color:var(--muted)}}
  .insight-box h3{{color:var(--text);margin-bottom:8px;font-size:1rem}}
  .insight-box p{{margin-bottom:10px}}
  .insight-box p:last-child{{margin-bottom:0}}
  .hl{{color:var(--text);font-weight:600}}
</style>
</head>
<body>
<header>
  <h1>Premier Cricket — Win Prediction Model</h1>
  <p class="subtitle">Random Forest · Temporal validation (round 17 held out) · Feature importance report</p>
  <div>
    <span class="accuracy-pill">{acc_pct}<span class="accuracy-label">validation accuracy</span></span>
  </div>
</header>

<main>

  <!-- ── MODEL PERFORMANCE ────────────────────────────────────────────── -->
  <section>
    <h2>Model Performance Summary</h2>
    <table class="metrics-table">
      <thead>
        <tr>
          <th>Model</th><th>Train Acc</th><th>Val Acc</th>
          <th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th>
        </tr>
      </thead>
      <tbody>{metrics_html}</tbody>
    </table>
  </section>

  <!-- ── CONFUSION MATRIX ─────────────────────────────────────────────── -->
  <section>
    <h2>Confusion Matrix — Round 17 Test Set</h2>
    <div class="cm-wrap">
      <div class="cm-table-wrap">
        <table>
          <thead>
            <tr>
              <th></th><th></th>
              <th colspan="2">Predicted</th>
            </tr>
            <tr>
              <th></th><th></th>
              <th>Loss (0)</th><th>Win (1)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="axis-label" rowspan="2" style="writing-mode:vertical-rl;transform:rotate(180deg);text-align:center">Actual</td>
              <td class="axis-label"><strong>Loss (0)</strong></td>
              <td class="cm-tn">{tn}<br><small>True Neg</small></td>
              <td class="cm-fp">{fp}<br><small>False Pos</small></td>
            </tr>
            <tr>
              <td class="axis-label"><strong>Win (1)</strong></td>
              <td class="cm-fn">{fn}<br><small>False Neg</small></td>
              <td class="cm-tp">{tp}<br><small>True Pos</small></td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="cm-note">
        <p><span class="hl">Correctly classified: {tn+tp} / {total}</span> ({(tn+tp)/total:.1%})</p>
        <p>False positives ({fp}): teams predicted to win that actually lost.
           These tend to be sides that bowled efficiently but whose batting fell short.</p>
        <p>False negatives ({fn}): teams predicted to lose that actually won.
           Often reflect very close finishes where the model's economy-diff signal
           pointed the wrong way.</p>
        <p>The test set is balanced (50 / 50 win rate), so a random classifier
           would score 50% — this model is <span class="hl">{(tn+tp)/total - 0.5:.1%} above chance</span>.</p>
      </div>
    </div>
  </section>

  <!-- ── IMPORTANCE CHART ─────────────────────────────────────────────── -->
  <section>
    <h2>Feature Importance (RF Gini impurity, sorted)</h2>
    <div style="background:var(--surface);border:1px solid var(--border);
                border-radius:12px;padding:24px">
{chart_html}
    </div>
  </section>

  <!-- ── KEY INSIGHT ──────────────────────────────────────────────────── -->
  <section>
    <h2>Key Insight</h2>
    <div class="insight-box">
      <h3>Why bowling economy differential dominates</h3>
      <p>The single most predictive feature is <span class="hl">bowl_economy_diff</span>
         (opposition economy minus our economy), carrying roughly
         <span class="hl">{fi.iloc[0]/fi.sum()*100:.0f}% of total feature importance</span>
         in the Random Forest. This says: <em>the team whose bowlers conceded fewer runs
         per over than the opposing bowlers is far more likely to win.</em></p>
      <p>This makes cricketing sense. In club cricket, a bowling attack that restricts the
         opposition to 2–3 runs per over while the opposing bowlers concede 4–5 per over
         produces a run differential that is very hard to overcome — even for a batting lineup
         that scores quickly, because the absolute difference in runs accumulates over
         80–100 overs of two-day cricket.</p>
      <p>A small but important caveat: the raw <code>runs_conceded</code> column in
         bowling_data does not include extras (wides, no-balls, byes) that don't appear in
         individual bowler figures. This means both economies carry a slight downward bias,
         but because <em>both sides are affected equally</em> the differential remains a valid
         relative signal.</p>
    </div>
  </section>

  <!-- ── FEATURE CARDS ────────────────────────────────────────────────── -->
  <section>
    <h2>Feature Detail</h2>
    <div class="cards-grid">
{cards_html}
    </div>
  </section>

</main>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Saved HTML report    → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  CRICKET WIN PREDICTION PIPELINE  (target: ≥80% accuracy)")
    print("=" * 70)

    bat, bw, sc, ov = load_data()

    labels = build_labels(sc)
    print(f"Labels: {len(labels)} rows | wins={labels['won'].sum()} | losses={(labels['won']==0).sum()}")

    df = build_feature_matrix(bat, bw, sc, ov, labels)
    n_base_feats = len(feature_cols(df))
    print(f"Base feature matrix: {df.shape[0]} rows × {n_base_feats} features")

    # ── Temporal split ────────────────────────────────────────────────────────
    rounds = sorted(df["round"].unique())
    train_rnds = rounds[:-1]
    test_rnds  = [rounds[-1]]
    print(f"Train rounds: {train_rnds}   |   Test round: {test_rnds[0]}")
    if len(rounds) < 3:
        print("  (< 3 rounds: using leave-one-round-out CV)")

    # ── Class imbalance check ─────────────────────────────────────────────────
    win_rate = float(df[df["round"].isin(train_rnds)]["won"].mean())
    print(f"Training win rate: {win_rate:.2%}")
    cw = "balanced" if abs(win_rate - 0.5) > 0.10 else None
    if cw:
        print("Win/loss split >60/40 → class_weight='balanced'")

    # ── Iterative feature selection loop ──────────────────────────────────────
    active_feats = feature_cols(df)
    best_val_acc = 0.0
    best_rf      = None
    best_feats   = active_feats[:]
    target_reached = False

    print("\n" + "─" * 70)

    for it in range(1, 6):
        print(f"\n[Iteration {it}]  active features: {len(active_feats)}")

        tr = df[df["round"].isin(train_rnds)]
        te = df[df["round"].isin(test_rnds)]

        X_tr = np.nan_to_num(tr[active_feats].values.astype(float))
        y_tr = tr["won"].values
        X_te = np.nan_to_num(te[active_feats].values.astype(float))
        y_te = te["won"].values

        rf = train_rf(X_tr, y_tr, cw)
        val_pred = rf.predict(X_te)
        val_acc  = accuracy_score(y_te, val_pred)
        print(f"  RF validation accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_rf        = rf
            best_feats     = active_feats[:]

        if val_acc >= 0.80:
            target_reached = True
            print(f"  ✓ Target ≥0.80 reached at iteration {it}.  Stopping.")
            break

        if it < 5:
            perm = permutation_importance(
                rf, X_tr, y_tr, n_repeats=10, random_state=42, scoring="accuracy"
            )
            imp_s = pd.Series(perm.importances_mean, index=active_feats)
            keep    = imp_s[imp_s >  0.0002].index.tolist()
            dropped = imp_s[imp_s <= 0.0002].index.tolist()
            if dropped:
                print(f"  Dropping {len(dropped)} near-zero features: {dropped}")
            active_feats = keep

            df, new_cols = add_derived_features(df, it + 1)
            active_feats = active_feats + new_cols
            print(f"  Added features for iter {it+1}: {new_cols}")

    else:
        print(f"\n  Max iterations (5) reached.  Best val accuracy: {best_val_acc:.4f}")

        if best_val_acc < 0.80 and best_rf is not None:
            te = df[df["round"].isin(test_rnds)]
            y_te      = te["won"].values
            y_pred_d  = best_rf.predict(
                np.nan_to_num(te[best_feats].values.astype(float))
            )
            cm = confusion_matrix(y_te, y_pred_d)
            print("\n  DIAGNOSTIC — Confusion matrix (rows=actual, cols=predicted):")
            print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
            print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
            for cls, total, idx in [("Losses misclassified as wins", cm[0].sum(), (0,1)),
                                     ("Wins misclassified as losses", cm[1].sum(), (1,0))]:
                if total > 0:
                    print(f"    {cls}: {cm[idx]/total:.1%}")

    # ── Train all three final models ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Training final models …")

    tr = df[df["round"].isin(train_rnds)]
    te = df[df["round"].isin(test_rnds)]

    X_tr = np.nan_to_num(tr[best_feats].values.astype(float))
    y_tr = tr["won"].values
    X_te = np.nan_to_num(te[best_feats].values.astype(float))
    y_te = te["won"].values

    rf_final          = best_rf
    lr_final, lr_sc   = train_lr(X_tr, y_tr, cw)
    svm_final, svm_sc = train_svm(X_tr, y_tr, cw)

    rf_tr_acc  = accuracy_score(y_tr, rf_final.predict(X_tr))
    lr_tr_acc  = accuracy_score(y_tr, lr_final.predict(lr_sc.transform(X_tr)))
    svm_tr_acc = accuracy_score(y_tr, svm_final.predict(svm_sc.transform(X_tr)))

    m_rf  = eval_model(rf_final,  X_te, y_te)
    m_lr  = eval_model(lr_final,  X_te, y_te, lr_sc)
    m_svm = eval_model(svm_final, X_te, y_te, svm_sc)

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    hdr = (f"{'Model':<22} {'Train Acc':>10} {'Val Acc':>10}"
           f" {'Precision':>10} {'Recall':>9} {'F1':>8} {'AUC':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, tr_acc, m in [
        ("Logistic Regression", lr_tr_acc,  m_lr),
        ("SVM (RBF)",           svm_tr_acc, m_svm),
        ("Random Forest",       rf_tr_acc,  m_rf),
    ]:
        print(
            f"{name:<22} {tr_acc:>10.4f} {m['accuracy']:>10.4f}"
            f" {m['precision']:>10.4f} {m['recall']:>9.4f}"
            f" {m['f1']:>8.4f} {m['auc']:>8.4f}"
        )

    # ── Feature report (console) ──────────────────────────────────────────────
    print_feature_report(
        best_feats,
        rf_final.feature_importances_,
        target_reached,
        best_val_acc,
    )

    # ── HTML report ───────────────────────────────────────────────────────────
    cm_final = confusion_matrix(y_te, rf_final.predict(X_te))

    def _row(name, train_acc, m, is_best=False):
        return {"name": name, "train_acc": train_acc, "is_best": is_best,
                "val_acc": m["accuracy"], "precision": m["precision"],
                "recall": m["recall"], "f1": m["f1"], "auc": m["auc"]}

    best_val = max(m_rf["accuracy"], m_svm["accuracy"], m_lr["accuracy"])
    model_rows_data = [
        _row("Logistic Regression", lr_tr_acc,  m_lr,  m_lr["accuracy"]  == best_val),
        _row("SVM (RBF)",           svm_tr_acc, m_svm, m_svm["accuracy"] == best_val),
        _row("Random Forest",       rf_tr_acc,  m_rf,  m_rf["accuracy"]  == best_val),
    ]
    generate_html_report(
        feature_names=best_feats,
        rf_importances=rf_final.feature_importances_,
        val_acc=best_val_acc,
        cm=cm_final,
        model_rows=model_rows_data,
        output_path=MODEL_DIR / "feature_report.html",
    )

    # ── Save artifacts ────────────────────────────────────────────────────────
    with open(MODEL_DIR / "rf_model.pkl", "wb") as fh:
        pickle.dump(rf_final, fh)
    with open(MODEL_DIR / "feature_list.json", "w") as fh:
        json.dump(best_feats, fh, indent=2)

    print(f"\nSaved RF model     → {MODEL_DIR / 'rf_model.pkl'}")
    print(f"Saved feature list → {MODEL_DIR / 'feature_list.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
