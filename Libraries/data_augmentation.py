import pandas as pd
import numpy as np

def add_ball_by_ball_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - partnership: sum of (runs+extras) since last wicket; written only on the wicket row,
                     and on the final row of the innings if no wicket ends it.
      - total_score: running total of (runs+extras) within the innings (bottom-up chronological).
      - runs_per_over: total (runs+extras) in that over; written only on the first row of that over
                       in the CURRENT sort order (i.e., the top-most row for that over).
    Resets counts per batting innings.

    Required cols (new names): ['matchup','current_batting','over_num','runs','wicket_flag']
    Extras cols expected: ['wides','noballs','byes','legbyes']
    Optional: ['round','grade'] will be included in the grouping if present.
    """

    df = df.copy()

    # ---- normalize column names (so your old logic still works cleanly) ----
    # If you've already renamed upstream, you can remove this block.
    rename_map = {}
    if 'batting_team' in df.columns and 'current_batting' not in df.columns:
        rename_map['batting_team'] = 'current_batting'
    if 'over' in df.columns and 'over_num' not in df.columns:
        rename_map['over'] = 'over_num'
    if 'wicket' in df.columns and 'wicket_flag' not in df.columns:
        rename_map['wicket'] = 'wicket_flag'
    if rename_map:
        df = df.rename(columns=rename_map)

    # ---- ensure numeric ----
    for c in ['runs', 'wides', 'noballs', 'byes', 'legbyes']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
        else:
            # if a column is missing, treat as 0 extras (safer than crashing)
            df[c] = 0

    df['wicket_flag'] = pd.to_numeric(df['wicket_flag'], errors='coerce').fillna(0).astype(int)
    df['over_num'] = pd.to_numeric(df['over_num'], errors='coerce').fillna(-1).astype(int)

    # ---- per-ball total credited to batting side ----
    df['ball_total'] = (
        df['runs'] + df['wides'] + df['noballs'] + df['byes'] + df['legbyes']
    ).astype(int)

    # Grouping keys per innings
    keys = [c for c in ['round', 'grade', 'matchup', 'current_batting'] if c in df.columns]

    # Prepare new columns
    df['partnership'] = np.nan
    df['total_score'] = np.nan
    df['runs_per_over'] = np.nan

    def _per_innings(g: pd.DataFrame) -> pd.DataFrame:
        idx = g.index.to_numpy()

        # total_score: bottom-up cumsum of ball_total
        rev = g.loc[idx[::-1], 'ball_total'].cumsum()
        total_score = rev.iloc[::-1].to_numpy()
        g.loc[idx, 'total_score'] = total_score

        # partnership: bottom-up accumulate ball_total until wicket, write only at wicket row.
        partner_vals = np.full(len(g), np.nan, dtype=float)
        acc = 0
        for pos, i in enumerate(idx[::-1]):  # bottom → top
            acc += int(g.at[i, 'ball_total'])
            if int(g.at[i, 'wicket_flag']) == 1:
                partner_vals[len(g) - 1 - pos] = acc
                acc = 0

        # If innings ended without a wicket on the top-most row, write partnership on that top row
        if np.isnan(partner_vals[0]) and acc > 0:
            partner_vals[0] = acc

        g['partnership'] = partner_vals

        # runs_per_over: total ball_total per over, write only on first row of that over (current order)
        over_totals = g.groupby('over_num', sort=False)['ball_total'].sum()
        first_idx_per_over = g.groupby('over_num', sort=False).head(1).index
        g.loc[first_idx_per_over, 'runs_per_over'] = g.loc[first_idx_per_over, 'over_num'].map(over_totals)

        return g

    if not keys:
        df = _per_innings(df)
    else:
        df = df.groupby(keys, group_keys=False, sort=False).apply(_per_innings)

    # Cast totals to int where filled
    for col in ['partnership', 'total_score', 'runs_per_over']:
        df[col] = df[col].astype('Int64')

    # keep your downstream step
    df = add_wickets_down(df)

    # optional: drop helper column if you don’t want it in output
    df = df.drop(columns=['ball_total'])

    return df


def add_wickets_down(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'wickets_down' as a running sum of 'wicket_flag' within each innings.
    Computed bottom-up to match your chronology.

    Ensures wickets_down does NOT merge across different:
      - round
      - grade
      - matchup
      - batting team (current_batting or batting_team)
    """
    out = df.copy()

    # Normalize batting team column name
    if 'current_batting' in out.columns and 'batting_team' not in out.columns:
        out = out.rename(columns={'current_batting': 'batting_team'})

    required = ['matchup', 'batting_team', 'wicket_flag']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"add_wickets_down missing required columns: {missing}. Present: {list(out.columns)}")

    out['wicket_flag'] = pd.to_numeric(out['wicket_flag'], errors='coerce').fillna(0).astype(int)

    # Strict grouping keys: use these if present, but NEVER drop matchup/batting_team
    keys = []
    if 'round' in out.columns:
        keys.append('round')
    if 'grade' in out.columns:
        keys.append('grade')
    keys += ['matchup', 'batting_team']

    def _per_innings(g: pd.DataFrame) -> pd.DataFrame:
        idx = g.index.to_numpy()
        rev_cum = g.loc[idx[::-1], 'wicket_flag'].cumsum()
        g.loc[idx, 'wickets_down'] = rev_cum.iloc[::-1].to_numpy()
        return g

    out = out.groupby(keys, group_keys=False, sort=False).apply(_per_innings)
    out['wickets_down'] = out['wickets_down'].astype('Int64')
    return out


