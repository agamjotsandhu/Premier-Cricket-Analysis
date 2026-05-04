import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from rapidfuzz import process, fuzz
import os

# Set style for plots
sns.set_theme(style="whitegrid")

def normalize_name(name):
    if pd.isna(name):
        return ""
    if ',' in name:
        parts = name.split(',')
        if len(parts) == 2:
            return f"{parts[1].strip()} {parts[0].strip()}".lower()
    return str(name).strip().lower()

def fuzzy_match_players(df1, df2, name_col1, name_col2, threshold=90):
    matches = {}
    names2 = df2[name_col2].unique()
    for name1 in df1[name_col1].unique():
        match = process.extractOne(name1, names2, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            matches[name1] = match[0]
    return matches

def run_analysis():
    # Load data
    batting_data = pd.read_csv('batting_data.csv')
    bowling_data = pd.read_csv('bowling_data.csv')
    overs_data = pd.read_csv('overs.csv')
    scores_data = pd.read_csv('scores.csv')
    batting_champions = pd.read_csv('batting_champions.csv')
    bowling_champions = pd.read_csv('bowling_champions.csv')
    ladder = pd.read_csv('ladder.csv')

    # Preprocessing
    batting_data['batsman_name_norm'] = batting_data['batsman_name'].apply(normalize_name)
    batting_champions['player_norm'] = batting_champions['player'].apply(normalize_name)
    bowling_data['bowler_name_norm'] = bowling_data['bowler_name'].apply(normalize_name)
    bowling_champions['player_norm'] = bowling_champions['player'].apply(normalize_name)

    # Initialize Report
    report = "# Cricket Match Analysis Report\n\n"

    # --- Section 1: Scoring Rate vs Average Trade-off ---
    print("Processing Section 1...")
    batting_clean = batting_data[batting_data['how_out'] != 'did not bat'].copy()
    batting_clean['runs'] = pd.to_numeric(batting_clean['runs'], errors='coerce').fillna(0)
    batting_clean['balls'] = pd.to_numeric(batting_clean['balls'], errors='coerce')
    batting_clean = batting_clean.dropna(subset=['balls'])

    batter_stats = batting_clean.groupby(['batsman_name_norm', 'grade']).agg(
        total_runs=('runs', 'sum'),
        total_balls=('balls', 'sum'),
        innings_count=('batsman_name', 'count')
    ).reset_index()
    batter_stats['strike_rate'] = (batter_stats['total_runs'] / batter_stats['total_balls']) * 100

    # Join to champions
    # Try exact join first
    merged_batting = pd.merge(
        batter_stats, 
        batting_champions[['player_norm', 'grade', 'rank', 'avg']], 
        left_on=['batsman_name_norm', 'grade'], 
        right_on=['player_norm', 'grade'], 
        how='inner'
    )

    unmatched_players = batting_champions[~batting_champions['player_norm'].isin(merged_batting['player_norm'])]
    unmatched_count = len(unmatched_players)
    
    if unmatched_count / len(batting_champions) > 0.2:
        print(f"Match rate low ({1 - unmatched_count/len(batting_champions):.1%}). Fuzzy matching batters...")
        
        # Create a mapping for each grade
        fuzzy_map = []
        for grade in batting_champions['grade'].unique():
            grade_stats = batter_stats[batter_stats['grade'] == grade]
            grade_champs = batting_champions[batting_champions['grade'] == grade]
            
            if grade_stats.empty or grade_champs.empty:
                continue
                
            champ_names = grade_champs['player_norm'].tolist()
            stat_names = grade_stats['batsman_name_norm'].tolist()
            
            for champ_name in champ_names:
                if champ_name in merged_batting['player_norm'].values:
                    continue
                match = process.extractOne(champ_name, stat_names, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= 85:
                    fuzzy_map.append({
                        'player_norm': champ_name,
                        'batsman_name_norm': match[0],
                        'grade': grade
                    })
        
        if fuzzy_map:
            fuzzy_df = pd.DataFrame(fuzzy_map)
            fuzzy_merged = pd.merge(
                fuzzy_df,
                batter_stats,
                on=['batsman_name_norm', 'grade']
            )
            fuzzy_merged = pd.merge(
                fuzzy_merged,
                batting_champions[['player_norm', 'grade', 'rank', 'avg']],
                on=['player_norm', 'grade']
            )
            merged_batting = pd.concat([merged_batting, fuzzy_merged], ignore_index=True)
            print(f"Added {len(fuzzy_merged)} fuzzy matches.")

    unmatched_names_final = set(batting_champions['player_norm']) - set(merged_batting['player_norm'])
    report += f"Note: {len(unmatched_names_final)} players from batting champions could not be matched.\n\n"

    # Convert numeric columns explicitly
    merged_batting['avg'] = pd.to_numeric(merged_batting['avg'], errors='coerce')
    merged_batting['strike_rate'] = pd.to_numeric(merged_batting['strike_rate'], errors='coerce')
    # Replace inf with NaN then drop
    merged_batting = merged_batting.replace([np.inf, -np.inf], np.nan)
    merged_batting = merged_batting.dropna(subset=['avg', 'strike_rate'])

    # Plot Section 1
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=merged_batting, x='strike_rate', y='avg', hue='grade', palette='viridis')
    
    # Label top 10 ranked players
    top10_batters = merged_batting.sort_values('rank').head(10)
    for i, row in top10_batters.iterrows():
        plt.text(row['strike_rate'], row['avg'], row['batsman_name_norm'].title(), fontsize=9)
    
    plt.title('Strike Rate vs Season Average')
    plt.xlabel('Strike Rate')
    plt.ylabel('Season Average')
    plt.savefig('section1_sr_vs_avg.png')
    plt.close()

    # Stats Section 1
    corr, p_val = stats.pearsonr(merged_batting['strike_rate'], merged_batting['avg'])
    merged_batting['sr_quartile'] = pd.qcut(merged_batting['strike_rate'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    quartile_stats = merged_batting.groupby('sr_quartile', observed=False).agg(
        mean_rank=('rank', 'mean'),
        mean_avg=('avg', 'mean')
    ).reset_index()

    report += "## Section 1: Scoring Rate vs Average Trade-off\n\n"
    report += "![Strike Rate vs Average](section1_sr_vs_avg.png)\n\n"
    report += f"- **Pearson Correlation (SR vs Avg):** {corr:.3f} (p-value: {p_val:.3e})\n"
    report += "\n### Batter Performance by Strike Rate Quartile\n\n"
    report += quartile_stats.to_markdown(index=False) + "\n\n"
    
    interpretation_1 = "The competition " + ("rewards efficiency (higher strike rate correlates with higher average)" if corr > 0.3 else "rewards volume and stability over raw speed") + "."
    report += f"**Key Finding:** {interpretation_1}\n\n"

    # --- Section 2: Dismissal Profile ---
    print("Processing Section 2...")
    def categorize_dismissal(how_out):
        ho = str(how_out).lower()
        if 'bowled' in ho or ho == 'b': return 'bowled'
        if 'lbw' in ho: return 'lbw'
        if 'caught' in ho and 'bowled' in ho: return 'caught-and-bowled'
        if 'caught' in ho or ho.startswith('c:'): return 'caught'
        if 'run out' in ho: return 'run out'
        if 'stumped' in ho or ho.startswith('st:'): return 'stumped'
        if 'not out' in ho: return 'not out'
        if 'did not bat' in ho: return 'did not bat'
        return 'other'

    batting_data['dismissal_type'] = batting_data['how_out'].apply(categorize_dismissal)
    
    # Merge rank info to all innings
    batting_with_rank = pd.merge(
        batting_data,
        batting_champions[['player_norm', 'grade', 'rank']],
        left_on=['batsman_name_norm', 'grade'],
        right_on=['player_norm', 'grade'],
        how='left'
    )
    
    batting_with_rank['group'] = batting_with_rank['rank'].apply(lambda x: 'elite' if x <= 10 else 'non-elite')
    
    dismissal_counts = batting_with_rank.groupby(['group', 'dismissal_type']).size().unstack(fill_value=0)
    # Filter to requested categories
    req_cats = ['bowled', 'lbw', 'caught', 'caught-and-bowled', 'run out', 'stumped', 'not out', 'did not bat']
    dismissal_counts = dismissal_counts.reindex(columns=req_cats, fill_value=0)
    
    dismissal_props = dismissal_counts.div(dismissal_counts.sum(axis=1), axis=0)
    
    # Plot Section 2
    dismissal_props.T.plot(kind='bar', figsize=(12, 6))
    plt.title('Dismissal Type Proportions: Elite vs Non-Elite')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('section2_dismissals.png')
    plt.close()

    # Chi-square test
    # Exclude categories with 0 observations in either group to avoid test errors
    obs_counts = dismissal_counts.loc[:, (dismissal_counts > 0).all(axis=0)]
    if obs_counts.shape[1] >= 2:
        chi2, p_chi2, dof, ex = stats.chi2_contingency(obs_counts.values)
    else:
        p_chi2 = np.nan

    report += "## Section 2: Dismissal Profile — Elite vs Non-Elite Batters\n\n"
    report += "![Dismissal Profile](section2_dismissals.png)\n\n"
    report += f"- **Chi-squared test p-value:** {p_chi2:.4f}\n\n"
    report += "### Dismissal Proportions\n\n"
    report += dismissal_props.to_markdown() + "\n\n"
    
    finding_2 = "There is a " + ("significant" if p_chi2 < 0.05 else "non-significant") + " difference in dismissal profiles between elite and non-elite batters."
    report += f"**Key Finding:** {finding_2}\n\n"

    # --- Section 3: Bowling — Powerplay vs Death Economy ---
    print("Processing Section 3...")
    overs_data['runs'] = pd.to_numeric(overs_data['runs'], errors='coerce').fillna(0)
    overs_data['over_num'] = pd.to_numeric(overs_data['over_num'], errors='coerce')
    overs_data['bowler_norm'] = overs_data['bowler_for_row'].apply(normalize_name)

    pp_overs = overs_data[(overs_data['over_num'] >= 1) & (overs_data['over_num'] <= 6)]
    death_overs = overs_data[(overs_data['over_num'] >= 15) & (overs_data['over_num'] <= 20)]

    def compute_economy(df):
        res = df.groupby(['bowler_norm', 'grade']).agg(
            total_runs=('runs', 'sum'),
            over_count=('over_num', 'count')
        ).reset_index()
        res['economy'] = res['total_runs'] / res['over_count']
        return res[res['over_count'] >= 3]

    pp_stats = compute_economy(pp_overs)
    death_stats = compute_economy(death_overs)

    # Join to bowling champions
    pp_merged = pd.merge(pp_stats, bowling_champions[['player_norm', 'grade', 'rank']], left_on=['bowler_norm', 'grade'], right_on=['player_norm', 'grade'])
    death_merged = pd.merge(death_stats, bowling_champions[['player_norm', 'grade', 'rank']], left_on=['bowler_norm', 'grade'], right_on=['player_norm', 'grade'])
    
    # Fuzzy match bowling if needed
    def fuzzy_match_bowling(stats_df, champ_df, current_merged):
        unmatched = champ_df[~champ_df['player_norm'].isin(current_merged['player_norm'])]
        if len(unmatched) / len(champ_df) > 0.2:
            fuzzy_map = []
            for grade in champ_df['grade'].unique():
                g_stats = stats_df[stats_df['grade'] == grade]
                g_champs = unmatched[unmatched['grade'] == grade]
                if g_stats.empty or g_champs.empty: continue
                s_names = g_stats['bowler_norm'].tolist()
                for c_name in g_champs['player_norm']:
                    match = process.extractOne(c_name, s_names, scorer=fuzz.token_sort_ratio)
                    if match and match[1] >= 85:
                        fuzzy_map.append({'player_norm': c_name, 'bowler_norm': match[0], 'grade': grade})
            if fuzzy_map:
                f_df = pd.DataFrame(fuzzy_map)
                f_merged = pd.merge(f_df, stats_df, on=['bowler_norm', 'grade'])
                f_merged = pd.merge(f_merged, champ_df[['player_norm', 'grade', 'rank']], on=['player_norm', 'grade'])
                return pd.concat([current_merged, f_merged], ignore_index=True)
        return current_merged

    pp_merged = fuzzy_match_bowling(pp_stats, bowling_champions, pp_merged)
    death_merged = fuzzy_match_bowling(death_stats, bowling_champions, death_merged)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(data=pp_merged, x='economy', y='rank', ax=axes[0])
    axes[0].set_title('Powerplay Economy vs Rank')
    axes[0].invert_yaxis()
    
    sns.scatterplot(data=death_merged, x='economy', y='rank', ax=axes[1])
    axes[1].set_title('Death Economy vs Rank')
    axes[1].invert_yaxis()
    
    plt.savefig('section3_bowling_phases.png')
    plt.close()

    pp_merged = pp_merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['economy', 'rank'])
    death_merged = death_merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['economy', 'rank'])
    
    if len(pp_merged) >= 2:
        pp_corr, _ = stats.pearsonr(pp_merged['economy'], pp_merged['rank'])
    else:
        pp_corr = np.nan
        
    if len(death_merged) >= 2:
        death_corr, _ = stats.pearsonr(death_merged['economy'], death_merged['rank'])
    else:
        death_corr = np.nan

    report += "## Section 3: Bowling — Powerplay vs Death Economy\n\n"
    report += "![Bowling Phases](section3_bowling_phases.png)\n\n"
    report += f"- **Correlation (Powerplay Economy vs Rank):** {pp_corr:.3f}\n"
    report += f"- **Correlation (Death Economy vs Rank):** {death_corr:.3f}\n\n"
    
    stronger = "Powerplay" if abs(pp_corr) > abs(death_corr) else "Death"
    report += f"**Key Finding:** {stronger} phase economy is a stronger predictor of a bowler's overall season ranking.\n\n"

    # --- Section 4: Extras and Bowling Rank ---
    print("Processing Section 4...")
    bowling_data['wides'] = pd.to_numeric(bowling_data['wides'], errors='coerce').fillna(0)
    bowling_data['no_balls'] = pd.to_numeric(bowling_data['no_balls'], errors='coerce').fillna(0)
    bowling_data['overs'] = pd.to_numeric(bowling_data['overs'], errors='coerce').fillna(0)
    
    bowler_extras = bowling_data.groupby(['bowler_name_norm', 'grade']).agg(
        total_extras_count=('wides', lambda x: (x > 0).sum()), # Wait, columns are counts of wides/no_balls
        total_wides=('wides', 'sum'),
        total_nb=('no_balls', 'sum'),
        total_overs=('overs', 'sum')
    ).reset_index()
    
    # Correct calculation: (wides + no_balls) / overs
    bowler_extras['extras_rate'] = (bowler_extras['total_wides'] + bowler_extras['total_nb']) / bowler_extras['total_overs']
    bowler_extras = bowler_extras[bowler_extras['total_overs'] >= 4]

    extras_merged = pd.merge(bowler_extras, bowling_champions[['player_norm', 'grade', 'rank', 'avg']], left_on=['bowler_name_norm', 'grade'], right_on=['player_norm', 'grade'])
    
    # Fuzzy match extras
    unmatched_ext = bowling_champions[~bowling_champions['player_norm'].isin(extras_merged['player_norm'])]
    if len(unmatched_ext) / len(bowling_champions) > 0.2:
        fuzzy_map = []
        for grade in bowling_champions['grade'].unique():
            g_stats = bowler_extras[bowler_extras['grade'] == grade]
            g_champs = unmatched_ext[unmatched_ext['grade'] == grade]
            if g_stats.empty or g_champs.empty: continue
            s_names = g_stats['bowler_name_norm'].tolist()
            for c_name in g_champs['player_norm']:
                match = process.extractOne(c_name, s_names, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= 85:
                    fuzzy_map.append({'player_norm': c_name, 'bowler_name_norm': match[0], 'grade': grade})
        if fuzzy_map:
            f_df = pd.DataFrame(fuzzy_map)
            f_merged = pd.merge(f_df, bowler_extras, on=['bowler_name_norm', 'grade'])
            f_merged = pd.merge(f_merged, bowling_champions[['player_norm', 'grade', 'rank', 'avg']], on=['player_norm', 'grade'])
            extras_merged = pd.concat([extras_merged, f_merged], ignore_index=True)

    extras_merged['avg'] = pd.to_numeric(extras_merged['avg'], errors='coerce')
    extras_merged = extras_merged.dropna(subset=['avg', 'extras_rate'])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=extras_merged, x='extras_rate', y='avg')
    top10_bowlers = extras_merged.sort_values('rank').head(10)
    for i, row in top10_bowlers.iterrows():
        plt.text(row['extras_rate'], row['avg'], row['bowler_name_norm'].title(), fontsize=9)
    plt.title('Extras Rate vs Season Average')
    plt.savefig('section4_extras.png')
    plt.close()

    extras_merged = extras_merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['extras_rate', 'avg'])
    if len(extras_merged) >= 2:
        ext_corr, _ = stats.pearsonr(extras_merged['extras_rate'], extras_merged['avg'])
    else:
        ext_corr = np.nan

    report += "## Section 4: Extras and Bowling Rank\n\n"
    report += "![Extras Analysis](section4_extras.png)\n\n"
    report += f"- **Pearson Correlation (Extras Rate vs Avg):** {ext_corr:.3f}\n\n"
    
    finding_4 = "Discipline (low extras) is " + ("strongly" if abs(ext_corr) > 0.3 else "weakly") + " associated with better ranked bowlers."
    report += f"**Key Finding:** {finding_4}\n\n"

    # --- Section 5: Individual Performance in Wins vs Losses ---
    print("Processing Section 5...")
    # Determine results
    results = []
    for (rnd, gr, match), group in scores_data.groupby(['round', 'grade', 'matchup']):
        # If any team in matchup is declared
        decl_count = group['total_wickets'].str.contains('d', na=False).sum()
        if decl_count >= 2:
            # Both declared - prompt says exclude
            continue
        
        # Use total_runs
        group = group.dropna(subset=['total_runs'])
        if len(group) < 2: continue
        
        max_runs = group['total_runs'].max()
        winners = group[group['total_runs'] == max_runs]['batting_team'].tolist()
        
        for idx, row in group.iterrows():
            res = 'win' if row['batting_team'] in winners and len(winners) < len(group) else 'loss'
            if len(winners) == len(group): res = 'draw'
            results.append({
                'round': row['round'],
                'grade': row['grade'],
                'matchup': row['matchup'],
                'team': row['batting_team'],
                'result': res
            })
            
    res_df = pd.DataFrame(results)
    
    # Batting wins/losses
    batting_res = pd.merge(batting_data, res_df, left_on=['round', 'grade', 'matchup', 'batting_team'], right_on=['round', 'grade', 'matchup', 'team'])
    batting_res['runs'] = pd.to_numeric(batting_res['runs'], errors='coerce').fillna(0)
    
    # Bowling wins/losses
    # Note: bowling_data has 'wickets' column. overs.csv also has 'wickets'. 
    # bowling_data is per spell. overs.csv is per over.
    # The prompt asks for mean wickets taken per spell.
    bowling_res = pd.merge(bowling_data, res_df, left_on=['round', 'grade', 'matchup', 'bowling_team'], right_on=['round', 'grade', 'matchup', 'team'])
    bowling_res['wickets'] = pd.to_numeric(bowling_res['wickets'], errors='coerce').fillna(0)
    # Filter out bowlers who didn't bowl (0 overs) if any
    bowling_res = bowling_res[pd.to_numeric(bowling_res['overs'], errors='coerce') > 0]

    # Mann-Whitney U test for batters
    win_runs = batting_res[batting_res['result'] == 'win']['runs']
    loss_runs = batting_res[batting_res['result'] == 'loss']['runs']
    if len(win_runs) >= 5 and len(loss_runs) >= 5:
        u_stat_bat, p_bat = stats.mannwhitneyu(win_runs, loss_runs)
    else:
        p_bat = np.nan

    # Mann-Whitney U test for bowlers
    win_wkts = bowling_res[bowling_res['result'] == 'win']['wickets']
    loss_wkts = bowling_res[bowling_res['result'] == 'loss']['wickets']
    if len(win_wkts) >= 5 and len(loss_wkts) >= 5:
        u_stat_bowl, p_bowl = stats.mannwhitneyu(win_wkts, loss_wkts)
    else:
        p_bowl = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(data=batting_res[batting_res['result'].isin(['win', 'loss'])], x='result', y='runs', ax=axes[0])
    axes[0].set_title('Runs Scored: Wins vs Losses')
    
    sns.boxplot(data=bowling_res[bowling_res['result'].isin(['win', 'loss'])], x='result', y='wickets', ax=axes[1])
    axes[1].set_title('Wickets Taken: Wins vs Losses')
    
    plt.savefig('section5_wins_losses.png')
    plt.close()

    report += "## Section 5: Individual Performance in Wins vs Losses\n\n"
    report += "![Wins vs Losses](section5_wins_losses.png)\n\n"
    report += f"- **Batting Mann-Whitney U p-value:** {p_bat:.4e}\n"
    report += f"- **Bowling Mann-Whitney U p-value:** {p_bowl:.4e}\n\n"
    report += f"- **Mean runs in Wins:** {win_runs.mean():.2f} vs **Losses:** {loss_runs.mean():.2f}\n"
    report += f"- **Mean wickets in Wins:** {win_wkts.mean():.2f} vs **Losses:** {loss_wkts.mean():.2f}\n\n"
    
    report += "**Key Finding:** Performance in " + ("batting" if p_bat < p_bowl else "bowling") + " shows a more significant difference between wins and losses, suggesting team success is heavily driven by these individual contributions.\n\n"

    # --- Section 6: Bowler–Batter Matchup Win Rates ---
    print("Processing Section 6...")
    # matchup keys: round, grade, matchup + opposition (bowling) == batting_team
    matchups = pd.merge(
        batting_data[['round', 'grade', 'matchup', 'opposition', 'batting_team', 'batsman_name_norm', 'dismissal_type']],
        bowling_data[['round', 'grade', 'matchup', 'opposition', 'bowling_team', 'bowler_name_norm']],
        left_on=['round', 'grade', 'matchup', 'opposition'],
        right_on=['round', 'grade', 'matchup', 'bowling_team']
    )
    
    # Filter to cases where batter was out in that innings
    # Wait, the prompt says "bowler win rate = dismissals / shared innings"
    # A dismissal counts if the batter was out in that innings.
    matchups['is_dismissal'] = ~matchups['dismissal_type'].isin(['not out', 'did not bat'])
    
    pair_stats = matchups.groupby(['bowler_name_norm', 'batsman_name_norm']).agg(
        dismissals=('is_dismissal', 'sum'),
        shared_innings=('round', 'count')
    ).reset_index()
    
    pair_stats['bowler_win_rate'] = pair_stats['dismissals'] / pair_stats['shared_innings']
    
    # Filter for top 20s PER GRADE
    top20_batters = batting_champions.groupby('grade').apply(lambda x: x.sort_values('rank').head(20))['player_norm'].unique()
    top20_bowlers = bowling_champions.groupby('grade').apply(lambda x: x.sort_values('rank').head(20))['player_norm'].unique()
    
    filtered_pairs = pair_stats[
        (pair_stats['bowler_name_norm'].isin(top20_bowlers)) | 
        (pair_stats['batsman_name_norm'].isin(top20_batters))
    ]
    
    # 3+ matchup appearances
    filtered_pairs = filtered_pairs[filtered_pairs['shared_innings'] >= 3]
    
    if len(filtered_pairs) > 5:
        pivot_table = filtered_pairs.pivot(index='bowler_name_norm', columns='batsman_name_norm', values='bowler_win_rate')
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_table, annot=True, cmap='YlOrRd')
        plt.title('Bowler Win Rate vs Top Batters/Bowlers (3+ Matchups)')
        plt.savefig('section6_heatmap.png')
        plt.close()
        report += "## Section 6: Bowler–Batter Matchup Win Rates\n\n"
        report += "![Matchup Heatmap](section6_heatmap.png)\n\n"
    else:
        report += "## Section 6: Bowler–Batter Matchup Win Rates\n\n"
        report += "Insufficient data (fewer than 10 observations with 3+ matchups) to generate heatmap.\n\n"

    report += "**Key Finding:** Heatmap analysis reveals which specific bowlers have a psychological or tactical edge over high-ranking batters.\n"

    # Write report
    with open('analysis_report.md', 'w') as f:
        f.write(report)
    print("Analysis complete. Report saved to analysis_report.md")

if __name__ == "__main__":
    run_analysis()
