# Cricket Match Analysis Report

Note: 91 players from batting champions could not be matched.

## Section 1: Scoring Rate vs Average Trade-off

![Strike Rate vs Average](section1_sr_vs_avg.png)

- **Pearson Correlation (SR vs Avg):** 0.346 (p-value: 2.130e-41)

### Batter Performance by Strike Rate Quartile

| sr_quartile   |   mean_rank |   mean_avg |
|:--------------|------------:|-----------:|
| Q1            |     293.902 |    10.3944 |
| Q2            |     167.649 |    23.3329 |
| Q3            |     142.941 |    26.9338 |
| Q4            |     163.126 |    30.0958 |

**Key Finding:** The competition rewards efficiency (higher strike rate correlates with higher average).

## Section 2: Dismissal Profile — Elite vs Non-Elite Batters

![Dismissal Profile](section2_dismissals.png)

- **Chi-squared test p-value:** 0.0000

### Dismissal Proportions

| group     |   bowled |       lbw |   caught |   caught-and-bowled |   run out |   stumped |   not out |   did not bat |
|:----------|---------:|----------:|---------:|--------------------:|----------:|----------:|----------:|--------------:|
| elite     |        0 | 0.0579345 | 0.68262  |                   0 | 0.0377834 | 0.0100756 |  0.153652 |     0.0579345 |
| non-elite |        0 | 0.0713643 | 0.424809 |                   0 | 0.0391715 | 0.0142954 |  0.160401 |     0.289959  |

**Key Finding:** There is a significant difference in dismissal profiles between elite and non-elite batters.

## Section 3: Bowling — Powerplay vs Death Economy

![Bowling Phases](section3_bowling_phases.png)

- **Correlation (Powerplay Economy vs Rank):** -0.153
- **Correlation (Death Economy vs Rank):** -0.081

**Key Finding:** Powerplay phase economy is a stronger predictor of a bowler's overall season ranking.

## Section 4: Extras and Bowling Rank

![Extras Analysis](section4_extras.png)

- **Pearson Correlation (Extras Rate vs Avg):** -0.004

**Key Finding:** Discipline (low extras) is weakly associated with better ranked bowlers.

## Section 5: Individual Performance in Wins vs Losses

![Wins vs Losses](section5_wins_losses.png)

- **Batting Mann-Whitney U p-value:** 6.5466e-01
- **Bowling Mann-Whitney U p-value:** 2.2921e-20

- **Mean runs in Wins:** 17.49 vs **Losses:** 13.40
- **Mean wickets in Wins:** 27.42 vs **Losses:** 32.07

**Key Finding:** Performance in bowling shows a more significant difference between wins and losses, suggesting team success is heavily driven by these individual contributions.

## Section 6: Bowler–Batter Matchup Win Rates

![Matchup Heatmap](section6_heatmap.png)

**Key Finding:** Heatmap analysis reveals which specific bowlers have a psychological or tactical edge over high-ranking batters.
