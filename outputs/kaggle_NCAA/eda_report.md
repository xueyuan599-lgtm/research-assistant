# NCAA March Madness — Data Exploration Report
Generated: 2026-07-07 16:26
## 1. Data Overview
Data root: `E:\wuyi\数学建模半自动\research-assistant\outputs\kaggle_NCAA\data`
Total files: 34 (Men: 16, Women: 14, Shared: 4)
### 1.1 Men's Data Files
| File | Rows | Cols | Time Range | Notes |
|------|------|------|-----------|-------|
| Teams | 381 | 4 | — | |
| Seasons | 42 | 6 | 1985-2026 | |
| RegularSeasonCompactResults | 198,577 | 8 | 1985-2026 | |
| RegularSeasonDetailedResults | 124,529 | 34 | 2003-2026 | |
| NCAATourneyCompactResults | 2,585 | 8 | 1985-2025 | |
| NCAATourneyDetailedResults | 1,449 | 34 | 2003-2025 | |
| NCAATourneySeeds | 2,694 | 3 | 1985-2026 | |
| TeamConferences | 13,753 | 3 | 1985-2026 | |
| TeamCoaches | 13,900 | 5 | 1985-2026 | |
| NCAATourneySlots | 2,653 | 4 | 1985-2026 | |
| NCAATourneySeedRoundSlots | 776 | 5 | — | |
| GameCities | 92,438 | 6 | 2010-2026 | |
| ConferenceTourneyGames | 7,093 | 5 | 2001-2026 | |
| SecondaryTourneyCompactResults | 1,865 | 9 | 1985-2025 | |
| SecondaryTourneyTeams | 1,895 | 3 | 1985-2025 | |
| TeamSpellings | 1,178 | 2 | — | |

### 1.2 Women's Data Files
| File | Rows | Cols | Time Range | Notes |
|------|------|------|-----------|-------|
| Teams | 379 | 2 | — | |
| Seasons | 29 | 6 | 1998-2026 | |
| RegularSeasonCompactResults | 142,507 | 8 | 1998-2026 | |
| RegularSeasonDetailedResults | 87,187 | 34 | 2010-2026 | |
| NCAATourneyCompactResults | 1,717 | 8 | 1998-2025 | |
| NCAATourneyDetailedResults | 961 | 34 | 2010-2025 | |
| NCAATourneySeeds | 1,812 | 3 | 1998-2026 | |
| TeamConferences | 9,853 | 3 | 1998-2026 | |
| NCAATourneySlots | 1,847 | 4 | 1998-2026 | |
| GameCities | 89,035 | 6 | 2010-2026 | |
| ConferenceTourneyGames | 6,777 | 5 | 2002-2026 | |
| SecondaryTourneyCompactResults | 906 | 9 | 2013-2025 | |
| SecondaryTourneyTeams | 904 | 3 | 2013-2025 | |
| TeamSpellings | 1,171 | 2 | — | |

### 1.3 Shared Files
| File | Rows | Cols | Notes |
|------|------|------|-------|
| Cities | 510 | 3 | |
| Conferences | 51 | 2 | |
| SampleSubmissionStage1 | 519,144 | 2 | |
| SampleSubmissionStage2 | 132,133 | 2 | |

## 2. Target Understanding
- **Task**: Predict P(Team_A beats Team_B) for each matchup in the tournament
- **Evaluation Metric**: Brier Score (mean squared error between prediction (0-1) and actual outcome)
- Better Brier = lower score. A naive prediction of 0.5 gives Brier = 0.25
- **Submission format**: Binary classification probabilities (0 to 1) for each matchup ID
- Stage 2 contains **132,133 matchups** across 1 seasons: 2026
- Stage 1 contains **519,144 matchups** across 4 seasons: 2022, 2023, 2024, 2025

## 3. Teams Overview
- **Men**: 381 teams
- **Women**: 379 teams
- **Team ID overlap** (same ID used for men & women): 0
  - (NCAA typically uses disjoint TeamID ranges for men vs women)

## 4. Tournament Seed Distribution
Seeds follow NCAA format: `{Region}{Number:02d}` — e.g. W01 = 1 seed in West region.
Extracted numeric seed (1–16) distributions:

**Men**: 2694 entries across 41 seasons
| Seed | Count |
|------|-------|
| 1 | 164 |
| 2 | 164 |
| 3 | 164 |
| 4 | 164 |
| 5 | 164 |
| 6 | 164 |
| 7 | 164 |
| 8 | 164 |
| 9 | 164 |
| 10 | 166 |
| 11 | 186 |
| 12 | 168 |
| 13 | 165 |
| 14 | 165 |
| 15 | 164 |
| 16 | 204 |

**Women**: 1812 entries across 28 seasons
| Seed | Count |
|------|-------|
| 1 | 112 |
| 2 | 112 |
| 3 | 112 |
| 4 | 112 |
| 5 | 112 |
| 6 | 112 |
| 7 | 112 |
| 8 | 112 |
| 9 | 112 |
| 10 | 113 |
| 11 | 120 |
| 12 | 113 |
| 13 | 112 |
| 14 | 112 |
| 15 | 112 |
| 16 | 122 |

Key observation: Each of the 16 seeds appears approximately equally, but note that play-in games (First Four) create slightly more teams at seeds 11, 12, 16.

## 5. Missing Seed Analysis
Check if any teams in tournament results have no seed entry.
- **Men**: 0 teams in tournament without seed records
- **Women**: 0 teams in tournament without seed records

## 6. Regular Season Statistics
| Metric | Men | Women |
|--------|-----|-------|
| N Games | 198577 | 142507 |
| Avg Score | 70.94 | 64.75 |
| Med Score | 70.0 | 64.0 |
| Std Score | 13.01 | 13.24 |
| Avg Margin | 12.09 | 14.44 |
| Med Margin | 10.0 | 12.0 |
| N Seasons | 42 | 29 |
| N Teams | 381 | 370 |
| Ot Games | 8163 | 4898 |

## 7. Tournament Statistics
| Metric | Men | Women |
|--------|-----|-------|
| N Tourney Games | 2585 | 1717 |
| Avg Margin | 11.82 | 16.73 |
| Med Margin | 10.0 | 14.0 |
| N Seasons | 40 | 27 |
| Ot Games | 151 | 40 |

## 8. 2020 COVID-19 Season Analysis
The 2020 NCAA tournament was cancelled due to COVID-19.
- **Men**:
  - Tournament games in 2020: 0
  - Seeds assigned for 2020: 0
  - Season 2020 in Season table: True
  - Season 2020 DayZero: 11/04/2019
- **Women**:
  - Tournament games in 2020: 0
  - Seeds assigned for 2020: 0
  - Season 2020 in Season table: True
  - Season 2020 DayZero: 11/04/2019

**Implication**: 2020 has regular season data but no tournament outcome. For training, 2020 regular season data can still be used for feature engineering, but no tournament matchup from 2020 is available for supervised learning.

## 9. Massey Ordinals Analysis (Men Only)
- **File size**: 129 MB on disk
- **Total rating entries**: 5,865,001
- **Number of ranking systems**: 197
- **Season range**: [2003, 2026]
- **Systems with full coverage** (all seasons): 7
  - ['AP', 'COL', 'DOL', 'MOR', 'POM', 'USA', 'WLK']

**Ranking days per season**: range 16–69 days

**Ordinal rank distribution**:
- count: 5865001.0
- mean: 174.32
- std: 101.81
- min: 1.0
- 10%: 33.0
- 25%: 86.0
- 50%: 174.0
- 75%: 262.0
- 90%: 315.0
- 95%: 333.0
- 99%: 351.0
- max: 365.0

**Ranking system correlations (latest season):**
- WEI_vs_PGH: 0.997
- PGH_vs_WEI: 0.997
- WEI_vs_WIL: 0.997
- WIL_vs_WEI: 0.997
- MB_vs_WEI: 0.996
- WEI_vs_MB: 0.996
- WIL_vs_PGH: 0.996
- PGH_vs_WIL: 0.996
- PGH_vs_MB: 0.995
- MB_vs_PGH: 0.995

**Systems per season**:
- Season 2003: 36 systems
- Season 2004: 38 systems
- Season 2005: 40 systems
- Season 2006: 40 systems
- Season 2007: 42 systems

**Women's note**: No MasseyOrdinals file available for women's tournament. All feature engineering for women must rely on game results and seeds only.

## 10. Adversarial Validation (Train/Test Distribution Shift)
Method: Train a classifier to distinguish earlier-season team features from later-season features. AUC > 0.65 suggests distribution shift across time.

### M
- **AUC**: 0.8302
- **Shift Detected**: True
- **Severity**: high
- **Top discriminating features**:
  - n_games: importance=200.00
  - pts_against: importance=172.00
  - net_pts: importance=164.00
  - pts_for: importance=158.00
  - SeedNum: importance=132.00
### W
- **AUC**: 0.6385
- **Shift Detected**: False
- **Severity**: low
- **Top discriminating features**:
  - pts_against: importance=178.00
  - net_pts: importance=171.00
  - pts_for: importance=168.00
  - n_games: importance=137.00
  - win_rate: importance=132.00

**Interpretation**:
- AUC ~ 0.5-0.6: Features are stable over time, no significant shift
- AUC 0.65-0.75: Moderate shift, consider time-aware cross-validation
- AUC > 0.75: Strong shift, use only recent data or model the trend

## 11. Missing Value Summary
| File | Missing Columns | Missing % |
|------|----------------|-----------|
| (All files) | None | 0% |

## 12. Feature Engineering Recommendations
### 12.1 Core Features (All Models)
1. **Season-level team strength**
   - Win rate, avg points scored/allowed, net rating
   - Strength of schedule (opponents' win rate)
2. **Seed-based features**
   - Seed difference: strong predictor of match outcome
   - Historical win rate by seed matchup
3. **Recent form**
   - Last 10 games win rate
   - Performance in conference tournament
4. **Massey Ordinals (men only)**
   - Composite rank across systems at latest ranking day
   - Rank momentum (trend over last few ranking days)
5. **Historical head-to-head**
   - Previous matchups in regular season
   - Win rate in tournament settings

### 12.2 Advanced Features
1. **Team efficiency metrics** (from DetailedResults)
   - Offensive/defensive efficiency (points per possession)
   - FG%, 3P%, FT%, rebounding, turnovers
   - Tempo-adjusted stats
2. **Coach experience**
   - Tournament win rate by coach
3. **Location features**
   - Game location (home/away/neutral) for regular season
   - Distance traveled for tournament games (from GameCities)
4. **Time-series features**
   - Team performance trajectory over the season (rolling windows)
   - Massey rank trajectory

### 12.3 Modeling Strategy Notes
1. **Men vs Women**: Train separate models unless pooling proves beneficial
2. **Massey Ordinals**: For men, include as features. For women, skip.
3. **Cross-validation**: Use leave-one-season-out or expanding window CV
4. **Loss function**: Brier score minimization (proper scoring rule)
5. **2020 season**: Usable for feature construction but not for tournament outcome labels

## 13. Men vs Women Processing Differences
| Aspect | Men | Women |
|--------|-----|-------|
| Teams | 381 | 379 |
| Regular season games | 198,577 | 142,507 |
| Regular season seasons | 42 (1985-2026) | 29 (1998-2026) |
| Tournament games | 2,585 | 1,717 |
| Tournament seasons | 40 | 27 |
| Massey Ordinals | **Available** (197 systems, 129MB, 5.9M ratings) | **Not available** |
| Detailed stats start | 2003 | 2010 |
| Coach data | Available (13,900 records) | Not directly available |
| Avg regular season score | 70.94 | 64.75 |
| Avg tournament margin | 11.82 | 16.73 |

## 14. Submission Summary
- Stage 1: 519,144 matchups across 4 seasons: 2022, 2023, 2024, 2025
- Stage 2: 132,133 matchups across 1 seasons: 2026
- Prediction column: `Pred` (float 0-1, probability TeamA beats TeamB)

---
*Report generated by Data Explorer Agent*
