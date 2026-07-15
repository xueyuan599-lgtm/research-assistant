#!/usr/bin/env python3
"""
International Football Data Collector (2021-2026)
==================================================
Collects and processes:
1. International A-level match results (2021-01-12 to 2026-07-09)
2. FIFA World Ranking history (38 dates from 2021-02 to 2026-06)
3. 2026 FIFA World Cup match data (group stage through semi-finals)

Data sources:
- Kaggle: martj42/international-football-results-from-1872-to-2017
- FIFA API: api.fifa.com/api/v3/rankings
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# Set UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"E:\wuyi\数学建模半自动\research-assistant\outputs\football_data"
KAGGLE_PATH = r"C:\Users\lenovo\.cache\kagglehub\datasets\martj42\international-football-results-from-1872-to-2017\versions\133"
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("INTERNATIONAL FOOTBALL DATA COLLECTOR")
print(f"Output directory: {DATA_DIR}")
print("=" * 60)

# ============================================================
# 1. MATCH RESULTS
# ============================================================
print("\n" + "=" * 60)
print("1. MATCH RESULTS (2021-2026)")
print("=" * 60)

results_df = pd.read_csv(os.path.join(KAGGLE_PATH, "results.csv"))
results_df['date'] = pd.to_datetime(results_df['date'])

# Filter 2021-2026
mask = (results_df['date'] >= '2021-01-01') & (results_df['date'] <= '2026-07-09')
matches = results_df[mask].copy()
matches = matches.dropna(subset=['home_score', 'away_score'])
matches['home_score'] = matches['home_score'].astype(int)
matches['away_score'] = matches['away_score'].astype(int)
matches = matches.sort_values('date').reset_index(drop=True)

# Save
matches.to_csv(os.path.join(DATA_DIR, "international_matches_2021_2026.csv"),
               index=False, encoding='utf-8')
print(f"  Saved: international_matches_2021_2026.csv")
print(f"  Rows: {len(matches)}")
print(f"  Columns: {list(matches.columns)}")
print(f"  Date range: {matches['date'].min().date()} to {matches['date'].max().date()}")
print(f"  Teams: {pd.concat([matches['home_team'], matches['away_team']]).nunique()}")
print(f"  Countries (venues): {matches['country'].nunique()}")

# ============================================================
# 2. FIFA RANKINGS
# ============================================================
print("\n" + "=" * 60)
print("2. FIFA RANKINGS (2021-2026)")
print("=" * 60)

rankings_df = pd.read_csv(os.path.join(DATA_DIR, "fifa_rankings_2021_2026.csv"))
print(f"  Loaded: fifa_rankings_2021_2026.csv")
print(f"  Rows: {len(rankings_df)}")
print(f"  Columns: {list(rankings_df.columns)}")
print(f"  Ranking dates: {rankings_df['rank_date'].nunique()}")
print(f"  Unique teams: {rankings_df['team_name'].nunique()}")

# ============================================================
# 3. 2026 WORLD CUP DATA
# ============================================================
print("\n" + "=" * 60)
print("3. 2026 WORLD CUP DATA")
print("=" * 60)

# Get all WC matches
wc_all = results_df[
    (results_df['date'] >= '2026-06-01') &
    (results_df['date'] <= '2026-07-15') &
    (results_df['tournament'] == 'FIFA World Cup')
].copy()
wc_all = wc_all.sort_values('date').reset_index(drop=True)

# Assign match stages based on date
def get_stage(date_str):
    d = pd.Timestamp(date_str)
    if d <= pd.Timestamp('2026-06-27'):
        return 'Group Stage'
    elif d <= pd.Timestamp('2026-07-03'):
        return 'Round of 32'
    elif d <= pd.Timestamp('2026-07-07'):
        return 'Round of 16'
    elif d <= pd.Timestamp('2026-07-09'):
        return 'Quarter-finals'
    elif d <= pd.Timestamp('2026-07-11'):
        return 'Semi-finals'
    elif d <= pd.Timestamp('2026-07-14'):
        return 'Final / 3rd Place'
    else:
        return 'Unknown'

# Group definitions based on opponent analysis
groups = {
    'A': ['Mexico', 'South Korea', 'Czech Republic', 'South Africa'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Switzerland', 'Qatar'],
    'C': ['United States', 'Paraguay', 'Turkey', 'Australia'],
    'D': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'E': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'F': ['Germany', 'Curaçao', 'Ecuador', 'Ivory Coast'],
    'G': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'H': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['England', 'Croatia', 'Ghana', 'Panama'],
    'L': ['Portugal', 'Colombia', 'DR Congo', 'Uzbekistan'],
}

# Build team -> group mapping
team_to_group = {}
for g, teams in groups.items():
    for t in teams:
        team_to_group[t] = g

# Build group standings from match results
gs = wc_all[wc_all['date'] <= '2026-06-27'].copy()
gs = gs.dropna(subset=['home_score', 'away_score'])
gs['home_score'] = gs['home_score'].astype(int)
gs['away_score'] = gs['away_score'].astype(int)

# Calculate standings for each group
standings_rows = []
for group_name, group_teams in groups.items():
    group_matches = gs[
        (gs['home_team'].isin(group_teams)) &
        (gs['away_team'].isin(group_teams))
    ]

    team_stats = {}
    for t in group_teams:
        team_stats[t] = {'MP': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0}

    for _, m in group_matches.iterrows():
        ht, at = m['home_team'], m['away_team']
        hs, as_ = m['home_score'], m['away_score']

        team_stats[ht]['MP'] += 1
        team_stats[at]['MP'] += 1
        team_stats[ht]['GF'] += hs
        team_stats[ht]['GA'] += as_
        team_stats[at]['GF'] += as_
        team_stats[at]['GA'] += hs

        if hs > as_:
            team_stats[ht]['W'] += 1
            team_stats[ht]['Pts'] += 3
            team_stats[at]['L'] += 1
        elif hs < as_:
            team_stats[at]['W'] += 1
            team_stats[at]['Pts'] += 3
            team_stats[ht]['L'] += 1
        else:
            team_stats[ht]['D'] += 1
            team_stats[at]['D'] += 1
            team_stats[ht]['Pts'] += 1
            team_stats[at]['Pts'] += 1

    # Sort by points, then GD, then GF
    sorted_teams = sorted(group_teams,
                          key=lambda t: (team_stats[t]['Pts'],
                                        team_stats[t]['GF'] - team_stats[t]['GA'],
                                        team_stats[t]['GF']),
                          reverse=True)

    for pos, t in enumerate(sorted_teams, 1):
        s = team_stats[t]
        s['GD'] = s['GF'] - s['GA']
        standings_rows.append({
            'group': group_name,
            'position': pos,
            'team': t,
            'MP': s['MP'], 'W': s['W'], 'D': s['D'], 'L': s['L'],
            'GF': s['GF'], 'GA': s['GA'], 'GD': s['GD'], 'Pts': s['Pts']
        })

standings_df = pd.DataFrame(standings_rows)
standings_df.to_csv(os.path.join(DATA_DIR, "wc2026_group_standings.csv"),
                    index=False, encoding='utf-8')
print(f"  Saved: wc2026_group_standings.csv ({len(standings_df)} rows)")

# Build full WC match data with groups/stages/status
wc_data_rows = []
for _, row in wc_all.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else ''
    stage = get_stage(row['date'])

    # Determine status
    if pd.isna(row['home_score']) or pd.isna(row['away_score']):
        status = 'Scheduled'
        home_score = None
        away_score = None
    else:
        status = 'Completed'
        home_score = int(row['home_score'])
        away_score = int(row['away_score'])

    ht = row['home_team']
    at = row['away_team']

    wc_data_rows.append({
        'date': date_str,
        'stage': stage,
        'group': team_to_group.get(ht, team_to_group.get(at, '')),
        'home_team': ht,
        'away_team': at,
        'home_score': home_score,
        'away_score': away_score,
        'home_group': team_to_group.get(ht, ''),
        'away_group': team_to_group.get(at, ''),
        'status': status,
        'venue_country': row['country'],
        'venue_city': row['city'] if pd.notna(row.get('city')) else '',
    })

wc_df = pd.DataFrame(wc_data_rows)
wc_df.to_csv(os.path.join(DATA_DIR, "wc2026_matches.csv"), index=False, encoding='utf-8')
print(f"  Saved: wc2026_matches.csv ({len(wc_df)} rows)")

completed = wc_df[wc_df['status'] == 'Completed']
scheduled = wc_df[wc_df['status'] == 'Scheduled']
print(f"  Completed matches: {len(completed)}")
print(f"  Scheduled matches: {len(scheduled)}")
print(f"  Date range: {wc_df['date'].min()} to {wc_df['date'].max()}")
print(f"  Stages: {wc_df['stage'].unique().tolist()}")
print(f"  Teams: {pd.concat([wc_df['home_team'], wc_df['away_team']]).nunique()}")

# ============================================================
# 4. DATA DICTIONARY
# ============================================================
print("\n" + "=" * 60)
print("4. DATA DICTIONARY")
print("=" * 60)

data_dict = {
    "international_matches_2021_2026.csv": {
        "description": "International A-level football match results from 2021-01-12 to 2026-07-09",
        "rows": len(matches),
        "columns": {
            "date": "Match date (YYYY-MM-DD)",
            "home_team": "Home team name",
            "away_team": "Away team name",
            "home_score": "Goals scored by home team",
            "away_score": "Goals scored by away team",
            "tournament": "Tournament/competition name (e.g. FIFA World Cup, Friendly, UEFA Euro qualification)",
            "city": "City where match was played",
            "country": "Country where match was played",
            "neutral": "Whether the match was at a neutral venue (True/False)",
        }
    },
    "fifa_rankings_2021_2026.csv": {
        "description": "FIFA World Ranking history with 38 publication dates from 2021-02-18 to 2026-06-11",
        "rows": len(rankings_df),
        "columns": {
            "rank_date": "Ranking publication date (YYYY-MM-DD)",
            "rank": "Current ranking position (1 = best)",
            "prev_rank": "Previous ranking position",
            "team_name": "National team name",
            "country_code": "FIFA 3-letter country code",
            "confederation": "Confederation (AFC, CAF, CONCACAF, CONMEBOL, OFC, UEFA)",
            "total_points": "Current total ranking points",
            "prev_points": "Previous total ranking points",
            "matches_played": "Number of matches played in ranking period",
        }
    },
    "wc2026_matches.csv": {
        "description": "2026 FIFA World Cup match data - 100 matches including group stage (72), Round of 32 (16), Round of 16 (8), Quarter-finals (4 scheduled)",
        "rows": len(wc_df),
        "columns": {
            "date": "Match date (YYYY-MM-DD)",
            "stage": "Tournament stage (Group Stage, Round of 32, Round of 16, Quarter-finals, Semi-finals, Final)",
            "group": "Group letter for group stage matches",
            "home_team": "Home/named-first team",
            "away_team": "Away/named-second team",
            "home_score": "Goals by home team (NaN if not yet played)",
            "away_score": "Goals by away team (NaN if not yet played)",
            "home_group": "Group letter of home team",
            "away_group": "Group letter of away team",
            "status": "Match status (Completed, Scheduled)",
            "venue_country": "Host country of the match",
            "venue_city": "Host city of the match",
        }
    },
    "wc2026_group_standings.csv": {
        "description": "2026 FIFA World Cup group stage final standings for all 12 groups",
        "rows": len(standings_df),
        "columns": {
            "group": "Group letter (A through L)",
            "position": "Final position in group (1-4)",
            "team": "National team name",
            "MP": "Matches played",
            "W": "Wins",
            "D": "Draws",
            "L": "Losses",
            "GF": "Goals for",
            "GA": "Goals against",
            "GD": "Goal difference",
            "Pts": "Points",
        }
    },
}

dict_rows = []
for fname, info in data_dict.items():
    dict_rows.append({
        'filename': fname,
        'description': info['description'],
        'rows': info['rows'],
        'columns_detail': json.dumps(info['columns'], ensure_ascii=False),
    })

dict_df = pd.DataFrame(dict_rows)
dict_df.to_csv(os.path.join(DATA_DIR, "data_dictionary.csv"), index=False, encoding='utf-8')
print(f"  Saved: data_dictionary.csv")

# ============================================================
# 5. SUMMARY REPORT
# ============================================================
print("\n" + "=" * 60)
print("5. DATA COLLECTION SUMMARY")
print("=" * 60)

print(f"""
MATCH RESULTS (2021-2026):
  File: international_matches_2021_2026.csv
  Total matches: {len(matches):,}
  Date range: {matches['date'].min().date()} to {matches['date'].max().date()}
  Teams: {pd.concat([matches['home_team'], matches['away_team']]).nunique():,}
  Tournament types: {matches['tournament'].nunique():,}
  Avg goals/match: {(matches['home_score'].sum() + matches['away_score'].sum()) / len(matches):.2f}
  Home win rate: {(matches['home_score'] > matches['away_score']).mean()*100:.1f}%
  Draw rate: {(matches['home_score'] == matches['away_score']).mean()*100:.1f}%
  Away win rate: {(matches['home_score'] < matches['away_score']).mean()*100:.1f}%

FIFA RANKINGS (2021-2026):
  File: fifa_rankings_2021_2026.csv
  Total records: {len(rankings_df):,}
  Ranking dates: {rankings_df['rank_date'].nunique():,}
  Ranked teams: {rankings_df['team_name'].nunique():,}
  Period: {rankings_df['rank_date'].min()} to {rankings_df['rank_date'].max()}
  Current #1: {rankings_df[rankings_df['rank_date'] == rankings_df['rank_date'].max()].sort_values('rank').iloc[0]['team_name']}

2026 WORLD CUP:
  File: wc2026_matches.csv
  Total matches in dataset: {len(wc_df)}
  Completed: {len(completed)}
  Scheduled (to be played): {len(scheduled)}
  Teams: {pd.concat([wc_df['home_team'], wc_df['away_team']]).nunique():,}
  Groups: 12 (A through L)

  Current knockout stage (top teams remaining):
    Quarter-finals: France vs Morocco (Jul 9), Spain vs Belgium (Jul 10),
                    Norway vs England (Jul 11), Argentina vs Switzerland (Jul 11)
    Semi-finals: TBD
    Final: TBD

TOP TEAMS BY FIFA RANKING (2026-06-11):
  {chr(10)+'  '.join([f'{r["rank"]}. {r["team_name"]} ({r["country_code"]}) - {r["total_points"]} pts'
                       for _, r in rankings_df[rankings_df["rank_date"]=="2026-06-11"].sort_values("rank").head(10).iterrows()])}

TOP GROUP STAGE RESULTS:
""")

# Print top teams from each group
for g in sorted(groups.keys()):
    grp = standings_df[standings_df['group'] == g].sort_values('position')
    top = grp.iloc[0]
    print(f"  Group {g}: {top['team']} ({top['Pts']} pts, +{top['GD']})")

print(f"""
OUTPUT FILES:
  {DATA_DIR}/international_matches_2021_2026.csv - Match results
  {DATA_DIR}/fifa_rankings_2021_2026.csv - FIFA rankings
  {DATA_DIR}/wc2026_matches.csv - World Cup 2026 matches
  {DATA_DIR}/wc2026_group_standings.csv - World Cup group standings
  {DATA_DIR}/data_dictionary.csv - Data dictionary
  {DATA_DIR}/collect_data.py - This script
""")
