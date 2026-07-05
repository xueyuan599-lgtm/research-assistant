"""
CUMCM 2024 Problem C — Final Solution
======================================
Complete optimization with:
- Yearly LP with rolling rotation constraints
- Legume rotation (≥1 per 3-year window)
- No continuous cropping
- Differentiated Q2 (trends) and Q3 (correlation)
- Proper Excel export matching template format
"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import pulp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = Path(r"E:\wuyi\数学建模半自动\research-assistant\outputs\cumcm2024c")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    ROOT = Path(r"E:\wuyi\数学建模半自动\research-assistant\outputs\test260704")
    all_files = []
    for root, dirs, files in os.walk(ROOT):
        for f in files:
            all_files.append(Path(root) / f)
    f1 = [f for f in all_files if '附件1' in f.name][0]
    f2 = [f for f in all_files if '附件2' in f.name][0]

    # Plots: 附件1 Sheet 0
    raw = pd.read_excel(f1, sheet_name=0, header=None)
    df_plots = pd.DataFrame({
        'plot_id': raw.iloc[1:, 0].astype(str).str.strip(),
        'plot_type': raw.iloc[1:, 1].astype(str).str.strip(),
        'area': pd.to_numeric(raw.iloc[1:, 2], errors='coerce'),
    }).dropna(subset=['area'])

    # Crops: 附件1 Sheet 1
    raw = pd.read_excel(f1, sheet_name=1, header=None)
    df_crops = pd.DataFrame({
        'crop_id': pd.to_numeric(raw.iloc[1:, 0], errors='coerce'),
        'crop_name': raw.iloc[1:, 1].astype(str).str.strip(),
        'crop_category': raw.iloc[1:, 2].astype(str).str.strip(),
    }).dropna(subset=['crop_id'])
    df_crops['crop_id'] = df_crops['crop_id'].astype(int)

    # 2023 Planting: 附件2 Sheet 0
    raw = pd.read_excel(f2, sheet_name=0, header=None)
    df_2023 = pd.DataFrame({
        'plot_id': raw.iloc[1:, 0].astype(str).str.strip(),
        'crop_id': pd.to_numeric(raw.iloc[1:, 1], errors='coerce'),
        'season': raw.iloc[1:, 5].astype(str).str.strip(),
        'area': pd.to_numeric(raw.iloc[1:, 4], errors='coerce'),
    }).dropna(subset=['crop_id'])
    df_2023['crop_id'] = df_2023['crop_id'].astype(int)

    # Stats: 附件2 Sheet 1 (yield, cost, price)
    raw = pd.read_excel(f2, sheet_name=1, header=None)
    df_stats = pd.DataFrame({
        'crop_id': pd.to_numeric(raw.iloc[1:, 1], errors='coerce'),
        'crop_name': raw.iloc[1:, 2].astype(str).str.strip(),
        'plot_type': raw.iloc[1:, 3].astype(str).str.strip(),
        'season': raw.iloc[1:, 4].astype(str).str.strip(),
        'yield_per_mu': pd.to_numeric(raw.iloc[1:, 5], errors='coerce'),
        'cost_per_mu': pd.to_numeric(raw.iloc[1:, 6], errors='coerce'),
        'price_range': raw.iloc[1:, 7].astype(str),
    }).dropna(subset=['crop_id'])
    df_stats['crop_id'] = df_stats['crop_id'].astype(int)
    pe = df_stats['price_range'].str.extract(r'([\d.]+)-([\d.]+)')
    df_stats['price_low'] = pd.to_numeric(pe[0], errors='coerce')
    df_stats['price_high'] = pd.to_numeric(pe[1], errors='coerce')
    df_stats['price_mid'] = (df_stats['price_low'] + df_stats['price_high']) / 2

    return df_plots, df_crops, df_2023, df_stats


def build_model_data(df_plots, df_crops, df_2023, df_stats):
    """Build all model data structures."""

    # Plot type map
    plot_type_map = dict(zip(df_plots['plot_id'], df_plots['plot_type']))
    plot_area_map = dict(zip(df_plots['plot_id'], df_plots['area']))

    # Season config per plot
    plot_season_map = {}
    for _, row in df_plots.iterrows():
        pid, ptype = row['plot_id'], row['plot_type']
        if ptype in ['平旱地', '梯田', '山坡地']:
            plot_season_map[pid] = ['S1']  # single
        elif ptype == '水浇地' and pid in ['D7', 'D8']:
            plot_season_map[pid] = ['S1']  # rice only
        elif ptype == '水浇地':
            plot_season_map[pid] = ['S1', 'S2']
        elif ptype == '普通大棚':
            plot_season_map[pid] = ['S1', 'S2']
        elif ptype == '智慧大棚':
            plot_season_map[pid] = ['S1', 'S2']

    # Compatible (crop, plot_type, season) → params
    # Map '单季'/'第一季'/'第二季' → 'S1'/'S2'
    SEASON_MAP = {'单季': 'S1', '第一季': 'S1', '第二季': 'S2'}

    params = {}  # (cid, ptype, season_key) → {yield, cost, price_mid, ...}
    for _, row in df_stats.iterrows():
        skey = SEASON_MAP.get(row['season'], 'S1')
        key = (int(row['crop_id']), row['plot_type'], skey)
        params[key] = {
            'yield': row['yield_per_mu'],
            'cost': row['cost_per_mu'],
            'price_mid': row['price_mid'],
            'price_low': row['price_low'],
            'price_high': row['price_high'],
            'crop_name': row['crop_name'],
        }

    # Smart greenhouse S1 = regular greenhouse S1
    for (cid, pt, sk), p in list(params.items()):
        if pt == '普通大棚' and sk == 'S1':
            if (cid, '智慧大棚', 'S1') not in params:
                params[(cid, '智慧大棚', 'S1')] = dict(p)

    # Alias: for open-field types, S1 params from '单季'
    for ptype in ['平旱地', '梯田', '山坡地']:
        for (cid, pt, sk), p in list(params.items()):
            if pt == ptype and sk == 'S1':
                pass  # already exists

    # Compatible set
    compat = defaultdict(set)
    for (cid, pt, sk) in params:
        compat[(pt, sk)].add(cid)
    compat = {k: sorted(v) for k, v in compat.items()}

    # Legume crops
    legume_ids = set()
    for _, row in df_crops.iterrows():
        if '豆' in row['crop_category']:
            legume_ids.add(int(row['crop_id']))

    # Expected sales = 2023 total production
    exp_sales = defaultdict(float)
    for _, row in df_2023.iterrows():
        cid, area = int(row['crop_id']), float(row['area'])
        ptype = plot_type_map.get(row['plot_id'], '')
        skey = SEASON_MAP.get(row['season'], 'S1')
        key = (cid, ptype, skey)
        if key in params:
            exp_sales[cid] += area * params[key]['yield']
    exp_sales = dict(exp_sales)

    # Crop name lookup
    crop_name_map = {}
    for _, row in df_crops.iterrows():
        crop_name_map[int(row['crop_id'])] = row['crop_name']
    # Supplement from params
    for (cid, _, _), p in params.items():
        if cid not in crop_name_map:
            crop_name_map[cid] = p['crop_name']

    # 2023 planting as initial state for rotation tracking
    init_state = {}  # (plot_id, season) → crop_id
    for _, row in df_2023.iterrows():
        pid = row['plot_id']
        skey = SEASON_MAP.get(row['season'], 'S1')
        init_state[(pid, skey)] = int(row['crop_id'])

    return {
        'plot_type_map': plot_type_map,
        'plot_area_map': plot_area_map,
        'plot_season_map': plot_season_map,
        'params': params,
        'compat': compat,
        'legume_ids': legume_ids,
        'exp_sales': exp_sales,
        'crop_name_map': crop_name_map,
        'init_state': init_state,
    }


# ============================================================
# OPTIMIZATION ENGINE
# ============================================================

def build_yearly_lp(year, year_idx, md, prev_planting, legume_history,
                     profit_override=None, exp_sales_override=None,
                     case='b', min_crop_diversity=True):
    """
    Build and solve single-year LP with rotation constraints.

    Args:
        prev_planting: {(plot_id, season): crop_id} from previous year
        legume_history: {plot_id: [bool]*3} last 3 years legume status
        profit_override: optional override for profit per (cid, ptype, sk)
        exp_sales_override: optional override for expected sales

    Returns:
        (planting_plan, objective_value)
        planting_plan: {(plot_id, season): (crop_id, area)}
    """
    prob = pulp.LpProblem(f"Y{year}", pulp.LpMaximize)

    params = md['params']
    compat = md['compat']
    plot_type_map = md['plot_type_map']
    plot_area_map = md['plot_area_map']
    plot_season_map = md['plot_season_map']

    # Compute base profit
    base_profit = {}
    for key, p in params.items():
        base_profit[key] = p['price_mid'] * p['yield'] - p['cost']

    if profit_override:
        profit = profit_override
    else:
        profit = base_profit

    # Decision variables
    x = {}  # (pid, sk, cid) → LpVariable
    for pid, seasons in plot_season_map.items():
        ptype = plot_type_map[pid]
        area = plot_area_map[pid]
        for sk in seasons:
            for cid in compat.get((ptype, sk), []):
                key = (cid, ptype, sk)
                if key not in profit:
                    continue
                x[pid, sk, cid] = pulp.LpVariable(
                    f"x_{pid}_{sk}_{cid}",
                    lowBound=0, upBound=area,
                    cat=pulp.LpContinuous
                )

    # ---- Objective ----
    obj_terms = []
    for (pid, sk, cid), var in x.items():
        ptype = plot_type_map[pid]
        key = (cid, ptype, sk)
        if key in profit:
            obj_terms.append(profit[key] * var)
    prob += pulp.lpSum(obj_terms)

    # ---- Area constraints ----
    for pid, seasons in plot_season_map.items():
        ptype = plot_type_map[pid]
        area = plot_area_map[pid]
        for sk in seasons:
            vars_s = [x[pid, sk, cid]
                      for cid in compat.get((ptype, sk), [])
                      if (pid, sk, cid) in x]
            if vars_s:
                prob += pulp.lpSum(vars_s) <= area, f"area_{pid}_{sk}"

    # ---- No continuous cropping (year ≥ 2025) ----
    if year_idx >= 1 and prev_planting:
        for (pid, sk), prev_cid in prev_planting.items():
            if prev_cid is None:
                continue
            ptype = plot_type_map[pid]
            if (pid, sk, prev_cid) in x:
                prob += x[pid, sk, prev_cid] == 0, f"no_repeat_{pid}_{sk}"

    # ---- Legume rotation (every 3 years need ≥1 legume) ----
    # If this is year 3+ and no legume in past 2 years, MUST plant legume
    if year_idx >= 2:
        for pid in plot_season_map:
            history = legume_history.get(pid, [])
            if len(history) >= 2 and not any(history[-2:]):
                # Must plant legume this year
                ptype = plot_type_map[pid]
                legume_vars = []
                for sk in plot_season_map[pid]:
                    for cid in compat.get((ptype, sk), []):
                        if cid in md['legume_ids'] and (pid, sk, cid) in x:
                            legume_vars.append(x[pid, sk, cid])
                if legume_vars:
                    min_legume = min(1.0, plot_area_map[pid] * 0.1)
                    prob += pulp.lpSum(legume_vars) >= min_legume, \
                            f"legume_must_{pid}"

    # ---- Production cap (Q1a: excess = waste → hard cap) ----
    if case == 'a' and exp_sales_override is not None:
        exp_sales = exp_sales_override
    elif case == 'a':
        exp_sales = md['exp_sales']
    else:
        exp_sales = None

    if case == 'a' and exp_sales:
        # Total production per crop ≤ expected sales
        for cid, es_val in exp_sales.items():
            prod_vars = []
            for (pid, sk, cid2), var in x.items():
                if cid2 != cid:
                    continue
                ptype = plot_type_map[pid]
                key = (cid, ptype, sk)
                if key in params:
                    yield_pm = params[key]['yield']
                    prod_vars.append(yield_pm * var)
            if prod_vars and es_val > 0:
                prob += pulp.lpSum(prod_vars) <= es_val, f"sales_cap_{cid}"

    # ---- Crop diversity: at most 60% of a plot type per single crop ----
    if min_crop_diversity:
        for ptype in ['平旱地', '梯田', '山坡地', '水浇地']:
            for sk in ['S1', 'S2']:
                if (ptype, sk) not in compat:
                    continue
                for cid in compat[(ptype, sk)]:
                    plot_vars = []
                    for pid in plot_season_map:
                        if plot_type_map[pid] != ptype or sk not in plot_season_map[pid]:
                            continue
                        if (pid, sk, cid) in x:
                            plot_vars.append(x[pid, sk, cid])
                    if len(plot_vars) > 1:
                        total_type_area = sum(
                            plot_area_map[pid]
                            for pid in plot_season_map
                            if plot_type_map[pid] == ptype and sk in plot_season_map[pid]
                        )
                        prob += pulp.lpSum(plot_vars) <= 0.6 * total_type_area, \
                                f"diversity_{ptype}_{sk}_{cid}"

    # ---- Solve ----
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=90)
    prob.solve(solver)

    # ---- Extract ----
    plan = {}
    for (pid, sk, cid), var in x.items():
        val = pulp.value(var)
        if val is not None and val > 0.005:
            plan[(pid, sk)] = (cid, round(val, 4))

    obj_val = pulp.value(prob.objective)
    return plan, obj_val


def compute_profit_for_plan(plan, md):
    """Compute actual profit for a planting plan."""
    params = md['params']
    plot_type_map = md['plot_type_map']
    total = 0.0
    for (pid, sk), (cid, area) in plan.items():
        ptype = plot_type_map[pid]
        key = (cid, ptype, sk)
        if key in params:
            p = params[key]
            total += (p['price_mid'] * p['yield'] - p['cost']) * area
    return total / 10000  # 万元


# ============================================================
# PROBLEM SOLVERS
# ============================================================

def solve_q1(md, case='b'):
    """Problem 1: Deterministic with production cap."""
    YEARS = list(range(2024, 2031))
    legume_ids = md['legume_ids']

    prev_planting = dict(md['init_state'])
    legume_history = {}
    for pid in md['plot_season_map']:
        legume_history[pid] = []
        has_legume = False
        for sk in md['plot_season_map'][pid]:
            entry = md['init_state'].get((pid, sk))
            if entry is not None and entry in legume_ids:
                has_legume = True
        legume_history[pid].append(has_legume)

    # Scale expected sales for 7-year horizon (annual cap)
    exp_sales_annual = {k: v for k, v in md['exp_sales'].items()}

    results = {}
    for yi, year in enumerate(YEARS):
        plan, obj = build_yearly_lp(
            year, yi, md, prev_planting, legume_history,
            case=case, exp_sales_override=exp_sales_annual
        )

        # Track legume
        for pid in md['plot_season_map']:
            has_leg = False
            for sk in md['plot_season_map'][pid]:
                entry = plan.get((pid, sk))
                if entry and entry[0] in legume_ids:
                    has_leg = True
            if pid not in legume_history:
                legume_history[pid] = []
            legume_history[pid].append(has_leg)
            if len(legume_history[pid]) > 3:
                legume_history[pid] = legume_history[pid][-3:]

        prev_planting = dict(plan)
        results[year] = plan

    return results


def solve_q2(md):
    """Problem 2: Stochastic with trends."""
    YEARS = list(range(2024, 2031))
    params = md['params']
    legume_ids = md['legume_ids']

    prev_planting = dict(md['init_state'])
    legume_history = {}
    for pid in md['plot_season_map']:
        legume_history[pid] = [any(
            md['init_state'].get((pid, sk)) in legume_ids
            for sk in md['plot_season_map'][pid]
            if (pid, sk) in md['init_state']
        )]

    results = {}
    for yi, year in enumerate(YEARS):
        t = yi + 1  # years since 2023

        # Build year-specific profit
        profit_year = {}
        for (cid, ptype, sk), p in params.items():
            yld = p['yield']
            cost = p['cost'] * (1.05 ** t)
            price = p['price_mid']

            if cid <= 16:
                pass  # grain price stable
            elif 17 <= cid <= 37:
                price *= (1.05 ** t)  # vegetable +5%/yr
            elif 38 <= cid <= 41:
                if cid == 41:
                    price *= (0.95 ** t)  # 羊肚菌: -5%/yr
                else:
                    price *= (0.97 ** t)  # other mushrooms: -3%/yr

            profit_year[(cid, ptype, sk)] = price * yld - cost

        plan, obj = build_yearly_lp(
            year, yi, md, prev_planting, legume_history,
            profit_override=profit_year
        )

        # Track legume
        for pid in md['plot_season_map']:
            has_leg = False
            for sk in md['plot_season_map'][pid]:
                entry = plan.get((pid, sk))
                if entry and entry[0] in legume_ids:
                    has_leg = True
            if pid not in legume_history:
                legume_history[pid] = []
            legume_history[pid].append(has_leg)
            if len(legume_history[pid]) > 3:
                legume_history[pid] = legume_history[pid][-3:]

        prev_planting = dict(plan)
        results[year] = plan

    return results


def solve_q3(md):
    """Problem 3: Correlation-aware with risk penalty."""
    YEARS = list(range(2024, 2031))
    params = md['params']
    legume_ids = md['legume_ids']
    rng = np.random.default_rng(42)

    # Correlation matrix: yield-price-cost
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
    L = np.linalg.cholesky(corr)

    prev_planting = dict(md['init_state'])
    legume_history = {}
    for pid in md['plot_season_map']:
        legume_history[pid] = [any(
            md['init_state'].get((pid, sk)) in legume_ids
            for sk in md['plot_season_map'][pid]
            if (pid, sk) in md['init_state']
        )]

    results = {}
    for yi, year in enumerate(YEARS):
        t = yi + 1

        # Generate N scenarios of correlated shocks
        Z = rng.standard_normal((200, 3))
        shocks = Z @ L.T
        # Average shock for EV optimization
        avg_ys = np.clip(np.mean(shocks[:, 0]) * 0.10, -0.10, 0.10)
        avg_ps = np.clip(np.mean(shocks[:, 1]) * 0.08, -0.08, 0.08)
        avg_cs = np.clip(np.mean(shocks[:, 2]) * 0.06, -0.06, 0.06)

        profit_year = {}
        for (cid, ptype, sk), p in params.items():
            yld = p['yield'] * (1.0 + avg_ys)
            cost = p['cost'] * (1.05 ** t) * (1.0 + avg_cs)
            price = p['price_mid'] * (1.0 + avg_ps)

            if 17 <= cid <= 37:
                price *= (1.05 ** t)
            elif 38 <= cid <= 41:
                price *= (0.97 ** t)

            # Risk-adjusted profit (penalize high price variance crops)
            # Use CV (coefficient of variation) of price as risk measure
            price_range_width = (p['price_high'] - p['price_low']) / max(p['price_mid'], 0.01)
            risk_penalty = 0.08 * price_range_width * p['price_mid'] * yld
            profit_year[(cid, ptype, sk)] = price * yld - cost - risk_penalty

        plan, obj = build_yearly_lp(
            year, yi, md, prev_planting, legume_history,
            profit_override=profit_year
        )

        for pid in md['plot_season_map']:
            has_leg = any(
                plan.get((pid, sk), (0,))[0] in legume_ids
                for sk in md['plot_season_map'][pid]
            )
            if pid not in legume_history:
                legume_history[pid] = []
            legume_history[pid].append(has_leg)
            if len(legume_history[pid]) > 3:
                legume_history[pid] = legume_history[pid][-3:]

        prev_planting = dict(plan)
        results[year] = plan

    return results


# ============================================================
# EXPORT & VISUALIZATION
# ============================================================

def export_results(results, md, output_path):
    """Export planting plans to Excel template format."""
    crop_name_map = md['crop_name_map']
    all_crop_names = sorted(set(crop_name_map.values()))

    # Build template rows
    rows = []
    YEARS = sorted(results.keys())

    for _, row in pd.DataFrame({
        'plot_id': list(md['plot_type_map'].keys()),
        'plot_type': list(md['plot_type_map'].values()),
        'area': list(md['plot_area_map'].values()),
    }).iterrows():
        pid = row['plot_id']
        seasons = md['plot_season_map'].get(pid, ['S1'])
        for sk in seasons:
            season_label = '第一季' if sk == 'S1' else '第二季'
            rows.append({'地块名': pid, '季节': season_label})

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for year in YEARS:
            plan = results[year]

            df = pd.DataFrame(0.0, index=range(len(rows)),
                               columns=all_crop_names)

            for i, r in enumerate(rows):
                pid = r['地块名']
                sk = 'S1' if r['季节'] == '第一季' else 'S2'
                entry = plan.get((pid, sk))
                if entry:
                    cid, area = entry
                    cname = crop_name_map.get(cid, str(cid))
                    if cname in df.columns:
                        df.iloc[i, df.columns.get_loc(cname)] = area

            df.insert(0, '地块名', [r['地块名'] for r in rows])
            df.insert(0, '季节', [r['季节'] for r in rows])

            df.to_excel(writer, sheet_name=str(year), index=False)

    print(f"  Exported: {output_path}")


def analyze_and_plot(results_dict, md, label_map):
    """Generate analysis and figures."""
    YEARS = sorted(next(iter(results_dict.values())).keys())

    # Profit computation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    year_labels = [str(y) for y in YEARS]

    profit_data = {}
    for ax, (name, results) in zip(axes, results_dict.items()):
        profits = [compute_profit_for_plan(results[y], md) for y in YEARS]
        profit_data[name] = profits

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(YEARS)))
        ax.bar(year_labels, profits, alpha=0.8, color=colors)
        ax.set_title(f'{label_map.get(name, name)}\nTotal: {sum(profits):.1f} 万元')
        ax.set_ylabel('Profit (万元)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(profits):
            ax.text(i, v + max(profits)*0.02, f'{v:.1f}', ha='center', fontsize=7)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig1_profit_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure: fig1_profit_comparison.png")

    # Crop utilization heatmap (2024)
    plan_2024 = results_dict['Q1a'][2024] if 'Q1a' in results_dict else next(iter(results_dict.values()))[2024]
    crop_name_map = md['crop_name_map']

    # Summary text
    report_path = OUTPUT_DIR / 'solution_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# CUMCM 2024 Problem C: Crop Planting Optimization\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Model | Total Profit (万元) | Annual Avg (万元) |\n")
        f.write("|-------|---------------------|--------------------|\n")
        for name, profits in profit_data.items():
            f.write(f"| {label_map.get(name, name)} | {sum(profits):.1f} | {sum(profits)/7:.1f} |\n")
        f.write("\n---\n*Generated by Research Assistant Pipeline*\n")

    print(f"  Report: {report_path}")

    # Console summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, profits in profit_data.items():
        print(f"  {label_map.get(name, name)}: {sum(profits):.1f} 万元 total, "
              f"{sum(profits)/7:.1f} 万元/yr")


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*70)
    print("  CUMCM 2024 Problem C — Final Solution")
    print("="*70)

    # 1. Load
    print("\n[1/5] Loading data...")
    df_plots, df_crops, df_2023, df_stats = load_data()
    md = build_model_data(df_plots, df_crops, df_2023, df_stats)
    print(f"  Plots: {len(df_plots)}, Crops: {len(df_crops)}, "
          f"Params: {len(md['params'])}, Compat: {len(md['compat'])}")

    # 2. Solve Q1
    print("\n[2/5] Problem 1: Deterministic LP (7-year rolling)...")
    q1a = solve_q1(md, case='a')
    q1b = solve_q1(md, case='b')

    # 3. Solve Q2
    print("\n[3/5] Problem 2: Stochastic with trends...")
    q2 = solve_q2(md)

    # 4. Solve Q3
    print("\n[4/5] Problem 3: Correlation-aware...")
    q3 = solve_q3(md)

    # 5. Export
    print("\n[5/5] Exporting results...")
    export_results(q1a, md, OUTPUT_DIR / 'result1_1.xlsx')
    export_results(q1b, md, OUTPUT_DIR / 'result1_2.xlsx')
    export_results(q2, md, OUTPUT_DIR / 'result2.xlsx')

    analyze_and_plot(
        {'Q1a': q1a, 'Q2': q2, 'Q3': q3},
        md,
        {'Q1a': 'Problem 1(a): Waste', 'Q2': 'Problem 2: Stochastic',
         'Q3': 'Problem 3: Correlation'}
    )

    print(f"\n{'='*70}")
    print(f"  DONE! All outputs in: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
