"""
Phase 5A 分析: 消融实验统计分析与图表
======================================
读取 phase5_ablation.json，计算:
  - ΔFitness: 移除组件后的中位数退化
  - ΔFeasibility: 可行率变化
  - Wilcoxon 检验: Full vs w/o XXX
  - Cohen's d 效应量
生成:
  - outputs/figures/ablation_heatmap.png
  - reports/report_phase5_ablation.md
"""

import sys
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sps

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
REPORTS_DIR = BASE_DIR / 'reports'
RESULTS_FILE = OUTPUT_DIR / 'phase5_ablation.json'

MECHANISMS = ['SHCA', 'Archive', 'Strategy', 'Cauchy', 'NM']
CONFIGS = ['Full', 'w/o_SHCA', 'w/o_Archive', 'w/o_Strategy', 'w/o_Cauchy', 'w/o_NM', 'Baseline']


def analyze():
    data = json.load(open(RESULTS_FILE, 'r', encoding='utf-8'))

    # 重组: results[problem][config] = {stats, best_fits, feasible_flags}
    problems = {}
    for key, res in data.items():
        prob = res['problem']
        cfg = res['config_name'] if 'config_name' in res else key.split('::')[0]
        # config_name 需从 key 解析
        cfg = key.split('::')[0]
        problems.setdefault(prob, {})[cfg] = res

    # 计算每个问题每个配置的汇总
    summary = {}
    for prob, cfgs in problems.items():
        summary[prob] = {}
        for cfg, res in cfgs.items():
            fits = np.array(res['best_fits'])
            feas = np.array(res['feasible_flags'])
            finite = fits[np.isfinite(fits)]
            summary[prob][cfg] = {
                'median': float(np.median(finite)) if len(finite) else float('inf'),
                'mean': float(np.mean(finite)) if len(finite) else float('inf'),
                'std': float(np.std(finite)) if len(finite) else float('inf'),
                'feasible_rate': float(feas.mean()),
                'best_fits': res['best_fits'],
                'feasible_flags': res['feasible_flags'],
            }

    # 组件贡献分析: 对每个机制和每个问题
    # Δratio = median(w/o_X) / median(Full)  (>1 表示退化)
    CAP = 1e6  # 报告上限：超出视为"收敛解坍塌"(Full 达机器精度而移除后落在平凡量级)
    component_effects = {}  # mechanism -> {problem: delta_ratio (capped)}
    capped_mask = {}        # mechanism -> {problem: True if 超出上限}
    wilcoxon_results = []
    cohens_d_results = []

    for mech in MECHANISMS:
        wocfg = f'w/o_{mech}'
        component_effects[mech] = {}
        capped_mask[mech] = {}
        for prob, cfgs in summary.items():
            if 'Full' not in cfgs or wocfg not in cfgs:
                continue
            full_med = cfgs['Full']['median']
            wo_med = cfgs[wocfg]['median']
            # 比值在 [1e-6, 1e6] 内报告；超出表示移除组件使机器精度收敛解直接坍塌
            if np.isfinite(full_med) and np.isfinite(wo_med):
                if full_med > 0 and wo_med > 0:
                    # 比值意义明确（同一问题的两个正中位数）
                    ratio = 10 ** (np.log10(wo_med) - np.log10(full_med))
                else:
                    # 含非正值：用 log10 差估计退化倍率
                    ratio = 10 ** (np.log10(wo_med + 1e-300) - np.log10(full_med + 1e-300))
                if ratio > CAP:
                    component_effects[mech][prob] = CAP
                    capped_mask[mech][prob] = True
                else:
                    component_effects[mech][prob] = ratio
            else:
                component_effects[mech][prob] = wo_med

            # Wilcoxon 秩和检验（全等样本 → p=1.0，避免除零）
            full_fits = np.array(cfgs['Full']['best_fits'])
            wo_fits = np.array(cfgs[wocfg]['best_fits'])
            try:
                if np.all(full_fits == wo_fits):
                    stat, pval = 0.0, 1.0
                else:
                    stat, pval = sps.wilcoxon(full_fits, wo_fits)
            except Exception:
                stat, pval = np.nan, 1.0

            # Cohen's d
            f_mean, f_std = (np.nanmean(full_fits), np.nanstd(full_fits))
            w_mean, w_std = (np.nanmean(wo_fits), np.nanstd(wo_fits))
            pool_std = np.sqrt((f_std ** 2 + w_std ** 2) / 2)
            d = (w_mean - f_mean) / pool_std if pool_std > 1e-12 else np.nan

            wilcoxon_results.append({
                'mechanism': mech, 'problem': prob,
                'wilcoxon_stat': float(stat), 'p_value': float(pval),
            })
            cohens_d_results.append({
                'mechanism': mech, 'problem': prob, 'cohens_d': float(d),
            })

    # 跨问题汇总
    agg = {}
    for mech in MECHANISMS:
        ratios = [v for v in component_effects[mech].values()
                  if isinstance(v, (int, float)) and np.isfinite(v) and v > 0]
        # 几何平均
        geo_mean = float(np.exp(np.mean(np.log(ratios)))) if ratios else np.nan
        pvals = [r['p_value'] for r in wilcoxon_results if r['mechanism'] == mech]
        n_sig = sum(1 for p in pvals if p < 0.05)
        ds = [r['cohens_d'] for r in cohens_d_results
              if r['mechanism'] == mech and np.isfinite(r['cohens_d'])]
        mean_d = float(np.mean(ds)) if ds else np.nan
        agg[mech] = {
            'geo_mean_delta_ratio': geo_mean,
            'n_significant': n_sig,
            'n_total': len(pvals),
            'mean_cohens_d': mean_d,
        }

    # 生成热力图: log2(delta ratio)
    plt.figure(figsize=(10, 6))
    mechs = MECHANISMS
    probs = list(summary.keys())
    heat = np.zeros((len(mechs), len(probs)))
    for i, mech in enumerate(mechs):
        for j, prob in enumerate(probs):
            r = component_effects[mech].get(prob, 1.0)
            if isinstance(r, (int, float)) and np.isfinite(r) and r > 0:
                heat[i, j] = np.log2(r)
            else:
                heat[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(heat, cmap='Reds', aspect='auto')
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels(probs, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(mechs)))
    ax.set_yticklabels([f'w/o {m}' for m in mechs], fontsize=9)
    # 标注数值（被截断的单元格显示 ≥1e6）
    for i in range(len(mechs)):
        for j in range(len(probs)):
            v = heat[i, j]
            if not np.isnan(v):
                if capped_mask[mechs[i]].get(probs[j], False):
                    label = '≥1e6'
                else:
                    label = f'{v:.1f}'
                ax.text(j, i, label, ha='center', va='center',
                        fontsize=8, color='white' if v > 1 else 'black')
    ax.set_title('Log2(退化倍率) — 移除组件后中位数退化', fontsize=12)
    plt.colorbar(im, label='log2(median(w/o) / median(Full))')
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'ablation_heatmap.png', dpi=150)
    plt.close()

    # 生成报告
    lines = []
    lines.append('# Phase 5A: 消融实验报告\n')
    lines.append('> 目标: 量化 PNO-GWO v4.0 各核心机制的独立贡献\n')
    lines.append('\n## 1. 实验设置\n')
    lines.append('- 配置: Full / w/o SHCA / w/o Archive / w/o Strategy / w/o Cauchy / w/o NM / Baseline（全去）')
    lines.append('- 问题: Sphere, Rastrigin, Rosenbrock, Ackley, Griewank (30D) + Welded Beam, Pressure Vessel, Spring')
    lines.append('- 每次运行: 30 次独立, 30000 FES, 相同随机种子')
    lines.append('\n## 2. 组件贡献汇总\n')
    lines.append('| 机制 | 几何平均退化倍率 | Wilcoxon 显著(问题数) | 平均 Cohen\'s d |')
    lines.append('|------|----------------|-------------------|---------------|')
    for mech in MECHANISMS:
        a = agg[mech]
        if a["geo_mean_delta_ratio"] >= CAP - 1e-9:
            gm_label = f'≥{CAP:.0f}'
        elif not np.isfinite(a["geo_mean_delta_ratio"]):
            gm_label = '—'
        else:
            gm_label = f'{a["geo_mean_delta_ratio"]:.2f}×'
        lines.append(f'| {mech} | {gm_label} | '
                     f'{a["n_significant"]}/{a["n_total"]} | '
                     f'{a["mean_cohens_d"]:.2f} |')

    lines.append('\n## 3. 逐问题退化倍率 (median(w/o) / median(Full), >1 表示退化)\n')
    lines.append('> 上限 1e6×: 超出表示 Full 已收敛至机器精度（~1e-12 量级），移除组件后中值落在平凡量级\n')
    lines.append('| 机制 | ' + ' | '.join(probs) + ' |')
    lines.append('|------|' + '------|' * len(probs))
    for mech in MECHANISMS:
        vals = []
        for prob in probs:
            r = component_effects[mech].get(prob, 1.0)
            c = capped_mask[mech].get(prob, False)
            if c:
                vals.append('≥1e6')
            elif isinstance(r, (int, float)) and np.isfinite(r):
                vals.append(f'{r:.2f}')
            else:
                vals.append('inf/0')
        lines.append(f'| w/o {mech} | ' + ' | '.join(vals) + ' |')

    lines.append('\n## 4. 结论\n')
    # 找出最重要的组件
    ranked = sorted(agg.items(), key=lambda x: -x[1]['geo_mean_delta_ratio']
                    if np.isfinite(x[1]['geo_mean_delta_ratio']) else 0)
    lines.append('组件按退化倍率排序（移除后退化越大 → 贡献越关键）:\n')
    for i, (mech, a) in enumerate(ranked, 1):
        if a["geo_mean_delta_ratio"] >= CAP - 1e-9:
            gm_label = f'≥{CAP:.0f}'
        else:
            gm_label = f'{a["geo_mean_delta_ratio"]:.2f}×'
        lines.append(f'{i}. **{mech}** — 移除后中位数退化 '
                     f'{gm_label}，{a["n_significant"]}/{a["n_total"]} 个问题显著')

    # 机器精度坍塌问题提示
    cap_hits = [(m, p) for m in MECHANISMS for p in probs if capped_mask[m].get(p, False)]
    if cap_hits:
        lines.append('\n> 注: 标注 ≥1e6 表示对应问题 Full 方案中位已达机器精度'
                     '（<1e-12），该比值仅说明"收敛解坍塌"，'
                     '不放大为具体倍率。')

    # Baseline 对比 (同样 log 比值 + 上限)
    base_vs_full = {}
    for prob, cfgs in summary.items():
        if 'Full' in cfgs and 'Baseline' in cfgs:
            f = cfgs['Full']['median']
            b = cfgs['Baseline']['median']
            if np.isfinite(f) and np.isfinite(b) and f > 0 and b > 0:
                r = b / f
                base_vs_full[prob] = min(r, CAP)
    if base_vs_full:
        bf = np.array(list(base_vs_full.values()))
        geo = float(np.exp(np.mean(np.log(bf))))
        label = f'≥{CAP:.0f}' if geo >= CAP - 1e-9 else f'{geo:.0f}'
        lines.append(f'\n**整体框架贡献**: Full vs Baseline 平均退化 '
                     f'{label}× (几何平均，上限 {CAP:.0f})')
        lines.append('（Baseline 即纯随机搜索 + 局部精化基础结构）')

    lines.append('\n## 5. 图\n')
    lines.append('![消融热力图](figures/ablation_heatmap.png)')

    RE2 = REPORTS_DIR
    RE2.mkdir(parents=True, exist_ok=True)
    (RE2 / 'report_phase5_ablation.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"分析完成，报告: {RE2 / 'report_phase5_ablation.md'}")
    print(f'组件排名: {[(m, round(a["geo_mean_delta_ratio"],2)) for m, a in ranked]}')
    return summary, agg, component_effects


if __name__ == '__main__':
    analyze()