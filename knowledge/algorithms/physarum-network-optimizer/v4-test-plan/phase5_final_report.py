"""
Phase 5F: 综合报告生成器
========================
合并 5A-5E 全部结果，生成综合报告 + 论文定位建议。

用法:
    python phase5_final_report.py
"""

import json
from pathlib import Path

import numpy as np
import scipy.stats as sps

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'outputs'
REPORTS_DIR = BASE_DIR / 'reports'
RESULT_FILES = {
    'ablation': OUTPUT_DIR / 'phase5_ablation.json',
    'cec2017': OUTPUT_DIR / 'phase5_cec2017.json',
    'scalability': OUTPUT_DIR / 'phase5_scalability.json',
    'complexity': OUTPUT_DIR / 'phase5_complexity.json',
    'improvement': OUTPUT_DIR / 'phase5_improvement_verify.json',
}


def load_all():
    """加载全部实验结果，缺的返回 None"""
    data = {}
    for k, path in RESULT_FILES.items():
        if path.exists():
            data[k] = json.load(open(path, 'r', encoding='utf-8'))
        else:
            data[k] = None
    return data


def load_report(name):
    p = REPORTS_DIR / f'report_phase5_{name}.md'
    return p.read_text(encoding='utf-8') if p.exists() else None


def build_ablation_section(data):
    if data is None:
        return '## 1. 消融实验\n\n⚠️ 数据未就绪\n'
    # 定稿数字取自 report_phase5_ablation.md（对数空间比率、1e6 上限、
    # 收敛至机器精度时比率封顶）。此处为权威摘要，非重算。
    lines = ['## 1. 消融实验\n']
    lines.append('| 机制 | 中位退化倍率 | 显著性 (Wilcoxon) | 效应量 |')
    lines.append('|------|------------|------------------|--------|')
    lines.append('| **StrategyPool（PNO↔GWO 切换）** | **193×（封顶）** | **8/8 函数显著** | **d=1.70** |')
    lines.append('| SHCA 参数自适应 | 1.46× | 1/8 | 弱 |')
    lines.append('| Cauchy 变异 | 1.30× | 2/8 | 弱 |')
    lines.append('| 外部归档 | 1.05× | 0/8 | 名义贡献 |')
    lines.append('| Nelder-Mead 精炼 | 1.01× | 4/8 | 收敛精度型 |')
    lines.append('\n- **Full vs Baseline（全关）：中位退化 291×**，混合结构整体有效'
                 '，但贡献高度集中于 StrategyPool。')
    lines.append('\n> 权威报告见 `reports/report_phase5_ablation.md` '
                 '（含逐函数比率、热力图 `outputs/figures/ablation_heatmap.png`）')
    return '\n'.join(lines)


def build_cec_section(data):
    if data is None:
        return '## 2. CEC 2017\n\n⚠️ 数据未就绪\n'
    # 计算平均排名（用每函数的序）
    funcs = {}
    for key, res in data.items():
        funcs.setdefault(res['function'], {})[res['algorithm']] = res['stats']['median']
    algos = sorted({res['algorithm'] for res in data.values()})
    ranks = {}
    for fn, adata in funcs.items():
        meds = {a: adata[a] for a in algos if a in adata}
        order = sorted(meds, key=lambda a: meds[a])
        for i, a in enumerate(order, 1):
            ranks.setdefault(a, []).append(i)
    mean_ranks = {a: np.mean(v) for a, v in ranks.items()}
    winner = min(mean_ranks, key=mean_ranks.get)

    lines = ['## 2. CEC 2017 对比\n']
    lines.append('> 基于 opfunu 1.0.4 的 CEC 2017 无约束套件'
                 '（28 函数 = F1 + F3-F29，D=10，30K FES，10 次；官方 F2 排除，'
                 'F30 该库未提供）')
    lines.append('\n平均排名（越低越好）:\n')
    lines.append('| 排名 | 算法 | 平均排名 |')
    lines.append('|------|------|---------|')
    for i, (a, r) in enumerate(sorted(mean_ranks.items(), key=lambda x: x[1]), 1):
        marker = ' ★' if a == 'PNO-GWO-v4.0' else ''
        lines.append(f'| {i} | {a}{marker} | {r:.2f} |')
    lines.append('\n> 完整报告见 `reports/report_phase5_cec2017.md`')
    return '\n'.join(lines)


def build_scalability_section(data):
    if data is None:
        return '## 3. 高维可扩展性\n\n⚠️ 数据未就绪\n'
    lines = ['## 3. 高维可扩展性\n']
    # 每个算法的 log10 中位数均值随维度
    algos = sorted({res['algorithm'] for res in data.values()})
    dims = sorted({res['dim'] for res in data.values()})
    lines.append('| 算法 | ' + ' | '.join(f'{d}D' for d in dims) + ' |')
    lines.append('|------|' + '------|' * len(dims))
    for a in algos:
        row = []
        for d in dims:
            vals = [np.log10(max(res['stats']['median'], 1e-300))
                    for res in data.values()
                    if res['dim'] == d and res['algorithm'] == a]
            row.append(f'{np.mean(vals):.2f}' if vals else '-')
        lines.append(f'| {a} | ' + ' | '.join(row) + ' |')
    lines.append('\n> **解读**：表内为 5 函数 log10 中位均值的均值，被可解到机器精度的'
                 ' Sphere/Griewank 支配——PNO-GWO 与 GWO 的负量级差异（1e-130 vs 1e-150）'
                 '无实际意义，两者均达标；真正的分水岭是 **PSO/DE 从 300D 起崩溃'
                 '（log10 从 0 翻正）**，而 PNO-GWO 在 500D（30 万 FES）仍保持负量级。')
    lines.append('\n> 完整报告见 `reports/report_phase5_scalability.md`')
    return '\n'.join(lines)


def build_improvement_section(data):
    if data is None:
        return '## 5. v4.1 改进验证\n\n⚠️ 数据未就绪\n'
    funcs = sorted({k.split('|')[0] for k in data})
    lines = ['## 5. v4.1 改进验证 (基于 Phase 5 结果)\n']
    lines.append('> 依据 5B 实测: CEC 2017 旋转/混合函数 PNO 落后 DE 族'
                 ' (SHADE/jSO/CMODE) 100-1000×, F27 落入组合陷阱\n'
                 '> → v4.1 在策略切换中加入 SHADE 式 pbest-DE 臂'
                 ' (current-to-pbest/1 + 二项交叉 + F/CR 成功历史)')
    lines.append('\n| 函数 | v4.0 中位 | v4.1 中位 | Δ(v4.1-v4.0) | Wilcoxon p |')
    lines.append('|------|----------|----------|------------|-----------|')
    n_worse = 0
    for fn in funcs:
        a = np.array(data[f'{fn}|v4.0'])
        b = np.array(data[f'{fn}|v4.1'])
        ma, mb = float(np.median(a)), float(np.median(b))
        p = sps.wilcoxon(a, b).pvalue if not np.all(a == b) else 1.0
        arrow = '↓' if mb < ma else ('↑' if mb > ma else '=')
        if mb > ma:
            n_worse += 1
        lines.append(f'| {fn} | {ma:.2e} | {mb:.2e} | {mb-ma:.1e} {arrow} | {p:.3f} |')
    # 显著项（p<0.1）判定
    sig = []
    for fn in funcs:
        a = np.array(data[f'{fn}|v4.0'])
        b = np.array(data[f'{fn}|v4.1'])
        p = sps.wilcoxon(a, b).pvalue if not np.all(a == b) else 1.0
        if p < 0.1:
            sig.append(f'{fn}(Δ={"↓" if np.median(b)<np.median(a) else "↑"}, p={p:.3f})')
    sig_txt = '，'.join(sig) if sig else '无'
    lines.append(f'\n> 中位方向性上升 {n_worse}/{len(funcs)} 个，'
                 f'但 p<0.1 的显著项仅 {sig_txt}；'
                 '其余方向性差异为抽样噪音（n=20）。'
                 '完整报告见 `reports/report_phase5_improvement.md`')
    return '\n'.join(lines)


def build_complexity_section(data):
    if data is None:
        return '## 4. 计算复杂度\n\n⚠️ 数据未就绪\n'
    comp = data['component_marginal']
    timing = data['timing']
    # 快速拟合
    pts = [(d['N'], d['D'], d['median_time']) for d in timing.values()
           if d.get('median_time')]
    if len(pts) > 3:
        X = np.array([[np.log(p[0]), np.log(p[1]), 1] for p in pts])
        y = np.array([np.log(p[2]) for p in pts])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        b, c, loga = coef
        fit = f'T ≈ {np.exp(loga):.3f} × N^{{{b:.2f}}} × D^{{{c:.2f}}}'
    else:
        fit = '数据不足'

    lines = ['## 4. 计算复杂度\n']
    lines.append(f'时间拟合: **{fit}**\n')
    lines.append('| 组件 | 边际成本 (s) |')
    lines.append('|------|-------------|')
    for comp_name, m in comp.items():
        lines.append(f'| {comp_name} | {m["marginal_cost"]:.4f} |')
    lines.append('\n> 完整报告见 `reports/report_phase5_complexity.md`')
    return '\n'.join(lines)


def main():
    data = load_all()
    lines = []
    lines.append('# Phase 5 综合报告: PNO-GWO v4.0 冲击 Q2 评估\n')
    from datetime import datetime
    lines.append(f'> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')

    lines.append('## 目录\n')
    lines.append('1. [消融实验](#1-消融实验)')
    lines.append('2. [CEC 2017 对比](#2-cec-2017-对比)')
    lines.append('3. [高维可扩展性](#3-高维可扩展性)')
    lines.append('4. [计算复杂度](#4-计算复杂度)')
    lines.append('5. [v4.1 改进验证](#5-v41-改进验证)')
    lines.append('6. [论文定位建议](#6-论文定位建议)')
    lines.append('7. [后续改进方向](#7-后续改进方向)\n')

    lines.append(build_ablation_section(data['ablation']))
    lines.append('\n' + build_cec_section(data['cec2017']))
    lines.append('\n' + build_scalability_section(data['scalability']))
    lines.append('\n' + build_complexity_section(data['complexity']))
    lines.append('\n' + build_improvement_section(data['improvement']))

    lines.append('\n## 6. 论文定位建议\n')
    lines.append('**数据画像**：CEC 2017 排名第 7（弱项集中在旋转/混合/组合函数，'
                 'v4.1 已针对最大失分点 F1 显著修复 p=0.015）；高维可扩展性稳'
                 '（500D 仍保持负量级，PSO/DE 同期崩溃）；复杂度近线性'
                 '（T≈0.002·N^1.02·D^0.06）；消融贡献高度集中于 StrategyPool'
                 '（193×，d=1.70）。\n')
    lines.append('**定位建议**：不宜以"CEC 全能冠军"行文，应主打三条实证主线——\n')
    lines.append('1. **图-狼群混合结构的可竞争性**：消融揭示策略池为核心贡献，'
                 '分层机制各司其职；')
    lines.append('2. **高维下的稳定性**：100D→500D 单调保持解精度，'
                 '对比算法（PSO/DE/CMA-ES）在 300D+ 失效；')
    lines.append('3. **旋转鲁棒补强**（v4.1）：针对 CEC 诊断加入 SHADE 式第三臂，'
                 '显著修复旋转 Bent Cigar（−30%），机制保真实验设计可复现。\n')
    lines.append('- 目标期刊：Applied Soft Computing / EAAI（Q2）可行，但需在正文'
                 '诚实声明 CEC 2017 的 7/10 位次并将其列为未来工作；')
    lines.append('- 摘要要点：混合结构 + 消融完备性 + 高维稳定性 + 复杂度近线性 + '
                 'v4.1 旋转补强的显著增益。')

    lines.append('\n## 7. 后续改进方向\n')
    lines.append('- **CEC 全量升级**：FES 升至官方标准（100K+）并补约束版'
                 ' C01–C28，验证 v4.1 在小预算旋转函数上的增益是否随预算扩展；')
    lines.append('- **策略池真接线**：激活当前死代码的图差分/存档引导策略，'
                 '让 4+1 臂真正参与成功历史竞争（当前仅 PNO↔GWO 二元切换）；')
    lines.append('- **CMA 方向融合**：CEC 显示 CMA-ES 在组合函数上的梯度方向优势，'
                 '可作为第 4 臂；')
    lines.append('- **v4.1 复测**：n=20 为聚焦验证，正式提交前对旋转/组合靶区'
                 '扩至 n=30 并增补工程约束函数。')

    report_path = REPORTS_DIR / 'report_phase5_final.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'综合报告已生成: {report_path}')


if __name__ == '__main__':
    main()