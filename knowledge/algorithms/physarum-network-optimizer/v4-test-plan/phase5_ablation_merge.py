"""
Phase 5A 合并: 并行消融输出 → 统一 JSON
========================================
合并 phase5_ablation_{suffix}.json 为 phase5_ablation.json
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'outputs'

CONFIGS = ['Full', 'w/o_SHCA', 'w/o_Archive', 'w/o_Strategy', 'w/o_Cauchy', 'w/o_NM', 'Baseline']


def main():
    merged = {}
    total = 0
    for cfg in CONFIGS:
        sfx = cfg.replace('/', '_')
        p = OUTPUT_DIR / f'phase5_ablation_{sfx}.json'
        if not p.exists():
            print(f'⚠️ 缺失: {p.name}')
            continue
        data = json.load(open(p, 'r', encoding='utf-8'))
        for key, val in data.items():
            merged[key] = val
            total += 1
        print(f'{cfg}: {len(data)} 组合')

    out = OUTPUT_DIR / 'phase5_ablation.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=1, ensure_ascii=False)
    print(f'\n合并完成: {total} 组合 → {out.name}')


if __name__ == '__main__':
    main()