# -*- coding: utf-8 -*-
"""Compile, render, audit, and catalogue every publication figure template."""

import importlib
from pathlib import Path
import py_compile
import shutil
import sys
import traceback

import matplotlib.image as mpimg

THIS_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = THIS_DIR / 'samples'
sys.path.insert(0, str(THIS_DIR))

TEMPLATES = [
    'template_line', 'template_bar', 'template_errorbar', 'template_heatmap',
    'template_scatter', 'template_tornado', 'template_3d',
    'template_bar_timeseries', 'template_distribution', 'template_forecast',
    'template_diagnostics', 'template_classification', 'template_pareto',
    'template_convergence', 'template_waterfall', 'template_multicriteria',
    'template_schedule', 'template_network', 'template_flow', 'template_map',
    'template_radar', 'template_pca',
]
RASTER_ONLY = {'template_bar_timeseries'}
COMPILE_MODULES = [
    '__init__', 'export_figure', 'figure_audit', 'palette', 'stat_helpers',
    'catalog', *TEMPLATES,
]
FORMATS = ('pdf', 'svg', 'png')
KEPT_SAMPLE_OUTPUTS = {'catalog_zh.png', 'catalog_en.png'}


def validate_outputs(stem, formats=FORMATS):
    from figure_audit import audit_export

    problems = []
    for extension in formats:
        path = SAMPLES_DIR / f'{stem}.{extension}'
        report = audit_export(path)
        problems.extend(f'{path.name}: {message}' for message in report.errors)
    png = SAMPLES_DIR / f'{stem}.png'
    if png.exists():
        image = mpimg.imread(png)
        if min(image.shape[:2]) < 900:
            problems.append(f'{png.name} 分辨率不足: {image.shape[1]}×{image.shape[0]}')
    return problems


def _render_bilingual(module, stem):
    default_png = SAMPLES_DIR / f'{stem}.png'
    zh_png = SAMPLES_DIR / f'{stem}_zh.png'
    shutil.copyfile(default_png, zh_png)
    module.demo(language='en', stem=f'{stem}_en')
    return [*validate_outputs(f'{stem}_zh', formats=('png',)),
            *validate_outputs(f'{stem}_en', formats=('png',))]


def cleanup_sample_outputs():
    """Keep the review catalogues while removing intermediate demo renders."""
    for path in SAMPLES_DIR.iterdir():
        if path.is_file() and path.name not in KEPT_SAMPLE_OUTPUTS:
            path.unlink()
    for cache in (THIS_DIR / '__pycache__', THIS_DIR / 'tests' / '__pycache__'):
        if cache.exists():
            shutil.rmtree(cache)


def main():
    from catalog import build_catalog

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for module_name in COMPILE_MODULES:
        try:
            py_compile.compile(str(THIS_DIR / f'{module_name}.py'), doraise=True)
        except Exception as error:
            results.append((module_name, f'FAIL 编译: {error}'))

    stems = []
    for module_name in TEMPLATES:
        try:
            module = importlib.import_module(module_name)
            module.demo()
            stem = f'{module_name.removeprefix("template_")}_demo'
            stems.append(module_name.removeprefix('template_'))
            formats = ('png',) if module_name in RASTER_ONLY else FORMATS
            problems = validate_outputs(stem, formats=formats)
            problems.extend(_render_bilingual(module, stem))
            results.append((module_name, 'PASS' if not problems
                            else 'FAIL ' + '; '.join(problems)))
        except Exception:
            traceback.print_exc()
            results.append((module_name, 'FAIL 渲染异常'))

    try:
        build_catalog('zh', stems=stems)
        build_catalog('en', stems=stems)
        results.append(('catalog', 'PASS'))
    except Exception:
        traceback.print_exc()
        results.append(('catalog', 'FAIL 生成异常'))

    cleanup_sample_outputs()

    print('\n===== MCM 期刊级模板验证 =====')
    all_pass = True
    for name, status in results:
        print(f'  {name:26s}  {status}')
        all_pass &= status == 'PASS'
    print('\n' + ('全部模板通过' if all_pass else '存在失败项'))
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
