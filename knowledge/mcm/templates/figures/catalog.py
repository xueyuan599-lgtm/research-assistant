# -*- coding: utf-8 -*-
"""Build compact PNG contact sheets for visual acceptance."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

THIS_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = THIS_DIR / 'samples'


def build_catalog(language='zh', stems=None, columns=4, thumb=(420, 300)):
    """Combine generated demo PNGs into a review-friendly contact sheet."""
    if language not in {'zh', 'en'}:
        raise ValueError("language 必须是 'zh' 或 'en'")
    suffix = f'_demo_{language}.png'
    paths = ([SAMPLES_DIR / f'{stem}_demo_{language}.png' for stem in stems]
             if stems else sorted(SAMPLES_DIR.glob(f'*{suffix}')))
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise ValueError(f'未找到 {language} 示例 PNG')
    cell_width, cell_height = thumb[0] + 24, thumb[1] + 42
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * cell_width, rows * cell_height), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = ImageOps.contain(source.convert('RGB'), thumb)
        x = (index % columns) * cell_width + (cell_width - image.width) // 2
        y = (index // columns) * cell_height + 24
        sheet.paste(image, (x, y))
        title = path.stem.removesuffix(f'_{language}').replace('_demo', '').replace('_', ' ')
        draw.text((index % columns * cell_width + 8, index // columns * cell_height + 6),
                  title, fill='#222222')
    output = SAMPLES_DIR / f'catalog_{language}.png'
    sheet.save(output, dpi=(300, 300))
    return output


if __name__ == '__main__':
    build_catalog('zh')
    build_catalog('en')
