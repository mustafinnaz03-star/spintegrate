"""Presentation preferences independent of numerical NMR processing."""
from dataclasses import dataclass
import math
import re


AREA_COLORS = ('#56B4E9', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#0072B2')


@dataclass
class AppearanceSettings:
    language: str = 'ru'
    export_language: str = 'auto'
    export_dpi: int = 300
    export_width_mm: float = 180.
    export_height_mm: float = 140.
    export_format: str = 'png'
    export_layout: str = 'full'
    export_region_names: bool = False
    ui_scale_percent: int = 100


def validate_color(color):
    if color is not None and (not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', color)):
        raise ValueError('Цвет области должен быть задан как #RRGGBB.')


def area_color(area, index=0):
    return area.color or AREA_COLORS[index % len(AREA_COLORS)]


def validate_appearance(a):
    if type(a.ui_scale_percent) is not int or a.ui_scale_percent not in (80, 90, 100, 110, 125, 150):
        raise ValueError('Масштаб интерфейса: 80, 90, 100, 110, 125 или 150%.')
    if a.language not in ('ru', 'en') or a.export_language not in ('auto', 'ru', 'en'):
        raise ValueError('Язык должен быть ru или en.')
    if a.export_format not in ('png', 'tiff', 'svg') or a.export_layout not in ('full', 'fit', 'region'):
        raise ValueError('Неверный формат или компоновка рисунка.')
    if not isinstance(a.export_dpi, int) or not 72 <= a.export_dpi <= 2400:
        raise ValueError('DPI должен быть целым числом от 72 до 2400.')
    if not all(math.isfinite(v) and 40 <= v <= 500 for v in (a.export_width_mm, a.export_height_mm)):
        raise ValueError('Ширина и высота рисунка должны быть от 40 до 500 мм.')
    pixels = a.export_width_mm * a.export_height_mm * (a.export_dpi/25.4)**2
    if a.export_format != 'svg' and pixels > 80_000_000:
        raise ValueError('Рисунок превышает 80 мегапикселей. Уменьшите размер или DPI.')
