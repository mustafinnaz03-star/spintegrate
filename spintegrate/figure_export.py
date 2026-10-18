"""Publication figure rendering without modifying the live GUI or fitted values."""
from pathlib import Path

from matplotlib.figure import Figure

from .appearance import validate_appearance
from .plotting import draw_spectrum


def render_fit_figure(analysis, session, region, options, interface_language='ru', title=''):
    validate_appearance(options)
    language = interface_language if options.export_language == 'auto' else options.export_language
    figure = Figure(figsize=(options.export_width_mm/25.4, options.export_height_mm/25.4),
                    dpi=100, facecolor='white', layout='constrained')
    if options.export_layout == 'full':
        grid = figure.add_gridspec(3, 1, height_ratios=(1, 2, .65))
        overview = figure.add_subplot(grid[0])
        detail = figure.add_subplot(grid[1])
        residual = figure.add_subplot(grid[2], sharex=detail)
    elif options.export_layout == 'fit':
        grid = figure.add_gridspec(2, 1, height_ratios=(3, 1))
        detail = figure.add_subplot(grid[0])
        residual = figure.add_subplot(grid[1], sharex=detail)
        overview = figure.add_axes((0, 0, .001, .001), visible=False)
    else:
        detail = figure.add_subplot(111)
        residual = figure.add_axes((0, 0, .001, .001), visible=False)
        overview = figure.add_axes((0, 0, .001, .001), visible=False)
    draw_spectrum(overview, detail, residual, analysis.spectrum, session, region, analysis,
                  component_limit=None, language=language, show_region_names=options.export_region_names)
    if title.strip():
        detail.set_title(title.strip(), fontsize=10, loc='left')
    # Explicit point sizes give the same physical typography at every DPI.
    for ax in figure.axes:
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.title.set_size(9)
    return figure


def save_figure(path, figure, options):
    validate_appearance(options)
    suffix = Path(path).suffix.lower()
    expected = '.tiff' if options.export_format == 'tiff' else '.'+options.export_format
    if suffix not in ({'.tif', '.tiff'} if options.export_format == 'tiff' else {expected}):
        raise ValueError(f'Expected {expected} file extension.')
    # Do not use bbox_inches="tight": it changes the requested physical size.
    kwargs = {'format': options.export_format, 'dpi': options.export_dpi, 'facecolor': 'white'}
    if options.export_format == 'tiff':
        kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
    figure.savefig(path, **kwargs)
