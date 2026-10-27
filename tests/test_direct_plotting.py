"""Scientific plot checks for direct numerical integration (no fitted components)."""
import re

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
import numpy as np
import pytest

from spintegrate.appearance import AppearanceSettings
from spintegrate.core import Analysis, BaselineSettings, FitRegion, FitResult, IntegrationRegion, Session, Spectrum
from spintegrate.figure_export import render_fit_figure
from spintegrate.plotting import draw_spectrum


def direct_example():
    x = np.linspace(0., 2., 21)
    y = x - .8  # includes both negative and positive signed areas
    baseline = np.full_like(x, 2.)
    spectrum = Spectrum(x, y+baseline, (y+baseline).astype(complex), 400., 800.,
                        'synthetic processed spectrum', (0., 0.), [])
    session = Session()
    session.analysis_method = 'direct'
    session.regions = [FitRegion('Контекст', 0., 2., BaselineSettings(method='none'))]
    session.integrals = [IntegrationRegion('Отрицательный', 'Контекст', .13, .57, '#CC79A7'),
                         IntegrationRegion('Положительный', 'Контекст', 1.07, 1.83, '#E69F00')]
    result = FitResult('Контекст', x, y+baseline, baseline, y, np.array([]), np.array([]),
                       np.array([]), np.array([]), y.copy(), True, [], {'method': 'direct'})
    analysis = Analysis(spectrum, {'Контекст': result}, {}, {}, 'ok', [])
    analysis.method = 'direct'
    return analysis, session


def axes():
    figure = Figure(figsize=(8, 7))
    overview, detail, residual = figure.subplots(3, 1)
    return figure, overview, detail, residual


def test_direct_plot_uses_exact_signed_interval_fill_and_cumulative_integrals(monkeypatch):
    analysis, session = direct_example()
    monkeypatch.setattr(analysis.results['Контекст'], 'components',
                        lambda: pytest.fail('Direct plotting must not generate Lorentzian components'))
    figure, overview, detail, cumulative = axes()
    draw_spectrum(overview, detail, cumulative, analysis.spectrum, session, 'Контекст', analysis)
    FigureCanvasAgg(figure).draw()
    fills = detail.collections
    assert len(fills) == 2
    assert to_hex(fills[0].get_facecolor()[0]) == '#cc79a7'
    assert to_hex(fills[1].get_facecolor()[0]) == '#e69f00'
    for fill, area in zip(fills, session.integrals):
        vertices = fill.get_paths()[0].vertices
        assert np.min(vertices[:, 0]) == pytest.approx(area.low)
        assert np.max(vertices[:, 0]) == pytest.approx(area.high)
    assert np.min(fills[0].get_paths()[0].vertices[:, 1]) < 0
    curves = [line for line in cumulative.lines if len(line.get_xdata()) > 2]
    assert len(curves) == 2
    for line, area in zip(curves, session.integrals):
        expected = .5*(area.high**2-area.low**2) - .8*(area.high-area.low)
        assert line.get_xdata()[0] == pytest.approx(area.low)
        assert line.get_xdata()[-1] == pytest.approx(area.high)
        assert line.get_ydata()[-1] == pytest.approx(expected)
    labels = [line.get_label() for line in detail.lines]
    assert not any('Lorentz' in label or 'Лоренц' in label or 'Threshold' in label for label in labels)
    assert 'пиков' not in detail.get_title(loc='left')
    assert 'Остаток' not in cumulative.get_title(loc='left')


@pytest.mark.parametrize('layout', ['full', 'fit', 'region'])
def test_direct_publication_export_uses_method_specific_english_labels(layout):
    analysis, session = direct_example()
    options = AppearanceSettings(export_language='en', export_layout=layout)
    figure = render_fit_figure(analysis, session, 'Контекст', options)
    FigureCanvasAgg(figure).draw()
    texts = []
    for ax in figure.axes:
        texts.extend([ax.get_title(loc='left'), ax.get_xlabel(), ax.get_ylabel()])
        if ax.get_legend():
            texts.extend(t.get_text() for t in ax.get_legend().get_texts())
        texts.extend(t.get_text() for t in ax.texts)
    assert not re.search('[А-Яа-яЁё]', '\n'.join(texts))
    assert any('Direct integration' in text for text in texts)
    assert not any('Lorentzian' in text or 'peaks;' in text or 'Residual:' in text for text in texts)


def test_direct_preview_hides_irrelevant_peak_markers_and_threshold():
    analysis, session = direct_example()
    result = analysis.results['Контекст']
    local = (result.x, result.raw, result.corrected, result.baseline,
             (np.array([1.]), np.array([.2])), [])
    _, overview, detail, cumulative = axes()
    draw_spectrum(overview, detail, cumulative, analysis.spectrum, session, 'Контекст', local=local)
    assert len(detail.collections) == 2  # fills only; no peak-picking scatter
    assert not any(line.get_label() == 'Threshold' for line in detail.lines)


def test_direct_plot_matches_actual_analysis_with_fractional_parent_boundaries():
    from spintegrate.core import analyze
    original, session = direct_example()
    x = original.spectrum.ppm
    y = x-.8
    spectrum = Spectrum(x, y, y.astype(complex), 400., 800., 'synthetic', (0., 0.), [])
    session.regions[0].low, session.regions[0].high = .03, 1.97
    analysis = analyze(spectrum, session)
    assert not analysis.errors
    assert analysis.results['Контекст'].x[0] == pytest.approx(.03)
    assert analysis.results['Контекст'].x[-1] == pytest.approx(1.97)
    _, overview, detail, cumulative = axes()
    draw_spectrum(overview, detail, cumulative, spectrum, session, 'Контекст', analysis)
    curves = [line for line in cumulative.lines if len(line.get_xdata()) > 2]
    for curve, area in zip(curves, session.integrals):
        assert curve.get_ydata()[-1] == pytest.approx(analysis.integrals[area.name])


def test_direct_publication_vertical_labels_fit_inside_canvas():
    analysis, session = direct_example()
    figure = render_fit_figure(analysis, session, 'Контекст', AppearanceSettings())
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    height = canvas.get_width_height()[1]
    for ax in figure.axes:
        if ax.get_ylabel():
            box = ax.yaxis.label.get_window_extent(canvas.get_renderer())
            assert box.y0 >= 0 and box.y1 <= height
