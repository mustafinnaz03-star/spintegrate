import numpy as np
import pytest
from matplotlib.figure import Figure

from spintegrate.core import analyze, process
from spintegrate.demo import demo_raw, demo_session
from spintegrate.plotting import draw_spectrum


@pytest.fixture
def automatic_result():
    session = demo_session()
    session.appearance.language = 'en'
    analysis = analyze(process(demo_raw(), session.processing), session)
    name = session.regions[0].name
    result = analysis.results[name]
    assert len(result.centers) > 0
    stale_preview = (result.x, result.raw, result.corrected, result.baseline,
                     (result.centers, result.initial_amplitudes), [])
    return session, analysis, name, stale_preview


@pytest.mark.parametrize('marks', [[], [3.544], [3.544, 3.595]])
def test_manual_preview_draws_only_saved_marks_even_with_old_automatic_fit_and_preview(automatic_result, marks):
    session, analysis, name, stale_preview = automatic_result
    session.analysis_method = 'lorentzian_manual'
    session.regions[0].manual_peaks = marks
    axes = Figure().subplots(3, 1)
    draw_spectrum(*axes, analysis.spectrum, session, name, analysis=analysis, local=stale_preview)
    detail, residual = axes[1:]
    labels = detail.get_legend_handles_labels()[1]
    assert 'Lorentzian sum' not in labels
    assert 'Peak-picked positions (fixed)' not in labels
    assert 'Optimized centers' not in labels
    markers = next(artist for artist in detail.collections if artist.get_label() == 'Manual marks (initial)')
    np.testing.assert_allclose(np.asarray(markers.get_offsets())[:, 0], marks)
    assert len(residual.lines) == 1  # the zero line, without stale fitted residuals


def test_stale_automatic_analysis_is_not_rendered_when_manual_mode_has_no_preview(automatic_result):
    session, analysis, name, _ = automatic_result
    session.analysis_method = 'lorentzian_manual'
    axes = Figure().subplots(3, 1)
    draw_spectrum(*axes, analysis.spectrum, session, name, analysis=analysis)
    assert not axes[1].collections
    assert axes[1].get_legend_handles_labels()[1] == ['Spectrum']
