import copy
import json
from dataclasses import asdict

import nmrglue as ng
import numpy as np
import pytest

from spintegrate import core
from spintegrate.core import (BaselineSettings, FitRegion, IntegrationRegion, Session,
                              Spectrum, analyze, batch_analyze, load_analysis, lorentzian)
from spintegrate.region_editing import RegionHistory, move_or_resize
from test_pdata_direct import write_pdata_fixture


def settings():
    s = Session(analysis_method='lorentzian_manual')
    s.processing.lb_hz = 0.
    s.fit.restarts = 2
    s.fit.threshold = 1e20  # manual mode must not use automatic peak picking
    s.regions = [FitRegion('fit', 3., 4., BaselineSettings(method='none'), [3.49, 3.605])]
    s.integrals = [IntegrationRegion('first', 'fit', 3.495, 3.505),
                   IntegrationRegion('all', 'fit', 3.4, 3.7)]
    return s


def spectrum(x, y):
    return Spectrum(x, y, y.astype(complex), 400., 2400., 'synthetic', (0., 0.), [])


@pytest.mark.parametrize('signed', [False, True])
@pytest.mark.parametrize('loss', ['linear', 'soft_l1'])
def test_manual_centers_refine_without_peak_picking_and_final_centers_determine_integrals(monkeypatch, signed, loss):
    x = np.linspace(2., 8., 12001)
    amplitudes = [12000., -6500. if signed else 6500.]
    y = lorentzian(x, 3.5, .006, amplitudes[0])+lorentzian(x, 3.6, .009, amplitudes[1])
    source, s = spectrum(x, y), settings()
    s.fit.signed = signed
    s.fit.loss = loss
    original = copy.deepcopy(s)
    monkeypatch.setattr(core, 'pick_peaks', lambda *_: pytest.fail('Manual mode called peak picking'))
    result = analyze(source, s)
    fit = result.results['fit']
    assert not result.errors and fit.success
    assert s == original
    np.testing.assert_allclose(fit.centers, [3.5, 3.6], atol=1e-7)
    np.testing.assert_allclose(fit.amplitudes, amplitudes, rtol=1e-5)
    np.testing.assert_allclose(fit.widths, [.006, .009], rtol=1e-5)
    np.testing.assert_allclose(fit.initial_amplitudes, np.interp([3.49, 3.605], x, y))
    assert result.integrals['first'] == pytest.approx(72., rel=1e-5)  # initial 3.49 is OUTSIDE
    assert result.integrals['all'] == pytest.approx(sum(np.array(amplitudes)*[.006, .009]), rel=1e-5)
    assert fit.diagnostics['initial_centers_ppm'] == [3.49, 3.605]
    assert fit.diagnostics['method'] == 'lorentzian_manual'


def test_center_bounds_are_enforced_and_reported():
    x = np.linspace(3., 4., 2001)
    s = settings()
    s.regions[0].manual_peaks = [3.5]
    s.fit.center_tolerance_ppm = .01
    s.fit.initial_width_hz = 8.
    result = analyze(spectrum(x, lorentzian(x, 3.52, .008, 5000.)), s)
    fit = result.results['fit']
    assert fit.centers[0] <= 3.51+1e-12
    assert any('Центр ручного пика достиг' in note for note in fit.notes)
    assert fit.diagnostics['center_bounds_ppm'] == [[3.49, 3.51]]


@pytest.mark.parametrize('peaks', [[3.5, 3.5], [3.], [4.1], [float('nan')], [[3.5]]])
def test_invalid_manual_marks_are_rejected(peaks):
    s = settings()
    s.regions[0].manual_peaks = peaks
    with pytest.raises(ValueError, match='ручные пики'):
        s.validate()


def test_missing_marks_are_allowed_while_editing_but_block_analysis_and_batch(tmp_path):
    s = settings()
    s.regions[0].manual_peaks = []
    s.validate()
    with pytest.raises(ValueError, match='отметьте'):
        s.validate(require_regions=True)
    with pytest.raises(ValueError, match='отметьте'):
        batch_analyze(tmp_path, s, tmp_path/'out.csv')
    assert not (tmp_path/'out.csv').exists()


def test_old_json_defaults_and_new_manual_recipe_roundtrip(tmp_path):
    s = settings()
    s.save(tmp_path/'manual.json')
    assert Session.load(tmp_path/'manual.json') == s
    legacy = asdict(s)
    legacy['analysis_method'] = 'lorentzian'
    legacy['fit'].pop('center_tolerance_ppm')
    legacy['regions'][0].pop('manual_peaks')
    restored = Session.from_dict(legacy)
    assert restored.regions[0].manual_peaks == []
    assert restored.fit.center_tolerance_ppm == .03


def test_region_geometry_translates_and_protects_manual_marks_with_undo():
    s = settings()
    history = RegionHistory()
    history.record(s)
    moved = move_or_resize(s, 'fit', 'fit', 4., 5., (0., 10.))
    assert moved.regions[0].manual_peaks == pytest.approx([4.49, 4.605])
    restored = history.undo(moved)
    assert restored.regions[0].manual_peaks == pytest.approx(s.regions[0].manual_peaks)
    narrowed = move_or_resize(s, 'fit', 'fit', 3.5, 3.55, (0., 10.))
    assert narrowed.regions[0].low < 3.49 < 3.605 < narrowed.regions[0].high


def test_manual_mode_recomputes_baseline_and_initial_heights_per_spectrum():
    x = np.linspace(2., 8., 12001)
    signal = lorentzian(x, 3.5, .006, 12000.)+lorentzian(x, 3.6, .009, 6500.)
    s = settings()
    s.regions[0].baseline = BaselineSettings(method='asls', p=.001, tol=1e-6)
    first = analyze(spectrum(x, signal+500.+30.*x), s)
    second = analyze(spectrum(x, .6*signal+1500.-60.*x), s)
    a, b = first.results['fit'], second.results['fit']
    assert not first.errors and not second.errors
    assert not np.allclose(a.baseline, b.baseline)
    np.testing.assert_allclose(b.initial_amplitudes, np.interp([3.49, 3.605], b.x, b.corrected))
    assert second.integrals['all']/first.integrals['all'] == pytest.approx(.6, rel=.005)
    assert set(asdict(s.regions[0])) == {'name', 'low', 'high', 'baseline', 'manual_peaks'}


@pytest.mark.parametrize('source_mode', ['raw', 'pdata'])
@pytest.mark.parametrize('mode', ['global', 'regions'])
def test_manual_batch_forces_alignment_refines_residual_drift_and_saves_actual_recipe(tmp_path, source_mode, mode):
    directory = tmp_path/'series'
    directory.mkdir()
    x = 2.+np.arange(8192)*6./8192
    for number, drift, gain in [(1, 0., 1.), (2, .042, .6)]:
        y = gain*(lorentzian(x, 3.5+drift, .006, 12000.)+
                  lorentzian(x, 3.6+drift+.004*(number-1), .009, 6500.))
        if source_mode == 'raw':
            np.savez(directory/f'{number}.npz', fid=ng.proc_base.ifft(y.astype(complex)),
                     sf_mhz=400., sw_hz=2400., carrier_ppm=5.)
        else:
            write_pdata_fixture(directory/str(number)/'pdata'/'1', y[::-1]*10000.,
                                nc_proc=0, offset=float(x[-1]), sf=400., sw=2400.)
    s = settings()
    s.processing.source_mode = source_mode
    s.alignment.mode = mode
    assert not s.alignment.enabled
    rows, stopped = batch_analyze(directory, s, tmp_path/'out.csv')
    assert not stopped and not s.alignment.enabled  # no mutation of caller's recipe
    saved = Session.load(tmp_path/'out.settings.json')
    assert saved.alignment.enabled and saved.alignment.mode == mode
    assert all(row['status'] != 'error' for row in rows), rows
    for row in rows:
        fitted, stored = load_analysis(row['fit_file'])
        assert fitted.method == 'lorentzian_manual' and stored == saved
        assert len(fitted.results['fit'].centers) == 2
        assert fitted.results['fit'].diagnostics['initial_centers_ppm'] == [3.49, 3.605]
    assert rows[1]['integral__all']/rows[0]['integral__all'] == pytest.approx(.6, rel=.003)
    report = json.loads((tmp_path/'out.diagnostics.json').read_text(encoding='utf-8'))
    assert all('alignment' in item for item in report['spectra'])
    assert report['spectra'][1]['diagnostics']['fit']['center_shifts_ppm']
