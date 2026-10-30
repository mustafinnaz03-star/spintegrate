import copy
import json
from dataclasses import replace

import nmrglue as ng
import numpy as np
import pytest

from spintegrate.alignment import AlignmentSettings, align_regions, align_spectrum, estimate_shift
from spintegrate.core import (BaselineSettings, FitRegion, IntegrationRegion, Session, Spectrum,
                              analyze, batch_analyze, load_analysis)
from test_pdata_direct import write_pdata_fixture


def trace(x, left=0., right=0., gain=1., offset=20., slope=3.):
    return offset+slope*x+gain*(500.*np.exp(-.5*((x-3.5-left)/.009)**2)+300.*np.exp(-.5*((x-5.3-right)/.012)**2))


def spectrum(x, y):
    return Spectrum(x, y, y.astype(complex), 400., 2400., 'synthetic', (0., 0.), [])


def recipe(mode='global'):
    s = Session(analysis_method='direct', alignment=AlignmentSettings(enabled=True, mode=mode))
    s.processing.lb_hz = 0.
    s.regions = [FitRegion('left', 3.2, 3.8, BaselineSettings(method='asls', p=.001, tol=1e-6)),
                 FitRegion('right', 5., 5.6, BaselineSettings(method='asls', p=.001, tol=1e-6))]
    s.integrals = [IntegrationRegion('a', 'left', 3.45, 3.55), IntegrationRegion('b', 'right', 5.25, 5.35)]
    return s


@pytest.mark.parametrize('drift', [-.043, -.0003, 0., .036])
def test_global_shift_sign_fractional_accuracy_and_no_intensity_resampling(drift):
    x = np.linspace(2., 8., 12001)
    reference = spectrum(x, trace(x))
    y = trace(x, drift, drift, gain=.4, offset=500., slope=-20.)
    y += np.random.default_rng(12).normal(0, .1, len(x))
    source = spectrum(x, y)
    original_x, original_y = source.ppm.copy(), source.intensity.copy()
    aligned, info = align_spectrum(reference, source, AlignmentSettings(), 'first')
    assert info['shift_ppm'] == pytest.approx(-drift, abs=5e-5)
    assert info['correlation'] > .95
    np.testing.assert_array_equal(aligned.intensity, original_y)
    np.testing.assert_array_equal(aligned.complex_spectrum, source.complex_spectrum)
    np.testing.assert_array_equal(source.ppm, original_x)
    assert aligned.processing_metadata['alignment_reference'] == 'first'


def test_regions_align_opposite_drifts_and_preserve_quantitative_areas():
    x = np.linspace(2., 8., 12001)
    ref = spectrum(x, trace(x))
    source = spectrum(x, trace(x, .034, -.026))
    s = recipe('regions')
    unchanged = copy.deepcopy(s)
    context, views, errors, info = align_regions(ref, source, s, 'first')
    assert not errors
    assert info['left']['shift_ppm'] == pytest.approx(-.034, abs=1e-5)
    assert info['right']['shift_ppm'] == pytest.approx(.026, abs=1e-5)
    result = analyze(context, s, views, errors)
    expected = analyze(ref, s)
    assert not result.errors and s == unchanged
    for name in ('a', 'b'):
        assert result.integrals[name] == pytest.approx(expected.integrals[name], rel=.002)
    np.testing.assert_array_equal(context.intensity, source.intensity)
    assert result.results['left'].diagnostics['alignment']['shift_ppm'] == info['left']['shift_ppm']


def test_regional_alignment_also_precedes_lorentzian_peak_picking_and_fitting():
    x = np.linspace(2., 8., 12001)
    def lines(left, right):
        return 20.+3*x+500.*.006**2/((x-3.5-left)**2+.006**2)+300.*.008**2/((x-5.3-right)**2+.008**2)
    reference = spectrum(x, lines(0., 0.))
    source = spectrum(x, lines(.031, -.024))
    s = recipe('regions')
    s.analysis_method = 'lorentzian'
    s.fit.threshold, s.fit.restarts = 30., 1
    aligned, views, errors, _ = align_regions(reference, source, s, 'first')
    result = analyze(aligned, s, views, errors)
    expected = analyze(reference, s)
    assert not result.errors and not expected.errors
    assert result.results['left'].centers[0] == pytest.approx(3.5, abs=.001)
    assert result.results['right'].centers[0] == pytest.approx(5.3, abs=.001)
    for name in ('a', 'b'):
        assert result.integrals[name] == pytest.approx(expected.integrals[name], rel=.005)


def test_flat_region_alignment_failure_does_not_discard_good_regions():
    x = np.linspace(2., 8., 12001)
    ref = spectrum(x, trace(x))
    y = trace(x, .02, -.02)
    y[(x > 4.8) & (x < 5.8)] = 20.
    source = spectrum(x, y)
    s = recipe('regions')
    context, views, errors, _ = align_regions(ref, source, s, 'first')
    assert 'right' in errors and 'left' not in errors
    result = analyze(context, s, views, errors)
    assert np.isfinite(result.integrals['a'])
    assert np.isnan(result.integrals['b'])


@pytest.mark.parametrize('mode', ['global', 'regions'])
@pytest.mark.parametrize('source_mode', ['raw', 'pdata'])
def test_batch_first_reference_json_snapshot_and_alignment_diagnostics(tmp_path, mode, source_mode):
    x = 2.+np.arange(8192)*6./8192
    directory = tmp_path/'series'
    directory.mkdir()
    for number, left, right in [(1, 0., 0.), (2, .031, .031 if mode == 'global' else -.024),
                                (10, -.018, -.018 if mode == 'global' else .022)]:
        y = trace(x, left, right, gain=1.-number*.025)
        if source_mode == 'raw':
            np.savez(directory/f'{number}.npz', fid=ng.proc_base.ifft(y.astype(complex)),
                     sf_mhz=400., sw_hz=2400., carrier_ppm=5.)
        else:
            write_pdata_fixture(directory/str(number)/'pdata'/'1', y[::-1]*10000.,
                                    nc_proc=0, offset=float(x[-1]), sf=400., sw=2400.)
    s = recipe(mode)
    s.processing.source_mode = source_mode
    s.save(tmp_path/'recipe.json')
    loaded = Session.load(tmp_path/'recipe.json')
    assert loaded == s
    rows, stopped = batch_analyze(directory, loaded, tmp_path/'out.csv')
    assert not stopped and [row['spectrum_number'] for row in rows] == ['1', '2', '10']
    reference_id = rows[0]['spectrum_id']
    assert all(row['alignment_reference'] == reference_id for row in rows)
    assert all(row['status'] != 'error' for row in rows), rows
    for index, row in enumerate(rows):
        analysis, saved = load_analysis(row['fit_file'])
        assert saved == s and analysis.spectrum.processing_metadata['alignment_mode'] == mode
        expected_gain = [1., .95/.975, .75/.975][index]
        assert row['integral__a']/rows[0]['integral__a'] == pytest.approx(expected_gain, rel=.005)
        assert row['integral__b']/rows[0]['integral__b'] == pytest.approx(expected_gain, rel=.005)
    report = json.loads((tmp_path/'out.diagnostics.json').read_text(encoding='utf-8'))
    assert all('alignment' in item for item in report['spectra'])
    if mode == 'global':
        assert rows[0]['alignment_shift_ppm'] == 0.
        assert rows[1]['alignment_shift_ppm'] == pytest.approx(-.031, abs=.0001)
    else:
        assert all(info['shift_ppm'] == 0. for info in report['spectra'][0]['alignment'].values())


def test_first_reference_failure_is_explicit_and_does_not_write_outputs(tmp_path):
    directory = tmp_path/'series'
    directory.mkdir()
    np.savez(directory/'1.npz', fid=np.zeros(1024, dtype=complex), sf_mhz=400., sw_hz=2400., carrier_ppm=5.)
    with pytest.raises(ValueError, match='first spectrum'):
        batch_analyze(directory, recipe(), tmp_path/'out.csv')
    assert not (tmp_path/'out.csv').exists()
    assert not (tmp_path/'out.settings.json').exists()


def test_alignment_handles_different_grids_and_bounds():
    x = np.linspace(2., 8., 12001)
    other = np.linspace(2.01, 7.98, 8500)
    info = estimate_shift(spectrum(x, trace(x)), spectrum(other, trace(other, .027, .027)), AlignmentSettings())
    assert info['shift_ppm'] == pytest.approx(-.027, abs=.0001)
    with pytest.raises(ValueError):
        estimate_shift(spectrum(x, trace(x)), spectrum(other, trace(other)), AlignmentSettings(low_ppm=3., high_ppm=3.0001))
