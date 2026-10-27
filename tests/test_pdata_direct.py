"""Independent processed-Bruker IO and direct-integration regression checks."""
import csv
import json
from dataclasses import asdict
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

import spintegrate.core as core
from spintegrate.core import (BaselineSettings, FitRegion, IntegrationRegion,
                             Session, Spectrum, analyze, batch_analyze,
                             load_analysis, process, save_analysis)
from spintegrate.demo import write_demo_series


def write_pdata_fixture(directory, values=None, imaginary=None, nc_proc=3,
                        offset=12., sf=400., sw=4000., phase=(123.25, -77.5)):
    """Use actual Bruker binary and JCAMP files, with a known storage scale."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if values is None:
        values = np.arange(128, dtype=float)*16 - 256
    values = np.asarray(values, dtype=float)
    dic = {'procs': {
        '_coreheader': ['##TITLE= Synthetic Spintegrate processed NMR', '##JCAMPDX= 5.00'],
        '_comments': [], 'SI': values.size, 'XDIM': values.size, 'PPARMOD': 0,
        'BYTORDP': 0, 'DTYPP': 0, 'NC_proc': nc_proc,
        'SF': sf, 'SW_p': sw, 'OFFSET': offset,
        'PHC0': phase[0], 'PHC1': phase[1], 'LB': 1.25,
    }}
    ng.bruker.write_pdata(str(directory), dic, values, scale_data=True,
                         write_procs=True, isfloat=False)
    if imaginary is not None:
        ng.bruker.write_pdata(str(directory), dic, np.asarray(imaginary),
                             scale_data=True, bin_file='1i', isfloat=False)
    ppm = offset - np.arange(values.size)*sw/(sf*values.size)
    return directory, values, ppm


def pdata_recipe():
    recipe = Session()
    recipe.processing.source_mode = 'pdata'
    recipe.processing.pdata_procno = '1'
    recipe.processing.lb_hz = 0.
    recipe.processing.phase = 'none'
    recipe.processing.reference_mode = 'metadata'
    return recipe


def direct_recipe(low=3., high=4.):
    recipe = Session()
    recipe.analysis_method = 'direct'
    recipe.regions = [FitRegion('parent', low, high, BaselineSettings(method='none'))]
    recipe.integrals = [IntegrationRegion('area', 'parent', low+.123, high-.137)]
    return recipe


def spectrum(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return Spectrum(x, y, y.astype(complex), 400., 4000., 'synthetic', (0., 0.), [])


@pytest.mark.parametrize('nc_proc', [-2, 0, 3])
def test_pdata_exact_topspin_axis_scale_and_imported_phase(tmp_path, nc_proc):
    directory, physical, descending_ppm = write_pdata_fixture(tmp_path/'37'/'pdata'/'1', nc_proc=nc_proc)
    source = core.load_pdata(directory)
    assert source.source_type == 'pdata'
    assert source.sf_mhz == 400.
    assert source.sw_hz == 4000.
    np.testing.assert_array_equal(source.ppm, descending_ppm[::-1])
    np.testing.assert_array_equal(source.intensity, physical[::-1])
    np.testing.assert_array_equal(source.complex_spectrum.real, physical[::-1])
    assert source.processing_metadata['has_imag'] is False
    procs = source.processing_metadata['procs']
    assert procs['PHC0'] == 123.25 and procs['PHC1'] == -77.5
    assert procs['NC_proc'] == nc_proc
    # Loading a real-only processed spectrum must not run FFT, rephase or re-reference it.
    result = process(source, pdata_recipe().processing)
    np.testing.assert_array_equal(result.intensity, source.intensity)
    np.testing.assert_array_equal(result.ppm, source.ppm)
    np.testing.assert_array_equal(source.intensity, physical[::-1])


def test_pdata_reference_uses_processed_parameters_even_if_acquisition_differs(tmp_path):
    directory, physical, ppm = write_pdata_fixture(tmp_path/'37'/'pdata'/'1', sf=400., sw=4000., offset=12.)
    # SFO1/O1 refer to the acquisition, not the saved TopSpin processed reference.
    acq = {'_coreheader': ['##TITLE= Synthetic acquisition'], '_comments': [],
           'SFO1': 499.9, 'SW_h': 7777., 'O1': 2000., 'AQ_mod': 3}
    ng.bruker.write_jcamp(acq, str(directory.parent.parent/'acqus'))
    loaded = core.load_pdata(directory)
    assert loaded.sf_mhz == 400.
    np.testing.assert_array_equal(loaded.ppm, ppm[::-1])
    np.testing.assert_array_equal(loaded.intensity, physical[::-1])


@pytest.mark.parametrize('selection', ['experiment', 'pdata', 'procno', '1r'])
def test_pdata_loader_accepts_topspin_selection_levels(tmp_path, selection):
    directory, expected, ppm = write_pdata_fixture(tmp_path/'37'/'pdata'/'1')
    paths = {'experiment': directory.parent.parent, 'pdata': directory.parent,
             'procno': directory, '1r': directory/'1r'}
    actual = core.load_pdata(paths[selection])
    np.testing.assert_array_equal(actual.intensity, expected[::-1])
    np.testing.assert_array_equal(actual.ppm, ppm[::-1])


def test_pdata_complex_component_and_magnitude_use_processed_values(tmp_path):
    imag = np.arange(128, dtype=float)*8+32
    directory, real, _ = write_pdata_fixture(tmp_path/'pdata'/'1', imaginary=imag)
    loaded = core.load_pdata(directory)
    assert loaded.processing_metadata['has_imag'] is True
    np.testing.assert_array_equal(loaded.complex_spectrum, (real+1j*imag)[::-1])
    recipe = pdata_recipe()
    recipe.processing.phase = 'abs'
    recipe.processing.abs_source = 'spectrum'
    processed = process(loaded, recipe.processing)
    np.testing.assert_allclose(processed.intensity, np.hypot(real, imag)[::-1])
    np.testing.assert_array_equal(processed.ppm, loaded.ppm)


@pytest.mark.parametrize('change', [
    {'lb_hz': 1.}, {'phase': 'acme'}, {'phase': 'peak_minima'},
    {'phase': 'abs', 'abs_source': 'spectrum'},
    {'phase': 'abs', 'abs_source': 'fid'}, {'zero_fill_factor': 2},
])
def test_real_only_pdata_does_not_invent_imaginary_information(tmp_path, change):
    directory, _, _ = write_pdata_fixture(tmp_path/'pdata'/'1')
    source = core.load_pdata(directory)
    recipe = pdata_recipe()
    for key, value in change.items():
        setattr(recipe.processing, key, value)
    with pytest.raises(ValueError, match='1i'):
        process(source, recipe.processing)


def test_complex_pdata_allows_extra_apodization_without_mutating_import(tmp_path):
    size, sf, sw, offset = 2048, 400., 4000., 12.
    ppm = offset-np.arange(size)*sw/(sf*size)
    d = ppm-7.5
    ft = 800000.*.015/(.015+1j*d)
    # Integer storage quantization is deliberate and separate from exact preservation checks.
    directory, _, _ = write_pdata_fixture(tmp_path/'pdata'/'1',
                         values=np.round(ft.real), imaginary=np.round(ft.imag), nc_proc=0,
                         offset=offset, sf=sf, sw=sw)
    source = core.load_pdata(directory)
    original = source.complex_spectrum.copy()
    recipe = pdata_recipe()
    recipe.processing.lb_hz = 2.
    processed = process(source, recipe.processing)
    assert processed.intensity.max() < source.intensity.max()
    np.testing.assert_array_equal(source.complex_spectrum, original)
    np.testing.assert_array_equal(processed.ppm, source.ppm)


def test_direct_integral_interpolates_both_exact_boundaries_and_skips_fitting(monkeypatch):
    x = np.linspace(3., 4., 101)
    y = 2*x-7.5  # signed signal whose exact antiderivative is x**2-7.5*x
    recipe = direct_recipe()
    recipe.fit.threshold = -123.  # not a meaningful setting for direct integration
    recipe.fit.min_width_hz = -1.
    recipe.fit.max_width_hz = -2.
    for name in ('pick_peaks', 'least_squares', 'fit_region'):
        monkeypatch.setattr(core, name, lambda *_a, **_k: pytest.fail('Direct mode must not call peak picking or fitting'))
    analysis = analyze(spectrum(x, y), recipe)
    low, high = recipe.integrals[0].low, recipe.integrals[0].high
    expected = (high**2-7.5*high)-(low**2-7.5*low)
    assert analysis.method == 'direct'
    assert analysis.integrals['area'] == pytest.approx(expected, abs=1e-12)
    assert analysis.integrals['area'] < 0  # retain the sign, as in signed region integration
    result = analysis.results['parent']
    assert result.centers.size == result.amplitudes.size == result.widths.size == 0
    np.testing.assert_array_equal(result.corrected, y)


def test_direct_helper_is_orientation_independent_and_rejects_outside_bounds():
    x = np.array([0., .25, .5, .75, 1.])
    y = 3*x+2
    expected = (1.5*.91**2+2*.91)-(1.5*.13**2+2*.13)
    assert core.integrate_region(x, y, .13, .91) == pytest.approx(expected)
    assert core.integrate_region(x[::-1], y[::-1], .13, .91) == pytest.approx(expected)
    with pytest.raises(ValueError):
        core.integrate_region(x, y, -.01, .5)
    with pytest.raises(ValueError):
        core.integrate_region(x, y, .5, 1.01)


def test_direct_full_parent_region_integrates_non_grid_parent_boundaries():
    x = np.linspace(3., 4., 101)
    recipe = direct_recipe(3.005, 3.995)
    recipe.integrals = [IntegrationRegion('whole_parent', 'parent', 3.005, 3.995)]
    analysis = analyze(spectrum(x, 3*x+2), recipe)
    assert analysis.status != 'error'
    expected = (1.5*3.995**2+2*3.995)-(1.5*3.005**2+2*3.005)
    assert analysis.integrals['whole_parent'] == pytest.approx(expected)


def test_direct_parent_baseline_is_local_and_integrates_corrected_data(monkeypatch):
    x = np.linspace(1., 8., 701)
    y = 5+2*x
    recipe = direct_recipe(3., 4.)
    recipe.regions[0].baseline.method = 'asls'
    calls = []
    def local_baseline(xx, yy, settings):
        calls.append((xx.copy(), yy.copy()))
        baseline = np.full_like(yy, 5.)
        return yy-baseline, baseline, []
    monkeypatch.setattr(core, 'baseline_correct', local_baseline)
    original = y.copy()
    analysis = analyze(spectrum(x, y), recipe)
    assert len(calls) == 1
    assert calls[0][0][0] >= 3. and calls[0][0][-1] <= 4.
    lo, hi = recipe.integrals[0].low, recipe.integrals[0].high
    assert analysis.integrals['area'] == pytest.approx(hi**2-lo**2)
    np.testing.assert_array_equal(original, y)


def test_direct_mode_is_saved_in_recipe_and_spfit_without_refitting(tmp_path, monkeypatch):
    recipe = direct_recipe()
    recipe.processing.source_mode = 'pdata'
    recipe.processing.pdata_procno = '2'
    recipe.processing.lb_hz = 0.
    source = spectrum(np.linspace(3., 4., 101), np.linspace(5., 7., 101))
    source.source_type = 'pdata'
    source.processing_metadata = {'has_imag': False, 'procno': '2', 'procs': {'PHC0': 123.25}}
    analysis = analyze(source, recipe)
    recipe.save(tmp_path/'settings.json')
    assert Session.load(tmp_path/'settings.json') == recipe
    save_analysis(tmp_path/'direct.spfit', analysis, recipe)
    monkeypatch.setattr(core, 'analyze', lambda *_: pytest.fail('Loading must not recalculate'))
    restored, saved = load_analysis(tmp_path/'direct.spfit')
    assert saved == recipe
    assert restored.method == 'direct'
    assert restored.spectrum.source_type == 'pdata'
    assert restored.spectrum.processing_metadata == source.processing_metadata
    assert restored.integrals == analysis.integrals


def test_legacy_recipe_defaults_to_raw_lorentzian(tmp_path):
    original = asdict(Session())
    original.pop('analysis_method', None)
    original['processing'].pop('source_mode', None)
    original['processing'].pop('pdata_procno', None)
    path = tmp_path/'legacy.json'
    path.write_text(json.dumps(original), encoding='utf-8')
    restored = Session.load(path)
    assert restored.analysis_method == 'lorentzian'
    assert restored.processing.source_mode == 'raw'


def test_pdata_discovery_selects_procno_and_uses_experiment_numbers_in_batch(tmp_path):
    for expno, level in [('10', 48.), ('2', 24.)]:
        for procno in ('1', '2'):
            amplitude = level if procno == '2' else level*10
            write_pdata_fixture(tmp_path/expno/'pdata'/procno, np.full(128, amplitude), nc_proc=3)
    recipe = pdata_recipe()
    recipe.processing.pdata_procno = '2'
    recipe.analysis_method = 'direct'
    recipe.regions = [FitRegion('parent', 3., 10., BaselineSettings(method='none'))]
    recipe.integrals = [IntegrationRegion('signal', 'parent', 4.123, 5.789)]
    found = core.discover_spectra(tmp_path, source_mode='pdata', pdata_procno='2')
    assert [p.relative_to(tmp_path).as_posix() for p in found] == ['2/pdata/2', '10/pdata/2']
    rows, stopped = batch_analyze(tmp_path, recipe, tmp_path/'out.csv')
    assert not stopped
    assert [r['spectrum_number'] for r in rows] == ['2', '10']
    assert len(rows) == 2
    for row, amplitude in zip(rows, [24., 48.]):
        assert row['status'] != 'error'
        assert row['integral__signal'] == pytest.approx(amplitude*(5.789-4.123))
        restored, saved = load_analysis(row['fit_file'])
        assert restored.method == 'direct'
        assert restored.spectrum.source_type == 'pdata'
        assert saved.processing.pdata_procno == '2'
    with (tmp_path/'out.csv').open(encoding='utf-8-sig') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert list(csv_rows[0]) == ['spectrum_number', 'spectrum_id', 'signal']
    assert float(csv_rows[0]['signal']) == pytest.approx(24.*(5.789-4.123))
    assert len(core.discover_spectra(tmp_path, source_mode='pdata', pdata_procno='2')) == 2


def test_raw_direct_batch_reuses_saved_regions_and_preserves_scaling(tmp_path):
    write_demo_series(tmp_path)
    recipe = direct_recipe(5., 5.55)
    recipe.processing.lb_hz = 0.
    recipe.integrals = [IntegrationRegion('signal', 'parent', 5.15, 5.35)]
    rows, _ = batch_analyze(tmp_path, recipe, tmp_path/'raw_direct.csv')
    assert len(rows) == 4
    assert all(r['status'] != 'error' for r in rows)
    # Synthetic fixed baseline is small relative to signals; compare near expected scaling.
    assert rows[3]['integral__signal']/rows[0]['integral__signal'] == pytest.approx(.4, abs=.04)
