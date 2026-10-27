"""A recipe reuses baseline parameters, never a curve from an earlier spectrum."""
import copy
import json

import numpy as np
import pytest

import spintegrate.core as core
from spintegrate.core import BaselineSettings, FitRegion, IntegrationRegion, Session
from test_pdata_direct import pdata_recipe, write_pdata_fixture


@pytest.mark.parametrize('method', ['asls', 'arpls', 'airpls'])
def test_pdata_direct_json_recalculates_each_region_in_each_spectrum(tmp_path, monkeypatch, method):
    size = 4096
    ppm = 12. - np.arange(size)*10./size
    signal = sum(40000.*np.exp(-.5*((ppm-center)/.035)**2) for center in (6., 8.))
    for number, offset, slope in [(1, 2000., 100.), (2, 17000., -1800.), (3, 80000., 2300.)]:
        values = np.round(signal + offset + slope*(ppm-7.))
        write_pdata_fixture(tmp_path/'series'/str(number)/'pdata'/'1', values, nc_proc=0)
    recipe = pdata_recipe()
    recipe.analysis_method = 'direct'
    baseline = BaselineSettings(method=method, lam=1e6, p=.001, tol=1e-6, max_iter=100)
    recipe.regions = [FitRegion('left', 5.2, 6.8, copy.deepcopy(baseline)),
                      FitRegion('right', 7.2, 8.8, copy.deepcopy(baseline))]
    recipe.integrals = [IntegrationRegion('a', 'left', 5.82, 6.18),
                        IntegrationRegion('b', 'right', 7.82, 8.18)]
    recipe.save(tmp_path/'recipe.json')
    saved = json.loads((tmp_path/'recipe.json').read_text(encoding='utf-8'))
    for region in saved['regions']:
        assert region['baseline'] == vars(baseline)
        assert not any(isinstance(v, (list, dict)) for v in region['baseline'].values())
    loaded = Session.load(tmp_path/'recipe.json')
    original = copy.deepcopy(loaded)
    actual_baseline = core.baseline_correct
    inputs = []
    def record(x, y, settings):
        inputs.append((x.copy(), y.copy(), copy.deepcopy(settings)))
        return actual_baseline(x, y, settings)
    monkeypatch.setattr(core, 'baseline_correct', record)
    rows, stopped = core.batch_analyze(tmp_path/'series', loaded, tmp_path/'out.csv')
    assert not stopped and len(rows) == 3
    assert loaded == original  # Running the series cannot turn parameters into fitted data.
    assert len(inputs) == 6  # Both local regions, independently, for all three pdata spectra.
    assert all(settings == baseline for _, _, settings in inputs)
    assert not np.allclose(inputs[0][1], inputs[2][1])
    curves = []
    expected_area = 40000.*.035*np.sqrt(2*np.pi)
    for row in rows:
        assert row['status'] != 'error'
        for name in ('a', 'b'):
            assert row[f'integral__{name}'] == pytest.approx(expected_area, rel=.02)
        analysis, restored_recipe = core.load_analysis(row['fit_file'])
        assert restored_recipe == loaded
        result = analysis.results['left']
        assert result.diagnostics['baseline_method'] == method
        assert result.diagnostics['baseline_parameters'] == vars(baseline)
        np.testing.assert_allclose(result.raw-result.baseline, result.corrected)
        curves.append(result.baseline)
    assert not np.allclose(curves[0], curves[1])
    assert not np.allclose(curves[1], curves[2])
    assert json.loads((tmp_path/'out.settings.json').read_text(encoding='utf-8')) == saved
