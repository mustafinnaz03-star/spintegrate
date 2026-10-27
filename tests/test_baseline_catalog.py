"""Complete method coverage, safe parameter recipes and per-spectrum evaluation."""
import copy
import inspect
import json
from collections import Counter

import numpy as np
import pytest
from pybaselines import Baseline

import spintegrate.core as core
from spintegrate.baseline_methods import CATALOG, FAMILIES, optimizer_methods
from spintegrate.baseline_ui import parse_parameter
from spintegrate.demo import write_demo_series, write_pdata_demo_series


def test_catalog_covers_every_public_1d_method_and_official_family():
    methods = {name: function.__module__.split('.')[-1] for name, function in inspect.getmembers(Baseline, inspect.isfunction)
               if not name.startswith('_') and function.__module__.split('.')[-1] in FAMILIES}
    assert CATALOG == methods
    assert len(CATALOG) == 62
    assert Counter(CATALOG.values()) == dict(whittaker=11, morphological=10, polynomial=7,
        spline=15, smooth=6, classification=7, optimizers=4, misc=2)
    assert len(core.BASELINE_METHODS) == 63


def test_optimizer_choices_are_limited_to_compatible_inner_methods():
    assert 'asls' in optimizer_methods('collab_pls')
    assert 'mor' not in optimizer_methods('collab_pls')
    assert 'corner_cutting' not in optimizer_methods('collab_pls')
    assert all(CATALOG[m] == 'polynomial' for m in optimizer_methods('adaptive_minmax'))
    with pytest.raises(ValueError):
        core.validate_baseline(core.BaselineSettings(method='adaptive_minmax', parameters={'method': 'asls'}))


@pytest.mark.parametrize('method, parameters', [
    ('rolling_ball', {'half_window': 17, 'smooth_half_window': 2}),
    ('pspline_arpls', {'lam': 1500., 'num_knots': 25}),
    ('adaptive_minmax', {'poly_order': [1, 2], 'method': 'imodpoly', 'method_kwargs': {'max_iter': 30}}),
    ('collab_pls', {'method': 'asls', 'method_kwargs': {'lam': 1e5, 'p': .001}}),
    ('interp_pts', {'anchor_ppm': []}),
])
@pytest.mark.parametrize('mode', ['raw', 'pdata'])
def test_extended_recipe_runs_fresh_on_raw_and_pdata_series(tmp_path, monkeypatch, method, parameters, mode):
    if mode == 'raw':
        directory = write_demo_series(tmp_path/'series')
        recipe = core.Session.load(directory/'demo.settings.json')
    else:
        directory = write_pdata_demo_series(tmp_path/'series')
        recipe = core.Session.load(directory/'demo.pdata.settings.json')
    recipe.analysis_method = 'direct'
    for region in recipe.regions:
        region.baseline = core.BaselineSettings(method=method, parameters=copy.deepcopy(parameters))
    recipe.save(tmp_path/'recipe.json')
    loaded = core.Session.load(tmp_path/'recipe.json')
    original = copy.deepcopy(loaded)
    calls = []
    real = core.baseline_correct
    def record(x, y, settings):
        calls.append(y.copy())
        return real(x, y, settings)
    monkeypatch.setattr(core, 'baseline_correct', record)
    rows, stopped = core.batch_analyze(directory, loaded, tmp_path/'out.csv')
    assert not stopped and len(rows) == 4
    assert len(calls) == 8
    assert not np.array_equal(calls[0], calls[2])
    assert original == loaded
    for row in rows:
        assert row['status'] != 'error', row['warnings']
        result, saved = core.load_analysis(row['fit_file'])
        assert saved == loaded
        assert all(r.baseline.parameters == parameters for r in saved.regions)
        assert all(np.isfinite(r.baseline).all() for r in result.results.values())
    payload = json.loads((tmp_path/'recipe.json').read_text(encoding='utf-8'))
    assert all(r['baseline']['parameters'] == parameters for r in payload['regions'])


def test_interpolation_samples_new_heights_from_saved_ppm_positions():
    x = np.linspace(3., 4., 501)
    peak = 4.*np.exp(-.5*((x-3.5)/.02)**2)
    b = core.BaselineSettings(method='interp_pts', parameters={'anchor_ppm': [3.1, 3.9]})
    before = copy.deepcopy(b)
    lines = []
    for offset, slope in [(5., 2.), (90., -10.)]:
        background = offset+slope*(x-3.)
        corrected, line, _ = core.baseline_correct(x, peak+background, b)
        np.testing.assert_allclose(line, background, atol=1e-8)
        np.testing.assert_allclose(corrected, peak, atol=1e-8)
        lines.append(line)
    assert not np.array_equal(*lines) and b == before


def test_new_method_parameters_change_the_computed_baseline():
    x = np.linspace(3., 4., 501)
    y = 2+x+5*np.exp(-.5*((x-3.5)/.06)**2)
    first = core.baseline_correct(x, y, core.BaselineSettings(method='pspline_asls', parameters={'lam': 10., 'p': .001}))[1]
    second = core.baseline_correct(x, y, core.BaselineSettings(method='pspline_asls', parameters={'lam': 1e6, 'p': .1}))[1]
    assert np.max(abs(first-second)) > .1


def test_auto_numeric_parameters_restore_defaults_without_changing_recipe():
    x = np.linspace(3., 4., 501)
    y = 2+x+5*np.exp(-.5*((x-3.5)/.03)**2)
    settings = core.BaselineSettings(method='asls', lam=100., parameters={'lam': None, 'p': None})
    before = copy.deepcopy(settings)
    actual = core.baseline_correct(x, y, settings)[1]
    expected = core.baseline_correct(x, y, core.BaselineSettings(method='asls'))[1]
    np.testing.assert_array_equal(actual, expected)
    assert settings == before


@pytest.mark.parametrize('parameters', [{'unknown': 4}, {'lam': -1}, {'lam': float('nan')},
    {'weights': np.ones(10)}, {'method_kwargs': {'not_a_parameter': 1}}])
def test_invalid_parameters_are_rejected_before_saving(parameters):
    method = 'custom_bc' if 'method_kwargs' in parameters else 'asls'
    with pytest.raises(ValueError):
        core.validate_baseline(core.BaselineSettings(method=method, parameters=parameters))


def test_safe_parameter_parser_and_asymmetric_snip_windows():
    assert parse_parameter('.001', .01) == .001
    assert parse_parameter('auto', None) is None
    assert parse_parameter('[3, 7]', None) == [3, 7]
    with pytest.raises(ValueError):
        parse_parameter('__import__("os").system("echo unexpected")', None)
    x = np.linspace(3., 4., 101)
    y = x+np.exp(-100*(x-3.5)**2)
    _, line, _ = core.baseline_correct(x, y, core.BaselineSettings(method='snip', parameters={'max_half_window': [3, 7]}))
    assert np.isfinite(line).all()
