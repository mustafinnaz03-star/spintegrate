import copy
import json
from dataclasses import asdict

import nmrglue as ng
import numpy as np
import pytest

from spintegrate.core import (BaselineSettings, FitRegion, IntegrationRegion, PhaseState,
                              ProcessingSettings, RawSpectrum, Session, Spectrum, batch_analyze,
                              load_analysis, load_source, process)
from spintegrate.phase_transfer import (BatchPhaseSettings, batch_processing_settings,
                                       capture_batch_phase, process_for_batch)
from test_pdata_direct import write_pdata_fixture


ANGLES = (21.5, -28.)  # auto (22,-38) + trims (3,10) about fractional pivot .35


def phase_recipe(source_mode='raw'):
    s = Session(analysis_method='direct')
    s.processing = ProcessingSettings(source_mode=source_mode, phase='acme', manual_phase_p0=3.,
        manual_phase_p1=10., phase_pivot_fraction=.35, lb_hz=0.)
    s.regions = [FitRegion('fit', 3., 4., BaselineSettings(method='none'))]
    s.integrals = [IntegrationRegion('area', 'fit', 3.3, 3.8)]
    return s


def write_series(directory, mode, gains=(1., .6, -.3)):
    directory.mkdir()
    x = 2.+np.arange(8192)*6./8192
    ideal = 12000.*.008/(.008+1j*(x-3.5))
    observed = ng.proc_base.ps(ideal, p0=-ANGLES[0], p1=-ANGLES[1])
    paths = []
    for i, gain in enumerate(gains, 1):
        if mode == 'raw':
            path = directory/f'{i}.npz'
            np.savez(path, fid=ng.proc_base.ifft(gain*observed), sf_mhz=400., sw_hz=2400., carrier_ppm=5.)
        else:
            path = directory/str(i)/'pdata'/'1'
            values = gain*observed[::-1]*10000.
            write_pdata_fixture(path, values.real, values.imag, nc_proc=0,
                                offset=float(x[-1]), sf=400., sw=2400., phase=(100.+i, -200.-i))
        paths.append(path)
    return paths


@pytest.mark.parametrize('mode', ['raw', 'pdata'])
@pytest.mark.parametrize('alignment', ['off', 'global', 'regions'])
def test_fixed_phase_batch_never_rephases_and_preserves_signal_polarity(tmp_path, monkeypatch, mode, alignment):
    gains = (1., .6, -.3) if alignment == 'off' else (1., .6, .3)
    paths = write_series(tmp_path/'series', mode, gains)
    s = phase_recipe(mode)
    estimates = []
    def estimate(data, *_args, **_kwargs):
        estimates.append(data.copy())
        return data, (22., -38.)
    monkeypatch.setattr(ng.proc_autophase, 'autops', estimate)
    source, state = load_source(paths[0], s.processing), PhaseState()
    selected = process(source, s.processing, state)
    s.batch_phase = capture_batch_phase(source, s.processing, state)
    assert len(estimates) == 1
    assert (s.batch_phase.p0, s.batch_phase.p1) == ANGLES
    np.testing.assert_allclose(process_for_batch(source, s).complex_spectrum, selected.complex_spectrum, atol=1e-7)
    monkeypatch.setattr(ng.proc_autophase, 'autops', lambda *_args, **_kwargs: pytest.fail('Fixed batch reran autophase'))
    s.alignment.enabled = alignment != 'off'
    s.alignment.mode = alignment if alignment != 'off' else 'global'
    original = copy.deepcopy(s)
    rows, stopped = batch_analyze(paths[0].parent if mode == 'raw' else paths[0].parents[2], s, tmp_path/'out.csv')
    assert not stopped and s == original
    assert all(row['status'] != 'error' for row in rows), rows
    assert [(row['phase_p0_deg'], row['phase_p1_deg']) for row in rows] == [ANGLES]*3
    np.testing.assert_allclose([row['integral__area']/rows[0]['integral__area'] for row in rows], gains, rtol=1e-6)
    loaded = Session.load(tmp_path/'out.settings.json')
    assert loaded == s and loaded.processing.phase == 'acme'  # original tuning settings remain documented
    for row in rows:
        actual, recipe = load_analysis(row['fit_file'])
        assert recipe == loaded and actual.spectrum.phases == ANGLES
        assert actual.spectrum.processing_metadata['batch_phase'] == asdict(s.batch_phase)
        if mode == 'pdata':
            assert actual.spectrum.processing_metadata['additional_phases'] == list(ANGLES)
    assert json.loads((tmp_path/'out.settings.json').read_text(encoding='utf-8'))['batch_phase']['captured']


@pytest.mark.parametrize('source_mode', ['raw', 'pdata', 'pdata_positive'])
def test_fixed_phase_uses_pre_window_path_and_does_not_mutate_tuning_settings(monkeypatch, source_mode):
    x = np.linspace(2., 8., 4096, endpoint=False)
    ft = 2000.*.01/(.01+1j*(x-3.5))
    if source_mode == 'pdata_positive':
        ft = ft.conj()
    source = (RawSpectrum(ng.proc_base.ifft(ft), 400., 2400., 5., 'selected') if source_mode == 'raw' else
              Spectrum(x, ft.real, ft, 400., 2400., 'selected', (123., -57.), [], 'pdata', {'has_imaginary': True}))
    mode = 'raw' if source_mode == 'raw' else 'pdata'
    s = phase_recipe(mode)
    s.processing.lb_hz, s.processing.zero_fill_factor = 2., 2
    s.batch_phase = BatchPhaseSettings('fixed', True, *ANGLES, 'chosen', mode)
    before = copy.deepcopy(s)
    monkeypatch.setattr(ng.proc_autophase, 'autops', lambda *_args, **_kwargs: pytest.fail('Unexpected autophase'))
    actual = process_for_batch(source, s)
    p = batch_processing_settings(s)
    forward = ng.proc_base.fft_positive if source_mode == 'pdata_positive' else ng.proc_base.fft
    inverse = ng.proc_base.ifft_positive if source_mode == 'pdata_positive' else ng.proc_base.ifft
    expected = forward(ng.proc_base.zf_size(ng.proc_base.em(
        inverse(ng.proc_base.ps(ft, p0=ANGLES[0], p1=ANGLES[1])), lb=2./2400.), 8192))
    np.testing.assert_allclose(actual.complex_spectrum, expected, atol=1e-8)
    assert s == before and p.phase == 'none' and p.phase_pivot_fraction == 0.


def test_pdata_capture_never_reapplies_imported_topspin_phases_without_1i():
    x = np.linspace(3., 4., 256)
    y = np.exp(-((x-3.5)/.01)**2)*1000.
    source = Spectrum(x, y, y.astype(complex), 400., 400., 'pdata/1', (123., -77.), [], 'pdata',
                      {'has_imaginary': False, 'imported_phases': [123., -77.]})
    s = phase_recipe('pdata')
    s.processing = ProcessingSettings(source_mode='pdata', phase='none', lb_hz=0.)
    s.batch_phase = capture_batch_phase(source, s.processing)
    assert s.batch_phase.p0 == s.batch_phase.p1 == 0.
    actual = process_for_batch(source, s)
    np.testing.assert_array_equal(actual.intensity, y)
    assert actual.phases == (0., 0.)
    assert actual.processing_metadata['imported_phases'] == [123., -77.]


def test_fixed_phase_guards_and_legacy_recipe_defaults(tmp_path):
    old = asdict(phase_recipe())
    old.pop('batch_phase')
    assert Session.from_dict(old).batch_phase.mode == 'per_spectrum'
    s = phase_recipe()
    s.batch_phase.mode = 'fixed'
    with pytest.raises(ValueError, match='зафиксируйте'):
        batch_analyze(tmp_path, s, tmp_path/'out.csv')
    assert not (tmp_path/'out.settings.json').exists()
    s.batch_phase = BatchPhaseSettings('fixed', True, 10., -5., 'source', 'raw')
    s.processing.phase = 'abs'
    with pytest.raises(ValueError, match='abs'):
        batch_processing_settings(s)
    with pytest.raises(ValueError, match='abs'):
        capture_batch_phase(None, s.processing)
    s.processing.phase, s.processing.source_mode = 'none', 'pdata'
    with pytest.raises(ValueError, match='raw/pdata'):
        batch_processing_settings(s)
    with pytest.raises(ValueError, match='конечные'):
        BatchPhaseSettings(p0=float('nan')).validate()


def test_default_batch_still_estimates_phase_independently(tmp_path, monkeypatch):
    s = phase_recipe()
    paths = write_series(tmp_path/'series', 'raw')
    calls = []
    def estimate(data, *_args, **_kwargs):
        calls.append(1)
        return data, (22.+len(calls), -38.)
    monkeypatch.setattr(ng.proc_autophase, 'autops', estimate)
    rows, _ = batch_analyze(paths[0].parent, s, tmp_path/'out.csv')
    assert len(calls) == 3
    assert [row['phase_p0_deg'] for row in rows] == [22.5, 23.5, 24.5]
