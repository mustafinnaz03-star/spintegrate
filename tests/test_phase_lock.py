"""Phase is estimated before apodization and reused independently of display settings."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import nmrglue as ng
import numpy as np
import pytest

import spintegrate.core as core
from spintegrate.core import ProcessingSettings, Session, process
from spintegrate.demo import demo_raw, demo_session, write_demo_series, write_pdata_demo_series


@pytest.fixture
def phaser(monkeypatch):
    inputs = []
    guard = threading.Lock()
    def fake(data, fn, return_phases=False, **kwargs):
        with guard:
            inputs.append(np.asarray(data).copy())
        # Keep estimation slow enough to expose duplicate concurrent evaluations.
        time.sleep(.015)
        phases = np.array([37., -11.])
        result = ng.proc_base.ps(data, p0=phases[0], p1=phases[1])
        return (result, phases) if return_phases else result
    monkeypatch.setattr(ng.proc_autophase, 'autops', fake)
    return inputs


@pytest.mark.parametrize('mode', ['acme', 'peak_minima'])
def test_cached_phase_is_not_reestimated_for_lb_reference_or_zero_fill(mode, phaser):
    source = demo_raw()
    state = core.PhaseState()
    settings = ProcessingSettings(phase=mode, lb_hz=0.)
    first = process(source, settings, phase_cache=state)
    expected_input = ng.proc_base.fft(source.fid)
    np.testing.assert_array_equal(phaser[0], expected_input)
    for lb in (1., 3., 7.):
        settings.lb_hz = lb
        current = process(source, settings, phase_cache=state)
        assert current.phases == first.phases == (37., -11.)
        phased_fid = ng.proc_base.ifft(ng.proc_base.ps(expected_input, p0=37., p1=-11.))
        expected = ng.proc_base.fft(ng.proc_base.em(phased_fid, lb=lb/source.sw_hz))
        np.testing.assert_allclose(current.complex_spectrum, expected)
        np.testing.assert_allclose(current.intensity, expected.real)
        late_phase = ng.proc_base.ps(ng.proc_base.fft(ng.proc_base.em(source.fid, lb=lb/source.sw_hz)),
                                    p0=37., p1=-11.)
        # First-order phase and apodization do not commute.
        assert not np.allclose(expected, late_phase)
    settings.zero_fill_factor = 2
    settings.reference_mode = 'strongest'
    settings.reference_ppm = 7.35
    settings.ppm_offset = .05
    changed = process(source, settings, phase_cache=state)
    assert changed.phases == first.phases
    assert len(phaser) == 1
    assert len(changed.intensity) == 2*len(source.fid)
    expected = ng.proc_base.fft(ng.proc_base.zf_size(
        ng.proc_base.em(phased_fid, lb=settings.lb_hz/source.sw_hz), 2*len(source.fid)))
    np.testing.assert_allclose(changed.complex_spectrum, expected)


def test_uncached_processing_estimates_from_same_unapodized_data_at_all_lb(phaser):
    source = demo_raw()
    for lb, factor in [(0., 1), (2., 2), (6., 4)]:
        result = process(source, ProcessingSettings(phase='acme', lb_hz=lb, zero_fill_factor=factor))
        assert result.phases == (37., -11.)
    assert len(phaser) == 3
    expected = ng.proc_base.fft(source.fid)
    for data in phaser:
        np.testing.assert_array_equal(data, expected)


def test_explicit_rephase_phase_inputs_and_new_source_each_resolve_again(phaser):
    state = core.PhaseState()
    source = demo_raw()
    settings = ProcessingSettings(phase='acme')
    process(source, settings, phase_cache=state)
    state.clear()
    process(source, settings, phase_cache=state)
    assert len(phaser) == 2
    settings.p0 = 10.
    process(source, settings, phase_cache=state)
    settings.p1 = 5.
    process(source, settings, phase_cache=state)
    settings.phase_peak_width = 60
    process(source, settings, phase_cache=state)
    settings.phase = 'peak_minima'
    process(source, settings, phase_cache=state)
    process(demo_raw(.8, source='another experiment'), settings, phase_cache=state)
    assert len(phaser) == 7


def test_shared_preview_cache_is_thread_safe(phaser):
    source = demo_raw()
    state = core.PhaseState()
    choices = [ProcessingSettings(phase='acme', lb_hz=lb) for lb in (0., 1., 4.)]
    with ThreadPoolExecutor(max_workers=3) as workers:
        values = list(workers.map(lambda p: process(source, p, phase_cache=state), choices))
    assert len(phaser) == 1
    assert all(s.phases == (37., -11.) for s in values)


def test_abs_is_always_processed_spectrum_magnitude_including_legacy_settings():
    source = demo_raw()
    fid = source.fid.copy()
    expected = np.abs(ng.proc_base.fft(ng.proc_base.em(fid, lb=2./source.sw_hz)))
    for legacy_source in ('fid', 'spectrum'):
        settings = ProcessingSettings(phase='abs', abs_source=legacy_source, lb_hz=2.)
        result = process(source, settings)
        np.testing.assert_array_equal(result.intensity, expected)
    np.testing.assert_array_equal(source.fid, fid)
    data = asdict(Session(processing=ProcessingSettings(phase='abs')))
    data['processing']['abs_source'] = 'fid'
    restored = Session.from_dict(data)
    assert restored.processing.abs_source == 'spectrum'
    np.testing.assert_array_equal(process(source, restored.processing).intensity,
                                  process(source, ProcessingSettings(phase='abs')).intensity)


def test_pdata_additional_phase_reuses_original_processed_reference(tmp_path, phaser):
    directory = write_pdata_demo_series(tmp_path/'processed')
    source = core.load_pdata(directory/'1'/'pdata'/'1')
    original = source.complex_spectrum.copy()
    settings = ProcessingSettings(source_mode='pdata', phase='acme', lb_hz=0.)
    state = core.PhaseState()
    first = process(source, settings, phase_cache=state)
    settings.lb_hz = 3.
    current = process(source, settings, phase_cache=state)
    assert len(phaser) == 1
    np.testing.assert_array_equal(phaser[0], original)
    np.testing.assert_array_equal(source.complex_spectrum, original)
    phased = ng.proc_base.ps(original, p0=37., p1=-11.)
    expected = ng.proc_base.fft(ng.proc_base.em(ng.proc_base.ifft(phased), lb=3./source.sw_hz))
    np.testing.assert_allclose(current.complex_spectrum, expected)
    late_phase = ng.proc_base.ps(ng.proc_base.fft(ng.proc_base.em(ng.proc_base.ifft(original),
                                                               lb=3./source.sw_hz)), p0=37., p1=-11.)
    assert not np.allclose(expected, late_phase)
    assert current.phases == first.phases


def test_pdata_existing_phasing_is_not_reapplied_when_only_lb_changes(tmp_path, phaser):
    directory = write_pdata_demo_series(tmp_path/'processed')
    source = core.load_pdata(directory/'1'/'pdata'/'1')
    source.phases = (120., -70.)
    source.processing_metadata['imported_phases'] = [120., -70.]
    source.processing_metadata['procs'].update(PHC0=120., PHC1=-70.)
    settings = ProcessingSettings(source_mode='pdata', phase='none', lb_hz=0.)
    plain = process(source, settings, phase_cache=core.PhaseState())
    np.testing.assert_array_equal(plain.intensity, source.intensity)
    settings.lb_hz = 2.
    broadened = process(source, settings, phase_cache=core.PhaseState())
    expected = ng.proc_base.fft(ng.proc_base.em(ng.proc_base.ifft(source.complex_spectrum),
                                              lb=2./source.sw_hz))
    np.testing.assert_allclose(broadened.intensity, expected.real)
    assert broadened.phases == plain.phases == (120., -70.)
    assert not phaser


def test_batch_estimates_independently_once_per_spectrum(tmp_path, phaser):
    directory = write_demo_series(tmp_path/'inputs')
    settings = demo_session()
    settings.processing.phase = 'acme'
    settings.processing.lb_hz = 2.
    settings.analysis_method = 'direct'
    rows, cancelled = core.batch_analyze(directory, settings, tmp_path/'batch.csv')
    assert len(rows) == 4 and not cancelled
    assert len(phaser) == 4
    for index, transformed in enumerate(phaser, 1):
        with np.load(directory/f'{index}.npz', allow_pickle=False) as data:
            expected = ng.proc_base.fft(data['fid'])
        np.testing.assert_array_equal(transformed, expected)
