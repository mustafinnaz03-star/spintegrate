from dataclasses import asdict

import nmrglue as ng
import numpy as np
import pytest

from spintegrate.core import PhaseState, ProcessingSettings, Session, load_pdata, process
from spintegrate.demo import demo_raw, write_pdata_demo_series


@pytest.mark.parametrize('mode', ['raw', 'pdata', 'pdata_positive'])
def test_manual_phase_uses_same_pivot_curve_before_window_and_zero_fill(tmp_path, mode):
    source = demo_raw()
    if mode.startswith('pdata'):
        directory = write_pdata_demo_series(tmp_path/'pdata')
        source = load_pdata(directory/'1'/'pdata'/'1')
        if mode == 'pdata_positive':
            source.complex_spectrum = source.complex_spectrum.conj()
        original = source.complex_spectrum.copy()
    else:
        original = source.fid.copy()
    p = ProcessingSettings(source_mode='pdata' if mode.startswith('pdata') else 'raw', phase='none', manual_phase_p0=24., manual_phase_p1=-36.,
                           phase_pivot_fraction=.3, lb_hz=2., zero_fill_factor=2)
    base = ng.proc_base.fft(source.fid) if mode == 'raw' else source.complex_spectrum
    total = (24.-(-36.)*.3, -36.)
    forward = ng.proc_base.fft_positive if mode == 'pdata_positive' else ng.proc_base.fft
    inverse = ng.proc_base.ifft_positive if mode == 'pdata_positive' else ng.proc_base.ifft
    expected = forward(ng.proc_base.zf_size(ng.proc_base.em(
        inverse(ng.proc_base.ps(base, p0=total[0], p1=total[1])), lb=2./source.sw_hz), 2*len(base)))
    actual = process(source, p)
    np.testing.assert_allclose(actual.complex_spectrum, expected)
    assert actual.phases == total
    np.testing.assert_array_equal(source.fid if mode == 'raw' else source.complex_spectrum, original)


def test_manual_offsets_and_pivot_do_not_rerun_automatic_phase(monkeypatch):
    estimates = []
    def phaser(data, *_args, **_kwargs):
        estimates.append(data.copy())
        return data, (12., -8.)
    monkeypatch.setattr(ng.proc_autophase, 'autops', phaser)
    source, state = demo_raw(), PhaseState()
    p = ProcessingSettings(phase='acme', lb_hz=0.)
    process(source, p, state)
    p.manual_phase_p0, p.manual_phase_p1 = 3., 10.
    first = process(source, p, state)
    old_fraction = p.phase_pivot_fraction
    p.phase_pivot_fraction = .2
    p.manual_phase_p0 += p.manual_phase_p1*(p.phase_pivot_fraction-old_fraction)
    moved = process(source, p, state)
    np.testing.assert_allclose(moved.complex_spectrum, first.complex_spectrum)
    p.lb_hz = 4.
    changed = process(source, p, state)
    assert changed.phases == moved.phases
    assert len(estimates) == 1


def test_abs_ignores_manual_phase_and_legacy_json_defaults():
    source = demo_raw()
    a = process(source, ProcessingSettings(phase='abs', lb_hz=2.))
    b = process(source, ProcessingSettings(phase='abs', lb_hz=2., manual_phase_p0=50., manual_phase_p1=100.))
    np.testing.assert_array_equal(a.intensity, b.intensity)
    data = asdict(Session())
    for name in ('manual_phase_p0', 'manual_phase_p1', 'phase_pivot_fraction'):
        data['processing'].pop(name)
    data.pop('alignment')
    restored = Session.from_dict(data)
    assert restored.processing.manual_phase_p0 == restored.processing.manual_phase_p1 == 0.
    assert restored.processing.phase_pivot_fraction == .5
    assert not restored.alignment.enabled


def test_manual_parameters_roundtrip_json(tmp_path):
    s = Session(processing=ProcessingSettings(manual_phase_p0=3.2, manual_phase_p1=21., phase_pivot_fraction=.27))
    s.save(tmp_path/'phase.json')
    assert Session.load(tmp_path/'phase.json') == s
