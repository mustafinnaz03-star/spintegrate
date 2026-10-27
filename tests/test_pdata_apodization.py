"""Processed spectra must broaden like a known causal raw FID, with either sign of 1i."""
import nmrglue as ng
import numpy as np
import pytest

from spintegrate.core import ProcessingSettings, RawSpectrum, load_pdata, process
from test_pdata_direct import write_pdata_fixture


def physical_pair(tmp_path, sign):
    n, sw, sf, carrier = 32768, 4000., 400., 6.
    t = np.arange(n)/sw
    fid = 10000.*np.exp(-np.pi*3.*t)*np.exp(2j*np.pi*320.*t)
    fid[0] *= .5
    raw = RawSpectrum(fid, sf, sw, carrier, 'known 3 Hz line')
    original = process(raw, ProcessingSettings(lb_hz=0.))
    ft = original.complex_spectrum
    directory, _, _ = write_pdata_fixture(tmp_path/'pdata'/'1', ft.real[::-1],
                                         (sign*ft.imag)[::-1], nc_proc=0,
                                         offset=float(original.ppm[-1]), sf=sf, sw=sw)
    return raw, load_pdata(directory)


@pytest.mark.parametrize('sign', [1., -1.])
@pytest.mark.parametrize('lb', [0., 1., 2., 5.])
@pytest.mark.parametrize('factor', [1, 2])
def test_same_physical_line_broadens_equally_in_raw_and_pdata(tmp_path, sign, lb, factor):
    raw, source = physical_pair(tmp_path, sign)
    original = source.complex_spectrum.copy()
    settings = ProcessingSettings(lb_hz=lb, zero_fill_factor=factor)
    expected = process(raw, settings)
    actual = process(source, settings)
    np.testing.assert_allclose(actual.ppm, expected.ppm, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual.intensity, expected.intensity, atol=5., rtol=1e-5)
    np.testing.assert_allclose(actual.complex_spectrum.imag,
                               sign*expected.complex_spectrum.imag, atol=5., rtol=1e-5)
    np.testing.assert_array_equal(source.complex_spectrum, original)
    assert abs(actual.ppm[np.argmax(actual.intensity)]-6.8) < .0004
    if lb:
        assert actual.processing_metadata['additional_fft_convention'] == ('positive' if sign < 0 else 'negative')
    # Independently check the physical FWHM: original 3 Hz plus additional LB.
    peak = np.argmax(actual.intensity)
    half = actual.intensity[peak]/2
    lo = np.interp(half, actual.intensity[:peak+1], actual.ppm[:peak+1])
    hi = np.interp(half, actual.intensity[peak:][::-1], actual.ppm[peak:][::-1])
    assert (hi-lo)*source.sf_mhz == pytest.approx(3.+lb, abs=.03)


def test_large_dc_baseline_cannot_reverse_the_resolved_time_direction(tmp_path):
    raw, source = physical_pair(tmp_path, -1.)
    source.complex_spectrum += 1e7
    source.intensity += 1e7
    settings = ProcessingSettings(lb_hz=2.)
    result = process(source, settings)
    expected = process(raw, settings)
    np.testing.assert_allclose(result.intensity-1e7, expected.intensity, atol=5., rtol=1e-5)


def test_opposite_imaginary_convention_keeps_fixed_phase_before_window(tmp_path, monkeypatch):
    _, source = physical_pair(tmp_path, -1.)
    from spintegrate.core import PhaseState
    estimates = []
    def phaser(data, *_args, **_kwargs):
        estimates.append(np.asarray(data).copy())
        return data, (20., -6.)
    monkeypatch.setattr(ng.proc_autophase, 'autops', phaser)
    state = PhaseState()
    settings = ProcessingSettings(phase='acme', lb_hz=0.)
    process(source, settings, phase_cache=state)
    settings.lb_hz = 2.
    actual = process(source, settings, phase_cache=state)
    phased = ng.proc_base.ps(source.complex_spectrum, p0=20., p1=-6.)
    expected = ng.proc_base.fft_positive(ng.proc_base.em(
        ng.proc_base.ifft_positive(phased), lb=2./source.sw_hz))
    assert len(estimates) == 1
    assert actual.phases == (20., -6.)
    np.testing.assert_allclose(actual.complex_spectrum, expected)
