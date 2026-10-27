"""A saved local reference aligns raw/pdata series with different native axes."""
import copy
import json

import nmrglue as ng
import numpy as np
import pytest

import spintegrate.core as core
from spintegrate.core import BaselineSettings, FitRegion, IntegrationRegion, Session
from test_pdata_direct import write_pdata_fixture


@pytest.mark.parametrize('sign', [1., -1.])
def test_local_reference_uses_signal_instead_of_complex_background(sign):
    x = np.linspace(6.8, 7.5, 701)
    y = 1e6+3000.*(x-7.15)+sign*100.*np.exp(-.5*((x-7.15)/.015)**2)
    complex_values = y+1j*1e9*(x-6.7)
    p = core.ProcessingSettings(reference_mode='window', reference_low=6.9,
                                reference_high=7.4, reference_ppm=7.2)
    shifted, _, metadata = core.reference_axis(x, complex_values, p, intensity=y)
    assert metadata['reference_observed_ppm'] == pytest.approx(7.15)
    assert shifted[np.argmin(abs(x-7.15))] == pytest.approx(7.2)


def test_local_reference_rejects_a_window_too_small_to_locate_a_peak():
    p = core.ProcessingSettings(reference_mode='window', reference_low=1., reference_high=2.)
    with pytest.raises(ValueError, match='трёх точек'):
        core.reference_axis(np.array([1., 2.]), np.array([1., 2.]), p)


@pytest.mark.parametrize('source_mode', ['raw', 'pdata'])
def test_saved_reference_applies_to_every_spectrum_before_region_analysis(tmp_path, source_mode):
    directory = tmp_path/'series'
    directory.mkdir()
    native_peaks = []
    for number, sf, drift in [(1, 400., 0.), (2, 500., .045), (3, 600., -.06)]:
        n, sw = 4096, 4000.
        nominal = 6.+np.fft.fftshift(np.fft.fftfreq(n, 1/sw))/sf
        ft = sum(amplitude*width/(width+1j*(nominal-center)) for center, width, amplitude in
                 [(7.15, .008, 20000.), (5.23, .012, 12000.), (8., .01, number*200000.)])
        if source_mode == 'raw':
            path = directory/f'{number}.npz'
            np.savez(path, fid=ng.proc_base.ifft(ft), sf_mhz=sf, sw_hz=sw, carrier_ppm=6.+drift)
        else:
            path, _, _ = write_pdata_fixture(directory/str(number)/'pdata'/'1',
                ft.real[::-1], nc_proc=0, offset=float(nominal[-1]+drift), sf=sf, sw=sw)
        p = core.ProcessingSettings(source_mode=source_mode, lb_hz=0.)
        plain = core.process(core.load_source(path, p), p)
        mask = (plain.ppm > 6.9) & (plain.ppm < 7.4)
        native_peaks.append(plain.ppm[mask][np.argmax(plain.intensity[mask])])
    assert np.ptp(native_peaks) > .1  # metadata alone intentionally keeps the input shifts
    recipe = Session(analysis_method='direct')
    recipe.processing.source_mode = source_mode
    recipe.processing.lb_hz = 0.
    recipe.processing.reference_mode = 'window'
    recipe.processing.reference_low, recipe.processing.reference_high = 6.9, 7.4
    recipe.processing.reference_ppm = 7.138
    recipe.processing.ppm_offset = .012  # final reference = 7.15, in both implementations
    recipe.regions = [FitRegion('reference', 6.9, 7.4, BaselineSettings(method='none')),
                      FitRegion('signal', 5., 5.5, BaselineSettings(method='none'))]
    recipe.integrals = [IntegrationRegion('ref_area', 'reference', 7.1, 7.2),
                        IntegrationRegion('signal_area', 'signal', 5.15, 5.32)]
    recipe.save(tmp_path/'recipe.json')
    loaded = Session.load(tmp_path/'recipe.json')
    before = copy.deepcopy(loaded)
    rows, stopped = core.batch_analyze(directory, loaded, tmp_path/'out.csv')
    assert not stopped and len(rows) == 3 and loaded == before
    for row, original_peak in zip(rows, native_peaks):
        assert row['status'] != 'error'
        analysis, saved = core.load_analysis(row['fit_file'])
        assert saved == loaded
        s = analysis.spectrum
        mask = (s.ppm > 6.9) & (s.ppm < 7.4)
        assert s.ppm[mask][np.argmax(s.intensity[mask])] == pytest.approx(7.15)
        mask = (s.ppm > 5.1) & (s.ppm < 5.4)
        assert s.ppm[mask][np.argmax(s.intensity[mask])] == pytest.approx(5.23, abs=.004)
        assert row['reference_shift_ppm'] == pytest.approx(7.15-original_peak, abs=.0001)
        assert s.processing_metadata['reference_mode'] == 'window'
        assert np.isfinite(row['integral__signal_area'])
    report = json.loads((tmp_path/'out.diagnostics.json').read_text(encoding='utf-8'))
    assert all('reference_shift_ppm' in spectrum['row'] for spectrum in report['spectra'])
