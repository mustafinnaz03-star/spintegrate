"""Synthetic complex FIDs for a demo and numerical verification (not real data)."""
from pathlib import Path
import nmrglue as ng
import numpy as np

from .core import (BaselineSettings, FitRegion, IntegrationRegion,
                   ProcessingSettings, RawSpectrum, Session, process)


def demo_raw(multiplier=1., source='Синтетический спектр'):
    sf, sw, carrier, size = 400., 4800., 6., 8192
    ppm = carrier + np.fft.fftshift(np.fft.fftfreq(size, d=1/sw))/sf
    # Analytic absorption + dispersion; the inverse transform yields a complex FID.
    ft = np.zeros(size, dtype=complex)
    for center, width, amplitude in [(3.55, .006, 70000.), (3.59, .008, 42000.),
                                      (5.23, .01, 57000.), (5.28, .007, 30000.),
                                      (7.15, .004, 180000.)]:
        d = ppm-center
        ft += multiplier * amplitude * width / (width + 1j*d)
    ft += 700 + 25*(ppm-6)
    rng = np.random.default_rng(7)
    ft += rng.normal(0, 35, size) + 1j*rng.normal(0, 35, size)
    return RawSpectrum(ng.proc_base.ifft(ft), sf, sw, carrier, source,
                       ['Демонстрационные синтетические данные.'])


def demo_session():
    s = Session()
    s.processing.lb_hz = 0.
    s.fit.threshold = 4000.
    s.fit.initial_width_hz = 5.
    b = BaselineSettings(method='asls', lam=1e6, p=.001)
    s.regions = [FitRegion('Стандарт', 3.3, 3.85, b), FitRegion('Сигнал', 5., 5.55, b)]
    s.integrals = [IntegrationRegion('standard', 'Стандарт', 3.5, 3.65),
                   IntegrationRegion('signal', 'Сигнал', 5.15, 5.35)]
    return s


def write_demo_series(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for number, multiplier in enumerate((1., .8, .6, .4), 1):
        raw = demo_raw(multiplier)
        np.savez(directory / f'{number}.npz', fid=raw.fid, sf_mhz=raw.sf_mhz,
                 sw_hz=raw.sw_hz, carrier_ppm=raw.carrier_ppm)
    demo_session().save(directory / 'demo.settings.json')
    return directory


def write_pdata_demo_series(directory, procno='1'):
    """Create self-contained synthetic TopSpin-like 1r/1i/procs examples."""
    directory = Path(directory)
    for number, multiplier in enumerate((1., .8, .6, .4), 1):
        spectrum = process(demo_raw(multiplier), ProcessingSettings(lb_hz=0.))
        target = directory / str(number) / 'pdata' / str(procno)
        target.mkdir(parents=True, exist_ok=True)
        size = len(spectrum.ppm)
        dic = {'procs': {
            '_coreheader': ['##TITLE= Synthetic Spintegrate pdata demo', '##JCAMPDX= 5.00'],
            '_comments': [], 'SI': size, 'XDIM': size, 'FTSIZE': size,
            'STSI': 0, 'STSR': 0, 'PPARMOD': 0, 'BYTORDP': 0, 'DTYPP': 2, 'NC_proc': 0,
            'SF': spectrum.sf_mhz, 'SW_p': spectrum.sw_hz, 'OFFSET': float(spectrum.ppm[-1]),
            'PHC0': 0., 'PHC1': 0., 'LB': 0.,
        }}
        ng.bruker.write_pdata(str(target), dic, spectrum.complex_spectrum.real[::-1],
                             scale_data=True, write_procs=True, isfloat=True)
        ng.bruker.write_pdata(str(target), dic, spectrum.complex_spectrum.imag[::-1],
                             scale_data=True, bin_file='1i', isfloat=True)
    recipe = demo_session()
    recipe.processing.source_mode = 'pdata'
    recipe.processing.pdata_procno = str(procno)
    recipe.save(directory / 'demo.pdata.settings.json')
    return directory
