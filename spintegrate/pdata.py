"""Bruker processed-spectrum input with exact saved TopSpin frequency bins."""
from pathlib import Path
import warnings

import nmrglue as ng
import numpy as np


def resolve_pdata_path(path, procno='1'):
    """Accept an experiment, its pdata directory, a processing directory or 1r."""
    procno = str(procno)
    if not procno.isdigit() or int(procno) < 1:
        raise ValueError('Номер обработки pdata должен быть положительным целым числом.')
    path = Path(path)
    if path.is_file():
        if path.name != '1r':
            raise ValueError('Для pdata выберите файл 1r или папку с 1r и procs.')
        path = path.parent
    if (path / '1r').is_file() and (path / 'procs').is_file():
        return path
    candidate = path / procno if path.name.lower() == 'pdata' else path / 'pdata' / procno
    if (candidate / '1r').is_file() and (candidate / 'procs').is_file():
        return candidate
    raise ValueError(f'Не найдены 1r и procs для обработки pdata/{procno}: {path}')


def load_pdata(path, procno='1'):
    from .core import Spectrum
    directory = resolve_pdata_path(path, procno)
    has_imag = (directory / '1i').is_file()
    components = ['1r', '1i'] if has_imag else ['1r']
    # Acquisition metadata must never override the processed SF/OFFSET/SW_p.
    with warnings.catch_warnings(record=True) as caught:
        dic, values = ng.bruker.read_pdata(str(directory), bin_files=components,
                                          read_acqus=False, scale_data=True)
    procs = dic.get('procs', {})
    if 'proc2s' in dic:
        raise ValueError('Поддерживаются только одномерные обработанные спектры pdata.')
    try:
        sf, sw, offset = float(procs['SF']), float(procs['SW_p']), float(procs['OFFSET'])
        si = int(procs['SI'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('В procs нужны корректные SF, SW_p, OFFSET и SI для шкалы TopSpin.') from exc
    if not np.isfinite([sf, sw, offset]).all() or sf <= 0 or sw <= 0 or si < 2:
        raise ValueError('Некорректная шкала TopSpin: SF, SW_p, OFFSET или SI.')
    real, imag = values if has_imag else (values, None)
    real = np.asarray(real, dtype=float)
    imag = np.asarray(imag, dtype=float) if imag is not None else None
    if real.ndim != 1 or (imag is not None and imag.ndim != 1):
        raise ValueError('Поддерживаются только одномерные обработанные спектры pdata.')
    # Ordinary processed spectra have SI == FTSIZE (or omit FTSIZE), STSR=0
    # and STSI=0/SI. Strip transforms use a different original FFT grid.
    # nmrglue.guess_udic(strip_fake=True) applies a special carrier correction,
    # while TopSpin documents OFFSET as the first spectral point. Without an
    # actual stripped TopSpin reference we cannot disambiguate saved OFFSET
    # conventions safely, so never silently shift a cropped spectrum's ppm.
    stsi = int(procs.get('STSI', 0))
    stsr = int(procs.get('STSR', 0))
    ftsize = int(procs.get('FTSIZE', si))
    if (stsr > 0 or (stsi > 0 and stsi != si)
            or (ftsize > 0 and ftsize != si)):
        raise ValueError('Обрезанные pdata (STSR/STSI/FTSIZE) пока не поддерживаются: '
                         'сохраните полный спектр в TopSpin или откройте raw, чтобы не исказить шкалу ppm.')
    stored, grid_size = si, si
    if stored < 2 or stsi < 0 or stsr < 0:
        raise ValueError('Некорректные STSI/STSR/FTSIZE в обработанном спектре.')
    if len(real) < stored or (imag is not None and len(imag) < stored):
        raise ValueError('Файлы 1r/1i короче размера, указанного в procs.')
    real = real[:stored]
    imag = imag[:stored] if imag is not None else None
    if not np.isfinite(real).all() or (imag is not None and not np.isfinite(imag).all()):
        raise ValueError('В обработанном спектре есть нечисловые значения.')
    step = sw / (sf * grid_size)
    ppm_descending = offset - np.arange(stored)*step
    spectrum = real.astype(complex)
    if imag is not None:
        spectrum += 1j*imag
    phases = tuple(float(procs.get(k, 0.)) for k in ('PHC0', 'PHC1'))
    if not np.isfinite(phases).all():
        raise ValueError('Некорректные PHC0/PHC1 в procs.')
    metadata = dict(procs=procs, procno=directory.name, has_imaginary=has_imag,
                    has_imag=has_imag, imported_phases=list(phases),
                    axis_step_ppm=step, cropped=False,
                    stored_points=stored, transform_size=grid_size)
    notes = [str(w.message) for w in caught]
    return Spectrum(ppm_descending[::-1].copy(), real[::-1].copy(), spectrum[::-1].copy(),
                    sf, sw, str(directory), phases, notes, 'pdata', metadata)


def pdata_uses_positive_transform(source):
    """Resolve time direction from the original complex spectrum, before edits.

    Reordering the saved descending ppm axis can leave an opposite imaginary
    convention to our raw FFT. Its inverse then has the decay at *negative*
    times (the end of the cyclic array). A one-sided EM window on that array
    erases the peaks. Use the matched positive FFT/IFFT pair in that case.
    The DC point is excluded so a large constant baseline cannot set direction.
    No phase angles or imported real/imaginary values are changed here.
    """
    fid = ng.proc_base.ifft(source.complex_spectrum)
    midpoint = len(fid)//2
    early = np.linalg.norm(fid[1:midpoint])
    late = np.linalg.norm(fid[-1:-midpoint:-1])
    return bool(late > early * (1. + 1e-6))


def process_pdata(source, p, phase_cache=None):
    from .core import (Spectrum, process_frequency_data, resolve_phase_angles,
                       reference_axis, spectrum_from_fourier, has_manual_phase, add_manual_phase)
    metadata = dict(source.processing_metadata)
    has_imag = bool(metadata.get('has_imaginary', metadata.get('has_imag', False)))
    extra_transform = p.lb_hz > 0 or p.zero_fill_factor != 1 or p.phase != 'none' or has_manual_phase(p)
    if extra_transform:
        if not has_imag:
            raise ValueError('Для дополнительной аподизации, фазировки, abs или zero filling pdata нужен файл 1i. '
                             'Без 1i доступны исходная фазировка TopSpin, базовая линия и анализ областей.')
        if metadata.get('cropped', False):
            raise ValueError('Дополнительная обработка через IFFT недоступна для обрезанного pdata. '
                             'Используйте исходный raw; анализ сохранённых областей pdata остаётся доступен.')
        # Start from the saved, already phased TopSpin frequency data. Extra
        # phase is applied before any additional window; saved PHC0/PHC1 is
        # never applied again. With no window/padding there is no FFT roundtrip.
        bins = np.fft.fftshift(np.fft.fftfreq(len(source.complex_spectrum), d=1/source.sw_hz))
        carrier = source.ppm[0] - bins[0]/source.sf_mhz
        phases, notes = resolve_phase_angles(source, p, phase_cache,
                                             reference=source.complex_spectrum)
        phases = add_manual_phase(phases, p)
        positive = pdata_uses_positive_transform(source)
        ft = process_frequency_data(source.complex_spectrum, p, source.sw_hz, phases,
                                    positive=positive)
        result = spectrum_from_fourier(ft, p, source.sf_mhz, source.sw_hz, carrier,
                                       source.source, phases, list(source.notes) + notes)
        result.source_type = 'pdata'
        result.processing_metadata = {**metadata, **result.processing_metadata}
        result.processing_metadata['additional_phases'] = list(result.phases)
        result.processing_metadata['additional_fft_convention'] = 'positive' if positive else 'negative'
        if p.phase == 'none' and not has_manual_phase(p):
            result.phases = source.phases
        result.notes.append('Дополнительная обработка применена к спектру после TopSpin; исходная фазировка не повторяется.')
        return result
    ppm, reference_notes, reference_info = reference_axis(source.ppm, source.complex_spectrum, p,
                                                        intensity=source.intensity)
    y = source.intensity.copy()
    notes = list(source.notes) + reference_notes
    return Spectrum(ppm, y, source.complex_spectrum.copy(), source.sf_mhz, source.sw_hz,
                    source.source, source.phases, notes, 'pdata', {**metadata, **reference_info})
