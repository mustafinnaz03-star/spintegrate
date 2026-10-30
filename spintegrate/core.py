"""Numerical pipeline, independent of the desktop interface.

Lorentzian width is HWHM in ppm. The historical Spintegrate integral is A*w;
the whole-line mathematical area is pi*A*w. Automatic peak-picking fixes centers;
the separate manual-peak mode also refines centers within bounded windows.
"""
from __future__ import annotations

import csv
import json
import warnings
import threading
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import nmrglue as ng
import numpy as np
from pybaselines import Baseline
from scipy.optimize import least_squares
from .appearance import AppearanceSettings, validate_appearance, validate_color
from .baseline_methods import BASELINE_METHODS, effective_parameters, fit_baseline, parameter_specs, validate_parameters
from .alignment import AlignmentSettings, align_regions, align_spectrum


PHASE_MODES = ('none', 'acme', 'peak_minima', 'abs')


@dataclass
class BaselineSettings:
    method: str = 'snip'
    lam: float = 1e6
    p: float = .01
    max_iter: int = 50
    tol: float = .001
    half_window: int = 30
    smooth_half_window: int = 3
    poly_order: int = 3
    parameters: dict = field(default_factory=dict)


@dataclass
class ProcessingSettings:
    phase: str = 'none'
    abs_source: str = 'spectrum'  # deprecated compatibility field; abs always means |spectrum|
    lb_hz: float = 1.0
    p0: float = 0.0
    p1: float = 0.0
    phase_peak_width: int = 100
    reference_mode: str = 'metadata'  # metadata / strongest / window
    reference_ppm: float = 7.15
    reference_low: float = 6.8
    reference_high: float = 7.5
    ppm_offset: float = 0.0
    zero_fill_factor: int = 1
    remove_digital_filter: bool = True
    source_mode: str = 'raw'
    pdata_procno: str = '1'
    manual_phase_p0: float = 0.
    manual_phase_p1: float = 0.
    phase_pivot_fraction: float = .5

    def __post_init__(self):
        # Older recipes offered abs(FID). That option is intentionally retired.
        self.abs_source = 'spectrum'


@dataclass
class FitSettings:
    threshold: float = 1000.0
    min_separation_points: int = 2
    signed: bool = False
    initial_width_hz: float = 1.0  # fallback FWHM if lb_hz=0
    min_width_hz: float = .05
    max_width_hz: float = 50.0
    max_nfev: int = 2000
    restarts: int = 3
    max_peaks: int = 150
    loss: str = 'linear'
    residual_warning: float = .15
    cancellation_warning: float = .7
    center_tolerance_ppm: float = .03  # residual drift after alignment, manual mode only


@dataclass
class FitRegion:
    name: str
    low: float
    high: float
    baseline: BaselineSettings = field(default_factory=BaselineSettings)
    manual_peaks: list[float] = field(default_factory=list)


@dataclass
class IntegrationRegion:
    name: str
    parent: str
    low: float
    high: float
    color: str | None = None


@dataclass
class Session:
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    fit: FitSettings = field(default_factory=FitSettings)
    baseline_default: BaselineSettings = field(default_factory=BaselineSettings)
    regions: list[FitRegion] = field(default_factory=list)
    integrals: list[IntegrationRegion] = field(default_factory=list)
    appearance: AppearanceSettings = field(default_factory=AppearanceSettings)
    schema_version: int = 1
    analysis_method: str = 'lorentzian'
    alignment: AlignmentSettings = field(default_factory=AlignmentSettings)

    def validate(self, require_regions: bool = False):
        validate_appearance(self.appearance)
        self.alignment.validate()
        p, f = self.processing, self.fit
        if self.analysis_method not in ('lorentzian', 'lorentzian_manual', 'direct'):
            raise ValueError('Неизвестный метод анализа: lorentzian, lorentzian_manual или direct.')
        if p.source_mode not in ('raw', 'pdata'):
            raise ValueError('Источник спектра должен быть raw или pdata.')
        if p.source_mode == 'pdata' and (not str(p.pdata_procno).isdigit() or int(p.pdata_procno) < 1):
            raise ValueError('Номер обработки pdata должен быть положительным целым числом.')
        if p.phase not in PHASE_MODES:
            raise ValueError('Неизвестный режим фазирования/abs.')
        if p.reference_mode not in ('metadata', 'strongest', 'window'):
            raise ValueError('Неизвестный режим привязки ppm.')
        if not np.isfinite(p.lb_hz) or p.lb_hz < 0:
            raise ValueError('Аподизация LB должна быть >= 0 Гц.')
        if p.zero_fill_factor not in (1, 2, 4, 8):
            raise ValueError('Zero filling: 1, 2, 4 или 8.')
        if p.phase_peak_width < 1:
            raise ValueError('Окно peak_minima должно быть положительным.')
        if p.reference_mode == 'window' and p.reference_low >= p.reference_high:
            raise ValueError('Неверное окно референса.')
        if self.analysis_method != 'direct':
            if self.analysis_method == 'lorentzian' and not (np.isfinite(f.threshold) and f.threshold > 0):
                raise ValueError('Threshold должен быть положительным.')
            if not (0 < f.min_width_hz < f.max_width_hz) or f.initial_width_hz <= 0:
                raise ValueError('Ширины: 0 < min < max; initial > 0.')
            if f.max_nfev < 1 or f.restarts < 1 or f.max_peaks < 1 or f.min_separation_points < 1:
                raise ValueError('Лимиты оптимизации и число точек должны быть >= 1.')
            if f.loss not in ('linear', 'soft_l1'):
                raise ValueError('Loss: linear или soft_l1.')
            if not 0 < f.cancellation_warning < 1 or f.residual_warning <= 0:
                raise ValueError('Неверные пороги диагностики.')
            if self.analysis_method == 'lorentzian_manual' and (not np.isfinite(f.center_tolerance_ppm) or f.center_tolerance_ppm <= 0):
                raise ValueError('Допуск смещения ручных пиков должен быть положительным, в ppm.')
        if require_regions and (not self.regions or not self.integrals):
            raise ValueError('Сохраните области обработки и интегрирования.')
        names = [r.name for r in self.regions]
        if len(names) != len(set(names)) or any(not n.strip() for n in names):
            raise ValueError('Имена областей обработки должны быть уникальны.')
        ordered = sorted(self.regions, key=lambda r: r.low)
        for r in ordered:
            if not np.isfinite([r.low, r.high]).all() or r.low >= r.high:
                raise ValueError(f'{r.name}: нужны две различные конечные границы.')
            validate_baseline(r.baseline)
            peaks = np.asarray(r.manual_peaks, dtype=float)
            if (peaks.ndim != 1 or not np.isfinite(peaks).all() or len(set(peaks)) != len(peaks)
                    or np.any(peaks <= r.low) or np.any(peaks >= r.high)):
                raise ValueError(f'{r.name}: ручные пики должны быть уникальными и строго внутри области.')
            if require_regions and self.analysis_method == 'lorentzian_manual':
                if not len(peaks):
                    raise ValueError(f'{r.name}: сначала отметьте ручные пики.')
                if len(peaks) > f.max_peaks:
                    raise ValueError(f'{r.name}: число ручных пиков превышает лимит {f.max_peaks}.')
        intnames = [r.name for r in self.integrals]
        if len(intnames) != len(set(intnames)) or any(not n.strip() for n in intnames):
            raise ValueError('Имена интегралов должны быть уникальны.')
        if set(intnames) & {'spectrum_number', 'spectrum_id'}:
            raise ValueError('Имена spectrum_number и spectrum_id зарезервированы для столбцов CSV.')
        parents = {r.name: r for r in self.regions}
        for r in self.integrals:
            validate_color(r.color)
            parent = parents.get(r.parent)
            if (parent is None or not np.isfinite([r.low, r.high]).all()
                    or not parent.low <= r.low < r.high <= parent.high):
                raise ValueError(f'{r.name}: область интегрирования должна лежать внутри {r.parent}.')
        validate_baseline(self.baseline_default)
        numeric = [p.p0, p.p1, p.reference_ppm, p.reference_low, p.reference_high, p.ppm_offset,
                   p.manual_phase_p0, p.manual_phase_p1, p.phase_pivot_fraction]
        if not 0 <= p.phase_pivot_fraction <= 1:
            raise ValueError('Phase pivot must lie inside the spectrum.')
        if self.analysis_method != 'direct':
            numeric += [f.initial_width_hz, f.min_width_hz, f.max_width_hz]
        if not np.isfinite(numeric).all():
            raise ValueError('Настройки должны содержать конечные числа.')

    def save(self, path):
        """Save region bounds and algorithm parameters; no fitted baseline arrays."""
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def from_dict(cls, d):
        if d.get('schema_version') != 1:
            raise ValueError('Неподдерживаемая версия настроек.')
        session = cls(
            processing=ProcessingSettings(**d['processing']), fit=FitSettings(**d['fit']),
            baseline_default=BaselineSettings(**d['baseline_default']),
            appearance=AppearanceSettings(**d.get('appearance', {})),
            analysis_method=d.get('analysis_method', 'lorentzian'),
            alignment=AlignmentSettings(**d.get('alignment', {})),
            regions=[FitRegion(**{**r, 'baseline': BaselineSettings(**r['baseline'])}) for r in d['regions']],
            integrals=[IntegrationRegion(**r) for r in d['integrals']])
        session.validate()
        return session


def validate_baseline(b: BaselineSettings):
    if b.method not in BASELINE_METHODS:
        raise ValueError(f'Неизвестный метод базовой линии: {b.method}')
    if any(isinstance(getattr(b, key), bool) or not isinstance(getattr(b, key), int)
           for key in ('max_iter', 'half_window', 'smooth_half_window', 'poly_order')):
        raise ValueError('Window sizes, polynomial order and iteration limits must be integers.')
    if (not np.isfinite([b.lam, b.p, b.tol]).all() or b.lam <= 0 or not 0 < b.p < 1
            or b.tol <= 0 or b.max_iter < 1 or b.half_window < 1
            or b.smooth_half_window < 0 or b.poly_order < 0):
        raise ValueError('Некорректные параметры базовой линии.')
    validate_parameters(b)


@dataclass
class RawSpectrum:
    fid: np.ndarray
    sf_mhz: float
    sw_hz: float
    carrier_ppm: float
    source: str
    notes: list[str] = field(default_factory=list)


@dataclass
class Spectrum:
    ppm: np.ndarray  # ascending internally; invert only the displayed axis
    intensity: np.ndarray
    complex_spectrum: np.ndarray
    sf_mhz: float
    sw_hz: float
    source: str
    phases: tuple[float, float]
    notes: list[str]
    source_type: str = 'raw'
    processing_metadata: dict = field(default_factory=dict)


def load_raw(path, remove_digital_filter=True) -> RawSpectrum:
    path = Path(path)
    notes = []
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            fid = np.asarray(data['fid'], dtype=complex)
            sf, sw = float(data['sf_mhz']), float(data['sw_hz'])
            carrier = float(data['carrier_ppm'])
    else:
        dic, fid = ng.bruker.read(str(path))
        if np.asarray(fid).ndim != 1:
            raise ValueError('Поддерживаются только одномерные Bruker FID.')
        acq = dic['acqus']
        sf, sw = float(acq['SFO1']), float(acq['SW_h'])
        carrier = float(acq['O1']) / sf
        if remove_digital_filter:
            # Skip only an explicitly absent filter; malformed metadata must not be hidden.
            if float(acq.get('GRPDLY', 0)) > 0 or int(acq.get('DECIM', 1)) > 1:
                fid = ng.bruker.remove_digital_filter(dic, fid)
            else:
                notes.append('Цифровой фильтр отсутствует по метаданным; удаление пропущено.')
        else:
            notes.append('Удаление цифрового фильтра отключено.')
    fid = np.asarray(fid, dtype=complex)
    if fid.ndim != 1 or fid.size < 16 or not np.isfinite(fid).all():
        raise ValueError('Нужен конечный одномерный FID длиной >= 16 точек.')
    if not np.isfinite([sf, sw, carrier]).all() or sf <= 0 or sw <= 0:
        raise ValueError('Неверные SFO1/SW_h/O1 в метаданных.')
    return RawSpectrum(fid, sf, sw, carrier, str(path), notes)


def resolve_pdata_path(path, procno='1'):
    from .pdata import resolve_pdata_path as resolve
    return resolve(path, procno)


def load_pdata(path, procno='1') -> Spectrum:
    from .pdata import load_pdata as load
    return load(path, procno)


def load_source(path, p: ProcessingSettings):
    if p.source_mode == 'pdata':
        return load_pdata(path, p.pdata_procno)
    if p.source_mode == 'raw':
        return load_raw(path, p.remove_digital_filter)
    raise ValueError('Источник спектра должен быть raw или pdata.')


class PhaseState:
    """Thread-safe resolved phases for immutable loaded source objects.

    Keep one state per open spectrum and share it between preview and analysis.
    LB, zero filling, references and baseline settings do not change phase fits.
    ``clear()`` is the explicit request to estimate the same source again.
    Strong source references prevent Python object-id reuse from mixing files.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._entries = {}

    def clear(self):
        with self._lock:
            self._entries.clear()

    def resolve(self, source, settings, estimate):
        key = (id(source), settings.phase, float(settings.p0), float(settings.p1),
               int(settings.phase_peak_width))
        with self._lock:
            cached = self._entries.get(key)
            if cached is None:
                phases, notes = estimate()
                cached = (source, tuple(phases), tuple(notes))
                self._entries[key] = cached
            return cached[1], list(cached[2])


def resolve_phase_angles(source, p, phase_cache=None, reference=None):
    """Estimate auto phases from the original, unapodized frequency data only."""
    if p.phase not in ('acme', 'peak_minima'):
        return (0., 0.), []
    def estimate():
        data = reference
        if data is None:
            data = (source.complex_spectrum if isinstance(source, Spectrum)
                    else ng.proc_base.fft(source.fid))
        data = np.asarray(data)
        counts = {'evaluations': 0, 'iterations': 0}
        peak_width = min(p.phase_peak_width, max(1, len(data)//4))
        def score(phases, values):
            counts['evaluations'] += 1
            if p.phase == 'acme':
                return ng.proc_autophase._ps_acme_score(phases, values)
            try:
                return ng.proc_autophase._ps_peak_minima_score(phases, values, peak_width)
            except ValueError as exc:
                raise ValueError('peak_minima: недостаточно точек около максимума. '
                                 'Уменьшите окно peak_minima или выберите ACME.') from exc
        def iteration(_):
            counts['iterations'] += 1
        with warnings.catch_warnings(record=True) as caught:
            _, optimized = ng.proc_autophase.autops(
                data, score, p0=p.p0, p1=p.p1, return_phases=True,
                disp=False, maxiter=1000, maxfun=2000, ftol=1e-6, callback=iteration)
        phases = tuple(float(v) for v in optimized)
        if len(phases) != 2 or not np.isfinite(phases).all():
            raise ValueError('Автокоррекция фазы вернула некорректные значения.')
        notes = [str(w.message) for w in caught]
        if counts['evaluations'] >= 2000 or counts['iterations'] >= 999:
            notes.append('Автокоррекция фазы достигла лимита оптимизации; проверьте фазу вручную.')
        return phases, notes
    return phase_cache.resolve(source, p, estimate) if phase_cache is not None else estimate()


def process(raw: RawSpectrum | Spectrum, p: ProcessingSettings, phase_cache=None) -> Spectrum:
    Session(processing=p).validate()
    if isinstance(raw, Spectrum):
        from .pdata import process_pdata
        return process_pdata(raw, p, phase_cache=phase_cache)
    # Phase the original frequency data first. First-order phase and a
    # time-domain window do not commute, so applying fixed angles *after*
    # apodization would still change the result of this ordered workflow.
    if p.phase in ('acme', 'peak_minima') or has_manual_phase(p):
        base = ng.proc_base.fft(raw.fid)
        phases, resolved_notes = resolve_phase_angles(raw, p, phase_cache, reference=base)
        phases = add_manual_phase(phases, p)
        ft = process_frequency_data(base, p, raw.sw_hz, phases)
    else:
        phases, resolved_notes = (0., 0.), []
        fid = apodize_fid(raw.fid.copy(), p, raw.sw_hz)
        ft = ng.proc_base.fft(fid)
    notes = list(raw.notes) + resolved_notes
    return spectrum_from_fourier(ft, p, raw.sf_mhz, raw.sw_hz, raw.carrier_ppm,
                                 raw.source, phases, notes)


def apodize_fid(fid, p, sw_hz):
    """Apply the requested time-domain window and padding, without phasing."""
    original_size = len(fid)
    # nmrglue expects normalized LB; the UI and JSON use physical Hz.
    if p.lb_hz > 0:
        fid = ng.proc_base.em(fid, lb=p.lb_hz / sw_hz)
    if p.zero_fill_factor > 1:
        fid = ng.proc_base.zf_size(fid, original_size * p.zero_fill_factor)
    return fid


def has_manual_phase(p):
    return p.phase != 'abs' and bool(p.manual_phase_p0 or p.manual_phase_p1)


def add_manual_phase(phases, p):
    if p.phase == 'abs':
        return phases
    return (phases[0]+p.manual_phase_p0-p.manual_phase_p1*p.phase_pivot_fraction,
            phases[1]+p.manual_phase_p1)


def process_frequency_data(base, p, sw_hz, phases, *, positive=False):
    """Phase original frequency data before any additional apodization."""
    if p.phase in ('acme', 'peak_minima') or has_manual_phase(p):
        phased = ng.proc_base.ps(base, p0=phases[0], p1=phases[1])
    else:
        phased = np.asarray(base).copy()
    if p.lb_hz == 0 and p.zero_fill_factor == 1:
        return phased
    inverse = ng.proc_base.ifft_positive if positive else ng.proc_base.ifft
    forward = ng.proc_base.fft_positive if positive else ng.proc_base.fft
    fid = inverse(phased)
    return forward(apodize_fid(fid, p, sw_hz))


def spectrum_from_fourier(ft, p, sf_mhz, sw_hz, carrier_ppm, source, phases, notes):
    """Construct the shared raw/pdata output after all spectral operations."""
    notes = list(notes)
    if p.phase == 'abs':
        y = np.abs(ft)
        notes.append('abs спектра: форма magnitude-пика отличается от чистого Лоренца.')
    else:
        y = ft.real
    # ng.fft implements fftshift(fft). The original notebook assigns increasing
    # frequencies to these bins. Keep that orientation, with exact bin spacing
    # instead of endpoint-inclusive linspace; sort/invert only for display.
    frequencies = np.fft.fftshift(np.fft.fftfreq(len(ft), d=1/sw_hz))
    ppm = carrier_ppm + frequencies / sf_mhz
    ppm, reference_notes, reference_info = reference_axis(ppm, ft, p, intensity=y)
    reference_info.update(manual_phase_p0_deg=0. if p.phase == 'abs' else p.manual_phase_p0,
                          manual_phase_p1_deg=0. if p.phase == 'abs' else p.manual_phase_p1,
                          phase_pivot_fraction=p.phase_pivot_fraction)
    notes.extend(reference_notes)
    order = np.argsort(ppm)
    if not np.isfinite(y).all():
        raise ValueError('После обработки появились нечисловые значения.')
    return Spectrum(ppm[order], np.asarray(y)[order], ft[order], sf_mhz,
                    sw_hz, source, phases, notes, processing_metadata=reference_info)


def window_reference_index(ppm, intensity, low, high):
    """Locate the selected absorption/magnitude signal above a local linear background."""
    indexes = np.flatnonzero((ppm >= low) & (ppm <= high))
    if not len(indexes):
        raise ValueError('Окно референса находится вне спектра.')
    if len(indexes) < 3:
        raise ValueError('Окно референса должно содержать не менее трёх точек.')
    x, y = ppm[indexes], np.asarray(intensity)[indexes]
    background = y[0] + (y[-1]-y[0])*(x-x[0])/(x[-1]-x[0])
    return int(indexes[np.argmax(np.abs(y-background))])


def reference_axis(ppm, values, p, intensity=None):
    """Apply the same saved reference rule independently to each spectrum."""
    ppm = np.asarray(ppm)
    notes, shift, observed = [], 0., None
    if p.reference_mode != 'metadata':
        if p.reference_mode == 'window':
            index = window_reference_index(ppm, np.abs(values) if intensity is None else intensity,
                                           p.reference_low, p.reference_high)
        else:
            # Preserve the legacy strongest-magnitude rule outside a chosen window.
            index = int(np.argmax(np.abs(values)))
        observed = float(ppm[index])
        shift = p.reference_ppm - observed
        if p.reference_mode == 'strongest':
            notes.append('Референс = самый сильный пик: проверяйте идентичность сигнала в серии.')
    info = dict(reference_mode=p.reference_mode, reference_observed_ppm=observed,
                reference_target_ppm=p.reference_ppm if observed is not None else None,
                reference_shift_ppm=float(shift+p.ppm_offset))
    return ppm + shift + p.ppm_offset, notes, info


def baseline_correct(x, y, b: BaselineSettings):
    validate_baseline(b)
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 8 or not np.all(np.diff(x) > 0):
        raise ValueError('Область должна содержать >= 8 точек с возрастающей ppm-шкалой.')
    if b.method == 'none':
        return y.copy(), np.zeros_like(y), []
    fitter = Baseline(x_data=x, check_finite=True)
    # Baseline algorithms operate on normalized intensities for numeric stability.
    scale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    with warnings.catch_warnings(record=True) as caught:
        line, params = fit_baseline(fitter, x, y / scale, b)
    line = np.asarray(line) * scale
    if line.shape != y.shape or not np.isfinite(line).all():
        raise ValueError('Коррекция базовой линии вернула нечисловые значения.')
    notes = [str(w.message) for w in caught]
    history = params.get('tol_history')
    tolerance = effective_parameters(b).get('tol', parameter_specs(b.method).get('tol'))
    if tolerance is None:
        tolerance = parameter_specs(b.method).get('tol')
    if history is not None and len(history) and tolerance is not None and np.asarray(history).ravel()[-1] > tolerance:
        notes.append(f'{b.method}: базовая линия не достигла tol={b.tol:g}.')
    return y - line, line, notes


def pick_peaks(x, y, f: FitSettings):
    locations = ng.peakpick.pick(np.asarray(y), pthres=f.threshold,
                            nthres=-f.threshold if f.signed else None,
                            algorithm='downward', msep=[f.min_separation_points],
                            est_params=False, cluster=False, table=False)
    if not locations:
        return np.array([], dtype=float), np.array([], dtype=float)
    indices = np.unique(np.clip(np.rint(np.asarray(locations)[:, 0]).astype(int), 0, len(x)-1))
    # Downward picking does not apply msep in all nmrglue versions. Explicitly
    # enforce it by retaining the strongest local extrema first.
    chosen = []
    for i in sorted(indices, key=lambda i: abs(y[i]), reverse=True):
        if all(abs(i-j) >= f.min_separation_points for j in chosen):
            chosen.append(i)
    indices = np.array(sorted(chosen), dtype=int)
    return np.asarray(x)[indices], np.asarray(y)[indices]


def lorentzian(x, center, width, amplitude):
    return amplitude * width**2 / ((np.asarray(x) - center)**2 + width**2)


def manual_peak_guesses(x, corrected, region):
    """Use the saved ppm marks; sample fresh baseline-corrected heights per trace."""
    centers = np.sort(np.asarray(region.manual_peaks, dtype=float))
    return centers, np.interp(centers, x, corrected) if len(centers) else np.array([], dtype=float)


@dataclass
class FitResult:
    region: str
    x: np.ndarray
    raw: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    centers: np.ndarray
    initial_amplitudes: np.ndarray
    amplitudes: np.ndarray
    widths: np.ndarray
    fitted: np.ndarray
    success: bool
    notes: list[str]
    diagnostics: dict

    def components(self):
        return lorentzian(self.x[:, None], self.centers[None, :],
                          self.widths[None, :], self.amplitudes[None, :])


def cancellation_diagnostics(components, centers, amplitudes, limit=.7):
    """Flag opposite-sign overlapping components, with pair IDs and scores."""
    pairs = []
    for i in range(len(amplitudes)):
        for j in range(i+1, len(amplitudes)):
            if amplitudes[i] * amplitudes[j] >= 0:
                continue
            a, b = components[:, i], components[:, j]
            denom = float(np.sum(np.abs(a) + np.abs(b)))
            if denom == 0:
                continue
            cancelled = 1 - float(np.sum(np.abs(a+b))) / denom
            if cancelled >= limit:
                pairs.append(dict(peak1=i+1, peak2=j+1, ppm1=float(centers[i]),
                                  ppm2=float(centers[j]), cancellation=cancelled))
    return pairs


def fit_region(spectrum: Spectrum, region: FitRegion, settings: Session) -> FitResult:
    xall = spectrum.ppm
    if region.low < xall[0] or region.high > xall[-1]:
        raise ValueError(f'{region.name}: границы области выходят за ppm-диапазон спектра.')
    mask = (xall >= region.low) & (xall <= region.high)
    x, y = xall[mask], spectrum.intensity[mask]
    corrected, line, notes = baseline_correct(x, y, region.baseline)
    f = settings.fit
    manual = settings.analysis_method == 'lorentzian_manual'
    centers, picked = manual_peak_guesses(x, corrected, region) if manual else pick_peaks(x, corrected, f)
    n = len(centers)
    if n > f.max_peaks:
        raise ValueError(f'{region.name}: найдено {n} пиков (лимит {f.max_peaks}). '
                         'Повысьте threshold/разделение или сузьте область.')
    if not n:
        if manual:
            raise ValueError(f'{region.name}: сначала отметьте ручные пики.')
        return FitResult(region.name, x, y, line, corrected, centers, picked,
                         np.array([]), np.array([]), np.zeros_like(y), True,
                         notes + ['Пики выше threshold не найдены; интеграл = 0.'],
                         {'n_peaks': 0, 'nfev': 0, 'relative_rmse': None})
    parameter_count = (3 if manual else 2)*n
    if len(x) <= parameter_count:
        raise ValueError(f'{region.name}: недостаточно точек для {parameter_count} параметров.')
    scale = max(float(np.max(np.abs(corrected))), np.finfo(float).tiny)
    target, initial_amp = corrected / scale, picked / scale
    if not f.signed:
        initial_amp = np.maximum(initial_amp, 1e-6)
        if manual and np.any(picked <= 0):
            notes.append('Начальная высота ручного пика неположительна; проверьте фазу, базовую линию или разрешите отрицательные амплитуды.')
    spacing = float(np.median(np.diff(x)))
    # A width appreciably below digitization resolution is unidentifiable.
    lower_w = max(f.min_width_hz / (2*spectrum.sf_mhz), spacing * .25)
    upper_w = f.max_width_hz / (2*spectrum.sf_mhz)
    if upper_w <= lower_w:
        raise ValueError('Максимальная ширина меньше цифрового разрешения спектра.')
    initial_w = max(settings.processing.lb_hz, f.initial_width_hz) / (2*spectrum.sf_mhz)
    initial_w = np.clip(initial_w, lower_w*1.01, upper_w*.99)
    seed_centers = centers.copy()
    center_unit = f.center_tolerance_ppm if manual else 1.
    if manual:
        # Midpoints keep component identities ordered and prevent center swaps.
        middle = (centers[:-1]+centers[1:])/2
        center_low = np.maximum(centers-center_unit, np.r_[region.low, middle])
        center_high = np.minimum(centers+center_unit, np.r_[middle, region.high])
    fixed_distances = None if manual else (x[:, None]-seed_centers[None, :])**2
    def model_jac(params, jac=False):
        amp, widths = params[:n], np.exp(params[n:2*n])
        if manual:
            current_centers = seed_centers+params[2*n:]*center_unit
            delta = x[:, None]-current_centers[None, :]
            distances = delta**2
        else:
            distances = fixed_distances
        w2 = widths**2
        profile = w2 / (distances + w2)
        if jac:
            dlogw = amp * 2*w2 * distances / (distances + w2)**2
            columns = [profile, dlogw]
            if manual:
                columns.append(amp*2*w2*delta/(distances+w2)**2*center_unit)
            return np.concatenate(columns, axis=1)
        return profile @ amp
    lower_amp = np.full(n, -np.inf if f.signed else 0.)
    lower = np.r_[lower_amp, np.full(n, np.log(lower_w))]
    upper = np.r_[np.full(n, np.inf), np.full(n, np.log(upper_w))]
    if manual:
        lower = np.r_[lower, (center_low-seed_centers)/center_unit]
        upper = np.r_[upper, (center_high-seed_centers)/center_unit]
    candidates = []
    failures = []
    for attempt in range(f.restarts):
        factor = (1., .5, 2., .25, 4.)[attempt % 5]
        width0 = np.clip(initial_w*factor, lower_w*1.001, upper_w*.999)
        start = np.r_[initial_amp, np.full(n, np.log(width0))]
        if manual:
            start = np.r_[start, np.zeros(n)]
        try:
            result = least_squares(lambda z: model_jac(z)-target, start,
                                   jac=lambda z: model_jac(z, True), bounds=(lower, upper),
                                   method='trf', x_scale='jac', max_nfev=f.max_nfev,
                                   tr_solver='lsmr' if n >= 20 else 'exact',
                                   loss=f.loss, f_scale=.03, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            candidates.append(result)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append(str(exc))
    if not candidates:
        raise ValueError('Все попытки подгонки завершились ошибкой: ' + '; '.join(failures))
    good = [r for r in candidates if r.success and np.isfinite(r.x).all()]
    result = min(good or candidates, key=lambda r: r.cost)
    amplitudes = result.x[:n] * scale
    widths = np.exp(result.x[n:2*n])
    if manual:
        centers = seed_centers+result.x[2*n:]*center_unit
    fitted = model_jac(result.x) * scale
    components = lorentzian(x[:, None], centers[None, :], widths[None, :], amplitudes[None, :])
    cancellation = cancellation_diagnostics(components, centers, amplitudes, f.cancellation_warning)
    rmse = float(np.sqrt(np.mean((target - fitted/scale)**2)))
    absolute_sum = np.sum(np.abs(components), axis=1)
    total_cancel = 1 - float(np.sum(np.abs(fitted))) / max(float(np.sum(absolute_sum)), np.finfo(float).tiny)
    jac = model_jac(result.x, True)
    # Normalize columns so condition reflects overlap, not amplitude units.
    norms = np.linalg.norm(jac, axis=0)
    cond = float(np.linalg.cond(jac / np.maximum(norms, np.finfo(float).tiny)))
    diagnostics = dict(n_peaks=n, nfev=int(result.nfev), attempts=len(candidates),
                       solver_message=str(result.message), relative_rmse=rmse,
                       condition_number=cond if np.isfinite(cond) else None, cancellation_fraction=total_cancel,
                       cancellation_pairs=cancellation,
                       initial_width_ppm=float(initial_w), width_bounds_ppm=[lower_w, upper_w])
    if manual:
        diagnostics.update(method='lorentzian_manual', initial_centers_ppm=seed_centers.tolist(),
                           center_bounds_ppm=np.column_stack([center_low, center_high]).tolist(),
                           center_shifts_ppm=(centers-seed_centers).tolist(),
                           center_tolerance_ppm=f.center_tolerance_ppm)
        if np.any(np.minimum(centers-center_low, center_high-centers) < .02*(center_high-center_low)):
            notes.append('Центр ручного пика достиг границы допуска; проверьте разметку и выравнивание.')
    if not result.success:
        notes.append('ОПТИМИЗАЦИЯ НЕ СОШЛАСЬ: ' + str(result.message))
    if rmse > f.residual_warning:
        notes.append(f'Большой остаток: относительный RMSE={rmse:.3g}.')
    if cond > 1e6:
        notes.append(f'Параметры плохо различимы: condition={cond:.3g}.')
    if np.any((widths < lower_w*1.02) | (widths > upper_w*.98)):
        notes.append('Одна или несколько ширин достигли границы ограничений.')
    if np.max(np.abs(amplitudes)) > scale*5:
        notes.append('Амплитуда компоненты превышает максимум данных более чем в 5 раз.')
    if cancellation:
        notes.append('Взаимная компенсация противоположных линий: ' +
                     ', '.join(f"{r['peak1']} / {r['peak2']} ({r['cancellation']:.0%})" for r in cancellation))
    if total_cancel >= f.cancellation_warning:
        notes.append(f'Суммарная компенсация линий {total_cancel:.0%}.')
    return FitResult(region.name, x, y, line, corrected, centers, picked, amplitudes,
                     widths, fitted, bool(result.success), notes, diagnostics)


@dataclass
class Analysis:
    spectrum: Spectrum
    results: dict[str, FitResult]
    integrals: dict[str, float]
    errors: dict[str, str]
    status: str
    notes: list[str]
    method: str = 'lorentzian'


def integration_segment(x, y, low, high):
    """Extract a linear-interpolated interval including its exact endpoints."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Для интегрирования нужны конечные одномерные массивы одинаковой длины.')
    differences = np.diff(x)
    if np.all(differences < 0):
        x, y = x[::-1], y[::-1]
    elif not np.all(differences > 0):
        raise ValueError('Шкала интегрирования должна быть строго монотонной без повторов.')
    low, high = float(low), float(high)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError('Для интегрирования нужны две различные границы low < high.')
    if low < x[0] or high > x[-1]:
        raise ValueError('Область интегрирования выходит за диапазон спектра.')
    inner = (x > low) & (x < high)
    return np.r_[low, x[inner], high], np.r_[np.interp(low, x, y), y[inner], np.interp(high, x, y)]


def integrate_region(x, y, low, high):
    """Signed trapezoidal area in intensity*ppm, independent of ppm direction."""
    xx, yy = integration_segment(x, y, low, high)
    return float(np.trapezoid(yy, xx))


def direct_region(spectrum: Spectrum, region: FitRegion) -> FitResult:
    """Estimate a fresh local baseline from this spectrum, then integrate its data."""
    x, y = integration_segment(spectrum.ppm, spectrum.intensity, region.low, region.high)
    if region.baseline.method == 'none':
        corrected, line, notes = y.copy(), np.zeros_like(y), []
    else:
        corrected, line, notes = baseline_correct(x, y, region.baseline)
    empty = np.array([], dtype=float)
    return FitResult(region.name, x, y, line, corrected, empty.copy(), empty.copy(),
                     empty.copy(), empty.copy(), corrected.copy(), True, notes,
                     dict(method='direct', n_peaks=0, nfev=0, relative_rmse=None,
                          integral_units='intensity*ppm', baseline_method=region.baseline.method,
                          baseline_parameters=asdict(region.baseline)))


def analyze(spectrum: Spectrum, session: Session, region_spectra=None, region_errors=None) -> Analysis:
    session.validate(require_regions=True)
    results, errors = {}, dict(region_errors or {})
    notes = list(spectrum.notes)
    notes.extend(f'{name}: {message}' for name, message in errors.items())
    for region in session.regions:
        if region.name in errors:
            continue
        try:
            source = (region_spectra or {}).get(region.name, spectrum)
            result = (direct_region(source, region) if session.analysis_method == 'direct'
                      else fit_region(source, region, session))
            if source.processing_metadata.get('alignment'):
                result.diagnostics['alignment'] = source.processing_metadata['alignment']
            results[region.name] = result
            notes.extend(f'{region.name}: {n}' for n in result.notes)
            if not result.success:
                errors[region.name] = result.diagnostics['solver_message']
        except Exception as exc:
            errors[region.name] = str(exc)
            notes.append(f'{region.name}: ОШИБКА: {exc}')
    integrals = {}
    for area in session.integrals:
        result = results.get(area.parent)
        if result is None or area.parent in errors:
            integrals[area.name] = float('nan')
        else:
            if session.analysis_method == 'direct':
                integrals[area.name] = integrate_region(result.x, result.corrected, area.low, area.high)
            else:
                keep = (result.centers >= area.low) & (result.centers <= area.high)
                integrals[area.name] = float(np.sum(result.amplitudes[keep] * result.widths[keep]))
    return Analysis(spectrum, results, integrals, errors,
                    'error' if errors else 'warning' if notes else 'ok', notes, session.analysis_method)


def discover_spectra(root, recursive=True, source_mode='raw', pdata_procno='1'):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('Выберите существующую папку серии.')
    found = set()
    if source_mode == 'pdata':
        procno = str(pdata_procno)
        if not procno.isdigit() or int(procno) < 1:
            raise ValueError('Номер обработки pdata должен быть положительным целым числом.')
        # A directly chosen processed directory is unambiguous, even if its
        # processing number differs from the dataset default.
        if (root / '1r').is_file() and (root / 'procs').is_file():
            found.add(root)
        elif root.name.lower() == 'pdata':
            candidate = root / procno
            if (candidate / '1r').is_file() and (candidate / 'procs').is_file():
                found.add(candidate)
        else:
            candidates = [root / 'pdata' / procno]
            candidates.extend(root.glob(f'**/pdata/{procno}' if recursive else f'*/pdata/{procno}'))
            found.update(candidate for candidate in candidates
                         if (candidate / '1r').is_file() and (candidate / 'procs').is_file())
    elif source_mode == 'raw':
        pattern = '**/fid' if recursive else '*/fid'
        for fid in root.glob(pattern):
            if (fid.parent / 'acqus').is_file():
                found.add(fid.parent)
        if (root / 'fid').is_file() and (root / 'acqus').is_file():
            found.add(root)
        found.update(root.glob('**/*.npz' if recursive else '*.npz'))
    else:
        raise ValueError('Источник спектра должен быть raw или pdata.')
    def key(path):
        # Natural sorting, preserving unique relative paths as spectrum IDs.
        import re
        return [(0, int(s)) if s.isdigit() else (1, s.casefold())
                for s in re.split(r'(\d+)', path.relative_to(root).as_posix())]
    return sorted(found, key=key)


def write_batch_csv(path, rows, session: Session):
    """One numeric column per named area, with no labels embedded in values."""
    session.validate()
    fields = ['spectrum_number', 'spectrum_id'] + [r.name for r in session.integrals]
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for row in rows:
            writer.writerow([row['spectrum_number'], row['spectrum_id']]
                            + [row[f'integral__{r.name}'] for r in session.integrals])


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_analysis(path, analysis: Analysis, session: Session):
    """Save the actual fitted arrays in a portable, non-pickle .spfit archive."""
    session.validate(require_regions=True)
    s = analysis.spectrum
    metadata = dict(format='spintegrate-fit', version=1, session=asdict(session),
                    spectrum=dict(sf_mhz=s.sf_mhz, sw_hz=s.sw_hz, source=s.source,
                                  phases=s.phases, notes=s.notes, source_type=s.source_type,
                                  processing_metadata=s.processing_metadata),
                    integrals=analysis.integrals, errors=analysis.errors,
                    status=analysis.status, notes=analysis.notes, method=analysis.method, regions=[])
    arrays = dict(spectrum_ppm=s.ppm, spectrum_intensity=s.intensity,
                  spectrum_complex=s.complex_spectrum)
    for index, (name, result) in enumerate(analysis.results.items()):
        key = f'r{index}'
        metadata['regions'].append(dict(key=key, name=name, success=result.success,
                                        notes=result.notes, diagnostics=result.diagnostics))
        for attr in ('x', 'raw', 'baseline', 'corrected', 'centers', 'initial_amplitudes',
                     'amplitudes', 'widths', 'fitted'):
            arrays[f'{key}_{attr}'] = getattr(result, attr)
    arrays['metadata'] = np.array(json.dumps(_json_safe(metadata), ensure_ascii=False, allow_nan=False))
    # A stream preserves the .spfit suffix (numpy otherwise appends .npz).
    with Path(path).open('wb') as stream:
        np.savez_compressed(stream, **arrays)


def load_analysis(path):
    """Return (Analysis, Session) without accessing the original FID or refitting."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata'].item()))
            if metadata.get('format') != 'spintegrate-fit' or metadata.get('version') != 1:
                raise ValueError('Неподдерживаемый формат файла подгонки.')
            session = Session.from_dict(metadata['session'])
            session.validate(require_regions=True)
            def array(key, complex_ok=False):
                value = np.asarray(archive[key])
                if (value.ndim != 1 or value.dtype.kind not in ('f', 'i', 'u', 'c')
                        or (not complex_ok and np.iscomplexobj(value)) or not np.isfinite(value).all()):
                    raise ValueError(f'Повреждён массив {key} в файле подгонки.')
                return value.copy()
            data = metadata['spectrum']
            ppm, y = array('spectrum_ppm'), array('spectrum_intensity')
            ft = array('spectrum_complex', complex_ok=True)
            if len(ppm) < 2 or len(y) != len(ppm) or len(ft) != len(ppm) or not np.all(np.diff(ppm) > 0):
                raise ValueError('Повреждена шкала спектра в файле подгонки.')
            spectrum = Spectrum(ppm, y, ft, data['sf_mhz'], data['sw_hz'],
                                data['source'], tuple(data['phases']), data['notes'],
                                data.get('source_type', 'raw'), data.get('processing_metadata', {}))
            results = {}
            region_names = {r.name for r in session.regions}
            for entry in metadata['regions']:
                key, name = entry['key'], entry['name']
                if name not in region_names or name in results:
                    raise ValueError('Неверный список областей в файле подгонки.')
                attrs = {attr: array(f'{key}_{attr}') for attr in
                         ('x', 'raw', 'baseline', 'corrected', 'centers', 'initial_amplitudes',
                          'amplitudes', 'widths', 'fitted')}
                minimum = 2 if entry.get('diagnostics', {}).get('method') == 'direct' else 8
                if (len(attrs['x']) < minimum or not np.all(np.diff(attrs['x']) > 0)
                        or any(len(attrs[a]) != len(attrs['x']) for a in ('raw', 'baseline', 'corrected', 'fitted'))
                        or any(len(attrs[a]) != len(attrs['centers']) for a in ('initial_amplitudes', 'amplitudes', 'widths'))
                        or np.any(attrs['widths'] <= 0)):
                    raise ValueError(f'Несогласованные массивы области {name}.')
                results[name] = FitResult(name, **attrs, success=entry['success'],
                                          notes=entry['notes'], diagnostics=entry['diagnostics'])
            integrals = {name: float('nan') if v is None else float(v)
                         for name, v in metadata['integrals'].items()}
            analysis = Analysis(spectrum, results, integrals, metadata['errors'],
                                metadata['status'], metadata['notes'],
                                metadata.get('method', session.analysis_method))
            return analysis, session
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Не удалось прочитать файл подгонки: {exc}') from exc


def load_batch_results(csv_path):
    """Reopen current or legacy CSV + sidecars; resolve fit files relative to CSV."""
    csv_path = Path(csv_path).resolve()
    settings_path = csv_path.with_suffix('.settings.json')
    if not settings_path.is_file():
        raise ValueError(f'Для просмотра CSV нужен сохранённый рецепт рядом с ним: {settings_path.name}. '
                         'Можно также открыть отдельный файл .spfit без CSV и JSON.')
    session = Session.load(settings_path)
    session.validate(require_regions=True)
    report_path = csv_path.with_suffix('.diagnostics.json')
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    details = report.get('spectra', [])
    by_id = {r['spectrum_id']: r for r in details}
    rows = []
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not {'spectrum_number', 'spectrum_id'} <= set(fields):
            raise ValueError('В CSV отсутствуют столбцы spectrum_number / spectrum_id.')
        legacy = all(f'integral__{r.name}' in fields for r in session.integrals)
        for number, source in enumerate(reader, 1):
            extra = by_id.get(source['spectrum_id'], {})
            row = dict(extra.get('row', {}))
            # Legacy files contain technical columns directly in the table.
            technical = {'sequence_index', 'spectrum_number', 'spectrum_id', 'source_path',
                         'status', 'warnings', 'phase_p0_deg', 'phase_p1_deg', 'peak_count', 'max_relative_rmse'}
            row.update({k: v for k, v in source.items()
                        if k in (technical if legacy else {'spectrum_number', 'spectrum_id'})})
            row.setdefault('sequence_index', number)
            row.setdefault('status', 'unknown')
            row.setdefault('warnings', 'Подробная диагностика недоступна.' if not extra else '')
            row.setdefault('source_path', '')
            for area in session.integrals:
                column = f'integral__{area.name}' if legacy else area.name
                try:
                    row[f'integral__{area.name}'] = float(source[column])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f'Неверное значение интеграла «{area.name}» в строке {number}.') from exc
            fit_file = extra.get('fit_file') or row.get('fit_file', '')
            row['fit_file'] = str((csv_path.parent / fit_file).resolve()) if fit_file else ''
            if row['source_path'] and not Path(row['source_path']).is_absolute():
                candidate = csv_path.parent / row['source_path']
                if candidate.exists():
                    row['source_path'] = str(candidate.resolve())
            rows.append(row)
    return rows, session


def batch_analyze(root, session: Session, csv_path, progress: Callable | None = None,
                  cancelled: Callable | None = None, recursive=True):
    if session.analysis_method == 'lorentzian_manual':
        session = copy.deepcopy(session)
        session.alignment.enabled = True
    session.validate(require_regions=True)
    paths = discover_spectra(root, recursive, session.processing.source_mode, session.processing.pdata_procno)
    if not paths:
        if session.processing.source_mode == 'pdata':
            raise ValueError(f'Не найдены 1r/procs для обработки pdata/{session.processing.pdata_procno}.')
        raise ValueError('Не найдены папки с fid + acqus или демонстрационные .npz.')
    root = Path(root)
    csv_path = Path(csv_path)
    reference, first_alignment, reference_id = None, None, ''
    if session.alignment.enabled:
        reference_id = paths[0].relative_to(root).as_posix()
        try:
            reference = process(load_source(paths[0], session.processing), session.processing)
            if session.alignment.mode == 'global':
                aligned, info = align_spectrum(reference, reference, session.alignment, reference_id, first=True)
                first_alignment = (aligned, None, {}, info)
            else:
                first_alignment = align_regions(reference, reference, session, reference_id, first=True)
                if len(first_alignment[2]) == len(session.regions):
                    raise ValueError('No processing region in the first spectrum can be aligned.')
        except Exception as exc:
            raise ValueError(f'Cannot align to the first spectrum ({reference_id}): {exc}') from exc
    # Save the exact recipe alongside the table for reproducibility.
    session.save(csv_path.with_suffix('.settings.json'))
    fit_directory = csv_path.with_suffix('.fits')
    fit_directory.mkdir(parents=True, exist_ok=True)
    rows, report = [], []
    was_cancelled = False
    for number, path in enumerate(paths, 1):
        if cancelled and cancelled():
            was_cancelled = True
            break
        ident = path.relative_to(root).as_posix()
        experiment = path.parent.parent if session.processing.source_mode == 'pdata' and path.parent.name.lower() == 'pdata' else path
        row = dict(sequence_index=number, spectrum_number=experiment.stem if experiment.is_file() else experiment.name,
                   spectrum_id=ident, source_path=str(path.resolve()), fit_file='',
                   analysis_method=session.analysis_method, source_mode=session.processing.source_mode,
                   status='error', warnings='', phase_p0_deg='', phase_p1_deg='',
                   peak_count='', max_relative_rmse='', reference_shift_ppm='',
                   alignment_mode=session.alignment.mode if session.alignment.enabled else 'off',
                   alignment_reference=reference_id, alignment_shift_ppm='', alignment_correlation='',
                   alignment_max_abs_shift_ppm='')
        row.update({f'integral__{r.name}': float('nan') for r in session.integrals})
        details = dict(spectrum_id=ident, errors={}, diagnostics={})
        try:
            region_views, alignment_errors = None, {}
            if session.alignment.enabled and number == 1:
                spectrum, region_views, alignment_errors, alignment_info = first_alignment
            else:
                spectrum = process(load_source(path, session.processing), session.processing)
                if session.alignment.enabled:
                    if session.alignment.mode == 'global':
                        spectrum, alignment_info = align_spectrum(reference, spectrum, session.alignment, reference_id)
                    else:
                        spectrum, region_views, alignment_errors, alignment_info = align_regions(reference, spectrum, session, reference_id)
            if session.alignment.enabled:
                details['alignment'] = alignment_info
                if session.alignment.mode == 'global':
                    row.update(alignment_shift_ppm=alignment_info['shift_ppm'],
                               alignment_correlation=alignment_info['correlation'],
                               alignment_max_abs_shift_ppm=abs(alignment_info['shift_ppm']))
                else:
                    valid = [info for info in alignment_info.values() if 'shift_ppm' in info]
                    row.update(alignment_correlation=min((info['correlation'] for info in valid), default=''),
                               alignment_max_abs_shift_ppm=max((abs(info['shift_ppm']) for info in valid), default=''))
            analysis = analyze(spectrum, session, region_views, alignment_errors)
            row.update({f'integral__{n}': v for n, v in analysis.integrals.items()})
            row.update(status=analysis.status, warnings=' | '.join(analysis.notes),
                       phase_p0_deg=analysis.spectrum.phases[0], phase_p1_deg=analysis.spectrum.phases[1],
                       reference_shift_ppm=analysis.spectrum.processing_metadata.get('reference_shift_ppm', 0.),
                       peak_count=sum(len(r.centers) for r in analysis.results.values()))
            rmses = [r.diagnostics.get('relative_rmse') for r in analysis.results.values()]
            row['max_relative_rmse'] = max((v for v in rmses if v is not None), default='')
            details.update(errors=analysis.errors,
                           diagnostics={n: r.diagnostics for n, r in analysis.results.items()})
            try:
                fit_path = fit_directory / f'{number:06d}.spfit'
                save_analysis(fit_path, analysis, session)
                row['fit_file'] = str(fit_path.resolve())
                details['fit_file'] = fit_path.relative_to(csv_path.parent).as_posix()
            except Exception as exc:
                if row['status'] == 'ok':
                    row['status'] = 'warning'
                row['warnings'] += f' | Не удалось сохранить файл подгонки: {exc}'
        except Exception as exc:
            row['warnings'] = str(exc)
            details['errors'] = {'spectrum': str(exc)}
        details['row'] = {k: v for k, v in row.items() if not k.startswith('integral__') and k != 'fit_file'}
        report.append(details)
        rows.append(row)
        if progress:
            progress(number, len(paths), row)
    write_batch_csv(csv_path, rows, session)
    csv_path.with_suffix('.diagnostics.json').write_text(
        json.dumps(_json_safe(dict(cancelled=was_cancelled, processed=len(rows), total=len(paths), spectra=report)),
                   ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return rows, was_cancelled
