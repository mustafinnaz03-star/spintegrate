"""Bounded registration to the first batch spectrum without resampling quantitative data."""
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, detrend


@dataclass
class AlignmentSettings:
    enabled: bool = False
    mode: str = 'global'  # global or regions
    max_shift_ppm: float = .1
    low_ppm: float | None = None
    high_ppm: float | None = None
    min_correlation: float = .6

    def validate(self):
        if not isinstance(self.enabled, bool) or self.mode not in ('global', 'regions'):
            raise ValueError('Invalid spectrum alignment mode.')
        if not np.isfinite(self.max_shift_ppm) or self.max_shift_ppm <= 0:
            raise ValueError('Maximum alignment drift must be positive, in ppm.')
        if not np.isfinite(self.min_correlation) or not 0 <= self.min_correlation <= 1:
            raise ValueError('Minimum alignment correlation must be between 0 and 1.')
        if (self.low_ppm is None) != (self.high_ppm is None):
            raise ValueError('Enter both alignment window bounds, or leave both blank.')
        if self.low_ppm is not None and (not np.isfinite([self.low_ppm, self.high_ppm]).all() or self.low_ppm >= self.high_ppm):
            raise ValueError('Alignment window must have finite bounds low < high.')


def _features(values):
    # Registration-only background removal and amplitude compression. Quantitative
    # intensities are never replaced by these temporary matching features.
    values = np.asarray(values, dtype=float)
    centered = detrend(values, type='linear')
    features = gaussian_filter1d(centered, sigma=1., order=1)
    features -= np.median(features)
    scale = float(np.max(np.abs(features)))
    if scale <= np.finfo(float).eps*max(float(np.max(np.abs(values))), 1.)*100:
        raise ValueError('Alignment window has no usable signal structure.')
    noise = 1.4826*float(np.median(abs(features-np.median(features))))
    amplitude = np.maximum(np.abs(features)-min(3.*noise, .1*scale), 0.)
    return np.sign(features)*np.sqrt(amplitude/scale)


def estimate_shift(reference, spectrum, settings):
    """Normalized derivative cross-correlation, refined to a fractional grid step."""
    settings.validate()
    limit = settings.max_shift_ppm
    step = max(float(np.median(np.diff(reference.ppm))), float(np.median(np.diff(spectrum.ppm))))
    if step <= 0 or not np.isfinite(step):
        raise ValueError('Alignment needs ascending finite ppm axes.')
    low = max(float(reference.ppm[0]), float(spectrum.ppm[0])+limit)
    high = min(float(reference.ppm[-1]), float(spectrum.ppm[-1])-limit)
    if settings.low_ppm is not None:
        low, high = max(low, settings.low_ppm), min(high, settings.high_ppm)
    if high-low < 16*step:
        raise ValueError('Too few overlapping points in the alignment window.')
    step = max(step, (high-low)/65535, limit/8192)
    lag_count = max(1, int(np.floor(limit/step)))
    # A sub-point limit still needs neighboring samples for parabolic refinement.
    margin = lag_count*step
    low = max(low, float(spectrum.ppm[0])+margin)
    high = min(high, float(spectrum.ppm[-1])-margin)
    n = int(np.floor((high-low)/step))+1
    if n < 16:
        raise ValueError('Too few overlapping points in the alignment window.')
    grid = low+np.arange(n)*step
    extended = low+np.arange(-lag_count, n+lag_count)*step
    r = _features(np.interp(grid, reference.ppm, reference.intensity))
    c = _features(np.interp(extended, spectrum.ppm, spectrum.intensity))
    r -= r.mean()
    ref_energy = float(np.dot(r, r))
    if ref_energy < np.finfo(float).eps:
        raise ValueError('Alignment window has no usable signal structure.')
    sums = np.r_[0., np.cumsum(c)]
    squares = np.r_[0., np.cumsum(c*c)]
    energies = squares[n:]-squares[:-n]-(sums[n:]-sums[:-n])**2/n
    denominators = np.sqrt(ref_energy*np.maximum(energies, 0.))
    scores = np.divide(correlate(c, r, mode='valid', method='fft'), denominators,
                       out=np.full_like(denominators, -1.), where=denominators > np.finfo(float).eps)
    # Prefer the smallest shift if multiple lags have indistinguishable scores.
    best_score = float(np.max(scores))
    candidates = np.flatnonzero(scores >= best_score-1e-10)
    best = int(candidates[np.argmin(abs(candidates-lag_count))])
    offset = 0.
    if 0 < best < len(scores)-1:
        curvature = scores[best-1]-2*scores[best]+scores[best+1]
        if curvature < 0:
            offset = float(np.clip(.5*(scores[best-1]-scores[best+1])/curvature, -.5, .5))
    shift = float(np.clip((lag_count-best-offset)*step, -limit, limit))
    segment = np.interp(np.arange(n)+lag_count-shift/step, np.arange(len(c)), c)
    segment -= segment.mean()
    denominator = np.sqrt(ref_energy*np.dot(segment, segment))
    score = float(np.clip(np.dot(r, segment)/denominator, -1., 1.)) if denominator else 0.
    if score < settings.min_correlation:
        raise ValueError(f'Alignment correlation {score:.3f} is below {settings.min_correlation:.3f}.')
    return dict(shift_ppm=shift, correlation=score, window_low_ppm=float(grid[0]),
                window_high_ppm=float(grid[-1]), at_limit=bool(best in (0, len(scores)-1)))


def align_spectrum(reference, spectrum, settings, reference_id, first=False):
    info = estimate_shift(reference, spectrum, replace(settings, min_correlation=0.) if first else settings)
    if first:
        info.update(shift_ppm=0., correlation=1., at_limit=False)
    info['reference_id'] = reference_id
    notes = list(spectrum.notes)
    if info['at_limit']:
        notes.append('Alignment reached the maximum drift; inspect this spectrum.')
    metadata = {**spectrum.processing_metadata, 'alignment_mode': 'global',
                'alignment_reference': reference_id, 'alignment_shift_ppm': info['shift_ppm'],
                'alignment_correlation': info['correlation'], 'alignment': info}
    return replace(spectrum, ppm=spectrum.ppm+info['shift_ppm'], processing_metadata=metadata, notes=notes), info


def align_regions(reference, spectrum, session, reference_id, first=False):
    views, reports, errors = {}, {}, {}
    for region in session.regions:
        settings = replace(session.alignment, low_ppm=region.low, high_ppm=region.high)
        try:
            views[region.name], reports[region.name] = align_spectrum(reference, spectrum, settings, reference_id, first)
        except ValueError as exc:
            errors[region.name] = f'Alignment: {exc}'
            reports[region.name] = dict(error=str(exc), reference_id=reference_id)
    metadata = {**spectrum.processing_metadata, 'alignment_mode': 'regions',
                'alignment_reference': reference_id, 'alignment_regions': reports}
    notes = list(spectrum.notes)
    notes.extend(f'{name}: alignment reached the maximum drift; inspect this region.'
                 for name, info in reports.items() if info.get('at_limit'))
    return replace(spectrum, processing_metadata=metadata, notes=notes), views, errors, reports
