"""Transfer a resolved phase from one selected spectrum to an entire batch."""
import copy
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BatchPhaseSettings:
    mode: str = 'per_spectrum'  # per_spectrum / fixed
    captured: bool = False
    p0: float = 0.  # complete phase at the first original frequency bin, degrees
    p1: float = 0.  # complete first-order phase ramp, degrees
    source: str = ''
    source_mode: str = 'raw'

    def validate(self, require_capture=False):
        if self.mode not in ('per_spectrum', 'fixed') or type(self.captured) is not bool:
            raise ValueError('Некорректный режим фазировки серии.')
        if not np.isfinite([self.p0, self.p1]).all():
            raise ValueError('Фаза серии P0/P1 должна содержать конечные числа.')
        if self.source_mode not in ('raw', 'pdata') or not isinstance(self.source, str):
            raise ValueError('Некорректный источник фиксированной фазы.')
        if require_capture and self.mode == 'fixed' and not self.captured:
            raise ValueError('Сначала зафиксируйте фазу выбранного спектра для серии.')


def capture_batch_phase(source, processing, phase_cache=None):
    """Resolve exactly the selected source's phase, including manual pivot trims."""
    from .core import process
    if processing.phase == 'abs':
        raise ValueError('В режиме abs фаза не определяется. Выберите фазировку перед фиксацией для серии.')
    spectrum = process(source, processing, phase_cache=phase_cache)
    # pdata already contains TopSpin PHC0/PHC1. Only our ADDITIONAL correction
    # may be transferred; imported metadata are not angles to apply again.
    phases = (spectrum.processing_metadata.get('additional_phases', (0., 0.))
              if spectrum.source_type == 'pdata' else spectrum.phases)
    result = BatchPhaseSettings(mode='fixed', captured=True, p0=float(phases[0]), p1=float(phases[1]),
                                source=str(spectrum.source), source_mode=spectrum.source_type)
    result.validate(require_capture=True)
    return result


def batch_processing_settings(session):
    """Apply the saved total angles, bypassing every per-file automatic phase fit."""
    phase = session.batch_phase
    phase.validate(require_capture=True)
    p = copy.deepcopy(session.processing)
    if phase.mode == 'fixed':
        if p.phase == 'abs':
            raise ValueError('Фиксированная фаза серии несовместима с abs. Выберите независимую обработку фазы.')
        if phase.source_mode != p.source_mode:
            raise ValueError('Источник фиксированной фазы должен совпадать с режимом raw/pdata серии.')
        # Total angles already include the automatic base and all manual trims.
        # Use the existing pre-apodization manual path with its pivot at bin 0.
        p.phase, p.p0, p.p1 = 'none', 0., 0.
        p.manual_phase_p0, p.manual_phase_p1 = phase.p0, phase.p1
        p.phase_pivot_fraction = 0.
    return p


def process_for_batch(source, session):
    from .core import process
    spectrum = process(source, batch_processing_settings(session))
    if session.batch_phase.mode == 'fixed':
        phase = session.batch_phase
        spectrum.phases = (phase.p0, phase.p1)
        spectrum.processing_metadata['batch_phase'] = asdict(phase)
        if spectrum.source_type == 'pdata':
            spectrum.processing_metadata['additional_phases'] = [phase.p0, phase.p1]
    return spectrum
