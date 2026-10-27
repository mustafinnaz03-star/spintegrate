"""Constrained region geometry edits; undo never rolls back processing settings."""
import copy
import math


def shift_anchors(region, delta):
    if region.baseline.method == 'interp_pts' and region.baseline.parameters.get('anchor_ppm'):
        region.baseline.parameters['anchor_ppm'] = [v+delta for v in region.baseline.parameters['anchor_ppm']]


def move_or_resize(session, kind, name, low, high, spectrum_limits):
    low, high = sorted((float(low), float(high)))
    if not all(math.isfinite(v) for v in (low, high)) or low >= high:
        raise ValueError('Region bounds must be finite and distinct.')
    result = copy.deepcopy(session)
    items = result.regions if kind == 'fit' else result.integrals
    region = next((r for r in items if r.name == name), None)
    if region is None:
        raise ValueError('Select a saved region to edit.')
    translating = math.isclose(high-low, region.high-region.low, rel_tol=1e-7, abs_tol=1e-9)
    if kind == 'fit':
        minimum, maximum = sorted(spectrum_limits)
        children = [area for area in result.integrals if area.parent == name]
    else:
        parent = next(r for r in result.regions if r.name == region.parent)
        minimum, maximum, children = parent.low, parent.high, []
    if translating:
        delta = max(minimum-region.low, min(low-region.low, maximum-region.high))
        region.low += delta
        region.high += delta
        if kind == 'fit':
            shift_anchors(region, delta)
        for area in children:
            area.low += delta
            area.high += delta
    else:
        region.low, region.high = max(low, minimum), min(high, maximum)
        if children:
            region.low = min(region.low, min(area.low for area in children))
            region.high = max(region.high, max(area.high for area in children))
        if kind == 'fit' and region.baseline.method == 'interp_pts':
            anchors = region.baseline.parameters.get('anchor_ppm', [])
            if anchors:
                region.low = min(region.low, min(anchors))
                region.high = max(region.high, max(anchors))
    if not minimum <= region.low < region.high <= maximum:
        raise ValueError('The region must fit inside its spectrum or parent region.')
    result.validate()
    return result


class RegionHistory:
    def __init__(self):
        self.states = []

    def clear(self):
        self.states.clear()

    def record(self, session):
        self.states.append((
            {r.name: (r.low, r.high) for r in session.regions},
            {r.name: (r.parent, r.low, r.high) for r in session.integrals}))
        del self.states[:-30]

    def undo(self, session):
        if not self.states:
            return session
        regions, integrals = self.states[-1]
        if (set(regions) != {r.name for r in session.regions}
                or set(integrals) != {r.name for r in session.integrals}
                or any(integrals[r.name][0] != r.parent for r in session.integrals)):
            self.clear()
            raise ValueError('Region structure changed; geometry undo was cleared.')
        result = copy.deepcopy(session)
        for r in result.regions:
            old_low, old_high = regions[r.name]
            if math.isclose(old_high-old_low, r.high-r.low, rel_tol=1e-7, abs_tol=1e-9):
                shift_anchors(r, old_low-r.low)
            r.low, r.high = regions[r.name]
        for r in result.integrals:
            _, r.low, r.high = integrals[r.name]
        result.validate()
        self.states.pop()
        return result
