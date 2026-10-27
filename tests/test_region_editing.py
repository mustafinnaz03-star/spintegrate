import copy

import pytest

from spintegrate.core import BaselineSettings, FitRegion, IntegrationRegion, Session
from spintegrate.region_editing import RegionHistory, move_or_resize


def recipe():
    s = Session()
    s.regions = [FitRegion('parent', 3., 4., BaselineSettings(method='interp_pts', parameters={'anchor_ppm': [3.2, 3.8]}))]
    s.integrals = [IntegrationRegion('signal', 'parent', 3.4, 3.6)]
    return s


def test_moving_parent_moves_children_and_interpolation_anchors_without_mutation():
    s = recipe()
    before = copy.deepcopy(s)
    changed = move_or_resize(s, 'fit', 'parent', 4., 5., (0., 10.))
    assert s == before
    assert changed.integrals[0].low == pytest.approx(4.4)
    assert changed.regions[0].baseline.parameters['anchor_ppm'] == pytest.approx([4.2, 4.8])


def test_resizing_parent_protects_children_and_baseline_anchors():
    changed = move_or_resize(recipe(), 'fit', 'parent', 3.5, 3.9, (0., 10.))
    assert changed.regions[0].low == 3.2
    assert changed.regions[0].high == 3.9
    changed.validate()


def test_integral_moves_and_resizes_stay_inside_parent():
    s = recipe()
    moved = move_or_resize(s, 'integral', 'signal', 5., 5.2, (0., 10.))
    assert moved.integrals[0].high == pytest.approx(4.)
    assert moved.integrals[0].high-moved.integrals[0].low == pytest.approx(.2)
    resized = move_or_resize(s, 'integral', 'signal', 2., 3.6, (0., 10.))
    assert resized.integrals[0].low == 3.
    with pytest.raises(ValueError):
        move_or_resize(s, 'integral', 'signal', 8., 9., (0., 10.))


def test_geometry_undo_preserves_new_processing_and_baseline_parameters():
    s = recipe()
    history = RegionHistory()
    history.record(s)
    changed = move_or_resize(s, 'fit', 'parent', 4., 5., (0., 10.))
    changed.processing.lb_hz = 3.
    changed.regions[0].baseline.parameters['interp_method'] = 'nearest'
    restored = history.undo(changed)
    assert restored.regions[0].low == 3.
    assert restored.integrals[0].low == 3.4
    assert restored.processing.lb_hz == 3.
    assert restored.regions[0].baseline.parameters == {'anchor_ppm': pytest.approx([3.2, 3.8]), 'interp_method': 'nearest'}
    assert not history.states
