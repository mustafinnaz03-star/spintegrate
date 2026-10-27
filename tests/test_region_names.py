"""Automatic area names remain unique as regions are edited or deleted."""
from types import SimpleNamespace

import pytest

from spintegrate.region_names import next_region_name


def region(name):
    return SimpleNamespace(name=name)


def test_empty_collections_start_at_one():
    assert next_region_name('fit', [], []) == 'region_1'
    assert next_region_name('integral', [], []) == 'signal_1'


def test_both_collections_reserve_names_even_with_custom_cross_kind_names():
    regions = [region('region_1'), region('signal_2')]
    integrals = [region('region_2'), region('signal_1')]
    assert next_region_name('fit', regions, integrals) == 'region_3'
    assert next_region_name('integral', regions, integrals) == 'signal_3'


def test_deleted_numbers_are_reused_without_colliding_with_later_names():
    assert next_region_name('fit', ['region_1', 'region_3'], []) == 'region_2'
    assert next_region_name('integral', [], ['signal_1', 'signal_3']) == 'signal_2'
    assert next_region_name('fit', ['region_2', 'region_3'], []) == 'region_1'


def test_custom_names_are_unchanged_and_matching_is_literal():
    regions = [region('Ароматическая область'), region('region_01'), region('Region_1')]
    integrals = [region('reference'), region('signal_100')]
    before = [r.name for r in regions + integrals]
    assert next_region_name('fit', regions, integrals) == 'region_1'
    assert next_region_name('integral', regions, integrals) == 'signal_1'
    assert [r.name for r in regions + integrals] == before


def test_iterables_of_strings_and_name_objects_can_be_mixed():
    assert next_region_name('fit', iter(['region_1', region('region_2')]),
                            (region(name) for name in ['region_3', 'custom'])) == 'region_4'


@pytest.mark.parametrize('kind', ['raw', 'region', '', None])
def test_unknown_kind_is_rejected(kind):
    with pytest.raises(ValueError, match='kind'):
        next_region_name(kind, [], [])


def test_invalid_name_type_is_rejected():
    with pytest.raises(TypeError, match='string'):
        next_region_name('fit', [region(3)], [])
