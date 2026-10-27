"""Collision-free automatic names, independent of editable GUI draft fields."""
from itertools import chain


def next_region_name(kind, regions, integrals):
    """Return the first available ``region_N`` or ``signal_N`` (N starts at 1).

    ``kind`` is ``'fit'`` for a processing region or ``'integral'`` for an
    integration region. Both iterables can contain strings or objects with a
    string ``name`` attribute. Names in *both* collections are reserved, so a
    custom name cannot accidentally collide with an automatic name of either
    kind. Matching is literal and case-sensitive, like Session validation.

    The function only proposes a name: it does not rename existing areas or
    change either collection. The caller can keep a user-entered custom name.
    """
    try:
        prefix = {'fit': 'region_', 'integral': 'signal_'}[kind]
    except (KeyError, TypeError):
        raise ValueError("Region kind must be 'fit' or 'integral'.") from None
    occupied = set()
    for item in chain(regions, integrals):
        name = getattr(item, 'name', item)
        if not isinstance(name, str):
            raise TypeError('Regions must be names or objects with string name attributes.')
        occupied.add(name)
    number = 1
    while f'{prefix}{number}' in occupied:
        number += 1
    return f'{prefix}{number}'
