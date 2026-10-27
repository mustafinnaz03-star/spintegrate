"""Complete pybaselines 1D catalogue and parameter-only recipe adapters."""
import copy
import inspect
import json
from types import SimpleNamespace

import numpy as np
from pybaselines import Baseline


FAMILIES = {
    'whittaker': 'Whittaker / penalized least squares',
    'morphological': 'Morphological', 'polynomial': 'Polynomial',
    'spline': 'Splines', 'smooth': 'Smoothing / peak clipping',
    'classification': 'Baseline / peak classification',
    'optimizers': 'Optimizers', 'misc': 'Other methods',
}
CATALOG = {}
for _name in dir(Baseline):
    _function = getattr(Baseline, _name)
    _family = (getattr(_function, '__module__', '') or '').split('.')[-1]
    if not _name.startswith('_') and callable(_function) and _family in FAMILIES:
        CATALOG[_name] = _family
BASELINE_METHODS = ('none', *CATALOG)
LEGACY_METHODS = {'asls', 'iasls', 'arpls', 'airpls', 'drpls', 'snip', 'modpoly', 'imodpoly', 'mor'}
# Avoid zero-weight singular systems on clean, sparsely peaked NMR regions.
APP_DEFAULTS = {'mpls': {'p': .01}, 'mpspline': {'p': .01},
                'pspline_mpls': {'p': .01}, 'loess': {'use_threshold': True}}


def optimizer_methods(method):
    if method == 'adaptive_minmax':
        return [m for m, family in CATALOG.items() if family == 'polynomial']
    if method == 'collab_pls':
        return [m for m, family in CATALOG.items() if family == 'whittaker'
                or (family == 'spline' and m != 'corner_cutting') or m in ('mpls', 'fabc')]
    if method == 'optimize_extended_range':
        return [m for m, family in CATALOG.items() if family in ('whittaker', 'polynomial')
                or (family == 'spline' and m != 'corner_cutting')
                or m in ('mpls', 'mpspline', 'fabc', 'dietrich', 'cwt_br', 'rubberband')]
    return [m for m, family in CATALOG.items() if family != 'optimizers' and m != 'interp_pts']


def parameter_specs(method):
    """Ordered defaults from the installed API; no executable user expressions."""
    if method == 'none':
        return {}
    specs = {p.name: copy.deepcopy(p.default) for p in inspect.signature(getattr(Baseline, method)).parameters.values()
             if p.name not in ('self', 'data') and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    if method == 'interp_pts':
        specs.pop('baseline_points')
        specs = {'anchor_ppm': [], **specs}
    specs.update(APP_DEFAULTS.get(method, {}))
    return specs


def legacy_parameters(settings):
    b = settings
    shared = dict(lam=b.lam, max_iter=b.max_iter, tol=b.tol)
    if b.method in ('asls', 'iasls'):
        return {**shared, 'p': b.p}
    if b.method in ('arpls', 'airpls', 'drpls'):
        return shared
    if b.method == 'snip':
        return dict(max_half_window=b.half_window, decreasing=True, smooth_half_window=b.smooth_half_window)
    if b.method in ('modpoly', 'imodpoly'):
        return dict(poly_order=b.poly_order, max_iter=b.max_iter, tol=b.tol)
    if b.method == 'mor':
        return dict(half_window=b.half_window)
    return {}


def effective_parameters(settings):
    return {**APP_DEFAULTS.get(settings.method, {}), **legacy_parameters(settings),
            **copy.deepcopy(settings.parameters)}


def validate_parameters(settings):
    if not isinstance(settings.parameters, dict):
        raise ValueError('Baseline parameters must be a JSON object.')
    if any(not isinstance(key, str) for key in settings.parameters):
        raise ValueError('Baseline parameter names must be strings.')
    try:
        json.dumps(settings.parameters, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('Baseline parameters must contain finite JSON values.') from exc
    allowed = parameter_specs(settings.method)
    unknown = set(settings.parameters)-set(allowed)
    if unknown:
        raise ValueError(f'{settings.method}: unknown parameters: {", ".join(sorted(unknown))}')
    for name, value in settings.parameters.items():
        if value is None:
            continue
        if name in ('lam', 'lam_0', 'lam_1', 'lam_2', 'lam_smooth', 'tol', 'tol_2', 'tol_3'):
            allow_zero = name in ('lam_0', 'lam_1', 'lam_2', 'lam_smooth') or (name == 'lam' and settings.method in ('rubberband', 'custom_bc'))
            if not isinstance(value, (int, float)) or value < 0 or (value == 0 and not allow_zero):
                raise ValueError(f'{settings.method}: invalid {name}.')
        if name in ('max_iter', 'max_iter_2', 'num_knots', 'diff_order', 'spline_degree', 'poly_order',
                    'half_window', 'max_half_window', 'min_half_window', 'smooth_half_window'):
            sequence_allowed = (name == 'poly_order' and settings.method == 'adaptive_minmax') or (name == 'max_half_window' and settings.method == 'snip')
            values = value if isinstance(value, list) and sequence_allowed else [value]
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
                raise ValueError(f'{settings.method}: {name} must be a nonnegative integer.')
        if name == 'p' and (not isinstance(value, (int, float)) or not 0 <= value < 1
                            or (value == 0 and settings.method not in ('mpls', 'mpspline', 'pspline_mpls'))):
            raise ValueError(f'{settings.method}: p must be in [0, 1).')
    if settings.method == 'interp_pts':
        anchors = settings.parameters.get('anchor_ppm', [])
        anchors = [] if anchors is None else anchors
        if not isinstance(anchors, list) or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in anchors):
            raise ValueError('anchor_ppm must be a JSON list of ppm positions, without intensities.')
    if settings.method in ('collab_pls', 'optimize_extended_range', 'adaptive_minmax', 'custom_bc'):
        nested = settings.parameters.get('method', allowed.get('method'))
        if nested not in optimizer_methods(settings.method):
            raise ValueError('Select a compatible inner baseline method for this optimizer.')
        kws = settings.parameters.get('method_kwargs')
        if kws is not None:
            if not isinstance(kws, dict) or set(kws)-set(parameter_specs(nested)):
                raise ValueError(f'{nested}: method_kwargs contains unsupported parameters.')
            validate_parameters(SimpleNamespace(method=nested, parameters=kws))


def automatic_parameters(method, values):
    defaults = parameter_specs(method)
    return {key: copy.deepcopy(defaults[key]) if value is None and defaults.get(key) is not None else value
            for key, value in values.items()}


def fit_baseline(fitter, x, y, settings):
    """Return normalized baseline and library diagnostics for one fresh region."""
    kwargs = automatic_parameters(settings.method, effective_parameters(settings))
    if CATALOG[settings.method] == 'optimizers':
        nested = kwargs.get('method', parameter_specs(settings.method)['method'])
        if nested in APP_DEFAULTS:
            kwargs['method_kwargs'] = {**APP_DEFAULTS[nested], **(kwargs.get('method_kwargs') or {})}
        if kwargs.get('method_kwargs') is not None:
            kwargs['method_kwargs'] = automatic_parameters(nested, kwargs['method_kwargs'])
    n = len(x)
    # Preserve the previous adapters for legacy recipes and small regions.
    if settings.method in LEGACY_METHODS:
        for key in ('half_window', 'max_half_window', 'smooth_half_window'):
            if key in kwargs and kwargs[key] is not None:
                value = kwargs[key]
                kwargs[key] = [min(v, (n-1)//2) for v in value] if isinstance(value, (list, tuple)) else min(value, (n-1)//2)
        if 'poly_order' in kwargs:
            kwargs['poly_order'] = min(kwargs['poly_order'], n-2)
    if settings.method == 'interp_pts':
        anchors = np.asarray(kwargs.pop('anchor_ppm', []), dtype=float)
        if anchors.size and (anchors.min() < x[0] or anchors.max() > x[-1]):
            raise ValueError('Baseline anchor positions must lie inside this processing region.')
        anchors = np.unique(np.r_[x[0], anchors, x[-1]])
        kwargs['baseline_points'] = np.column_stack((anchors, np.interp(anchors, x, y)))
    data = y[np.newaxis, :] if settings.method == 'collab_pls' else y
    line, params = getattr(fitter, settings.method)(data, **kwargs)
    if settings.method == 'collab_pls':
        line = line[0]  # independent single-spectrum dataset, never weights shared across files
    return line, params


def method_summary(method):
    if method == 'none':
        return 'No baseline subtraction.'
    summary = inspect.getdoc(getattr(Baseline, method)).split('\n\n', 1)[0].replace('\n', ' ')
    if method == 'collab_pls':
        summary += ' Each spectrum is used as a one-trace dataset; spectra are corrected independently.'
    if method == 'interp_pts':
        summary += ' Anchor ppm positions are reused; heights are sampled anew from every spectrum. Region endpoints are included.'
    return summary
