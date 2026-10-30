from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from spintegrate.navigation import PlotNavigation, bounded_x_limits


def example():
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    a = figure.add_subplot(211)
    b = figure.add_subplot(212, sharex=a)
    for ax in (a, b):
        ax.plot(np.linspace(0, 10, 101), np.sin(np.linspace(0, 10, 101)))
        ax.set_xlim(10, 0)
    phases = []
    mode = ['intensity']
    nav = PlotNavigation(canvas, (a, b), lambda: (0, 10), lambda: mode[0], lambda *args: phases.append(args))
    nav.before_draw('source')
    nav.after_draw()
    return nav, a, b, mode, phases


def wheel(ax, key='', step=1):
    return SimpleNamespace(inaxes=ax, step=step, button='up', key=key, xdata=7., modifiers=())


def test_intensity_zoom_ppm_zoom_pan_and_reset_preserve_reversed_axis():
    nav, a, b, _, _ = example()
    y = a.get_ylim()
    nav.on_scroll(wheel(a))
    assert np.ptp(a.get_ylim()) < np.ptp(y)
    nav.on_scroll(wheel(a, 'ctrl'))
    assert a.get_xlim()[0] > a.get_xlim()[1]
    assert np.ptp(a.get_xlim()) < 10
    assert a.get_xlim() == b.get_xlim()
    nav.on_scroll(wheel(a, 'shift'))
    assert 0 <= min(a.get_xlim()) < max(a.get_xlim()) <= 10
    nav.reset(a)
    assert a.get_xlim() == (10, 0)
    np.testing.assert_allclose(a.get_ylim(), y)


def test_zoom_survives_redraw_but_new_source_resets_it():
    nav, a, b, _, _ = example()
    nav.on_scroll(wheel(a, 'ctrl'))
    saved = a.get_xlim()
    nav.before_draw('source')
    a.set_xlim(10, 0)
    nav.after_draw()
    assert a.get_xlim() == saved
    nav.before_draw('another source')
    a.set_xlim(10, 0)
    nav.after_draw()
    assert a.get_xlim() == b.get_xlim() == (10, 0)


def test_toolbar_view_changes_are_preserved_after_processing_redraw():
    nav, a, _, _, _ = example()
    a.set_xlim(7., 4.)
    a.set_ylim(-.5, .75)
    nav.before_draw('source')
    a.set_xlim(10., 0.)
    a.set_ylim(-2., 2.)
    nav.after_draw()
    assert a.get_xlim() == (7., 4.)
    assert a.get_ylim() == (-.5, .75)


def test_phase_mode_modifiers_change_offsets_without_changing_plot_limits():
    nav, a, _, mode, phases = example()
    limits = a.get_xlim(), a.get_ylim()
    mode[0] = 'phase_p0'
    nav.on_scroll(wheel(a))
    nav.on_scroll(wheel(a, 'ctrl+shift'))
    assert phases == [('phase_p0', 1.), ('phase_p1', .1)]
    assert limits == (a.get_xlim(), a.get_ylim())
    mode[0] = 'intensity'
    nav.on_scroll(wheel(a, 'alt'))
    assert phases[-1] == ('phase_p0', 1.)


def test_zoom_limits_remain_finite_and_bounded():
    assert bounded_x_limits((10, 0), 7, .001, (0, 10)) == (10, 0)
    result = bounded_x_limits((10, 0), 7, 1e12, (0, 10))
    assert 0 <= min(result) < max(result) <= 10
