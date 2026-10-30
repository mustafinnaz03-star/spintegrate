"""Wheel navigation for Matplotlib spectra, independent of Tk and numerical processing."""
import math

import numpy as np


def bounded_x_limits(limits, center, factor, bounds):
    inverted = limits[0] > limits[1]
    low, high = sorted(limits)
    minimum, maximum = sorted(bounds)
    full = maximum-minimum
    if full <= 0:
        return limits
    width = min(full, max(full/1e6, (high-low)/factor))
    fraction = .5 if high == low else np.clip((center-low)/(high-low), 0., 1.)
    new_low = center-fraction*width
    new_low = max(minimum, min(new_low, maximum-width))
    result = (float(new_low), float(new_low+width))
    return result[::-1] if inverted else result


def event_modifiers(event):
    result = set(str(getattr(event, 'key', '') or '').lower().split('+'))
    result.update(getattr(event, 'modifiers', ()) or ())
    if 'control' in result or 'cmd' in result or 'super' in result:
        result.add('ctrl')
    return result


class PlotNavigation:
    def __init__(self, canvas, axes, bounds, mode=None, phase=None, blocked=None):
        self.canvas, self.axes, self.bounds = canvas, tuple(axes), bounds
        self.mode = mode or (lambda: 'intensity')
        self.phase, self.blocked = phase, blocked or (lambda: False)
        self.overrides, self.homes, self.context = {}, {}, None
        canvas.spectrum_navigation = self
        self.connections = [canvas.mpl_connect('scroll_event', self.on_scroll),
                            canvas.mpl_connect('button_press_event', self.on_press)]

    def before_draw(self, context):
        if context != self.context:
            self.overrides.clear()
            self.context = context
        else:
            # Include zoom/pan performed with the standard Matplotlib toolbar.
            for axis, (x, y) in self.homes.items():
                if axis.get_xlim() != x:
                    self.overrides.setdefault(axis, {})['x'] = axis.get_xlim()
                if axis.get_ylim() != y:
                    self.overrides.setdefault(axis, {})['y'] = axis.get_ylim()

    def after_draw(self):
        self.homes = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in self.axes}
        for ax, values in self.overrides.items():
            if 'x' in values:
                ax.set_xlim(values['x'])
            if 'y' in values:
                ax.set_ylim(values['y'])

    def _remember_x(self, axis):
        for other in self.axes:
            if other is axis or axis.get_shared_x_axes().joined(axis, other):
                self.overrides.setdefault(other, {})['x'] = other.get_xlim()

    def on_scroll(self, event):
        axis = event.inaxes
        if axis not in self.axes or self.blocked():
            return
        step = float(getattr(event, 'step', 0.) or (1 if event.button == 'up' else -1))
        if not math.isfinite(step) or not step:
            return
        step = max(-20., min(20., step))
        modifiers, mode = event_modifiers(event), self.mode()
        if mode.startswith('phase_') or 'alt' in modifiers:
            if self.phase is not None:
                target = 'phase_p1' if mode == 'phase_p1' or 'shift' in modifiers else 'phase_p0'
                self.phase(target, step*(.1 if 'ctrl' in modifiers else 1.))
            return
        bounds = self.bounds()
        if bounds is None:
            return
        if 'shift' in modifiers:
            limits = axis.get_xlim()
            low, high = sorted(limits)
            delta = -.1*(high-low)*step
            centered = (low+high)/2+delta
            shifted = bounded_x_limits((low+delta, high+delta), centered, 1., bounds)
            axis.set_xlim(shifted[::-1] if limits[0] > limits[1] else shifted)
            self._remember_x(axis)
        elif mode == 'ppm' or 'ctrl' in modifiers:
            limits = axis.get_xlim()
            center = event.xdata if event.xdata is not None else sum(limits)/2
            axis.set_xlim(bounded_x_limits(limits, center, 1.2**step, bounds))
            self._remember_x(axis)
        else:
            low, high = axis.get_ylim()
            anchor = 0. if low <= 0 <= high else (low if low > 0 else high)
            factor = 1.15**step
            if abs(high-low) > np.finfo(float).tiny:
                axis.set_ylim(anchor+(low-anchor)/factor, anchor+(high-anchor)/factor)
                self.overrides.setdefault(axis, {})['y'] = axis.get_ylim()
        self.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes in self.axes and event.button == 1 and event.dblclick and not self.blocked():
            self.reset(event.inaxes)

    def reset(self, axis=None):
        targets = self.axes if axis is None else (axis,)
        for target in targets:
            if target not in self.homes:
                continue
            x, y = self.homes[target]
            target.set_xlim(x)
            target.set_ylim(y)
            for other in self.axes:
                if other is target or target.get_shared_x_axes().joined(target, other):
                    values = self.overrides.get(other, {})
                    values.pop('x', None)
                    if other is target:
                        values.pop('y', None)
                    if not values:
                        self.overrides.pop(other, None)
        self.canvas.draw_idle()

    def disconnect(self):
        for connection in self.connections:
            self.canvas.mpl_disconnect(connection)
