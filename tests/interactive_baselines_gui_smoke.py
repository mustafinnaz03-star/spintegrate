"""Exercise all method forms and real Matplotlib drag/resize events in both workspaces."""
import copy
import json
import sys
import time
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.baseline_methods import CATALOG
from spintegrate.core import BaselineSettings, validate_baseline
from spintegrate.gui import SpintegrateApp


def main():
    root = tk.Tk()
    root.withdraw()
    app = SpintegrateApp(root)
    errors = []
    root.report_callback_exception = lambda _type, exc, _tb: errors.append(exc)
    def fail(message):
        raise AssertionError(message)
    app.error = fail
    def settle(workspace):
        until = time.monotonic()+45
        while time.monotonic() < until:
            root.update()
            assert not errors, errors
            popup = workspace.preprocess_window
            if (not workspace.jobs and not workspace.pending_preview and not workspace.preview_timer
                    and not (popup and popup.timer)):
                root.update()
                return
            time.sleep(.01)
        raise AssertionError(f'GUI did not settle: {workspace.status.get()}')
    def drag(workspace, start, end):
        canvas, ax = workspace.canvas, workspace.overview_ax
        canvas.draw()
        y = np.mean(ax.get_ylim())
        for kind, x in [('button_press_event', start), ('motion_notify_event', end), ('button_release_event', end)]:
            px, py = ax.transData.transform((x, y))
            event = MouseEvent(kind, canvas, px, py, button=1)
            canvas.callbacks.process(kind, event)
        settle(workspace)
    try:
        child = app.open_source_window('pdata')
        child.root.withdraw()
        child.error = fail
        for workspace in (app, child):
            workspace.load_demo()
            settle(workspace)
            workspace.set_language('en')
            workspace.open_preprocess()
            popup = workspace.preprocess_window
            popup.window.withdraw()
            settle(workspace)
            saved = copy.deepcopy(next(r for r in workspace.session.regions if r.name == workspace.selected_region).baseline)
            popup.loading = True
            for method in ('none', *CATALOG):
                popup.fields.vars['method'].set(method)
                settings = BaselineSettings(**popup.fields.values())
                validate_baseline(settings)
                assert settings.method == method
                assert popup.fields.group == CATALOG.get(method, 'none')
            popup.fields.set_values(asdict(saved))
            popup.loading = False
            popup.changed()
            settle(workspace)
            popup.fields.group_var.set('Morphological')
            assert all(CATALOG[m] == 'morphological' for m in popup.fields.method_combo.cget('values'))
            popup.fields.search_var.set('pspline_ar')
            assert tuple(popup.fields.method_combo.cget('values')) == ('pspline_arpls',)
            popup.fields.vars['method'].set('asls')
            settle(workspace)
            popup.fields.lambda_slider.set(8.)
            settle(workspace)
            assert workspace.session.regions[0].baseline.lam == 1e8
            old_p = workspace.session.regions[0].baseline.p
            popup.fields.vars['p'].set('.005')
            settle(workspace)
            assert workspace.session.regions[0].baseline.p == .005
            popup.undo_baseline()
            settle(workspace)
            assert workspace.session.regions[0].baseline.p == old_p
            popup.curve_visibility['raw'].set(False)
            popup.redraw_preview()
            assert not popup.ax_base.lines[0].get_visible()
            popup.curve_visibility['raw'].set(True)
            popup.redraw_preview()
            popup.pick_anchors.set(True)
            popup.enable_anchors()
            settle(workspace)
            for x in (3.35, 3.8):
                popup.on_anchor_click(SimpleNamespace(inaxes=popup.ax_base, xdata=x, button=1))
                settle(workspace)
            assert workspace.session.regions[0].baseline.parameters['anchor_ppm'] == [3.35, 3.8]
            popup.on_anchor_click(SimpleNamespace(inaxes=popup.ax_base, xdata=3.35, button=3))
            settle(workspace)
            assert workspace.session.regions[0].baseline.parameters['anchor_ppm'] == [3.8]
            popup.undo_baseline()
            settle(workspace)
            assert workspace.session.regions[0].baseline.parameters['anchor_ppm'] == [3.35, 3.8]
            print(f'{workspace.source_mode}: all 62 forms, grouping/search, slider, baseline undo, visibility and anchors: OK', flush=True)

            original = copy.deepcopy(workspace.session)
            region = workspace.session.regions[0]
            workspace.edit_regions.set(True)
            workspace.toggle_region_editing()
            drag(workspace, region.high, region.high+.1)
            assert abs(workspace.session.regions[0].high-(region.high+.1)) < .003
            resized = copy.deepcopy(workspace.session)
            region = workspace.session.regions[0]
            drag(workspace, (region.low+region.high)/2, (region.low+region.high)/2+.12)
            assert abs(workspace.session.regions[0].low-(region.low+.12)) < .003
            delta = workspace.session.regions[0].low-region.low
            assert abs(workspace.session.integrals[0].low-resized.integrals[0].low-delta) < 1e-8
            assert np.allclose(workspace.session.regions[0].baseline.parameters['anchor_ppm'], np.array([3.35, 3.8])+delta)
            event = KeyEvent('key_press_event', workspace.canvas, key='ctrl+z')
            workspace.canvas.callbacks.process('key_press_event', event)
            settle(workspace)
            assert abs(workspace.session.regions[0].low-resized.regions[0].low) < 1e-8
            workspace.undo_region_bounds()
            settle(workspace)
            assert workspace.session.regions[0].low == original.regions[0].low
            assert workspace.session.regions[0].high == original.regions[0].high
            workspace.area_tree.selection_set(workspace.session.integrals[0].name)
            settle(workspace)
            area = copy.deepcopy(workspace.session.integrals[0])
            drag(workspace, (area.low+area.high)/2, (area.low+area.high)/2+1.)
            parent = workspace.session.regions[0]
            area = workspace.session.integrals[0]
            assert parent.low <= area.low < area.high <= parent.high
            assert abs(area.high-parent.high) < 1e-8
            assert abs((area.high-area.low)-(original.integrals[0].high-original.integrals[0].low)) < 1e-8
            workspace.undo_region_bounds()
            settle(workspace)
            assert workspace.session.integrals[0].low == original.integrals[0].low
            workspace.region_tree.selection_set(workspace.session.regions[0].name)
            settle(workspace)
            before_cancel = copy.deepcopy(workspace.session)
            canvas, ax = workspace.canvas, workspace.overview_ax
            canvas.draw()
            y = np.mean(ax.get_ylim())
            high = workspace.session.regions[0].high
            for kind, x in [('button_press_event', high), ('motion_notify_event', high+.1)]:
                px, py = ax.transData.transform((x, y))
                canvas.callbacks.process(kind, MouseEvent(kind, canvas, px, py, button=1))
            canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', canvas, key='escape'))
            settle(workspace)
            assert not workspace.region_dragging and workspace.session == before_cancel
            assert abs(float(workspace.bound_high.get())-high) < 1e-8
            payload = asdict(workspace.read_settings())
            assert payload['regions'][0]['baseline']['parameters']['anchor_ppm'] == [3.35, 3.8]
            json.dumps(payload, allow_nan=False)
            print(f'{workspace.source_mode}: mouse resize/move, child constraints, shifted anchors and geometry undo: OK', flush=True)
    finally:
        for workspace in [app, *app.source_windows]:
            workspace.batch_cancel.set()
            workspace.executor.shutdown(wait=True, cancel_futures=True)
        root.destroy()


if __name__ == '__main__':
    main()
