"""Automatic/manual transitions must never leave undeletable automatic markers."""
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import numpy as np
from matplotlib.backend_bases import MouseEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spintegrate.gui as gui
from spintegrate.gui import SpintegrateApp


def main():
    root = tk.Tk()
    root.withdraw()
    app = SpintegrateApp(root)
    errors = []
    root.report_callback_exception = lambda _kind, exc, _tb: errors.append(exc)
    original_picker = gui.pick_peaks
    release = threading.Event()
    def fail(message):
        raise AssertionError(message)
    app.error = fail
    def settle(workspace):
        until = time.monotonic()+40
        while time.monotonic() < until:
            root.update()
            assert not errors, errors
            if not workspace.jobs and not workspace.preview_timer and not workspace.pending_preview:
                root.update()
                return
            time.sleep(.01)
        raise AssertionError(f'GUI did not settle: {workspace.status.get()}')
    def count_marks(workspace):
        return sum(len(artist.get_offsets()) for artist in workspace.detail_ax.collections
                   if artist.get_label() in ('Peak-picking', 'Peak-picked positions (fixed)',
                                              'Manual marks (initial)', 'Optimized centers'))
    def click(workspace, ppm, button=1):
        workspace.canvas.draw()
        px, py = workspace.detail_ax.transData.transform((ppm, np.mean(workspace.detail_ax.get_ylim())))
        workspace.canvas.callbacks.process('button_press_event',
            MouseEvent('button_press_event', workspace.canvas, px, py, button=button))
    try:
        child = app.open_source_window('pdata')
        child.root.withdraw()
        child.error = fail
        for workspace in (app, child):
            workspace.load_demo()
            workspace.set_language('en')
            settle(workspace)
            workspace.run_analysis()
            settle(workspace)
            assert workspace.analysis.method == 'lorentzian' and count_marks(workspace) > 0
            assert not any(region.manual_peaks for region in workspace.session.regions)

            workspace.fit_fields.vars['analysis_label'].set('Lorentzian fit: manual peaks')
            # Do not pump Tk or wait for workers: the old fit must disappear now.
            assert count_marks(workspace) == 0, 'Automatic fit markers survived switching to manual mode'
            assert workspace.analysis is None
            workspace.set_manual_peaks([])
            assert count_marks(workspace) == 0
            assert workspace.manual_peak_edit.get(), 'Entering manual mode should allow peak editing'
            settle(workspace)
            assert count_marks(workspace) == 0

            # Hold an old automatic preview while changing modes and editing marks.
            workspace.fit_fields.vars['analysis_label'].set('Lorentzian deconvolution')
            settle(workspace)
            assert count_marks(workspace) > 0
            started = threading.Event()
            release.clear()
            def delayed_picker(*args, **kwargs):
                started.set()
                if not release.wait(15):
                    raise AssertionError('Test did not release automatic preview')
                return original_picker(*args, **kwargs)
            gui.pick_peaks = delayed_picker
            workspace.preview()
            assert started.wait(5), 'Automatic preview worker did not start'
            workspace.fit_fields.vars['analysis_label'].set('Lorentzian fit: manual peaks')
            assert count_marks(workspace) == 0
            assert workspace.manual_peak_edit.get()
            click(workspace, 3.55)
            assert len(workspace.session.regions[0].manual_peaks) == 1
            click(workspace, 3.55, button=3)
            assert workspace.session.regions[0].manual_peaks == []
            release.set()
            settle(workspace)
            gui.pick_peaks = original_picker
            assert workspace.session.analysis_method == 'lorentzian_manual'
            assert count_marks(workspace) == 0

            # Preserve deliberately saved manual marks across an automatic round trip.
            workspace.set_manual_peaks([3.55, 3.59])
            settle(workspace)
            assert count_marks(workspace) == 2
            workspace.fit_fields.vars['analysis_label'].set('Lorentzian deconvolution')
            settle(workspace)
            workspace.fit_fields.vars['analysis_label'].set('Lorentzian fit: manual peaks')
            assert workspace.session.regions[0].manual_peaks == [3.55, 3.59]
            settle(workspace)
            assert count_marks(workspace) == 2
            first_name, second_name = [region.name for region in workspace.session.regions]
            workspace.region_tree.selection_set(second_name)
            settle(workspace)
            workspace.set_manual_peaks([5.23, 5.28])
            settle(workspace)
            workspace.region_tree.selection_set(first_name)
            settle(workspace)
            workspace.run_analysis()
            settle(workspace)
            assert workspace.analysis.method == 'lorentzian_manual'
            workspace.set_manual_peaks([])
            assert workspace.analysis is None and count_marks(workspace) == 0
            settle(workspace)
            workspace.set_manual_peaks([])  # empty Clear must still refresh the view
            assert count_marks(workspace) == 0
            print(f'{workspace.source_mode}: immediate auto/manual reset, editing/deletion, delayed preview, saved marks and Clear after fit: OK', flush=True)
    finally:
        release.set()
        gui.pick_peaks = original_picker
        for workspace in [app, *app.source_windows]:
            workspace.executor.shutdown(wait=True, cancel_futures=True)
        root.destroy()


if __name__ == '__main__':
    main()
