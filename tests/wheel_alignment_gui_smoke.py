"""Real wheel/pivot events and both first-spectrum batch alignment modes."""
import copy
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

import nmrglue as ng
import numpy as np
from matplotlib.backend_bases import MouseEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.core import Session, load_analysis
from spintegrate.gui import ALIGNMENT_MODES, SpintegrateApp
from test_alignment import recipe, trace
from test_pdata_direct import write_pdata_fixture


def main():
    root = tk.Tk()
    root.withdraw()
    app = SpintegrateApp(root)
    errors, estimates = [], []
    root.report_callback_exception = lambda _kind, exc, _tb: errors.append(exc)
    original_phaser = ng.proc_autophase.autops
    def fake_phase(data, *_args, **_kwargs):
        estimates.append(data.copy())
        return data, (20., -6.)
    ng.proc_autophase.autops = fake_phase
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
        raise AssertionError(f'GUI did not settle: {workspace.status.get()} / {list(workspace.jobs)}')
    def wheel(canvas, axis, key='', step=1):
        canvas.draw()
        px, py = axis.transData.transform((np.mean(axis.get_xlim()), np.mean(axis.get_ylim())))
        canvas.callbacks.process('scroll_event', MouseEvent('scroll_event', canvas, px, py, button='up', step=step, key=key))
    try:
        child = app.open_source_window('pdata')
        child.root.withdraw()
        child.error = fail
        child.confirm = lambda _title, message: fail('Unexpected confirmation: '+message)
        for workspace in (app, child):
            workspace.load_demo()
            workspace.set_language('en')
            settle(workspace)
            workspace.processing_fields.vars['phase_label'].set('ACME (nmrglue)')
            settle(workspace)
            count = len(estimates)
            x0, y0 = workspace.detail_ax.get_xlim(), workspace.detail_ax.get_ylim()
            data = workspace.spectrum.intensity.copy()
            wheel(workspace.canvas, workspace.detail_ax)
            assert np.ptp(workspace.detail_ax.get_ylim()) < np.ptp(y0)
            wheel(workspace.canvas, workspace.detail_ax, 'ctrl')
            assert np.ptp(workspace.detail_ax.get_xlim()) < np.ptp(x0)
            np.testing.assert_array_equal(workspace.spectrum.intensity, data)
            xzoom, yzoom = workspace.detail_ax.get_xlim(), workspace.detail_ax.get_ylim()
            workspace.wheel_mode.set('Phase P0')
            wheel(workspace.canvas, workspace.detail_ax)
            settle(workspace)
            assert workspace.read_settings().processing.manual_phase_p0 == 1.
            wheel(workspace.canvas, workspace.detail_ax, 'ctrl+shift')
            settle(workspace)
            assert workspace.read_settings().processing.manual_phase_p1 == .5
            assert len(estimates) == count
            np.testing.assert_allclose(workspace.detail_ax.get_xlim(), xzoom)
            np.testing.assert_allclose(workspace.detail_ax.get_ylim(), yzoom)
            before = workspace.spectrum.complex_spectrum.copy()
            workspace.arm_phase_pivot()
            workspace.canvas.draw()
            px, py = workspace.overview_ax.transData.transform((3.55, np.mean(workspace.overview_ax.get_ylim())))
            workspace.canvas.callbacks.process('button_press_event', MouseEvent('button_press_event', workspace.canvas, px, py, button=1))
            settle(workspace)
            assert not workspace.phase_pivot_armed
            np.testing.assert_allclose(workspace.spectrum.complex_spectrum, before, atol=1e-7, rtol=1e-10)
            assert len(estimates) == count
            workspace.open_preprocess()
            popup = workspace.preprocess_window
            popup.window.withdraw()
            settle(workspace)
            workspace.wheel_mode.set('Intensity')
            old_y = popup.ax_apod.get_ylim()
            wheel(popup.canvas, popup.ax_apod)
            saved_y = popup.ax_apod.get_ylim()
            assert np.ptp(saved_y) < np.ptp(old_y)
            popup.lb_var.set('2')
            settle(workspace)
            np.testing.assert_allclose(popup.ax_apod.get_ylim(), saved_y)
            assert len(estimates) == count
            workspace.processing_fields.vars['phase_label'].set('abs')
            settle(workspace)
            workspace.wheel_mode.set('Phase P0')
            old_p0 = workspace.processing_fields.vars['manual_phase_p0'].get()
            wheel(workspace.canvas, workspace.detail_ax)
            assert workspace.processing_fields.vars['manual_phase_p0'].get() == old_p0
            popup.close()
            workspace.wheel_mode.set('Intensity')
            print(f'{workspace.source_mode}: wheel zoom, preserved viewport, phase offsets/pivot, abs guard and no repeated autophase: OK', flush=True)

            with tempfile.TemporaryDirectory(prefix='spintegrate_align_gui_') as folder:
                target = Path(folder)
                for mode in ('global', 'regions'):
                    directory = target/mode
                    directory.mkdir()
                    x = 2.+np.arange(8192)*6./8192
                    for number, left, right in [(1, 0., 0.), (2, .031, .031 if mode == 'global' else -.024)]:
                        y = trace(x, left, right, gain=1.-number*.025)
                        if workspace.source_mode == 'raw':
                            np.savez(directory/f'{number}.npz', fid=ng.proc_base.ifft(y.astype(complex)),
                                     sf_mhz=400., sw_hz=2400., carrier_ppm=5.)
                        else:
                            write_pdata_fixture(directory/str(number)/'pdata'/'1', y[::-1]*10000.,
                                                nc_proc=0, offset=float(x[-1]), sf=400., sw=2400.)
                    settings = recipe(mode)
                    settings.processing.source_mode = workspace.source_mode
                    settings.alignment.enabled = False
                    workspace.apply_recipe(settings)
                    settle(workspace)
                    workspace.remember_for_batch()
                    controls = workspace.alignment_controls
                    controls.vars['enabled'].set(True)
                    controls.vars['mode'].set(workspace.t(next(label for label, code in ALIGNMENT_MODES.items() if code == mode)))
                    controls.vars['max_shift_ppm'].set('.08')
                    if mode == 'regions':
                        controls.vars['low_ppm'].set('ignored in regional mode')
                        assert controls.read(settings.alignment).low_ppm is None
                    assert workspace.batch_recipe.alignment.enabled and workspace.batch_recipe.alignment.mode == mode
                    workspace.batch_root.set(str(directory))
                    workspace.batch_output.set(str(target/f'{mode}.csv'))
                    workspace.run_batch()
                    settle(workspace)
                    assert len(workspace.batch_rows) == 2
                    assert all(row['status'] != 'error' for row in workspace.batch_rows)
                    assert 'alignment' in workspace.batch_tree.cget('columns')
                    assert Session.load(target/f'{mode}.settings.json').alignment.mode == mode
                    item = workspace.batch_tree.get_children()[1]
                    workspace.batch_tree.selection_set(item)
                    workspace.review_selected_fit()
                    settle(workspace)
                    review = workspace.review_windows[-1]
                    review.window.withdraw()
                    assert review.analysis.spectrum.processing_metadata['alignment_mode'] == mode
                    stored = copy.deepcopy(review.analysis.integrals)
                    original_y = review.analysis.spectrum.intensity.copy()
                    review_y = review.detail_ax.get_ylim()
                    wheel(review.canvas, review.detail_ax)
                    assert np.ptp(review.detail_ax.get_ylim()) < np.ptp(review_y)
                    np.testing.assert_array_equal(review.analysis.spectrum.intensity, original_y)
                    workspace.batch_rows[1]['fit_file'] = ''  # simulate missing archive without deleting files
                    workspace.review_selected_fit()
                    settle(workspace)
                    restored = workspace.review_windows[-1]
                    restored.window.withdraw()
                    assert restored.analysis.integrals == stored
                    assert restored.analysis.spectrum.processing_metadata['alignment_mode'] == mode
                    print(f'{workspace.source_mode}: {mode} first-spectrum alignment, saved settings, drift column and exact review reconstruction: OK', flush=True)
                if workspace.source_mode == 'pdata':
                    workspace.load_path(str(directory/'1'/'pdata'/'1'))
                    settle(workspace)
                    workspace.wheel_mode.set('Phase P0')
                    before = workspace.processing_fields.vars['manual_phase_p0'].get()
                    wheel(workspace.canvas, workspace.detail_ax)
                    assert workspace.processing_fields.vars['manual_phase_p0'].get() == before
                    assert '1i' in workspace.status.get()
    finally:
        for workspace in [app, *app.source_windows]:
            workspace.batch_cancel.set()
            workspace.executor.shutdown(wait=True, cancel_futures=True)
        ng.proc_autophase.autops = original_phaser
        root.destroy()


if __name__ == '__main__':
    main()
