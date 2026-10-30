"""Manual peak clicks, undo, free-center fits and aligned batch review in raw/pdata."""
import copy
import csv
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

import nmrglue as ng
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.core import Session, load_analysis, lorentzian
from spintegrate.gui import ALIGNMENT_MODES, SpintegrateApp
from test_manual_peaks import settings
from test_pdata_direct import write_pdata_fixture


def main():
    root = tk.Tk()
    root.withdraw()
    app = SpintegrateApp(root)
    errors = []
    root.report_callback_exception = lambda _kind, exc, _tb: errors.append(exc)
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
    def click(workspace, value, button=1):
        workspace.canvas.draw()
        px, py = workspace.detail_ax.transData.transform((value, np.mean(workspace.detail_ax.get_ylim())))
        workspace.canvas.callbacks.process('button_press_event',
            MouseEvent('button_press_event', workspace.canvas, px, py, button=button))
        settle(workspace)
    try:
        child = app.open_source_window('pdata')
        child.root.withdraw()
        child.error = fail
        with tempfile.TemporaryDirectory(prefix='spintegrate_manual_peaks_') as folder:
            target = Path(folder)
            for workspace in (app, child):
                directory = target/f'{workspace.source_mode}_series'
                directory.mkdir()
                paths = []
                x = 2.+np.arange(8192)*6./8192
                for number, drift, gain in [(1, 0., 1.), (2, .042, .6)]:
                    y = gain*(lorentzian(x, 3.5+drift, .006, 12000.)+
                              lorentzian(x, 3.6+drift+.004*(number-1), .009, 6500.))
                    if workspace.source_mode == 'raw':
                        path = directory/f'{number}.npz'
                        np.savez(path, fid=ng.proc_base.ifft(y.astype(complex)), sf_mhz=400., sw_hz=2400., carrier_ppm=5.)
                    else:
                        path = directory/str(number)/'pdata'/'1'
                        write_pdata_fixture(path, y[::-1]*10000., nc_proc=0, offset=float(x[-1]), sf=400., sw=2400.)
                    paths.append(path)
                s = settings()
                s.appearance.language = 'en'
                s.processing.source_mode = workspace.source_mode
                s.regions[0].manual_peaks = []
                workspace.apply_recipe(s)
                workspace.load_path(str(paths[0]))
                settle(workspace)
                assert workspace.manual_peak_controls.pack_info()
                assert workspace.alignment_controls.vars['enabled'].get()
                assert workspace.alignment_controls.enabled_check.instate(['disabled'])
                assert workspace.fit_fields.widgets['threshold'].instate(['disabled'])
                assert not workspace.fit_fields.widgets['center_tolerance_ppm'].instate(['disabled'])
                workspace.manual_peak_edit.set(True)
                workspace.toggle_manual_peaks()
                assert not workspace.selectors  # clicks cannot create/change regions
                before_bounds = copy.deepcopy(workspace.session.integrals)
                click(workspace, 3.49)
                click(workspace, 3.605)
                assert len(workspace.session.regions[0].manual_peaks) == 2
                click(workspace, 3.49, button=3)
                assert len(workspace.session.regions[0].manual_peaks) == 1
                workspace.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', workspace.canvas, key='ctrl+z'))
                settle(workspace)
                assert len(workspace.session.regions[0].manual_peaks) == 2
                assert workspace.session.integrals == before_bounds
                workspace.manual_peak_text.set('3.49; 3.605')
                workspace.apply_manual_peak_text()
                settle(workspace)
                assert workspace.session.regions[0].manual_peaks == [3.49, 3.605]
                workspace.run_analysis()
                settle(workspace)
                assert workspace.analysis and not workspace.analysis.errors
                fit = workspace.analysis.results['fit']
                np.testing.assert_allclose(fit.centers, [3.5, 3.6], atol=1e-6)
                expected = 72.*(10000. if workspace.source_mode == 'pdata' else 1.)
                assert abs(workspace.analysis.integrals['first']/expected-1.) < .0003
                legend = workspace.detail_ax.get_legend_handles_labels()[1]
                assert 'Manual marks (initial)' in legend and 'Optimized centers' in legend
                assert 'ppm (fitted)' in workspace.result_text.get('1.0', 'end')
                workspace.set_language('ru')
                assert 'Ручная разметка (старт)' in workspace.detail_ax.get_legend_handles_labels()[1]
                workspace.set_language('en')
                assert workspace.read_settings().analysis_method == 'lorentzian_manual'
                output = target/f'{workspace.source_mode}.peaks.csv'
                original_dialog = workspace.dialog
                workspace.dialog = lambda *_args, **_kwargs: str(output)
                workspace.export_single()
                workspace.dialog = original_dialog
                with output.open(encoding='utf-8-sig', newline='') as stream:
                    assert 'center_ppm_optimized' in next(csv.reader(stream))
                workspace.session.save(target/'manual.json')
                assert Session.load(target/'manual.json').regions[0].manual_peaks == [3.49, 3.605]
                print(f'{workspace.source_mode}: real mark/remove/undo clicks, numeric marks, free-center fit, final-center integral, language and CSV: OK', flush=True)
                workspace.remember_for_batch()
                frozen = copy.deepcopy(workspace.batch_recipe)
                workspace.set_manual_peaks([3.48, 3.61])
                settle(workspace)
                assert workspace.batch_recipe == frozen
                for mode in ('global', 'regions'):
                    controls = workspace.alignment_controls
                    controls.vars['mode'].set(workspace.t(next(label for label, code in ALIGNMENT_MODES.items() if code == mode)))
                    workspace.batch_root.set(str(directory))
                    workspace.batch_output.set(str(target/f'{workspace.source_mode}.{mode}.csv'))
                    workspace.run_batch()
                    settle(workspace)
                    assert len(workspace.batch_rows) == 2
                    assert all(row['status'] != 'error' for row in workspace.batch_rows), workspace.batch_rows
                    recipe = Session.load(Path(workspace.batch_output.get()).with_suffix('.settings.json'))
                    assert recipe.alignment.enabled and recipe.alignment.mode == mode
                    assert recipe.regions[0].manual_peaks == [3.49, 3.605]
                    fitted, stored = load_analysis(workspace.batch_rows[1]['fit_file'])
                    assert stored == recipe and fitted.method == 'lorentzian_manual'
                    workspace.batch_tree.selection_set(workspace.batch_tree.get_children()[1])
                    workspace.review_selected_fit()
                    settle(workspace)
                    review = workspace.review_windows[-1]
                    review.window.withdraw()
                    assert 'ppm (fitted)' in review.details.get('1.0', 'end')
                    assert 'Optimized centers' in review.detail_ax.get_legend_handles_labels()[1]
                    workspace.batch_rows[1]['fit_file'] = ''
                    workspace.review_selected_fit()
                    settle(workspace)
                    reconstructed = workspace.review_windows[-1]
                    reconstructed.window.withdraw()
                    assert reconstructed.analysis.integrals == fitted.integrals
                    print(f'{workspace.source_mode}: {mode} mandatory alignment, frozen marks, batch .spfit and exact reconstruction: OK', flush=True)
    finally:
        for workspace in [app, *app.source_windows]:
            workspace.batch_cancel.set()
            workspace.executor.shutdown(wait=True, cancel_futures=True)
        root.destroy()


if __name__ == '__main__':
    main()
