"""Live baseline edits must reach JSON and every pdata spectrum without Apply."""
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.core import BaselineSettings, FitRegion, IntegrationRegion, Session, load_analysis
from spintegrate.gui import SpintegrateApp
from test_pdata_direct import pdata_recipe, write_pdata_fixture


def main():
    root = tk.Tk()
    root.withdraw()
    app = SpintegrateApp(root)
    errors = []
    root.report_callback_exception = lambda _type, exc, _tb: errors.append(exc)
    def fail(message):
        raise AssertionError(message)
    app.error = fail
    def settle(target):
        until = time.monotonic()+40
        while time.monotonic() < until:
            root.update()
            assert not errors, errors
            popup = target.preprocess_window
            if (not target.jobs and not target.pending_preview and not target.preview_timer
                    and not (popup and popup.timer)):
                root.update()
                return
            time.sleep(.01)
        raise AssertionError(f'GUI did not settle: {target.status.get()}')
    try:
        app.load_demo()
        settle(app)
        app.open_preprocess()
        app.preprocess_window.window.withdraw()
        settle(app)
        selected = next(r for r in app.session.regions if r.name == app.selected_region)
        assert app.preprocess_window.fields.vars['method'].get() == selected.baseline.method
        app.preprocess_window.fields.vars['method'].set('arpls')
        assert app.read_settings().regions[0].baseline.method == 'arpls'
        settle(app)
        print('Raw baseline controls immediately update the selected region: OK', flush=True)

        child = app.open_source_window('pdata')
        child.root.withdraw()
        child.error = fail
        child.confirm = lambda _title, message: fail('Unexpected confirmation: '+message)
        with tempfile.TemporaryDirectory(prefix='spintegrate_baseline_gui_') as folder:
            target = Path(folder)
            ppm = 12.-np.arange(4096)*10./4096
            for number, offset, slope in [(1, 2000., 100.), (2, 80000., 2300.)]:
                signal = 40000.*np.exp(-.5*((ppm-6.)/.035)**2)
                write_pdata_fixture(target/'series'/str(number)/'pdata'/'1',
                                    np.round(signal+offset+slope*(ppm-7.)), nc_proc=0)
            recipe = pdata_recipe()
            recipe.analysis_method = 'direct'
            recipe.regions = [FitRegion('left', 5.2, 6.8, BaselineSettings(method='none')),
                              FitRegion('right', 7.2, 8.8, BaselineSettings(method='modpoly'))]
            recipe.integrals = [IntegrationRegion('area', 'left', 5.82, 6.18)]
            child.apply_recipe(recipe)
            child.load_path(str(target/'series'/'1'/'pdata'/'1'))
            settle(child)
            child.open_preprocess()
            popup = child.preprocess_window
            popup.window.withdraw()
            settle(child)
            assert popup.fields.vars['method'].get() == 'none'
            popup.fields.vars['method'].set('asls')
            popup.fields.vars['p'].set('.001')
            popup.fields.vars['tol'].set('0.000001')
            # Do not wait for the debounce timer and do not click Apply.
            child.remember_for_batch()
            assert child.batch_recipe.regions[0].baseline.method == 'asls'
            assert child.batch_recipe.regions[1].baseline.method == 'modpoly'
            child.dialog = lambda _function, **_options: str(target/'recipe.json')
            child.save_settings()
            loaded = Session.load(target/'recipe.json')
            assert loaded.regions[0].baseline.method == 'asls'
            popup.fields.vars['lam'].set('-1')
            try:
                child.read_settings()
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid live baseline settings were silently ignored')
            popup.fields.vars['lam'].set('1000000')
            settle(child)
            child.region_tree.selection_set('right')
            settle(child)
            assert popup.fields.vars['method'].get() == 'modpoly'
            assert child.session.regions[0].baseline.method == 'asls'
            child.apply_recipe(loaded)
            assert child.read_settings() == loaded
            child.batch_root.set(str(target/'series'))
            child.batch_output.set(str(target/'results.csv'))
            child.run_batch()
            settle(child)
            assert len(child.batch_rows) == 2
            curves = []
            for row in child.batch_rows:
                assert row['status'] != 'error'
                result, saved_recipe = load_analysis(row['fit_file'])
                assert saved_recipe.regions[0].baseline.method == 'asls'
                assert abs(row['integral__area']-40000.*.035*np.sqrt(2*np.pi)) < 70.
                curves.append(result.results['left'].baseline)
            assert not np.allclose(*curves)
            print('Live pdata settings reach JSON and batch; per-spectrum curves differ: OK', flush=True)
    finally:
        for workspace in [app, *app.source_windows]:
            workspace.batch_cancel.set()
            workspace.executor.shutdown(wait=True, cancel_futures=True)
        root.destroy()


if __name__ == '__main__':
    main()
