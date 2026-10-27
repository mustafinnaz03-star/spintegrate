"""Physical raw/pdata apodization and a saved mouse-selected batch reference."""
import copy
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.core import BaselineSettings, FitRegion, IntegrationRegion, Session
from spintegrate.gui import SpintegrateApp
from test_pdata_apodization import physical_pair


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
        until = time.monotonic()+40
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
    try:
        with tempfile.TemporaryDirectory(prefix='spintegrate_apod_gui_') as folder:
            target = Path(folder)
            raw, pdata = physical_pair(target, -1.)
            raw_path = target/'raw.npz'
            np.savez(raw_path, fid=raw.fid, sf_mhz=raw.sf_mhz, sw_hz=raw.sw_hz, carrier_ppm=raw.carrier_ppm)
            recipe = Session(analysis_method='direct')
            recipe.processing.lb_hz = 0.
            recipe.regions = [FitRegion('parent', 6.4, 7.2, BaselineSettings(method='none'))]
            recipe.integrals = [IntegrationRegion('area', 'parent', 6.6, 7.)]
            app.apply_recipe(copy.deepcopy(recipe))
            app.load_path(str(raw_path))
            settle(app)
            child = app.open_source_window('pdata')
            child.root.withdraw()
            child.error = fail
            recipe.processing.source_mode = 'pdata'
            child.apply_recipe(copy.deepcopy(recipe))
            child.load_path(pdata.source)
            settle(child)
            for workspace in (app, child):
                workspace.open_preprocess()
                workspace.preprocess_window.window.withdraw()
                settle(workspace)
                workspace.preprocess_window.lb_var.set('2')
                settle(workspace)
            raw_preview = app.preprocess_window.cached_result[0]
            pdata_preview = child.preprocess_window.cached_result[0]
            np.testing.assert_allclose(pdata_preview.intensity, raw_preview.intensity, atol=5., rtol=1e-5)
            assert pdata_preview.intensity.max() > .5*child.preprocess_window.cached_result[1].intensity.max()
            assert pdata_preview.processing_metadata['additional_fft_convention'] == 'positive'
            print('Raw and opposite-sign pdata show the same physical LB=2 Hz broadening: OK', flush=True)
            for workspace in (app, child):
                workspace.processing_fields.vars['ppm_offset'].set('.125')
                settle(workspace)
                workspace.on_span(6.7, 7.1)
                workspace.anchor_reference()
                settle(workspace)
                settings = workspace.read_settings()
                assert settings.processing.reference_mode == 'window'
                assert abs(settings.processing.reference_low-6.575) < 1e-10
                assert abs(settings.processing.reference_high-6.975) < 1e-10
                expected = settings.processing.reference_ppm+settings.processing.ppm_offset
                mask = (workspace.spectrum.ppm > 6.7) & (workspace.spectrum.ppm < 7.1)
                observed = workspace.spectrum.ppm[mask][np.argmax(workspace.spectrum.intensity[mask])]
                assert abs(observed-expected) < 1e-8
                workspace.remember_for_batch()
                assert workspace.batch_recipe.processing.reference_mode == 'window'
                workspace.batch_recipe.save(target/f'{workspace.source_mode}.json')
                assert Session.load(target/f'{workspace.source_mode}.json') == workspace.batch_recipe
                workspace.set_language('en')
                assert 'Batch referencing: reference' in workspace.recipe_label.cget('text')
            print('Mouse-selected reference, offset conversion, JSON and frozen batch recipe in both windows: OK', flush=True)
    finally:
        for workspace in [app, *app.source_windows]:
            workspace.batch_cancel.set()
            workspace.executor.shutdown(wait=True, cancel_futures=True)
        root.destroy()


if __name__ == '__main__':
    main()
