"""Capture a manually refined phase once; transfer and reopen it without autophase."""
import copy
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

import nmrglue as ng
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.core import Session
from spintegrate.gui import SpintegrateApp
from test_batch_phase import ANGLES, phase_recipe, write_series


def main():
    root = tk.Tk()
    root.withdraw()
    app = SpintegrateApp(root)
    errors, calls = [], []
    root.report_callback_exception = lambda _kind, exc, _tb: errors.append(exc)
    original_phase = ng.proc_autophase.autops
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
    try:
        child = app.open_source_window('pdata')
        child.root.withdraw()
        child.error = fail
        with tempfile.TemporaryDirectory(prefix='spintegrate_batch_phase_') as folder:
            target = Path(folder)
            for workspace in (app, child):
                def phase(data, *_args, **_kwargs):
                    calls.append(1)
                    return data, (22., -38.)
                ng.proc_autophase.autops = phase
                s = phase_recipe(workspace.source_mode)
                s.appearance.language = 'en'
                paths = write_series(target/f'{workspace.source_mode}_series', workspace.source_mode)
                workspace.apply_recipe(s)
                workspace.load_path(str(paths[0]))
                settle(workspace)
                assert workspace.session.batch_phase.mode == 'per_spectrum'
                selected_curve = workspace.spectrum.complex_spectrum.copy()
                count = len(calls)
                workspace.batch_phase_controls.mode.set('Fixed from the selected spectrum')
                assert not workspace.session.batch_phase.captured
                workspace.remember_for_batch()  # initial capture can also be requested here
                settle(workspace)
                assert workspace.batch_recipe.batch_phase.captured
                workspace.freeze_phase_for_batch()
                assert workspace.batch_phase_controls.combo.instate(['disabled'])
                settle(workspace)
                profile = workspace.session.batch_phase
                assert (profile.p0, profile.p1) == ANGLES and profile.mode == 'fixed'
                assert len(calls) == count
                assert workspace.session.processing.phase == 'acme'
                assert 'No automatic phasing' in workspace.batch_phase_controls.summary.cget('text')
                workspace.set_language('ru')
                assert workspace.batch_phase_controls.mode.get() == 'Постоянная с выбранного спектра'
                workspace.set_language('en')
                workspace.remember_for_batch()
                settle(workspace)
                frozen = copy.deepcopy(workspace.batch_recipe)
                assert frozen.batch_phase == profile
                # Later single-spectrum edits must not change the frozen batch phase.
                workspace.processing_fields.vars['manual_phase_p0'].set('55')
                settle(workspace)
                assert workspace.batch_recipe == frozen
                assert workspace.spectrum.phases != ANGLES
                workspace.remember_for_batch()
                assert workspace.batch_recipe.batch_phase == profile
                frozen = copy.deepcopy(workspace.batch_recipe)
                ng.proc_autophase.autops = lambda *_args, **_kwargs: fail('Batch or reconstruction reran autophase')
                workspace.batch_root.set(str(paths[0].parent if workspace.source_mode == 'raw' else paths[0].parents[2]))
                workspace.batch_output.set(str(target/f'{workspace.source_mode}.csv'))
                workspace.run_batch()
                settle(workspace)
                rows = workspace.batch_rows
                assert len(rows) == 3 and all(row['status'] != 'error' for row in rows), rows
                assert [(row['phase_p0_deg'], row['phase_p1_deg']) for row in rows] == [ANGLES]*3
                np.testing.assert_allclose([row['integral__area']/rows[0]['integral__area'] for row in rows], [1., .6, -.3], rtol=1e-6)
                workspace.batch_tree.selection_set(workspace.batch_tree.get_children()[0])
                workspace.review_selected_fit()
                settle(workspace)
                review = workspace.review_windows[-1]
                review.window.withdraw()
                np.testing.assert_allclose(review.analysis.spectrum.complex_spectrum, selected_curve, atol=1e-7)
                saved_integrals = copy.deepcopy(review.analysis.integrals)
                workspace.batch_rows[0]['fit_file'] = ''
                workspace.review_selected_fit()
                settle(workspace)
                reconstructed = workspace.review_windows[-1]
                reconstructed.window.withdraw()
                assert reconstructed.analysis.integrals == saved_integrals
                recipe = Session.load(Path(workspace.batch_output.get()).with_suffix('.settings.json'))
                assert recipe == frozen
                workspace.raw = None  # recreate a session with only the saved recipe
                workspace.apply_recipe(recipe)
                settle(workspace)
                workspace.remember_for_batch()
                workspace.batch_output.set(str(target/f'{workspace.source_mode}.reopened.csv'))
                workspace.run_batch()
                settle(workspace)
                assert all(row['status'] != 'error' for row in workspace.batch_rows)
                assert [row['phase_p0_deg'] for row in workspace.batch_rows] == [ANGLES[0]]*3
                workspace.batch_phase_controls.mode.set('Independently for each spectrum')
                assert workspace.session.batch_phase.mode == 'per_spectrum'
                workspace.batch_phase_controls.mode.set('Fixed from the selected spectrum')
                assert workspace.session.batch_phase == profile
                print(f'{workspace.source_mode}: cached total-phase capture, frozen batch, negative signals, JSON-only rerun and exact review reconstruction: OK', flush=True)
    finally:
        ng.proc_autophase.autops = original_phase
        for workspace in [app, *app.source_windows]:
            workspace.executor.shutdown(wait=True, cancel_futures=True)
        root.destroy()


if __name__ == '__main__':
    main()
