"""Regression checks for phase locking, abs simplification and editable automatic names."""
import sys
import time
import tkinter as tk
from pathlib import Path

import nmrglue as ng
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.gui import SpintegrateApp


def main():
    root = tk.Tk()
    root.withdraw()
    errors = []
    root.report_callback_exception = lambda _kind, exc, _tb: errors.append(exc)
    original_phaser = ng.proc_autophase.autops
    estimates = []
    def phaser(data, fn, return_phases=False, **kwargs):
        estimates.append(np.asarray(data).copy())
        phase = np.array([20., -6.])
        result = ng.proc_base.ps(data, p0=phase[0], p1=phase[1])
        return (result, phase) if return_phases else result
    ng.proc_autophase.autops = phaser
    app = SpintegrateApp(root)
    def fail(message):
        raise AssertionError(message)
    app.error = fail
    app.confirm = lambda _title, message: fail('Unexpected confirmation: ' + message)
    def settle(target=app, timeout=30):
        until = time.monotonic()+timeout
        while time.monotonic() < until:
            root.update()
            if errors:
                raise AssertionError(errors)
            popup = target.preprocess_window
            if (not target.jobs and not target.pending_preview and not target.preview_timer
                    and not (popup and popup.timer)):
                root.update()
                return
            time.sleep(.01)
        raise AssertionError(f'GUI did not settle: {target.status.get()} / {list(target.jobs)}')
    def names():
        return {r.name for r in app.session.regions+app.session.integrals}
    try:
        app.load_demo()
        settle()
        assert 'abs_source' not in app.processing_fields.vars
        app.processing_fields.vars['phase_label'].set('ACME (nmrglue)')
        settle()
        assert len(estimates) == 1
        assert app.spectrum.phases == (20., -6.)
        app.open_preprocess()
        popup = app.preprocess_window
        popup.window.withdraw()
        settle()
        assert len(estimates) == 1  # LB=0 comparison must share the locked correction
        for lb in ('2.', '4.', '1.'):
            popup.lb_var.set(lb)
            settle()
            assert len(estimates) == 1
            assert app.spectrum.phases == (20., -6.)
        popup.fields.vars['lam'].set('2000000')
        settle()
        app.run_analysis()
        settle()
        assert len(estimates) == 1
        app.rephase_spectrum()
        settle()
        assert len(estimates) == 2
        assert app.spectrum.phases == (20., -6.)
        print('Shared phase correction survives live LB/baseline changes and fitting; explicit rephase works: OK', flush=True)

        # Selecting an existing region then dragging a new range must start a new name.
        app.region_tree.selection_set(app.session.regions[0].name)
        settle()
        app.selection_mode.set('fit')
        app.on_span(8.4, 8.0)
        auto_fit = app.region_name.get()
        assert auto_fit.startswith('region_') and auto_fit not in names()
        app.add_region()
        settle()
        assert auto_fit in names()
        assert app.region_name.get() not in names()
        app.selection_mode.set('integral')
        app.parent_region.set(auto_fit)
        app.on_span(8.1, 8.3)
        auto_area = app.region_name.get()
        assert auto_area.startswith('signal_') and auto_area not in names()
        app.add_region()
        settle()
        assert auto_area in names()
        assert app.region_name.get() not in names()

        # Explicit draft names survive a new selection; blank names are generated on save.
        app.selection_mode.set('fit')
        app.region_name.set('Custom region')
        app.on_span(9., 9.3)
        assert app.region_name.get() == 'Custom region'
        app.add_region()
        settle()
        assert 'Custom region' in names()
        app.selection_mode.set('fit')
        app.region_name.set('')
        app.bound_low.set('10.1')
        app.bound_high.set('10.4')
        before = names()
        app.add_region()
        settle()
        generated = names()-before
        assert len(generated) == 1 and next(iter(generated)).startswith('region_')
        assert len(names()) == len(app.session.regions)+len(app.session.integrals)
        app.region_tree.selection_set(auto_fit)
        settle()
        app.region_name.set('Renamed context')
        app.update_region()
        settle()
        assert 'Renamed context' in names() and auto_fit not in names()
        assert next(r for r in app.session.integrals if r.name == auto_area).parent == 'Renamed context'
        print('Unique processing/integral names, custom drafts, blank fallback and rename: OK', flush=True)

        # The simplified abs choice is always post-FFT magnitude.
        app.processing_fields.vars['phase_label'].set('abs')
        settle()
        assert app.read_settings().processing.abs_source == 'spectrum'
        assert np.min(app.spectrum.intensity) >= 0
        assert len(estimates) == 2
        child = app.open_source_window('pdata')
        child.root.withdraw()
        child.error = fail
        child.load_demo()
        settle(child)
        assert 'abs_source' not in child.processing_fields.vars
        child.processing_fields.vars['phase_label'].set('ACME (nmrglue)')
        settle(child)
        assert len(estimates) == 3
        child.open_preprocess()
        child.preprocess_window.window.withdraw()
        settle(child)
        child.preprocess_window.lb_var.set('3.')
        settle(child)
        assert len(estimates) == 3
        assert child.spectrum.phases == (20., -6.)
        print('Post-FFT abs and independent pdata phase lock: OK', flush=True)

        # Closing before idle selection callbacks run must cancel Tcl jobs.
        child.refresh_regions()
        pending = set(child._region_event_timers)
        assert pending
        child.close()
        assert not pending.intersection(root.tk.call('after', 'info'))
        root.update()
        assert not errors
        print('Immediate pdata close cancels deferred selection callbacks: OK', flush=True)
    finally:
        for child in app.source_windows:
            child.batch_cancel.set()
            child.executor.shutdown(wait=True, cancel_futures=True)
        app.batch_cancel.set()
        app.executor.shutdown(wait=True, cancel_futures=True)
        ng.proc_autophase.autops = original_phaser
        root.destroy()


if __name__ == '__main__':
    main()
