"""Exercise the actual Tk widgets, workers, previews and CSV export in hidden windows."""
import csv
import json
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.demo import write_demo_series
from spintegrate.gui import SpintegrateApp
import spintegrate.gui as gui


def main():
    root = tk.Tk()
    root.withdraw()
    app = SpintegrateApp(root)
    callback_errors = []
    root.report_callback_exception = lambda kind, exc, tb: callback_errors.append(exc)
    app.error = lambda message: (_ for _ in ()).throw(AssertionError(message))

    def settle(timeout=40):
        until = time.monotonic()+timeout
        while time.monotonic() < until:
            root.update()
            popup = app.preprocess_window
            if (not app.jobs and not app.pending_preview and not app.preview_timer
                    and not (popup and popup.timer)):
                root.update()
                assert not callback_errors, repr(callback_errors)
                return
            time.sleep(.01)
        raise AssertionError(f'UI work did not settle: {list(app.jobs)}, '
                             f'status={app.status.get()}, popup={popup.status.get() if popup else None}')

    try:
        app.load_demo()
        settle()
        assert app.spectrum is not None
        assert len(app.session.regions) == 2
        assert app.read_settings() == app.session
        print('Demo loading and local peak preview: OK', flush=True)

        # Dragging either direction fills normalized numerical bounds.
        app.on_span(8.4, 8.0)
        assert float(app.bound_low.get()) == 8.
        app.selection_mode.set('fit')
        app.region_name.set('extra')
        app.add_region()
        settle()
        assert app.selected_region == 'extra'
        app.selection_mode.set('integral')
        app.parent_region.set('extra')
        app.region_name.set('empty_signal')
        app.bound_low.set('8.1')
        app.bound_high.set('8.3')
        app.add_region()
        settle()
        assert len(app.session.integrals) == 3
        print('Manual region selection, saving and containment: OK', flush=True)

        app.region_tree.selection_set('Стандарт')
        root.update()
        app.open_preprocess()
        popup = app.preprocess_window
        popup.window.withdraw()
        settle()
        popup.lb_var.set('2.0')
        popup.fields.vars['method'].set('arpls')
        settle()
        assert app.session.processing.lb_hz == 2.
        assert len(popup.ax_apod.lines) == 2
        assert len(popup.ax_base.lines) == 3
        popup.apply_baseline()
        settle()
        standard = next(r for r in app.session.regions if r.name == 'Стандарт')
        assert standard.baseline.method == 'arpls'
        print('Live apodization, baseline preview and individual-region settings: OK', flush=True)

        app.run_analysis()
        settle()
        assert app.analysis is not None
        assert not app.analysis.errors
        assert len(app.integral_tree.get_children()) == 3
        app.remember_for_batch()
        threshold = app.batch_recipe.fit.threshold
        app.fit_fields.vars['threshold'].set(str(threshold*1.1))
        settle()
        assert app.analysis is None  # stale results cannot be exported
        assert app.batch_recipe.fit.threshold == threshold
        print('Deconvolution, integral table and frozen batch recipe: OK', flush=True)

        with tempfile.TemporaryDirectory(prefix='spintegrate_gui_') as directory:
            target = Path(directory)
            app.run_analysis()
            settle()
            gui.filedialog.asksaveasfilename = lambda **_: str(target/'peaks.csv')
            app.export_single()
            assert (target/'peaks.csv').is_file()
            assert (target/'peaks.integrals.csv').is_file()
            with (target/'peaks.integrals.csv').open(encoding='utf-8-sig') as stream:
                single_rows = list(csv.DictReader(stream))
            assert 'standard' in single_rows[0] and 'signal' in single_rows[0]
            assert len(single_rows) == 1
            gui.filedialog.asksaveasfilename = lambda **_: str(target/'single.spfit')
            app.save_single_fit()
            assert (target/'single.spfit').is_file()
            app.figure.savefig(Path.cwd()/'test_output'/'single_spectrum.png', dpi=130)
            popup.figure.savefig(Path.cwd()/'test_output'/'apodization_baseline.png', dpi=130)
            write_demo_series(target/'series')
            app.batch_root.set(str(target/'series'))
            app.batch_output.set(str(target/'batch.csv'))
            app.run_batch()
            settle()
            assert len(app.batch_rows) == 4
            assert all(r['status'] != 'error' for r in app.batch_rows)
            assert (target/'batch.csv').is_file()
            assert len(app.batch_tree.get_children()) == 4
            assert app.batch_tree.heading('area0')['text'] == 'standard'
            first = app.batch_tree.get_children()[0]
            cells = app.batch_tree.item(first, 'values')
            assert len(cells) == 6  # number, ID, status, three area columns
            assert '=' not in cells[3]
            float(cells[3])
            app.batch_tree.selection_set(first)
            app.review_selected_fit()
            settle()
            review = app.review_windows[-1]
            review.window.withdraw()
            assert review.analysis.integrals['signal'] == app.batch_rows[0]['integral__signal']
            assert len(review.detail_ax.lines) >= 3
            review.selected_region.set('Сигнал')
            review.draw()
            assert 'Сигнал' in review.detail_ax.get_title(loc='left')
            gui.filedialog.asksaveasfilename = lambda **_: str(target/'review.png')
            review.save_image()
            assert (target/'review.png').stat().st_size > 10000
            review.figure.savefig(Path.cwd()/'test_output'/'saved_fit_review.png', dpi=130)
            # A CSV opened after the analysis loads its own recipe and archives.
            gui.filedialog.askopenfilename = lambda **_: str(target/'batch.csv')
            app.open_batch_results()
            settle()
            app.batch_tree.selection_set(app.batch_tree.get_children()[0])
            # Current processing controls must not affect review of saved fits.
            app.fit_fields.vars['threshold'].set('1000000000')
            settle()
            before = app.read_settings()
            app.review_selected_fit()
            settle()
            assert app.read_settings() == before
            assert len(app.review_windows[-1].analysis.results['Сигнал'].centers) == 2
            app.review_windows[-1].window.withdraw()
            # Explicitly marked fallback for older batches with no .spfit.
            app.batch_rows[0]['fit_file'] = ''
            app.review_selected_fit()
            settle()
            fallback = app.review_windows[-1]
            fallback.window.withdraw()
            assert fallback.origin.startswith('ПОВТОРНЫЙ РАСЧЁТ')
            assert len(fallback.analysis.results['Сигнал'].centers) == 2
            gui.filedialog.askopenfilename = lambda **_: str(target/'single.spfit')
            app.open_fit_file()
            settle()
            assert app.review_windows[-1].analysis.integrals['signal'] > 0
            app.review_windows[-1].window.withdraw()
            print('Numeric CSV columns, archived fit review, PNG, CSV reopen and legacy fallback: OK', flush=True)
            recipe = target/'peaks.settings.json'
            gui.filedialog.askopenfilename = lambda **_: str(recipe)
            app.load_settings()
            settle()
            assert app.batch_recipe is not None
            assert app.read_settings() == app.session
            print('Single CSV, batch analysis/CSV and JSON reload: OK', flush=True)
        popup.close()
        settle()
        print('GUI SMOKE PASSED', flush=True)
    finally:
        app.batch_cancel.set()
        app.executor.shutdown(wait=True, cancel_futures=True)
        root.destroy()


if __name__ == '__main__':
    (Path.cwd()/'test_output').mkdir(exist_ok=True)
    main()
