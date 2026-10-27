"""Exercise independent raw/pdata windows and reusable direct region integration."""
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.core import Session, load_analysis
from spintegrate.gui import SpintegrateApp
from test_pdata_direct import write_pdata_fixture


def main():
    root = tk.Tk()
    root.withdraw()
    app = SpintegrateApp(root)
    children = []
    callback_errors = []
    root.report_callback_exception = lambda _kind, exc, _tb: callback_errors.append(exc)
    def raising(message):
        raise AssertionError(message)
    app.error = raising

    def settle(target, timeout=40):
        until = time.monotonic()+timeout
        while time.monotonic() < until:
            root.update()
            if callback_errors:
                raise AssertionError(callback_errors)
            popup = target.preprocess_window
            if (not target.jobs and not target.pending_preview and not target.preview_timer
                    and not (popup and popup.timer)):
                root.update()
                return
            time.sleep(.01)
        raise AssertionError(f'Window did not settle: {target.status.get()} / {list(target.jobs)}')

    try:
        app.load_demo()
        settle(app)
        app.run_analysis()
        settle(app)
        app.fit_fields.vars['analysis_label'].set('Прямое интегрирование областей')
        settle(app)
        app.run_analysis()
        settle(app)
        assert app.analysis.method == 'direct' and app.analysis.integrals['signal'] > 0
        assert app.analysis.results['Сигнал'].centers.size == 0
        app.fit_fields.vars['analysis_label'].set('Деконволюция Лоренцами')
        settle(app)
        app.run_analysis()
        settle(app)
        raw_analysis = app.analysis
        raw_recipe = app.read_settings()
        child = app.open_source_window('pdata')
        children.append(child)
        child.root.withdraw()
        child.error = raising
        child.confirm = lambda title, message: raising('Unexpected confirmation: ' + message)
        assert child is app.open_source_window('pdata')  # reopening focuses the same workspace
        child.root.withdraw()
        assert child is not app and child.root is not root
        assert child.source_mode == 'pdata'
        assert child.read_settings().processing.source_mode == 'pdata'
        assert child.read_settings().processing.lb_hz == 0.
        assert child.read_settings().processing.phase == 'none'
        child.set_language('en')
        app.set_language('ru')
        assert child.language == 'en' and app.language == 'ru'
        assert child.tabs.tab(child.single_tab, 'text') == 'Single spectrum'

        with tempfile.TemporaryDirectory(prefix='spintegrate_pdata_gui_') as directory:
            target = Path(directory)
            for expno, amplitude in [('2', 24.), ('10', 48.)]:
                write_pdata_fixture(target/'series'/expno/'pdata'/'1', np.full(128, amplitude))
            child.load_path(str(target/'series'/'2'/'pdata'/'1'))
            settle(child)
            np.testing.assert_array_equal(child.spectrum.intensity, np.full(128, 24.))
            child.session.baseline_default.method = 'none'
            child.selection_mode.set('fit')
            child.region_name.set('processed_parent')
            child.bound_low.set('3.0')
            child.bound_high.set('10.0')
            child.add_region()
            settle(child)
            child.selection_mode.set('integral')
            child.parent_region.set('processed_parent')
            child.region_name.set('processed_area')
            child.bound_low.set('4.123')
            child.bound_high.set('5.789')
            child.add_region()
            settle(child)
            child.fit_fields.vars['threshold'].set('')  # irrelevant fitting input must not block direct mode
            child.fit_fields.vars['analysis_label'].set('Прямое интегрирование областей')
            settle(child)
            assert child.read_settings().analysis_method == 'direct'
            child.run_analysis()
            settle(child)
            assert child.analysis.method == 'direct'
            assert abs(child.analysis.integrals['processed_area']-24.*(5.789-4.123)) < 1e-10
            assert child.analysis.results['processed_parent'].centers.size == 0
            assert app.analysis is raw_analysis
            assert app.read_settings() == raw_recipe
            child.session.save(target/'saved_recipe.settings.json')
            restored = Session.load(target/'saved_recipe.settings.json')
            assert restored.analysis_method == 'direct' and restored.processing.source_mode == 'pdata'
            child.remember_for_batch()
            child.batch_root.set(str(target/'series'))
            child.batch_output.set(str(target/'pdata.csv'))
            child.run_batch()
            settle(child)
            assert [r['spectrum_number'] for r in child.batch_rows] == ['2', '10']
            assert all(r['status'] != 'error' for r in child.batch_rows)
            stored, stored_recipe = load_analysis(child.batch_rows[0]['fit_file'])
            assert stored.method == 'direct'
            assert stored_recipe.analysis_method == 'direct'
            child.batch_tree.selection_set(child.batch_tree.get_children()[0])
            child.review_selected_fit()
            settle(child)
            review = child.review_windows[-1]
            review.window.withdraw()
            assert review.analysis.method == 'direct'
            labels = [line.get_label() for line in review.detail_ax.lines]
            assert 'Сумма Лоренцев' not in labels and 'Lorentzian sum' not in labels
            assert app.analysis is raw_analysis
            # A direct single-spectrum export is a numeric CSV, not an empty peak table.
            import spintegrate.gui as gui
            gui.filedialog.asksaveasfilename = lambda **_: str(target/'single_direct.csv')
            child.export_single()
            import csv
            with (target/'single_direct.csv').open(encoding='utf-8-sig') as stream:
                exported = list(csv.DictReader(stream))
            assert exported[0]['spectrum_number'] == '2'
            assert float(exported[0]['processed_area']) == child.analysis.integrals['processed_area']
            render = review.save_image()
            render.window.withdraw()
            assert render.layout_choices.get('Область + интеграл') == 'fit'
            gui.filedialog.asksaveasfilename = lambda **_: str(target/'direct.png')
            render.save()
            assert (target/'direct.png').is_file()
            print('Independent pdata window, imported data preservation and direct batch/review: OK', flush=True)

        assert child.close() is True
        assert child.closed
        assert root.winfo_exists() and app.analysis is raw_analysis
        print('Closing pdata leaves raw analysis intact: OK', flush=True)
    finally:
        for child in children:
            child.batch_cancel.set()
            child.executor.shutdown(wait=True, cancel_futures=True)
        app.batch_cancel.set()
        app.executor.shutdown(wait=True, cancel_futures=True)
        root.destroy()


if __name__ == '__main__':
    main()
