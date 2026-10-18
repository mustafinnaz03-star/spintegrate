"""Language switching, live color updates and article export through real widgets."""
import copy
import json
import re
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.gui import SpintegrateApp
from spintegrate.core import Session
from spintegrate.i18n import tr
import spintegrate.gui as gui


def main():
    root = tk.Tk()
    root.withdraw()
    app = SpintegrateApp(root)
    errors = []
    root.report_callback_exception = lambda kind, exc, tb: errors.append(exc)
    app.error = lambda message: (_ for _ in ()).throw(AssertionError(message))
    def settle():
        until = time.monotonic()+30
        while time.monotonic() < until:
            root.update()
            popup = app.preprocess_window
            if errors:
                raise AssertionError(errors)
            if (not app.jobs and not app.pending_preview and not app.preview_timer
                    and not (popup and popup.timer)):
                return
            time.sleep(.01)
        raise AssertionError('GUI failed to settle')
    try:
        app.set_language('en')
        assert app.tabs.tab(app.single_tab, 'text') == 'Single spectrum'
        app.load_demo()
        settle()
        assert app.language == 'en'
        assert app.processing_fields.vars['phase_label'].get() == 'No correction'
        assert app.read_settings().processing.phase == 'none'
        app.open_preprocess()
        app.preprocess_window.window.withdraw()
        settle()
        assert app.preprocess_window.ax_base.get_title(loc='left') == 'Local baseline'
        def check_static_labels(widget):
            source = getattr(widget, '_source_label', '')
            if re.search('[А-Яа-яЁё]', source) and tr(source, 'en') == source:
                assert source == 'Язык / Language' or source.startswith(('Стандарт:', 'Сигнал:')), source
            for child in widget.winfo_children():
                check_static_labels(child)
        check_static_labels(root)
        app.run_analysis()
        settle()
        analysis = app.analysis
        fitted = analysis.results['Сигнал'].fitted.copy()
        app.set_language('ru')
        assert app.tabs.tab(app.single_tab, 'text') == 'Один спектр'
        assert app.analysis is analysis
        app.set_language('en')
        assert app.analysis is analysis
        assert app.processing_fields.vars['reference_label'].get() == 'Bruker metadata'
        assert app.preprocess_window.ax_base.get_title(loc='left') == 'Local baseline'
        app.area_tree.selection_set('signal')
        settle()
        gui.colorchooser.askcolor = lambda **_: ((204, 121, 167), '#CC79A7')
        app.choose_integration_color()
        assert app.session.integrals[1].color == '#CC79A7'
        assert app.analysis is analysis
        np.testing.assert_array_equal(fitted, app.analysis.results['Сигнал'].fitted)
        with tempfile.TemporaryDirectory(prefix='spintegrate_presentation_') as directory:
            target = Path(directory)
            app.session.save(target/'settings.json')
            restored = Session.load(target/'settings.json')
            assert restored.appearance.language == 'en'
            assert restored.integrals[1].color == '#CC79A7'
            export = app.open_image_export(analysis, app.session, 'Сигнал')
            export.window.withdraw()
            export.fields.vars['language_label'].set('English')
            export.fields.vars['export_dpi'].set('600')
            export.fields.vars['export_width_mm'].set('101.6')
            export.fields.vars['export_height_mm'].set('76.2')
            export.fields.vars['layout_label'].set('Region only')
            check_static_labels(export.window)
            assert '2400' in export.info.get() and '1800' in export.info.get()
            gui.filedialog.asksaveasfilename = lambda **_: str(target/'article.png')
            export.save()
            with Image.open(target/'article.png') as image:
                assert image.size == (2400, 1800)
                assert abs(image.info['dpi'][0]-600) < .02
            assert app.session.appearance.export_dpi == 600
            assert app.analysis is analysis
            # Restoring Russian changes UI labels without changing the English export preference.
            app.set_language('ru')
            assert app.session.appearance.export_language == 'en'
            assert app.processing_fields.vars['phase_label'].get() == 'Без коррекции'
            app._show_fit_review(analysis, app.session, 'Сохранённая подгонка: demo.spfit')
            review = app.review_windows[-1]
            review.window.withdraw()
            review.selected_region.set('Сигнал')
            review.draw()
            review.color_area.set('signal')
            gui.colorchooser.askcolor = lambda **_: ((230, 159, 0), '#E69F00')
            review.choose_color()
            assert review.recipe.integrals[1].color == '#E69F00'
            assert app.session.integrals[1].color == '#CC79A7'
            np.testing.assert_array_equal(fitted, review.analysis.results['Сигнал'].fitted)
        print('Language switching, preserved fit, chosen colors and 600-DPI article export: OK', flush=True)
    finally:
        app.executor.shutdown(wait=True, cancel_futures=True)
        root.destroy()


if __name__ == '__main__':
    main()
