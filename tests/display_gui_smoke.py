"""Native-DPI backing-store bounds, adjustable UI and unchanged numerical fits."""
import copy
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.desktop_display import display_factor, enable_windows_dpi
from spintegrate.core import Session
from spintegrate.figure_export import render_fit_figure, save_figure
from spintegrate.gui import SpintegrateApp


def main():
    enable_windows_dpi()  # same entry path as Start_Spintegrate.cmd / main()
    root = tk.Tk()
    root.withdraw()
    app = SpintegrateApp(root)
    errors = []
    root.report_callback_exception = lambda _kind, exc, _tb: errors.append(exc)
    app.error = lambda message: (_ for _ in ()).throw(AssertionError(message))
    def settle():
        until = time.monotonic()+40
        while time.monotonic() < until:
            root.update()
            assert not errors, errors
            popup = app.preprocess_window
            if (not app.jobs and not app.pending_preview and not app.preview_timer
                    and not (popup and popup.timer)):
                root.update()
                return
            time.sleep(.01)
        raise AssertionError(f'GUI did not settle: {app.status.get()}')
    def bounds(canvas):
        root.update()
        canvas.draw()
        widget = canvas.get_tk_widget()
        size = widget.winfo_width(), widget.winfo_height()
        assert min(size) > 80, size
        np.testing.assert_allclose(canvas.get_width_height(physical=True), size, atol=1)
        renderer = canvas.get_renderer()
        assert abs(renderer.width-size[0]) < 1 and abs(renderer.height-size[1]) < 1
        for axis in canvas.figure.axes:
            extent = axis.get_window_extent(renderer)
            assert 0 <= extent.x0 < extent.x1 <= size[0]+1
            assert 0 <= extent.y0 < extent.y1 <= size[1]+1
    try:
        app.set_language('en')
        app.load_demo()
        settle()
        app.run_analysis()
        settle()
        analysis = app.analysis
        original = copy.deepcopy(analysis.integrals)
        curves = {name: result.fitted.copy() for name, result in analysis.results.items()}
        root.deiconify()
        settle()
        bounds(app.canvas)
        normal_width = app.canvas.get_tk_widget().winfo_width()
        app.toggle_sidebar()
        settle()
        bounds(app.canvas)
        assert app.canvas.get_tk_widget().winfo_width() > normal_width
        app.toggle_sidebar()
        for percent in (80, 100, 125, 150, 100):
            app.set_display_scale(percent)
            settle()
            bounds(app.canvas)
            assert app.analysis is analysis and analysis.integrals == original
            assert app.session.appearance.ui_scale_percent == percent
            assert app.overview_ax.title.get_fontsize() == 10*percent/100.
        scale = display_factor(root)
        for width, height in ((1200, 790), (1600, 960)):
            root.geometry(f'{round(width*scale)}x{round(height*scale)}')
            settle()
            bounds(app.canvas)
        app.set_display_scale(125)
        child = app.open_source_window('pdata')
        assert child.session.appearance.ui_scale_percent == 125
        child.root.withdraw()
        app.open_preprocess()
        settle()
        bounds(app.preprocess_window.canvas)
        app._show_fit_review(analysis, app.session, 'Synthetic display test')
        review = app.review_windows[-1]
        settle()
        bounds(review.canvas)
        app.set_display_scale(100)
        settle()
        bounds(app.canvas)
        bounds(app.preprocess_window.canvas)
        bounds(review.canvas)
        for name, result in analysis.results.items():
            np.testing.assert_array_equal(result.fitted, curves[name])
        app.set_display_scale(150)
        export = app.open_image_export(analysis, app.session, 'Сигнал')
        settle()
        assert export.save_button.winfo_ismapped()
        assert export.save_button.winfo_rooty()+export.save_button.winfo_height() <= export.window.winfo_rooty()+export.window.winfo_height()
        export.window.destroy()
        app.set_display_scale(100)
        with tempfile.TemporaryDirectory(prefix='spintegrate_display_') as folder:
            target = Path(folder)
            app.session.save(target/'settings.json')
            assert Session.load(target/'settings.json').appearance.ui_scale_percent == 100
            options = copy.deepcopy(app.session.appearance)
            options.export_width_mm, options.export_height_mm = 101.6, 76.2
            options.export_dpi, options.export_layout = 300, 'region'
            options.ui_scale_percent = 150
            figure = render_fit_figure(analysis, app.session, 'Сигнал', options)
            save_figure(target/'article.png', figure, options)
            with Image.open(target/'article.png') as image:
                assert image.size == (1200, 900)  # UI scale never changes article output
        print('Native-DPI raw/preview/review canvases, resize, 80–150% scale, sidebar, pdata inheritance and unchanged fits/export: OK', flush=True)
    finally:
        for workspace in [app, *app.source_windows]:
            workspace.executor.shutdown(wait=True, cancel_futures=True)
        root.destroy()


if __name__ == '__main__':
    main()
