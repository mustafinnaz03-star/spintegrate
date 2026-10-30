"""Tk desktop UI. All numerical work is submitted to workers; Tk stays on main thread."""
from __future__ import annotations

import copy
import csv
import json
import queue
import re
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from .core import (BASELINE_METHODS, PHASE_MODES, BaselineSettings, FitRegion,
                   IntegrationRegion, Session, analyze, baseline_correct,
                   batch_analyze, discover_spectra, load_analysis, load_batch_results,
                   direct_region, load_source, PhaseState, ProcessingSettings, pick_peaks, manual_peak_guesses, process, save_analysis,
                   validate_baseline, window_reference_index, write_batch_csv)
from .demo import demo_raw, demo_session, write_demo_series, write_pdata_demo_series
from .plotting import draw_spectrum
from .appearance import AREA_COLORS, AppearanceSettings, area_color, validate_appearance
from .figure_export import render_fit_figure, save_figure
from .i18n import TranslatedVar, canonical, set_label, tr, translate_widgets, widget_language
from .region_names import next_region_name
from .baseline_ui import BaselineControls
from .region_editing import RegionHistory, move_or_resize
from .alignment import AlignmentSettings, align_regions, align_spectrum
from .navigation import PlotNavigation
from .desktop_display import (SURFACE, UI_SCALES, SpectrumCanvas, apply_style, display_factor, enable_windows_dpi,
                              fit_window, scale_explicit_fonts, style_live_figure)


PHASE_LABELS = {'Без коррекции': 'none', 'ACME (nmrglue)': 'acme',
                'Peak minima (nmrglue)': 'peak_minima', 'abs': 'abs'}
REF_LABELS = {'Из метаданных Bruker': 'metadata', 'Самый сильный пик': 'strongest',
              'Пик в заданном ppm-окне': 'window'}
ANALYSIS_LABELS = {'Деконволюция Лоренцами': 'lorentzian',
                   'Лоренцы: ручной выбор пиков': 'lorentzian_manual',
                   'Прямое интегрирование областей': 'direct'}
WHEEL_MODES = {'Интенсивность': 'intensity', 'Масштаб ppm': 'ppm', 'Фаза P0': 'phase_p0', 'Фаза P1': 'phase_p1'}
ALIGNMENT_MODES = {'Единый сдвиг спектра': 'global', 'По областям обработки': 'regions'}


class SpectrumToolbar(NavigationToolbar2Tk):
    def home(self, *args):
        navigation = getattr(self.canvas, 'spectrum_navigation', None)
        if navigation is not None:
            navigation.reset()
        else:
            super().home(*args)


class WheelControls(ttk.Frame):
    def __init__(self, parent, app, reset):
        super().__init__(parent, padding=3)
        self.app = app
        toolbar = ttk.Frame(self)
        toolbar.pack(fill='x')
        ttk.Label(toolbar, text='Колесо мыши:').pack(side='left', padx=3)
        self.combo = ttk.Combobox(toolbar, textvariable=app.wheel_mode, state='readonly', width=16)
        self.combo.pack(side='left')
        ttk.Button(toolbar, text='Сбросить вид', command=reset).pack(side='left', padx=4)
        ttk.Button(toolbar, text='Опора фазы…', command=app.arm_phase_pivot).pack(side='left', padx=4)
        self.pivot_label = ttk.Label(self, text='Ctrl+колесо: ppm; Shift+колесо: сдвиг; двойной щелчок: сброс.', wraplength=650)
        self.pivot_label.pack(fill='x', padx=3)
        self.pivot_label.bind('<Configure>', lambda event: self.pivot_label.configure(wraplength=max(150, event.width-10)))
        app.wheel_controls.append(self)
        self.translate_choices(app.language)

    def translate_choices(self, language):
        value = canonical(self.app.wheel_mode.get())
        self.combo.configure(values=[tr(label, language) for label in WHEEL_MODES])
        self.app.wheel_mode.set(tr(value, language))


class AlignmentControls(ttk.LabelFrame):
    def __init__(self, parent, app):
        super().__init__(parent, text='Выравнивание спектров', padding=6)
        self.app, self.loading = app, True
        self.vars = dict(enabled=tk.BooleanVar(value=False), mode=tk.StringVar(),
                        max_shift_ppm=tk.StringVar(), min_correlation=tk.StringVar(),
                        low_ppm=tk.StringVar(), high_ppm=tk.StringVar())
        self.enabled_check = ttk.Checkbutton(self, text='Выравнивание спектров', variable=self.vars['enabled'])
        self.enabled_check.grid(row=0, column=0, sticky='w')
        self.combo = ttk.Combobox(self, textvariable=self.vars['mode'], state='readonly', width=25)
        self.combo.grid(row=0, column=1, columnspan=2, padx=6)
        self.entries = {}
        for name, label, row, column in [('max_shift_ppm', 'Макс. дрейф, ppm', 0, 3),
                ('min_correlation', 'Мин. корреляция', 0, 5), ('low_ppm', 'Окно: от', 1, 0), ('high_ppm', 'до', 1, 2)]:
            ttk.Label(self, text=label).grid(row=row, column=column, sticky='w', padx=4, pady=3)
            entry = ttk.Entry(self, textvariable=self.vars[name], width=10)
            entry.grid(row=row, column=column+1, sticky='w', padx=4)
            self.entries[name] = entry
        ttk.Label(self, text='Опора — первый спектр списка. Пустое окно — весь спектр; '
                  'для локального режима используются области обработки.',
                  wraplength=630, foreground='#65758a').grid(row=1, column=4, columnspan=3, sticky='w', padx=5)
        for variable in self.vars.values():
            variable.trace_add('write', lambda *_: self.changed())
        self.set_values(app.session.alignment)
        self.loading = False

    def translate_choices(self, language):
        loading, self.loading = self.loading, True
        self.combo.configure(values=[tr(label, language) for label in ALIGNMENT_MODES])
        self.vars['mode'].set(tr(canonical(self.vars['mode'].get()), language))
        self.loading = loading

    def set_values(self, settings):
        loading, self.loading = self.loading, True
        for key, variable in self.vars.items():
            value = next(label for label, code in ALIGNMENT_MODES.items() if code == settings.mode) if key == 'mode' else getattr(settings, key)
            variable.set(self.app.t(value) if key == 'mode' else ('' if value is None else value))
        self.translate_choices(self.app.language)
        self.loading = loading
        self.refresh_states()

    def refresh_states(self):
        required = self.required()
        if required and not self.vars['enabled'].get():
            loading, self.loading = self.loading, True
            self.vars['enabled'].set(True)
            self.loading = loading
        self.enabled_check.configure(state='disabled' if self.app.batch_running or required else 'normal')
        enabled = self.vars['enabled'].get() and not self.app.batch_running
        self.combo.configure(state='readonly' if enabled else 'disabled')
        for key, entry in self.entries.items():
            active = enabled and (key not in ('low_ppm', 'high_ppm') or ALIGNMENT_MODES[canonical(self.vars['mode'].get())] == 'global')
            entry.configure(state='normal' if active else 'disabled')

    def required(self):
        return (self.app.session.analysis_method == 'lorentzian_manual' or
                bool(self.app.batch_recipe and self.app.batch_recipe.analysis_method == 'lorentzian_manual'))

    def read(self, default, required=False):
        if not self.vars['enabled'].get() and not required:
            result = copy.deepcopy(default)
            result.enabled = False
            return result
        values = dict(enabled=True, mode=ALIGNMENT_MODES[canonical(self.vars['mode'].get())])
        for key in ('max_shift_ppm', 'min_correlation', 'low_ppm', 'high_ppm'):
            if key in ('low_ppm', 'high_ppm') and values['mode'] == 'regions':
                values[key] = None
                continue
            text = self.vars[key].get().strip()
            values[key] = None if key in ('low_ppm', 'high_ppm') and not text else float(text)
        if values['mode'] == 'regions':
            values.update(low_ppm=None, high_ppm=None)
        settings = AlignmentSettings(**values)
        settings.validate()
        return settings

    def changed(self):
        if self.loading or self.app.loading_settings:
            return
        self.refresh_states()
        try:
            settings = self.read(self.app.session.alignment, required=self.required())
            self.app.session.alignment = settings
            if self.app.batch_recipe is not None:
                self.app.batch_recipe.alignment = copy.deepcopy(settings)
                self.app._recipe_description()
        except ValueError as exc:
            self.app.status.set(str(exc))


class FieldPanel(ttk.Frame):
    """Compact labeled input rows with typed values and a single edit callback."""
    def __init__(self, parent, callback=None):
        super().__init__(parent, padding=8)
        self.vars, self.types, self.choice_widgets, self.widgets = {}, {}, {}, {}
        self.callback = callback
        self.columnconfigure(1, weight=1)

    def add(self, key, label, value, choices=None, kind=float, hint=None):
        row = len(self.vars)*2
        ttk.Label(self, text=label, wraplength=round(185*display_factor(self))).grid(
            row=row, column=0, sticky='w', padx=(0, 8), pady=3)
        var = tk.BooleanVar(value=value) if kind is bool else tk.StringVar(value=str(value))
        if kind is bool:
            widget = ttk.Checkbutton(self, variable=var)
        elif choices:
            widget = ttk.Combobox(self, textvariable=var, values=choices, state='readonly', width=20)
            self.choice_widgets[key] = (widget, tuple(choices))
        else:
            widget = ttk.Entry(self, textvariable=var, width=16)
        widget.grid(row=row, column=1, sticky='ew', pady=3)
        if hint:
            ttk.Label(self, text=hint, foreground='#65758a', wraplength=355,
                      font=('Segoe UI', 8)).grid(row=row+1, column=0, columnspan=2, sticky='w')
        self.vars[key], self.types[key] = var, kind
        self.widgets[key] = widget
        if self.callback:
            var.trace_add('write', lambda *_: self.callback())
        return var

    def values(self):
        result = {}
        for k, var in self.vars.items():
            try:
                result[k] = self.types[k](var.get())
            except (ValueError, tk.TclError):
                raise ValueError(f'Проверьте значение поля «{k}».') from None
        return result

    def set_values(self, values):
        for k, var in self.vars.items():
            if k in values:
                value = values[k]
                if k in self.choice_widgets:
                    value = tr(value, widget_language(self))
                var.set(value)

    def translate_choices(self, language):
        for key, (widget, choices) in self.choice_widgets.items():
            value = canonical(self.vars[key].get())
            widget.configure(values=[tr(v, language) for v in choices])
            translated = tr(value, language)
            if translated != self.vars[key].get():
                self.vars[key].set(translated)

    def enable(self, key, enabled):
        widget = self.widgets[key]
        widget.configure(state=('readonly' if key in self.choice_widgets else 'normal') if enabled else 'disabled')


def scrolled_text(parent, height=8):
    frame = ttk.Frame(parent)
    text = tk.Text(frame, height=height, wrap='word', font=('Consolas', 9),
                   relief='flat', background='#f5f7fa', padx=8, pady=8)
    scroll = ttk.Scrollbar(frame, command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    text.pack(side='left', fill='both', expand=True)
    scroll.pack(side='right', fill='y')
    return frame, text


def set_text(widget, value):
    widget._source_text = str(value)
    widget.configure(state='normal')
    widget.delete('1.0', 'end')
    widget.insert('1.0', tr(value, widget_language(widget)))
    widget.configure(state='disabled')


class ScrollPanel(ttk.Frame):
    """Scrollable settings page for smaller screens / enlarged Windows fonts."""
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background=SURFACE)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.body = ttk.Frame(self.canvas)
        self.item = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.body.bind('<Configure>', lambda _: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure(self.item, width=e.width))
        self.canvas.bind_all('<MouseWheel>', self._wheel, add='+')

    def _wheel(self, event):
        # Only the settings page below the pointer receives the wheel event.
        if not self.winfo_exists() or not self.winfo_ismapped():
            return
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget == self:
                self.canvas.yview_scroll(-int(event.delta/120), 'units')
                return
            widget = getattr(widget, 'master', None)


class SpintegrateApp:
    def __init__(self, root: tk.Tk, source_mode='raw', parent_app=None):
        if source_mode not in ('raw', 'pdata'):
            raise ValueError('Unknown spectrum source')
        self.root = root
        root._spintegrate_source_window = True
        self.source_mode = source_mode
        self.parent_app = parent_app
        self.source_windows = []
        self.closed = False
        self.poll_timer = None
        self.language = 'ru'
        root._spintegrate_language = self.language
        root.title('Spintegrate — анализ из pdata' if source_mode == 'pdata' else 'Spintegrate — анализ спектров ЯМР (raw)')
        self.session = Session()
        if parent_app:
            self.session.appearance.ui_scale_percent = parent_app.session.appearance.ui_scale_percent
        apply_style(root, self.session.appearance.ui_scale_percent)
        fit_window(root)
        self.ui_scale_var = tk.StringVar(root, value=f'{self.session.appearance.ui_scale_percent}%')
        self.session.processing.source_mode = source_mode
        if source_mode == 'pdata':
            self.session.processing.lb_hz = 0.
            self.session.baseline_default.method = 'none'
        self.raw = self.spectrum = self.analysis = None
        self.loaded_filter = True
        self.loaded_procno = None
        self.source_path = None
        self.selected_region = None
        self.generation = 0
        self.preview_revision = 0
        self.preview_timer = None
        self.loading_settings = False
        self.jobs = {}
        self.pending_preview = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='spintegrate')
        self.queue = queue.Queue()
        self.batch_cancel = threading.Event()
        self.batch_running = False
        self.batch_rows = []
        self.batch_recipe = None
        self.batch_results_recipe = None
        self.review_windows = []
        self.export_windows = []
        self.preview_local = None
        self.phase_state = PhaseState()
        self._last_auto_name = 'region_1'
        self._naming_sync = False
        self._region_events_suspended = 0
        self._region_event_timers = set()
        self.region_history = RegionHistory()
        self.region_dragging = False
        self.manual_peak_edit = tk.BooleanVar(root, value=False)
        self.manual_peak_history = []
        self.manual_peak_text = tk.StringVar(root)
        self._manual_peak_entry_region = None
        self.wheel_mode = tk.StringVar(root, value='Интенсивность')
        self.wheel_controls = []
        self.phase_pivot_armed = False
        self.preprocess_window = None
        self.status = TranslatedVar(root, value='Откройте папку эксперимента Bruker или демонстрационный спектр.')
        self._build()
        self.wheel_mode.trace_add('write', lambda *_: self.wheel_mode_changed())
        self._sync_widgets()
        self.set_language(self.language)
        self.poll_timer = root.after(80, self._poll)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def _build(self):
        top = ttk.Frame(self.root, padding=(12, 8))
        top.pack(fill='x')
        ttk.Label(top, text='Spintegrate', font=('Segoe UI', 18, 'bold')).pack(side='left')
        ttk.Button(top, text='Загрузить настройки JSON', command=self.load_settings).pack(side='right', padx=4)
        ttk.Button(top, text='Сохранить настройки JSON', command=self.save_settings).pack(side='right', padx=4)
        self.language_var = tk.StringVar(value='Русский')
        language_combo = ttk.Combobox(top, textvariable=self.language_var, values=('Русский', 'English'),
                                      state='readonly', width=10)
        language_combo.pack(side='right', padx=8)
        language_combo.bind('<<ComboboxSelected>>', lambda _: self.set_language(
                            'en' if self.language_var.get() == 'English' else 'ru'))
        ttk.Label(top, text='Язык / Language').pack(side='right')
        scale_combo = ttk.Combobox(top, textvariable=self.ui_scale_var,
                                  values=[f'{v}%' for v in UI_SCALES], state='readonly', width=5)
        scale_combo.pack(side='right', padx=6)
        scale_combo.bind('<<ComboboxSelected>>', lambda _: self.set_display_scale())
        ttk.Label(top, text='Масштаб UI').pack(side='right')
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=10)
        self.single_tab, self.batch_tab = ttk.Frame(self.tabs), ttk.Frame(self.tabs)
        self.tabs.add(self.single_tab, text='Один спектр')
        self.tabs.add(self.batch_tab, text='Много спектров')
        source = ttk.Frame(self.single_tab, padding=8)
        source.pack(fill='x')
        ttk.Button(source, text='Открыть pdata…' if self.source_mode == 'pdata' else 'Открыть Bruker…',
                   command=self.open_bruker).pack(side='left', padx=3)
        if self.source_mode == 'raw':
            ttk.Button(source, text='Открыть NPZ…', command=self.open_npz).pack(side='left', padx=3)
        ttk.Button(source, text='Демо', command=self.load_demo).pack(side='left', padx=3)
        ttk.Button(source, text='Аподизация / базовая линия…', command=self.open_preprocess).pack(side='left', padx=8)
        self.fit_button = ttk.Button(source, text='Выполнить анализ', command=self.run_analysis, style='Accent.TButton')
        self.fit_button.pack(side='left', padx=4)
        actions = ttk.Frame(self.single_tab, padding=(8, 0, 8, 6))
        actions.pack(fill='x')
        ttk.Button(actions, text='Экспорт картинки…', command=self.export_single_image).pack(side='left', padx=4)
        alternative = 'raw' if self.source_mode == 'pdata' else 'pdata'
        ttk.Button(actions, text='Анализировать raw…' if alternative == 'raw' else 'Анализировать pdata…',
                   command=lambda: self.open_source_window(alternative)).pack(side='left', padx=4)
        self.sidebar_button = ttk.Button(actions, text='Скрыть настройки', command=self.toggle_sidebar)
        self.sidebar_button.pack(side='left', padx=4)
        ttk.Button(actions, text='Развернуть окно', command=self.maximize_window).pack(side='left', padx=4)
        ttk.Button(actions, text='Запомнить → серия', command=self.remember_for_batch).pack(side='right', padx=4)
        self.source_label = ttk.Label(self.single_tab, text='Спектр не загружен', foreground='#65758a')
        self.source_label.pack(fill='x', padx=12, pady=(0, 6))
        split = self.main_split = ttk.Panedwindow(self.single_tab, orient='horizontal')
        split.pack(fill='both', expand=True)
        left, right = ttk.Frame(split, width=round(385*display_factor(self.root))), ttk.Frame(split)
        self.settings_sidebar, self.sidebar_visible = left, True
        left.pack_propagate(False)
        split.add(left, weight=0)
        split.add(right, weight=1)
        control_tabs = ttk.Notebook(left)
        control_tabs.pack(fill='both', expand=True, padx=5)
        pages = [ScrollPanel(control_tabs) for _ in range(3)]
        result_tab = ttk.Frame(control_tabs)
        processing_tab, region_tab, fit_tab = [p.body for p in pages]
        for frame, title in zip((*pages, result_tab),
                                ('Обработка', 'Области', 'Подгонка', 'Результат')):
            control_tabs.add(frame, text=title)
        self.control_tabs = control_tabs
        self.processing_fields = FieldPanel(processing_tab, self.settings_changed)
        self.processing_fields.pack(fill='x')
        p = self.session.processing
        add = self.processing_fields.add
        add('phase_label', 'Фаза', 'Без коррекции', list(PHASE_LABELS), str)
        add('p0', 'Начальная фаза P0, °', p.p0)
        add('p1', 'Начальная фаза P1, °', p.p1)
        add('phase_peak_width', 'Окно peak_minima, точки', p.phase_peak_width, kind=int)
        add('manual_phase_p0', 'Ручная поправка P0, °', p.manual_phase_p0,
            hint='Поправки добавляются к выбранной фазе до LB; колесо не запускает новую автофазировку.')
        add('manual_phase_p1', 'Ручная поправка P1, °', p.manual_phase_p1)
        add('reference_label', 'Привязка ppm', 'Из метаданных Bruker', list(REF_LABELS), str)
        add('reference_ppm', 'Референс, ppm', p.reference_ppm)
        add('reference_low', 'Окно референса: от', p.reference_low)
        add('reference_high', 'Окно референса: до', p.reference_high)
        add('ppm_offset', 'Сдвиг шкалы, ppm', p.ppm_offset)
        add('zero_fill_factor', 'Zero filling ×', p.zero_fill_factor, [1, 2, 4, 8], int)
        if self.source_mode == 'raw':
            add('remove_digital_filter', 'Удалить цифровой фильтр', p.remove_digital_filter, kind=bool)
        else:
            add('pdata_procno', 'Номер pdata (procno)', p.pdata_procno, kind=str)
        ttk.Button(processing_tab, text='Привязать к выделенному сигналу',
                   command=self.anchor_reference).pack(fill='x', padx=8, pady=5)
        ttk.Label(processing_tab,
                  text='Выделите мышью окно одного референсного сигнала. Привязка сохранит его текущую ppm '
                       'и будет находить этот сигнал заново в каждом спектре серии. '
                       'Режим «Из метаданных» сохраняет отдельную шкалу каждого файла.',
                  wraplength=365, foreground='#65758a').pack(padx=8, pady=5, anchor='w')
        self.rephase_button = ttk.Button(processing_tab, text='Повторить автофазировку', command=self.rephase_spectrum)
        self.rephase_button.pack(fill='x', padx=8, pady=5)
        ttk.Button(processing_tab, text='Сбросить ручную фазу', command=self.reset_manual_phase).pack(fill='x', padx=8, pady=3)
        self.phase_summary = ttk.Label(processing_tab, text='Фаза ещё не определена.', wraplength=365)
        self.phase_summary.pack(fill='x', padx=8, pady=3)
        explanation = ('P0/P1 — старт автоматического фазирования. «Без коррекции» показывает реальную часть FFT.\n'
                       'LB настраивается в отдельном окне.' if self.source_mode == 'raw' else
                       'По умолчанию используются 1r и ppm-шкала TopSpin. Фазы PHC0/PHC1 повторно не применяются. '
                       'Выбранные здесь фаза, LB и zero filling — дополнительная обработка; для неё нужен канал 1i.')
        ttk.Label(processing_tab, text=explanation, wraplength=365,
                  foreground='#65758a').pack(padx=8, pady=10, anchor='w')
        ttk.Label(processing_tab,
                  text='Автофаза определяется один раз без дополнительного LB. Изменение аподизации не запускает новый поиск. '
                       'В режиме abs используется модуль спектра после FFT.',
                  wraplength=365, foreground='#65758a').pack(padx=8, pady=(0, 8), anchor='w')
        self._build_regions(region_tab)
        self.fit_fields = FieldPanel(fit_tab, self.fit_settings_changed)
        self.fit_fields.pack(fill='x')
        f = self.session.fit
        add = self.fit_fields.add
        add('analysis_label', 'Метод анализа', 'Деконволюция Лоренцами', tuple(ANALYSIS_LABELS), str)
        add('threshold', 'Threshold (высота пика)', f.threshold,
            hint='Абсолютная высота после локальной коррекции базовой линии.')
        add('min_separation_points', 'Разделение, точки', f.min_separation_points, kind=int)
        add('signed', 'Искать и отрицательные пики', f.signed, kind=bool,
            hint='Выключено: A ≥ 0. Включено: амплитуды любого знака + диагностика компенсации.')
        add('initial_width_hz', 'Initial FWHM, Гц', f.initial_width_hz)
        add('min_width_hz', 'Минимум FWHM, Гц', f.min_width_hz)
        add('max_width_hz', 'Максимум FWHM, Гц', f.max_width_hz)
        add('center_tolerance_ppm', 'Допуск центра ±, ppm', f.center_tolerance_ppm,
            hint='Для ручных пиков: допустимое остаточное смещение после выравнивания.')
        add('max_nfev', 'Лимит вычислений / попытку', f.max_nfev, kind=int)
        add('restarts', 'Попытки с разными ширинами', f.restarts, kind=int)
        add('max_peaks', 'Максимум пиков / область', f.max_peaks, kind=int)
        add('loss', 'Функция ошибки', f.loss, ['linear', 'soft_l1'], str)
        add('residual_warning', 'Предупреждать при RMSE >', f.residual_warning)
        add('cancellation_warning', 'Компенсация: порог 0…1', f.cancellation_warning)
        self.fit_hint = ttk.Label(fit_tab, text='Центры фиксированы по peak-picking.\nШирина модели = HWHM (ppm); '
                  'старт = max(LB, initial FWHM) / (2 × SFO1).\nИнтеграл = Σ A × HWHM; '
                  'полная площадь Лоренца = π × этот интеграл.', wraplength=365,
                  foreground='#65758a')
        self.fit_hint.pack(padx=8, pady=10, anchor='w')
        self.integral_tree = ttk.Treeview(result_tab, columns=('integral',), show='tree headings', height=7)
        self.integral_tree.heading('#0', text='Сигнал')
        self.integral_tree.heading('integral', text='Интеграл')
        self.integral_tree.column('#0', width=145)
        self.integral_tree.column('integral', width=140)
        self.integral_tree.pack(fill='x', padx=6, pady=8)
        self.export_csv_button = ttk.Button(result_tab, text='Сохранить пики и интегралы CSV…', command=self.export_single)
        self.export_csv_button.pack(pady=4)
        ttk.Button(result_tab, text='Сохранить файл подгонки .spfit…', command=self.save_single_fit).pack(pady=4)
        frame, self.result_text = scrolled_text(result_tab, 17)
        frame.pack(fill='both', expand=True, padx=6, pady=8)
        set_text(self.result_text, 'После анализа здесь появятся остаток, ограничения ширин, '
                 'сходимость и обнаруженные пары компенсирующихся линий.')
        self.figure = Figure(figsize=(9, 7), dpi=100, facecolor='white', layout='constrained')
        grid = self.figure.add_gridspec(3, 1, height_ratios=(1, 2.5, .7))
        self.overview_ax = self.figure.add_subplot(grid[0])
        self.detail_ax = self.figure.add_subplot(grid[1])
        self.residual_ax = self.figure.add_subplot(grid[2], sharex=self.detail_ax)
        WheelControls(right, self, lambda: self.navigation.reset()).pack(fill='x')
        self.manual_peak_controls = ttk.Frame(right, padding=4)
        peak_row = ttk.Frame(self.manual_peak_controls)
        peak_row.pack(fill='x')
        ttk.Checkbutton(peak_row, text='Размечать пики', variable=self.manual_peak_edit,
                        command=self.toggle_manual_peaks).pack(side='left')
        ttk.Button(peak_row, text='Отменить пик', command=self.undo_manual_peaks).pack(side='left', padx=4)
        ttk.Button(peak_row, text='Очистить пики', command=lambda: self.set_manual_peaks([])).pack(side='left')
        self.manual_peak_label = ttk.Label(peak_row)
        self.manual_peak_label.pack(side='left', padx=6)
        peak_entry_row = ttk.Frame(self.manual_peak_controls)
        peak_entry_row.pack(fill='x')
        ttk.Label(peak_entry_row, text='Начальные ppm:').pack(side='left')
        self.manual_peak_entry = ttk.Entry(peak_entry_row, textvariable=self.manual_peak_text)
        self.manual_peak_entry.pack(side='left', fill='x', expand=True, padx=4)
        self.manual_peak_entry.bind('<Return>', lambda _: self.apply_manual_peak_text())
        ttk.Button(peak_entry_row, text='Применить пики', command=self.apply_manual_peak_text).pack(side='left')
        self.canvas = SpectrumCanvas(self.figure, master=right)
        self.toolbar = SpectrumToolbar(self.canvas, right)
        self.toolbar.update()
        self.canvas.mpl_connect('key_press_event', self.region_edit_key)
        self.canvas.mpl_connect('button_press_event', self.pick_phase_pivot)
        self.canvas.mpl_connect('button_press_event', self.on_manual_peak_click)
        self.selectors = []
        self.selection_label = ttk.Label(right, text='Во вкладке «Области» выберите режим, '
                                        'затем выделите диапазон мышью на графике.')
        self.selection_label.pack(side='bottom', fill='x', padx=8, pady=4)
        self.selection_label.bind('<Configure>', lambda event: self.selection_label.configure(wraplength=max(150, event.width-20)))
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.navigation = PlotNavigation(self.canvas, (self.overview_ax, self.detail_ax, self.residual_ax),
            self.spectrum_bounds, self.wheel_mode_code, self.adjust_phase_wheel,
            lambda: self.region_dragging or self.phase_pivot_armed)
        self._draw_empty()
        self._build_batch()
        ttk.Label(self.root, textvariable=self.status, padding=(12, 8),
                  wraplength=1350, anchor='w').pack(fill='x')

    def _build_regions(self, parent):
        ttk.Label(parent, text='1. Сохраните области для базовой линии и подгонки.\n'
                  '2. Внутри них сохраните области интегрирования.', wraplength=380).pack(padx=8, pady=8)
        self.selection_mode = tk.StringVar(value='fit')
        modes = ttk.Frame(parent)
        modes.pack(fill='x', padx=8)
        ttk.Radiobutton(modes, text='Обработка', variable=self.selection_mode, value='fit').pack(side='left')
        ttk.Radiobutton(modes, text='Интеграл', variable=self.selection_mode, value='integral').pack(side='left')
        self.region_name = tk.StringVar(value='region_1')
        self.selection_mode.trace_add('write', lambda *_: self.selection_mode_changed())
        self.bound_low, self.bound_high = tk.StringVar(value='3.0'), tk.StringVar(value='4.0')
        form = ttk.Frame(parent, padding=8)
        form.pack(fill='x')
        ttk.Label(form, text='Имя').grid(row=0, column=0, sticky='w')
        ttk.Entry(form, textvariable=self.region_name).grid(row=0, column=1, columnspan=2, sticky='ew')
        ttk.Label(form, text='Границы ppm').grid(row=1, column=0, sticky='w', pady=7)
        ttk.Entry(form, textvariable=self.bound_low, width=12).grid(row=1, column=1, padx=3)
        ttk.Entry(form, textvariable=self.bound_high, width=12).grid(row=1, column=2, padx=3)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, weight=1)
        self.parent_region = tk.StringVar()
        ttk.Label(form, text='Обработка для интеграла').grid(row=2, column=0, sticky='w')
        self.parent_combo = ttk.Combobox(form, textvariable=self.parent_region, state='readonly', width=17)
        self.parent_combo.grid(row=2, column=1, columnspan=2, sticky='ew')
        self.integration_color = tk.StringVar(value=AREA_COLORS[0])
        ttk.Label(form, text='Цвет интегрирования').grid(row=3, column=0, sticky='w', pady=6)
        self.color_swatch = tk.Label(form, textvariable=self.integration_color,
                                     background=self.integration_color.get(), width=10)
        self.color_swatch.grid(row=3, column=1, sticky='ew', padx=3)
        ttk.Button(form, text='Цвет области…', command=self.choose_integration_color).grid(row=3, column=2, sticky='ew')
        buttons = ttk.Frame(parent)
        buttons.pack(fill='x', padx=8)
        ttk.Button(buttons, text='Сохранить область', command=self.add_region).pack(side='left', padx=2)
        ttk.Button(buttons, text='Изменить выбранную', command=self.update_region).pack(side='left', padx=2)
        editing = ttk.Frame(parent)
        editing.pack(fill='x', padx=8, pady=6)
        self.edit_regions = tk.BooleanVar(value=False)
        ttk.Checkbutton(editing, text='Двигать границы на графике', variable=self.edit_regions,
                        command=self.toggle_region_editing).pack(side='left')
        self.region_undo_button = ttk.Button(editing, text='Отменить границы', command=self.undo_region_bounds, state='disabled')
        self.region_undo_button.pack(side='left', padx=5)
        self.region_tree = ttk.Treeview(parent, columns=('bounds', 'baseline'), show='tree headings', height=7)
        self.region_tree.heading('#0', text='Обработка')
        self.region_tree.heading('bounds', text='ppm')
        self.region_tree.heading('baseline', text='База')
        self.region_tree.column('#0', width=105)
        self.region_tree.column('bounds', width=120)
        self.region_tree.column('baseline', width=65)
        self.region_tree.pack(fill='x', padx=8, pady=8)
        self.region_tree.bind('<<TreeviewSelect>>', self.select_region)
        ttk.Button(parent, text='Удалить область обработки', command=self.delete_region).pack(anchor='w', padx=8)
        self.area_tree = ttk.Treeview(parent, columns=('parent', 'bounds'), show='tree headings', height=5)
        self.area_tree.heading('#0', text='Интеграл')
        self.area_tree.heading('parent', text='Обработка')
        self.area_tree.heading('bounds', text='ppm')
        self.area_tree.column('#0', width=100)
        self.area_tree.column('parent', width=100)
        self.area_tree.column('bounds', width=120)
        self.area_tree.pack(fill='x', padx=8, pady=8)
        self.area_tree.bind('<<TreeviewSelect>>', self.select_area)
        ttk.Button(parent, text='Удалить область интегрирования', command=self.delete_area).pack(anchor='w', padx=8)
        ttk.Label(parent, text='Выделение мышью заполняет границы. Область сохраняется кнопкой. '
                  'Границы можно вводить вручную в любом порядке.\n'
                  'Базовая линия применяется только в сохранённых областях обработки.',
                  wraplength=375, foreground='#65758a').pack(padx=8, pady=12)

    def _build_batch(self):
        frame = ttk.Frame(self.batch_tab, padding=16)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='Повторение анализа с сохранёнными параметрами',
                  font=('Segoe UI', 15, 'bold')).pack(anchor='w')
        self.recipe_summary_controls = ttk.Frame(frame)
        self.recipe_summary_controls.pack(fill='x', pady=5)
        self.recipe_summary = ttk.Label(self.recipe_summary_controls,
            text='Нажмите «Запомнить → серия» или загрузите JSON настроек.')
        self.recipe_summary.pack(side='left')
        self.recipe_details_open = False
        ttk.Button(self.recipe_summary_controls, text='Подробности рецепта', command=self.toggle_recipe_details).pack(side='left', padx=10)
        self.recipe_label = ttk.Label(frame, text='Нажмите «Запомнить → серия» во вкладке одного спектра '
                                     'или загрузите JSON.', wraplength=1200)
        self.batch_root = tk.StringVar()
        self.batch_output = tk.StringVar(value=str(Path.cwd() / 'integrals.csv'))
        self.batch_recursive = tk.BooleanVar(value=True)
        paths = ttk.Frame(frame)
        paths.pack(fill='x', pady=8)
        paths.columnconfigure(1, weight=1)
        ttk.Label(paths, text='Папка серии').grid(row=0, column=0, padx=6, sticky='w')
        ttk.Entry(paths, textvariable=self.batch_root).grid(row=0, column=1, sticky='ew', pady=6)
        ttk.Button(paths, text='Выбрать…', command=self.choose_batch_root).grid(row=0, column=2, padx=6)
        ttk.Label(paths, text='CSV результата').grid(row=1, column=0, padx=6, sticky='w')
        ttk.Entry(paths, textvariable=self.batch_output).grid(row=1, column=1, sticky='ew', pady=6)
        ttk.Button(paths, text='Выбрать…', command=self.choose_batch_output).grid(row=1, column=2, padx=6)
        ttk.Checkbutton(frame, text='Искать эксперименты во вложенных папках',
                        variable=self.batch_recursive).pack(anchor='w')
        self.alignment_controls = AlignmentControls(frame, self)
        self.alignment_controls.pack(fill='x', pady=6)
        buttons = ttk.Frame(frame)
        buttons.pack(fill='x', pady=12)
        ttk.Button(buttons, text='Показать найденные спектры', command=self.show_batch_paths).pack(side='left', padx=4)
        self.batch_button = ttk.Button(buttons, text='Анализировать серию → CSV', command=self.run_batch)
        self.batch_button.pack(side='left', padx=4)
        self.cancel_button = ttk.Button(buttons, text='Остановить после текущего спектра',
                                       command=self.batch_cancel.set, state='disabled')
        self.cancel_button.pack(side='left', padx=4)
        ttk.Button(buttons, text='Создать демо-серию…', command=self.create_demo_series).pack(side='right', padx=4)
        self.progress = ttk.Progressbar(frame, mode='determinate')
        self.progress.pack(fill='x', pady=4)
        review_buttons = ttk.Frame(frame)
        review_buttons.pack(fill='x', pady=(2, 8))
        ttk.Button(review_buttons, text='Показать фит выбранного спектра', command=self.review_selected_fit).pack(side='left', padx=4)
        ttk.Button(review_buttons, text='Открыть результаты CSV…', command=self.open_batch_results).pack(side='left', padx=4)
        ttk.Button(review_buttons, text='Открыть файл подгонки .spfit…', command=self.open_fit_file).pack(side='left', padx=4)
        table = ttk.Frame(frame)
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.batch_tree = ttk.Treeview(table, show='headings', height=11)
        vertical = ttk.Scrollbar(table, orient='vertical', command=self.batch_tree.yview)
        horizontal = ttk.Scrollbar(table, orient='horizontal', command=self.batch_tree.xview)
        self.batch_tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.batch_tree.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        self.configure_batch_columns(Session())
        self.batch_tree.bind('<<TreeviewSelect>>', self.show_batch_row)
        self.batch_tree.bind('<Double-1>', lambda _: self.review_selected_fit())
        block, self.batch_text = scrolled_text(frame, 5)
        block.pack(side='bottom', fill='x')
        table.pack(fill='both', expand=True, pady=8)
        set_text(self.batch_text, 'Каждый спектр обрабатывается независимо. Ошибки не останавливают '
                 'серию; недостоверные интегралы записываются как NaN. '
                 'CSV: номер спектра и отдельные числовые колонки областей. '
                 'Для проверки фита выберите строку и нажмите «Показать фит». '
                 'Рядом сохраняются JSON и папка .fits с файлами .spfit.')

    def _sync_widgets(self):
        self.region_history.clear()
        self.manual_peak_history.clear()
        self.manual_peak_edit.set(False)
        self.region_dragging = False
        self.phase_pivot_armed = False
        self.loading_settings = True
        p = asdict(self.session.processing)
        p['phase_label'] = next(k for k, v in PHASE_LABELS.items() if v == p['phase'])
        p['reference_label'] = next(k for k, v in REF_LABELS.items() if v == p['reference_mode'])
        self.processing_fields.set_values(p)
        self.fit_fields.set_values(asdict(self.session.fit))
        self.fit_fields.set_values({'analysis_label': next(k for k, v in ANALYSIS_LABELS.items()
                                                         if v == self.session.analysis_method)})
        self.alignment_controls.set_values(self.session.alignment)
        self.update_analysis_controls()
        self.loading_settings = False
        self.refresh_regions()
        self.prepare_region_name(force=True)
        self.update_phase_controls()
        if self.preprocess_window and self.preprocess_window.window.winfo_exists():
            self.preprocess_window.load_settings()
        if self.language != self.session.appearance.language:
            self.set_language(self.session.appearance.language)
        self.set_display_scale(self.session.appearance.ui_scale_percent)

    def t(self, text):
        return tr(text, self.language)

    def set_display_scale(self, percent=None):
        percent = int(self.ui_scale_var.get().rstrip('%')) if percent is None else percent
        if percent not in UI_SCALES:
            return
        owner = self
        while owner.parent_app:
            owner = owner.parent_app
        workspaces = list(dict.fromkeys([owner, *owner.source_windows, self]))
        apply_style(owner.root, percent)
        for workspace in workspaces:
            if workspace.closed:
                continue
            workspace.session.appearance.ui_scale_percent = percent
            workspace.ui_scale_var.set(f'{percent}%')
            scale_explicit_fonts(workspace.root, percent)
            if hasattr(workspace, 'settings_sidebar'):
                workspace.settings_sidebar.configure(width=round(385*display_factor(workspace.root)*percent/100.))
            if hasattr(workspace, 'navigation'):
                workspace.draw(local=workspace.preview_local)
                popup = workspace.preprocess_window
                if popup and popup.window.winfo_exists() and popup.cached_result:
                    popup.draw_cached()
                for review in workspace.review_windows:
                    if review.window.winfo_exists():
                        review.draw()

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self._sidebar_width = self.settings_sidebar.winfo_width()
            self.main_split.forget(self.settings_sidebar)
        else:
            self.main_split.insert(0, self.settings_sidebar, weight=0)
            self.main_split.sashpos(0, getattr(self, '_sidebar_width', round(385*display_factor(self.root))))
        self.sidebar_visible = not self.sidebar_visible
        set_label(self.sidebar_button, 'Скрыть настройки' if self.sidebar_visible else 'Показать настройки')

    def maximize_window(self):
        try:
            self.root.state('normal' if self.root.state() == 'zoomed' else 'zoomed')
        except tk.TclError:
            fit_window(self.root)

    def set_language(self, language):
        if language not in ('ru', 'en'):
            raise ValueError('Unsupported language')
        self.language = language
        self.root._spintegrate_language = language
        self.session.appearance.language = language
        self.language_var.set('English' if language == 'en' else 'Русский')
        previous = self.loading_settings
        self.loading_settings = True
        popup = self.preprocess_window
        was_loading = popup.loading if popup else None
        if popup:
            popup.loading = True
        try:
            translate_widgets(self.root, language)
        finally:
            self.loading_settings = previous
            if popup:
                popup.loading = was_loading
        self.status.refresh()
        self.draw(local=self.preview_local)
        for window in self.review_windows:
            if window.window.winfo_exists():
                window.draw()
        for window in self.export_windows:
            if window.window.winfo_exists():
                window.update_info()
        if popup and popup.window.winfo_exists():
            popup.status.refresh()
            if popup.cached_result is not None:
                popup.draw_cached()

    def dialog(self, function, **options):
        options.setdefault('parent', self.root)
        if 'title' in options:
            options['title'] = self.t(options['title'])
        if 'filetypes' in options:
            options['filetypes'] = [(self.t(label), pattern) for label, pattern in options['filetypes']]
        return function(**options)

    def confirm(self, title, message):
        return messagebox.askyesno(self.t(title), self.t(message), parent=self.root)

    def read_settings(self):
        if self.preprocess_window and self.preprocess_window.window.winfo_exists():
            self.preprocess_window.store_controls()
        s = copy.deepcopy(self.session)
        p = self.processing_fields.values()
        p['phase'] = PHASE_LABELS[canonical(p.pop('phase_label'))]
        p['reference_mode'] = REF_LABELS[canonical(p.pop('reference_label'))]
        for k, v in p.items():
            setattr(s.processing, k, v)
        s.processing.source_mode = self.source_mode
        s.processing.abs_source = 'spectrum'
        s.analysis_method = ANALYSIS_LABELS[canonical(self.fit_fields.vars['analysis_label'].get())]
        s.alignment = self.alignment_controls.read(s.alignment, required=s.analysis_method == 'lorentzian_manual')
        if s.analysis_method != 'direct':
            for k, variable in self.fit_fields.vars.items():
                if k == 'analysis_label' or (k in ('threshold', 'min_separation_points') and s.analysis_method == 'lorentzian_manual') or (k == 'center_tolerance_ppm' and s.analysis_method != 'lorentzian_manual'):
                    continue
                try:
                    setattr(s.fit, k, self.fit_fields.types[k](variable.get()))
                except (ValueError, tk.TclError):
                    raise ValueError(f'Проверьте значение поля «{k}».') from None
        s.validate()
        return s

    def invalidate(self):
        self.generation += 1
        self.analysis = None
        self.preview_local = None
        self.fit_button.configure(state='normal')
        for item in self.integral_tree.get_children():
            self.integral_tree.delete(item)
        set_text(self.result_text, 'Параметры или области изменились. Выполните анализ заново.')

    def settings_changed(self):
        if self.loading_settings:
            return
        self.update_phase_controls()
        self.invalidate()
        self.schedule_preview()

    def update_phase_controls(self):
        phase = PHASE_LABELS[canonical(self.processing_fields.vars['phase_label'].get())]
        self.rephase_button.configure(state='normal' if phase in ('acme', 'peak_minima') else 'disabled')
        for key in ('manual_phase_p0', 'manual_phase_p1'):
            self.processing_fields.enable(key, phase != 'abs')

    def wheel_mode_code(self):
        return WHEEL_MODES.get(canonical(self.wheel_mode.get()), 'intensity')

    def wheel_mode_changed(self):
        if not self.loading_settings:
            self.draw(local=self.preview_local)

    def spectrum_bounds(self):
        return (self.spectrum.ppm[0], self.spectrum.ppm[-1]) if self.spectrum is not None else None

    def navigation_context(self):
        p = self.session.processing
        region = next((r for r in self.session.regions if r.name == self.selected_region), None)
        return (id(self.raw), self.selected_region, (region.low, region.high) if region else None,
                self.session.analysis_method, p.reference_mode, p.reference_ppm, p.ppm_offset,
                p.reference_low, p.reference_high)

    def adjust_phase_wheel(self, mode, notches):
        if self.raw is None:
            return
        phase = PHASE_LABELS[canonical(self.processing_fields.vars['phase_label'].get())]
        if phase == 'abs':
            self.status.set('Ручная фазировка недоступна в режиме abs.')
            return
        if self.source_mode == 'pdata' and not self.raw.processing_metadata.get('has_imaginary', self.raw.processing_metadata.get('has_imag', False)):
            self.status.set('Для ручной фазировки pdata нужен файл 1i.')
            return
        key = 'manual_phase_p1' if mode == 'phase_p1' else 'manual_phase_p0'
        try:
            value = float(self.processing_fields.vars[key].get())
            value += notches*(5. if mode == 'phase_p1' else 1.)
            self.processing_fields.vars[key].set(f'{value:.6g}')
        except ValueError as exc:
            self.status.set(str(exc))

    def reset_manual_phase(self):
        self.loading_settings = True
        self.processing_fields.set_values(dict(manual_phase_p0=0., manual_phase_p1=0.))
        self.session.processing.phase_pivot_fraction = .5
        self.loading_settings = False
        self.phase_pivot_armed = False
        self.settings_changed()

    def arm_phase_pivot(self):
        if self.raw is None:
            return self.error('Сначала загрузите спектр.')
        self.manual_peak_edit.set(False)
        if PHASE_LABELS[canonical(self.processing_fields.vars['phase_label'].get())] == 'abs':
            self.status.set('Ручная фазировка недоступна в режиме abs.')
            return
        if self.source_mode == 'pdata' and not self.raw.processing_metadata.get('has_imaginary', self.raw.processing_metadata.get('has_imag', False)):
            self.status.set('Для ручной фазировки pdata нужен файл 1i.')
            return
        self.wheel_mode.set(self.t('Фаза P1'))
        self.phase_pivot_armed = True
        for selector in self.selectors:
            selector.set_active(False)
        self.status.set('Щёлкните по спектру, чтобы выбрать опору фазы P1.')

    def pick_phase_pivot(self, event):
        if not self.phase_pivot_armed or event.button != 1 or event.xdata is None:
            return
        spectrum = self.spectrum
        popup = self.preprocess_window
        if popup and event.canvas is popup.canvas and popup.cached_result:
            spectrum = popup.cached_result[0]
        if spectrum is None:
            return
        try:
            p = self.read_settings().processing
            fraction = float(np.clip((event.xdata-spectrum.ppm[0])/(spectrum.sw_hz/spectrum.sf_mhz), 0., 1.))
            self.loading_settings = True
            # Changing the pivot alone leaves the current phase curve unchanged.
            p0 = p.manual_phase_p0+p.manual_phase_p1*(fraction-p.phase_pivot_fraction)
            self.processing_fields.vars['manual_phase_p0'].set(f'{p0:.12g}')
            self.session.processing.phase_pivot_fraction = fraction
        except ValueError as exc:
            self.status.set(str(exc))
            return
        finally:
            self.loading_settings = False
            self.phase_pivot_armed = False
        self.settings_changed()

    def draw_phase_pivot(self, axes, spectrum):
        if spectrum is None:
            return
        pivot = spectrum.ppm[0]+self.session.processing.phase_pivot_fraction*spectrum.sw_hz/spectrum.sf_mhz
        p = self.session.processing
        message = (f'Ручная фаза P0={p.manual_phase_p0:.3f}°, P1={p.manual_phase_p1:.3f}°; опора {pivot:.4f} ppm. Ctrl — точно; Shift — P1.'
                   if self.wheel_mode_code().startswith('phase_') else
                   'Ctrl+колесо: ppm; Shift+колесо: сдвиг; двойной щелчок: сброс.')
        for controls in self.wheel_controls:
            if controls.winfo_exists():
                set_label(controls.pivot_label, message)
        if self.wheel_mode_code().startswith('phase_'):
            for axis in axes:
                if min(axis.get_xlim()) <= pivot <= max(axis.get_xlim()):
                    axis.axvline(pivot, color='#8957a7', ls=':', lw=1.)

    def rephase_spectrum(self):
        phase = PHASE_LABELS[canonical(self.processing_fields.vars['phase_label'].get())]
        if phase not in ('acme', 'peak_minima'):
            self.status.set('Для автофазировки выберите ACME или peak_minima.')
            return
        # Swap state rather than waiting on an optimization already running in a worker.
        self.phase_state = PhaseState()
        self.invalidate()
        self.schedule_preview()

    def fit_settings_changed(self):
        if self.loading_settings:
            return
        self.update_analysis_controls()
        self.invalidate()
        self.schedule_preview()

    def anchor_reference(self):
        if self.raw is None:
            return self.error('Сначала загрузите спектр.')
        if 'anchor_reference' in self.jobs:
            return
        try:
            recipe = self.read_settings()
            low, high = sorted((float(self.bound_low.get()), float(self.bound_high.get())))
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError('Выделите окно одного референсного сигнала на графике.')
        except ValueError as exc:
            return self.error(str(exc))
        source, phase_state = self.raw, self.phase_state
        def work():
            current = process(source, recipe.processing, phase_cache=phase_state)
            index = window_reference_index(current.ppm, current.intensity, low, high)
            shift = current.processing_metadata['reference_shift_ppm']
            # The search window is stored in the unreferenced input coordinates;
            # target+offset reproduces the selected spectrum's displayed position.
            return low-shift, high-shift, current.ppm[index]-recipe.processing.ppm_offset
        def ready(result):
            low_native, high_native, target = result
            self.loading_settings = True
            try:
                self.processing_fields.set_values(dict(reference_label='Пик в заданном ppm-окне',
                    reference_low=f'{low_native:.12g}', reference_high=f'{high_native:.12g}',
                    reference_ppm=f'{target:.12g}'))
            finally:
                self.loading_settings = False
            self.settings_changed()
            self.status.set('Привязка задана. При необходимости уточните «Референс, ppm», затем нажмите «Запомнить → серия».')
        self.status.set('Определяю референс по выбранному спектру…')
        self.submit('anchor_reference', work, ready, self.generation)

    def refresh_manual_peaks(self):
        if not hasattr(self, 'manual_peak_label'):
            return
        region = next((r for r in self.session.regions if r.name == self.selected_region), None)
        label = f'{region.name}: {len(region.manual_peaks)} пиков' if region else 'Сначала выберите область обработки.'
        set_label(self.manual_peak_label, label)
        # Do not overwrite an unapplied numeric draft while its entry has focus.
        name = region.name if region else None
        if name != self._manual_peak_entry_region or self.root.focus_get() is not self.manual_peak_entry:
            self.manual_peak_text.set('; '.join(f'{v:.8g}' for v in sorted(region.manual_peaks)) if region else '')
            self._manual_peak_entry_region = name

    def toggle_manual_peaks(self):
        if self.manual_peak_edit.get():
            self.edit_regions.set(False)
            self.region_dragging = False
            self.phase_pivot_armed = False
            if self.toolbar.mode:
                self.toolbar.zoom() if 'zoom rect' in str(self.toolbar.mode) else self.toolbar.pan()
        self.draw(local=self.preview_local)

    def set_manual_peaks(self, peaks, record=True):
        try:
            recipe = self.read_settings()
            region = next((r for r in recipe.regions if r.name == self.selected_region), None)
            if region is None:
                raise ValueError('Сначала выберите область обработки.')
            old = list(region.manual_peaks)
            region.manual_peaks = sorted(float(v) for v in peaks)
            recipe.validate()
            if old == region.manual_peaks:
                return
            if record:
                self.manual_peak_history.append((region.name, old))
                del self.manual_peak_history[:-30]
            self.session = recipe
            self.manual_peak_text.set('; '.join(f'{v:.8g}' for v in region.manual_peaks))
            self.invalidate()
            self.draw()
            self.schedule_preview()
        except ValueError as exc:
            self.status.set(str(exc))

    def apply_manual_peak_text(self):
        try:
            self.set_manual_peaks([float(v) for v in re.split(r'[;,\s]+', self.manual_peak_text.get().strip()) if v])
        except ValueError:
            self.status.set('Введите ppm через точку с запятой; десятичный разделитель — точка.')

    def undo_manual_peaks(self):
        if not self.manual_peak_history:
            return
        name, peaks = self.manual_peak_history[-1]
        if name != self.selected_region:
            self.status.set('Для отмены выберите область последнего изменённого пика.')
            return
        self.manual_peak_history.pop()
        self.set_manual_peaks(peaks, record=False)

    def on_manual_peak_click(self, event):
        if (not self.manual_peak_edit.get() or self.session.analysis_method != 'lorentzian_manual'
                or self.phase_pivot_armed or self.toolbar.mode or event.dblclick
                or event.inaxes not in (self.overview_ax, self.detail_ax) or event.xdata is None
                or event.button not in (1, 3) or self.spectrum is None):
            return
        region = next((r for r in self.session.regions if r.name == self.selected_region), None)
        if region is None:
            self.status.set('Сначала выберите область обработки.')
            return
        peaks = list(region.manual_peaks)
        if event.button == 1:
            if not region.low < event.xdata < region.high:
                self.status.set('Отметьте пик внутри выбранной области обработки.')
                return
            spacing = float(np.median(np.diff(self.spectrum.ppm)))
            if any(abs(v-event.xdata) < spacing*.25 for v in peaks):
                return
            peaks.append(float(event.xdata))
        elif peaks:
            pixels = event.inaxes.transData.transform(np.column_stack([peaks, np.zeros(len(peaks))]))[:, 0]
            nearest = int(np.argmin(abs(pixels-event.x)))
            if abs(pixels[nearest]-event.x) > 14:
                return
            peaks.pop(nearest)
        self.set_manual_peaks(peaks)

    def update_analysis_controls(self):
        mode = ANALYSIS_LABELS[canonical(self.fit_fields.vars['analysis_label'].get())]
        self.session.analysis_method = mode
        for key in self.fit_fields.vars:
            if key != 'analysis_label':
                active = mode != 'direct'
                if key in ('threshold', 'min_separation_points'):
                    active = mode == 'lorentzian'
                if key == 'center_tolerance_ppm':
                    active = mode == 'lorentzian_manual'
                self.fit_fields.enable(key, active)
        if hasattr(self, 'manual_peak_controls'):
            if mode == 'lorentzian_manual':
                self.session.alignment.enabled = True
                self.alignment_controls.vars['enabled'].set(True)
                self.manual_peak_controls.pack(before=self.canvas.get_tk_widget(), fill='x')
            else:
                self.manual_peak_controls.pack_forget()
                self.manual_peak_edit.set(False)
            self.alignment_controls.refresh_states()
            self.refresh_manual_peaks()
        if hasattr(self, 'fit_hint'):
            if mode == 'direct':
                text = ('Прямой интеграл = ∫ I(ppm) dppm после локальной коррекции базовой линии. '
                        'Поиск пиков и подгонка не выполняются. Отрицательные сигналы дают отрицательный вклад.')
            elif mode == 'lorentzian_manual':
                text = ('Отметьте пики в каждой области. Центры, амплитуды и ширины оптимизируются. '
                        'В серии выравнивание обязательно. Интеграл = Σ A × HWHM по итоговым центрам.')
            else:
                text = ('Центры фиксированы по peak-picking.\nШирина модели = HWHM (ppm); '
                        'старт = max(LB, initial FWHM) / (2 × SFO1).\nИнтеграл = Σ A × HWHM; '
                        'полная площадь Лоренца = π × этот интеграл.')
            set_label(self.fit_hint, text)
            set_label(self.export_csv_button, 'Сохранить интегралы CSV…' if mode == 'direct'
                      else 'Сохранить пики и интегралы CSV…')
            heading = 'Интеграл (инт. × ppm)' if mode == 'direct' else 'Σ A × HWHM'
            self.integral_tree._source_headings = {**getattr(self.integral_tree, '_source_headings', {}),
                                                    'integral': heading}
            self.integral_tree.heading('integral', text=self.t(heading))

    def open_source_window(self, source_mode):
        owner = self
        while owner.parent_app is not None:
            owner = owner.parent_app
        for app in [owner, *owner.source_windows]:
            if app.source_mode == source_mode and not app.closed:
                app.root.deiconify()
                app.root.lift()
                return app
        window = tk.Toplevel(owner.root)
        app = SpintegrateApp(window, source_mode=source_mode, parent_app=owner)
        app.session.appearance = copy.deepcopy(self.session.appearance)
        app.set_language(self.language)
        owner.source_windows.append(app)
        return app

    def schedule_preview(self):
        if self.preview_timer:
            self.root.after_cancel(self.preview_timer)
        self.preview_timer = self.root.after(220, self.preview)

    def submit(self, name, function, callback, generation=None):
        future = self.executor.submit(function)
        self.jobs[name] = (future, callback, generation)

    def _poll(self):
        if self.closed:
            return
        for name, (future, callback, generation) in list(self.jobs.items()):
            if not future.done():
                continue
            del self.jobs[name]
            if generation is not None and generation != self.generation:
                continue
            try:
                callback(future.result())
            except Exception as exc:
                if name == 'batch':
                    self._batch_unlock()
                if name == 'fit':
                    self.fit_button.configure(state='normal')
                self.error(str(exc))
        while True:
            try:
                message = self.queue.get_nowait()
            except queue.Empty:
                break
            if message[0] == 'batch_progress':
                _, number, total, row = message
                self.progress.configure(maximum=total, value=number)
                self.batch_rows.append(row)
                self.insert_batch_row(row)
                self.status.set(f"Серия: {number}/{total}, {row['spectrum_id']} — {row['status']}")
        if self.pending_preview and 'preview' not in self.jobs:
            task = self.pending_preview
            self.pending_preview = None
            self._submit_preview(*task)
        self.poll_timer = self.root.after(80, self._poll)

    def error(self, text):
        self.status.set('Ошибка: ' + text)
        messagebox.showerror('Spintegrate', self.t(text), parent=self.root)

    def open_bruker(self):
        path = self.dialog(filedialog.askdirectory,
                           title='Папка pdata с 1r и procs или папка эксперимента' if self.source_mode == 'pdata'
                           else 'Папка эксперимента с fid и acqus')
        if path:
            self.load_path(path)

    def open_npz(self):
        path = self.dialog(filedialog.askopenfilename, filetypes=[('FID NPZ', '*.npz')])
        if path:
            self.load_path(path)

    def load_path(self, path):
        try:
            recipe = self.read_settings()
        except ValueError as exc:
            return self.error(str(exc))
        if self.source_mode == 'pdata':
            selected = Path(path)
            if selected.is_file():
                selected = selected.parent
            if selected.parent.name.lower() == 'pdata' and selected.name.isdigit():
                self.loading_settings = True
                self.processing_fields.vars['pdata_procno'].set(selected.name)
                self.loading_settings = False
                recipe.processing.pdata_procno = selected.name
        self.invalidate()
        self.phase_state = PhaseState()
        set_label(self.phase_summary, 'Фаза ещё не определена.')
        self.source_path = str(path)
        self.raw = self.spectrum = None
        self.pending_preview = None
        self.status.set('Загрузка pdata…' if self.source_mode == 'pdata' else 'Загрузка FID…')
        remove_filter = recipe.processing.remove_digital_filter
        def loaded(raw):
            self.raw = raw
            self.loaded_filter = remove_filter
            self.loaded_procno = str(recipe.processing.pdata_procno)
            set_label(self.source_label, raw.source)
            self.preview()
        self.submit('load', lambda: load_source(path, recipe.processing), loaded, self.generation)

    def load_demo(self):
        appearance = copy.deepcopy(self.session.appearance)
        method = self.session.analysis_method
        self.session = demo_session()
        self.session.appearance = appearance
        self.session.analysis_method = method
        self.session.processing.source_mode = self.source_mode
        self._sync_widgets()
        self.invalidate()
        self.phase_state = PhaseState()
        set_label(self.phase_summary, 'Фаза ещё не определена.')
        self.raw = demo_raw()
        if self.source_mode == 'pdata':
            self.session.processing.lb_hz = 0.
            self.raw = process(self.raw, ProcessingSettings(lb_hz=0.))
            self.raw.source_type = 'pdata'
            self.raw.processing_metadata = {'has_imaginary': True, 'has_imag': True, 'imported_phases': [0., 0.]}
            self._sync_widgets()
        self.source_path = None
        set_label(self.source_label, self.raw.source)
        self.selected_region = self.session.regions[0].name
        self.region_tree.selection_set(self.selected_region)
        self.preview()

    def preview(self):
        self.preview_timer = None
        if self.raw is None:
            return
        try:
            recipe = self.read_settings()
        except ValueError as exc:
            self.status.set(str(exc))
            return
        if self.source_mode == 'raw' and self.source_path and recipe.processing.remove_digital_filter != self.loaded_filter:
            return self.load_path(self.source_path)
        if self.source_mode == 'pdata' and self.source_path and str(recipe.processing.pdata_procno) != self.loaded_procno:
            return self.load_path(self.input_path(recipe.processing))
        self.preview_revision += 1
        task = (self.raw, recipe, self.generation, self.preview_revision)
        if 'preview' in self.jobs:
            self.pending_preview = task
        else:
            self._submit_preview(*task)

    def _submit_preview(self, raw, recipe, generation, revision):
        self.status.set('Обновляю спектр и локальную базовую линию…')
        selected = next((r for r in recipe.regions if r.name == self.selected_region), None)
        phase_state = self.phase_state
        def work():
            spectrum = process(raw, recipe.processing, phase_cache=phase_state)
            local = None
            if selected:
                if recipe.analysis_method == 'direct':
                    result = direct_region(spectrum, selected)
                    x, y, corrected, baseline, notes = result.x, result.raw, result.corrected, result.baseline, result.notes
                    peaks = ([], [])
                else:
                    mask = (spectrum.ppm >= selected.low) & (spectrum.ppm <= selected.high)
                    x, y = spectrum.ppm[mask], spectrum.intensity[mask]
                    corrected, baseline, notes = baseline_correct(x, y, selected.baseline)
                    peaks = (manual_peak_guesses(x, corrected, selected) if recipe.analysis_method == 'lorentzian_manual'
                             else pick_peaks(x, corrected, recipe.fit))
                local = (x, y, corrected, baseline, peaks, notes)
            return spectrum, local
        def ready(result):
            if revision != self.preview_revision:
                return
            self.session.processing = recipe.processing
            self.session.fit = recipe.fit
            self.session.analysis_method = recipe.analysis_method
            self.spectrum, local = result
            set_label(self.phase_summary, f'Фаза P0={self.spectrum.phases[0]:.3f}°, P1={self.spectrum.phases[1]:.3f}°. '
                      'Сохраняется при изменении LB.')
            self.preview_local = local
            self.draw(local=local)
            self.status.set(f'Спектр готов. LB={recipe.processing.lb_hz:g} Гц; '
                            f'P0/P1={self.spectrum.phases[0]:.2f}/{self.spectrum.phases[1]:.2f}°. '
                            + (' | '.join(local[-1]) if local and local[-1] else 'Выберите области для анализа.'))
            if self.preprocess_window and self.preprocess_window.window.winfo_exists():
                self.preprocess_window.refresh()
        self.submit('preview', work, ready, generation)

    def input_path(self, settings):
        path = Path(self.source_path)
        if self.source_mode == 'pdata':
            if path.is_file():
                path = path.parent
            if path.parent.name.lower() == 'pdata':
                path = path.parent / str(settings.pdata_procno)
        return str(path)

    def _draw_empty(self):
        for ax, title in ((self.overview_ax, 'Весь спектр'), (self.detail_ax, 'Выбранная область'),
                          (self.residual_ax, 'Остаток: данные − подгонка')):
            ax.clear()
            ax.set_title(title, fontsize=10, loc='left')
            ax.set_xlabel('ppm')
            ax.grid(alpha=.15)
        self.canvas.draw_idle()

    def draw(self, local=None):
        if self.region_dragging:
            return
        self.navigation.before_draw(self.navigation_context())
        # Clear selectors before axes; retain references for interactive dragging.
        for selector in self.selectors:
            selector.disconnect_events()
        self.selectors.clear()
        draw_spectrum(self.overview_ax, self.detail_ax, self.residual_ax, self.spectrum,
                      self.session, self.selected_region, self.analysis, local, language=self.language)
        style_live_figure(self.figure, self.session.appearance.ui_scale_percent, shared_labels=True)
        self.refresh_manual_peaks()
        if self.spectrum is not None:
            if self.manual_peak_edit.get() and self.session.analysis_method == 'lorentzian_manual':
                set_label(self.selection_label, 'Ручные пики: левая кнопка добавляет, правая удаляет ближайший. Ctrl+Z — отмена; Esc — закончить.')
                self.draw_phase_pivot((self.overview_ax, self.detail_ax), self.spectrum)
                self.navigation.after_draw()
                self.canvas.draw_idle()
                return
            if not self.edit_regions.get():
                set_label(self.selection_label,
                    'Включите «Размечать пики» для кликов; выключите для выбора областей мышью.'
                    if self.session.analysis_method == 'lorentzian_manual' else
                    'Выделение мышью заполняет границы новой области.')
            kind = self.selection_mode.get()
            if kind == 'integral':
                selected = self.area_tree.selection()
                editable = next((a for a in self.session.integrals if selected and a.name == selected[0]), None)
            else:
                editable = next((r for r in self.session.regions if r.name == self.selected_region), None)
            if self.edit_regions.get():
                set_label(self.selection_label, 'Выберите сохранённую область в таблице. Потяните край для изменения размера '
                          'или середину для перемещения. Ctrl+Z отменяет изменение; Esc отменяет перетаскивание.')
                if editable is None:
                    self.draw_phase_pivot((self.overview_ax, self.detail_ax), self.spectrum)
                    self.navigation.after_draw()
                    self.canvas.draw_idle()
                    return
            if self.edit_regions.get() and kind == 'fit' and editable:
                margin = .12*(editable.high-editable.low)
                self.detail_ax.set_xlim(editable.high+margin, editable.low-margin)
            self.draw_phase_pivot((self.overview_ax, self.detail_ax), self.spectrum)
            self.navigation.after_draw()
            for ax in (self.overview_ax, self.detail_ax):
                editing = self.edit_regions.get() and editable is not None
                callback = (lambda low, high, k=kind, n=editable.name: self.edit_saved_bounds(k, n, low, high)) if editing else self.on_span
                span_pixels = abs(ax.transData.transform((editable.high, 0))[0]-ax.transData.transform((editable.low, 0))[0]) if editing else 40.
                selector = SpanSelector(ax, callback, 'horizontal', useblit=True,
                    props=dict(alpha=.25, facecolor='#e6a24a'), button=1,
                    interactive=editing, drag_from_anywhere=editing, ignore_event_outside=editing,
                    grab_range=max(1., min(8., span_pixels/4)),
                    onmove_callback=self.preview_region_bounds if editing else None, minspan=1e-10)
                if editing:
                    selector.extents = (editable.low, editable.high)
                self.selectors.append(selector)
        self.canvas.draw_idle()

    def preview_region_bounds(self, low, high):
        self.region_dragging = True
        self.bound_low.set(f'{min(low, high):.6f}')
        self.bound_high.set(f'{max(low, high):.6f}')

    def selection_mode_changed(self):
        self.prepare_region_name()
        if (hasattr(self, 'edit_regions') and self.edit_regions.get()
                and not self.loading_settings and not self._naming_sync):
            self.toggle_region_editing()

    def toggle_region_editing(self):
        if self.edit_regions.get():
            self.manual_peak_edit.set(False)
        self.region_dragging = False
        self.restore_region_bound_fields()
        if not self.edit_regions.get():
            set_label(self.selection_label, 'Выделение мышью заполняет границы новой области.')
        self.draw(local=self.preview_local)

    def restore_region_bound_fields(self):
        if self.selection_mode.get() == 'integral':
            selected = self.area_tree.selection()
            area = next((a for a in self.session.integrals if selected and a.name == selected[0]), None)
        else:
            area = next((r for r in self.session.regions if r.name == self.selected_region), None)
        if area:
            self.region_name.set(area.name)
            self.bound_low.set(f'{area.low:.6f}')
            self.bound_high.set(f'{area.high:.6f}')

    def edit_saved_bounds(self, kind, name, low, high):
        self.region_dragging = False
        if self.toolbar.mode or self.spectrum is None:
            return
        try:
            recipe = self.read_settings()
            edited = move_or_resize(recipe, kind, name, low, high,
                                    (self.spectrum.ppm[0], self.spectrum.ppm[-1]))
            if edited.regions == recipe.regions and edited.integrals == recipe.integrals:
                self.draw(local=self.preview_local)
                return
            self.region_history.record(recipe)
            self.session = edited
            self.manual_peak_history.clear()
            if self.preprocess_window:
                self.preprocess_window.load_settings()
            self.refresh_regions()
            if kind == 'integral':
                self.area_tree.selection_set(name)
                area = next(a for a in edited.integrals if a.name == name)
            else:
                area = next(r for r in edited.regions if r.name == name)
            self.region_name.set(name)
            self.bound_low.set(f'{area.low:.6f}')
            self.bound_high.set(f'{area.high:.6f}')
            self.invalidate()
            self.draw()
            self.schedule_preview()
        except ValueError as exc:
            self.status.set(str(exc))
            self.draw(local=self.preview_local)

    def undo_region_bounds(self):
        self.region_dragging = False
        if not self.region_history.states:
            return
        try:
            self.session = self.region_history.undo(self.read_settings())
            self.manual_peak_history.clear()
            selected = self.area_tree.selection() if self.selection_mode.get() == 'integral' else ()
            if self.preprocess_window:
                self.preprocess_window.load_settings()
            self.refresh_regions()
            if selected:
                self.area_tree.selection_set(selected[0])
            area = next((a for a in self.session.integrals if selected and a.name == selected[0]), None)
            area = area or next((r for r in self.session.regions if r.name == self.selected_region), None)
            if area:
                self.bound_low.set(f'{area.low:.6f}')
                self.bound_high.set(f'{area.high:.6f}')
            self.invalidate()
            self.draw()
            self.schedule_preview()
        except ValueError as exc:
            self.status.set(str(exc))

    def region_edit_key(self, event):
        if self.manual_peak_edit.get():
            if event.key == 'escape':
                self.manual_peak_edit.set(False)
                self.draw(local=self.preview_local)
                return
            if event.key in ('ctrl+z', 'cmd+z'):
                self.undo_manual_peaks()
                return
        if event.key == 'escape' and self.phase_pivot_armed:
            self.phase_pivot_armed = False
            self.draw(local=self.preview_local)
            return
        if event.key == 'escape' and self.region_dragging:
            self.region_dragging = False
            self.restore_region_bound_fields()
            self.draw(local=self.preview_local)
        elif event.key in ('ctrl+z', 'cmd+z') and self.edit_regions.get():
            self.undo_region_bounds()

    def on_span(self, low, high):
        if self.toolbar.mode or abs(high-low) < 1e-9:
            return
        self.bound_low.set(f'{min(low, high):.6f}')
        self.bound_high.set(f'{max(low, high):.6f}')
        self.prepare_region_name()
        self._suspend_region_events()
        for tree in (self.region_tree, self.area_tree):
            tree.selection_remove(*tree.selection())
        set_label(self.selection_label, f'Выбрано {min(low, high):.6f}…{max(low, high):.6f} ppm. '
                                       'Нажмите «Сохранить область».')
        self.control_tabs.select(1)

    def _suspend_region_events(self):
        self._region_events_suspended += 1
        def resume():
            self._region_event_timers.discard(timer)
            self._region_events_suspended = max(0, self._region_events_suspended-1)
        timer = self.root.after_idle(resume)
        self._region_event_timers.add(timer)

    def prepare_region_name(self, force=False):
        if self.loading_settings or self._naming_sync or not hasattr(self, 'region_name'):
            return
        current = self.region_name.get().strip()
        existing = {r.name for r in [*self.session.regions, *self.session.integrals]}
        if force or not current or current == self._last_auto_name or current in existing:
            name = next_region_name(self.selection_mode.get(), self.session.regions, self.session.integrals)
            self._last_auto_name = name
            self.region_name.set(name)

    def refresh_regions(self):
        self.region_undo_button.configure(state='normal' if self.region_history.states else 'disabled')
        self._suspend_region_events()
        self.loading_settings = True
        for tree in (self.region_tree, self.area_tree):
            for item in tree.get_children():
                tree.delete(item)
        for r in self.session.regions:
            self.region_tree.insert('', 'end', iid=r.name, text=r.name,
                                    values=(f'{r.low:g}…{r.high:g}', r.baseline.method))
        for r in self.session.integrals:
            self.area_tree.insert('', 'end', iid=r.name, text=r.name,
                                  values=(r.parent, f'{r.low:g}…{r.high:g}'))
        names = [r.name for r in self.session.regions]
        self.parent_combo.configure(values=names)
        if self.parent_region.get() not in names:
            self.parent_region.set(names[0] if names else '')
        if self.selected_region not in names:
            self.selected_region = names[0] if names else None
        if self.selected_region:
            self.region_tree.selection_set(self.selected_region)
        self.loading_settings = False

    def _region_from_fields(self):
        low, high = sorted((float(self.bound_low.get()), float(self.bound_high.get())))
        name = self.region_name.get().strip()
        if not name:
            name = next_region_name(self.selection_mode.get(), self.session.regions, self.session.integrals)
            self.region_name.set(name)
        if self.selection_mode.get() == 'fit':
            return FitRegion(name, low, high, copy.deepcopy(self.session.baseline_default))
        return IntegrationRegion(name, self.parent_region.get(), low, high, self.integration_color.get())

    def choose_integration_color(self):
        _, color = colorchooser.askcolor(color=self.integration_color.get(),
                                         title=self.t('Цвет интегрирования'), parent=self.root)
        if not color:
            return
        self.integration_color.set(color)
        self.color_swatch.configure(background=color)
        selection = self.area_tree.selection()
        if self.selection_mode.get() == 'integral' and selection and selection[0] == self.region_name.get():
            area = next(r for r in self.session.integrals if r.name == selection[0])
            area.color = color
            self.draw(local=self.preview_local)

    def add_region(self):
        try:
            self.session = self.read_settings()
            s = copy.deepcopy(self.session)
            region = self._region_from_fields()
            target = s.regions if isinstance(region, FitRegion) else s.integrals
            target.append(region)
            s.validate()
            self.session = s
            self.region_history.clear()
            if isinstance(region, FitRegion):
                self.selected_region = region.name
            else:
                self.selected_region = region.parent
            self.integration_color.set(AREA_COLORS[len(s.integrals) % len(AREA_COLORS)])
            self.color_swatch.configure(background=self.integration_color.get())
            self.refresh_regions()
            self.prepare_region_name(force=True)
            for tree in (self.region_tree, self.area_tree):
                tree.selection_remove(*tree.selection())
            self.invalidate()
            self.preview()
        except (ValueError, KeyError) as exc:
            self.error(str(exc))

    def update_region(self):
        try:
            s = self.read_settings()
            new = self._region_from_fields()
            tree = self.region_tree if isinstance(new, FitRegion) else self.area_tree
            selection = tree.selection()
            if not selection:
                raise ValueError('Выберите область в соответствующей таблице.')
            old = selection[0]
            if isinstance(new, FitRegion):
                previous = next(r for r in s.regions if r.name == old)
                new.baseline = previous.baseline
                new.manual_peaks = list(previous.manual_peaks)
                s.regions = [new if r.name == old else r for r in s.regions]
                for area in s.integrals:
                    if area.parent == old:
                        area.parent = new.name
                self.selected_region = new.name
            else:
                s.integrals = [new if r.name == old else r for r in s.integrals]
            s.validate()
            self.session = s
            self.region_history.clear()
            self.manual_peak_history.clear()
            self.refresh_regions()
            self.invalidate()
            self.preview()
        except (ValueError, StopIteration) as exc:
            self.error(str(exc))

    def select_region(self, _=None):
        if self.loading_settings or self._region_events_suspended:
            return
        selection = self.region_tree.selection()
        if not selection:
            return
        self.selected_region = selection[0]
        r = next(r for r in self.session.regions if r.name == self.selected_region)
        self.parent_region.set(r.name)
        self._naming_sync = True
        self.selection_mode.set('fit')
        self.region_name.set(r.name)
        self._naming_sync = False
        self.bound_low.set(str(r.low))
        self.bound_high.set(str(r.high))
        if self.analysis:
            self.draw()
        else:
            self.preview()
        if self.preprocess_window and self.preprocess_window.window.winfo_exists():
            self.preprocess_window.refresh()

    def select_area(self, _=None):
        if self.loading_settings or self._region_events_suspended:
            return
        selection = self.area_tree.selection()
        if not selection:
            return
        r = next(r for r in self.session.integrals if r.name == selection[0])
        self._naming_sync = True
        self.selection_mode.set('integral')
        self.region_name.set(r.name)
        self._naming_sync = False
        self.parent_region.set(r.parent)
        self.bound_low.set(str(r.low))
        self.bound_high.set(str(r.high))
        self.selected_region = r.parent
        self.integration_color.set(area_color(r, self.session.integrals.index(r)))
        self.color_swatch.configure(background=self.integration_color.get())
        if self.analysis:
            self.draw()
        else:
            self.preview()

    def delete_region(self):
        selection = self.region_tree.selection()
        if not selection:
            return
        name = selection[0]
        dependent = [r for r in self.session.integrals if r.parent == name]
        if dependent and not self.confirm('Удалить область',
                    f'Также удалить {len(dependent)} связанных областей интегрирования?'):
            return
        self.session.regions = [r for r in self.session.regions if r.name != name]
        self.session.integrals = [r for r in self.session.integrals if r.parent != name]
        self.region_history.clear()
        self.manual_peak_history.clear()
        self.refresh_regions()
        self.invalidate()
        self.preview()

    def delete_area(self):
        selection = self.area_tree.selection()
        if selection:
            self.session.integrals = [r for r in self.session.integrals if r.name != selection[0]]
            self.region_history.clear()
            self.refresh_regions()
            self.invalidate()
            self.preview()

    def run_analysis(self):
        if self.raw is None:
            return self.error('Сначала загрузите спектр.')
        if 'fit' in self.jobs:
            self.status.set('Предыдущая подгонка ещё выполняется.')
            return
        try:
            recipe = self.read_settings()
            recipe.validate(require_regions=True)
        except ValueError as exc:
            return self.error(str(exc))
        self.session = recipe
        self.invalidate()
        self.fit_button.configure(state='disabled')
        self.status.set('Интегрирование областей…' if recipe.analysis_method == 'direct' else
                        'Подгонка областей… Интерфейс доступен; изменение параметров отменит показ старого результата.')
        raw = self.raw
        phase_state = self.phase_state
        path = self.source_path
        reload = bool(path and ((self.source_mode == 'raw' and self.loaded_filter != recipe.processing.remove_digital_filter)
                      or (self.source_mode == 'pdata' and str(recipe.processing.pdata_procno) != self.loaded_procno)))
        selected_path = self.input_path(recipe.processing) if reload else path
        def work():
            source = load_source(selected_path, recipe.processing) if reload else raw
            return analyze(process(source, recipe.processing, phase_cache=phase_state), recipe)
        self.submit('fit', work, self.analysis_ready, self.generation)

    def analysis_ready(self, analysis):
        self.analysis = analysis
        self.spectrum = analysis.spectrum
        self.fit_button.configure(state='normal')
        for name, value in analysis.integrals.items():
            self.integral_tree.insert('', 'end', text=name, values=(f'{value:.9g}',))
        method = next(k for k, v in ANALYSIS_LABELS.items() if v == analysis.method)
        lines = [f'Статус: {analysis.status}', 'Режим анализа: ' + method,
                 f'P0/P1: {analysis.spectrum.phases}', '']
        for name, result in analysis.results.items():
            d = result.diagnostics
            if analysis.method == 'direct':
                lines.append(f'{name}: прямое интегрирование, точек {len(result.x)}')
                lines.append('Единицы: интенсивность × ppm; численный интеграл по точным границам.')
                continue
            lines.append(f"{name}: {len(result.centers)} пиков; сошлось={result.success}")
            lines.append(f"RMSE={d.get('relative_rmse')}; nfev={d.get('nfev')}; condition={d.get('condition_number')}")
            lines.append(' №     ppm (оптим.)  A            HWHM ppm     A×HWHM' if analysis.method == 'lorentzian_manual' else
                         ' №     ppm (фикс.)   A            HWHM ppm     A×HWHM')
            if analysis.method == 'lorentzian_manual':
                lines.append('Начальные ppm: ' + ', '.join(f'{v:.6f}' for v in d.get('initial_centers_ppm', [])))
            for i, (c, a, w) in enumerate(zip(result.centers, result.amplitudes, result.widths), 1):
                lines.append(f'{i:2}  {c:11.6f} {a:12.5g} {w:12.5g} {a*w:12.5g}')
            lines.append('')
        lines.extend(analysis.notes)
        set_text(self.result_text, '\n'.join(lines))
        self.control_tabs.select(3)
        self.draw()
        self.status.set(f'Анализ завершён: {analysis.status}. '
                        'Подробная диагностика во вкладке «Результат».')

    def save_settings(self):
        try:
            recipe = self.read_settings()
        except ValueError as exc:
            return self.error(str(exc))
        path = self.dialog(filedialog.asksaveasfilename, defaultextension='.json',
                    initialfile='spintegrate.settings.json', filetypes=[('Настройки JSON', '*.json')])
        if path:
            try:
                recipe.save(path)
                self.session = recipe
                self.status.set('Настройки и области сохранены: ' + path)
            except Exception as exc:
                self.error(str(exc))

    def load_settings(self):
        path = self.dialog(filedialog.askopenfilename, filetypes=[('Настройки JSON', '*.json')])
        if not path:
            return
        try:
            loaded = Session.load(path)
            if loaded.processing.source_mode != self.source_mode:
                destination = self.open_source_window(loaded.processing.source_mode)
                destination.apply_recipe(loaded)
                return
            self.apply_recipe(loaded)
        except Exception as exc:
            self.error(str(exc))

    def apply_recipe(self, loaded):
        self.session = loaded
        self._sync_widgets()
        self.invalidate()
        self.batch_recipe = copy.deepcopy(self.session)
        self._recipe_description()
        self.preview()

    def remember_for_batch(self):
        try:
            recipe = self.read_settings()
            recipe.validate(require_regions=True)
            self.session = recipe
            self.batch_recipe = copy.deepcopy(recipe)
            self.alignment_controls.set_values(recipe.alignment)
            self._recipe_description()
            self.tabs.select(self.batch_tab)
            self.status.set('Параметры запомнены для серии. Последующие изменения одиночного анализа не изменят этот снимок.')
        except ValueError as exc:
            self.error(str(exc))

    def _recipe_description(self):
        s = self.batch_recipe
        set_label(self.recipe_summary, f'Рецепт: {len(s.regions)} областей обработки, {len(s.integrals)} интегралов.')
        intro = f'Источник спектра: {s.processing.source_mode}\nРежим анализа: '
        intro += next(k for k, v in ANALYSIS_LABELS.items() if v == s.analysis_method) + '\n'
        if s.processing.source_mode == 'pdata':
            intro += f'pdata/{s.processing.pdata_procno}\n'
        text = (f'Запомнено: {len(s.regions)} областей обработки, '
                 f'{len(s.integrals)} интегралов; фаза={s.processing.phase}; '
                 f'LB={s.processing.lb_hz:g} Гц; threshold={s.fit.threshold:g}. '
                 'Центры и амплитуды определяются заново в каждом спектре.' if s.analysis_method == 'lorentzian' else
                 f'Сохранено областей интегрирования: {len(s.integrals)}. Численный интеграл: интенсивность × ppm.')
        if s.analysis_method == 'lorentzian_manual':
            text = (f'Ручных пиков: {sum(len(r.manual_peaks) for r in s.regions)}; '
                    f'допуск центра ±{s.fit.center_tolerance_ppm:g} ppm. Центры, амплитуды и ширины оптимизируются. '
                    'Выравнивание в серии обязательно; интеграл = Σ A × HWHM.')
        p = s.processing
        reference = (('Исходная привязка ppm: метаданные каждого файла.' if s.alignment.enabled else
                      'Привязка серии: шкала каждого файла, без автоматического выравнивания.')
                     if p.reference_mode == 'metadata' else
                     f'Привязка серии: референс {p.reference_ppm:g} ppm; режим {p.reference_mode}.')
        if p.reference_mode == 'window':
            reference += f'\nОкно поиска референса: {p.reference_low:g}…{p.reference_high:g} ppm.'
        alignment = ('Выравнивание: выключено.' if not s.alignment.enabled else
                     f'Выравнивание к первому спектру: {s.alignment.mode}; максимум {s.alignment.max_shift_ppm:g} ppm.')
        set_label(self.recipe_label, intro + text + '\n' + reference + '\n' + alignment)

    def toggle_recipe_details(self):
        self.recipe_details_open = not self.recipe_details_open
        if self.recipe_details_open:
            self.recipe_label.pack(after=self.recipe_summary_controls, anchor='w', pady=5)
        else:
            self.recipe_label.pack_forget()

    def export_single(self):
        if self.analysis is None:
            return self.error('Сначала выполните анализ с текущими параметрами.')
        direct = self.analysis.method == 'direct'
        path = self.dialog(filedialog.asksaveasfilename, defaultextension='.csv',
                           initialfile='spectrum.integrals.csv' if direct else 'spectrum.peaks.csv')
        if not path:
            return
        try:
            if not direct:
                with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.writer(stream)
                    manual = self.analysis.method == 'lorentzian_manual'
                    writer.writerow(['region', 'peak', 'center_ppm_optimized' if manual else 'center_ppm_fixed', 'initial_amplitude',
                                     'amplitude', 'hwhm_ppm', 'fwhm_hz', 'amplitude_x_hwhm', 'converged'])
                    for name, result in self.analysis.results.items():
                        for i, (c, a0, a, w) in enumerate(zip(result.centers, result.initial_amplitudes,
                                                            result.amplitudes, result.widths), 1):
                            writer.writerow([name, i, c, a0, a, w, 2*w*self.spectrum.sf_mhz, a*w, result.success])
            integral_path = Path(path) if direct else Path(path).with_suffix('.integrals.csv')
            source = Path(self.analysis.spectrum.source)
            experiment = source.parent.parent if self.analysis.spectrum.source_type == 'pdata' and source.parent.name.lower() == 'pdata' else source
            row = dict(spectrum_number=experiment.stem if experiment.is_file() else experiment.name,
                       spectrum_id=source.relative_to(experiment.parent).as_posix()
                       if self.analysis.spectrum.source_type == 'pdata' else source.name)
            row.update({f'integral__{n}': v for n, v in self.analysis.integrals.items()})
            write_batch_csv(integral_path, [row], self.session)
            self.session.save(Path(path).with_suffix('.settings.json'))
            self.status.set(('Сохранены CSV интегралов и JSON настроек: ' if direct else
                             'Сохранены CSV пиков, CSV интегралов и JSON настроек: ') + path)
        except Exception as exc:
            self.error(str(exc))

    def save_single_fit(self):
        if self.analysis is None:
            return self.error('Сначала выполните анализ с текущими параметрами.')
        path = self.dialog(filedialog.asksaveasfilename, defaultextension='.spfit', initialfile='spectrum.spfit',
                                          filetypes=[('Подгонка Spintegrate', '*.spfit')])
        if path:
            try:
                save_analysis(path, self.analysis, self.session)
                self.status.set('Сохранён файл подгонки: ' + path)
            except Exception as exc:
                self.error(str(exc))

    def choose_batch_root(self):
        path = self.dialog(filedialog.askdirectory, title='Корневая папка серии')
        if path:
            self.batch_root.set(path)

    def choose_batch_output(self):
        path = self.dialog(filedialog.asksaveasfilename, defaultextension='.csv', initialfile='integrals.csv',
                                          filetypes=[('CSV', '*.csv')])
        if path:
            self.batch_output.set(path)

    def show_batch_paths(self):
        try:
            recipe = self.batch_recipe or self.read_settings()
            paths = discover_spectra(self.batch_root.get(), self.batch_recursive.get(),
                                     recipe.processing.source_mode, recipe.processing.pdata_procno)
            set_text(self.batch_text, f'Найдено {len(paths)} спектров:\n' + '\n'.join(map(str, paths)))
        except Exception as exc:
            self.error(str(exc))

    def create_demo_series(self):
        directory = self.dialog(filedialog.askdirectory, title='Папка для синтетической демо-серии')
        if directory:
            try:
                target = Path(directory) / ('spintegrate_pdata_demo' if self.source_mode == 'pdata' else 'spintegrate_demo')
                if target.exists() and any(target.iterdir()):
                    raise ValueError('Демо-файлы уже существуют. Выберите другую папку.')
                recipe = demo_session()
                recipe.processing.source_mode = self.source_mode
                recipe.analysis_method = self.session.analysis_method
                if self.source_mode == 'pdata':
                    recipe.processing.pdata_procno = self.read_settings().processing.pdata_procno
                    write_pdata_demo_series(target, recipe.processing.pdata_procno)
                else:
                    write_demo_series(target)
                self.batch_root.set(str(target))
                self.batch_output.set(str(target / 'integrals.csv'))
                self.batch_recipe = recipe
                self._recipe_description()
                self.status.set('Создана синтетическая серия из четырёх pdata.' if self.source_mode == 'pdata'
                                else 'Создана синтетическая серия из четырёх FID.')
            except Exception as exc:
                self.error(str(exc))

    def run_batch(self):
        if self.batch_running:
            return
        if 'load_batch_results' in self.jobs:
            return self.error('Дождитесь загрузки открываемого CSV перед запуском новой серии.')
        if self.batch_recipe is None:
            return self.error('Сначала «Запомнить → серия» или загрузить JSON настроек.')
        try:
            recipe = copy.deepcopy(self.batch_recipe)
            recipe.alignment = self.alignment_controls.read(recipe.alignment, required=recipe.analysis_method == 'lorentzian_manual')
            recipe.validate(require_regions=True)
            root = self.batch_root.get()
            paths = discover_spectra(root, self.batch_recursive.get(), recipe.processing.source_mode,
                                     recipe.processing.pdata_procno)
            if not paths:
                raise ValueError('Спектры не найдены.')
            out = Path(self.batch_output.get())
            if out.suffix.lower() != '.csv' or not out.parent.is_dir():
                raise ValueError('Выберите путь .csv в существующей папке.')
            outputs = [out, out.with_suffix('.settings.json'), out.with_suffix('.diagnostics.json'),
                       out.with_suffix('.fits')]
            if any(p.exists() for p in outputs) and not self.confirm(
                    'Заменить результаты?', 'CSV, JSON или папка .fits уже существуют. '
                    'Заменить результаты и файлы подгонок с совпадающими номерами?'):
                return
        except Exception as exc:
            return self.error(str(exc))
        self.batch_running = True
        self.alignment_controls.refresh_states()
        self.batch_cancel.clear()
        self.batch_rows.clear()
        self.batch_results_recipe = copy.deepcopy(recipe)
        self.configure_batch_columns(recipe)
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        self.progress.configure(value=0, maximum=len(paths))
        self.batch_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        recursive = self.batch_recursive.get()
        def progress(n, total, row):
            self.queue.put(('batch_progress', n, total, row))
        def finished(result):
            rows, cancelled = result
            self._batch_unlock()
            self.status.set(f"{'Остановлено' if cancelled else 'Готово'}: {len(rows)} спектров; CSV: {out}")
            set_text(self.batch_text, f'CSV: {out}\nОшибок: {sum(r["status"] == "error" for r in rows)}; '
                     f'предупреждений: {sum(r["status"] == "warning" for r in rows)}.\n'
                     'Двойной щелчок по строке открывает сохранённую подгонку. '
                     'Для просмотра после перезапуска используйте «Открыть результаты CSV…».')
        self.submit('batch', lambda: batch_analyze(root, recipe, out, progress,
                           self.batch_cancel.is_set, recursive), finished)

    def _batch_unlock(self):
        self.batch_running = False
        self.alignment_controls.refresh_states()
        self.batch_button.configure(state='normal')
        self.cancel_button.configure(state='disabled')

    def show_batch_row(self, _=None):
        selection = self.batch_tree.selection()
        if selection:
            i = self.batch_tree.index(selection[0])
            if i < len(self.batch_rows):
                row = self.batch_rows[i]
                set_text(self.batch_text, f"{row['spectrum_id']} — {row['status']}\n{row['warnings']}\n"
                         f"Файл подгонки: {row.get('fit_file') or 'не сохранён'}")

    def configure_batch_columns(self, recipe):
        columns = ['number', 'id', 'status'] + (['alignment', 'alignment_score'] if recipe.alignment.enabled else []) + [f'area{i}' for i in range(len(recipe.integrals))]
        self.batch_tree.configure(columns=columns)
        for key, label, width in [('number', 'Номер спектра', 110), ('id', 'ID спектра', 200),
                                  ('status', 'Статус', 95)]:
            self.batch_tree.heading(key, text=label)
            self.batch_tree.column(key, width=width, minwidth=75, stretch=False)
        if recipe.alignment.enabled:
            self.batch_tree.heading('alignment', text='Выравнивание, ppm')
            self.batch_tree.column('alignment', width=155, minwidth=100)
            self.batch_tree.heading('alignment_score', text='Корреляция')
            self.batch_tree.column('alignment_score', width=95, minwidth=80)
        for i, area in enumerate(recipe.integrals):
            self.batch_tree.heading(f'area{i}', text=area.name)
            self.batch_tree.column(f'area{i}', width=155, minwidth=100)
        translate_widgets(self.batch_tree, self.language)

    def insert_batch_row(self, row):
        recipe = self.batch_results_recipe
        values = [row['spectrum_number'], row['spectrum_id'], row['status']]
        if recipe.alignment.enabled:
            value = row.get('alignment_shift_ppm') if recipe.alignment.mode == 'global' else row.get('alignment_max_abs_shift_ppm')
            values.append((f'{value:+.6g}' if recipe.alignment.mode == 'global' else f'max |Δ| {value:.6g}')
                          if isinstance(value, (float, int)) else '')
            score = row.get('alignment_correlation')
            values.append(f'{score:.3f}' if isinstance(score, (float, int)) else '')
        values.extend(f"{row[f'integral__{r.name}']:.9g}" for r in recipe.integrals)
        self.batch_tree.insert('', 'end', values=values)

    def open_batch_results(self):
        if self.batch_running:
            return self.error('Дождитесь окончания или остановите текущую серию перед открытием другого CSV.')
        path = self.dialog(filedialog.askopenfilename, title='Открыть CSV вместе с JSON и папкой .fits',
                                          filetypes=[('Результаты Spintegrate', '*.csv')])
        if not path:
            return
        self.status.set('Чтение результатов серии…')
        def ready(result):
            rows, recipe = result
            self.batch_rows = rows
            self.batch_results_recipe = recipe
            self.configure_batch_columns(recipe)
            for item in self.batch_tree.get_children():
                self.batch_tree.delete(item)
            for row in rows:
                self.insert_batch_row(row)
            self.progress.configure(maximum=max(1, len(rows)), value=len(rows))
            self.tabs.select(self.batch_tab)
            self.status.set(f'Загружено {len(rows)} спектров из {path}. Выберите строку для проверки подгонки.')
        self.submit('load_batch_results', lambda: load_batch_results(path), ready)

    def open_fit_file(self):
        path = self.dialog(filedialog.askopenfilename, title='Открыть сохранённую подгонку без повторного расчёта',
                                          filetypes=[('Подгонка Spintegrate', '*.spfit')])
        if path:
            self._open_fit_path(path)

    def _open_fit_path(self, path):
        self.status.set('Загрузка сохранённой подгонки…')
        def ready(result):
            analysis, recipe = result
            self._show_fit_review(analysis, recipe, f'Сохранённая подгонка: {path}')
        self.submit('review', lambda: load_analysis(path), ready)

    def review_selected_fit(self):
        selection = self.batch_tree.selection()
        if not selection:
            return self.error('Сначала выберите спектр в таблице серии.')
        row = self.batch_rows[self.batch_tree.index(selection[0])]
        fit_file = row.get('fit_file')
        if fit_file and Path(fit_file).is_file():
            return self._open_fit_path(fit_file)
        # Older CSVs did not save fitted arrays. Recalculate only as a clearly
        # labeled fallback, using that series' recipe rather than current widgets.
        source = row.get('source_path')
        recipe = copy.deepcopy(self.batch_results_recipe)
        if not source or not Path(source).exists() or recipe is None:
            return self.error('Файл .spfit не найден и исходный спектр недоступен. '
                              'Откройте сохранённый .spfit или верните исходный FID для старой серии.')
        self.status.set('Для этой серии нет сохранённого фита. Повторный расчёт по её сохранённым настройкам…')
        reference_id = row.get('alignment_reference')
        reference_row = next((entry for entry in self.batch_rows if entry['spectrum_id'] == reference_id), None)
        reference_source = reference_row.get('source_path') if reference_row else None
        def work():
            raw = load_source(source, recipe.processing)
            spectrum = process(raw, recipe.processing)
            if recipe.alignment.enabled:
                if not reference_source or not Path(reference_source).exists():
                    raise ValueError('The first reference spectrum is required to reconstruct an aligned result.')
                reference = process(load_source(reference_source, recipe.processing), recipe.processing)
                first = Path(source).resolve() == Path(reference_source).resolve()
                if recipe.alignment.mode == 'global':
                    spectrum, _ = align_spectrum(reference, spectrum, recipe.alignment, reference_id, first)
                else:
                    spectrum, views, errors, _ = align_regions(reference, spectrum, recipe, reference_id, first)
                    return analyze(spectrum, recipe, views, errors)
            return analyze(spectrum, recipe)
        def ready(analysis):
            self._show_fit_review(analysis, recipe, 'ПОВТОРНЫЙ РАСЧЁТ из исходного спектра: ' + source)
        self.submit('review', work, ready)

    def _show_fit_review(self, analysis, recipe, origin):
        self.review_windows = [w for w in self.review_windows if w.window.winfo_exists()]
        self.review_windows.append(FitReviewWindow(self, analysis, recipe, origin))
        self.status.set('Открыто окно проверки подгонки. Настройки одиночного анализа не изменены.')

    def export_single_image(self):
        if self.analysis is None:
            return self.error('Сначала выполните анализ с текущими параметрами.')
        self.open_image_export(self.analysis, self.session, self.selected_region)

    def open_image_export(self, analysis, recipe, region):
        self.export_windows = [w for w in self.export_windows if w.window.winfo_exists()]
        window = ImageExportWindow(self, analysis, recipe, region)
        self.export_windows.append(window)
        return window

    def open_preprocess(self):
        if self.preprocess_window and self.preprocess_window.window.winfo_exists():
            self.preprocess_window.window.lift()
        else:
            self.preprocess_window = PreprocessWindow(self)

    def close(self):
        if self.closed:
            return True
        if self.batch_running:
            if not self.confirm('Закрыть Spintegrate',
                    'Серия выполняется. Остановить после текущего спектра и закрыть окно?'):
                return False
            self.batch_cancel.set()
        for child in self.source_windows:
            if not child.close():
                return False
        self.closed = True
        if self.poll_timer:
            self.root.after_cancel(self.poll_timer)
        if self.preview_timer:
            self.root.after_cancel(self.preview_timer)
        for timer in self._region_event_timers:
            self.root.after_cancel(timer)
        self._region_event_timers.clear()
        self._region_events_suspended = 0
        if self.preprocess_window and self.preprocess_window.window.winfo_exists():
            self.preprocess_window.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()
        return True


class FitReviewWindow:
    """Read-only review of actual saved results; never changes live fit settings."""
    def __init__(self, app, analysis, recipe, origin):
        self.app, self.analysis = app, analysis
        self.recipe = copy.deepcopy(recipe)
        self.origin = origin
        self.window = tk.Toplevel(app.root)
        self.window.title(('Проверка интегрирования — ' if analysis.method == 'direct' else
                           'Проверка фита Лоренцами — ') + Path(analysis.spectrum.source).name)
        fit_window(self.window, 1500, 1050)
        ttk.Label(self.window, text=origin, wraplength=1100, padding=(10, 6),
                  foreground='#65758a').pack(fill='x')
        ttk.Label(self.window, text='Исходный спектр: ' + analysis.spectrum.source,
                  wraplength=1100, padding=(10, 2)).pack(fill='x')
        controls = ttk.Frame(self.window, padding=(10, 6))
        controls.pack(fill='x')
        ttk.Label(controls, text='Область обработки:').pack(side='left')
        names = [r.name for r in recipe.regions]
        self.selected_region = tk.StringVar(value=next(iter(analysis.results), names[0]))
        selector = ttk.Combobox(controls, textvariable=self.selected_region,
                                values=names, state='readonly', width=24)
        selector.pack(side='left', padx=8)
        selector.bind('<<ComboboxSelected>>', lambda _: self.draw())
        self.color_area = tk.StringVar()
        self.color_combo = ttk.Combobox(controls, textvariable=self.color_area, state='readonly', width=18)
        self.color_combo.pack(side='left', padx=4)
        ttk.Button(controls, text='Цвет области…', command=self.choose_color).pack(side='left', padx=4)
        ttk.Button(controls, text='Экспорт картинки…', command=self.save_image).pack(side='right', padx=4)
        self.figure = Figure(figsize=(10, 6), dpi=100, facecolor='white', layout='constrained')
        grid = self.figure.add_gridspec(3, 1, height_ratios=(1, 2.5, .7))
        self.overview_ax = self.figure.add_subplot(grid[0])
        self.detail_ax = self.figure.add_subplot(grid[1])
        self.residual_ax = self.figure.add_subplot(grid[2], sharex=self.detail_ax)
        plot_frame = ttk.Frame(self.window)
        plot_frame.pack(fill='both', expand=True)
        self.canvas = SpectrumCanvas(self.figure, master=plot_frame)
        self.toolbar = SpectrumToolbar(self.canvas, plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.toolbar.update()
        scale_explicit_fonts(self.window, app.session.appearance.ui_scale_percent)
        self.navigation = PlotNavigation(self.canvas, (self.overview_ax, self.detail_ax, self.residual_ax),
            lambda: (self.analysis.spectrum.ppm[0], self.analysis.spectrum.ppm[-1]))
        block, self.details = scrolled_text(self.window, 7)
        block.pack(fill='x', padx=8, pady=6)
        self.draw()
        translate_widgets(self.window, app.language)

    def draw(self):
        name = self.selected_region.get()
        self.navigation.before_draw((id(self.analysis), name))
        areas = [area.name for area in self.recipe.integrals if area.parent == name]
        self.color_combo.configure(values=areas)
        if self.color_area.get() not in areas:
            self.color_area.set(areas[0] if areas else '')
        draw_spectrum(self.overview_ax, self.detail_ax, self.residual_ax,
                      self.analysis.spectrum, self.recipe, name, self.analysis, component_limit=None,
                      language=self.app.language)
        style_live_figure(self.figure, self.app.session.appearance.ui_scale_percent, shared_labels=True)
        self.navigation.after_draw()
        lines = [self.origin, 'Статус спектра: ' + self.analysis.status]
        metadata = self.analysis.spectrum.processing_metadata
        if metadata.get('alignment_mode') == 'global':
            lines.append(f"Alignment to {metadata.get('alignment_reference')}: {metadata.get('alignment_shift_ppm', 0):+.6g} ppm; correlation={metadata.get('alignment_correlation', 0):.3f}")
        elif metadata.get('alignment_mode') == 'regions':
            for region_name, info in metadata.get('alignment_regions', {}).items():
                lines.append(f"Alignment {region_name}: "+(info.get('error') or f"{info['shift_ppm']:+.6g} ppm; correlation={info['correlation']:.3f}"))
        for area in self.recipe.integrals:
            if area.parent == name:
                value = self.analysis.integrals.get(area.name, float('nan'))
                units = 'интенсивность × ppm' if self.analysis.method == 'direct' else 'Σ A × HWHM'
                lines.append(f'Интеграл {area.name}: {value:.9g} ({units})')
        result = self.analysis.results.get(name)
        if result and self.analysis.method != 'direct':
            d = result.diagnostics
            lines.append(f"RMSE: {d.get('relative_rmse')}; сошлось: {result.success}; nfev: {d.get('nfev')}")
            lines.append(' №       ppm (оптим.)      A            HWHM ppm        A×HWHM' if self.analysis.method == 'lorentzian_manual' else
                         ' №       ppm (фикс.)       A            HWHM ppm        A×HWHM')
            if self.analysis.method == 'lorentzian_manual':
                lines.append('Начальные ppm: ' + ', '.join(f'{v:.6f}' for v in d.get('initial_centers_ppm', [])))
            for i, (c, a, w) in enumerate(zip(result.centers, result.amplitudes, result.widths), 1):
                lines.append(f'{i:3}  {c:14.7f} {a:14.6g} {w:14.6g} {a*w:14.6g}')
        elif result:
            lines.append('Прямое интегрирование: поиск пиков и оптимизация не используются.')
        if name in self.analysis.errors:
            lines.append('ОШИБКА: ' + self.analysis.errors[name])
        lines.extend(self.analysis.notes)
        set_text(self.details, '\n'.join(lines))
        self.canvas.draw_idle()

    def choose_color(self):
        area = next((r for r in self.recipe.integrals if r.name == self.color_area.get()), None)
        if area is None:
            return
        _, color = colorchooser.askcolor(color=area_color(area, self.recipe.integrals.index(area)),
                                         title=self.app.t('Цвет интегрирования'), parent=self.window)
        if color:
            area.color = color
            self.draw()

    def save_image(self):
        return self.app.open_image_export(self.analysis, self.recipe, self.selected_region.get())


class ImageExportWindow:
    def __init__(self, app, analysis, recipe, region):
        self.app, self.analysis = app, analysis
        self.recipe = copy.deepcopy(recipe)
        self.region = region or next(iter(analysis.results), recipe.regions[0].name)
        self.window = tk.Toplevel(app.root)
        self.window.title('Экспорт рисунка')
        a = app.session.appearance
        fit_window(self.window, 540*a.ui_scale_percent/100., 680*a.ui_scale_percent/100., min_width=480, min_height=540)
        buttons = ttk.Frame(self.window, padding=8)
        buttons.pack(side='bottom', fill='x')
        controls = ScrollPanel(self.window)
        controls.pack(fill='both', expand=True)
        self.fields = FieldPanel(controls.body, self.update_info)
        self.fields.pack(fill='x')
        add = self.fields.add
        self.language_choices = {'Как в программе': 'auto', 'Русский': 'ru', 'English': 'en'}
        self.layout_choices = ({'Весь спектр + область + интеграл': 'full', 'Область + интеграл': 'fit', 'Только область': 'region'}
                               if analysis.method == 'direct' else
                               {'Весь спектр + область + остаток': 'full', 'Область + остаток': 'fit', 'Только область': 'region'})
        add('export_format', 'Формат', a.export_format, ('png', 'tiff', 'svg'), str)
        add('export_dpi', 'Качество, DPI', a.export_dpi, kind=int,
            hint='150 / 300 / 600 / 1200 DPI')
        add('export_width_mm', 'Ширина, мм', a.export_width_mm)
        add('export_height_mm', 'Высота, мм', a.export_height_mm)
        add('language_label', 'Язык рисунка', next(k for k, v in self.language_choices.items() if v == a.export_language),
            tuple(self.language_choices), str)
        add('layout_label', 'Компоновка', next(k for k, v in self.layout_choices.items() if v == a.export_layout),
            tuple(self.layout_choices), str)
        add('export_region_names', 'Названия областей на рисунке', a.export_region_names, kind=bool)
        add('title', 'Заголовок (необязательно)', '', kind=str)
        self.info = TranslatedVar(self.window)
        ttk.Label(controls.body, textvariable=self.info, padding=8, wraplength=450).pack(fill='x')
        ttk.Label(controls.body, text='SVG — векторный формат; DPI влияет на PNG и TIFF.',
                  padding=8, wraplength=450).pack(fill='x')
        self.save_button = ttk.Button(buttons, text='Сохранить рисунок…', command=self.save, style='Accent.TButton')
        self.save_button.pack(side='left')
        ttk.Button(buttons, text='Отмена', command=self.window.destroy).pack(side='right')
        scale_explicit_fonts(self.window, a.ui_scale_percent)
        translate_widgets(self.window, app.language)
        self.update_info()

    def options(self):
        values = self.fields.values()
        title = values.pop('title')
        values['export_language'] = self.language_choices[canonical(values.pop('language_label'))]
        values['export_layout'] = self.layout_choices[canonical(values.pop('layout_label'))]
        options = AppearanceSettings(**{**asdict(self.app.session.appearance), **values})
        validate_appearance(options)
        return options, title

    def update_info(self):
        if not hasattr(self, 'info'):
            return
        try:
            a, _ = self.options()
            if a.export_format == 'svg':
                text = f'Размер: {a.export_width_mm:g} × {a.export_height_mm:g} мм (вектор)'
            else:
                w, h = round(a.export_width_mm/25.4*a.export_dpi), round(a.export_height_mm/25.4*a.export_dpi)
                text = f'Размер: {w} × {h} пикселей'
            self.info.set(text)
        except (ValueError, KeyError) as exc:
            self.info.set(str(exc))

    def save(self):
        try:
            options, title = self.options()
        except (ValueError, KeyError) as exc:
            self.info.set(str(exc))
            return
        suffix = '.'+options.export_format
        path = self.app.dialog(filedialog.asksaveasfilename, parent=self.window,
                  defaultextension=suffix, initialfile='lorentzian_fit'+suffix,
                  filetypes=[(options.export_format.upper(), '*'+suffix)])
        if not path:
            return
        try:
            figure = render_fit_figure(self.analysis, self.recipe, self.region, options, self.app.language, title)
            save_figure(path, figure, options)
            self.app.session.appearance = options
            self.app.status.set('Рисунок сохранён: ' + path)
            self.window.destroy()
        except Exception as exc:
            self.app.error(str(exc))


class PreprocessWindow:
    def __init__(self, app: SpintegrateApp):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title('Аподизация и локальная базовая линия')
        fit_window(self.window, 1600, 1000)
        self.timer = None
        self.revision = 0
        self.loading = True
        self.pending = None
        self.preview_key = None
        self.requested_key = None
        self.cached_result = None
        self.cached_region = None
        self.cached_baseline = None
        self.cached_lb = 0.
        self.status = TranslatedVar(self.window, value='Выберите сохранённую область в главном окне.')
        self.status_label = ttk.Label(self.window, textvariable=self.status, padding=8, wraplength=1100)
        self.status_label.pack(side='bottom', fill='x')
        self.status_label.bind('<Configure>', lambda event: self.status_label.configure(wraplength=max(150, event.width-20)))
        split = ttk.Panedwindow(self.window, orient='horizontal')
        split.pack(fill='both', expand=True)
        controls, right = ScrollPanel(split), ttk.Frame(split)
        controls.canvas.configure(width=round(385*display_factor(app.root)))
        left = controls.body
        split.add(controls, weight=0)
        split.add(right, weight=1)
        ttk.Label(left, text='Аподизация (экспоненциальная)', font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        self.lb_var = tk.StringVar(value=str(app.session.processing.lb_hz))
        self.lb_slider = tk.DoubleVar(value=app.session.processing.lb_hz)
        ttk.Label(left, text='LB = добавочное уширение FWHM, Гц').pack(anchor='w', pady=(8, 3))
        entry = ttk.Entry(left, textvariable=self.lb_var)
        entry.pack(fill='x')
        self.slider = ttk.Scale(left, from_=0, to=max(20., app.session.processing.lb_hz*1.2),
                                variable=self.lb_slider, command=self.slider_changed)
        self.slider.pack(fill='x', pady=8)
        ttk.Label(left, text='Сравнение с LB=0 при тех же прочих настройках.\n'
                  'График обновляется после короткой паузы.', foreground='#65758a', wraplength=340).pack(anchor='w')
        ttk.Separator(left).pack(fill='x', pady=12)
        ttk.Label(left, text='Базовая линия выбранной области', font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        self.region_label = ttk.Label(left, text='', wraplength=345)
        self.region_label.pack(anchor='w', pady=8)
        self.editing_region = app.selected_region
        self.baseline_history = {}
        self.restoring_baseline = False
        selected = next((r for r in app.session.regions if r.name == self.editing_region), None)
        b = selected.baseline if selected else app.session.baseline_default
        self.fields = BaselineControls(left, b, self.changed)
        self.fields.pack(fill='x', padx=8)
        self.undo_baseline_button = ttk.Button(left, text='Отменить изменение базы', command=self.undo_baseline, state='disabled')
        self.undo_baseline_button.pack(fill='x', pady=3)
        ttk.Button(left, text='Применить ко всем областям', command=lambda: self.apply_baseline(True)).pack(fill='x', pady=3)
        ttk.Label(left, text='Изменения сразу сохраняются для выбранной области и новых областей. '
                  'В JSON сохраняются только метод и параметры; базовая линия рассчитывается заново для каждого спектра. '
                  'Для остальных областей нажмите «Применить ко всем областям».',
                  wraplength=350, foreground='#65758a').pack(anchor='w', pady=8)
        self.figure = Figure(figsize=(8, 7), dpi=100, facecolor='white', layout='constrained')
        self.ax_apod, self.ax_base = self.figure.subplots(2, 1)
        WheelControls(right, app, lambda: self.navigation.reset()).pack(fill='x')
        preview_controls = ttk.Frame(right, padding=4)
        preview_controls.pack(fill='x')
        self.curve_visibility = {}
        for key, label in (('raw', 'Исходная область'), ('baseline', 'Базовая линия'), ('corrected', 'После коррекции')):
            self.curve_visibility[key] = tk.BooleanVar(value=True)
            ttk.Checkbutton(preview_controls, text=label, variable=self.curve_visibility[key],
                            command=self.redraw_preview).pack(side='left', padx=4)
        self.pick_anchors = tk.BooleanVar(value=False)
        ttk.Checkbutton(right, text='Выбирать опорные точки: левая кнопка добавляет, правая удаляет',
                        variable=self.pick_anchors, command=self.enable_anchors).pack(anchor='w', padx=5)
        self.canvas = SpectrumCanvas(self.figure, master=right)
        self.toolbar = SpectrumToolbar(self.canvas, right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_anchor_click)
        self.canvas.mpl_connect('button_press_event', app.pick_phase_pivot)
        self.canvas.mpl_connect('key_press_event', app.region_edit_key)
        scale_explicit_fonts(self.window, app.session.appearance.ui_scale_percent)
        self.navigation = PlotNavigation(self.canvas, (self.ax_apod, self.ax_base),
            lambda: (self.cached_result[0].ppm[0], self.cached_result[0].ppm[-1]) if self.cached_result else None,
            app.wheel_mode_code, app.adjust_phase_wheel, lambda: app.phase_pivot_armed)
        self.lb_var.trace_add('write', lambda *_: self.changed())
        self.loading = False
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        translate_widgets(self.window, app.language)
        self.refresh()

    def slider_changed(self, value):
        self.lb_var.set(f'{float(value):.3f}')

    def changed(self):
        if self.loading:
            return
        try:
            self.store_controls()
        except (ValueError, tk.TclError) as exc:
            self.status.set(str(exc))
        self.revision += 1
        if self.timer:
            self.window.after_cancel(self.timer)
        self.timer = self.window.after(220, self.refresh)

    def sync_region_controls(self, force=False):
        if not force and self.editing_region == self.app.selected_region:
            return
        r = next((r for r in self.app.session.regions if r.name == self.app.selected_region), None)
        self.loading = True
        self.fields.set_values(asdict(r.baseline if r else self.app.session.baseline_default))
        self.editing_region = self.app.selected_region
        self.loading = False
        self.update_undo_button()

    def load_region_baseline(self):
        self.sync_region_controls(force=True)
        self.changed()

    def load_settings(self):
        self.baseline_history.clear()
        self.update_undo_button()
        self.sync_region_controls(force=True)
        self.loading = True
        self.lb_var.set(str(self.app.session.processing.lb_hz))
        self.lb_slider.set(self.app.session.processing.lb_hz)
        self.loading = False

    def store_controls(self):
        """Commit parameters, never a fitted baseline curve, before analysis/export."""
        if self.loading:
            return
        self.sync_region_controls()
        b = BaselineSettings(**self.fields.values())
        validate_baseline(b)
        lb = float(self.lb_var.get())
        if not np.isfinite(lb) or lb < 0:
            raise ValueError('Аподизация LB должна быть >= 0 Гц.')
        r = next((r for r in self.app.session.regions if r.name == self.editing_region), None)
        baseline_changed = (r.baseline if r is not None else self.app.session.baseline_default) != b
        changed = self.app.session.processing.lb_hz != lb or (r is not None and baseline_changed)
        self.app.session.processing.lb_hz = lb
        if baseline_changed:
            self.app.session.baseline_default = copy.deepcopy(b)
        if r is not None and baseline_changed:
            if not self.restoring_baseline:
                history = self.baseline_history.setdefault(r.name, [])
                history.append(copy.deepcopy(r.baseline))
                del history[:-30]
            r.baseline = copy.deepcopy(b)
        self.update_undo_button()
        if changed:
            self.app.refresh_regions()
            self.app.invalidate()
            self.app.schedule_preview()

    def refresh(self):
        self.timer = None
        if not self.window.winfo_exists():
            return
        try:
            self.sync_region_controls()
            b = BaselineSettings(**self.fields.values())
            recipe = self.app.read_settings()
            lb = float(self.lb_var.get())
            recipe.processing.lb_hz = lb
            recipe.baseline_default = b
            recipe.validate()
        except (ValueError, tk.TclError) as exc:
            self.status.set(str(exc))
            return
        self.lb_slider.set(lb)
        r = next((r for r in recipe.regions if r.name == self.app.selected_region), None)
        set_label(self.region_label, f'{r.name}: {r.low:g}…{r.high:g} ppm' if r else 'Сначала сохраните область обработки.')
        if self.app.raw is None:
            return
        key = (id(self.app.raw), repr(recipe.processing), repr(b), repr(r), self.app.generation)
        if key == self.preview_key:
            return
        if key != self.requested_key:
            self.revision += 1
            self.requested_key = key
        revision, generation, raw = self.revision, self.app.generation, self.app.raw
        phase_state = self.app.phase_state
        if 'preprocess' in self.app.jobs:
            # Do not queue unbounded jobs while dragging a slider.
            if self.timer:
                self.window.after_cancel(self.timer)
            self.timer = self.window.after(120, self.refresh)
            return
        def work():
            current = process(raw, recipe.processing, phase_cache=phase_state)
            no_apod = copy.deepcopy(recipe.processing)
            no_apod.lb_hz = 0.
            reference = process(raw, no_apod, phase_cache=phase_state)
            baseline_data = None
            if r:
                if recipe.analysis_method == 'direct':
                    result = direct_region(current, r)
                    x, y, corrected, line, notes = result.x, result.raw, result.corrected, result.baseline, result.notes
                else:
                    mask = (current.ppm >= r.low) & (current.ppm <= r.high)
                    x, y = current.ppm[mask], current.intensity[mask]
                    corrected, line, notes = baseline_correct(x, y, b)
                baseline_data = (x, y, corrected, line, notes)
            return current, reference, baseline_data
        def ready(result):
            if not self.window.winfo_exists() or revision != self.revision or generation != self.app.generation:
                return
            current, reference, local = result
            self.preview_key = key
            self.cached_result = result
            self.cached_region, self.cached_baseline, self.cached_lb = r, b, lb
            self.draw_cached()
        self.app.submit('preprocess', work, ready)

    def draw_cached(self):
        current, reference, local = self.cached_result
        self.navigation.before_draw(self.app.navigation_context())
        r, b, lb = self.cached_region, self.cached_baseline, self.cached_lb
        t = self.app.t
        self.ax_apod.clear()
        self.ax_base.clear()
        ref_mask = (reference.ppm >= r.low) & (reference.ppm <= r.high) if r else slice(None)
        cur_mask = (current.ppm >= r.low) & (current.ppm <= r.high) if r else slice(None)
        self.ax_apod.plot(reference.ppm[ref_mask], reference.intensity[ref_mask],
                          color='#a8b4c1', lw=.9, label=t('LB = 0 Гц'))
        self.ax_apod.plot(current.ppm[cur_mask], current.intensity[cur_mask],
                          color='#254b79', lw=1., label=t(f'LB = {lb:g} Гц'))
        self.ax_apod.set_title(t('Аподизация: текущая область' if r else 'Аподизация: весь спектр'), fontsize=11, loc='left')
        self.ax_apod.set_xlim(r.high, r.low) if r else self.ax_apod.set_xlim(current.ppm[-1], current.ppm[0])
        if local:
            x, y, corrected, line, notes = local
            self.ax_base.plot(x, y, color='#a8b4c1', lw=.9, label=t('Исходная область'))
            self.ax_base.plot(x, line, color='#d26335', lw=1., label=t(f'База: {b.method}'))
            self.ax_base.plot(x, corrected, color='#254b79', lw=1., label=t('После коррекции'))
            for key, line_artist in zip(('raw', 'baseline', 'corrected'), self.ax_base.lines):
                line_artist.set_visible(self.curve_visibility[key].get())
            if b.method == 'interp_pts':
                anchors = b.parameters.get('anchor_ppm', [])
                if anchors:
                    self.ax_base.scatter(anchors, np.interp(anchors, x, y), color='#d26335', marker='o', zorder=5)
            self.ax_base.set_xlim(r.high, r.low)
            self.status.set(' | '.join(notes) if notes else 'Предпросмотр готов; база применяется только внутри выбранной области.')
        else:
            self.ax_base.text(.5, .5, t('Выберите сохранённую область обработки'),
                              transform=self.ax_base.transAxes, ha='center')
            self.status.set('Аподизация обновлена. Сохраните область для предпросмотра базовой линии.')
        self.ax_base.set_title(t('Локальная базовая линия'), fontsize=11, loc='left')
        for ax in (self.ax_apod, self.ax_base):
            ax.set_xlabel(t('Химический сдвиг (ppm)'))
            ax.grid(alpha=.15)
            if ax.lines:
                ax.legend(fontsize=8)
        self.app.draw_phase_pivot((self.ax_apod, self.ax_base), current)
        style_live_figure(self.figure, self.app.session.appearance.ui_scale_percent)
        self.navigation.after_draw()
        self.canvas.draw_idle()

    def redraw_preview(self):
        if self.cached_result is not None:
            self.draw_cached()

    def enable_anchors(self):
        if self.pick_anchors.get():
            self.fields.vars['method'].set('interp_pts')

    def on_anchor_click(self, event):
        if self.app.phase_pivot_armed or self.toolbar.mode or not self.pick_anchors.get() or event.inaxes != self.ax_base or event.xdata is None:
            return
        if self.fields.method != 'interp_pts' or not self.cached_region or event.button not in (1, 3):
            return
        region = self.cached_region
        if not region.low <= event.xdata <= region.high:
            return
        try:
            b = BaselineSettings(**self.fields.values())
            anchors = list(b.parameters.get('anchor_ppm', []))
            if event.button == 1:
                anchors.append(float(event.xdata))
            elif anchors:
                anchors.remove(min(anchors, key=lambda x: abs(x-event.xdata)))
            self.fields.vars['anchor_ppm'].set(json.dumps(sorted(set(anchors))))
        except ValueError as exc:
            self.status.set(str(exc))

    def undo_baseline(self):
        history = self.baseline_history.get(self.app.selected_region, [])
        if not history:
            return
        self.restoring_baseline = True
        try:
            self.fields.set_values(asdict(history.pop()))
            self.changed()
        finally:
            self.restoring_baseline = False
            self.update_undo_button()

    def update_undo_button(self):
        if hasattr(self, 'undo_baseline_button'):
            self.undo_baseline_button.configure(state='normal' if self.baseline_history.get(self.app.selected_region) else 'disabled')

    def apply_baseline(self, all_regions=False):
        try:
            b = BaselineSettings(**self.fields.values())
            s = self.app.read_settings()
            s.baseline_default = b
            targets = s.regions if all_regions else [r for r in s.regions if r.name == self.app.selected_region]
            if not targets:
                raise ValueError('Выберите сохранённую область обработки.')
            for r in targets:
                if r.baseline != b:
                    history = self.baseline_history.setdefault(r.name, [])
                    history.append(copy.deepcopy(r.baseline))
                    del history[:-30]
                r.baseline = copy.deepcopy(b)
            s.validate()
            self.app.session = s
            self.app.refresh_regions()
            self.app.invalidate()
            self.app.preview()
            self.status.set('Настройки базовой линии сохранены для: ' + ', '.join(r.name for r in targets))
        except ValueError as exc:
            self.status.set(str(exc))

    def close(self):
        self.revision += 1
        if self.timer:
            self.window.after_cancel(self.timer)
        self.window.destroy()
        self.app.preprocess_window = None
        self.app.wheel_controls = [controls for controls in self.app.wheel_controls if controls.winfo_exists()]
        if self.app.phase_pivot_armed:
            self.app.phase_pivot_armed = False
            self.app.draw(local=self.app.preview_local)


def main():
    enable_windows_dpi()
    root = tk.Tk()
    SpintegrateApp(root)
    root.mainloop()
