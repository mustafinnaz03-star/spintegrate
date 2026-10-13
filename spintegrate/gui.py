"""Tk desktop UI. All numerical work is submitted to workers; Tk stays on main thread."""
from __future__ import annotations

import copy
import csv
import queue
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from .core import (BASELINE_METHODS, PHASE_MODES, BaselineSettings, FitRegion,
                   IntegrationRegion, Session, analyze, baseline_correct,
                   batch_analyze, discover_spectra, load_raw, pick_peaks, process)
from .demo import demo_raw, demo_session, write_demo_series


PHASE_LABELS = {'Без коррекции': 'none', 'ACME (nmrglue)': 'acme',
                'Peak minima (nmrglue)': 'peak_minima', 'abs': 'abs'}
REF_LABELS = {'Из метаданных Bruker': 'metadata', 'Самый сильный пик': 'strongest',
              'Пик в заданном ppm-окне': 'window'}


class FieldPanel(ttk.Frame):
    """Compact labeled input rows with typed values and a single edit callback."""
    def __init__(self, parent, callback=None):
        super().__init__(parent, padding=8)
        self.vars, self.types = {}, {}
        self.callback = callback
        self.columnconfigure(1, weight=1)

    def add(self, key, label, value, choices=None, kind=float, hint=None):
        row = len(self.vars)*2
        ttk.Label(self, text=label).grid(row=row, column=0, sticky='w', padx=(0, 8), pady=3)
        var = tk.BooleanVar(value=value) if kind is bool else tk.StringVar(value=str(value))
        if kind is bool:
            widget = ttk.Checkbutton(self, variable=var)
        elif choices:
            widget = ttk.Combobox(self, textvariable=var, values=choices, state='readonly', width=20)
        else:
            widget = ttk.Entry(self, textvariable=var, width=16)
        widget.grid(row=row, column=1, sticky='ew', pady=3)
        if hint:
            ttk.Label(self, text=hint, foreground='#65758a', wraplength=355,
                      font=('Segoe UI', 8)).grid(row=row+1, column=0, columnspan=2, sticky='w')
        self.vars[key], self.types[key] = var, kind
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
                var.set(values[k])


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
    widget.configure(state='normal')
    widget.delete('1.0', 'end')
    widget.insert('1.0', value)
    widget.configure(state='disabled')


class ScrollPanel(ttk.Frame):
    """Scrollable settings page for smaller screens / enlarged Windows fonts."""
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background='#dcdad5')
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
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('Spintegrate — анализ спектров ЯМР')
        root.geometry('1400x920')
        root.minsize(1080, 750)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10))
        style.configure('TNotebook.Tab', padding=(12, 7))
        style.configure('Treeview', rowheight=25, font=('Segoe UI', 9))
        self.session = Session()
        self.raw = self.spectrum = self.analysis = None
        self.loaded_filter = True
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
        self.preprocess_window = None
        self.status = tk.StringVar(value='Откройте папку эксперимента Bruker или демонстрационный спектр.')
        self._build()
        self._sync_widgets()
        root.after(80, self._poll)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def _build(self):
        top = ttk.Frame(self.root, padding=(12, 8))
        top.pack(fill='x')
        ttk.Label(top, text='Spintegrate', font=('Segoe UI', 18, 'bold')).pack(side='left')
        ttk.Button(top, text='Загрузить настройки JSON', command=self.load_settings).pack(side='right', padx=4)
        ttk.Button(top, text='Сохранить настройки JSON', command=self.save_settings).pack(side='right', padx=4)
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=10)
        self.single_tab, self.batch_tab = ttk.Frame(self.tabs), ttk.Frame(self.tabs)
        self.tabs.add(self.single_tab, text='Один спектр')
        self.tabs.add(self.batch_tab, text='Много спектров')
        source = ttk.Frame(self.single_tab, padding=8)
        source.pack(fill='x')
        ttk.Button(source, text='Открыть Bruker…', command=self.open_bruker).pack(side='left', padx=3)
        ttk.Button(source, text='Открыть NPZ…', command=self.open_npz).pack(side='left', padx=3)
        ttk.Button(source, text='Демо', command=self.load_demo).pack(side='left', padx=3)
        ttk.Button(source, text='Аподизация / базовая линия…', command=self.open_preprocess).pack(side='left', padx=8)
        self.fit_button = ttk.Button(source, text='Выполнить анализ', command=self.run_analysis)
        self.fit_button.pack(side='left', padx=4)
        ttk.Button(source, text='Запомнить → серия', command=self.remember_for_batch).pack(side='right', padx=4)
        self.source_label = ttk.Label(self.single_tab, text='Спектр не загружен', foreground='#65758a')
        self.source_label.pack(fill='x', padx=12, pady=(0, 6))
        split = ttk.Panedwindow(self.single_tab, orient='horizontal')
        split.pack(fill='both', expand=True)
        left, right = ttk.Frame(split, width=410), ttk.Frame(split)
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
        add('abs_source', 'abs применяется к', p.abs_source, ['spectrum', 'fid'], str,
            'spectrum = |FFT(FID)|; fid = Re FFT(|FID|).')
        add('p0', 'Начальная фаза P0, °', p.p0)
        add('p1', 'Начальная фаза P1, °', p.p1)
        add('phase_peak_width', 'Окно peak_minima, точки', p.phase_peak_width, kind=int)
        add('reference_label', 'Привязка ppm', 'Из метаданных Bruker', list(REF_LABELS), str)
        add('reference_ppm', 'Референс, ppm', p.reference_ppm)
        add('reference_low', 'Окно референса: от', p.reference_low)
        add('reference_high', 'Окно референса: до', p.reference_high)
        add('ppm_offset', 'Сдвиг шкалы, ppm', p.ppm_offset)
        add('zero_fill_factor', 'Zero filling ×', p.zero_fill_factor, [1, 2, 4, 8], int)
        add('remove_digital_filter', 'Удалить цифровой фильтр', p.remove_digital_filter, kind=bool)
        ttk.Label(processing_tab, text='P0/P1 — старт автоматического фазирования. '
                  '«Без коррекции» показывает реальную часть FFT.\nLB настраивается в отдельном окне.',
                  wraplength=365, foreground='#65758a').pack(padx=8, pady=10, anchor='w')
        self._build_regions(region_tab)
        self.fit_fields = FieldPanel(fit_tab, self.fit_settings_changed)
        self.fit_fields.pack(fill='x')
        f = self.session.fit
        add = self.fit_fields.add
        add('threshold', 'Threshold (высота пика)', f.threshold,
            hint='Абсолютная высота после локальной коррекции базовой линии.')
        add('min_separation_points', 'Разделение, точки', f.min_separation_points, kind=int)
        add('signed', 'Искать и отрицательные пики', f.signed, kind=bool,
            hint='Выключено: A ≥ 0. Включено: амплитуды любого знака + диагностика компенсации.')
        add('initial_width_hz', 'Initial FWHM, Гц', f.initial_width_hz)
        add('min_width_hz', 'Минимум FWHM, Гц', f.min_width_hz)
        add('max_width_hz', 'Максимум FWHM, Гц', f.max_width_hz)
        add('max_nfev', 'Лимит вычислений / попытку', f.max_nfev, kind=int)
        add('restarts', 'Попытки с разными ширинами', f.restarts, kind=int)
        add('max_peaks', 'Максимум пиков / область', f.max_peaks, kind=int)
        add('loss', 'Функция ошибки', f.loss, ['linear', 'soft_l1'], str)
        add('residual_warning', 'Предупреждать при RMSE >', f.residual_warning)
        add('cancellation_warning', 'Компенсация: порог 0…1', f.cancellation_warning)
        ttk.Label(fit_tab, text='Центры фиксированы по peak-picking.\nШирина модели = HWHM (ppm); '
                  'старт = max(LB, initial FWHM) / (2 × SFO1).\nИнтеграл = Σ A × HWHM; '
                  'полная площадь Лоренца = π × этот интеграл.', wraplength=365,
                  foreground='#65758a').pack(padx=8, pady=10, anchor='w')
        self.integral_tree = ttk.Treeview(result_tab, columns=('integral',), show='tree headings', height=7)
        self.integral_tree.heading('#0', text='Сигнал')
        self.integral_tree.heading('integral', text='Σ A × HWHM')
        self.integral_tree.column('#0', width=145)
        self.integral_tree.column('integral', width=140)
        self.integral_tree.pack(fill='x', padx=6, pady=8)
        ttk.Button(result_tab, text='Сохранить пики и интегралы CSV…', command=self.export_single).pack(pady=4)
        frame, self.result_text = scrolled_text(result_tab, 17)
        frame.pack(fill='both', expand=True, padx=6, pady=8)
        set_text(self.result_text, 'После анализа здесь появятся остаток, ограничения ширин, '
                 'сходимость и обнаруженные пары компенсирующихся линий.')
        self.figure = Figure(figsize=(9, 7), dpi=100, facecolor='white')
        grid = self.figure.add_gridspec(3, 1, height_ratios=(1, 2, .65), hspace=.4)
        self.overview_ax = self.figure.add_subplot(grid[0])
        self.detail_ax = self.figure.add_subplot(grid[1])
        self.residual_ax = self.figure.add_subplot(grid[2], sharex=self.detail_ax)
        self.figure.subplots_adjust(left=.09, right=.98, top=.95, bottom=.08)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.selectors = []
        self.selection_label = ttk.Label(right, text='Во вкладке «Области» выберите режим, '
                                        'затем выделите диапазон мышью на графике.')
        self.selection_label.pack(fill='x', padx=8, pady=4)
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
        buttons = ttk.Frame(parent)
        buttons.pack(fill='x', padx=8)
        ttk.Button(buttons, text='Сохранить область', command=self.add_region).pack(side='left', padx=2)
        ttk.Button(buttons, text='Изменить выбранную', command=self.update_region).pack(side='left', padx=2)
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
        self.recipe_label = ttk.Label(frame, text='Нажмите «Запомнить → серия» во вкладке одного спектра '
                                     'или загрузите JSON.', wraplength=1200)
        self.recipe_label.pack(anchor='w', pady=10)
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
        self.batch_tree = ttk.Treeview(frame, columns=('status', 'integrals'), show='tree headings', height=13)
        self.batch_tree.heading('#0', text='Номер / ID спектра')
        self.batch_tree.heading('status', text='Статус')
        self.batch_tree.heading('integrals', text='Интегралы Σ A × HWHM')
        self.batch_tree.column('#0', width=280)
        self.batch_tree.column('status', width=90)
        self.batch_tree.column('integrals', width=700)
        self.batch_tree.pack(fill='both', expand=True, pady=8)
        self.batch_tree.bind('<<TreeviewSelect>>', self.show_batch_row)
        block, self.batch_text = scrolled_text(frame, 5)
        block.pack(fill='x')
        set_text(self.batch_text, 'Каждый спектр обрабатывается независимо. Ошибки не останавливают '
                 'серию; недостоверные интегралы записываются как NaN. '
                 'Рядом с CSV сохраняются JSON настроек и подробная диагностика.')

    def _sync_widgets(self):
        self.loading_settings = True
        p = asdict(self.session.processing)
        p['phase_label'] = next(k for k, v in PHASE_LABELS.items() if v == p['phase'])
        p['reference_label'] = next(k for k, v in REF_LABELS.items() if v == p['reference_mode'])
        self.processing_fields.set_values(p)
        self.fit_fields.set_values(asdict(self.session.fit))
        self.loading_settings = False
        self.refresh_regions()

    def read_settings(self):
        s = copy.deepcopy(self.session)
        p = self.processing_fields.values()
        p['phase'] = PHASE_LABELS[p.pop('phase_label')]
        p['reference_mode'] = REF_LABELS[p.pop('reference_label')]
        for k, v in p.items():
            setattr(s.processing, k, v)
        for k, v in self.fit_fields.values().items():
            setattr(s.fit, k, v)
        s.validate()
        return s

    def invalidate(self):
        self.generation += 1
        self.analysis = None
        self.fit_button.configure(state='normal')
        for item in self.integral_tree.get_children():
            self.integral_tree.delete(item)
        set_text(self.result_text, 'Параметры или области изменились. Выполните анализ заново.')

    def settings_changed(self):
        if self.loading_settings:
            return
        self.invalidate()
        self.schedule_preview()

    def fit_settings_changed(self):
        if self.loading_settings:
            return
        self.invalidate()
        self.schedule_preview()

    def schedule_preview(self):
        if self.preview_timer:
            self.root.after_cancel(self.preview_timer)
        self.preview_timer = self.root.after(220, self.preview)

    def submit(self, name, function, callback, generation=None):
        future = self.executor.submit(function)
        self.jobs[name] = (future, callback, generation)

    def _poll(self):
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
                values = ', '.join(f'{k[10:]}={v:.6g}' for k, v in row.items() if k.startswith('integral__'))
                self.batch_tree.insert('', 'end', text=f"{number}: {row['spectrum_id']}",
                                       values=(row['status'], values))
                self.status.set(f"Серия: {number}/{total}, {row['spectrum_id']} — {row['status']}")
        if self.pending_preview and 'preview' not in self.jobs:
            task = self.pending_preview
            self.pending_preview = None
            self._submit_preview(*task)
        self.root.after(80, self._poll)

    def error(self, text):
        self.status.set('Ошибка: ' + text)
        messagebox.showerror('Spintegrate', text, parent=self.root)

    def open_bruker(self):
        path = filedialog.askdirectory(title='Папка эксперимента с fid и acqus')
        if path:
            self.load_path(path)

    def open_npz(self):
        path = filedialog.askopenfilename(filetypes=[('FID NPZ', '*.npz')])
        if path:
            self.load_path(path)

    def load_path(self, path):
        try:
            recipe = self.read_settings()
        except ValueError as exc:
            return self.error(str(exc))
        self.invalidate()
        self.source_path = str(path)
        self.raw = self.spectrum = None
        self.pending_preview = None
        self.status.set('Загрузка FID…')
        remove_filter = recipe.processing.remove_digital_filter
        def loaded(raw):
            self.raw = raw
            self.loaded_filter = remove_filter
            self.source_label.configure(text=raw.source)
            self.preview()
        self.submit('load', lambda: load_raw(path, remove_filter), loaded, self.generation)

    def load_demo(self):
        self.session = demo_session()
        self._sync_widgets()
        self.invalidate()
        self.raw = demo_raw()
        self.source_path = None
        self.source_label.configure(text=self.raw.source)
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
        if self.source_path and recipe.processing.remove_digital_filter != self.loaded_filter:
            return self.load_path(self.source_path)
        self.preview_revision += 1
        task = (self.raw, recipe, self.generation, self.preview_revision)
        if 'preview' in self.jobs:
            self.pending_preview = task
        else:
            self._submit_preview(*task)

    def _submit_preview(self, raw, recipe, generation, revision):
        self.status.set('Обновляю спектр и локальную базовую линию…')
        selected = next((r for r in recipe.regions if r.name == self.selected_region), None)
        def work():
            spectrum = process(raw, recipe.processing)
            local = None
            if selected:
                mask = (spectrum.ppm >= selected.low) & (spectrum.ppm <= selected.high)
                x, y = spectrum.ppm[mask], spectrum.intensity[mask]
                corrected, baseline, notes = baseline_correct(x, y, selected.baseline)
                peaks = pick_peaks(x, corrected, recipe.fit)
                local = (x, y, corrected, baseline, peaks, notes)
            return spectrum, local
        def ready(result):
            if revision != self.preview_revision:
                return
            self.session.processing = recipe.processing
            self.session.fit = recipe.fit
            self.spectrum, local = result
            self.draw(local=local)
            self.status.set(f'Спектр готов. LB={recipe.processing.lb_hz:g} Гц; '
                            f'P0/P1={self.spectrum.phases[0]:.2f}/{self.spectrum.phases[1]:.2f}°. '
                            + (' | '.join(local[-1]) if local and local[-1] else 'Выберите области для анализа.'))
            if self.preprocess_window and self.preprocess_window.window.winfo_exists():
                self.preprocess_window.refresh()
        self.submit('preview', work, ready, generation)

    def _draw_empty(self):
        for ax, title in ((self.overview_ax, 'Весь спектр'), (self.detail_ax, 'Выбранная область'),
                          (self.residual_ax, 'Остаток: данные − подгонка')):
            ax.clear()
            ax.set_title(title, fontsize=10, loc='left')
            ax.set_xlabel('ppm')
            ax.grid(alpha=.15)
        self.canvas.draw_idle()

    def draw(self, local=None):
        # Clear selectors before axes; retain references for interactive dragging.
        for selector in self.selectors:
            selector.disconnect_events()
        self.selectors.clear()
        self._draw_empty()
        if self.spectrum is None:
            return
        spectrum = self.spectrum
        self.overview_ax.plot(spectrum.ppm, spectrum.intensity, color='#254b79', lw=.8)
        self.overview_ax.set_xlim(spectrum.ppm[-1], spectrum.ppm[0])
        for r in self.session.regions:
            self.overview_ax.axvspan(r.low, r.high, color='#4d8bbb', alpha=.17)
            self.overview_ax.text((r.low+r.high)/2, .95, r.name, transform=self.overview_ax.get_xaxis_transform(),
                                   ha='center', va='top', fontsize=8)
        for r in self.session.integrals:
            self.overview_ax.axvspan(r.low, r.high, color='#2c9874', alpha=.23)
        result = self.analysis.results.get(self.selected_region) if self.analysis else None
        region = next((r for r in self.session.regions if r.name == self.selected_region), None)
        if result:
            x, y = result.x, result.corrected
            self.detail_ax.plot(x, y, color='#254b79', lw=1., label='После базовой линии')
            self.detail_ax.plot(x, result.fitted, color='#d26335', lw=1.4, label='Сумма Лоренцев')
            # All components are exported; limit drawn curves to keep the UI usable.
            components = result.components()
            for i in range(min(len(result.centers), 60)):
                self.detail_ax.plot(x, components[:, i], '--', lw=.75, alpha=.6)
            self.detail_ax.scatter(result.centers, result.initial_amplitudes, color='#c33b43',
                                    s=20, zorder=4, label='Peak-picking (фикс. центры)')
            self.residual_ax.plot(x, y-result.fitted, color='#687689', lw=.85)
            self.detail_ax.set_title(f'{result.region}: {len(result.centers)} пиков; '
                                    + ('сошлось' if result.success else 'НЕ СОШЛОСЬ'), fontsize=10, loc='left')
        elif local:
            x, raw, y, line, peaks, _ = local
            self.detail_ax.plot(x, raw, color='#b0b9c5', lw=.9, label='До базовой линии')
            self.detail_ax.plot(x, line, color='#d26335', lw=1., label='Базовая линия')
            self.detail_ax.plot(x, y, color='#254b79', lw=1., label='После коррекции')
            self.detail_ax.scatter(*peaks, color='#c33b43', s=22, zorder=4, label='Peak-picking')
            self.detail_ax.axhline(self.session.fit.threshold, color='#c33b43', ls=':', lw=.8, label='Threshold')
            if self.session.fit.signed:
                self.detail_ax.axhline(-self.session.fit.threshold, color='#c33b43', ls=':', lw=.8)
        else:
            self.detail_ax.plot(spectrum.ppm, spectrum.intensity, color='#254b79', lw=.8, label='Спектр')
        if region:
            self.detail_ax.set_xlim(region.high, region.low)
            for area in self.session.integrals:
                if area.parent == region.name:
                    self.detail_ax.axvspan(area.low, area.high, color='#2c9874', alpha=.12)
        else:
            self.detail_ax.set_xlim(spectrum.ppm[-1], spectrum.ppm[0])
        self.detail_ax.legend(fontsize=8, loc='upper right')
        self.residual_ax.axhline(0, color='#a0a8b5', lw=.7)
        for ax in (self.overview_ax, self.detail_ax):
            self.selectors.append(SpanSelector(ax, self.on_span, 'horizontal', useblit=True,
                                  props=dict(alpha=.25, facecolor='#e6a24a'), button=1,
                                  interactive=False, minspan=0))
        self.canvas.draw_idle()

    def on_span(self, low, high):
        if self.toolbar.mode or abs(high-low) < 1e-9:
            return
        self.bound_low.set(f'{min(low, high):.6f}')
        self.bound_high.set(f'{max(low, high):.6f}')
        self.selection_label.configure(text=f'Выбрано {min(low, high):.6f}…{max(low, high):.6f} ppm. '
                                      'Нажмите «Сохранить область».')
        self.control_tabs.select(1)

    def refresh_regions(self):
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
        if self.selection_mode.get() == 'fit':
            return FitRegion(name, low, high, copy.deepcopy(self.session.baseline_default))
        return IntegrationRegion(name, self.parent_region.get(), low, high)

    def add_region(self):
        try:
            self.session = self.read_settings()
            s = copy.deepcopy(self.session)
            region = self._region_from_fields()
            target = s.regions if isinstance(region, FitRegion) else s.integrals
            target.append(region)
            s.validate()
            self.session = s
            if isinstance(region, FitRegion):
                self.selected_region = region.name
            self.region_name.set(('region_' if isinstance(region, FitRegion) else 'signal_') + str(len(target)+1))
            self.refresh_regions()
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
                s.regions = [new if r.name == old else r for r in s.regions]
                for area in s.integrals:
                    if area.parent == old:
                        area.parent = new.name
                self.selected_region = new.name
            else:
                s.integrals = [new if r.name == old else r for r in s.integrals]
            s.validate()
            self.session = s
            self.refresh_regions()
            self.invalidate()
            self.preview()
        except (ValueError, StopIteration) as exc:
            self.error(str(exc))

    def select_region(self, _=None):
        if self.loading_settings:
            return
        selection = self.region_tree.selection()
        if not selection:
            return
        self.selected_region = selection[0]
        r = next(r for r in self.session.regions if r.name == self.selected_region)
        self.parent_region.set(r.name)
        self.selection_mode.set('fit')
        self.region_name.set(r.name)
        self.bound_low.set(str(r.low))
        self.bound_high.set(str(r.high))
        if self.analysis:
            self.draw()
        else:
            self.preview()
        if self.preprocess_window and self.preprocess_window.window.winfo_exists():
            self.preprocess_window.refresh()

    def select_area(self, _=None):
        if self.loading_settings:
            return
        selection = self.area_tree.selection()
        if not selection:
            return
        r = next(r for r in self.session.integrals if r.name == selection[0])
        self.selection_mode.set('integral')
        self.region_name.set(r.name)
        self.parent_region.set(r.parent)
        self.bound_low.set(str(r.low))
        self.bound_high.set(str(r.high))
        self.selected_region = r.parent
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
        if dependent and not messagebox.askyesno('Удалить область',
                    f'Также удалить {len(dependent)} связанных областей интегрирования?', parent=self.root):
            return
        self.session.regions = [r for r in self.session.regions if r.name != name]
        self.session.integrals = [r for r in self.session.integrals if r.parent != name]
        self.refresh_regions()
        self.invalidate()
        self.preview()

    def delete_area(self):
        selection = self.area_tree.selection()
        if selection:
            self.session.integrals = [r for r in self.session.integrals if r.name != selection[0]]
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
        self.status.set('Подгонка областей… Интерфейс доступен; изменение параметров отменит показ старого результата.')
        raw = self.raw
        path = self.source_path
        reload = path and self.loaded_filter != recipe.processing.remove_digital_filter
        def work():
            source = load_raw(path, recipe.processing.remove_digital_filter) if reload else raw
            return analyze(process(source, recipe.processing), recipe)
        self.submit('fit', work, self.analysis_ready, self.generation)

    def analysis_ready(self, analysis):
        self.analysis = analysis
        self.spectrum = analysis.spectrum
        self.fit_button.configure(state='normal')
        for name, value in analysis.integrals.items():
            self.integral_tree.insert('', 'end', text=name, values=(f'{value:.9g}',))
        lines = [f'Статус: {analysis.status}', f'P0/P1: {analysis.spectrum.phases}', '']
        for name, result in analysis.results.items():
            d = result.diagnostics
            lines.append(f"{name}: {len(result.centers)} пиков; сошлось={result.success}")
            lines.append(f"RMSE={d.get('relative_rmse')}; nfev={d.get('nfev')}; condition={d.get('condition_number')}")
            lines.append(' №     ppm (фикс.)   A            HWHM ppm     A×HWHM')
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
        path = filedialog.asksaveasfilename(defaultextension='.json',
                    initialfile='spintegrate.settings.json', filetypes=[('Настройки JSON', '*.json')])
        if path:
            try:
                recipe.save(path)
                self.session = recipe
                self.status.set('Настройки и области сохранены: ' + path)
            except Exception as exc:
                self.error(str(exc))

    def load_settings(self):
        path = filedialog.askopenfilename(filetypes=[('Настройки JSON', '*.json')])
        if not path:
            return
        try:
            self.session = Session.load(path)
            self._sync_widgets()
            self.invalidate()
            self.batch_recipe = copy.deepcopy(self.session)
            self._recipe_description()
            self.preview()
        except Exception as exc:
            self.error(str(exc))

    def remember_for_batch(self):
        try:
            recipe = self.read_settings()
            recipe.validate(require_regions=True)
            self.session = recipe
            self.batch_recipe = copy.deepcopy(recipe)
            self._recipe_description()
            self.tabs.select(self.batch_tab)
            self.status.set('Параметры запомнены для серии. Последующие изменения одиночного анализа не изменят этот снимок.')
        except ValueError as exc:
            self.error(str(exc))

    def _recipe_description(self):
        s = self.batch_recipe
        self.recipe_label.configure(text=f'Запомнено: {len(s.regions)} областей обработки, '
                 f'{len(s.integrals)} интегралов; фаза={s.processing.phase}; '
                 f'LB={s.processing.lb_hz:g} Гц; threshold={s.fit.threshold:g}. '
                 'Центры и амплитуды определяются заново в каждом спектре.')

    def export_single(self):
        if self.analysis is None:
            return self.error('Сначала выполните анализ с текущими параметрами.')
        path = filedialog.asksaveasfilename(defaultextension='.csv', initialfile='spectrum.peaks.csv')
        if not path:
            return
        try:
            with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['region', 'peak', 'center_ppm_fixed', 'initial_amplitude',
                                 'amplitude', 'hwhm_ppm', 'fwhm_hz', 'amplitude_x_hwhm', 'converged'])
                for name, result in self.analysis.results.items():
                    for i, (c, a0, a, w) in enumerate(zip(result.centers, result.initial_amplitudes,
                                                        result.amplitudes, result.widths), 1):
                        writer.writerow([name, i, c, a0, a, w, 2*w*self.spectrum.sf_mhz, a*w, result.success])
            integral_path = Path(path).with_suffix('.integrals.csv')
            with integral_path.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['signal', 'integral_A_x_HWHM', 'status'])
                for name, value in self.analysis.integrals.items():
                    writer.writerow([name, value, self.analysis.status])
            self.session.save(Path(path).with_suffix('.settings.json'))
            self.status.set('Сохранены CSV пиков, CSV интегралов и JSON настроек: ' + path)
        except Exception as exc:
            self.error(str(exc))

    def choose_batch_root(self):
        path = filedialog.askdirectory(title='Корневая папка серии')
        if path:
            self.batch_root.set(path)

    def choose_batch_output(self):
        path = filedialog.asksaveasfilename(defaultextension='.csv', initialfile='integrals.csv',
                                          filetypes=[('CSV', '*.csv')])
        if path:
            self.batch_output.set(path)

    def show_batch_paths(self):
        try:
            paths = discover_spectra(self.batch_root.get(), self.batch_recursive.get())
            set_text(self.batch_text, f'Найдено {len(paths)} спектров:\n' + '\n'.join(map(str, paths)))
        except Exception as exc:
            self.error(str(exc))

    def create_demo_series(self):
        directory = filedialog.askdirectory(title='Папка для синтетической демо-серии')
        if directory:
            try:
                target = Path(directory) / 'spintegrate_demo'
                if any((target / f'{n}.npz').exists() for n in range(1, 5)):
                    raise ValueError('Демо-файлы уже существуют. Выберите другую папку.')
                write_demo_series(target)
                self.batch_root.set(str(target))
                self.batch_output.set(str(target / 'integrals.csv'))
                self.batch_recipe = demo_session()
                self._recipe_description()
                self.status.set('Создана синтетическая серия из четырёх FID.')
            except Exception as exc:
                self.error(str(exc))

    def run_batch(self):
        if self.batch_running:
            return
        if self.batch_recipe is None:
            return self.error('Сначала «Запомнить → серия» или загрузить JSON настроек.')
        try:
            recipe = copy.deepcopy(self.batch_recipe)
            recipe.validate(require_regions=True)
            root = self.batch_root.get()
            paths = discover_spectra(root, self.batch_recursive.get())
            if not paths:
                raise ValueError('Спектры не найдены.')
            out = Path(self.batch_output.get())
            if out.suffix.lower() != '.csv' or not out.parent.is_dir():
                raise ValueError('Выберите путь .csv в существующей папке.')
            outputs = [out, out.with_suffix('.settings.json'), out.with_suffix('.diagnostics.json')]
            if any(p.exists() for p in outputs) and not messagebox.askyesno(
                    'Заменить результаты?', 'CSV или связанные JSON уже существуют. Заменить их?', parent=self.root):
                return
        except Exception as exc:
            return self.error(str(exc))
        self.batch_running = True
        self.batch_cancel.clear()
        self.batch_rows.clear()
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
                     'Выберите строку таблицы, чтобы прочитать диагностику спектра.')
        self.submit('batch', lambda: batch_analyze(root, recipe, out, progress,
                           self.batch_cancel.is_set, recursive), finished)

    def _batch_unlock(self):
        self.batch_running = False
        self.batch_button.configure(state='normal')
        self.cancel_button.configure(state='disabled')

    def show_batch_row(self, _=None):
        selection = self.batch_tree.selection()
        if selection:
            i = self.batch_tree.index(selection[0])
            if i < len(self.batch_rows):
                row = self.batch_rows[i]
                set_text(self.batch_text, f"{row['spectrum_id']} — {row['status']}\n{row['warnings']}")

    def open_preprocess(self):
        if self.preprocess_window and self.preprocess_window.window.winfo_exists():
            self.preprocess_window.window.lift()
        else:
            self.preprocess_window = PreprocessWindow(self)

    def close(self):
        if self.batch_running:
            if not messagebox.askyesno('Закрыть Spintegrate',
                    'Серия выполняется. Остановить после текущего спектра и закрыть окно?', parent=self.root):
                return
            self.batch_cancel.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


class PreprocessWindow:
    def __init__(self, app: SpintegrateApp):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title('Аподизация и локальная базовая линия')
        self.window.geometry('1180x790')
        self.window.minsize(1000, 700)
        self.timer = None
        self.revision = 0
        self.loading = True
        self.pending = None
        self.preview_key = None
        self.requested_key = None
        split = ttk.Panedwindow(self.window, orient='horizontal')
        split.pack(fill='both', expand=True)
        left, right = ttk.Frame(split, padding=10), ttk.Frame(split)
        split.add(left, weight=0)
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
        self.fields = FieldPanel(left, self.changed)
        self.fields.pack(fill='x')
        b = app.session.baseline_default
        add = self.fields.add
        add('method', 'Метод', b.method, BASELINE_METHODS, str)
        add('lam', 'λ (Whittaker)', b.lam)
        add('p', 'p (asls / iasls)', b.p)
        add('max_iter', 'Максимум итераций', b.max_iter, kind=int)
        add('tol', 'Допуск tol', b.tol)
        add('half_window', 'Полуокно SNIP / mor', b.half_window, kind=int)
        add('smooth_half_window', 'Сглаживание SNIP, точки', b.smooth_half_window, kind=int)
        add('poly_order', 'Степень modpoly / imodpoly', b.poly_order, kind=int)
        ttk.Button(left, text='Взять базу из выбранной области', command=self.load_region_baseline).pack(fill='x', pady=3)
        ttk.Button(left, text='Применить к выбранной области', command=lambda: self.apply_baseline(False)).pack(fill='x', pady=3)
        ttk.Button(left, text='Применить ко всем областям', command=lambda: self.apply_baseline(True)).pack(fill='x', pady=3)
        ttk.Label(left, text='Изменения здесь задают базу для новых областей. '
                  'Для сохранённых областей нажмите «Применить». LB применяется сразу.',
                  wraplength=350, foreground='#65758a').pack(anchor='w', pady=8)
        self.figure = Figure(figsize=(8, 7), dpi=100, facecolor='white')
        self.ax_apod, self.ax_base = self.figure.subplots(2, 1)
        self.figure.subplots_adjust(left=.10, right=.98, top=.94, bottom=.08, hspace=.4)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(self.canvas, right)
        self.status = tk.StringVar(value='Выберите сохранённую область в главном окне.')
        ttk.Label(self.window, textvariable=self.status, wraplength=1100, padding=8).pack(fill='x')
        self.lb_var.trace_add('write', lambda *_: self.changed())
        self.loading = False
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.refresh()

    def slider_changed(self, value):
        self.lb_var.set(f'{float(value):.3f}')

    def changed(self):
        if self.loading:
            return
        self.revision += 1
        if self.timer:
            self.window.after_cancel(self.timer)
        self.timer = self.window.after(220, self.refresh)

    def load_region_baseline(self):
        r = next((r for r in self.app.session.regions if r.name == self.app.selected_region), None)
        if r:
            self.loading = True
            self.fields.set_values(asdict(r.baseline))
            self.loading = False
            self.changed()

    def refresh(self):
        self.timer = None
        if not self.window.winfo_exists():
            return
        try:
            b = BaselineSettings(**self.fields.values())
            recipe = self.app.read_settings()
            lb = float(self.lb_var.get())
            recipe.processing.lb_hz = lb
            recipe.baseline_default = b
            recipe.validate()
        except (ValueError, tk.TclError) as exc:
            self.status.set(str(exc))
            return
        oldlb = self.app.session.processing.lb_hz
        self.app.session.processing.lb_hz = lb
        self.app.session.baseline_default = b
        self.lb_slider.set(lb)
        if oldlb != lb:
            self.app.invalidate()
            self.app.schedule_preview()
        r = next((r for r in recipe.regions if r.name == self.app.selected_region), None)
        self.region_label.configure(text=f'{r.name}: {r.low:g}…{r.high:g} ppm' if r else 'Сначала сохраните область обработки.')
        if self.app.raw is None:
            return
        key = (id(self.app.raw), repr(recipe.processing), repr(b), repr(r), self.app.generation)
        if key == self.preview_key:
            return
        if key != self.requested_key:
            self.revision += 1
            self.requested_key = key
        revision, generation, raw = self.revision, self.app.generation, self.app.raw
        if 'preprocess' in self.app.jobs:
            # Do not queue unbounded jobs while dragging a slider.
            if self.timer:
                self.window.after_cancel(self.timer)
            self.timer = self.window.after(120, self.refresh)
            return
        def work():
            current = process(raw, recipe.processing)
            no_apod = copy.deepcopy(recipe.processing)
            no_apod.lb_hz = 0.
            reference = process(raw, no_apod)
            baseline_data = None
            if r:
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
            self.ax_apod.clear()
            self.ax_base.clear()
            # Compare at the same ppm coordinates, including independently applied reference rules.
            ref_mask = (reference.ppm >= r.low) & (reference.ppm <= r.high) if r else slice(None)
            cur_mask = (current.ppm >= r.low) & (current.ppm <= r.high) if r else slice(None)
            self.ax_apod.plot(reference.ppm[ref_mask], reference.intensity[ref_mask],
                              color='#a8b4c1', lw=.9, label='LB = 0 Гц')
            self.ax_apod.plot(current.ppm[cur_mask], current.intensity[cur_mask],
                              color='#254b79', lw=1., label=f'LB = {lb:g} Гц')
            self.ax_apod.set_title('Аподизация: текущая область' if r else 'Аподизация: весь спектр', fontsize=11, loc='left')
            self.ax_apod.set_xlim(r.high, r.low) if r else self.ax_apod.set_xlim(current.ppm[-1], current.ppm[0])
            if local:
                x, y, corrected, line, notes = local
                self.ax_base.plot(x, y, color='#a8b4c1', lw=.9, label='Исходная область')
                self.ax_base.plot(x, line, color='#d26335', lw=1., label=f'База: {b.method}')
                self.ax_base.plot(x, corrected, color='#254b79', lw=1., label='После коррекции')
                self.ax_base.set_xlim(r.high, r.low)
                self.status.set(' | '.join(notes) if notes else 'Предпросмотр готов; база применяется только внутри выбранной области.')
            else:
                self.ax_base.text(.5, .5, 'Выберите сохранённую область обработки',
                                  transform=self.ax_base.transAxes, ha='center')
                self.status.set('Аподизация обновлена. Сохраните область для предпросмотра базовой линии.')
            self.ax_base.set_title('Локальная базовая линия', fontsize=11, loc='left')
            for ax in (self.ax_apod, self.ax_base):
                ax.set_xlabel('ppm')
                ax.grid(alpha=.15)
                if ax.lines:
                    ax.legend(fontsize=8)
            self.canvas.draw_idle()
        self.app.submit('preprocess', work, ready)

    def apply_baseline(self, all_regions=False):
        try:
            b = BaselineSettings(**self.fields.values())
            s = self.app.read_settings()
            s.baseline_default = b
            targets = s.regions if all_regions else [r for r in s.regions if r.name == self.app.selected_region]
            if not targets:
                raise ValueError('Выберите сохранённую область обработки.')
            for r in targets:
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


def main():
    root = tk.Tk()
    SpintegrateApp(root)
    root.mainloop()
