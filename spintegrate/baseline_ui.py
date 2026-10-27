"""Grouped baseline chooser with method-specific, safely parsed parameter controls."""
import copy
import json
import math
import tkinter as tk
import webbrowser
from dataclasses import asdict
from tkinter import ttk

from .baseline_methods import CATALOG, LEGACY_METHODS, effective_parameters, method_summary, optimizer_methods, parameter_specs
from .core import BaselineSettings
from .i18n import canonical, set_label, tr, translate_widgets, widget_language


GROUP_LABELS = {
    'all': 'Все методы', 'none': 'Без коррекции',
    'whittaker': 'Уиттекер / штрафные наименьшие квадраты',
    'morphological': 'Морфологические', 'polynomial': 'Полиномиальные',
    'spline': 'Сплайны', 'smooth': 'Сглаживание / отсечение пиков',
    'classification': 'Классификация базы и пиков', 'optimizers': 'Оптимизаторы', 'misc': 'Другие методы',
}


def parse_parameter(text, default):
    if isinstance(text, bool):
        return text
    text = str(text).strip()
    if text.lower() in ('', 'auto', 'none', 'null'):
        return None
    if isinstance(default, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            value = float(text)
            return int(value) if isinstance(default, int) and value.is_integer() else value
        except ValueError:
            if text in ('l1_v1', 'l1_v2'):
                return text
            raise ValueError('Use a number, auto, or valid JSON for lists and objects.') from None


class BaselineControls(ttk.Frame):
    def __init__(self, parent, settings, callback):
        super().__init__(parent)
        self.callback, self.busy, self.drafts = callback, True, {}
        self.settings = copy.deepcopy(settings)
        self.method = settings.method
        self.group = CATALOG.get(self.method, 'none')
        self.columnconfigure(1, weight=1)
        self.group_var = tk.StringVar(self)
        self.search_var = tk.StringVar(self)
        self.method_var = tk.StringVar(self, value=self.method)
        for row, text in enumerate(('Семейство алгоритмов', 'Поиск метода', 'Метод')):
            ttk.Label(self, text=text).grid(row=row, column=0, sticky='w', padx=(0, 8), pady=3)
        self.group_combo = ttk.Combobox(self, textvariable=self.group_var, state='readonly', width=28)
        self.group_combo.grid(row=0, column=1, sticky='ew')
        ttk.Entry(self, textvariable=self.search_var).grid(row=1, column=1, sticky='ew')
        self.method_combo = ttk.Combobox(self, textvariable=self.method_var, state='readonly', width=28)
        self.method_combo.grid(row=2, column=1, sticky='ew')
        self.summary = ttk.Label(self, wraplength=370, foreground='#65758a')
        self.summary.grid(row=3, column=0, columnspan=2, sticky='ew', pady=6)
        actions = ttk.Frame(self)
        actions.grid(row=4, column=0, columnspan=2, sticky='ew')
        ttk.Button(actions, text='Параметры по умолчанию', command=self.reset_defaults).pack(side='left')
        ttk.Button(actions, text='Справка по методу', command=self.open_help).pack(side='left', padx=5)
        self.advanced = tk.BooleanVar(self, value=False)
        ttk.Checkbutton(self, text='Расширенные параметры', variable=self.advanced,
                        command=self.toggle_advanced).grid(row=5, column=0, columnspan=2, sticky='w')
        self.form = ttk.Frame(self)
        self.form.columnconfigure(1, weight=1)
        self.form.grid(row=6, column=0, columnspan=2, sticky='ew', pady=5)
        ttk.Label(self, text='auto — выбор библиотеки. Списки и объекты вводятся в JSON. '
                  'Пороговые интенсивности задаются для сигнала, нормированного к max|I| = 1.',
                  wraplength=370, foreground='#65758a').grid(row=7, column=0, columnspan=2, sticky='ew', pady=5)
        self.group_var.trace_add('write', self.group_changed)
        self.search_var.trace_add('write', lambda *_: self.filter_methods())
        self.method_var.trace_add('write', self.method_changed)
        self.load(settings)
        self.busy = False

    def translate_choices(self, language):
        busy = self.busy
        self.busy = True
        self.group_combo.configure(values=[tr(label, language) for label in GROUP_LABELS.values()])
        self.group_var.set(tr(GROUP_LABELS[self.group], language))
        self.busy = busy

    def filter_methods(self):
        query = self.search_var.get().strip().lower()
        methods = ['none', *CATALOG]
        if query:
            methods = [m for m in methods if query in m]
        elif self.group != 'all':
            methods = [m for m in methods if CATALOG.get(m, 'none') == self.group]
        self.method_combo.configure(values=methods)
        return methods

    def group_changed(self, *_):
        if self.busy:
            return
        source = canonical(self.group_var.get())
        self.group = next(key for key, label in GROUP_LABELS.items() if label == source)
        self.search_var.set('')
        methods = self.filter_methods()
        if self.method not in methods and methods:
            self.method_var.set(methods[0])

    def method_changed(self, *_):
        name = self.method_var.get()
        if self.busy or name == self.method or name not in ('none', *CATALOG):
            return
        try:
            self.drafts[self.method] = BaselineSettings(**self.values())
        except ValueError:
            pass
        self.load(self.drafts.get(name, BaselineSettings(method=name)))
        self.notify()

    def load(self, settings):
        busy, self.busy = self.busy, True
        self.settings = copy.deepcopy(settings)
        self.method = settings.method
        self.method_var.set(self.method)
        self.search_var.set('')
        self.group = CATALOG.get(self.method, 'none')
        self.translate_choices(widget_language(self))
        self.filter_methods()
        set_label(self.summary, method_summary(self.method))
        for child in self.form.winfo_children():
            child.destroy()
        self.vars, self.specs, self.names = {'method': self.method_var}, parameter_specs(self.method), {}
        self.advanced_widgets = []
        self.initial = {**self.specs, **effective_parameters(settings)}
        row = 0
        for name, default in self.specs.items():
            value = self.initial[name]
            key = 'inner_method' if name == 'method' else ('half_window' if self.method == 'snip' and name == 'max_half_window' else name)
            self.names[name] = key
            label = 'Опорные позиции, ppm' if name == 'anchor_ppm' else name
            parameter_label = ttk.Label(self.form, text=label)
            parameter_label.grid(row=row, column=0, sticky='w', padx=(0, 8), pady=3)
            var = tk.BooleanVar(self, value=value) if isinstance(default, bool) else tk.StringVar(self,
                value='auto' if value is None else (value if isinstance(value, str) else json.dumps(value)))
            self.vars[key] = var
            if isinstance(default, bool):
                widget = ttk.Checkbutton(self.form, variable=var)
            elif name == 'method':
                choices = optimizer_methods(self.method)
                widget = ttk.Combobox(self.form, textvariable=var, values=choices, state='readonly', width=23)
            elif name == 'interp_method':
                widget = ttk.Combobox(self.form, textvariable=var,
                    values=('linear', 'nearest', 'slinear', 'quadratic', 'cubic', 'previous', 'next'), state='readonly', width=23)
            else:
                widget = ttk.Entry(self.form, textvariable=var, width=25)
            widget.grid(row=row, column=1, sticky='ew', pady=3)
            if name in ('weights', 'alpha', 'pad_kwargs', 'window_kwargs', 'method_kwargs',
                        'return_coef', 'conserve_memory', 'roi'):
                self.advanced_widgets.extend((parameter_label, widget))
            var.trace_add('write', lambda *_: self.notify())
            row += 1
            if name == 'lam' and value is not None and value > 0:
                slider_var = tk.DoubleVar(self, value=math.log10(value))
                self.lambda_slider = ttk.Scale(self.form, from_=-2, to=14, variable=slider_var,
                    command=lambda value, v=var: v.set(f'{10**float(value):.6g}'))
                self.lambda_slider.grid(row=row, column=0, columnspan=2, sticky='ew')
                def sync_slider(*_, variable=var, slider_value=slider_var):
                    try:
                        numeric = float(variable.get())
                        if numeric > 0 and math.isfinite(numeric):
                            slider_value.set(max(-2., min(14., math.log10(numeric))))
                    except ValueError:
                        pass
                var.trace_add('write', sync_slider)
                row += 1
        self.busy = busy
        self.toggle_advanced()
        translate_widgets(self.form, widget_language(self))

    def toggle_advanced(self):
        for widget in self.advanced_widgets:
            widget.grid() if self.advanced.get() else widget.grid_remove()

    def values(self):
        settings = copy.deepcopy(self.settings)
        legacy = effective_parameters(BaselineSettings(method=self.method))
        for name, default in self.specs.items():
            value = parse_parameter(self.vars[self.names[name]].get(), default)
            if value == self.initial[name]:
                continue
            target = 'half_window' if self.method == 'snip' and name == 'max_half_window' else name
            if self.method in LEGACY_METHODS and name in legacy and hasattr(settings, target) and isinstance(value, (int, float)) and name not in settings.parameters:
                setattr(settings, target, value)
            else:
                settings.parameters[name] = value
        return asdict(settings)

    def set_values(self, values):
        self.drafts.clear()
        self.load(BaselineSettings(**values))

    def notify(self):
        if not self.busy:
            self.callback()

    def reset_defaults(self):
        self.load(BaselineSettings(method=self.method))
        self.notify()

    def open_help(self):
        if self.method != 'none':
            webbrowser.open('https://pybaselines.readthedocs.io/en/stable/generated/api/pybaselines.Baseline.'+self.method+'.html')
