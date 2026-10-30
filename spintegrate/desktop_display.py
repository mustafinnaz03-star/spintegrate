"""Desktop-only rendering preferences; never changes spectral or exported data."""
import ctypes
import sys
from types import SimpleNamespace
from tkinter import font as tkfont, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


UI_SCALES = (80, 90, 100, 110, 125, 150)
SURFACE = '#f3f6fb'
INK = '#213650'


class SpectrumCanvas(FigureCanvasTkAgg):
    """Keep the native-DPI backing store the same size as the allocated Tk widget.

    Matplotlib 3.10.3 changes the requested widget size when its pixel ratio
    changes on Map. In an embedded expanding pane that can clip the figure.
    Rebuild at the allocated physical size without changing the layout request.
    """
    def _update_device_pixel_ratio(self, event=None):
        if sys.platform != 'win32':
            return super()._update_device_pixel_ratio(event)
        ratio = round(float(self._tkcanvas.tk.call('tk', 'scaling'))/(96/72), 2)
        if self._set_device_pixel_ratio(ratio):
            width, height = self._tkcanvas.winfo_width(), self._tkcanvas.winfo_height()
            if width > 1 and height > 1:
                self.resize(SimpleNamespace(width=width, height=height))


def enable_windows_dpi():
    """Call before creating Tk. Existing hosts (e.g. Jupyter) retain their context."""
    if sys.platform != 'win32':
        return
    try:
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        set_context = user32.SetProcessDpiAwarenessContext
        set_context.argtypes = [ctypes.c_void_p]
        set_context.restype = ctypes.c_bool
        if set_context(ctypes.c_void_p(-4)):  # PER_MONITOR_AWARE_V2
            return
        if ctypes.get_last_error() == 5:  # awareness already chosen by this host
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def display_factor(window):
    return max(.75, min(3., float(window.winfo_fpixels('1i'))/96.))


def fit_window(window, width=1800, height=1080, min_width=960, min_height=620):
    """Size in logical pixels, bounded by this display's usable dimensions."""
    factor = display_factor(window)
    screen_w, screen_h = window.winfo_screenwidth(), window.winfo_screenheight()
    work_w, work_h = screen_w, screen_h-int(48*factor)
    if sys.platform == 'win32':
        try:
            from ctypes import wintypes
            rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0):
                work_w, work_h = rect.right-rect.left, rect.bottom-rect.top
        except (AttributeError, OSError):
            pass
    available_w, available_h = max(600, work_w-int(40*factor)), max(440, work_h-int(55*factor))
    window.geometry(f'{min(round(width*factor), available_w)}x{min(round(height*factor), available_h)}')
    window.minsize(min(round(min_width*factor), available_w), min(round(min_height*factor), available_h))


def apply_style(root, percent=100):
    ratio, pixels = percent/100., display_factor(root)
    style = ttk.Style(root)
    style.theme_use('clam')
    size = max(8, round(10*ratio))
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont'):
        tkfont.nametofont(name, root=root).configure(family='Segoe UI', size=size)
    style.configure('.', font=('Segoe UI', size), background=SURFACE, foreground=INK)
    style.configure('TFrame', background=SURFACE)
    style.configure('TLabel', background=SURFACE)
    style.configure('TButton', padding=(round(10*ratio), round(5*ratio)),
                    background='#e7eef8', foreground=INK, bordercolor='#ccd8e8', relief='flat')
    style.map('TButton', background=[('active', '#d5e4f6'), ('pressed', '#c2d8f2')],
              foreground=[('disabled', '#8995a6')])
    style.configure('Accent.TButton', background='#2563a6', foreground='white', bordercolor='#2563a6')
    style.map('Accent.TButton', background=[('disabled', '#adc0d4'), ('pressed', '#174b83'), ('active', '#3477bd')],
              foreground=[('disabled', '#eef3f9'), ('!disabled', 'white')])
    style.configure('TEntry', padding=round(4*ratio), fieldbackground='white', bordercolor='#c9d5e5')
    style.configure('TCombobox', padding=round(4*ratio), fieldbackground='white', bordercolor='#c9d5e5')
    style.map('TCombobox', fieldbackground=[('readonly', 'white')], selectbackground=[('readonly', '#e7eef8')],
              selectforeground=[('readonly', INK)])
    style.configure('TNotebook', background=SURFACE, borderwidth=0, tabmargins=(0, 2, 0, 0))
    style.configure('TNotebook.Tab', padding=(round(13*ratio), round(7*ratio)), background='#e4ebf5', borderwidth=0)
    style.map('TNotebook.Tab', background=[('selected', 'white'), ('active', '#d6e4f5')],
              foreground=[('selected', '#1d5f9f')])
    style.configure('TLabelframe', background=SURFACE, bordercolor='#d3deec', relief='solid')
    style.configure('TLabelframe.Label', background=SURFACE, foreground='#285780', font=('Segoe UI', size, 'bold'))
    style.configure('Treeview', background='white', fieldbackground='white', bordercolor='#d3deec',
                    rowheight=round(27*ratio*pixels), font=('Segoe UI', size))
    style.configure('Treeview.Heading', background='#e6edf7', foreground=INK,
                    font=('Segoe UI', size, 'bold'), padding=(7, 5), relief='flat')
    style.map('Treeview', background=[('selected', '#d9eafa')], foreground=[('selected', '#173b60')])
    root.configure(background=SURFACE)


def scale_explicit_fonts(widget, percent):
    """Keep explicit hint/heading/text fonts consistent with the shared ttk style."""
    value = str(widget.cget('font')) if 'font' in widget.keys() else ''
    if value and value not in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont'):
        if not hasattr(widget, '_display_font'):
            actual = tkfont.Font(root=widget, font=widget.cget('font')).actual()
            widget._display_font = actual
        base = widget._display_font
        widget.configure(font=(base['family'], max(7, round(abs(base['size'])*percent/100.)),
                               base['weight'], base['slant']))
    for child in widget.winfo_children():
        scale_explicit_fonts(child, percent)


def style_live_figure(figure, percent=100, shared_labels=False):
    """Render to the Tk backing store at native DPI; export uses its own settings."""
    ratio = percent/100.
    axes = figure.axes
    for axis in axes:
        axis.title.set_fontsize(10*ratio)
        axis.xaxis.label.set_fontsize(9*ratio)
        axis.yaxis.label.set_fontsize(9*ratio)
        axis.tick_params(labelsize=8*ratio)
        for text in axis.texts:
            text.set_fontsize(8*ratio)
        if axis.get_legend():
            for text in axis.get_legend().get_texts():
                text.set_fontsize(8*ratio)
    if shared_labels and len(axes) == 3:
        axes[0].set_xlabel('')
        axes[1].set_xlabel('')
