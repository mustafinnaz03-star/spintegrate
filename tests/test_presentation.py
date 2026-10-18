import copy
import json
import re
from dataclasses import asdict

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from PIL import Image
import numpy as np
import pytest

from spintegrate.appearance import AppearanceSettings, area_color, validate_appearance
from spintegrate.core import Session, analyze, load_analysis, process, save_analysis
from spintegrate.demo import demo_raw, demo_session
from spintegrate.figure_export import render_fit_figure, save_figure
from spintegrate.i18n import canonical, tr


@pytest.fixture
def fitted():
    recipe = demo_session()
    return analyze(process(demo_raw(), recipe.processing), recipe), recipe


def test_old_settings_and_fit_files_get_compatible_presentation_defaults(tmp_path):
    old = asdict(demo_session())
    old.pop('appearance')
    for area in old['integrals']:
        area.pop('color')
    path = tmp_path/'old.json'
    path.write_text(json.dumps(old), encoding='utf-8')
    restored = Session.load(path)
    assert restored.appearance.language == 'ru'
    assert area_color(restored.integrals[0], 0) != area_color(restored.integrals[1], 1)
    # Recreate a pre-presentation archive, even if bundled examples are updated later.
    with np.load('examples/demo_integrals.fits/000001.spfit', allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    metadata = json.loads(str(arrays['metadata'].item()))
    metadata['session'].pop('appearance', None)
    for area in metadata['session']['integrals']:
        area.pop('color', None)
    arrays['metadata'] = np.array(json.dumps(metadata))
    with (tmp_path/'old.spfit').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    analysis, recipe = load_analysis(tmp_path/'old.spfit')
    assert analysis.results and recipe.appearance.export_dpi == 300


def test_language_color_and_export_preferences_survive_json_and_spfit(tmp_path, fitted):
    analysis, recipe = fitted
    recipe.integrals[0].color = '#CC79A7'
    recipe.appearance = AppearanceSettings(language='en', export_dpi=600, export_format='tiff')
    recipe.save(tmp_path/'settings.json')
    assert Session.load(tmp_path/'settings.json') == recipe
    save_analysis(tmp_path/'fit.spfit', analysis, recipe)
    restored, restored_recipe = load_analysis(tmp_path/'fit.spfit')
    assert restored_recipe == recipe
    np.testing.assert_array_equal(restored.results['Сигнал'].fitted, analysis.results['Сигнал'].fitted)


@pytest.mark.parametrize('layout', ['full', 'fit', 'region'])
def test_publication_figure_has_english_labels_and_custom_region_color(layout, fitted):
    analysis, recipe = fitted
    recipe.integrals[1].color = '#CC79A7'
    options = AppearanceSettings(export_language='en', export_layout=layout)
    figure = render_fit_figure(analysis, recipe, 'Сигнал', options)
    FigureCanvasAgg(figure).draw()
    texts = []
    for ax in figure.axes:
        texts += [ax.get_title(loc='left'), ax.get_xlabel(), ax.get_ylabel()]
        if ax.get_legend():
            texts += [t.get_text() for t in ax.get_legend().get_texts()]
        texts += [t.get_text() for t in ax.texts]
    assert not re.search('[А-Яа-яЁё]', '\n'.join(texts))
    assert any('Lorentzian' in text for text in texts)
    detail = next(ax for ax in figure.axes if 'peaks;' in ax.get_title(loc='left'))
    assert to_hex(detail.patches[0].get_facecolor()) == '#cc79a7'
    assert recipe.appearance.language == 'ru'  # independent export language


@pytest.mark.parametrize('dpi', [150, 300, 600])
def test_png_size_and_dpi_are_exact_and_fitted_arrays_unchanged(tmp_path, dpi, fitted):
    analysis, recipe = fitted
    original = analysis.results['Сигнал'].fitted.copy()
    options = AppearanceSettings(export_dpi=dpi, export_width_mm=101.6, export_height_mm=76.2,
                                 export_layout='region', export_language='en')
    figure = render_fit_figure(analysis, recipe, 'Сигнал', options)
    path = tmp_path/'plot.png'
    save_figure(path, figure, options)
    with Image.open(path) as picture:
        assert picture.size == (4*dpi, 3*dpi)
        assert picture.info['dpi'][0] == pytest.approx(dpi, abs=.02)
    np.testing.assert_array_equal(original, analysis.results['Сигнал'].fitted)


@pytest.mark.parametrize('format', ['tiff', 'svg'])
def test_tiff_and_vector_svg_export(tmp_path, format, fitted):
    analysis, recipe = fitted
    options = AppearanceSettings(export_format=format, export_layout='region', export_language='en')
    path = tmp_path/f'plot.{format}'
    save_figure(path, render_fit_figure(analysis, recipe, 'Сигнал', options), options)
    if format == 'svg':
        xml = path.read_text(encoding='utf-8')
        assert '<svg' in xml and '<path' in xml
        assert not re.search('[А-Яа-яЁё]', xml)
    else:
        with Image.open(path) as picture:
            assert picture.format == 'TIFF'
            assert picture.info['compression'] == 'tiff_lzw'


def test_invalid_color_and_excessive_raster_dimensions_are_rejected():
    recipe = demo_session()
    recipe.integrals[0].color = 'not a color'
    with pytest.raises(ValueError, match='RRGGBB'):
        recipe.validate()
    with pytest.raises(ValueError, match='мегапикселей'):
        validate_appearance(AppearanceSettings(export_width_mm=500, export_height_mm=500, export_dpi=2400))


def test_translations_keep_variable_values_and_choice_codes():
    assert tr('Сумма Лоренцев', 'en') == 'Lorentzian sum'
    assert tr('Стандарт: 2 пиков; сошлось', 'en') == 'Стандарт: 2 peaks; converged'
    assert tr('Сигнал: 2 пиков; сошлось', 'en') == 'Сигнал: 2 peaks; converged'
    assert canonical(tr('Без коррекции', 'en')) == 'Без коррекции'
    assert tr('Датасет 167', 'en') == 'Датасет 167'  # an unrecognized user name is not rewritten
