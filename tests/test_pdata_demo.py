import numpy as np

from spintegrate.core import Session, batch_analyze, discover_spectra, load_pdata, process
from spintegrate.demo import demo_raw, write_pdata_demo_series


def test_processed_demo_generator_and_direct_batch(tmp_path):
    directory = write_pdata_demo_series(tmp_path/'dataset', procno='2')
    recipe = Session.load(directory/'demo.pdata.settings.json')
    assert recipe.processing.source_mode == 'pdata'
    assert recipe.processing.pdata_procno == '2'
    assert len(discover_spectra(directory, source_mode='pdata', pdata_procno='2')) == 4
    assert discover_spectra(directory, source_mode='raw') == []
    imported = load_pdata(directory/'1'/'pdata'/'2')
    expected = process(demo_raw(), recipe.processing)
    np.testing.assert_array_equal(imported.intensity, expected.intensity)
    np.testing.assert_allclose(imported.ppm, expected.ppm, rtol=0, atol=1e-12)
    assert imported.processing_metadata['has_imaginary'] is True
    recipe.analysis_method = 'direct'
    rows, stopped = batch_analyze(directory, recipe, tmp_path/'direct.csv')
    assert not stopped and len(rows) == 4
    assert [r['spectrum_number'] for r in rows] == ['1', '2', '3', '4']
    assert all(r['status'] != 'error' for r in rows)
    assert rows[3]['integral__signal'] / rows[0]['integral__signal'] > .35
