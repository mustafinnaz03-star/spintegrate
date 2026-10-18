import csv
import json
import shutil
from dataclasses import asdict

import numpy as np
import pytest

from spintegrate.core import (Session, analyze, batch_analyze, discover_spectra,
                             load_analysis, load_batch_results, process,
                             save_analysis, write_batch_csv)
from spintegrate.demo import demo_raw, demo_session, write_demo_series


def test_saved_fit_is_exact_and_self_contained_without_refitting(tmp_path, monkeypatch):
    import spintegrate.core as core
    recipe = demo_session()
    analysis = analyze(process(demo_raw(), recipe.processing), recipe)
    target = tmp_path/'spectrum.spfit'
    save_analysis(target, analysis, recipe)
    monkeypatch.setattr(core, 'analyze', lambda *_: pytest.fail('Loading must not refit'))
    monkeypatch.setattr(core, 'load_raw', lambda *_: pytest.fail('Loading must not access FID'))
    restored, saved_recipe = load_analysis(target)
    assert asdict(saved_recipe) == asdict(recipe)
    assert restored.integrals == analysis.integrals
    assert restored.errors == analysis.errors
    np.testing.assert_array_equal(restored.spectrum.ppm, analysis.spectrum.ppm)
    np.testing.assert_array_equal(restored.spectrum.intensity, analysis.spectrum.intensity)
    np.testing.assert_array_equal(restored.spectrum.complex_spectrum, analysis.spectrum.complex_spectrum)
    for name, result in analysis.results.items():
        other = restored.results[name]
        for attr in ('x', 'raw', 'baseline', 'corrected', 'centers', 'initial_amplitudes',
                     'amplitudes', 'widths', 'fitted'):
            np.testing.assert_array_equal(getattr(other, attr), getattr(result, attr))
        assert other.diagnostics == result.diagnostics


def test_failed_optimization_stays_failed_after_loading(tmp_path):
    recipe = demo_session()
    recipe.fit.max_nfev = 1
    recipe.fit.restarts = 1
    analysis = analyze(process(demo_raw(), recipe.processing), recipe)
    assert analysis.status == 'error'
    save_analysis(tmp_path/'failed.spfit', analysis, recipe)
    restored, _ = load_analysis(tmp_path/'failed.spfit')
    assert restored.status == 'error'
    assert np.isnan(restored.integrals['signal'])
    assert not restored.results['Сигнал'].success


def test_empty_peak_list_survives_saved_fit(tmp_path):
    recipe = demo_session()
    recipe.fit.threshold = 1e12
    analysis = analyze(process(demo_raw(), recipe.processing), recipe)
    save_analysis(tmp_path/'empty.spfit', analysis, recipe)
    restored, _ = load_analysis(tmp_path/'empty.spfit')
    assert restored.integrals['signal'] == 0.
    assert restored.results['Сигнал'].centers.size == 0


def test_batch_csv_numeric_columns_and_portable_saved_fit_paths(tmp_path):
    write_demo_series(tmp_path/'inputs')
    recipe = demo_session()
    recipe.integrals[0].name = 'Стандарт, площадь'
    recipe.integrals[1].name = 'status'  # must not overwrite technical diagnostics
    out = tmp_path/'export'/'areas.csv'
    out.parent.mkdir()
    rows, _ = batch_analyze(tmp_path/'inputs', recipe, out)
    with out.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        csvrows = list(reader)
        assert reader.fieldnames == ['spectrum_number', 'spectrum_id', 'Стандарт, площадь', 'status']
    for row in csvrows:
        float(row['Стандарт, площадь'])
        float(row['status'])
        assert not any('=' in v for v in row.values())
    loaded_rows, loaded_recipe = load_batch_results(out)
    assert asdict(loaded_recipe) == asdict(recipe)
    assert loaded_rows[0]['status'] == rows[0]['status']
    assert loaded_rows[0]['integral__status'] == rows[0]['integral__status']
    assert loaded_rows[0]['fit_file'].endswith('.spfit')
    original, _ = load_analysis(loaded_rows[0]['fit_file'])
    assert original.integrals['status'] == rows[0]['integral__status']
    # CSV + JSONs + .fits can be moved together; no original FID is required.
    shutil.move(str(out.parent), str(tmp_path/'moved'))
    shutil.rmtree(tmp_path/'inputs')
    moved_rows, _ = load_batch_results(tmp_path/'moved'/'areas.csv')
    assert (tmp_path/'moved'/'areas.fits'/'000001.spfit').is_file()
    moved, _ = load_analysis(moved_rows[0]['fit_file'])
    np.testing.assert_array_equal(moved.results['Сигнал'].fitted, original.results['Сигнал'].fitted)


def test_saved_fit_files_are_not_discovered_as_input_spectra(tmp_path):
    write_demo_series(tmp_path)
    batch_analyze(tmp_path, demo_session(), tmp_path/'run.csv')
    assert len(discover_spectra(tmp_path)) == 4
    # Repeating a batch must not process its own stored results as new FIDs.
    rows, _ = batch_analyze(tmp_path, demo_session(), tmp_path/'again.csv')
    assert len(rows) == 4


def test_legacy_csv_can_be_reopened_with_its_original_recipe(tmp_path):
    recipe = demo_session()
    recipe.save(tmp_path/'old.settings.json')
    path = tmp_path/'old.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence_index', 'spectrum_number', 'spectrum_id',
                               'source_path', 'integral__standard', 'integral__signal', 'status', 'warnings'])
        writer.writeheader()
        writer.writerow(dict(sequence_index=1, spectrum_number='167', spectrum_id='167',
                             source_path=str(tmp_path/'167'), integral__standard=123.,
                             integral__signal=456., status='warning', warnings='legacy warning'))
    rows, loaded = load_batch_results(path)
    assert rows[0]['integral__signal'] == 456.
    assert rows[0]['status'] == 'warning'
    assert rows[0]['fit_file'] == ''
    assert asdict(loaded) == asdict(recipe)


def test_reserved_csv_identifiers_are_rejected():
    recipe = demo_session()
    recipe.integrals[0].name = 'spectrum_number'
    with pytest.raises(ValueError, match='зарезервированы'):
        recipe.validate()


def test_saved_fit_loader_rejects_arbitrary_npz_and_object_arrays(tmp_path):
    raw = tmp_path/'raw.spfit'
    with raw.open('wb') as stream:
        np.savez(stream, fid=np.array([1j]))
    with pytest.raises(ValueError, match='файл подгонки'):
        load_analysis(raw)
    with raw.open('wb') as stream:
        np.savez(stream, metadata=np.array({'malformed': True}, dtype=object))
    with pytest.raises(ValueError, match='Object arrays'):
        load_analysis(raw)
