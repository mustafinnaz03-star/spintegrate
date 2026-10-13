"""Generate clearly labeled synthetic example inputs and outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spintegrate.core import batch_analyze
from spintegrate.demo import demo_session, write_demo_series


if __name__ == '__main__':
    project = Path(__file__).resolve().parents[1]
    directory = write_demo_series(project/'examples'/'demo_series')
    try:
        input_directory = directory.relative_to(Path.cwd())
    except ValueError:
        input_directory = directory
    rows, cancelled = batch_analyze(input_directory, demo_session(), project/'examples'/'demo_integrals.csv')
    print(f'Created {len(rows)} synthetic example spectra and demo_integrals.csv')
