"""Bundle source, instructions and demos, excluding machine-specific environments."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


if __name__ == '__main__':
    project = Path(__file__).resolve().parents[1]
    files = [project/name for name in ('README.md', 'requirements.txt', 'requirements-test.txt',
             'Start_Spintegrate.cmd', 'spintegrate_app.py', 'Spintegrate_GUI.ipynb')]
    for folder in ('spintegrate', 'tests', 'scripts', 'examples', 'docs'):
        files.extend(p for p in (project/folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc')
    target = project/'Spintegrate_GUI.zip'
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, 'Spintegrate_GUI/'+path.relative_to(project).as_posix())
    print(target)
