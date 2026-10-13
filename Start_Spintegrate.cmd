@echo off
setlocal
cd /d "%~dp0"
if exist ".appenv\Scripts\python.exe" (
    ".appenv\Scripts\python.exe" spintegrate_app.py
) else (
    python spintegrate_app.py
)
if errorlevel 1 (
    echo.
    echo Launch failed. See README.md for Python / Anaconda setup.
    pause
)
