@echo off
echo ===================================================
echo Starting AI Health & Medicine Recommendation System
echo ===================================================

cd /d "%~dp0"

IF EXIST ".venv\Scripts\python.exe" (
    echo Using virtual environment at .venv...
    .venv\Scripts\python.exe main.py
) ELSE (
    echo Using system Python...
    python main.py
)

pause
