@echo off
title OPS-SAT
set PYTHONUNBUFFERED=1
set PYTHONPATH=%~dp0

"%~dp0.venv\Scripts\python.exe" -u "%~dp0scripts\start_ui.py"

echo.
pause
