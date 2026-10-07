@echo off
setlocal
title OPS-SAT Telemetry Diagnostics - Start All Services

rem -------------------------------------------------------------
rem Thin launcher. ALL startup logic lives in scripts\start_ui.py,
rem so there is only one source of truth: the same script that
rem README.md tells people to run, and the only one that also
rem works on Linux / macOS.
rem
rem Double-click this file, or run:  start_all.bat --no-open
rem -------------------------------------------------------------

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "LAUNCHER=%ROOT%scripts\start_ui.py"

echo ============================================================
echo   OPS-SAT Telemetry Diagnostics
echo   One-click launcher  -  scripts\start_ui.py
echo ============================================================
echo.

if not exist "%PY%" (
  echo       [ERROR] Not found: %PY%
  echo              Create it with:  python -m venv .venv
  echo.
  pause
  exit /b 1
)
if not exist "%LAUNCHER%" (
  echo       [ERROR] Not found: %LAUNCHER%
  echo.
  pause
  exit /b 1
)

rem Put the bundled Node.js first on PATH, so a cmd window opened
rem from Windows Explorer still finds node.exe even when the user
rem PATH has no Node at all.

set "PATH=%ROOT%node-runtime;%PATH%"

rem Forward optional arguments (for example --no-open) to the launcher.
set "ARGS="
if not "%~1"=="" set "ARGS=%~1"

start "OPS-SAT Services (API + UI)" /D "%ROOT%" "%PY%" "%LAUNCHER%" %ARGS%

echo       OK   Launching backend + frontend.
echo.
echo   Everything runs in the window titled "OPS-SAT Services".
echo   Close that window (or press Ctrl+C) to stop both services.
echo.
pause
