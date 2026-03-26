@echo off
REM Cross-platform build wrapper for CPS cps_core extension (Windows).
REM Requires: Python 3.x with numpy, gfortran via MSYS2 or Conda.
REM Usage: build.bat [--clean]
python "%~dp0build.py" %*
