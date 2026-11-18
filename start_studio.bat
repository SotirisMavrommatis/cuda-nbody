@echo off
REM Starts the N-body studio: build scenarios with the mouse, run them on the GPU,
REM and watch them in the browser. Double-click this file.

cd /d "%~dp0"
if not exist output mkdir output

set PY=python
if exist "%USERPROFILE%\anaconda3\python.exe" set PY=%USERPROFILE%\anaconda3\python.exe

REM Compile nbody.exe only if it is missing or older than nbody.cu.
"%PY%" -c "import os,sys; sys.exit(0 if os.path.exists('nbody.exe') and os.path.getmtime('nbody.exe') >= os.path.getmtime('nbody.cu') else 1)"
if errorlevel 1 (
    echo Compiling nbody.cu, this takes a few seconds...
    set LOG=output\studio_build_log.txt
    echo ==== studio build ==== > output\studio_build_log.txt
    call _build.bat output\studio_build_log.txt
    if errorlevel 1 (
        type output\studio_build_log.txt
        pause
        exit /b 1
    )
)

"%PY%" server.py
pause
