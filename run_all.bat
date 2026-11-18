@echo off
REM Run every scenario in the scenarios folder and make a GIF for each.

cd /d "%~dp0"
if not exist output mkdir output
set LOG=output\run_all_log.txt
echo ==== all scenarios ==== > "%LOG%"

call _build.bat "%LOG%"
if errorlevel 1 goto :show

set PY=python
if exist "%USERPROFILE%\anaconda3\python.exe" set PY=%USERPROFILE%\anaconda3\python.exe

for %%F in (scenarios\*.txt) do (
    echo. >> "%LOG%"
    nbody.exe "%%F" output\frames.bin >> "%LOG%" 2>&1
    "%PY%" render.py output\frames.bin "output\%%~nF.gif" "%%~nF" >> "%LOG%" 2>&1
)
if exist output\frames.bin del output\frames.bin
echo. >> "%LOG%"
echo ALL DONE >> "%LOG%"

:show
type "%LOG%"
echo.
pause
