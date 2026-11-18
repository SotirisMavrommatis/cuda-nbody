@echo off
REM Run every scenario in the scenarios folder.
REM For each one you get output\<name>.frames (for viewer.html) and output\<name>.gif

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
    nbody.exe "%%F" "output\%%~nF.frames" >> "%LOG%" 2>&1
    "%PY%" render.py "output\%%~nF.frames" "output\%%~nF.gif" "%%~nF" >> "%LOG%" 2>&1
)
echo. >> "%LOG%"
echo ALL DONE >> "%LOG%"

:show
type "%LOG%"
echo.
pause
