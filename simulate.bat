@echo off
REM Run one scenario and make a GIF.
REM   - Drag a scenario .txt file onto this .bat, or
REM   - double-click it to run scenarios\galaxies.txt
REM Results go to the output folder.

cd /d "%~dp0"
if not exist output mkdir output

set SCEN=%~1
if "%SCEN%"=="" set SCEN=scenarios\galaxies.txt
for %%F in ("%SCEN%") do set NAME=%%~nF
set LOG=output\%NAME%_log.txt
echo ==== %NAME% ==== > "%LOG%"

call _build.bat "%LOG%"
if errorlevel 1 goto :show

echo ---- simulate ---- >> "%LOG%"
nbody.exe "%SCEN%" output\frames.bin >> "%LOG%" 2>&1
if errorlevel 1 goto :show

echo ---- render GIF ---- >> "%LOG%"
set PY=python
if exist "%USERPROFILE%\anaconda3\python.exe" set PY=%USERPROFILE%\anaconda3\python.exe
"%PY%" render.py output\frames.bin "output\%NAME%.gif" "%NAME%" >> "%LOG%" 2>&1
del output\frames.bin

:show
type "%LOG%"
echo.
echo Output is in the output folder.
pause
