@echo off
REM Run one scenario, then open it in the interactive viewer.
REM   - Drag a scenario .txt file onto this .bat, or
REM   - double-click it to run scenarios\galaxies.txt
REM Results go to the output folder:
REM   <name>.frames  open it in viewer.html (play, pause, rewind, 3D)
REM   <name>.gif     animated GIF of the same simulation

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
nbody.exe "%SCEN%" "output\%NAME%.frames" >> "%LOG%" 2>&1
if errorlevel 1 goto :show

echo ---- render GIF ---- >> "%LOG%"
set PY=python
if exist "%USERPROFILE%\anaconda3\python.exe" set PY=%USERPROFILE%\anaconda3\python.exe
"%PY%" render.py "output\%NAME%.frames" "output\%NAME%.gif" "%NAME%" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo Opening the viewer: drag output\%NAME%.frames into the browser window. >> "%LOG%"
start "" viewer.html
start "" explorer /select,"%~dp0output\%NAME%.frames"

:show
type "%LOG%"
echo.
pause
