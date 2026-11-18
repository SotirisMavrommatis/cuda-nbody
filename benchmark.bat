@echo off
REM Verify CPU vs GPU and benchmark. Double-click this file.

cd /d "%~dp0"
if not exist output mkdir output
set LOG=output\benchmark_log.txt
echo ==== benchmark ==== > "%LOG%"

call _build.bat "%LOG%"
if errorlevel 1 goto :show

echo ---- run ---- >> "%LOG%"
nbody.exe bench >> "%LOG%" 2>&1

:show
type "%LOG%"
echo.
pause
