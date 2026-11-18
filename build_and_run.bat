@echo off
REM Builds nbody.cu, runs verification + benchmark + galaxy simulation,
REM then renders the GIF with Python. Double-click this file.

cd /d "%~dp0"
set LOG=%~dp0run_log.txt
echo ==== cuda-nbody build and run log ==== > "%LOG%"

set VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (
    echo ERROR: Could not find vcvars64.bat >> "%LOG%"
    goto :show
)
call "%VCVARS%" >nul

echo ---- compile ---- >> "%LOG%"
if exist nbody.exe del nbody.exe
nvcc -O3 -o nbody nbody.cu >> "%LOG%" 2>&1
if not exist nbody.exe (
    echo ERROR: compile failed >> "%LOG%"
    goto :show
)

echo ---- run ---- >> "%LOG%"
nbody.exe >> "%LOG%" 2>&1

echo ---- render GIF ---- >> "%LOG%"
set PY=python
if exist "%USERPROFILE%\anaconda3\python.exe" set PY=%USERPROFILE%\anaconda3\python.exe
"%PY%" render.py frames.bin galaxy_collision.gif >> "%LOG%" 2>&1

:show
type "%LOG%"
echo.
echo Output also saved to run_log.txt
pause
