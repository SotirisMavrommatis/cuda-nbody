@echo off
REM Helper used by the other .bat files: loads the Visual Studio compiler
REM environment and compiles nbody.cu. Usage: call _build.bat <logfile>

set BUILDLOG=%~1
set VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (
    echo ERROR: Could not find vcvars64.bat at "%VCVARS%" >> "%BUILDLOG%"
    exit /b 1
)
call "%VCVARS%" >nul

echo ---- compile ---- >> "%BUILDLOG%"
if exist nbody.exe del nbody.exe
nvcc -O3 -o nbody nbody.cu >> "%BUILDLOG%" 2>&1
if not exist nbody.exe (
    echo ERROR: compile failed >> "%BUILDLOG%"
    exit /b 1
)
exit /b 0
