@echo off
setlocal
call "%~dp0src\go.bat"
if errorlevel 1 exit /b 1
rem NASA sources are supplied separately; build the fast reference when present.
if exist "%~dp0Mishchenko\NASA\ampld.lp.f" if exist "%~dp0Mishchenko\NASA\ampld.par.f" if exist "%~dp0Mishchenko\NASA\lpd.f" (
    call "%~dp0Mishchenko\build_windows.bat"
    if errorlevel 1 exit /b 1
)
exit /b 0
