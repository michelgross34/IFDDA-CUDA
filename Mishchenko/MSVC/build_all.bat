@echo off
setlocal
set "source_dir=%~dp0."
set "build_dir=%~dp0build"
if not defined IFDDA_MISHCHENKO_GFORTRAN set "IFDDA_MISHCHENKO_GFORTRAN=F:/mingw64_11.2/bin/gfortran.exe"

cmake -S "%source_dir%" -B "%build_dir%" -G "Visual Studio 16 2019" -A x64 "-DIFDDA_MISHCHENKO_GFORTRAN=%IFDDA_MISHCHENKO_GFORTRAN%"
if errorlevel 1 exit /b %errorlevel%
cmake --build "%build_dir%" --config Release --target ifdda_mishchenko_shared ifdda_mishchenko_sections
if errorlevel 1 exit /b %errorlevel%

set "output_dir=%build_dir%\mishchenko\Release"
cmake -E copy_if_different "%output_dir%\ifdda_mishchenko.dll" "%output_dir%\ifdda_mishchenko_sections.exe" "%output_dir%\ifdda_mishchenko.lib" "%build_dir%"
if errorlevel 1 exit /b %errorlevel%
echo %build_dir%\ifdda_mishchenko.dll
echo %build_dir%\ifdda_mishchenko_sections.exe
