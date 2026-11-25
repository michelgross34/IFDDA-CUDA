@echo off
setlocal
set "source_dir=%~dp0."
set "build_dir=%~dp0build"

cmake -S "%source_dir%" -B "%build_dir%" -G "Visual Studio 16 2019" -A x64
if errorlevel 1 exit /b %errorlevel%
cmake --build "%build_dir%" --config Release --target ifdda_scattnlay_shared ifdda_scattnlay_sections
if errorlevel 1 exit /b %errorlevel%

set "output_dir=%build_dir%\scattnlay\Release"
cmake -E copy_if_different "%output_dir%\ifdda_scattnlay.dll" "%output_dir%\ifdda_scattnlay_sections.exe" "%output_dir%\ifdda_scattnlay.lib" "%build_dir%"
if errorlevel 1 exit /b %errorlevel%
echo %build_dir%\ifdda_scattnlay.dll
echo %build_dir%\ifdda_scattnlay_sections.exe
