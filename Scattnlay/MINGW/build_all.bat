@echo off
setlocal
set "source_dir=%~dp0."
set "build_dir=%~dp0build"
if not defined IFDDA_SCATTNLAY_MINGW_BIN set "IFDDA_SCATTNLAY_MINGW_BIN=F:\mingw64_11.2\bin"
set "PATH=%IFDDA_SCATTNLAY_MINGW_BIN%;%PATH%"

cmake -S "%source_dir%" -B "%build_dir%" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=%IFDDA_SCATTNLAY_MINGW_BIN%/g++.exe"
if errorlevel 1 exit /b %errorlevel%
cmake --build "%build_dir%" --target ifdda_scattnlay_shared ifdda_scattnlay_sections
if errorlevel 1 exit /b %errorlevel%

set "output_dir=%build_dir%\scattnlay"
cmake -E copy_if_different "%output_dir%\ifdda_scattnlay.dll" "%output_dir%\ifdda_scattnlay_sections.exe" "%output_dir%\libifdda_scattnlay.dll.a" "%build_dir%"
if errorlevel 1 exit /b %errorlevel%
echo %build_dir%\ifdda_scattnlay.dll
echo %build_dir%\ifdda_scattnlay_sections.exe
