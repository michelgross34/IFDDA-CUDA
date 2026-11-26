@echo off
setlocal
set "source_dir=%~dp0."
set "build_dir=%~dp0build"
if not defined IFDDA_MISHCHENKO_MINGW_BIN set "IFDDA_MISHCHENKO_MINGW_BIN=F:\mingw64_11.2\bin"
set "PATH=%IFDDA_MISHCHENKO_MINGW_BIN%;%PATH%"

cmake -S "%source_dir%" -B "%build_dir%" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=%IFDDA_MISHCHENKO_MINGW_BIN%/g++.exe" "-DCMAKE_Fortran_COMPILER=%IFDDA_MISHCHENKO_MINGW_BIN%/gfortran.exe" -DIFDDA_MISHCHENKO_BUILD_STANDALONE=ON -DIFDDA_MISHCHENKO_ENABLE_CUDA=ON
if errorlevel 1 exit /b %errorlevel%
cmake --build "%build_dir%" --parallel 4
if errorlevel 1 exit /b %errorlevel%

set "output_dir=%build_dir%\mishchenko\Release"
for %%F in (ifdda_mishchenko.dll ifdda_mishchenko_cpu.dll ifdda_mishchenko_openmp.dll ifdda_mishchenko_cuda_kernel.dll cudart64_12.dll ifdda_mishchenko_sections_cpu.exe ifdda_mishchenko_sections_openmp.exe ifdda_mishchenko_sections_cuda.exe) do (
    cmake -E copy_if_different "%output_dir%\%%F" "%build_dir%"
    if errorlevel 1 exit /b 1
)
for %%F in (libifdda_mishchenko.dll.a libifdda_mishchenko_cpu.dll.a libifdda_mishchenko_openmp.dll.a) do (
    cmake -E copy_if_different "%output_dir%\%%F" "%build_dir%"
    if errorlevel 1 exit /b 1
)
python "%source_dir%\..\test_backends.py" "%build_dir%"
if errorlevel 1 exit /b %errorlevel%
python "%source_dir%\..\test_backends.py" "%build_dir%" --qp
exit /b %errorlevel%
