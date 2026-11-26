@echo off
setlocal
set "source_dir=%~dp0."
set "build_dir=%~dp0build"
if not defined IFDDA_MISHCHENKO_GFORTRAN set "IFDDA_MISHCHENKO_GFORTRAN=F:/mingw64_11.2/bin/gfortran.exe"

cmake -S "%source_dir%" -B "%build_dir%" -G "Visual Studio 16 2019" -A x64 "-DIFDDA_MISHCHENKO_GFORTRAN=%IFDDA_MISHCHENKO_GFORTRAN%" -DIFDDA_MISHCHENKO_BUILD_STANDALONE=ON -DIFDDA_MISHCHENKO_ENABLE_CUDA=ON
if errorlevel 1 exit /b %errorlevel%
cmake --build "%build_dir%" --config Release --parallel 4
if errorlevel 1 exit /b %errorlevel%

set "output_dir=%build_dir%\mishchenko\Release"
for %%F in (ifdda_mishchenko.dll ifdda_mishchenko_cpu.dll ifdda_mishchenko_openmp.dll ifdda_mishchenko_cuda_kernel.dll cudart64_12.dll ifdda_mishchenko_sections_cpu.exe ifdda_mishchenko_sections_openmp.exe ifdda_mishchenko_sections_cuda.exe) do (
    cmake -E copy_if_different "%output_dir%\%%F" "%build_dir%"
    if errorlevel 1 exit /b 1
)
for %%F in (ifdda_mishchenko.lib ifdda_mishchenko_cpu.lib ifdda_mishchenko_openmp.lib) do (
    cmake -E copy_if_different "%output_dir%\%%F" "%build_dir%"
    if errorlevel 1 exit /b 1
)
python "%source_dir%\..\test_backends.py" "%build_dir%"
if errorlevel 1 exit /b %errorlevel%
python "%source_dir%\..\test_backends.py" "%build_dir%" --qp
exit /b %errorlevel%
