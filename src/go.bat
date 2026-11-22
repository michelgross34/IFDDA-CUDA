@echo off
setlocal
pushd "%~dp0"

set "OUTPUT_DIR=%~1"
if not defined OUTPUT_DIR set "OUTPUT_DIR=%~dp0"
set "PRECISION=%~2"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if errorlevel 1 goto build_failed

if /i "%PRECISION%"=="float" goto build_float
nvcc --shared -m64 -cudart shared ^
    -DPRODUITMATVECTFFT_SINGLE_PRECISION=0 ^
    -o "%OUTPUT_DIR%\kernel_double.dll" kernel.cu ^
    -lcufft ^
    -Xlinker "/IMPLIB:%OUTPUT_DIR%\kernel_double.lib"
if errorlevel 1 goto build_failed
if not exist "%OUTPUT_DIR%\kernel_double.dll" goto build_failed
if not exist "%OUTPUT_DIR%\kernel_double.lib" goto build_failed
if /i "%PRECISION%"=="double" goto build_succeeded

if /i not "%PRECISION%"=="" if /i not "%PRECISION%"=="all" goto build_failed
:build_float
nvcc --shared -m64 -cudart shared ^
    -DPRODUITMATVECTFFT_SINGLE_PRECISION=1 ^
    -DUSE_SIMP ^
    -o "%OUTPUT_DIR%\kernel_float.dll" kernel.cu ^
    -lcufft ^
    -Xlinker "/IMPLIB:%OUTPUT_DIR%\kernel_float.lib"
if errorlevel 1 goto build_failed
if not exist "%OUTPUT_DIR%\kernel_float.dll" goto build_failed
if not exist "%OUTPUT_DIR%\kernel_float.lib" goto build_failed

:build_succeeded
if /i "%PRECISION%"=="double" (
    echo Created double CUDA wrapper DLL and import library in "%OUTPUT_DIR%"
    popd
    exit /b 0
)
echo Created double and float CUDA wrapper DLLs and import libraries in "%OUTPUT_DIR%"
popd
exit /b 0

:build_failed
echo Failed to create both CUDA wrapper DLLs/import libraries
popd
exit /b 1
