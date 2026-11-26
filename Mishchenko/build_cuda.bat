@echo off
setlocal
rem Compile the CUDA DLL separately with nvcc and MSVC, never MinGW.
rem Optional argument: output directory. IFDDA_MISHCHENKO_CUDA_ROOT and IFDDA_MISHCHENKO_VCVARS64 override defaults.
if "%~1"=="" (set "OUT=%~dp0build") else (set "OUT=%~f1")
pushd "%~dp0" || exit /b 1
set "SRC=%~dp0."
rem MSBuild can be a 32-bit process; use the native 64-bit Program Files.
set "PF64=%ProgramW6432%"
if not defined PF64 set "PF64=%ProgramFiles%"
if not defined IFDDA_MISHCHENKO_CUDA_ROOT set "IFDDA_MISHCHENKO_CUDA_ROOT=%PF64%\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not defined IFDDA_MISHCHENKO_VCVARS64 set "IFDDA_MISHCHENKO_VCVARS64=%PF64%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%IFDDA_MISHCHENKO_VCVARS64%" set "IFDDA_MISHCHENKO_VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%IFDDA_MISHCHENKO_VCVARS64%" goto :missing
if not exist "%IFDDA_MISHCHENKO_CUDA_ROOT%\bin\nvcc.exe" goto :missing
if not exist "%IFDDA_MISHCHENKO_CUDA_ROOT%\bin\cudart64_12.dll" goto :missing
if /I "%VSCMD_ARG_TGT_ARCH%"=="x64" (
  where cl.exe >nul 2>nul
  if not errorlevel 1 goto :host_ready
)
call "%IFDDA_MISHCHENKO_VCVARS64%"
if errorlevel 1 goto :failed
:host_ready
if not exist "%OUT%" mkdir "%OUT%"
if errorlevel 1 goto :failed
echo === CUDA DLL: %OUT%\ifdda_mishchenko_cuda_kernel.dll ===
"%IFDDA_MISHCHENKO_CUDA_ROOT%\bin\nvcc.exe" -shared -O3 -std=c++17 -arch=sm_70 --fmad=false -Xcompiler=/EHsc,/MD -cudart shared "%SRC%\mish_cuda.cu" -o "%OUT%\ifdda_mishchenko_cuda_kernel.dll"
if errorlevel 1 goto :failed
rem Deploy the runtime DLLs from the SAME CUDA toolkit. The driver is installed separately.
for %%F in (cudart64_12.dll) do (
  if exist "%IFDDA_MISHCHENKO_CUDA_ROOT%\bin\%%F" (
    copy /Y "%IFDDA_MISHCHENKO_CUDA_ROOT%\bin\%%F" "%OUT%\%%F" >nul
    if errorlevel 1 goto :failed
  )
)
echo CUDA DLL built successfully.
popd
exit /b 0
:missing
echo ERROR: nvcc or MSVC vcvars64.bat not found. Set IFDDA_MISHCHENKO_CUDA_ROOT and IFDDA_MISHCHENKO_VCVARS64. 1>&2
:failed
popd
exit /b 1
