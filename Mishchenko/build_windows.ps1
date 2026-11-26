param([string]$Fortran = 'gfortran', [string]$Python = 'python', [string[]]$DeployTo = @())
$ErrorActionPreference = 'Stop'
$runtimeDirectories = $DeployTo | ForEach-Object { (Resolve-Path -LiteralPath $_).Path }
$fc = (Get-Command $Fortran -ErrorAction SilentlyContinue).Source
if (-not $fc) {
    if ($Fortran -ne 'gfortran') { throw "GNU Fortran not found: $Fortran" }
    $fc = 'F:/mingw64_11.2/bin/gfortran.exe'
}
$cxx = Join-Path (Split-Path -Parent $fc) 'g++.exe'
$py = (Get-Command $Python -ErrorAction Stop).Source
$build = Join-Path $PSScriptRoot 'build/windows-fast'
cmake -S "$PSScriptRoot/MINGW" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=$cxx" "-DIFDDA_MISHCHENKO_GFORTRAN=$fc" "-DPython3_EXECUTABLE=$py" -DIFDDA_MISHCHENKO_BUILD_STANDALONE=OFF -DIFDDA_MISHCHENKO_ENABLE_CUDA=ON
if ($LASTEXITCODE -ne 0) { throw 'Mishchenko configuration failed' }
cmake --build "$build" --target ifdda_mishchenko_shared --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'Mishchenko build failed' }
foreach ($directory in @($PSScriptRoot)+@($runtimeDirectories)) {
    foreach ($name in 'ifdda_mishchenko.dll','ifdda_mishchenko_cuda_kernel.dll','cudart64_12.dll') {
        Copy-Item -LiteralPath (Join-Path "$build/mishchenko/Release" $name) -Destination $directory -Force
    }
}
