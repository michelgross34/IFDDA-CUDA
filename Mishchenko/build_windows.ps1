param([string]$Fortran = 'gfortran', [string]$Python = 'python', [string[]]$DeployTo = @())
$ErrorActionPreference = 'Stop'
$runtimeDirectories = $DeployTo | ForEach-Object { (Resolve-Path -LiteralPath $_).Path }
Push-Location $PSScriptRoot
try {
    & $Python prepare_core.py
    if ($LASTEXITCODE -ne 0) { throw 'NASA source preparation failed' }
    Push-Location build
    try {
        & $Fortran -O2 -std=legacy -fallow-argument-mismatch -I../NASA -c mish_core.f lpd.f ../mishchenko_reference.f90
        if ($LASTEXITCODE -ne 0) { throw 'Fortran compilation failed' }
        & $Fortran -shared -static -static-libgfortran -static-libgcc -o ../ifdda_mishchenko.dll mish_core.o lpd.o mishchenko_reference.o ../ifdda_mishchenko.def
        if ($LASTEXITCODE -ne 0) { throw 'DLL link failed' }
    } finally { Pop-Location }
    foreach ($directory in $runtimeDirectories) {
        Copy-Item -LiteralPath ifdda_mishchenko.dll -Destination $directory
    }
} finally { Pop-Location }
