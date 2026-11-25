# Scattnlay multilayer Mie reference for IF-DDA

Scattnlay computes the extinction, scattering, and absorption cross sections of spheres made of concentric, isotropic, nonmagnetic layers in vacuum. The shared library, standalone command-line program, and IF-DDA reference all use the same C++ solver. Scattnlay runs on the CPU in double precision, including when called from a single-precision IF-DDA executable.

The solver takes the interface radii and complex relative permittivities in order from the core outward. It uses the size parameter x = 2πr/λ at each interface and the relative refractive index m = √εᵣ. The cross sections in m² are the efficiencies Qext, Qsca, and Qabs multiplied by πR², where R is the outer radius of the sphere.

## Build the shared library and standalone executable

Run the commands from `E:\msna6\test_chat_on_steroid\if-dda-master\CUDA\Scattnlay`. Under WSL, the same directory is `/mnt/e/msna6/test_chat_on_steroid/if-dda-master/CUDA/Scattnlay`. The build requires CMake 3.18 or later and a C++20 compiler. The upstream source files are included in `upstream`.

| CMake project | Release output directory | Shared library | Standalone executable |
| --- | --- | --- | --- |
| `WSL/CMakeLists.txt` | `build/wsl/scattnlay` | `libifdda_scattnlay.so` | `ifdda_scattnlay_sections` |
| `MSVC/CMakeLists.txt` | `build/msvc/scattnlay/Release` | `ifdda_scattnlay.dll` | `ifdda_scattnlay_sections.exe` |
| `MINGW/CMakeLists.txt` | `build/mingw/scattnlay` | `ifdda_scattnlay.dll` | `ifdda_scattnlay_sections.exe` |

### Windows build scripts

From the Scattnlay directory in PowerShell, choose the script for your toolchain.

```powershell
.\MSVC\build_all.bat
.\MINGW\build_all.bat
```

Each script builds the Release DLL, standalone executable, and import library, then places the three files in `MSVC/build` or `MINGW/build`. MSVC uses Visual Studio 2019 x64. MinGW uses Ninja and defaults to `F:\mingw64_11.2\bin`. Set `IFDDA_SCATTNLAY_MINGW_BIN` to use a different MinGW compiler directory.

### WSL

```bash
bash WSL/build_all.sh
```

The script configures a Release build, builds both targets, and prints the binary paths. Set `IFDDA_SCATTNLAY_BUILD_DIR` to choose a different build directory. Additional script arguments are forwarded to the CMake configure command. The equivalent commands for the default build directory are shown below.

```bash
cmake -S WSL -B build/wsl -DCMAKE_BUILD_TYPE=Release
cmake --build build/wsl --target ifdda_scattnlay_shared ifdda_scattnlay_sections
```

### Windows with MSVC 2019 x64

Run these commands in PowerShell with Visual Studio 2019 and its C++ toolchain installed.

```powershell
cmake -S MSVC -B build/msvc -G 'Visual Studio 16 2019' -A x64
cmake --build build/msvc --config Release --target ifdda_scattnlay_shared ifdda_scattnlay_sections
```

The MSVC import library `ifdda_scattnlay.lib` is in the same directory as the DLL.

### Windows with MinGW-w64 x64

Run these commands in PowerShell with Ninja on `PATH`. The compiler path below refers to the installation on this PC.

```powershell
$env:Path = 'F:\mingw64_11.2\bin;' + $env:Path
cmake -S MINGW -B build/mingw -G Ninja -DCMAKE_CXX_COMPILER=F:/mingw64_11.2/bin/g++.exe -DCMAKE_BUILD_TYPE=Release
cmake --build build/mingw --target ifdda_scattnlay_shared ifdda_scattnlay_sections
```

The MinGW import library `libifdda_scattnlay.dll.a` is in the same directory as the DLL. The MinGW runtime is linked statically into the DLL and the executable.

In CLion, open `WSL`, `MSVC`, or `MINGW` and select the matching toolchain. To build one artifact, specify only the corresponding target after `--target`. The top-level `CMakeLists.txt` can also be used for a standalone build.

```bash
cmake -S . -B build/standalone -DCMAKE_BUILD_TYPE=Release
cmake --build build/standalone --config Release
```

With a multi-configuration generator on Windows, the executable from this top-level build is in `build/standalone/Release`.

## Run the standalone executable

```text
ifdda_scattnlay_sections [wavelength_nm radius1_nm eps1_re eps1_im [radius2_nm eps2_re eps2_im ...]]
```

The first argument is the vacuum wavelength **in nanometres**. For each layer, give its outer radius **in nanometres** followed by the real and imaginary parts of its relative permittivity. List the layers from the core outward, with strictly increasing radii. The program infers the number of layers from the number of triplets. For a refractive index n + ik, the permittivity arguments are n² − k² and 2nk.

This lossless example uses radii of 2.5 and 5 nm and relative permittivities of 2.25 and 1.25. The wavelength gives an outer size parameter of 1.

Running without arguments uses this lossless validation case. Use `--help` or `-h` to display the syntax and default values. If arguments are supplied, the complete syntax is required.

```bash
./build/wsl/scattnlay/ifdda_scattnlay_sections
```

```bash
./build/wsl/scattnlay/ifdda_scattnlay_sections 31.41592653589793 2.5 2.25 0 5 1.25 0
```

The Windows executables take the same arguments. For example, run the MSVC binary in PowerShell.

```powershell
.\build\msvc\scattnlay\Release\ifdda_scattnlay_sections.exe 31.41592653589793 2.5 2.25 0 5 1.25 0
```

The following WSL example uses three absorbing layers.

```bash
./build/wsl/scattnlay/ifdda_scattnlay_sections 31.41592653589793 1.5 1.9596 0.056 3 3.2391 0.108 5 1.6899 0.026
```

The executable prints `Cext`, `Csca`, and `Cabs` on separate lines in m². The exit code is `0` on success, `1` for an invalid value or calculation failure, and `2` for an incorrect number of arguments.

## Call the shared library from C

The DLL and WSL shared library export two versioned functions using the C calling convention. Include `scattnlay_c.h`. Each function computes all three cross sections in one call. The C++ API in `concentric_mie.hpp` also takes lengths in metres.

| Function | Material input for each layer |
| --- | --- |
| `ifdda_scattnlay_sections_eps_v1` | Real and imaginary parts of the complex relative permittivity εᵣ |
| `ifdda_scattnlay_sections_index_v1` | Real part n and imaginary part k of the relative refractive index n + ik |

```c
int ifdda_scattnlay_sections_eps_v1(
    int layer_count, double wavelength_m, const double* radii_m,
    const double* eps_real, const double* eps_imag, double sections_m2[3],
    char* error_message, size_t error_capacity);

int ifdda_scattnlay_sections_index_v1(
    int layer_count, double wavelength_m, const double* radii_m,
    const double* index_real, const double* index_imag, double sections_m2[3],
    char* error_message, size_t error_capacity);
```

- `layer_count` is the number of layers. Each of the three input arrays must contain exactly that many elements, ordered from the core outward.
- `wavelength_m` is the vacuum wavelength in metres. `radii_m` contains the **outer radii** of the layers in metres. Each radius must be positive and strictly greater than the preceding one.
- Use passive materials with Im(εᵣ) ≥ 0, or with n ≥ 0 and k ≥ 0. The complex permittivity or refractive index must be finite and nonzero.
- `sections_m2` is a caller-allocated array of three doubles. It receives `Cext`, `Csca`, and `Cabs`, in that order and in m². On failure, the function zeros a supplied output array.
- `error_message` is optional. Pass `NULL, 0` to omit it. A supplied buffer with positive `error_capacity` in bytes receives a NUL-terminated message, which may be truncated. Use disjoint input and output buffers.

| Return code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Invalid argument or parameter outside the supported range |
| `2` | Reference calculation failed |

Save the following two-layer example as `example.c`.

```c
#include "scattnlay_c.h"
#include <stdio.h>

int main(void) {
    const double radii[] = {2.5e-9, 5e-9};
    const double eps_real[] = {2.25, 1.25};
    const double eps_imag[] = {0.0, 0.0};
    double sections[3];
    char error[160];

    int code = ifdda_scattnlay_sections_eps_v1(2, 31.41592653589793e-9, radii,
        eps_real, eps_imag, sections, error, sizeof error);
    if (code != IFDDA_SCATTNLAY_OK) {
        fprintf(stderr, "%s\n", error);
        return code;
    }
    printf("Cext=%.10e Csca=%.10e Cabs=%.10e m2\n", sections[0], sections[1], sections[2]);
    return 0;
}
```

Build and run the C example against the WSL library.

```bash
cc -I. example.c -Lbuild/wsl/scattnlay -lifdda_scattnlay -o example
LD_LIBRARY_PATH="$PWD/build/wsl/scattnlay" ./example
```

On Windows, link against the import library for the selected toolchain. Place the DLL next to the application's executable or make its directory available to the Windows DLL loader.

## Use the reference in IF-DDA

For `concentricsphere` under linearly or circularly polarized plane-wave illumination in vacuum, the C++ and Fortran IF-DDA cores print the Scattnlay cross sections after the DDA results. The reference uses the interface radii and complex permittivities from the initialized DDA geometry. IF-DDA prints a relative error for each positive reference value and reports any failure of the reference solver.

After rebuilding IF-DDA, run this example from `CUDA/WSL`.

```bash
./cmake-build-debug/test_main0_cpp_cdmlib_cpp_cuda_double \
    --geometry concentricsphere --illumination pwavelinear \
    --wavelength 31.41592653589793 --grid-size 32 --cross-section 1 --solver gpbicg
```

This uses the lossless two-layer case described above.

## Validation

The six WSL executables, the two MSVC executables, and the MinGW drivers with C++ and Fortran cores were checked on an NVIDIA TITAN V in single and double precision. For the lossless two-layer example, the double-precision results are as follows.

| Cross section | Scattnlay (m²) | IF-DDA (m²) | Relative DDA error |
| --- | ---: | ---: | ---: |
| Extinction | 1.981837342664075e-18 | 2.01618108021270e-18 | +1.733% |
| Scattering | 1.981837342664075e-18 | 2.01618108021270e-18 | +1.733% |
| Absorption | 0 | 0 | — |

The standalone program reproduced the upstream Scattnlay efficiencies for the two-layer lossless case and the three-layer absorbing case. With the same outer radius, splitting a homogeneous material into two identical layers gave the same cross sections as a single layer. For x = 1 and m = 1.5 + 0.02i, extinction and scattering agreed with the homogeneous Mie reference in IF-DDA. The absorbing two-layer DDA run with that same material in both layers and a 5 nm outer radius reported a Scattnlay absorption cross section of 4.50355445310864e-18 m².

Both exported functions and the standalone executable gave the same cross sections across all three Release x64 builds. For the three-layer absorbing example, extinction, scattering, and absorption were 1.4766494845e-17, 1.1534296725e-17, and 3.2321981198e-18 m², respectively. The tests also checked invalid radius ordering and NUL termination of truncated error messages. Test logs are in `build/validation`.

## Source and license

The far-field Scattnlay headers come from [ovidiopr/scattnlay](https://github.com/ovidiopr/scattnlay) at revision `ab27f6dbe063190f42d140cfc1f24a0c7fa53b1c`. They are stored in `upstream` with the GPL v3 or later license. The method is described by O. Peña and U. Pal in *Computer Physics Communications* 180, 2348–2354 (2009). [10.1016/j.cpc.2009.07.010](https://doi.org/10.1016/j.cpc.2009.07.010)
