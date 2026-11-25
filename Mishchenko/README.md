# Mishchenko T-matrix for IF-DDA

The NASA GISS fixed-orientation solver computes extinction, scattering, and absorption in double precision for homogeneous, isotropic, nonmagnetic particles in vacuum under plane-wave illumination. It supports spheroids (ellipsoids of revolution, including the sphere limit) and finite circular cylinders. Triaxial ellipsoids require a different solver, such as the DDA in IF-DDA.

## Build the shared library and standalone executable

Obtain the fixed-orientation double-precision sources from the [NASA GISS T-matrix page](https://www.giss.nasa.gov/staff/mmishchenko/tmatrix/) and place `ampld.lp.f`, `ampld.par.f`, and `lpd.f` in `Mishchenko/NASA/` before building. Follow the conditions in the NASA source headers. A fresh clone contains the IF-DDA wrapper and build scripts. The standalone build requires the three local NASA files. The IF-DDA WSL build defaults to T-matrix support when these files are present.

Run these commands from `E:\msna6\test_chat_on_steroid\if-dda-master\CUDA\Mishchenko`, or `/mnt/e/msna6/test_chat_on_steroid/if-dda-master/CUDA/Mishchenko` under WSL. Requirements are CMake 3.20 or later, Python 3, GNU Fortran, and a C++17 compiler. The build runs `prepare_core.py` to adapt the sources in `NASA` and preserve double precision in the solver and its amplitude tables.

| Build script | Release directory | Shared library | Executable |
| --- | --- | --- | --- |
| `WSL/build_all.sh` or `WSL/build_all.bat` | `WSL/build` | `libifdda_mishchenko.so` | `ifdda_mishchenko_sections` |
| `MSVC/build_all.bat` | `MSVC/build` | `ifdda_mishchenko.dll` | `ifdda_mishchenko_sections.exe` |
| `MINGW/build_all.bat` | `MINGW/build` | `ifdda_mishchenko.dll` | `ifdda_mishchenko_sections.exe` |

Each script configures its local `CMakeLists.txt`, builds both Release targets, and places the two binaries side by side at the top of that environment's `build` directory. The Windows scripts also place their import libraries there.

### WSL

```bash
bash WSL/build_all.sh
```

Set `IFDDA_MISHCHENKO_BUILD_DIR` to choose a different build directory. To start this build from Windows PowerShell, run the WSL batch launcher.

```powershell
.\WSL\build_all.bat
```

### Windows with MSVC 2019 x64

From PowerShell, run the MSVC batch script. Visual Studio builds the C++ executable and the import library `ifdda_mishchenko.lib`. GNU Fortran for MinGW-w64 builds the x64 DLL. Set `IFDDA_MISHCHENKO_GFORTRAN` to select a compiler other than `F:/mingw64_11.2/bin/gfortran.exe`.

```powershell
.\MSVC\build_all.bat
```

### Windows with MinGW-w64 x64

Run the MinGW batch script from PowerShell with Ninja on `PATH`. Set `IFDDA_MISHCHENKO_MINGW_BIN` if the compiler binaries are installed outside `F:\mingw64_11.2\bin`. The MinGW import library is `libifdda_mishchenko.dll.a`.

```powershell
.\MINGW\build_all.bat
```

In CLion, open the `WSL`, `MSVC`, or `MINGW` subdirectory and select its matching toolchain. For a single artifact, pass only its target to `--target`. Keep the Windows DLL and executable together. The Windows DLLs link the GNU Fortran runtime statically.

## Calculate cross sections with the standalone executable

Run the executable without arguments to calculate the reference prolate spheroid. This is equivalent to `spheroid 630 140 280 1.5 0.02 27` with Eθ polarization. In CLion, leave **Program arguments** empty to run this case.

```text
ifdda_mishchenko_sections
ifdda_mishchenko_sections spheroid wavelength_nm transverse_nm axial_nm mr mi theta_deg
ifdda_mishchenko_sections ellipsoid wavelength_nm a_nm b_nm c_nm mr mi theta_deg
ifdda_mishchenko_sections cylinder wavelength_nm radius_nm height_nm mr mi theta_deg
```

Use positive lengths in **nanometres**. The relative refractive index is `mr + i mi`, with `mr > 0` and `mi >= 0`. The incidence angle `theta_deg` is measured from the symmetry axis to the incoming wave vector, from 0 to 180 degrees.

Append `Ep_re Ep_im Es_re Es_im` to specify the incident polarization. `Ep` is the Eθ component in the incidence plane and `Es` is the Eφ component perpendicular to it. The solver normalizes the two components. The default is `1 0 0 0` for Eθ. Use `0 0 1 0` for Eφ and `1 0 0 1` for equal components in quadrature. Use `--help` to display the command syntax.

### Ellipsoid of revolution

Enter the three **semiaxes** with `ellipsoid`. At least two must be equal. Their common value is the transverse semiaxis, and the third is the axial semiaxis. With `spheroid`, supply the transverse and axial semiaxes directly. The axial semiaxis is larger for a prolate spheroid and smaller for an oblate spheroid.

These commands describe the same prolate spheroid.

```bash
./WSL/build/ifdda_mishchenko_sections ellipsoid 630 140 140 280 1.5 0.02 27
./WSL/build/ifdda_mishchenko_sections spheroid 630 140 280 1.5 0.02 27
```

### Finite circular cylinder

Give the base **radius** and the **full height**. This example has radius 180 nm, height 360 nm, and Eφ polarization.

```bash
./WSL/build/ifdda_mishchenko_sections cylinder 630 180 360 1.5 0.02 27 0 0 1 0
```

The Windows executables take the same arguments. For example, run the MSVC build in PowerShell.

```powershell
.\MSVC\build\ifdda_mishchenko_sections.exe cylinder 630 180 360 1.5 0.02 27 0 0 1 0
```

The program prints `Cext`, `Csca`, and `Cabs` in m², followed by the multipole order, number of surface quadrature points, and relative variation at the final refinement. It returns `0` for success, `1` for invalid or unsupported input, and `2` when the solver cannot return a converged reference.

## Call the DLL or shared library from C

Include `mishchenko_c.h`. The C function `ifdda_mishchenko_v1` computes all three cross sections.

```c
int ifdda_mishchenko_v1(
    int shape, double volume_radius, double ratio, double longest,
    double mr, double mi, double theta_deg,
    double ep_real, double ep_imag, double es_real, double es_imag,
    double sections[3], int *order, int *quadrature, double *variation);
```

Let λ be the vacuum wavelength, a and c the transverse and axial spheroid semiaxes, r the cylinder radius, and h its full height.

| Shape code | `volume_radius` | `ratio` | `longest` |
| --- | --- | --- | --- |
| `-1` spheroid | ∛(a²c)/λ | a/c | max(a,c)/λ |
| `-2` cylinder | ∛(3r²h/4)/λ | 2r/h | √(r² + h²/4)/λ |

The index, angle, and polarization arguments have the same meaning as in the command line. The caller provides storage for all outputs. On success, `sections` contains `Cext`, `Csca`, and `Cabs` in units of λ². Multiply the values by the square of the wavelength **in metres** to obtain m². `order` is the final multipole order, `quadrature` is the total number of surface quadrature points, and `variation` is a relative change expressed as a fraction. Use the cross sections only when the return code is `0`. Serialize calls to the same loaded library because the solver uses Fortran `COMMON` blocks.

Save this prolate spheroid example as `example.c`.

```c
#include "mishchenko_c.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    const double a = 140.0 / 630.0;
    const double c = 280.0 / 630.0;
    const double area = (630e-9) * (630e-9);
    double sections[3], variation;
    int order, quadrature;
    int status = ifdda_mishchenko_v1(IFDDA_MISHCHENKO_SPHEROID,
        cbrt(a*a*c), a/c, fmax(a,c), 1.5, 0.02, 27.0, 1.0, 0.0, 0.0, 0.0,
        sections, &order, &quadrature, &variation);
    if (status != IFDDA_MISHCHENKO_OK) {
        fprintf(stderr, "Mishchenko status %d\n", status);
        return status;
    }
    printf("Cext=%.10e Csca=%.10e Cabs=%.10e m2\n",
        sections[0]*area, sections[1]*area, sections[2]*area);
    return 0;
}
```

```bash
cc -I. example.c -LWSL/build -lifdda_mishchenko -lm -o example
LD_LIBRARY_PATH="$PWD/WSL/build" ./example
```

On Windows, link with the import library for your toolchain and place the DLL alongside the calling executable or make its directory available to the Windows loader.

## Use the reference from IF-DDA

Under WSL, configure IF-DDA with `-DIFDDA_WSL_TMATRIX=ON`, or run `IFDDA_WSL_TMATRIX=ON bash build_all.sh` from `CUDA/WSL`. Under Windows, deploy the DLL beside the IF-DDA executable. The MSVC IF-DDA build can also copy a DLL present in `CUDA/Mishchenko` at configuration time. The existing `build_windows.bat` and `build_windows.ps1` support the deployment workflow. For example, run `build_windows.bat -DeployTo ..\MSVC\cmake-build-debug` from `CUDA\Mishchenko` in Command Prompt. The PowerShell script accepts `-Fortran` to select GNU Fortran. `test_reference.py` tests the DLL in the source directory.

The C++ and Fortran IF-DDA cores report the T-matrix sections after DDA for eligible spheroids and cylinders in vacuum under linear or circular plane waves. The effective incident field and object rotation determine the polarization and incidence in the particle frame. The automatic `ellipsoid` reference requires exactly one pair of equal semiaxes. The `cylinder` driver takes the radius and full height in nm and requires a third entry as a placeholder. Its axis is the local z axis. The Fortran driver also reports its separate Mie reference for `sphere`.

Run these examples from `CUDA/WSL` after building IF-DDA.

```bash
./cmake-build-debug/test_main0_cpp_cdmlib_cpp_cuda_double \
    --geometry ellipsoid --geometry-dimensions 140,140,280 \
    --geometry-angles 0,0,0 --illumination pwavelinear \
    --theta 27 --phi 49 --pp 1 --epsilon 2.24,0.3 --wavelength 630 \
    --grid-size 32 --cross-section 1 --solver gpbicg

./cmake-build-debug/test_main0_cpp_cdmlib_cpp_cuda_double \
    --geometry cylinder --geometry-dimensions 180,360,1 \
    --geometry-angles 23,39,17 --illumination pwavelinear \
    --theta 27 --phi 49 --pp 1 --epsilon 2.24,0.3 --wavelength 630 \
    --grid-size 32 --cross-section 1 --solver gpbicg
```

The IF-DDA commands use the relative permittivity εᵣ = m², laboratory incidence, and geometry rotation. The standalone commands use the refractive index and incidence defined in the particle frame.

## Numerical checks and limits

The solver obtains Cext from the forward amplitude through the optical theorem and Csca by integrating the scattered intensity over 4π. It calculates Cabs as Cext − Csca and checks passivity. It compares Cext and Csca at two multipole orders and two surface quadratures with a relative tolerance of 0.1%. Cylinders use a larger order increment and require two consecutive stable refinements. Absorption can have a larger relative uncertainty near the lossless limit. IF-DDA reports unavailable references and computes relative DDA errors when the reference is positive.

The IF-DDA examples above were checked on a TITAN V. Their reference values are shown below.

| IF-DDA case | Cext (m²) | Csca (m²) | Cabs (m²) | Final variation |
| --- | ---: | ---: | ---: | ---: |
| Tilted cylinder, radius 180 nm, height 360 nm | 2.659051266e-13 | 1.776699783e-13 | 8.823514830e-14 | 0.0162% |
| Spheroid, semiaxes 140, 140, and 280 nm | 1.640699264e-13 | 1.000376720e-13 | 6.403225445e-14 | 0.000001% |

For the tilted cylinder, WSL double-precision DDA differed from T-matrix by −1.1046% in extinction, −0.9734% in scattering, and −1.3688% in absorption. The reference reached order 18 with 100 surface points. Checks covered six WSL and two MSVC Debug executables, MinGW C++ and Fortran builds, and two MSVC Release builds. The WSL Debug and MSVC Debug cylinder references agreed across drivers and precisions within 0.00001%. Spheroid checks covered all three symmetry axes. Logs are in `build/cylinder-ifdda-20261125` and `build/spheroid-only`.

The direct-library test agreed with the independent Mie result for x = 5 and m = 5 within 0.00002%. For the aspect-ratio-2 spheroid at x = 5, the variation against earlier references was under 0.05% for m = 2.5 + 0.1i, 4 + 0.1i, and 5 + 0.1i. The m = 6 + 0.1i case reached the refinement limit. For the cylinder at radius 0.45λ, height 0.9λ, m = 2.5 + 0.1i, θ = 27°, and Eφ polarization, the library agreed within 0.05% with a separate extended-precision NASA Fortran run at order 38 with 456 surface points. See `build/cylinder_qp`.

### Standalone Release validation

The WSL, MSVC, and MinGW builds were tested through both the shared-library API and the standalone executable. The three builds agreed within 1e-12 relative for the two command-line examples below. All three also reproduced the independent Mie sphere result at x = 5 and m = 5 within 0.00002%.

| Standalone example | Cext (m²) | Csca (m²) | Cabs (m²) |
| --- | ---: | ---: | ---: |
| Prolate spheroid, Eθ | 1.347709480e-13 | 1.192705349e-13 | 1.550041306e-14 |
| Circular cylinder, Eφ | 2.597371208e-13 | 2.356175667e-13 | 2.411955408e-14 |

The tests also checked the rejection of triaxial axes and a zero incident polarization. The logs are in `build/standalone-validation`. GNU Make reported clock skew on the Windows-mounted filesystem under WSL. The build and numerical tests completed successfully.

## Source

The NASA GISS code is available for nonprofit scientific research under the conditions in its source header. See M. I. Mishchenko, *Applied Optics* **39**, 1026–1031 (2000). [DOI 10.1364/AO.39.001026](https://doi.org/10.1364/AO.39.001026)
