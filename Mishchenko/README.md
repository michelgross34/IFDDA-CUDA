# Mishchenko T-matrix for IF-DDA

The NASA GISS fixed-orientation solver computes extinction, scattering, and absorption using double- or quadruple-precision T-matrix construction for homogeneous, isotropic, nonmagnetic particles in vacuum under plane-wave illumination. The standalone executable and shared library support spheroids (ellipsoids of revolution), finite circular cylinders, and capsules with hemispherical caps. Triaxial ellipsoids require a different solver, such as the DDA in IF-DDA.

## Execution backends

| Standalone executable (add .exe on Windows) | Backend ID | Algorithm |
| --- | ---: | --- |
| ifdda_mishchenko_sections_cpu | 0 | Serial DP/QP Fortran |
| ifdda_mishchenko_sections_openmp | 1 | OpenMP DP/QP Fortran |
| ifdda_mishchenko_sections_cuda | 2 | OpenMP T-matrix construction and CUDA FP64 angular integration |

The three versions use the same NASA fixed-orientation DP and QP solvers, geometries, convergence criteria and C interface. OpenMP distributes independent surface-integral terms in TMATR0 and TMATR, preserving the summation order within each term. Matrix factorization remains on the CPU. The CUDA backend contracts incident multipoles once per refinement on the CPU and evaluates the outgoing amplitudes in parallel on the GPU; the cost per direction is reduced from O(N^3) to O(N^2). The forward amplitude and the full-sphere scattering quadrature use double precision, with the original quadrature weights and deterministic host summation. The implementation is adapted from [TMatrix-Holo](https://github.com/michelgross34/TMatrix-Holo). No FFT is required.

Set OMP_NUM_THREADS to control the OpenMP worker count. When it is not set, the library uses up to eight workers and restores the caller's OpenMP thread-count setting on return. CUDA initialization and transfers can dominate very small calculations.

## Build the libraries and standalone executables

Obtain the six original NASA fixed-orientation files (DP: ampld.lp.f, ampld.par.f, lpd.f; QP: amplq.lp.f, amplq.par.f, lpq.f) and place them in NASA/. Both source variants carry the update date 08/06/2005. They are external dependencies, not distributed in this repository. Requirements: CMake 3.20 or later, Python 3, GNU Fortran with OpenMP and IEEE binary128 support (libquadmath), C++17 and NVIDIA CUDA for the GPU variant. The build prepares private copies of the original sources; it never edits the NASA files.

| Build script | Release output directory |
| --- | --- |
| WSL/build_all.sh (or WSL/build_all.bat from Windows) | WSL/build |
| MSVC/build_all.bat | MSVC/build |
| MINGW/build_all.bat | MINGW/build |

Each script builds all three standalone executables and their shared libraries and runs the numerical backend tests in both DP and QP. Windows builds include the corresponding import libraries.

### WSL

~~~bash
bash WSL/build_all.sh
~~~

Set IFDDA_MISHCHENKO_BUILD_DIR to select the output directory. Set CMAKE_CUDA_ARCHITECTURES using the script's additional CMake arguments for another GPU architecture. The default is SM 70 (Volta), supported by CUDA 12.9. The shared libraries are libifdda_mishchenko_cpu.so, libifdda_mishchenko_openmp.so and libifdda_mishchenko.so (the accelerated library).

### Windows

~~~powershell
.\MSVC\build_all.bat
.\MINGW\build_all.bat
~~~

Run the script appropriate to the C++ toolchain. MSVC 2019 x64 or MinGW-w64 x64 builds the standalone C++ programs; GNU Fortran builds the Fortran libraries. The GPU kernel is compiled separately with nvcc and an x64 MSVC host compiler, using CUDA 12.9 by default. The GNU Fortran and OpenMP runtimes are statically linked into the reference DLL. Set IFDDA_MISHCHENKO_GFORTRAN, IFDDA_MISHCHENKO_MINGW_BIN, IFDDA_MISHCHENKO_CUDA_ROOT and IFDDA_MISHCHENKO_VCVARS64 as appropriate for the installed tools. Ninja is required by the MinGW script. The default Windows kernel architecture is SM 70.

The serial and OpenMP DLLs are ifdda_mishchenko_cpu.dll and ifdda_mishchenko_openmp.dll. The accelerated variant requires these three runtime files together: ifdda_mishchenko.dll, ifdda_mishchenko_cuda_kernel.dll and cudart64_12.dll. Import libraries use .lib for MSVC and .dll.a for MinGW.

### CMake and CLion

Open the WSL, MSVC or MINGW subdirectory with its matching toolchain. Build All creates the three executables. In an IDE build the binaries are in the mishchenko subdirectory (Windows: mishchenko/<configuration>); the build_all scripts additionally copy the Release binaries to their build directory. Select Release for numerical performance measurements.

IFDDA_MISHCHENKO_ENABLE_CUDA=OFF builds the serial and OpenMP standalone variants without the CUDA toolkit. IFDDA_MISHCHENKO_BUILD_STANDALONE=OFF builds only the accelerated library and is used by the parent IF-DDA projects.

## T-matrix construction precision

Each of the three executables and libraries supports:

| Mode | T-matrix construction |
| --- | --- |
| dp | IEEE binary64, no fallback |
| qp | IEEE binary128 surface integrals, Bessel functions, LU inversion and matrix product |
| auto (default) | DP first; repeat the refinement in QP if the DP solver returns nonconvergence (status 2) |

QP uses the original NASA extended-precision LAPACK implementation, with
modern GNU Fortran binary128 intrinsics and OpenMP surface integration.
The matrix product in TT is accumulated in binary128 before export.
Both solvers share the final FP64 T-matrix storage, and the amplitude and
cross-section integration (CPU or GPU) use FP64. Inputs and returned
sections use the existing double-precision ABI. QP does not mean
end-to-end 128-bit CUDA arithmetic. The QP refinement allows order up to
72 and up to 1000 surface quadrature points, subject to the existing
input-size guard, and retains the original cross-section tolerances.

Append the precision option after the geometry and optional polarization:

~~~bash
ifdda_mishchenko_sections_cuda spheroid 630 140 280 1.5 0.02 27 --precision qp
~~~

Each successful standalone calculation reports the selected precision.
The C ABI retains ifdda_mishchenko_v1 (now automatic) and adds
ifdda_mishchenko_v2, which accepts a precision selector (0 auto, 1 dp,
2 qp) and returns the construction precision actually used (1 or 2).
The complete signature is in mishchenko_c.h.

IF-DDA uses the OpenMP + CUDA v2 library with automatic precision by
default, for both single- and double-precision DDA drivers.
To force the reference precision, set IFDDA_TMATRIX_PRECISION to dp
or qp; unset it or set auto to restore the default. For example:

~~~bash
export IFDDA_TMATRIX_PRECISION=qp
~~~

~~~bat
set IFDDA_TMATRIX_PRECISION=qp
~~~

The IF-DDA reference report prints the selected construction precision.
A CUDA error is not retried as a precision problem. QP is not a
guarantee of convergence for every size, aspect ratio or index.

## Calculate cross sections with the standalone executable

All three executables accept identical arguments; the following examples use the CUDA variant. Run the executable without arguments to calculate the reference prolate spheroid. This is equivalent to `spheroid 630 140 280 1.5 0.02 27` with Eθ polarization. In CLion, leave **Program arguments** empty to run this case.

```text
ifdda_mishchenko_sections_cuda
ifdda_mishchenko_sections_cuda spheroid wavelength_nm transverse_nm axial_nm mr mi theta_deg
ifdda_mishchenko_sections_cuda ellipsoid wavelength_nm a_nm b_nm c_nm mr mi theta_deg
ifdda_mishchenko_sections_cuda cylinder wavelength_nm radius_nm height_nm mr mi theta_deg
ifdda_mishchenko_sections_cuda capsule wavelength_nm radius_nm straight_length_nm mr mi theta_deg
```

Use **nanometres** for all lengths. The capsule straight length may be zero, which gives a sphere. All other lengths must be positive. The relative refractive index is `mr + i mi`, with `mr > 0` and `mi >= 0`. The incidence angle `theta_deg` is measured from the symmetry axis to the incoming wave vector, from 0 to 180 degrees.

Append `Ep_re Ep_im Es_re Es_im` to specify the incident polarization. `Ep` is the Eθ component in the incidence plane and `Es` is the Eφ component perpendicular to it. The solver normalizes the two components. The default is `1 0 0 0` for Eθ. Use `0 0 1 0` for Eφ and `1 0 0 1` for equal components in quadrature. Use `--help` to display the command syntax.

### Ellipsoid of revolution

Enter the three **semiaxes** with `ellipsoid`. At least two must be equal. Their common value is the transverse semiaxis, and the third is the axial semiaxis. With `spheroid`, supply the transverse and axial semiaxes directly. The axial semiaxis is larger for a prolate spheroid and smaller for an oblate spheroid.

These commands describe the same prolate spheroid.

```bash
./WSL/build/ifdda_mishchenko_sections_cuda ellipsoid 630 140 140 280 1.5 0.02 27
./WSL/build/ifdda_mishchenko_sections_cuda spheroid 630 140 280 1.5 0.02 27
```

### Finite circular cylinder

Give the base **radius** and the **full height**. This example has radius 180 nm, height 360 nm, and Eφ polarization.

```bash
./WSL/build/ifdda_mishchenko_sections_cuda cylinder 630 180 360 1.5 0.02 27 0 0 1 0
```

The Windows executables take the same arguments. For example, run the MSVC build in PowerShell.

```powershell
.\MSVC\build\ifdda_mishchenko_sections_cuda.exe cylinder 630 180 360 1.5 0.02 27 0 0 1 0
```

The program prints `Cext`, `Csca`, and `Cabs` in m², followed by the multipole order, number of surface quadrature points, relative variation at the final refinement, and backend ID. It returns `0` for success, `1` for invalid or unsupported input, and `2` when the solver cannot return a converged reference.

### Capsule with hemispherical caps

Supply the cap radius r and the straight cylindrical length ℓ between the cap centres. The total tip-to-tip length is ℓ + 2r. The example below has a diameter of 672 nm and a total length of 1 344 nm. The final four arguments select the laboratory Y polarization for the ADDA benchmark with an axis tilt of 27° and an azimuth of 48°. They are the real components sin(48°) and cos(48°) in the particle incidence basis.

~~~bash
./WSL/build/ifdda_mishchenko_sections_cuda capsule 630 336 672 3 0.1 27 0.7431448254773942 0 0.6691306063588582 0
~~~

Use the same arguments with `MINGW\build\ifdda_mishchenko_sections_cuda.exe` or `MSVC\build\ifdda_mishchenko_sections_cuda.exe`. With no polarization arguments, the executable uses Eθ. For pure Eφ, append `0 0 1 0`.

## Call the DLL or shared library from C

Include `mishchenko_c.h`. The C function `ifdda_mishchenko_v1` computes all three cross sections.

```c
int ifdda_mishchenko_v1(
    int shape, double volume_radius, double ratio, double longest,
    double mr, double mi, double theta_deg,
    double ep_real, double ep_imag, double es_real, double es_imag,
    double sections[3], int *order, int *quadrature, double *variation);
```

Let λ be the vacuum wavelength, a and c the transverse and axial spheroid semiaxes, r the radius of a cylinder or capsule, h the full cylinder height, and ℓ the straight capsule length.

| Shape code | `volume_radius` | `ratio` | `longest` |
| --- | --- | --- | --- |
| `-1` spheroid | ∛(a²c)/λ | a/c | max(a,c)/λ |
| `-2` cylinder | ∛(3r²h/4)/λ | 2r/h | √(r² + h²/4)/λ |
| `-4` capsule (`IFDDA_MISHCHENKO_CAPSULE`) | ∛(r³ + 3r²ℓ/4)/λ | ℓ/(2r) | (r + ℓ/2)/λ |

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

The parent CMake projects enable the accelerated reference automatically when the six external NASA sources are available. Under WSL use -DIFDDA_WSL_TMATRIX=ON (or IFDDA_WSL_TMATRIX=ON bash build_all.sh from CUDA/WSL); on Windows use -DIFDDA_WINDOWS_TMATRIX=ON. Explicitly enabling the option requires the NASA files. The parent projects build only the OpenMP + CUDA FP64 reference, not the standalone CPU executables. IF-DDA verifies backend ID 2 before accepting its results.

Under Windows, CUDA/build_all_cuda.bat builds the ordinary IF-DDA CUDA 11.8 wrappers and the reference DLLs when the NASA files are present. The parent CMake build deploys the three reference runtime DLLs alongside each IF-DDA executable. The reference GPU kernel uses CUDA 12.9 independently of the DDA CUDA 11.8 wrappers. For manual deployment, run build_windows.bat -DeployTo ..\\MSVC\\cmake-build-debug from CUDA/Mishchenko; the PowerShell script also accepts -Fortran to select GNU Fortran. Under WSL the reference is linked as libifdda_mishchenko.so using the same installed CUDA toolkit as IF-DDA.

To test the C API directly, run python test_reference.py <path-to-library>. To compare all standalone backends, run python test_backends.py <build-directory> (add --qp for quadruple precision, --large for the long capsule test). To check the parent executables, run python test_ifdda_integration.py <IF-DDA-build-directory> <log-directory> --count 6 for WSL/MinGW or --count 2 for MSVC. Add --precision qp to force and verify QP in the integration test.

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

The IF-DDA commands use the relative permittivity εᵣ = m², laboratory incidence, and geometry rotation. The standalone commands use the refractive index and incidence defined in the particle frame. Use the standalone executable or the shared-library API to calculate capsule references.

## Numerical checks and limits

The solver obtains Cext from the forward amplitude through the optical theorem and Csca by integrating the scattered intensity over 4π. It calculates Cabs as Cext − Csca and checks passivity. For spheroids and cylinders the relative refinement tolerance is 0.1% on extinction and scattering. Cylinders and capsules require two consecutive stable refinements. For capsules the surface quadrature is split at the junction of the flank and the cap, and the relative tolerance is 0.01% on extinction, scattering, and absorption for absorbing particles. IF-DDA reports unavailable references and computes relative DDA errors when the reference is positive.

The IF-DDA examples above were checked on a TITAN V. Their reference values are shown below.

| IF-DDA case | Cext (m²) | Csca (m²) | Cabs (m²) | Final variation |
| --- | ---: | ---: | ---: | ---: |
| Tilted cylinder, radius 180 nm, height 360 nm | 2.659051266e-13 | 1.776699783e-13 | 8.823514830e-14 | 0.0162% |
| Spheroid, semiaxes 140, 140, and 280 nm | 1.640699264e-13 | 1.000376720e-13 | 6.403225445e-14 | 0.000001% |

For the tilted cylinder, WSL double-precision DDA differed from T-matrix by −1.1046% in extinction, −0.9734% in scattering, and −1.3688% in absorption. The reference reached order 18 with 100 surface points. Checks covered six WSL and two MSVC Debug executables, MinGW C++ and Fortran builds, and two MSVC Release builds. The WSL Debug and MSVC Debug cylinder references agreed across drivers and precisions within 0.00001%. Spheroid checks covered all three symmetry axes. Logs are in `build/cylinder-ifdda-20261125` and `build/spheroid-only`.

The direct-library test agreed with the independent Mie result for x = 5 and m = 5 within 0.00002%. For the aspect-ratio-2 spheroid at x = 5, the variation against earlier references was under 0.05% for m = 2.5 + 0.1i, 4 + 0.1i, and 5 + 0.1i. The m = 6 + 0.1i case reached the refinement limit in DP and in the extended QP refinement. For the cylinder at radius 0.45λ, height 0.9λ, m = 2.5 + 0.1i, θ = 27°, and Eφ polarization, the library agreed within 0.05% with a separate extended-precision NASA Fortran run at order 38 with 456 surface points. See `build/cylinder_qp`.

### Standalone and IF-DDA validation

The three Release backends were built and tested under WSL, MinGW and MSVC (TITAN V). Eight cases per backend and precision cover a spheroid, circular cylinder, spherical capsule, general complex polarization, both poles, equator and transparent material. All three backends returned the same multipole order and surface quadrature in each case. The largest absolute difference in cross sections between backends was below 3e-28 m² in the eight standard DP cases and below 3e-28 m² in the corresponding QP cases. The spherical capsule was also checked against independent Mie values. The test rejects non-axisymmetric ellipsoids.

| Standalone example | Cext (m²) | Csca (m²) | Cabs (m²) | Order / quadrature |
| --- | ---: | ---: | ---: | ---: |
| Prolate spheroid, Eθ | 1.347709480e-13 | 1.192705349e-13 | 1.550041306e-14 | 13 / 82 |
| Circular cylinder, Eφ | 2.597371208e-13 | 2.356175667e-13 | 2.411955408e-14 | 18 / 100 |
| Spherical capsule | 8.837754602e-13 | 5.093785336e-13 | 3.743969266e-13 | 20 / 104 |

The large capsule test (radius 336 nm, straight length 672 nm, wavelength 630 nm, m=3+0.1i, 27 degrees, mixed linear polarization) also agreed across all three backends, reaching N=49 and 424 quadrature points. The serial values are 1.6596905470001064e-12, 9.87279495837564e-13 and 6.724110511625422e-13 m^2. The largest absolute backend difference was 5.05e-27 m^2. On WSL/TITAN V the observed wall times were 193.21 s (serial), 219.89 s (OpenMP) and 2.54 s (OpenMP+CUDA), with other compilation processes running concurrently. These are indicative elapsed times, not an isolated performance benchmark.

The accelerated direct-library test also reproduced the independent Mie result for x=5 and m=5, the three converged high-index spheroid cases and the extended-precision cylinder reference. For the m=6+0.1i spheroid, auto mode retries in QP but still returns refinement-limit status 2 (relative variation 0.00324, exceeding the 0.001 tolerance). It is not reported as a converged reference.

The parent builds were validated with the tilted cylinder and the spheroid described above, using grid size 16, in all six WSL and MinGW executable variants and both MSVC variants. Every run used backend ID 2 and completed the DDA calculation successfully. The maximum relative difference against the tabulated T-matrix references was 4.88e-8, including the single-precision IF-DDA drivers. The reference library uses DP/QP construction and FP64 angular integration.

The explicit QP direct-library tests under WSL gave a maximum relative error of 1.08e-7 against independent Mie for the sphere at x=5, m=5, and 8.19e-5 against the tabulated spheroid reference at x=5, m=5+0.1i. The spheroid converged at N=36 with 780 surface points. All 14 local IF-DDA executables and the six WSL publication-checkout executables passed both the automatic and forced-QP cylinder and spheroid integration tests.

## Source

The original `ampld.lp.f`, `ampld.par.f`, and `lpd.f` are in `NASA`. The code is available for nonprofit scientific research under the conditions in its source header. See the [NASA GISS T-matrix page](https://www.giss.nasa.gov/staff/mmishchenko/tmatrix/) and M. I. Mishchenko, *Applied Optics* **39**, 1026–1031 (2000). [DOI 10.1364/AO.39.001026](https://doi.org/10.1364/AO.39.001026)
