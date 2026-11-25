# Multilayer Mie scattering with Scattnlay in IF-DDA

The [build and C interface documentation](SCATTNLAY_GUIDE.md) covers WSL, MSVC and MinGW, including the Windows batch launchers and `WSL/build_all.sh`.

The reference calculation handles concentric spheres with isotropic, nonmagnetic layers in vacuum under linearly or circularly polarized plane-wave illumination. The C++ and Fortran IF-DDA cores print `Cext`, `Csca` and `Cabs` after the DDA cross sections for the `concentricsphere` geometry. Relative errors are printed when the reference cross section is positive. Any failure of the reference solver is reported in the output.

The code retrieves the interface radii and complex permittivities after the DDA geometry has been initialized. For each layer, from the core outward, it computes the size parameter `x = 2πr/λ` and the refractive index `m = sqrt(εr)`. The Scattnlay solver runs on the CPU in double precision, even in a single-precision IF-DDA executable. The efficiencies `Qext`, `Qsca` and `Qabs` are multiplied by `π R_ext²` to obtain cross sections in m².

## Use in IF-DDA

WSL example, run from `CUDA/WSL` after rebuilding.

```bash
./cmake-build-debug/test_main0_cpp_cdmlib_cpp_cuda_double \
    --geometry concentricsphere --illumination pwavelinear \
    --wavelength 31.41592653589793 --grid-size 32 --cross-section 1 --solver gpbicg
```

The built-in case uses radii of 2.5 and 5 nm, with relative permittivities of 2.25 and 1.25. The chosen wavelength gives an outer size parameter of 1.

## Standalone calculation

Run from `CUDA/Scattnlay` with CMake and a C++20 compiler.

```bash
cmake -S . -B build
cmake --build build --config Release
./build/ifdda_scattnlay_sections 31.41592653589793 2.5 2.25 0 5 1.25 0
```

On Windows, the program has an `.exe` extension and is located in `build/Release` when using a multi-configuration generator. The first argument is the vacuum wavelength in nm. The following arguments are triplets `radius_nm Re(εr) Im(εr)`, with strictly increasing radii. The C++ API in `concentric_mie.hpp` expects lengths in metres.

Running the executable without arguments uses the lossless two-layer example above. Use `--help` or `-h` to display the syntax and default values.

## Validation

The six WSL executables, two MSVC executables, and MinGW drivers with C++ and Fortran cores were tested on a TITAN V in single and double precision. The lossless test above gives `Cext = Csca = 1.981837342664075e-18 m²` and `Cabs = 0` in double precision. The double-precision DDA cross sections are `Cext = Csca ≈ 2.01618108021270e-18 m²`, corresponding to a relative error of +1.733%.

The standalone program reproduces the efficiencies of the original Scattnlay executable for two lossless layers and three absorbing layers. Two layers with the same permittivity give the same cross sections as a single layer with the same outer geometry. Extinction and scattering also agree with the homogeneous Mie reference in IF-DDA for `x = 1` and `m = 1.5 + 0.02i`. The absorbing multilayer DDA calculation with two layers of that same permittivity reports `Cabs (Scattnlay) = 4.50355445310864e-18 m²` for an outer radius of 5 nm. The test logs are in `build/validation`.

## Source

The Scattnlay headers required for far-field calculations come from the [ovidiopr/scattnlay](https://github.com/ovidiopr/scattnlay) repository, revision `ab27f6dbe063190f42d140cfc1f24a0c7fa53b1c`. They are stored in `upstream` under the GPL v3 or later license. The method is described by O. Peña and U. Pal in *Computer Physics Communications* 180, 2348–2354 (2009). [10.1016/j.cpc.2009.07.010](https://doi.org/10.1016/j.cpc.2009.07.010)
