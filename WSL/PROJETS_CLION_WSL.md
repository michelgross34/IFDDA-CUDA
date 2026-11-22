# Cibles CMake CUDA pour WSL

Le projet `CUDA/WSL` construit six executables : trois combinaisons de pilote
et d'implementation `cdmlib`, en simple et double precision.

| Cible | Sources | Precision |
|---|---|---|
| `test_main0_cpp_cdmlib_cpp_cuda_float` | `main0.cpp` + `cdmlib.cpp` | float |
| `test_main0_cpp_cdmlib_cpp_cuda_double` | `main0.cpp` + `cdmlib.cpp` | double |
| `test_main0_f90_cdmlib_cpp_cuda_float` | `main0.f90` + `cdmlib.cpp` | float |
| `test_main0_f90_cdmlib_cpp_cuda_double` | `main0.f90` + `cdmlib.cpp` | double |
| `test_main_f90_cdmlib_f90_cuda_float` | `main0.f90` + `cdmlib.f90` | float |
| `test_main_f90_cdmlib_f90_cuda_double` | `main0.f90` + `cdmlib.f90` | double |

Le CMake cree quatre archives statiques : `ifdda_cdmlib_f90_cuda_float/double`
et `ifdda_cdmlib_cpp_cuda_float/double`. Chacune regroupe les sources de son
implementation et de sa precision, y compris `kernel.cu`. Les modules de
precision requis par l'executable Fortran qui utilise `cdmlib.cpp` sont
compiles dans cet executable, sans archive auxiliaire.

## Construire les six executables

Sous Ubuntu WSL, avec CMake, Ninja, g++, gfortran, OpenMP et NVCC disponibles :

```bash
cd /mnt/e/msna6/test_chat_on_steroid/if-dda-master/CUDA/WSL
./build_all.sh
```

Le script construit les six cibles dans `CUDA/WSL/build/`. Par defaut, il
utilise Release, CUDA SM 70 et quatre taches paralleles. Les options peuvent
etre ajustees, par exemple :

```bash
IFDDA_BUILD_TYPE=Debug IFDDA_CUDA_ARCH=70 IFDDA_BUILD_JOBS=6 ./build_all.sh
```

Pour construire une seule cible :

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build --target test_main_f90_cdmlib_f90_cuda_double --parallel 4
```

Dans CLion, choisir une toolchain Linux/WSL et l'une des six cibles comme
configuration d'execution. Ne pas reutiliser un cache CMake Windows : les
chemins et les compilateurs sont differents.