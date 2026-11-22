# Compilation CUDA native sous WSL / CLion

Le projet CMake de ce dossier compile `src/kernel.cu` avec NVCC dans WSL et
produit six executables. Les kernels sont integres dans quatre archives
statiques, une par combinaison CDMLIB (Fortran ou C++) et precision (float ou
double). Les variantes Fortran utilisent OpenMP; aucun lien OpenBLAS ou FFTW
n'est configure dans ce projet WSL.

## CLion

1. Installer CMake, Ninja, g++, gfortran, OpenMP et le toolkit NVIDIA CUDA dans
   la distribution WSL; verifier que `nvidia-smi` voit le GPU.
2. Dans CLion, choisir une toolchain WSL et ouvrir `CUDA/WSL` comme projet CMake.
3. Configurer `CMAKE_CUDA_ARCHITECTURES=70` pour une TITAN V (SM 7.0), ou
   adapter cette valeur au GPU.
4. Les six cibles sont documentees dans `PROJETS_CLION_WSL.md`.

## Compilation par script

Dans le terminal WSL, depuis `CUDA/WSL` :

```bash
./build_all.sh
```

Le script configure et compile les six executables dans `CUDA/WSL/build/`.
Les valeurs par defaut sont Release, CUDA SM 70 et quatre taches paralleles.

## Cache Windows / WSL

Configurer et construire ce projet avec une toolchain Linux/WSL. Un cache
cree par CMake Windows ne peut pas etre reutilise avec les chemins Linux du
compilateur et du generateur Ninja/Make. CLion signale explicitement une
configuration faite depuis un hote Windows.

`Debug` passe `-G` a NVCC pour le debogage des kernels; utiliser `Release` pour
les mesures de performance.