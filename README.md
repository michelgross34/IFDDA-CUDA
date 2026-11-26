# IFDDA-CUDA

IFDDA-CUDA est un portage sur GPU CUDA du programme IFDDA. Le dépôt rassemble les sources C++, Fortran, CUDA et les en-têtes utilisés par les différentes variantes dans `src/`.

Les exécutables acceptent des options en ligne de commande. La liste des options, leurs valeurs et des exemples sont décrits dans [`doc/OPTIONS_LIGNE_DE_COMMANDE.md`](doc/OPTIONS_LIGNE_DE_COMMANDE.md).

Deux modules fournissent des sections efficaces de référence pour la comparaison DDA.

- [Scattnlay](Scattnlay/SCATTNLAY_GUIDE.md) calcule les sections de Mie pour les sphères concentriques. Le guide décrit les exécutables et bibliothèques autonomes sous WSL, MSVC et MinGW.
- [Mishchenko](Mishchenko/README.md) calcule les sections T-matrix pour les sphéroïdes et cylindres circulaires finis. Le guide décrit les trois exécutables autonomes (CPU, CPU + OpenMP, GPU + OpenMP). La référence intégrée à IF-DDA utilise exclusivement OpenMP + CUDA FP64 lorsque les sources NASA sont disponibles. Les options CMake IFDDA_WSL_TMATRIX et IFDDA_WINDOWS_TMATRIX activent cette référence.

## Prérequis généraux

- Un compilateur et un environnement CMake adaptés à la plateforme.
- Le toolkit NVIDIA CUDA. Les wrappers DDA Windows demandent CUDA Toolkit 11.8.x ; le module Mishchenko accéléré utilise CUDA 12.9 par défaut.
- Pour **Linux/WSL**, le dépôt est autonome : les en-têtes C99 nécessaires sont
  inclus dans `src/` ; le projet ne dépend pas d'un répertoire frère
  `cdmlib_C99`. Les variantes Windows MinGW/MSVC peuvent demander des
  bibliothèques externes (notamment FFTW/OpenBLAS), selon leur configuration.

## Linux et WSL

Dans un terminal Linux ou WSL, depuis la racine de ce dépôt :

```bash
cd WSL
./build_all.sh
```

Le script configure le projet avec CMake et Ninja puis compile les six exécutables, en simple et double précision : `main0.cpp + cdmlib.cpp`, `main0.f90 + cdmlib.cpp` et `main0.f90 + cdmlib.f90`. Les exécutables sont placés dans `WSL/build/`.

Il faut disposer de CMake, Ninja, d'un compilateur C++ et Fortran (par exemple GCC et gfortran), d'OpenMP et du toolkit CUDA avec `nvcc`. Le script choisit par défaut l'architecture CUDA SM 70, le mode Release et quatre tâches parallèles. Ces valeurs peuvent être ajustées avec `IFDDA_CUDA_ARCH`, `IFDDA_BUILD_TYPE`, `IFDDA_BUILD_JOBS` et `IFDDA_BUILD_DIR`.

**Compatibilité CUDA :** le compilateur CUDA 13 ne prend plus en charge les
architectures Volta `sm_70` (Titan V). Pour ce GPU, choisir un toolkit CUDA
12.9 ou antérieur compatible ; pour un GPU plus récent, indiquer son
architecture réelle, par exemple `IFDDA_CUDA_ARCH=86 ./build_all.sh` pour
un GPU `sm_86`. La variable `IFDDA_CUDA_ARCH` n'est pas une version CUDA.

En cas de mise à jour d'une ancienne archive, retélécharger la version corrigée :
`src/cdmlib_abi.h`, `src/cdmlib_config.h` et les cinq autres en-têtes
de compatibilité doivent être présents. Pour reconfigurer depuis zéro,
supprimer uniquement l'ancien répertoire de compilation `WSL/build`
(jamais `WSL` ni `src`), puis relancer `./build_all.sh`.

## Windows : préparation CUDA commune

Avant de configurer ou compiler les projets MinGW ou MSVC, ouvrez un terminal Windows à la racine du dépôt et lancez :

```bat
build_all_cuda.bat
```

Cette étape compile séparément les wrappers CUDA avec le compilateur CUDA Windows et crée les DLL ainsi que les bibliothèques d'import float et double dans `src/`. La compilation CMake de MinGW ou MSVC utilise ces fichiers : elle ne remplace pas cette étape. CUDA Toolkit 11.8.x et `nvcc` doivent être installés et accessibles.

Avec les trois fichiers NASA disponibles, le script construit aussi la bibliothèque de référence Mishchenko OpenMP + CUDA FP64. CMake déploie automatiquement les trois DLL de ce module. Le toolkit CUDA 12.9 est utilisé par défaut pour ce noyau distinct.

## Windows avec MinGW

Après la préparation CUDA ci-dessus, configurez puis compilez depuis la racine du dépôt dans PowerShell ou l'invite de commandes (avec CMake et Ninja disponibles) :

```powershell
cmake -S MINGW -B MINGW/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build MINGW/build --parallel 4
```

Le projet MinGW génère six exécutables, correspondant aux trois associations de pilotes et bibliothèques décrites dans la section Linux/WSL, chacune en simple et double précision. Il requiert également l'environnement MinGW-w64 avec gfortran, ainsi que les bibliothèques FFTW et OpenBLAS configurées par `MINGW/CMakeLists.txt`.

## Windows avec MSVC

Après la préparation CUDA commune, configurez et compilez avec Visual Studio 2019 (générateur CMake `Visual Studio 16 2019`) :

```powershell
cmake -S MSVC -B MSVC/build -G "Visual Studio 16 2019" -A x64
cmake --build MSVC/build --config Release --parallel 4
```

La configuration MSVC produit uniquement les deux variantes `main0.cpp + cdmlib.cpp`, en simple et double précision. Elle n'inclut pas les variantes Fortran : aucun compilateur Intel Fortran n'est nécessaire.

## Documentation complémentaire

- [Options de la ligne de commande](doc/OPTIONS_LIGNE_DE_COMMANDE.md)
- [Instructions du projet WSL/CLion](WSL/PROJETS_CLION_WSL.md)
