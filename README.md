# IFDDA-CUDA

IFDDA-CUDA est un portage sur GPU CUDA du programme IFDDA. Le dépôt rassemble les sources C++, Fortran, CUDA et les en-têtes utilisés par les différentes variantes dans `src/`.

Les exécutables acceptent des options en ligne de commande. La liste des options, leurs valeurs et des exemples sont décrits dans [`doc/OPTIONS_LIGNE_DE_COMMANDE.md`](doc/OPTIONS_LIGNE_DE_COMMANDE.md).

## Prérequis généraux

- Un compilateur et un environnement CMake adaptés à la plateforme.
- Le toolkit NVIDIA CUDA. Les projets Windows demandent CUDA Toolkit 11.8.x.
- Le dépôt doit se trouver dans l'arborescence IF-DDA attendue par les CMake : ceux-ci utilisent aussi des dépendances situées dans le répertoire parent, notamment `cdmlib_C99` et les bibliothèques FFTW/OpenBLAS selon la variante.

## Linux et WSL

Dans un terminal Linux ou WSL, depuis la racine de ce dépôt :

```bash
cd WSL
./build_all.sh
```

Le script configure le projet avec CMake et Ninja puis compile les six exécutables, en simple et double précision : `main0.cpp + cdmlib.cpp`, `main0.f90 + cdmlib.cpp` et `main0.f90 + cdmlib.f90`. Les exécutables sont placés dans `WSL/build/`.

Il faut disposer de CMake, Ninja, d'un compilateur C++ et Fortran (par exemple GCC et gfortran), d'OpenMP et du toolkit CUDA avec `nvcc`. Le script choisit par défaut l'architecture CUDA SM 70, le mode Release et quatre tâches parallèles. Ces valeurs peuvent être ajustées avec `IFDDA_CUDA_ARCH`, `IFDDA_BUILD_TYPE`, `IFDDA_BUILD_JOBS` et `IFDDA_BUILD_DIR`.

## Windows : préparation CUDA commune

Avant de configurer ou compiler les projets MinGW ou MSVC, ouvrez un terminal Windows à la racine du dépôt et lancez :

```bat
build_all_cuda.bat
```

Cette étape compile séparément les wrappers CUDA avec le compilateur CUDA Windows et crée les DLL ainsi que les bibliothèques d'import float et double dans `src/`. La compilation CMake de MinGW ou MSVC utilise ces fichiers : elle ne remplace pas cette étape. CUDA Toolkit 11.8.x et `nvcc` doivent être installés et accessibles.

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
