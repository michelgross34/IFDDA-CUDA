# IFDDA CUDA

Backend CUDA du projet IF-DDA. Les sources C++, Fortran, CUDA et les en-tetes
necessaires sont regroupes dans `src/`.

## Compilation

- **WSL** : ouvrir `WSL/` dans CLion avec une toolchain Linux/WSL, ou lancer
  `./build_all.sh` depuis ce dossier dans un terminal WSL. Le projet fournit
  six executables (trois associations main0/cdmlib, float et double).
- **MinGW** : configurer le projet `MINGW/` avec une toolchain MinGW et CMake.
  Les six variantes C++/Fortran float/double y sont disponibles.
- **MSVC / Visual Studio 2019** : configurer `MSVC/` avec CMake. Pour eviter
  Intel Fortran, MSVC ne propose que `main0.cpp + cdmlib.cpp` en float et
  double.
- **Wrappers CUDA Windows** : lancer `build_all_cuda.bat` depuis la racine du
  depot pour produire les DLL/import libraries float et double dans `src/`.

Les details WSL sont dans `WSL/PROJETS_CLION_WSL.md`; les options des pilotes
sont dans `doc/OPTIONS_LIGNE_DE_COMMANDE.md`. `REPOSITORY_PREPARATION.md`
conserve l'historique des modifications et validations.