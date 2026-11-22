# PrÃ©paration du dÃ©pÃ´t GitHub IFDDA-CUDA

Date : 22 novembre 2026.

## Ã‰tat avant intervention

- Le code CUDA Ã©tait rangÃ© Ã  la racine de `if-dda-master/CUDA` ; le rÃ©pertoire `src/` ne contenait pas les sources C++, Fortran et en-tÃªtes visÃ©s.
- Les configurations racine CMake, MinGW et MSVC pointaient vers lâ€™ancienne disposition.
- Les projets Visual Studio 2019 v142 Ã©taient gÃ©nÃ©rÃ©s dans `MSVC/vs2019-v142` et contenaient des chemins absolus vers les unitÃ©s de compilation et le rÃ©pertoire dâ€™inclusion.
- Les rapports de suivi `BORN_RYTOV_SCOPE_FIX_2026-11-12.md` et `CHAN_SINGLE_GLOBAL_MATRIX_2026-11-12.md` consignent des compilations/tests antÃ©rieurs, y compris certains tests float et double. Le rapport Born/Rytov prÃ©cise que les validations ciblÃ©es de cette correction ne couvrent que le solveur standard double ; elles ne prouvent ni toutes les branches Born/Rytov ni la paritÃ© Fortran.

## Modifications de cette Ã©tape

- DÃ©placement de 173 fichiers de la racine vers `src/` : extensions `.cpp`, `.f90`, `.h` et `.hpp`.
- Mise Ã  jour de `CMakeLists.txt`, `MINGW/CMakeLists.txt` et `MSVC/CMakeLists.txt` pour rÃ©soudre les unitÃ©s dÃ©placÃ©es et leurs en-tÃªtes. Les ressources CUDA `kernel.cu`, `.cuh`, scripts et dÃ©pendances restent dans leurs emplacements, car ils ne font pas partie des extensions demandÃ©es.
- Mise Ã  jour des entrÃ©es et rÃ©pertoires dâ€™inclusion dans les projets `.vcxproj` de `MSVC/vs2019-v142`.
- Ajout dâ€™un README de dÃ©pÃ´t et dâ€™un `.gitignore` pour exclure les sorties de compilation, caches, journaux, donnÃ©es de simulation et sauvegardes temporaires.

## Validation effectuÃ©e

- Inventaire aprÃ¨s dÃ©placement : 173 fichiers correspondants sont prÃ©sents dans `src/`.
- VÃ©rification statique des chemins CMake et Visual Studio en cours ; aucune compilation ni comparaison numÃ©rique Fortran nâ€™a Ã©tÃ© lancÃ©e pour cette rÃ©organisation.
- Les validations historiques consignÃ©es dans les rapports existants restent des rÃ©sultats antÃ©rieurs, pas une nouvelle validation de cette rÃ©organisation.

### Incident de gÃ©nÃ©ration MinGW aprÃ¨s dÃ©placement

- Erreur initiale signalÃ©e : Ninja ne pouvait pas ouvrir `CMakeFiles/rules.ninja`. Le cache existant contenait `build.ninja`, mais la rÃ¨gle Ã©tait absente.
- La configuration MinGW avec le CMake de CLion 2024.3.2 a rÃ©gÃ©nÃ©rÃ© `rules.ninja` et les rÃ¨gles de compilation.
- Cette rÃ©gÃ©nÃ©ration a rÃ©vÃ©lÃ© des rÃ©fÃ©rences oubliÃ©es dans les configurations MinGW, WSL et hybride ; elles pointaient sur les anciens emplacements de sources, tests et noyau. Les CMake associÃ©s ont Ã©tÃ© corrigÃ©s.
- Le dossier de build existant contenait aussi 11 `Fortran.dd` gÃ©nÃ©rÃ©s avant le dÃ©placement et mentionnant `CUDA/main0.f90`. Ces fichiers de dÃ©pendances gÃ©nÃ©rÃ©s seulement ont Ã©tÃ© retirÃ©s pour quâ€™ils puissent Ãªtre recalculÃ©s ; les objets, bibliothÃ¨ques, DLL et exÃ©cutables existants nâ€™ont pas Ã©tÃ© supprimÃ©s.
- MalgrÃ© la gÃ©nÃ©ration CMake rÃ©ussie, la compilation `all -j 14` est restÃ©e sans sortie avec Ninja inactif et a Ã©tÃ© interrompue. La compilation complÃ¨te reste donc Ã  confirmer.
## DÃ©pendances et limites

- Les configurations CMake utilisent des dÃ©pendances du dÃ©pÃ´t parent IF-DDA (`cdmlib_CPP`, `cdmlib_C99`, `cdmlib`, `fftw`) ainsi que CUDA Toolkit 11.8. Le dÃ©pÃ´t CUDA seul nâ€™est donc pas une archive autonome de toutes les dÃ©pendances.
- Les projets VS2019 prÃ©sents dans lâ€™arborescence sont des fichiers gÃ©nÃ©rÃ©s qui contiennent encore des chemins spÃ©cifiques Ã  lâ€™environnement Windows local pour certaines dÃ©pendances et outils. Les chemins des sources dÃ©placÃ©es ont Ã©tÃ© actualisÃ©s ; une rÃ©gÃ©nÃ©ration CMake avec les dÃ©pendances installÃ©es reste nÃ©cessaire pour un environnement diffÃ©rent.
- Le dÃ©pÃ´t distant `michelgross34/IFDDA-CUDA` existe et est privÃ© ; il Ã©tait vide lors de la vÃ©rification.

### Correction du build CUDA depuis `src` â€” 22 novembre 2026

- Copie de `cdmlib_CPP/cdmlib_cpp_precision.hpp` dans `src/cdmlib_cpp_precision.hpp` ; les fichiers sont identiques.
- `src/kernel.h` inclut maintenant lâ€™en-tÃªte local `cdmlib_cpp_precision.hpp` et ne rÃ©fÃ©rence plus lâ€™ancien rÃ©pertoire.
- Mise Ã  jour de `src/go.bat` pour inclure les `.cuh` restÃ©s dans le dossier parent et Ã©crire les bibliothÃ¨ques CUDA dans la racine `CUDA`, attendue par CMake. `go.bat` racine ajoute Ã©galement `src` au chemin dâ€™inclusion pour utiliser le mÃªme `kernel.h`.
- Compilation depuis `CUDA/src` : `cmd /c go_all.bat` â€” **PASS**. `kernel_double.dll/.lib` et `kernel_float.dll/.lib` gÃ©nÃ©rÃ©s dans `CUDA/`, code retour 0.
- Le lancement `go_all.bat` appelle `src/go.bat`, qui utilise `src/kernel.cu`, identique Ã  la copie racine, et produit les DLL/import libraries dans `CUDA/`. Le script `go_all.bat` est ensuite placÃ© Ã  la racine `CUDA` et appelle `src/go.bat`.
### Sorties CUDA placÃ©es dans `src` â€” 22 novembre 2026

- `src/go.bat` Ã©crit par dÃ©faut `kernel_double.dll/.lib` et `kernel_float.dll/.lib` dans son dossier `src`. Le `go.bat` racine Ã©crit Ã©galement dans `src`.
- Les chemins par dÃ©faut des bibliothÃ¨ques importÃ©es ont Ã©tÃ© dÃ©placÃ©s dans les CMake racine, MinGW, MinGW alternatif et MSVC.
- Les projets Visual Studio 2019 v142 ont Ã©tÃ© rÃ©gÃ©nÃ©rÃ©s avec les quatre artefacts importÃ©s sous `CUDA/src`.
- Compilation `cmd /c go_all.bat` depuis `CUDA/src` : **PASS**, double et float. Configuration MinGW et gÃ©nÃ©ration MSVC/VS2019 : **PASS** avec `IFDDA_KERNEL_DLL_*` et `IFDDA_KERNEL_IMPLIB_*` configurÃ©s vers `src`.### `go_all.bat` Ã  la racine de CUDA â€” 22 novembre 2026

- DÃ©placÃ© `src/go_all.bat` vers `CUDA/go_all.bat`.
- Le script racine appelle `src/go.bat` et transmet les arguments. Les DLL et fichiers `.lib` sont toujours produits dans `src/`.
- Le README dÃ©crit maintenant lâ€™emplacement actuel du script et des sorties.### Script de compilation WSL et sources sous `src`

- `WSL/build_all.sh` valide maintenant la prÃ©sence des fichiers dâ€™entrÃ©e sous `CUDA/src` et du noyau CUDA dans `CUDA/` avant de lancer CMake.
- Le rÃ©pertoire de build garde `CUDA/WSL/build` par dÃ©faut ; `IFDDA_BUILD_DIR` permet de le placer dans le systÃ¨me de fichiers Linux WSL. Les chemins de build relatifs sont rÃ©solus depuis `CUDA/WSL`.
- Aucune sortie de compilation existante nâ€™est supprimÃ©e par le script.- Tentative de vÃ©rification syntaxique par `wsl.exe -e bash -n` : non exÃ©cutÃ©e, le service WSL a renvoyÃ© `Wsl/Service/CreateInstance/E_ACCESSDENIED`. Le script nâ€™a pas Ã©tÃ© lancÃ© et aucun build WSL nâ€™a Ã©tÃ© fait dans cette Ã©tape.## Publication distante

- Les commits locaux contenant lâ€™arborescence et les correctifs sont sur la branche `main`.
- Le dÃ©pÃ´t distant existe dÃ©jÃ , est privÃ© et Ã©tait vide. Le push Git HTTPS a Ã©chouÃ© car aucune information dâ€™authentification nâ€™est disponible (`SEC_E_NO_CREDENTIALS`).
- Une Ã©criture par le connecteur GitHub a aussi Ã©tÃ© tentÃ©e, mais lâ€™outil a rÃ©pondu quâ€™une approbation Ã©tait requise alors que la politique actuelle ne permet pas de lâ€™accorder. Aucun fichier nâ€™a Ã©tÃ© Ã©crit sur GitHub ; le dÃ©pÃ´t distant reste vide.
- AprÃ¨s authentification GitHub sur cette machine, publier avec `git push -u origin main` depuis ce dossier.
## Reprise

Pour poursuivre, vÃ©rifier les diffÃ©rences et les fichiers ignorÃ©s, configurer CMake dans un rÃ©pertoire de build neuf, puis compiler et exÃ©cuter les variantes float et double disponibles. Pour dÃ©clarer une paritÃ© CUDA complÃ¨te, comparer les rÃ©sultats numÃ©riques Ã  lâ€™original Fortran sur les cas concernÃ©s.

### Diagnostic de l'echec float sous WSL - 22 novembre 2026

- Symptome signale : `test_cuda_float` s'arretait avec Â« Tolerance for the iterative method too small! Â» avant toute iteration, alors que le cas double aboutissait.
- Cause identifiee dans `WSL/CMakeLists.txt` : la bibliotheque `ifdda_cuda_cpp_float` etait compilee avec `USE_SIMP`, mais l'executable `test_cuda_float` (son `main0.cpp`) ne recevait pas cette definition. Le pilote utilisait donc des scalaires double avec une bibliotheque qui les interprete en float, ce qui corrompt notamment `tolinit` dans l'ABI.
- Correction : application de `ifdda_wsl_precision_settings` a chaque executable `test_cuda_${_precision}`. Le pilote et sa bibliotheque utilisent maintenant le meme type scalaire; le chemin double conserve aussi sa configuration explicite.
- Validation : verification du CMake et de la coherence des definitions par lecture statique. Compilation et execution WSL, puis comparaison numerique float/double contre le Fortran original, restent a faire : l'acces au service WSL avait renvoye `Wsl/Service/CreateInstance/E_ACCESSDENIED`. Ne pas considerer ce portage valide avant ces controles.

### Sources CUDA rassemblees dans src - 22 novembre 2026

- `src/kernel.cu` et `src/mod_precision.f90` etaient deja presents et sont identiques aux sources respectives `CUDA/kernel.cu` et `cdmlib/module/mod_precision.f90`.
- Copie de `cdmlib_CPP/rytov_vector_field.cpp` vers `CUDA/src/rytov_vector_field.cpp`.
- Le include MinGW `gfortran_rng_cpp.hpp`, requis par le `random_bridge_c.cpp` local, a aussi ete copie vers `CUDA/src` pour pouvoir retirer le chemin d'inclusion `cdmlib_CPP`.
- Les CMake CUDA racine, MinGW, MinGW alternatif, MSVC et WSL utilisent maintenant les sources locales pour `random_bridge_c.cpp`, `rytov_vector_field.cpp`, `mod_precision.f90` et `kernel.cu` lorsqu'ils en ont besoin. Le script WSL verifie `src/kernel.cu`.
- Les deux projets VS2019 et leurs fichiers `.filters` referencent maintenant le Rytov local; les chemins d'inclusion `cdmlib_CPP` ont ete retires.
- Controle statique : aucune reference de repertoire `cdmlib_CPP` ou `cdmlib/module/mod_precision` restante dans les CMakeLists, le script WSL et les projets VCXPROJ actifs. Les references a `cdmlib_C99`, `cdmlib` et FFTW restent des dependances externes distinctes.
- Aucune compilation n'a ete lancee pour cette etape. Les configurations MinGW/MSVC et WSL doivent etre regenerees et compilees pour valider les changements; la comparaison numerique avec Fortran reste requise.

### Dependances d'inclusion du noyau CUDA localisees - 22 novembre 2026

- Erreur signalee au build WSL : `src/kernel.cu` ne trouvait pas `force_green_derivative_kernels.cuh`, reste a la racine de CUDA.
- Copies dans `src/` de `force_green_derivative_kernels.cuh` et `gaussian_incident_kernel.cuh`; le fichier transitif `gaussian_gkp_tables.h` est deja dans `src/`.
- `src/go.bat` ne passe plus le repertoire parent comme chemin d'inclusion NVCC. `WSL/build_all.sh` controle maintenant la presence des trois en-tetes locaux en plus de `src/kernel.cu`.
- Controle statique des inclusions directes/transitives et des fichiers : effectue. Compilation WSL non relancee dans cet environnement; confirmer avec `./build_all.sh`.

### En-tete ABI manquant dans CUDA/src - 22 novembre 2026

- Erreur signalee a 15 pour cent : `random_bridge_c.cpp` ne trouvait pas `cdmlib_geometry_abi.hpp`, inclus par `cdmlib_cpp_abi.hpp`.
- Copie dans `src/` de `cdmlib_geometry_abi.hpp` et `cdmlib_cpp_fftw_precision.hpp` depuis le dossier source `cdmlib__CPP` present dans l'arbre parent. Les includes relatifs `../cdmlib_CPP/...` ont ete remplaces par les en-tetes locaux `src`.
- Audit statique des inclusions entre les sources et en-tetes de `src` : pas d'autre en-tete local introuvable dans `src`, CUDA racine, `cdmlib_C99`, `cdmlib__CPP` ou FFTW.
- `WSL/build_all.sh` verifie maintenant les en-tetes ABI locaux. Compilation WSL toujours a confirmer par l'utilisateur; le service WSL de cet environnement reste inaccessible.

### Preparation du depot local CUDA/github - 22 novembre 2026

- Tentative `git clone https://github.com/michelgross34/IFDDA-CUDA.git github` : echec, Git Credential Manager n'a pas de credentials (`SEC_E_NO_CREDENTIALS`). La cible n'a pas ete creee par la tentative.
- Creation de `CUDA/github` comme depot Git local vide sur la branche `main`, avec `origin` configure vers le depot GitHub demande. Aucun fichier source n'a ete copie, aucun commit n'a ete cree, et aucun fetch/push n'a reussi.
- Pour en faire un vrai clone synchronise, s'authentifier sur GitHub puis lancer `git -C CUDA/github fetch origin` et `git -C CUDA/github pull origin main` (si la branche distante contient un commit).

### Copie des sources dans le depot Git local CUDA/github - 22 novembre 2026

- Copie de 183 fichiers depuis `CUDA/src` vers `CUDA/github/src` : 122 `.cpp`, 34 `.f90` et 27 en-tetes (`.h`, `.hpp`, `.hh`, `.hxx`, `.cuh`).
- Controle SHA-256 : 0 difference entre sources et copies; le dossier de destination contient 183 fichiers.
- Les fichiers sont presents localement et non suivis par Git (`src/` apparait comme non suivi). Aucun commit ou push n'a ete effectue; le clone distant reste bloque par l'absence d'identifiants GitHub.

### Configurations de compilation ajoutees a CUDA/github - 22 novembre 2026

- Copie de `MINGW/CMakeLists.txt`, `MSVC/CMakeLists.txt`, de `WSL/CMakeLists.txt` et `WSL/build_all.sh` dans l'arborescence correspondante de `CUDA/github`.
- Copie des huit projets VS2019 `.vcxproj` et de leurs huit fichiers `.filters` vers `CUDA/github/MSVC/vs2019-v142/`. Les projets generes auxiliaires `ALL_BUILD`, `ZERO_CHECK` et `CMakeFiles` ne sont pas inclus.
- Copie de `CUDA/src/go.bat` vers `CUDA/github/src/go.bat`.
- Controle SHA-256 des 21 fichiers copies : aucune difference. Les dossiers MINGW, MSVC, WSL et src sont encore non suivis; aucun commit/push n'a ete effectue.

### Script de compilation racine ajoute - 22 novembre 2026

- Copie de `CUDA/build_all_cuda.bat` vers `CUDA/github/build_all_cuda.bat` (57 octets); SHA-256 identique.
- Ce script racine appelle `src/go.bat`, qui construit les DLL CUDA dans `src/`.

### Commit local pour le depot distant - 22 novembre 2026

- Tous les fichiers presents dans `CUDA/github` (`src`, `MINGW`, `MSVC`, `WSL`, `build_all_cuda.bat`) ont ete ajoutes au commit local `905988f` (`Add CUDA sources and build configurations`). Arbre de travail propre.
- `git push -u origin main` a echoue : `SEC_E_NO_CREDENTIALS`; aucune publication distante n'a eu lieu.
- Pour finaliser dans PowerShell : `Set-Location 'E:\msna6\test_chat_on_steroid\if-dda-master\CUDA\github'`, puis `git credential-manager github login` (terminer l'authentification dans le navigateur) et enfin `git push -u origin main`.

### Publication GitHub terminee - 22 novembre 2026

- Apres connexion GCM du compte `michelgross34`, le helper Git standard ne pouvait pas demarrer son shell (`couldn't create signal pipe`, Win32 error 5). Le jeton a ete recupere en memoire depuis Git Credential Manager, sans affichage, puis passe a Git via `http.extraheader` uniquement pour la commande de push.
- `git push -u origin main` : **PASS**, branche `main` publiee sur `origin/main`.
- Commit publie : `905988fdc5cdff7f5587794016c9e010ec20a77c` (`Add CUDA sources and build configurations`). L'arbre local est propre et suit `origin/main`.

### Documentation des options de ligne de commande - 22 novembre 2026

- Creation de `CUDA/doc/OPTIONS_LIGNE_DE_COMMANDE.md` et copie identique dans `CUDA/github/doc/OPTIONS_LIGNE_DE_COMMANDE.md`.
- Le document distingue les parsers `main0.f90` et `main0.cpp`, liste les options communes et celles propres au Fortran, les geometries/illuminations/solveurs, la syntaxe et des exemples. Contenu recoupe avec les options effectivement traitees dans les deux sources.
- Empreinte SHA-256 des deux copies identique. Pas de compilation lancee pour cette documentation.
- Commit `df3236a8cb4aac24789806db47de1f0dcce69af2` (`Match C++ CLI options to Fortran driver`) et correctif `1ec4d03` (`Preserve geometry option precedence`) pousses sur `origin/main`; arbre de travail du clone propre et synchronise.
- Commit `b7130501b942123d809e94cfa7385c12f0dd93e5` (`Document command line options`) pousse sur GitHub; `main` suit `origin/main` et l'arbre de travail est propre.

### Parite CLI de `main0.cpp` avec `main0.f90` - 22 novembre 2026

- `CUDA/src/main0.cpp` accepte maintenant les options nommees reconnues par `main0.f90`, dont `--grid-dimensions`, `--geometry-center`, `--gaussian-center`, `--near-padding`, les options microscope/NA, les choix de polarizabilite/isotropie et les options de fichiers.
- Ajout de la forme positionnelle historique a sept arguments (`GEOMETRY_ID BEAM_ID NRIG_ID METHOD_ID PRECONDITIONER_ID INITIAL_GUESS_ID OPTION_ID`) et du mappage des identifiants de methode, geometrie, illumination et option.
- Les options sont transmises aux parametres de l'appel `cdmlib`; `--near-padding` exige `--near-domain 2`, comme dans le pilote Fortran.
- Suppression de la dependance a `../tests/geometry_cli_cases.h`, absent du depot de travail; les noms de geometrie et l'aide correspondante sont definis dans `main0.cpp`. Les constantes complexes de la fixture sont egalement definies localement. Le comportement explicite `IFDDA_TEST_GEOMETRY_ASCII` est conserve.
- Documentation CLI mise a jour dans `CUDA/doc/OPTIONS_LIGNE_DE_COMMANDE.md` et copiee dans `CUDA/github` avec le source.
- Verification syntaxique C++ avec `g++ -std=c++17 -fsyntax-only` : PASS en double precision et avec `-DUSE_SIMP`.
- Aucun executable CUDA n'a ete compile ou execute et aucune comparaison numerique Fortran/CUDA n'a ete lancee. `wsl --status` echoue dans cette session avec `Wsl/EnumerateDistros/Service/E_ACCESSDENIED`; les compilations numeriques restent a faire sous WSL. La verification syntaxique ne valide ni le lien ABI ni les resultats du solveur.
- Commit `df3236a8cb4aac24789806db47de1f0dcce69af2` (`Match C++ CLI options to Fortran driver`) et correctif `1ec4d03` (`Preserve geometry option precedence`) pousses sur `origin/main`; arbre de travail du clone propre et synchronise.



### Reduction des variantes CMake/Visual Studio aux combinaisons demandees - 22 novembre 2026

- MINGW declare six executables : `main0.cpp + cdmlib.cpp`, `main0.f90 + cdmlib.cpp` et `main0.f90 + cdmlib.f90`, chacun en float et double. Suppression des executables de test GPU et de la cible Fortran de couverture qui etaient en plus.
- WSL declare les memes six combinaisons. Retrait de la famille C++ dupliquee `test_cuda_*`; le script `WSL/build_all.sh` configure les six cibles et supprime dans son repertoire de build les six anciens executables aux noms retires.
- Renommage coherent : `test_main0_cpp_cdmlib_cpp_cuda_*`, `test_main0_f90_cdmlib_cpp_cuda_*` et `test_main0_f90_cdmlib_f90_cuda_*`.
- MSVC suit la precision demandee par l'utilisateur : seulement les deux executables C++ `test_main0_cpp_cdmlib_cpp_cuda_float/double`; aucune variante F90 n'est ajoutee. Suppression des projets `.vcxproj` de test GPU devenus superflus et regeneration des projets MSVC.
- Ajout de `hybrid_fortran_cuda/CMakeLists.txt` au clone GitHub, car MINGW l'inclut pour les deux cibles `main0.f90 + cdmlib.f90`.
- Mise a jour des noms de cibles dans la documentation CLI et les instructions WSL.
- Verification : configuration CMake MSVC PASS; les quatre `.vcxproj` restants du clone (deux bibliotheques internes et deux executables) sont du XML valide; `git diff --check` PASS.
- Limite de validation : aucune compilation ni execution numerique effectuee. La configuration CMake MinGW est restee bloquee pendant la detection de l'ABI CXX (`Detecting CXX compiler ABI info`) et a ete interrompue; WSL n'a pas ete configure depuis cette session Windows. Les CMakeLists sont donc modifies mais MinGW/WSL restent a compiler.
- Publication GitHub a faire apres commit local.

### Selection des six exécutables et publication — 22 novembre 2026

- Dépôt GitHub `michelgross34/IFDDA-CUDA`, branche `main` : changements publiés par le commit `42c6e92` (`Keep only requested CUDA executables`). Le clone local `CUDA/github` est propre après publication.
- MinGW et WSL exposent les trois combinaisons demandées, en simple et double précision : `main0.cpp + cdmlib.cpp`, `main0.f90 + cdmlib.cpp`, et `main0.f90 + cdmlib.f90`. Les noms suivent `test_main0_<langage>_cdmlib_<langage>_cuda_<precision>`.
- MSVC ne conserve que `main0.cpp + cdmlib.cpp` en simple et double précision, conformément à la demande de ne pas dépendre d'Intel Fortran. Les projets Visual Studio correspondants ont été régénérés et les anciens projets de tests retirés du dépôt.
- La génération CMake MSVC a réussi et les fichiers `.vcxproj` restants ont été vérifiés comme XML. `git diff --check` n'a relevé aucune erreur d'espacement.
- Limites : la configuration MinGW est restée bloquée pendant la détection ABI du compilateur C++; l'accès Bash/WSL a échoué (`E_ACCESSDENIED`). Aucune compilation MinGW/WSL, exécution des six programmes ou comparaison numérique float/double avec Fortran n'a été réalisée. Le portage ne peut donc pas être déclaré validé.

### Controle du CMake WSL — 22 novembre 2026

- Controle demande : `WSL/CMakeLists.txt` declare les six executables attendus (trois couples main0/cdmlib en float et double), utilise les sources de `CUDA/src`, et ne declare plus les anciens executables `test_cuda_*`.
- Les fichiers `WSL/CMakeLists.txt` et `WSL/build_all.sh` dans `CUDA` et `CUDA/github` ont des empreintes SHA-256 identiques. La version du clone est deja publiee dans le commit `42c6e92`.
- Aucun changement supplementaire n'etait necessaire au CMake WSL. La configuration et la compilation WSL restent a executer dans un environnement WSL fonctionnel; aucune validation numerique n'est revendiquee.

### Renommage explicite des cibles Fortran WSL — 22 novembre 2026

- Renommage de l'executable `test_main0_f90_cdmlib_f90_cuda_<precision>` en `test_main_f90_cdmlib_f90_cuda_<precision>` dans `WSL/CMakeLists.txt`.
- Renommage des bibliotheques internes `ifdda_hybrid_f90_core_*` et `ifdda_hybrid_cuda_leaf_*` en `ifdda_main_f90_cdmlib_f90_core_*` et `ifdda_main_f90_cdmlib_f90_cuda_leaf_*`.
- Mise a jour du README et des instructions CLion WSL locaux; `build_all.sh` supprime aussi les deux anciens noms d'executable pour eviter les restes dans son repertoire de build.
- Le CMake et le script ont ete synchronises avec `CUDA/github`. Controle des references internes effectue; compilation WSL non lancee dans cette session.

### Nettoyage des liens de bibliotheques WSL — 22 novembre 2026

- Suppression d'OpenBLAS des dependances configurees et liees dans `WSL/CMakeLists.txt`; une recherche dans les sources WSL incluses ne trouve aucun appel BLAS/OpenBLAS.
- La bibliotheque C++ `ifdda_cdmlib_cpp_cuda_*` ne lie plus `OpenMP::OpenMP_Fortran`, inutile pour ses sources C++; elle conserve `OpenMP::OpenMP_CXX` pour les regions OpenMP de `cdmlib.cpp` et les bibliotheques CUDA transitives via `ifdda_cuda_kernels_*`.
- OpenMP Fortran est conserve sur `ifdda_main_f90_cdmlib_f90_core_*`, qui compile `cdmlib.f90`. Les bibliotheques necessaires aux noyaux (`cudart`, `cufft`, `cublas`) restent declarees sur la cible kernel partagee.
- Le README et les instructions CLion WSL locaux ne demandent plus OpenBLAS. Verification textuelle et `git diff --check` PASS; aucun build WSL n'a ete lance, donc la configuration et le lien doivent encore etre valides sous WSL.

### Noms des bibliotheques internes WSL clarifies — 22 novembre 2026

- Correction des noms internes pour decrire leur contenu, sans les associer a `main0` : `ifdda_cdmlib_f90_core_*` (cdmlib.f90 et precision), `ifdda_cuda_cpp_support_*` (sources C++/CUDA sans cdmlib.cpp), et `ifdda_fortran_precision_*` (mod_precision.f90 uniquement).
- Les executables conservent `test_main_f90_cdmlib_f90_*` et les autres noms `test_main...`, car ils incluent effectivement le point d'entree principal.
- Toutes les references CMake WSL vers ces bibliotheques ont ete mises a jour dans le projet local et le clone GitHub. Controle statique et `git diff --check` PASS; aucun build WSL n'a ete lance.

### Regroupement des bibliotheques WSL en quatre archives — 22 novembre 2026

- La configuration WSL cree maintenant exactement quatre archives statiques : `ifdda_cdmlib_f90_cuda_{float,double}` et `ifdda_cdmlib_cpp_cuda_{float,double}`.
- Les sources support C++/CUDA et `kernel.cu` sont integrees dans chacune des variantes correspondantes; les trois bibliotheques auxiliaires (`cuda_kernels`, support C++ et precision Fortran) ont ete retirees.
- Pour `main0.f90 + cdmlib.cpp`, `mod_precision.f90` est compile directement dans l'executable, afin de fournir le module sans creer une cinquieme archive. Les executables continuent de ne lier qu'une seule archive CDMLIB pour leur precision.
- Controle statique : deux declarations d'archive dans une boucle double/float, donc quatre cibles generees. `git diff --check` PASS et copie locale/clone identique. Pas de configuration ou compilation WSL effectuee; le build mixte Fortran/C++/CUDA doit encore etre valide sous WSL.

### Tentative de compilation WSL via build_all.sh — 22 novembre 2026

- Tentative depuis cette session Windows : `wsl.exe --exec bash -lc 'cd /mnt/e/msna6/test_chat_on_steroid/if-dda-master/CUDA/WSL && ./build_all.sh'`.
- Echec avant execution du script : `Wsl/Service/CreateInstance/E_ACCESSDENIED`. Le script et la toolchain Linux/WSL ne sont donc pas atteints; aucun résultat de compilation ne peut être déclaré.
- Le transfert de cette commande via l'outil de relai utilisateur n'a pas ete enregistre, car le contexte ne fournit pas l'identite exacte du chat. A lancer manuellement dans le terminal WSL de CLion, depuis `/mnt/e/msna6/test_chat_on_steroid/if-dda-master/CUDA/WSL` : `./build_all.sh`.
- En l'absence d'une compilation, la validation numérique float/double reste egalement non faite.

### Builds plateformes confirmes par l'utilisateur — 22 novembre 2026

- L'utilisateur confirme que le code compile et fonctionne avec WSL, MSVC et MinGW. Cette information concerne sa verification locale; aucune sortie detaillee de compilation n'a ete recue dans cette session.
- Les sources locales `src/kernel.cu` et `src/cdmlib_cpp_abi.hpp` ont ete synchronisees dans le clone GitHub. Les notices WSL, la documentation CLI et le README ont egalement ete actualises pour reflet les noms de cibles WSL et les quatre archives.
- Aucune comparaison numerique simple/double avec la reference Fortran n'a ete fournie ici; le portage n'est pas marque comme valide numeriquement sur cette seule confirmation de compilation.
