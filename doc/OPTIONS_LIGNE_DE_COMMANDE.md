# Options de la ligne de commande

Ce document décrit les options des pilotes de calcul CUDA définis dans
`src/main0.f90` et `src/main0.cpp`. Les deux pilotes ne reconnaissent pas
exactement les mêmes options : les options propres à chaque pilote sont
indiquées séparément.

## Choisir un exécutable

Les cibles hybrides et les variantes dont le nom contient
`test_main0_fortran_...` utilisent le pilote Fortran `main0.f90`. Elles
acceptent les options Fortran décrites ci-dessous.

Les cibles dont le nom contient `test_main0_cpp_...` et les anciennes cibles
`test_cuda_float` / `test_cuda_double` utilisent `main0.cpp`. Elles acceptent
le sous-ensemble C++ décrit plus bas. La précision `float` ou `double` ne
change pas la syntaxe.

Sous WSL, les noms des principales cibles sont notamment :

```text
test_hybrid_fortran_cdmlib_cuda_float
test_hybrid_fortran_cdmlib_cuda_double
test_main0_fortran_cdmlib_cpp_cuda_float
test_main0_fortran_cdmlib_cpp_cuda_double
test_main0_cpp_cdmlib_cpp_cuda_float
test_main0_cpp_cdmlib_cpp_cuda_double
```

## Syntaxe générale

Les options nommées acceptent les deux formes suivantes :

```text
--option valeur
--option=valeur
```

Pour afficher l’aide intégrée, utiliser `-h` ou `--help`. Avec le pilote
Fortran, placer cette option en premier argument. Les paramètres qui prennent
plusieurs nombres sont séparés par des virgules, par exemple `--epsilon 1.3,0.05` ou
`--grid-dimensions 32,24,16`.

Pour le pilote Fortran, le lancement sans argument utilise par défaut une
sphère `32^3`, l’illumination `pwavelinear`, `epsilon=1.6+0i`, le solveur
`GPBICG1` et le préconditionneur `0`. Les valeurs par défaut du pilote C++
sont définies séparément dans `main0.cpp`.

## Géométries, illuminations et solveurs

### Géométries nommées

Le pilote Fortran accepte :

| Nom | Alias ou identifiant historique |
|---|---|
| `concentricsphere` | `concentric`, `0` |
| `sphere` | `1` |
| `cube` | `2` |
| `cuboid1` | `3` |
| `cuboid2` | `4` |
| `inhomosphere` | `5` |
| `inhomocuboid1` | `6` |
| `inhomocuboid2` | `7` |
| `ellipsoid` | `8` |
| `nspheres` | `9` |
| `cylinder` | `10` |
| `randomsphere1` | `11` |
| `randomsphere2` | `12` |
| `arbitrary` | `13` |
| `randomspheremulti1` | `randommulti`, `14` |

Le pilote C++ utilise sa liste de géométries de test intégrée ; lancer son
`--help` pour l’afficher.

### Illuminations

Les deux pilotes acceptent les noms suivants avec `--illumination` :

`pwavelinear`, `pwavecircular`, `wavelinearmulti`, `antenna`,
`gparawavelinear`, `gparawavecircular`, `gwavelinear`, `gwavecircular`,
`gfftwavelinear`, `gfftwavecircular`, `arbitrary`, `greentensor`, `speckle`,
`demispeckle`, `confocal`, `demiconfocal`.

### Solveurs

Le pilote Fortran accepte les noms suivants avec `--solver` :

`gpbicg`, `gpbicg1`, `bornserie`, `gpbicg2`, `gpbicgplus`, `gpbicgsafe`,
`gpbicgar`, `gpbicgar2`, `gpbicgstarplus`, `bicgstarplus`, `gpbicor`,
`gcors2`, `cors`, `qmrcla`, `pimzqmr`, `ssbicgsafe2`, `tfqmr`, `zcg`,
`pimzbicgstab`, `qmrbicgstab`, `qmrbicgstab1`, `qmrbicgstab2`, `idrs2`,
`idrs4`, `idrs8`, `bicglstab2`, `bicglstab4`, `bicglstab8`,
`gpbicgstabl2`, `gpbicgstabl4`, `gpbicgstabl8`.

Le pilote C++ expose sa propre liste avec `--help`. Certains noms peuvent
différer selon la variante ; l’aide de l’exécutable fait foi.

## Options communes aux pilotes Fortran et C++

| Option | Valeur | Rôle |
|---|---|---|
| `--geometry` | Nom | Choisit une géométrie nommée. |
| `--illumination` | Nom | Choisit une illumination. |
| `--solver` | Nom | Choisit le solveur itératif. |
| `--grid-size` | Entier | Définit une grille cubique. |
| `--epsilon` | `réel,imaginaire` | Permittivité complexe du matériau. |
| `--wavelength` | nm | Longueur d’onde. |
| `--power` | W | Puissance incidente. |
| `--tolerance` | Réel | Tolérance demandée au solveur. |
| `--max-iterations` | Entier | Nombre maximal d’itérations. |
| `--nrig` | Entier | Paramètre entier du calcul / de la méthode. |
| `--preconditioner` | Entier | Sélectionne le préconditionneur. |
| `--initial-guess` | Entier | Sélectionne le mode d’initialisation. |
| `--nfft2d` | Entier | Dimension FFT 2D. |
| `--nquad` | Entier | Paramètre entier de quadrature. |
| `--theta`, `--phi` | Degrés | Angles d’incidence. |
| `--pp`, `--ss` | Réel | Composantes de polarisation. |
| `--waist` | nm | Taille du waist gaussien. |
| `--radius` | Réel | Rayon utilisé par les géométries compatibles. |
| `--geometry-angles` | `A,B,C` | Angles d’orientation de la géométrie. |
| `--geometry-dimensions` | `A,B,C` | Dimensions de la géométrie, selon le type choisi. |
| `--geometry-density` | Réel | Densité utilisée par les géométries compatibles ; le pilote C++ exige une valeur positive. |
| `--cell-size` | Réel | Taille d’une cellule de discrétisation ; le pilote C++ exige une valeur positive. |
| `--near-domain` | Entier | Sélectionne le domaine de calcul proche. |
| `--background-epsilon` | Réel | Partie imaginaire de la permittivité du milieu ambiant. |
| `--abc-type` | Entier | Type de condition absorbante. |
| `--abc-nrbc` | Réel positif ou nul | Paramètre NRBC de la condition absorbante. |
| `--abc-cos` | Réel positif ou nul | Paramètre cosinus de la condition absorbante. |
| `--quick-lens` | Entier | Sélectionne le mode lentille rapide. |
| `--quick-diffraction` | Entier | Sélectionne le mode diffraction rapide. |
| `--scattering` | Entier | Sélectionne le calcul de diffusion. |
| `--scattering-theta-points` | Entier | Nombre de points angulaires en theta. |
| `--scattering-phi-points` | Entier | Nombre de points angulaires en phi. |
| `--cross-section` | Entier | Sélectionne la section efficace à calculer. |
| `--force`, `--force-density` | Entier | Active ou sélectionne les calculs de force et de densité de force. |
| `--torque`, `--torque-density` | Entier | Active ou sélectionne les calculs de couple et de densité de couple. |
| `--energy` | Entier | Sélectionne le calcul énergétique. |
| `--local-field` | Entier | Sélectionne la sortie du champ local. Le pilote C++ accepte aussi `--local`. |
| `--macro-field` | Entier | Sélectionne la sortie du champ macroscopique. Le pilote C++ accepte aussi `--macro`. |
| `--microscope` | Entier | Sélectionne un mode microscope. |
| `--microscope-type` | Entier | Sélectionne le type de microscope. |
| `--na` | Réel | Ouverture numérique. |
| `--magnification` | Réel | Grossissement optique. |
| `--lens-z` | Réel | Position z de la lentille. |
| `--illumination-na` | Réel | Ouverture numérique de l’illumination. |
| `--speckle-seed` | Entier | Graine du générateur aléatoire speckle. |

Les valeurs entières sont des sélecteurs transmis au programme. Pour les
options dont la plage n’est pas indiquée ici, vérifier les contrôles de la
variante et utiliser `--help`.

## Options supplémentaires du pilote Fortran

Ces options sont reconnues par `main0.f90`; elles ne sont pas toutes
implémentées par `main0.cpp`.

| Option | Valeur | Rôle |
|---|---|---|
| `--method-case` | Entier `0..24` | Sélection historique de la méthode. |
| `--option-case` | Entier `0..28` | Sélection historique d’options du cas de test. |
| `--grid-dimensions` | `NX,NY,NZ` | Grille non cubique. |
| `--geometry-center` | `X,Y,Z` | Centre de la géométrie. |
| `--gaussian-center` | `X,Y,Z` | Centre du faisceau gaussien. |
| `--near-padding` | `X,Y,Z` | Marge du domaine proche ; utilisée avec `--near-domain 2`. |
| `--microscope-side` | Entier | Sélectionne le côté du microscope. |
| `--condenser-na` | Réel | Ouverture numérique du condenseur. |
| `--inner-na` | Réel | Ouverture numérique intérieure. |
| `--na-center` | `X,Y` | Centre de l’ouverture numérique. |
| `--isotropy` | `iso` ou `ani` | Choisit le traitement isotrope ou anisotrope. |
| `--polarizability` | Code de 2 caractères | Choisit le modèle de polarisabilité. |
| `--side-length` | Réel | Longueur du côté pour les géométries compatibles. |
| `--geometry-only` | Entier | Mode de vérification de géométrie seule. |
| `--save-mode` | Entier | Sélectionne le mode de sauvegarde. |
| `--read-field` | Entier | Sélectionne la relecture d’un champ. |
| `--field-file` | Chemin | Fichier de champ à lire ou écrire. |
| `--object-file` | Chemin | Fichier de description d’objet. |
| `--arbitrary-file` | Chemin | Fichier pour la géométrie arbitraire. |
| `--hdf5-file` | Chemin | Fichier HDF5 pour les variantes configurées avec HDF5. |
### Options de géométrie C++ supplémentaires

Le pilote C++ accepte aussi `--geometry-density`, `--geometry-angles`,
`--geometry-dimensions` et `--cell-size`. La validité de ces paramètres
dépend de la géométrie. Il ne prend pas en charge `--grid-dimensions` dans
son parseur actuel.

## Arguments positionnels historiques du pilote Fortran

Les sept premiers arguments positionnels restent acceptés pour les anciens
scripts :

```text
GEOMETRY_ID BEAM_ID NRIG_ID METHOD_ID PRECONDITIONER_ID INITIAL_GUESS_ID OPTION_ID
```

Les identifiants sont les sélections entières historiques du programme. Les
options nommées sont préférables pour les nouveaux scripts et peuvent être
ajoutées après les arguments positionnels pour remplacer les valeurs
correspondantes.

## Exemples

Cas Fortran hybride en simple précision sous WSL :

```bash
./test_hybrid_fortran_cdmlib_cuda_float \
  --geometry sphere --illumination pwavelinear \
  --grid-size 32 --epsilon 1.3,0.05 \
  --solver gpbicg1 --preconditioner 0 --tolerance 1e-4
```

Cas avec pilote C++ :

```bash
./test_main0_cpp_cdmlib_cpp_cuda_double \
  --geometry sphere --illumination pwavelinear \
  --grid-size 32 --epsilon 1.3,0.05 \
  --solver gpbicg --preconditioner 0 --tolerance 1e-4
```

Pour les builds Windows, remplacer les noms des exécutables par les fichiers
`.exe` produits par la configuration utilisée.
