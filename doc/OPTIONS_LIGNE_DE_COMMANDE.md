# Options de la ligne de commande

Ce document présente l’interface de commande commune de `main0.cpp` et
`main0.f90`. Les options de calcul ci-dessous ont les mêmes noms et la même
syntaxe dans les deux pilotes. Leurs valeurs par défaut peuvent toutefois
différer.

## Choisir un exécutable

Les cibles `test_main0_f90_*` utilisent `main0.f90`; les cibles
`test_main0_cpp_*` utilisent `main0.cpp`. Sous WSL, la combinaison
`main0.f90 + cdmlib.f90` est nommée `test_main_f90_cdmlib_f90_cuda_<precision>`.
Les autres noms de cibles sont listés dans les instructions de compilation.

Les deux pilotes acceptent les mêmes options de calcul documentées ci-dessous. La
précision `float` ou `double` ne change pas la syntaxe.

Sous WSL, les noms des principales cibles sont notamment :

```text
test_main_f90_cdmlib_f90_cuda_float
test_main_f90_cdmlib_f90_cuda_double
test_main0_f90_cdmlib_cpp_cuda_float
test_main0_f90_cdmlib_cpp_cuda_double
test_main0_cpp_cdmlib_cpp_cuda_float
test_main0_cpp_cdmlib_cpp_cuda_double
```

MINGW et WSL fournissent ces six combinaisons. La configuration MSVC conserve
uniquement les deux variantes `main0.cpp` + `cdmlib.cpp`, car elle ne requiert
pas le compilateur Intel Fortran.

## Syntaxe générale

Les options de calcul communes acceptent les deux formes suivantes :

```text
--option valeur
--option=valeur
```

Pour afficher l’aide intégrée, utiliser `-h` ou `--help` en premier argument.
Les paramètres qui prennent
plusieurs nombres sont séparés par des virgules, par exemple `--epsilon 1.3,0.05` ou
`--grid-dimensions 32,24,16`.

Les valeurs par défaut sans argument sont celles du pilote Fortran : sphère
`32^3`, illumination `pwavelinear`, `epsilon=1.6+0i`, solveur `GPBICG1` et
préconditionneur `0`. Les valeurs par défaut du pilote C++ sont définies dans
`main0.cpp` et peuvent différer ; cela ne change pas les noms d’options communs.

## Géométries, illuminations et solveurs

### Géométries nommées

Les deux pilotes acceptent les mêmes géométries et les mêmes alias :

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

### Illuminations

Les deux pilotes acceptent les noms suivants avec `--illumination` :

`pwavelinear`, `pwavecircular`, `wavelinearmulti`, `antenna`,
`gparawavelinear`, `gparawavecircular`, `gwavelinear`, `gwavecircular`,
`gfftwavelinear`, `gfftwavecircular`, `arbitrary`, `greentensor`, `speckle`,
`demispeckle`, `confocal`, `demiconfocal`.

### Solveurs

Les deux pilotes acceptent les noms suivants avec `--solver` :

`gpbicg`, `gpbicg1`, `bornserie`, `gpbicg2`, `gpbicgplus`, `gpbicgsafe`,
`gpbicgar`, `gpbicgar2`, `gpbicgstarplus`, `bicgstarplus`, `gpbicor`,
`gcors2`, `cors`, `qmrcla`, `pimzqmr`, `ssbicgsafe2`, `tfqmr`, `zcg`,
`pimzbicgstab`, `qmrbicgstab`, `qmrbicgstab1`, `qmrbicgstab2`, `idrs2`,
`idrs4`, `idrs8`, `bicglstab2`, `bicglstab4`, `bicglstab8`,
`gpbicgstabl2`, `gpbicgstabl4`, `gpbicgstabl8`.

Les alias `gpbicg` et `gpbicg1` sélectionnent tous deux `GPBICG1` dans les deux
pilotes.

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
| `--geometry-density` | Réel positif | Densité utilisée par les géométries compatibles. |
| `--cell-size` | Réel positif | Taille d’une cellule de discrétisation. |
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
| `--local-field` | Entier | Sélectionne la sortie du champ local. |
| `--macro-field` | Entier | Sélectionne la sortie du champ macroscopique. |
| `--microscope` | Entier | Sélectionne un mode microscope. |
| `--microscope-type` | Entier | Sélectionne le type de microscope. |
| `--na` | Réel | Ouverture numérique. |
| `--magnification` | Réel | Grossissement optique. |
| `--lens-z` | Réel | Position z de la lentille. |
| `--illumination-na` | Réel | Ouverture numérique de l’illumination. |
| `--speckle-seed` | Entier | Graine du générateur aléatoire speckle. |

Les options et leurs noms sont communs aux deux pilotes. Les entiers sont des
sélecteurs transmis au programme; les plages indiquées dans ce tableau
s’appliquent aux deux interfaces.

## Options étendues communes aux deux pilotes

Ces options sont reconnues par les deux pilotes et sont appliquées aux
paramètres correspondants de `cdmlib`.

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
## Extensions de validation propres au pilote C++

Les options de calcul standard restent communes. Le pilote C++ conserve en plus
les commutateurs de validation `--repeat-calculation N` (1 à 10) et
`--second-grid-size N` (à utiliser avec `--repeat-calculation` supérieur ou égal
à 2). Il accepte aussi `--local` comme alias de `--local-field` et `--macro`
comme alias de `--macro-field`. Ces extensions ne font pas partie de l’interface
commune de calcul. Les deux commutateurs de validation s’écrivent avec la valeur
séparée par un espace.

## Arguments positionnels historiques du pilote Fortran

Les sept arguments positionnels historiques sont acceptés par les deux
pilotes :

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
./test_main_f90_cdmlib_f90_cuda_float \
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
