#ifndef IFDDA_CDMLIB_GEOMETRY_ABI_HPP
#define IFDDA_CDMLIB_GEOMETRY_ABI_HPP

#include "cdmlib_cpp_precision.hpp"

// Host-side geometry setup entry point shared by CPU and CUDA geometry paths.
extern "C" void polaepsto0_(COMPLEX *polarizability, COMPLEX *epsilon,
                           int *dipole_map, const int *isotropic, const int *nmax);

extern "C" void initnrbc_(const REAL *k0, REAL *lcos, REAL *lnrbc, REAL *lpml,
                          REAL *taillemax, const int *nnnr, int *nrbc, int *npml,
                          int *pmlwrk, REAL *aretecube);

// Shared host-coordinate lookup used by the CPU and CUDA geometry paths.
extern "C" void dipoleinside_(REAL *xdip, REAL *ydip, REAL *zdip,
                              const REAL *xs, const REAL *ys, const REAL *zs,
                              const REAL *aretecube, const int *nmax,
                              const int *nbsphere);

#endif // IFDDA_CDMLIB_GEOMETRY_ABI_HPP
