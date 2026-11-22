#ifndef IFDDA_MICROSCONFOCAL_CUDA_H
#define IFDDA_MICROSCONFOCAL_CUDA_H

#include "cdmlib_cpp_abi.hpp"
#include "kernel.h"

#include <cstddef>

extern "C" int microsconfocal_validate_c99(const int* nfft2d,
                                           const REAL* nmtom,
                                           const REAL* numaperk,
                                           const REAL* deltak,
                                           const int* nside,
                                           const REAL* stepzin,
                                           const REAL* zin,
                                           const REAL* zsupin,
                                           REAL* stepz,
                                           REAL* zinf,
                                           REAL* zsup,
                                           int* nplanz,
                                           int* nplanze,
                                           int* nstop,
                                           char* infostr,
                                           std::size_t infostr_len);

extern "C" int microsconfocal_incident_field_c99(const int* nbsphere,
                                                 const REAL* k0,
                                                 const REAL* kxinc,
                                                 const REAL* kyinc,
                                                 const REAL* psi,
                                                 const COMPLEX* E0,
                                                 const REAL* xs,
                                                 const REAL* ys,
                                                 const REAL* zs,
                                                 COMPLEX* FF0,
                                                 COMPLEX* XI,
                                                 COMPLEX* Ax,
                                                 COMPLEX* Ay,
                                                 COMPLEX* Az,
                                                 REAL* kzinc);

extern "C" int microsconfocal_classical_diffraction_c99(const int* nfft2d,
                                                        const int* imaxk0,
                                                        const int* nside,
                                                        const int* nbsphere,
                                                        const REAL* k0,
                                                        const REAL* numaperk,
                                                        const REAL* deltakx,
                                                        const REAL* deltaky,
                                                        const REAL* xs,
                                                        const REAL* ys,
                                                        const REAL* zs,
                                                        const COMPLEX* field,
                                                        COMPLEX* diffraction);

extern "C" int microsconfocal_classical_diffraction_device_c99(const int* nfft2d,
                                                               const int* imaxk0,
                                                               const int* nside,
                                                               const int* nbsphere,
                                                               const REAL* k0,
                                                               const REAL* numaperk,
                                                               const REAL* deltakx,
                                                               const REAL* deltaky,
                                                               const REAL* xs,
                                                               const REAL* ys,
                                                               const REAL* zs,
                                                               const COMPLEX* field,
                                                               const CUCOMPLEX** diffraction);

#endif
