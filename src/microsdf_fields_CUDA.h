#ifndef IFDDA_MICROSDF_FIELDS_CUDA_H
#define IFDDA_MICROSDF_FIELDS_CUDA_H

#include "cuda_data.h"
#include "kernel.h"

#include <complex>
#include <cstddef>

extern "C" int ifdda_microsdf_expand_scalar_fields_c99(const std::complex<CU_REAL>* polarization,
                                                       const std::complex<CU_REAL>* scalar_local,
                                                       const std::complex<CU_REAL>* ux,
                                                       const std::complex<CU_REAL>* uy,
                                                       const std::complex<CU_REAL>* uz,
                                                       size_t dipole_count,
                                                       std::complex<CU_REAL>* xi,
                                                       std::complex<CU_REAL>* scattered,
                                                       std::complex<CU_REAL>* local);

extern "C" int ifdda_microsdf_expand_scalar_fields_device_c99(
    const CUCOMPLEX* device_polarization,
    const CUCOMPLEX* device_scalar_local,
    std::complex<CU_REAL> ux,
    std::complex<CU_REAL> uy,
    std::complex<CU_REAL> uz,
    size_t dipole_count,
    cuda_data::DarkFieldMicroscopeDeviceBuffers* buffers);

extern "C" int
ifdda_microsdf_plane_wave_device_c99(const void* host_incident_key,
                                     const CU_REAL* xs,
                                     const CU_REAL* ys,
                                     const CU_REAL* zs,
                                     size_t dipole_count,
                                     CU_REAL k0,
                                     CU_REAL kx,
                                     CU_REAL ky,
                                     std::complex<CU_REAL> amplitude,
                                     CU_REAL pp,
                                     cuda_data::MicroscopeIncidentDeviceBuffers* buffers);

#endif
