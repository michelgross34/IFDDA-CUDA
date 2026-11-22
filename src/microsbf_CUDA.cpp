#include "cdmlib_cpp_abi.hpp"
#define IFDDA_CUDA_CALLERS_MICROSBF_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_MICROSBF_CUDA
#include "cuda_data.h"
#include "cuda_solver_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"
#include <algorithm>
#include <array>
#include <climits>
#include <vector>
#define I (COMPLEX(0.0, 1.0))
/*
 * C99 implementation of cdmlib/microsbf.f90 and shared negative-side engine
 * used by microsbfneg.c99.  The ABI entry point, nrig field dispatch,
 * diffraction/image path, and ASCII/HDF output paths are linked from CMake.
 * Keep broader validation coverage current in CONVERSION_PROGRESS.md.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//-----------------------------------------------------------
static int microsbf_polarization_cuda_impl(const int *nbsphere, const int *nmax,
    const int *niso, const COMPLEX *polarisa, const COMPLEX *local_field,
    const CUCOMPLEX *device_local_field, COMPLEX *scattered_field)
{
    if (!nbsphere || !nmax || !niso || !polarisa ||
        (!local_field && !device_local_field) ||
        !scattered_field || *nbsphere < 0 || *nmax < *nbsphere ||
        (*niso != 1 && *niso != 3))
        return 1;
    const size_t field_count = size_t(3) * size_t(*nbsphere);
    const size_t alpha_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
    if (field_count == 0)
        return 0;
    std::vector<CUCOMPLEX> alpha(alpha_count);
    std::vector<CUCOMPLEX> local;
    for (size_t i = 0; i < alpha_count; ++i)
        alpha[i] = CUCOMPLEX{CU_REAL(std::real(polarisa[i])),
                             CU_REAL(std::imag(polarisa[i]))};
    if (!device_local_field) {
        local.resize(field_count);
        for (size_t i = 0; i < field_count; ++i)
            local[i] = CUCOMPLEX{CU_REAL(std::real(local_field[i])),
                                 CU_REAL(std::imag(local_field[i]))};
    }
    cuda_data::DarkFieldMicroscopeDeviceBuffers buffers{};
    if (!cuda_data::get_device_dark_field_microscope_buffers(
            0, alpha_count, field_count, 0, 0, field_count, 0,
            field_count, 0, &buffers) || !buffers.polarization ||
        !buffers.scalar_local || !buffers.detector_planes)
        return 1;
    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.polarization, alpha.data(), alpha_count * sizeof(CUCOMPLEX),
        cuda_utils::TransferCategory::Material);
    if (status == cudaSuccess) {
        if (device_local_field)
            status = cuda_data::copy_device_to_device(
                buffers.scalar_local, device_local_field,
                field_count * sizeof(CUCOMPLEX));
        else
            status = cuda_data::copy_host_to_device(
                buffers.scalar_local, local.data(),
                field_count * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::SolverVector);
    }
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_device(
            buffers.detector_planes, buffers.scalar_local,
            field_count * sizeof(CUCOMPLEX));
    if (status == cudaSuccess)
        status = polarization_inplace_wrapper(
            buffers.polarization, buffers.detector_planes,
            *niso, *nmax, *nbsphere);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            scattered_field, buffers.detector_planes,
            field_count * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result);
    return status == cudaSuccess ? 0 : 2;
}

//-----------------------------------------------------------
extern "C" int microsbf_polarization_fields_cuda(
    const int *nbsphere, const int *nmax,
    const int *niso, const COMPLEX *polarisa, const COMPLEX *local_field,
    COMPLEX *scattered_field)
{
    return microsbf_polarization_cuda_impl(nbsphere, nmax, niso, polarisa,
        local_field, nullptr, scattered_field);
}

//-----------------------------------------------------------
static int microsbf_polarization_cuda_resident(const int *nbsphere,
    const int *nmax, const int *niso, const COMPLEX *polarisa,
    const CUCOMPLEX *local_field, COMPLEX *scattered_field)
{
    return microsbf_polarization_cuda_impl(nbsphere, nmax, niso, polarisa,
        nullptr, local_field, scattered_field);
}






//-------------------------------------------

static void microsbf_set_info(char *dst, size_t length, const char *message)
{
    size_t n = strlen(message);
    if (n > length) n = length;
    memset(dst, ' ', length);
    memcpy(dst, message, n);
}

//-----------------------------------------------------------
static int microsbf_plane_wave_batch(
    const cuda_data::MicroscopeIncidentDeviceBuffers *buffers,
    int dipole_count, const REAL *xs, const REAL *ys, const REAL *zs,
    const REAL *k0, const COMPLEX *amplitude, const REAL *ss, const REAL *pp,
    const REAL *kx, const REAL *ky, COMPLEX *incident, COMPLEX *initial,
    int *nstop, char *infostr, size_t infostr_length,
    CUCOMPLEX *host_field, bool publish_host_fields)
{
    if (!buffers || !buffers->xs || !buffers->ys || !buffers->zs ||
        !buffers->incident || dipole_count <= 0 || !xs || !ys || !zs ||
        !k0 || !amplitude || !ss || !pp || !kx || !ky || !incident ||
        !initial || !nstop || !infostr || (publish_host_fields && !host_field))
        return 1;

    const REAL zero = REAL(0);
    COMPLEX ex0{}, ey0{}, ez0{};
    ondeplanekxky_(&zero, &zero, &zero, k0, amplitude, ss, pp, kx, ky,
                   &ex0, &ey0, &ez0, nstop, infostr, infostr_length);
    if (*nstop != 0)
        return 2;

    const REAL transverse_squared = (*kx) * (*kx) + (*ky) * (*ky);
    const REAL kz = std::sqrt(std::max(REAL(0), (*k0) * (*k0) - transverse_squared));
    if (incident_plane_wave_wrapper(buffers->xs, buffers->ys, buffers->zs,
            buffers->incident, dipole_count, *kx, *ky, COMPLEX(kz, REAL(0)),
            ex0, ey0, ez0) != cudaSuccess)
        return 3;

    if (!publish_host_fields)
        return 0;
    if (cuda_data::copy_device_to_host(host_field, buffers->incident,
            size_t(3) * size_t(dipole_count) * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::IncidentField) != cudaSuccess)
        return 4;

    const size_t value_count = size_t(3) * size_t(dipole_count);
    for (size_t index = 0; index < value_count; ++index) {
        incident[index] = COMPLEX(host_field[index].x, host_field[index].y);
        initial[index] = incident[index];
    }
    return 0;
}

/* C99 translation of one incidence of the Fortran nrig=1 path.  It accepts
 * the same column-major arrays and one-based Fortran ABI scalar values; all
 * pointers are caller-owned just as in the original routine. */
//-------------------------------------------
static int microsbf_nrig1_fields_c99(
    const int *nbsphere, const int *nbsphere3, const int *nmax,
    const int *niso, const REAL *xs, const REAL *ys, const REAL *zs,
    const REAL *k0, const COMPLEX *E0, const REAL *ss,
    const REAL *pp, const REAL *kxinc, const REAL *kyinc,
    const COMPLEX *polarisa, COMPLEX *FF0,
    COMPLEX *XI, COMPLEX *FFloc, COMPLEX *FF,
    int *nstop, char *infostr, size_t infostr_len)
{
    if (!nbsphere || !nbsphere3 || !nmax || !niso || !xs || !ys || !zs ||
        !k0 || !E0 || !ss || !pp || !kxinc || !kyinc || !polarisa ||
        !FF0 || !XI || !FFloc || !FF || !nstop || !infostr ||
        *nbsphere < 0 || *nbsphere3 < 0 ||
        *nbsphere3 > 3 * *nbsphere) {
        if (nstop) *nstop = 1;
        if (infostr && infostr_len) {
            memset(infostr, ' ', infostr_len);
            memcpy(infostr, "Invalid microsbf C99 arguments",
                   infostr_len < 30 ? infostr_len : 30);
        }
        return 1;
    }

    *nstop = 0;
    const size_t field_count = size_t(3) * size_t(*nbsphere);
    memcpy(XI, FF0, field_count * sizeof(COMPLEX));
    memcpy(FFloc, FF0, size_t(*nbsphere3) * sizeof(COMPLEX));

    const int status = microsbf_polarization_fields_cuda(
        nbsphere, nmax, niso, polarisa, FFloc, FF);
    if (status != 0) {
        *nstop = 1;
        if (infostr_len) {
            memset(infostr, ' ', infostr_len);
            memcpy(infostr, "CUDA polarization failed", infostr_len < 24 ? infostr_len : 24);
        }
    }
    return status;
}

//-------------------------------------------
extern "C" int microsbf_nrig2_fields_c99(
    const int *nbsphere, const int *nmax, const int *niso,
    const COMPLEX *epsilon, const COMPLEX *polarisa,
    const COMPLEX *FF0, const REAL *aretecube, const REAL *k0,
    COMPLEX *FFloc, COMPLEX *FF)
{
    if (!nbsphere || !nmax || !niso || !epsilon || !polarisa || !FF0 ||
        !aretecube || !k0 || !FFloc || !FF || *nbsphere < 0 || *nmax <= 0 ||
        *nbsphere > *nmax || (*niso != 1 && *niso != 3))
        return 1;

    const int nsens = -1;
    const size_t dipoles = static_cast<size_t>(*nbsphere);
    const size_t field_count = 3u * dipoles;
    const size_t tensor_count = static_cast<size_t>(*nmax) *
        static_cast<size_t>(*niso) * static_cast<size_t>(*niso);
    std::vector<CUCOMPLEX> epsilon_device;
    std::vector<CUCOMPLEX> incident_device;
    std::vector<CUCOMPLEX> local_x;
    std::vector<CUCOMPLEX> local_y;
    std::vector<CUCOMPLEX> local_z;
    std::vector<CU_REAL> field_magnitude;
    try {
        epsilon_device.resize(tensor_count);
        incident_device.resize(field_count);
        local_x.resize(dipoles);
        local_y.resize(dipoles);
        local_z.resize(dipoles);
        field_magnitude.resize(dipoles);
    } catch (...) {
        return 1;
    }

    for (size_t i = 0; i < tensor_count; ++i)
        epsilon_device[i] = CUCOMPLEX{static_cast<CU_REAL>(epsilon[i].real()),
                                      static_cast<CU_REAL>(epsilon[i].imag())};
    for (size_t i = 0; i < field_count; ++i)
        incident_device[i] = CUCOMPLEX{static_cast<CU_REAL>(FF0[i].real()),
                                       static_cast<CU_REAL>(FF0[i].imag())};

    int nstop = 0;
    if (dipoles != 0)
        ifdda_local_macro_batch_cuda_(epsilon_device.data(), incident_device.data(),
            local_x.data(), local_y.data(), local_z.data(), field_magnitude.data(),
            nmax, niso, nbsphere, aretecube, k0, &nsens, &nstop);
    if (nstop != 0)
        return 1;

    for (size_t dip = 0; dip < dipoles; ++dip) {
        const size_t off = 3u * dip;
        FFloc[off] = COMPLEX(local_x[dip].x, local_x[dip].y);
        FFloc[off + 1] = COMPLEX(local_y[dip].x, local_y[dip].y);
        FFloc[off + 2] = COMPLEX(local_z[dip].x, local_z[dip].y);
    }
    return microsbf_polarization_fields_cuda(
        nbsphere, nmax, niso, polarisa, FFloc, FF);
}

/* Device-resident nrig=3 Born update and nrig=4 Rytov local-field update. */
//-------------------------------------------
static int microsbf_nrig34_fields_c99(bool rytov,
    const int *nbsphere, const int *nmax, const int *niso,
    const int *ntotalm, const int *ntotal, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *tabdip,
    const COMPLEX *tensor_xx, const COMPLEX *tensor_xy,
    const COMPLEX *tensor_xz, const COMPLEX *tensor_yy,
    const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    const int64_t *planf, const int64_t *planb,
    const COMPLEX *polarisa, const COMPLEX *FF0,
    COMPLEX *XI, COMPLEX *XR, COMPLEX *FF,
    COMPLEX *FFloc)
{
    if (!nbsphere || !nmax || !niso || !ntotalm || !ntotal || !ndipole ||
        !nxm || !nym || !nzm || !nx || !ny || !nz || !nx2 || !ny2 ||
        !nxy2 || !nz2 || !tabdip || !tensor_xx || !tensor_xy || !tensor_xz ||
        !tensor_yy || !tensor_yz || !tensor_zz ||
        !polarisa || !FF0 || !XI || !XR || !FF || !FFloc || *nbsphere <= 0 ||
        *nmax < *nbsphere || (*niso != 1 && *niso != 3) ||
        *ntotal <= 0 || *nxm <= 0 || *nym <= 0 || *nzm <= 0)
        return 1;

    const size_t dipoles = size_t(*nbsphere);
    const size_t maximum = size_t(*nmax);
    if (maximum > std::numeric_limits<size_t>::max() / 3u ||
        maximum > std::numeric_limits<size_t>::max() / size_t(*niso) /
            size_t(*niso) || size_t(*ntotal) > size_t(INT_MAX) / 3u)
        return 1;
    const size_t field_count = 3u * dipoles;
    const size_t field_capacity = 3u * maximum;
    const size_t alpha_count = maximum * size_t(*niso) * size_t(*niso);
    const size_t tensor_count = (size_t(*nxm) + 1u) * (size_t(*nym) + 1u) *
        (size_t(*nzm) + 1u);
    if (tensor_count == 0)
        return 1;
    CUCOMPLEX *product_vectors = get_device_vectors(3u * size_t(*ntotal));
    if (!product_vectors)
        return 1;

    std::vector<CUCOMPLEX> incident(field_capacity,
        CUCOMPLEX{CU_REAL(0), CU_REAL(0)});
    std::vector<CUCOMPLEX> alpha(alpha_count);
    const void *host_tensor_keys[6] = {
        tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz, tensor_zz};
    GreenTensorDevicePointers device_green = find_device_green_for_host_keys(
        host_tensor_keys, tensor_count);
    if (!device_green)
        return 1;
    const CUCOMPLEX *device_tensors[6] = {
        device_green[0], device_green[1], device_green[2],
        device_green[3], device_green[4], device_green[5]};
    for (size_t i = 0; i < field_count; ++i)
        incident[i] = CUCOMPLEX{CU_REAL(std::real(FF0[i])),
                                CU_REAL(std::imag(FF0[i]))};
    for (size_t i = 0; i < alpha_count; ++i)
        alpha[i] = CUCOMPLEX{CU_REAL(std::real(polarisa[i])),
                             CU_REAL(std::imag(polarisa[i]))};
    cuda_data::DarkFieldMicroscopeDeviceBuffers buffers{};
    if (!cuda_data::get_device_dark_field_microscope_buffers(
            0, alpha_count, field_capacity, field_capacity, 0, field_capacity,
            field_capacity, field_capacity, field_capacity, &buffers) ||
        !buffers.polarization || !buffers.scalar_local || !buffers.image_planes ||
        !buffers.detector_planes || !buffers.solver_fields ||
        !buffers.scalar_solution || !buffers.image_scratch)
        return 1;

    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.polarization, alpha.data(), alpha_count * sizeof(CUCOMPLEX),
        cuda_utils::TransferCategory::Material);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.scalar_solution, incident.data(),
            field_capacity * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_device(
            buffers.detector_planes, buffers.scalar_solution,
            field_capacity * sizeof(CUCOMPLEX));
    if (status == cudaSuccess)
        status = polarization_inplace_wrapper(
            buffers.polarization, buffers.detector_planes, *niso, *nmax, *nbsphere);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_device(
            buffers.solver_fields, buffers.detector_planes,
            field_capacity * sizeof(CUCOMPLEX));
    if (status == cudaSuccess)
        status = complex_vector_scale_wrapper(
            buffers.solver_fields, buffers.solver_fields,
            CUCOMPLEX{CU_REAL(-1), CU_REAL(0)}, int(field_capacity));
    if (status == cudaSuccess) {
        if (rytov)
            status = cudaMemset(buffers.image_scratch, 0,
                                field_capacity * sizeof(CUCOMPLEX));
        else
            status = cuda_data::copy_device_to_device(
                buffers.image_scratch, buffers.scalar_solution,
                field_capacity * sizeof(CUCOMPLEX));
    }
    if (status != cudaSuccess)
        return 2;

    produitfftmatvect3host_(device_tensors[0], device_tensors[1],
        device_tensors[2], device_tensors[3], device_tensors[4],
        device_tensors[5], product_vectors,
        product_vectors + size_t(*ntotal),
        product_vectors + 2u * size_t(*ntotal), tabdip, ntotalm, ntotal,
        nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,
        buffers.solver_fields, buffers.image_scratch, nullptr, nullptr);
    status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        if (rytov)
            status = rytov_reference_local_wrapper(
                buffers.scalar_solution, buffers.image_scratch,
                buffers.scalar_local, *nbsphere);
        else
            status = cuda_data::copy_device_to_device(
                buffers.scalar_local, buffers.image_scratch,
                field_capacity * sizeof(CUCOMPLEX));
    }
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_device(
            buffers.image_planes, buffers.scalar_local,
            field_capacity * sizeof(CUCOMPLEX));
    if (status == cudaSuccess)
        status = polarization_inplace_wrapper(
            buffers.polarization, buffers.image_planes, *niso, *nmax, *nbsphere);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            XI, buffers.solver_fields, field_count * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::Result);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            XR, buffers.image_scratch, field_count * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::Result);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            FFloc, buffers.scalar_local, field_count * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::Result);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            FF, buffers.image_planes, field_count * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::Result);
    return status == cudaSuccess ? 0 : 3;
}

//-----------------------------------------------------------
extern "C" int microsbf_nrig4_fields_c99(
    const int *nbsphere, const int *nmax, const int *niso,
    const int *ntotalm, const int *ntotal, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *tabdip,
    const COMPLEX *tensor_xx, const COMPLEX *tensor_xy,
    const COMPLEX *tensor_xz, const COMPLEX *tensor_yy,
    const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    const int64_t *planf, const int64_t *planb,
    const COMPLEX *polarisa, const COMPLEX *FF0,
    COMPLEX *XI, COMPLEX *XR, COMPLEX *FF,
    COMPLEX *FFloc)
{
    return microsbf_nrig34_fields_c99(true, nbsphere, nmax, niso, ntotalm,
        ntotal, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,
        tabdip, tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz,
        tensor_zz, vectx, vecty, vectz, planf, planb, polarisa, FF0,
        XI, XR, FF, FFloc);
}

//-----------------------------------------------------------
extern "C" int microsbf_nrig3_fields_c99(
    const int *nbsphere, const int *nmax, const int *niso,
    const int *ntotalm, const int *ntotal, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *tabdip,
    const COMPLEX *tensor_xx, const COMPLEX *tensor_xy,
    const COMPLEX *tensor_xz, const COMPLEX *tensor_yy,
    const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    const int64_t *planf, const int64_t *planb,
    const COMPLEX *polarisa, const COMPLEX *FF0,
    COMPLEX *XI, COMPLEX *XR, COMPLEX *FF,
    COMPLEX *FFloc)
{
    return microsbf_nrig34_fields_c99(false, nbsphere, nmax, niso, ntotalm,
        ntotal, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,
        tabdip, tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz,
        tensor_zz, vectx, vecty, vectz, planf, planb, polarisa, FF0,
        XI, XR, FF, FFloc);
}

/* Translation of the nrig=5 macroscopic-field Rytov approximation. */
//-------------------------------------------
static int microsbf_nrig5_fields_c99(
    const int *nbsphere, const int *nmax, const int *niso,
    const int *ntotalm, const int *ntotal, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *tabdip,
    const COMPLEX *tensor_xx, const COMPLEX *tensor_xy,
    const COMPLEX *tensor_xz, const COMPLEX *tensor_yy,
    const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    const int64_t *planf, const int64_t *planb,
    const COMPLEX *polarisa, const COMPLEX *epsilon,
    const COMPLEX *eps0, const REAL *aretecube, const REAL *k0,
    const COMPLEX *FF0, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *FF, COMPLEX *FFloc,
    const cuda_data::LocalMacroDeviceBuffers *batch_buffers,
    const cuda_data::DarkFieldMicroscopeDeviceBuffers *device_buffers,
    const CUCOMPLEX *device_incident, GreenTensorDeviceView device_green,
    CUCOMPLEX *device_fft_vectors, CU_INT *device_tabdip, cufftHandle fft_plan,
    CUCOMPLEX *host_staging)
{
    const int nsens = -1;
    if (!batch_buffers || !device_buffers || !device_incident ||
        !batch_buffers->epsilon || !batch_buffers->output_field ||
        !batch_buffers->input_field || !batch_buffers->output_field ||
        !batch_buffers->field_magnitude || !batch_buffers->status ||
        !device_buffers->polarization || !device_buffers->solver_fields ||
        !device_buffers->scalar_solution || !device_buffers->image_scratch ||
        !device_green || !device_fft_vectors || !device_tabdip || !fft_plan ||
        !host_staging)
        return 1;
    const size_t field_count = size_t(3) * size_t(*nbsphere);
    const size_t padded_count = size_t(*ntotal);
    const size_t fft_vector_count = size_t(3) * padded_count;
    cudaError_t status = cuda_data::copy_device_to_device(
        batch_buffers->input_field, device_incident,
        field_count * sizeof(CUCOMPLEX));
    if (status == cudaSuccess)
        status = local_macro_batch_wrapper(
            batch_buffers->epsilon, batch_buffers->input_field,
            batch_buffers->output_field, batch_buffers->field_magnitude,
            batch_buffers->status, *niso, *nmax, *nbsphere,
            CU_REAL(*aretecube), CU_REAL(*k0), nsens);
    if (status != cudaSuccess)
        return 1;
    status = cuda_data::copy_device_to_device(
        device_buffers->solver_fields, batch_buffers->output_field,
        field_count * sizeof(CUCOMPLEX));
    if (status == cudaSuccess)
        status = polarization_inplace_wrapper(
            device_buffers->polarization, device_buffers->solver_fields,
            *niso, *nmax, *nbsphere);
    if (status == cudaSuccess)
        status = complex_vector_scale_wrapper(
            device_buffers->solver_fields, device_buffers->solver_fields,
            CUCOMPLEX{CU_REAL(-1), CU_REAL(0)}, int(field_count));
    if (status != cudaSuccess)
        return 2;

    const CUCOMPLEX background{CU_REAL(std::real(*eps0)),
                               CU_REAL(std::imag(*eps0))};
    status = rytov_macro_delta_wrapper(
        device_incident, batch_buffers->epsilon, device_buffers->scalar_solution,
        background, *niso, *nmax, *nbsphere);
    if (status != cudaSuccess || size_t(*ntotal) > size_t(INT_MAX) / 3u)
        return 3;
    status = zero_vectors_wrapper(device_fft_vectors, int(fft_vector_count));
    if (status == cudaSuccess)
        status = gather_incident_wrapper(device_buffers->solver_fields,
            device_tabdip, device_fft_vectors, *nx, *ny, *nz, *nx2,
            *nxy2, *ntotal, *nmax);
    for (int component = 0; status == cudaSuccess && component < 3; ++component) {
        const cufftResult fft_status = CUFFT_EXECUTE(
            fft_plan, device_fft_vectors + size_t(component) * padded_count,
            device_fft_vectors + size_t(component) * padded_count, CUFFT_INVERSE);
        if (fft_status != CUFFT_SUCCESS)
            status = cudaErrorUnknown;
    }
    if (status == cudaSuccess)
        status = multiply_green_tensor_wrapper(
            device_green, device_fft_vectors, *nx, *ny, *nz,
            *nx2, *ny2, *nz2,
            size_t(*nxm + 1) * size_t(*nym + 1) * size_t(*nzm + 1));
    for (int component = 0; status == cudaSuccess && component < 3; ++component) {
        const cufftResult fft_status = CUFFT_EXECUTE(
            fft_plan, device_fft_vectors + size_t(component) * padded_count,
            device_fft_vectors + size_t(component) * padded_count, CUFFT_FORWARD);
        if (fft_status != CUFFT_SUCCESS)
            status = cudaErrorUnknown;
    }
    if (status == cudaSuccess)
        status = update_output_wrapper(
            device_fft_vectors, device_tabdip, device_buffers->scalar_solution,
            *nx, *ny, *nz, *nx2, *nxy2, *nmax, *ntotal);
    if (status == cudaSuccess)
        status = rytov_reference_local_wrapper(
            batch_buffers->output_field, device_buffers->scalar_solution,
            device_buffers->image_scratch, *nbsphere);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_device(
            device_buffers->solver_fields, device_buffers->image_scratch,
            field_count * sizeof(CUCOMPLEX));
    if (status == cudaSuccess)
        status = polarization_inplace_wrapper(
            device_buffers->polarization, device_buffers->solver_fields,
            *niso, *nmax, *nbsphere);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            host_staging, device_buffers->solver_fields,
            field_count * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::Result);
    if (status != cudaSuccess ||
        !cuda_utils::check_cuda(cudaDeviceSynchronize(),
            "BF nrig=5 CUDA fields", "synchronize resident Rytov calculation"))
        return 4;
    for (size_t i = 0; i < field_count; ++i)
        FF[i] = COMPLEX(host_staging[i].x, host_staging[i].y);
    (void)FFloc;
    (void)XI;
    (void)XR;
    (void)vectx;
    (void)vecty;
    (void)vectz;
    (void)planf;
    (void)planb;
    (void)tensor_xx;
    (void)tensor_xy;
    (void)tensor_xz;
    (void)tensor_yy;
    (void)tensor_yz;
    (void)tensor_zz;
    (void)tabdip;
    (void)ntotalm;
    (void)ndipole;
    return 0;
}

//-------------------------------------------
extern "C" int microsbf_nrig5_fields_host_c99(
    const int *nbsphere, const int *nmax, const int *niso,
    const int *ntotalm, const int *ntotal, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *tabdip,
    const COMPLEX *tensor_xx, const COMPLEX *tensor_xy,
    const COMPLEX *tensor_xz, const COMPLEX *tensor_yy,
    const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    const int64_t *planf, const int64_t *planb,
    const COMPLEX *polarisa, const COMPLEX *epsilon,
    const COMPLEX *eps0, const REAL *aretecube, const REAL *k0,
    const COMPLEX *FF0, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *FF, COMPLEX *FFloc)
{
    if (!nbsphere || !nmax || !niso || !ntotal || !nx || !ny || !nz ||
        !nxm || !nym || !nzm || !nx2 || !ny2 || !nz2 || !tabdip ||
        !polarisa || !epsilon || !eps0 || !aretecube || !k0 || !FF0 ||
        !FF || !FFloc || *nbsphere <= 0 || *nmax < *nbsphere ||
        *ntotal <= 0 || *nx <= 0 || *ny <= 0 || *nz <= 0 ||
        *nxm < 0 || *nym < 0 || *nzm < 0 || *nx2 <= 0 || *ny2 <= 0 ||
        *nz2 <= 0 || (*niso != 1 && *niso != 3))
        return 1;

    const size_t dipole_count = size_t(*nbsphere);
    const size_t material_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
    const size_t tensor_x = size_t(*nxm) + 1u;
    const size_t tensor_y = size_t(*nym) + 1u;
    const size_t tensor_z = size_t(*nzm) + 1u;
    if (dipole_count > size_t(INT_MAX) / 3u ||
        size_t(*ntotal) > size_t(INT_MAX) / 3u ||
        tensor_x > SIZE_MAX / tensor_y ||
        tensor_x * tensor_y > SIZE_MAX / tensor_z)
        return 1;
    const size_t tensor_count = tensor_x * tensor_y * tensor_z;
    const size_t field_count = 3u * dipole_count;
    const size_t mesh_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
    const size_t fft_vector_count = 3u * size_t(*ntotal);
    const COMPLEX *host_tensors[6] = {
        tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz, tensor_zz};
    if (!host_tensors[0] || !host_tensors[1] || !host_tensors[2] ||
        !host_tensors[3] || !host_tensors[4] || !host_tensors[5])
        return 1;

    std::vector<CUCOMPLEX> epsilon_device, polarization_device, host_incident;
    try {
        epsilon_device.resize(material_count);
        polarization_device.resize(material_count);
        host_incident.resize(field_count);
        for (size_t i = 0; i < material_count; ++i) {
            epsilon_device[i] = CUCOMPLEX{
                CU_REAL(std::real(epsilon[i])), CU_REAL(std::imag(epsilon[i]))};
            polarization_device[i] = CUCOMPLEX{
                CU_REAL(std::real(polarisa[i])), CU_REAL(std::imag(polarisa[i]))};
        }
    } catch (...) {
        return 1;
    }

    if (!initialize_cuda_device())
        return 1;
    cuda_data::LocalMacroDeviceBuffers macro_buffers{};
    cuda_data::DarkFieldMicroscopeDeviceBuffers solver_buffers{};
    cuda_data::MicroscopeIncidentDeviceBuffers incident_buffers{};
    if (!cuda_data::get_device_local_macro_buffers(
            material_count, dipole_count, &macro_buffers) ||
        !cuda_data::get_device_dark_field_microscope_buffers(
            0, material_count, 0, 0, 0, 0, field_count, field_count,
            field_count, &solver_buffers) ||
        !cuda_data::get_device_microscope_incident_buffers(
            dipole_count, &incident_buffers))
        return 1;

    cudaError_t status = cuda_data::copy_host_to_device(
        macro_buffers.epsilon, epsilon_device.data(),
        material_count * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Material);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            solver_buffers.polarization, polarization_device.data(),
            material_count * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Material);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_complex_to_device(
            incident_buffers.incident, FF0, field_count,
            cuda_utils::TransferCategory::IncidentField);
    const void *green_keys[6] = {
        host_tensors[0], host_tensors[1], host_tensors[2],
        host_tensors[3], host_tensors[4], host_tensors[5]};
    GreenTensorDevicePointers device_green = find_device_green_for_host_keys(green_keys, tensor_count);
    if (status == cudaSuccess && !device_green)
        status = cudaErrorInvalidDevicePointer;
    CUCOMPLEX *device_fft_vectors = get_device_vectors(fft_vector_count);
    CU_INT *device_tabdip = get_device_tabdip(tabdip, mesh_count);
    const cufftHandle fft_plan = get_fft_plan(*nx2, *ny2, *nz2);
    if (status != cudaSuccess || !device_fft_vectors || !device_tabdip || !fft_plan ||
        !macro_buffers.epsilon || !macro_buffers.input_field ||
        !macro_buffers.output_field || !macro_buffers.field_magnitude ||
        !macro_buffers.status || !solver_buffers.polarization ||
        !solver_buffers.solver_fields || !solver_buffers.scalar_solution ||
        !solver_buffers.image_scratch || !incident_buffers.incident)
        return 1;
    status = cudaMemset(macro_buffers.status, 0, sizeof(int));
    if (status != cudaSuccess)
        return 1;
    return microsbf_nrig5_fields_c99(
        nbsphere, nmax, niso, ntotalm, ntotal, ndipole, nxm, nym, nzm,
        nx, ny, nz, nx2, ny2, nxy2, nz2, tabdip,
        tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz, tensor_zz,
        vectx, vecty, vectz, planf, planb, polarisa, epsilon, eps0,
        aretecube, k0, FF0, XI, XR, FF, FFloc, &macro_buffers,
        &solver_buffers, incident_buffers.incident, green_view(device_green),
        device_fft_vectors, device_tabdip, fft_plan, host_incident.data());
}

/* Translation of the nrig=6/7 beam-propagation approximations. */
//-------------------------------------------
extern "C" int microsbf_nrig67_fields_c99(
    const int *nrig, const int *nbsphere, const int *nmax, const int *niso,
    const int *ndipole, const int *nx, const int *ny, const int *nz,
    const int *nxm, const int *nym, const int *nzm, const int *nfft2d,
    const REAL *xs, const REAL *ys, const REAL *zs,
    const REAL *aretecube, const REAL *k0, const REAL *w0,
    const COMPLEX *E0, const REAL *ss, const REAL *pp,
    const REAL *kxinc, const REAL *kyinc,
    const COMPLEX *epsilon, const COMPLEX *polarisa,
    COMPLEX *Efourierx, COMPLEX *Efouriery,
    COMPLEX *Efourierz, COMPLEX *FF0, COMPLEX *FFloc,
    COMPLEX *FF, const int64_t *plan2f, const int64_t *plan2b,
    int *nstop, char *infostr, size_t infostr_len)
{
    const REAL pi = 3.141592653589793238462643383279502884;
    const REAL zero = 0.0;
    const REAL kp = sqrt(*kxinc * *kxinc + *kyinc * *kyinc);
    const REAL theta = asin(kp / *k0) * 180.0 / pi;
    const REAL phi = atan2(*kyinc, *kxinc) * 180.0 / pi;
    const char beam[64] = "pwavelinear";
    const int resident_mode = 0;
    const size_t nfield = (size_t)3 * (size_t)*nmax;

    if (*nrig != 6 && *nrig != 7) {
        if (nstop) *nstop = 1;
        return 1;
    }
    std::fill_n(FF0, nfield, COMPLEX{});
    *nstop = 0;
    if (*nrig == 6) {
        beampropagationmacro_(xs,ys,zs,aretecube,k0,w0,E0,ss,pp,
            &theta,&phi,&zero,&zero,&zero,beam,epsilon,niso,ndipole,nx,ny,nz,
            nxm,nym,nzm,nmax,nfft2d,Efourierx,Efouriery,Efourierz,FF0,FFloc,
            FF,plan2f,plan2b,nstop,infostr,64u,infostr_len);
    } else {
        beampropagation_(xs,ys,zs,aretecube,k0,w0,E0,ss,pp,
            &theta,&phi,&zero,&zero,&zero,beam,epsilon,niso,ndipole,nx,ny,nz,
            nxm,nym,nzm,nmax,nfft2d,Efourierx,Efouriery,Efourierz,FF0,FFloc,
            FF,plan2f,plan2b,nstop,infostr,&resident_mode,64u,infostr_len);
    }
    if (*nstop == 1) return 1;
    return microsbf_polarization_fields_cuda(
        nbsphere, nmax, niso, polarisa, FFloc, FF);
}

/* Translation of the nrig=8/9 scalar inverse-solver dispatch. */
//-------------------------------------------
static int microsbf_nrig89_fields_c99(
    const int *nrig, const int *nproche,
    const int *ntotalm, const int *ntotal, const int *ldabi,
    const int *nlar, const int *nmax, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *nbsphere, const int *nlim, const int *niso,
    const COMPLEX *tensor_xx, COMPLEX *vectx,
    const int *tabdip, COMPLEX *tensor_yy_scalar,
    COMPLEX *tensor_zz_scalar, const REAL *k0,
    const COMPLEX *E0, const REAL *ss, const REAL *pp,
    const REAL *kxinc, const REAL *kyinc, const REAL *tolinit,
    REAL *tol1, int *nloop, int *ncompte, const int64_t *planf,
    const int64_t *planb, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *wrk, COMPLEX *FF, COMPLEX *FF0,
    COMPLEX *FFloc, COMPLEX *polarisa, char *methodeit,
    size_t methodeit_len, int *nstop, char *infostr, size_t infostr_len)
{
    const REAL zero = 0.0;
    COMPLEX Uinc[3];
    REAL emod, tol = *tolinit;
    int status = 0;

    if (*nrig != 8 && *nrig != 9) {
        *nstop = 1;
        return 1;
    }
    ondeplanekxky_(&zero,&zero,&zero,k0,E0,ss,pp,kxinc,kyinc,
        &Uinc[0],&Uinc[1],&Uinc[2],nstop,infostr,infostr_len);
    if (*nstop == 1) return 1;
    emod = sqrt(std::real(Uinc[0]*std::conj(Uinc[0]) + Uinc[1]*std::conj(Uinc[1]) +
                       Uinc[2]*std::conj(Uinc[2])));
    Uinc[0] /= emod;
    Uinc[1] /= emod;
    Uinc[2] /= emod;
    *ncompte = 0;
    *nloop = 0;

    if (*nproche == -1) {
        inverserigscalaire_( tensor_xx, vectx, tabdip,
            tensor_yy_scalar, tensor_zz_scalar, &Uinc[0], &Uinc[1], &Uinc[2],
            ntotalm,ntotal,ldabi,nlar,
            nmax,ndipole,nxm,nym,nzm,
            nx,ny,nz,nx2,ny2,
            nxy2,nz2,nbsphere,XI,XR,wrk,FF,FF0,FFloc,
            polarisa,niso,methodeit,&tol,tol1,nloop,ncompte,nlim,
            planf,planb,nstop,infostr,methodeit_len,infostr_len);
    } else {
        inverserigscalaire_(tensor_xx,vectx,tabdip,tensor_yy_scalar,tensor_zz_scalar,
            &Uinc[0],&Uinc[1],&Uinc[2],ntotalm,ntotal,
            ldabi,nlar,nmax,ndipole,nxm,nym,nzm,nx,ny,nz,nx2,ny2,
            nxy2,nz2,nbsphere,XI,XR,wrk,FF,FF0,FFloc,
            polarisa,niso,methodeit,&tol,tol1,nloop,ncompte,nlim,
            planf,planb,nstop,infostr,methodeit_len,infostr_len);
    }
    if (*nstop == 1) status = 1;
    return status;
}

/* Translation of the nrig=0 vector inverse-solver dispatch. */
//-------------------------------------------
extern "C" int microsbf_nrig0_fields_c99(
    const int *nproche, const int *ntotalm, const int *ntotal,
    const int *ldabi, const int *nlar, const int *nmax, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *nbsphere, const int *nbsphere3, const int *nlim, const int *niso,
    COMPLEX *tensor_xx, COMPLEX *tensor_xy,
    COMPLEX *tensor_xz, COMPLEX *tensor_yy,
    COMPLEX *tensor_yz, COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    int *tabdip, COMPLEX *XI, COMPLEX *XR, COMPLEX *wrk,
    COMPLEX *FF, COMPLEX *FF0, COMPLEX *FFloc,
    COMPLEX *polarisa, char *methodeit, const REAL *tolinit,
    REAL *tol1, int *nloop, int *ncompte,
    const int64_t *planxf, const int64_t *planxb,
    const int64_t *planyf, const int64_t *planyb,
    const int64_t *planzf, const int64_t *planzb,
    const int64_t *planf, const int64_t *planb,
    int *nstop, char *infostr, size_t methodeit_len, size_t infostr_len)
{
    REAL tol = *tolinit;
    // Native C++ microscopy must call the common CUDA solver directly.
    // The inverserig_/inverserigopt_ legacy bridge is reserved for
    // historical Fortran ABI callers. In particular the old bridge
    // converted all three host FFT scratch vectors in both directions.
    if (!ntotal || *ntotal <= 0 || size_t(*ntotal) > SIZE_MAX / 3u) {
        *nstop = 1;
        microsbf_set_info(infostr, infostr_len, "Invalid microscopy CUDA FFT workspace size");
        return 1;
    }
    CUCOMPLEX *fft_work = get_device_vectors(3u * size_t(*ntotal));
    if (!fft_work) {
        *nstop = 1;
        microsbf_set_info(infostr, infostr_len, "Unable to acquire microscopy GPU FFT workspace");
        return 1;
    }
    (void)vectx;
    (void)vecty;
    (void)vectz;
    const int optimized_product = *nproche != -1 ? 1 : 0;
    // Preserve legacy bridge entry semantics for the initial field.
    const int initial_guess_mode = optimized_product ? 1 : -1;
    const int preconditioner = 0;
    const long long plans[8] = {
        static_cast<long long>(*planxf), static_cast<long long>(*planxb),
        static_cast<long long>(*planyf), static_cast<long long>(*planyb),
        static_cast<long long>(*planzf), static_cast<long long>(*planzb),
        static_cast<long long>(*planf), static_cast<long long>(*planb)
    };
    *ncompte = 0;
    *nloop = 0;
    *nstop = 0;
    inverserig_all_cuda_driver_(
        tensor_xx,tensor_xy,tensor_xz,tensor_yy,tensor_yz,tensor_zz,
        fft_work,fft_work+*ntotal,fft_work+size_t(2)*size_t(*ntotal),
        tabdip,ntotalm,ntotal,ldabi,nlar,nmax,ndipole,
        nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,nbsphere,nbsphere3,
        XI,XR,wrk,FF,FF0,FFloc,polarisa,niso,methodeit,&tol,
        tol1,nloop,ncompte,nlim,
        &plans[0],&plans[1],&plans[2],&plans[3],
        &plans[4],&plans[5],&plans[6],&plans[7],
        &optimized_product,&initial_guess_mode,&preconditioner,
        nstop,infostr,methodeit_len,infostr_len);
    return *nstop == 1 ? 1 : 0;
}

/* Translation of the BF polarization/incidence loops and nrig=1 solver
 * branch.  The caller supplies the mask already made by micromasque_. */
//-------------------------------------------
static int microsbf_nrig12_mask_c99(
    const int *nrig, const int *nbsphere, const int *nbsphere3,
    const int *nmax, const int *niso, const int *ideltam, const int *nbitermic,
    const int *npolainc, const COMPLEX *masque,
    const REAL *xs, const REAL *ys, const REAL *zs,
    const COMPLEX *epsilon, const REAL *aretecube,
    const REAL *deltak, const REAL *k0, REAL *P0,
    const REAL *w0, REAL *irra, const COMPLEX *polarisa,
    COMPLEX *FF0, COMPLEX *XI, COMPLEX *FFloc,
    COMPLEX *FF, int *nstop, char *infostr, size_t infostr_len)
{
    COMPLEX E0 = 0.0;
    int npol = 1;
    int niter = *nbitermic;
    REAL p0_per_illumination;

    if (*npolainc == 0) {
        npol = 2;
        niter *= 2;
    }
    if (niter <= 0) {
        if (nstop) *nstop = 1;
        if (infostr && infostr_len) {
            const char message[] = "No BF illumination samples";
            const size_t n = sizeof message - 1 < infostr_len
                ? sizeof message - 1 : infostr_len;
            memset(infostr, ' ', infostr_len);
            memcpy(infostr, message, n);
        }
        return 1;
    }

    *P0 /= (REAL)niter;
    p0_per_illumination = *P0;
    irradiance_(&p0_per_illumination, w0, &E0, irra);
    *nstop = 0;

    cuda_data::MicroscopeIncidentDeviceBuffers incident_buffers{};
    std::vector<CUCOMPLEX> host_incident(size_t(3) * size_t(*nbsphere));
    if (!cuda_data::get_device_microscope_incident_buffers(
            size_t(*nbsphere), &incident_buffers) ||
        cuda_data::copy_host_to_device(incident_buffers.xs, xs,
            size_t(*nbsphere) * sizeof(REAL),
            cuda_utils::TransferCategory::Geometry) != cudaSuccess ||
        cuda_data::copy_host_to_device(incident_buffers.ys, ys,
            size_t(*nbsphere) * sizeof(REAL),
            cuda_utils::TransferCategory::Geometry) != cudaSuccess ||
        cuda_data::copy_host_to_device(incident_buffers.zs, zs,
            size_t(*nbsphere) * sizeof(REAL),
            cuda_utils::TransferCategory::Geometry) != cudaSuccess) {
        *nstop = 1;
        microsbf_set_info(infostr, infostr_len,
                          "Unable to prepare BF CUDA incident coordinates");
        return 1;
    }

    for (int ipol = 0; ipol < npol; ++ipol) {
        REAL ss, pp;
        if (*npolainc == 1) {
            ss = 1.0;
            pp = 0.0;
        } else if (*npolainc == 2) {
            ss = 0.0;
            pp = 1.0;
        } else {
            ss = ipol == 0 ? 1.0 : 0.0;
            pp = ipol == 0 ? 0.0 : 1.0;
        }
        for (int idelta = -*ideltam; idelta <= *ideltam; ++idelta) {
            for (int jdelta = -*ideltam; jdelta <= *ideltam; ++jdelta) {
                const size_t index = (size_t)(*ideltam + idelta) +
                    (size_t)(*ideltam + jdelta) *
                    (size_t)(2 * *ideltam + 1);
                const REAL kxinc = (REAL)idelta * *deltak;
                const REAL kyinc = (REAL)jdelta * *deltak;
                if (std::abs(masque[index]) <= 2.0) {
                    if (microsbf_plane_wave_batch(&incident_buffers, *nbsphere,
                            xs, ys, zs, k0, &E0, &ss, &pp, &kxinc, &kyinc,
                            FF0, XI, nstop, infostr, infostr_len,
                            host_incident.data(), true) != 0) {
                        *nstop = 1;
                        return 1;
                    }
                    if (*nrig == 1) {
                        for (int component = 0; component < *nbsphere3;
                             ++component)
                            FFloc[component] = FF0[component];
                        if (microsbf_polarization_cuda_resident(
                                nbsphere,nmax,niso,polarisa,
                                incident_buffers.incident,FF) != 0)
                            return 1;
                    } else if (*nrig == 2) {
                        if (microsbf_nrig2_fields_c99(nbsphere,nmax,niso,
                                epsilon,polarisa,FF0,aretecube,k0,FFloc,FF) != 0)
                            return 1;
                    } else {
                        if (nstop) *nstop = 1;
                        if (infostr && infostr_len) {
                            const char message[] = "nrig not in partial BF port";
                            const size_t n = sizeof message - 1 < infostr_len
                                ? sizeof message - 1 : infostr_len;
                            memset(infostr, ' ', infostr_len);
                            memcpy(infostr, message, n);
                        }
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

/* Translate the one-incidence BF diffraction, Fourier-plane filtering,
 * polarization mapping and image accumulation.  Image output serialization
 * remains in the enclosing microsbf port. */
//-------------------------------------------
int microsbf_image_c99(
    const int *nx, const int *ny, const int *nz, const int *nxm,
    const int *nym, const int *nzm, const int *nfft2d,
    int *tabfft2, const int *nbsphere, int *imaxk0,
    const int *nquicklens, const int *nside, const int *ntypemic,
    const REAL *k0, const REAL *xs, const REAL *ys, const REAL *zs,
    const REAL *aretecube, const REAL *numaper, const REAL *numaperinc,
    const REAL *numaperinc2, const REAL *zlens, const REAL *gross,
    REAL *deltakx, REAL *deltaky, const REAL *kxinc,
    const REAL *kyinc, const COMPLEX *E0, const REAL *ss,
    const REAL *pp, const COMPLEX *scattered,
    const int64_t *plan2f, const int64_t *plan2b,
    COMPLEX *Ediffkzpos, COMPLEX *Efourierx,
    COMPLEX *Efouriery, COMPLEX *Efourierz,
    COMPLEX *Efourierincx, COMPLEX *Efourierincy,
    COMPLEX *Efourierincz, COMPLEX *Eimagex,
    COMPLEX *Eimagey, COMPLEX *Eimagez,
    COMPLEX *Eimageincx, COMPLEX *Eimageincy,
    COMPLEX *Eimageincz, int *nstop, char *infostr,
    size_t infostr_len,
    const cuda_data::MicroscopeIncidentDeviceBuffers *geometry_buffers,
    const CUCOMPLEX *device_scattered,
    cuda_data::BrightFieldMicroscopeDeviceBuffers *bf_image_buffers)
{
    const int nfft = *nfft2d;
    const size_t plane = (size_t)nfft * (size_t)nfft;
    const REAL kval = *k0, k02 = kval * kval;
    int imax = *imaxk0;
    int ikxinc, jkyinc;
    // The native C++ caller retains the user-facing 0 (transmission) side.
    // The original Fortran cdmlib normalizes 0 -> +1 before diffraction.
    // Pass the normalized direction to the lens, not the raw 0 value.
    const int normalized_side = *nside < 0 ? -1 : 1;
    const REAL side = REAL(normalized_side);
    REAL dkx = *deltakx, dky = *deltaky;

    if (!bf_image_buffers || !bf_image_buffers->fft_plan ||
        !bf_image_buffers->diffraction_field || !bf_image_buffers->fourier_fields ||
        !bf_image_buffers->image_intensity || !bf_image_buffers->scattered_field) {
        *nstop = 1;
        microsbf_set_info(infostr, infostr_len,
                          "Bright-field CUDA image buffers unavailable");
        return 1;
    }
    cuda_data::BrightFieldMicroscopeDeviceBuffers& buffers = *bf_image_buffers;
    const CUCOMPLEX *diffraction_field_device = buffers.diffraction_field;

    if (*nquicklens == 1) {
        diffractefft2dlens_set_resident_output(1);
        diffractefft2dlens_(nx,ny,nz,nxm,nym,nzm,nfft2d,tabfft2,k0,
            xs,ys,zs,aretecube,Efourierx,Efouriery,Efourierz,scattered,
            &imax,&dkx,&dky,Ediffkzpos,numaper,&normalized_side,
            plan2f,plan2b,
            nstop,infostr,infostr_len);
        diffractefft2dlens_set_resident_output(0);
        if (*nstop == 1) return 1;
        *imaxk0 = imax;
        *deltakx = dkx;
        *deltaky = dky;
        diffraction_field_device = diffractefft2dlens_get_resident_output();
        if (!diffraction_field_device) {
            *nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Lens diffraction did not return a device spectrum");
            return 1;
        }
    } else {
        if (!geometry_buffers || !geometry_buffers->xs ||
            !geometry_buffers->ys || !geometry_buffers->zs || !Ediffkzpos) {
            *nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Bright-field CUDA geometry buffers unavailable");
            return 1;
        }
        const size_t dipole_count = size_t(*nbsphere);
        const size_t field_count = size_t(3) * dipole_count;
        const CUCOMPLEX *device_field = device_scattered;
        std::vector<CUCOMPLEX> host_field;
        if (!device_field) {
            try {
                host_field.resize(field_count);
            } catch (...) {
                *nstop = 1;
                microsbf_set_info(infostr, infostr_len,
                                  "Unable to stage bright-field CUDA input");
                return 1;
            }
            for (size_t i = 0; i < field_count; ++i)
                host_field[i] = CUCOMPLEX{
                    CU_REAL(std::real(scattered[i])),
                    CU_REAL(std::imag(scattered[i]))};
            if (cuda_data::copy_host_to_device(
                    buffers.scattered_field, host_field.data(),
                    field_count * sizeof(CUCOMPLEX),
                    cuda_utils::TransferCategory::Result) != cudaSuccess) {
                *nstop = 1;
                microsbf_set_info(infostr, infostr_len,
                                  "Unable to upload bright-field CUDA input");
                return 1;
            }
            device_field = buffers.scattered_field;
        }
        const cudaError_t launch = confocal_direct_diffraction_wrapper(
            device_field, geometry_buffers->xs, geometry_buffers->ys,
            geometry_buffers->zs, *nbsphere, nfft, imax, int(side),
            CU_REAL(kval), CU_REAL(*numaper), CU_REAL(*deltakx),
            CU_REAL(*deltaky), buffers.diffraction_field);
        if (!cuda_utils::check_cuda(launch, "bright-field diffraction",
                                    "launch dipole spectrum kernel")) {
            *nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Bright-field CUDA diffraction failed");
            return 1;
        }
    }
        deltakroutine_(kxinc, kyinc, deltakx, deltaky, k0,
                       &ikxinc, &jkyinc);
        COMPLEX incident_x = COMPLEX(0.0, 0.0);
        COMPLEX incident_y = COMPLEX(0.0, 0.0);
        COMPLEX incident_z = COMPLEX(0.0, 0.0);
        const REAL incident_kx = REAL(ikxinc) * *deltakx;
        const REAL incident_ky = REAL(jkyinc) * *deltaky;
        const REAL incident_transverse = incident_kx * incident_kx +
                                         incident_ky * incident_ky;
        if (std::abs(ikxinc) <= imax && std::abs(jkyinc) <= imax &&
            incident_transverse <= *numaper * *numaper &&
            incident_transverse < k02) {
            const REAL zero = 0.0;
            ondeplanekxky_(&zero, &zero, &zero, k0, E0, ss, pp,
                &incident_kx, &incident_ky, &incident_x, &incident_y,
                &incident_z, nstop, infostr, infostr_len);
            if (*nstop == 1)
                return 1;
        }
        const REAL deltakxy = *deltakx * *deltaky;
        const REAL incident_scale = deltakxy == REAL(0)
            ? REAL(0) : REAL(1) / deltakxy;
        const cudaError_t prepare_status = bright_field_prepare_fourier_wrapper(
            diffraction_field_device, buffers.fourier_fields, nfft, imax,
            CU_REAL(*deltakx), CU_REAL(*deltaky), CU_REAL(kval),
            CU_REAL(*numaper), CU_REAL(*zlens), CU_REAL(*gross), int(side),
            // The kernel compares transverse k against aperture_inner*k0;
            // pass the dimensionless inner NA (Fortran: kp >= numaperinc2*k0).
            *ntypemic, CU_REAL(*numaperinc2), ikxinc, jkyinc,
            CU_REAL(incident_scale),
            CUCOMPLEX{CU_REAL(std::real(incident_x)), CU_REAL(std::imag(incident_x))},
            CUCOMPLEX{CU_REAL(std::real(incident_y)), CU_REAL(std::imag(incident_y))},
            CUCOMPLEX{CU_REAL(std::real(incident_z)), CU_REAL(std::imag(incident_z))});
        if (!cuda_utils::check_cuda(prepare_status, "bright-field Fourier plane",
                                    "prepare fields on GPU")) {
            *nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Bright-field CUDA Fourier preparation failed");
            return 1;
        }
        if (*nside < 0 && !cuda_utils::check_cuda(
                bright_field_image_accumulate_wrapper(
                    buffers.fourier_fields, buffers.image_intensity, plane, 3, 3),
                "bright-field image", "accumulate incident Fourier intensities")) {
            *nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Bright-field CUDA incident image accumulation failed");
            return 1;
        }
        const cufftResult fft_status = CUFFT_EXECUTE(
            buffers.fft_plan, buffers.fourier_fields, buffers.fourier_fields,
            CUFFT_FORWARD);
        if (!cuda_utils::check_cufft(fft_status, "bright-field image",
                                     "execute batched 2-D FFT")) {
            *nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Bright-field CUDA FFT failed");
            return 1;
        }
        const REAL image_scale = *deltakx * *deltaky / *gross;
        if (!cuda_utils::check_cuda(fourier_to_image_rearrange_wrapper(
                buffers.fourier_fields, nfft, CU_REAL(image_scale)),
                "bright-field image", "rearrange Fourier planes") ||
            !cuda_utils::check_cuda(bright_field_image_accumulate_wrapper(
                buffers.fourier_fields, buffers.image_intensity, plane, 0,
                *nside < 0 ? 3 : 6),
                "bright-field image", "accumulate intensities")) {
            *nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Bright-field CUDA image accumulation failed");
            return 1;
        }
        return 0;

}

/* Complete GNU Fortran ABI entry point.  It runs the BF incidence scheduler,
 * selects one of the already translated local-field branches, and accumulates
 * the detector image.  Explicit arguments retain the original by-address
 * convention; hidden CHARACTER lengths are the final three ABI arguments. */
//-------------------------------------------
extern "C" void microsbf_(
    COMPLEX *FFTTENSORxx, COMPLEX *FFTTENSORxy, COMPLEX *FFTTENSORxz,
    COMPLEX *FFTTENSORyy, COMPLEX *FFTTENSORyz, COMPLEX *FFTTENSORzz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    int *ntotalm, int *ntotal, int *ldabi, int *nlar, int *nmax,
    int *nxm, int *nym, int *nzm, int *nx, int *ny, int *nz,
    int *nx2, int *ny2, int *nxy2, int *nz2, int *ndipole,
    int *nbsphere, int *nbsphere3, int *nproche, int *nrig,
    int *nfft2d, int *tabfft2, int *Tabdip, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *wrk, COMPLEX *FF, COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *polarisa,
    int *niso, COMPLEX *epsilon, char *methodeit, char *trope,
    REAL *tolinit, REAL *tol1, int *nloop, int *ncompte, int *nlim,
    REAL *xs, REAL *ys, REAL *zs, REAL *aretecube, REAL *numaper,
    REAL *numaperinc, REAL *numaperinc2, int *npolainc_in,
    int *nquicklens, COMPLEX *eps0, REAL *k0, REAL *P0, REAL *irra,
    REAL *w0, REAL *gross, REAL *zlens, COMPLEX *Eimagex, COMPLEX *Eimagey,
    COMPLEX *Eimagez, COMPLEX *Efourierx, COMPLEX *Efouriery, COMPLEX *Efourierz,
    COMPLEX *Eimageincx, COMPLEX *Eimageincy, COMPLEX *Eimageincz,
    COMPLEX *Efourierincx, COMPLEX *Efourierincy, COMPLEX *Efourierincz,
    COMPLEX *Ediffkzpos, COMPLEX *Ediffkzneg, REAL *kxy, REAL *xy,
    int *nside, COMPLEX *masque, int *nmasque, int *ideltam,
    int *imaxk0, int *nbitermic, REAL *deltak, REAL *deltakx,
    int *ntypemic, int64_t *planXf, int64_t *planXb, int64_t *planYf,
    int64_t *planYb, int64_t *planZf, int64_t *planZb, int64_t *planf, int64_t *planb,
    int64_t *plan2f, int64_t *plan2b, int *nmat, int *file_id_in,
    int *group_idmic_in, int *nstop, char *infostr,
    size_t methodeit_len, size_t trope_len, size_t infostr_len)
{
    const int nfft = *nfft2d;
    const int half = nfft / 2;
    const int nsphere = *nbsphere;
    const int nmask = *nmasque;
    const int imaxmask = *ideltam;
    const int rig = *nrig;
    const int npolainc = *npolainc_in;
    const int npol = npolainc == 0 ? 2 : 1;
    const int niter = (npolainc == 0 ? 2 : 1) * *nbitermic;
    const size_t nimage = (size_t)nfft * (size_t)nfft;
    const size_t nfield = (size_t)3 * (size_t)nsphere;
    const REAL pi = 3.141592653589793238462643383279502884;
    const REAL kval = *k0;
    const REAL k02 = kval * kval;
    COMPLEX E0 = 0.0;
    REAL deltak_y = *deltakx;
    REAL deltakxy, deltax;
    int niterii = 0;
    COMPLEX *const incident = FF0;
    COMPLEX *const initial = XI;
    COMPLEX *const real_work = XR;
    COMPLEX *const scattered = FF;
    COMPLEX *const local = FFloc;
    COMPLEX *const pol = polarisa;
    COMPLEX *const eps = epsilon;
    COMPLEX *const tensor[6] = {
        FFTTENSORxx, FFTTENSORxy,
        FFTTENSORxz, FFTTENSORyy,
        FFTTENSORyz, FFTTENSORzz};
    COMPLEX *const vec[3] = {vectx,
        vecty, vectz};
    int *const p_nstop = nstop;
    int *const p_nloop = nloop;
    int *const p_ncompte = ncompte;
    REAL *const p_P0 = P0;
    REAL *const p_irra = irra;
    const int64_t *const p_planxf = planXf;
    const int64_t *const p_planxb = planXb;
    const int64_t *const p_planyf = planYf;
    const int64_t *const p_planyb = planYb;
    const int64_t *const p_planzf = planZf;
    const int64_t *const p_planzb = planZb;
    const int64_t *const p_planf = planf;
    const int64_t *const p_planb = planb;
    const int64_t *const p_plan2f = plan2f;
    const int64_t *const p_plan2b = plan2b;

    (void)nmask; (void)Ediffkzneg; (void)file_id_in; (void)group_idmic_in;
    (void)trope;
    if (nfft <= 0 || nfft > 16384 || niter <= 0 || nsphere <= 0 ||
        imaxmask < 0 || nmask <= 0 || 2 * imaxmask + 1 > nmask) {
        *p_nstop = 1;
        microsbf_set_info(infostr, infostr_len, "Invalid BF microscopy dimensions");
        return;
    }
    if (*deltak >= kval * *numaperinc) {
        *p_nstop = 1;
        microsbf_set_info(infostr, infostr_len, "In FFT lens nfft2d too small");
        return;
    }

    std::fill_n(Eimagex, nimage, COMPLEX{});
    std::fill_n(Eimagey, nimage, COMPLEX{});
    std::fill_n(Eimagez, nimage, COMPLEX{});
    std::fill_n(Eimageincx, nimage, COMPLEX{});
    std::fill_n(Eimageincy, nimage, COMPLEX{});
    std::fill_n(Eimageincz, nimage, COMPLEX{});
    cuda_data::BrightFieldMicroscopeDeviceBuffers bf_image_buffers{};
    if (!cuda_data::get_device_bright_field_microscope_buffers(
            size_t(nsphere), nimage, nfft, &bf_image_buffers) ||
        cudaMemset(bf_image_buffers.image_intensity, 0,
                   6 * nimage * sizeof(CU_REAL)) != cudaSuccess) {
        *p_nstop = 1;
        microsbf_set_info(infostr, infostr_len,
                          "Unable to initialize BF CUDA image buffers");
        return;
    }
    *p_P0 /= (REAL)niter;
    irradiance_(p_P0, w0, &E0, p_irra);
    *p_nstop = 0;
    cuda_data::MicroscopeIncidentDeviceBuffers plane_wave_buffers{};
    std::vector<CUCOMPLEX> host_incident(nfield);
    cuda_data::LocalMacroDeviceBuffers bf_macro_buffers{};
    cuda_data::DarkFieldMicroscopeDeviceBuffers bf_device_buffers{};
    std::vector<CUCOMPLEX> epsilon_cuda;
    std::vector<CUCOMPLEX> polarization_cuda;
    GreenTensorDevicePointers bf_green_device{};
    CUCOMPLEX *bf_fft_vectors = nullptr;
    CU_INT *bf_tabdip_device = nullptr;
    cufftHandle bf_fft_plan = 0;
    if (!initialize_cuda_device() ||
        !cuda_data::get_device_microscope_incident_buffers(
            size_t(nsphere), &plane_wave_buffers) ||
        cuda_data::copy_host_to_device(plane_wave_buffers.xs, xs,
            size_t(nsphere) * sizeof(REAL),
            cuda_utils::TransferCategory::Geometry) != cudaSuccess ||
        cuda_data::copy_host_to_device(plane_wave_buffers.ys, ys,
            size_t(nsphere) * sizeof(REAL),
            cuda_utils::TransferCategory::Geometry) != cudaSuccess ||
        cuda_data::copy_host_to_device(plane_wave_buffers.zs, zs,
            size_t(nsphere) * sizeof(REAL),
            cuda_utils::TransferCategory::Geometry) != cudaSuccess) {
        *p_nstop = 1;
        microsbf_set_info(infostr, infostr_len,
                          "Unable to prepare CUDA incident field buffers");
        return;
    }
    if (rig == 5) {
        const size_t material_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
        const size_t solver_field_count = size_t(3) * size_t(*nmax);
        const size_t physical_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
        const size_t tensor_count = size_t(*nxm + 1) * size_t(*nym + 1) * size_t(*nzm + 1);
        const size_t fft_vector_count = size_t(3) * size_t(*ntotal);
        const COMPLEX *host_tensors[6] = {
            FFTTENSORxx, FFTTENSORxy, FFTTENSORxz,
            FFTTENSORyy, FFTTENSORyz, FFTTENSORzz};
        const void *green_component_keys[6] = {
            host_tensors[0], host_tensors[1], host_tensors[2],
            host_tensors[3], host_tensors[4], host_tensors[5]};
        try {
            epsilon_cuda.resize(material_count);
            polarization_cuda.resize(material_count);
            for (size_t i = 0; i < material_count; ++i)
                epsilon_cuda[i] = CUCOMPLEX{
                    CU_REAL(std::real(eps[i])), CU_REAL(std::imag(eps[i]))};
            for (size_t i = 0; i < material_count; ++i)
                polarization_cuda[i] = CUCOMPLEX{
                    CU_REAL(std::real(pol[i])), CU_REAL(std::imag(pol[i]))};
        } catch (...) {
            *p_nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Unable to stage BF microscopy CUDA arrays");
            return;
        }
        if (!cuda_data::get_device_local_macro_buffers(
                material_count, size_t(nsphere), &bf_macro_buffers) ||
            !cuda_data::get_device_dark_field_microscope_buffers(
                0, material_count, 0, 0, 0, 0, solver_field_count,
                solver_field_count, solver_field_count, &bf_device_buffers) ||
            cuda_data::copy_host_to_device(bf_macro_buffers.epsilon,
                epsilon_cuda.data(), material_count * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::Material) != cudaSuccess) {
            *p_nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Unable to initialize BF CUDA material buffers");
            return;
        }
        if (cuda_data::copy_host_to_device(bf_device_buffers.polarization,
                polarization_cuda.data(), material_count * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::Material) != cudaSuccess) {
            *p_nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Unable to initialize BF CUDA polarizability");
            return;
        }
        // The six Fortran tensor arrays are identity keys, not host values.
        // Never upload them over the resident Green tensor generated on GPU.
        bf_green_device = find_device_green_for_host_keys(
            green_component_keys, tensor_count);
        bf_fft_vectors = get_device_vectors(fft_vector_count);
        bf_tabdip_device = get_device_tabdip(Tabdip, physical_count);
        bf_fft_plan = get_fft_plan(*nx2, *ny2, *nz2);
        if (!bf_green_device || !bf_fft_vectors || !bf_tabdip_device || !bf_fft_plan) {
            *p_nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Unable to initialize BF CUDA Green workspace");
            return;
        }
        if (cudaMemset(bf_macro_buffers.status, 0, sizeof(int)) != cudaSuccess) {
            *p_nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Unable to initialize BF CUDA status buffer");
            return;
        }
    }
    for (int ipol = 0; ipol < npol; ++ipol) {
        const REAL ss = npolainc == 1 ? 1.0 : (npolainc == 2 ? 0.0 : (ipol == 0 ? 1.0 : 0.0));
        const REAL pp = npolainc == 1 ? 0.0 : (npolainc == 2 ? 1.0 : (ipol == 0 ? 0.0 : 1.0));
        for (int idelta = -imaxmask; idelta <= imaxmask; ++idelta) {
            for (int jdelta = -imaxmask; jdelta <= imaxmask; ++jdelta) {
                const size_t mask_index = (size_t)(imaxmask + idelta) +
                    (size_t)(imaxmask + jdelta) * (size_t)(2 * imaxmask + 1);
                const REAL kxinc = (REAL)idelta * *deltak;
                const REAL kyinc = (REAL)jdelta * *deltak;
                int status = 0;
                if (std::abs((masque)[mask_index]) > 2.0) continue;
                ++niterii;
                if (microsbf_plane_wave_batch(&plane_wave_buffers, nsphere,
                        xs, ys, zs, k0, &E0, &ss, &pp, &kxinc, &kyinc,
                        incident, initial, p_nstop, infostr, infostr_len,
                        host_incident.data(), rig != 5) != 0)
                    return;
                if (rig == 0) {
                    const bool incident_alias_registered =
                        nbsphere3 != nullptr && *nbsphere3 > 0 &&
                        register_gpbicg_device_b_alias(
                            incident, plane_wave_buffers.incident,
                            size_t(*nbsphere3));
                    status = microsbf_nrig0_fields_c99(nproche,
                        ntotalm,ntotal,ldabi,
                        nlar,nmax,ndipole,
                        nxm,nym,nzm,
                        nx,ny,nz,
                        nx2,ny2,nxy2,nz2,
                        nbsphere,nbsphere3,nlim,
                        niso,tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],
                        vec[0],vec[1],vec[2],Tabdip,initial,real_work,wrk,
                        scattered,incident,local,pol,methodeit,tolinit,tol1,
                        p_nloop,p_ncompte,p_planxf,p_planxb,p_planyf,p_planyb,p_planzf,p_planzb,
                        p_planf,p_planb,p_nstop,infostr,methodeit_len,infostr_len);
                    if (incident_alias_registered)
                        release_gpbicg_device_b_alias(incident);
                } else if (rig == 1) {
                    status = microsbf_nrig1_fields_c99(nbsphere,nbsphere3,
                        nmax,niso,xs,ys,
                        zs,k0,&E0,&ss,&pp,&kxinc,&kyinc,pol,
                        incident,initial,local,scattered,p_nstop,infostr,infostr_len);
                } else if (rig == 2) {
                    status = microsbf_nrig2_fields_c99(nbsphere,nmax,
                        niso,eps,pol,incident,aretecube,
                        k0,local,scattered);
                } else if (rig >= 3 && rig <= 5) {
                    if (rig == 3) status = microsbf_nrig3_fields_c99(nbsphere,nmax,
                        niso,ntotalm,ntotal,ndipole,
                        nxm,nym,nzm,nx,ny,
                        nz,nx2,ny2,nxy2,nz2,
                        Tabdip,tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],
                        vec[0],vec[1],vec[2],p_planf,p_planb,pol,incident,initial,real_work,scattered,local);
                    else if (rig == 4) status = microsbf_nrig4_fields_c99(nbsphere,nmax,
                        niso,ntotalm,ntotal,ndipole,
                        nxm,nym,nzm,nx,ny,
                        nz,nx2,ny2,nxy2,nz2,
                        Tabdip,tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],
                        vec[0],vec[1],vec[2],p_planf,p_planb,pol,incident,initial,real_work,scattered,local);
                    else status = microsbf_nrig5_fields_c99(nbsphere,nmax,
                        niso,ntotalm,ntotal,ndipole,
                        nxm,nym,nzm,nx,ny,
                        nz,nx2,ny2,nxy2,nz2,
                        Tabdip,tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],
                        vec[0],vec[1],vec[2],p_planf,p_planb,pol,eps,eps0,
                        aretecube,k0,incident,initial,real_work,scattered,local,
                        &bf_macro_buffers,&bf_device_buffers,
                        plane_wave_buffers.incident,green_view(bf_green_device),
                        bf_fft_vectors,bf_tabdip_device,bf_fft_plan,
                        host_incident.data());
                } else if (rig == 6 || rig == 7) {
                    status = microsbf_nrig67_fields_c99(nrig,nbsphere,
                        nmax,niso,ndipole,nx,
                        ny,nz,nxm,nym,nzm,
                        nfft2d,xs,ys,zs,
                        aretecube,k0,w0,&E0,&ss,&pp,
                        &kxinc,&kyinc,eps,pol,Efourierx,Efouriery,
                        Efourierz,incident,local,scattered,p_plan2f,p_plan2b,
                        p_nstop,infostr,infostr_len);
                } else if (rig == 8 || rig == 9) {
                    status = microsbf_nrig89_fields_c99(nrig,nproche,
                        ntotalm,ntotal,ldabi,nlar,
                        nmax,ndipole,nxm,nym,nzm,
                        nx,ny,nz,nx2,ny2,
                        nxy2,nz2,nbsphere,nlim,
                        niso,tensor[0],vec[0],Tabdip,tensor[3],tensor[5],
                        k0,&E0,&ss,&pp,&kxinc,&kyinc,tolinit,tol1,
                        p_nloop,p_ncompte,p_planf,p_planb,initial,real_work,wrk,
                        scattered,incident,local,pol,methodeit,methodeit_len,p_nstop,infostr,infostr_len);
                } else {
                    *p_nstop = 1;
                    microsbf_set_info(infostr,infostr_len,"Method not implemented for microscopy");
                    return;
                }
                if (status != 0 || *p_nstop == 1) return;
                if (microsbf_image_c99(nx,ny,nz,
                    nxm,nym,nzm,nfft2d,
                    tabfft2,nbsphere,imaxk0,nquicklens,
                    nside,ntypemic,k0,xs,
                    ys,zs,aretecube,numaper,
                    numaperinc,numaperinc2,zlens,
                    gross,deltakx,&deltak_y,&kxinc,&kyinc,&E0,&ss,&pp,scattered,
                    p_plan2f,p_plan2b,Ediffkzpos,Efourierx,
                    Efouriery,Efourierz,Efourierincx,
                    Efourierincy,Efourierincz,Eimagex,
                    Eimagey,Eimagez,Eimageincx,
                    Eimageincy,Eimageincz,p_nstop,infostr,infostr_len,
                    &plane_wave_buffers,
                    rig == 5 ? bf_device_buffers.solver_fields : nullptr,
                    &bf_image_buffers) != 0)
                    return;
                if (*p_nstop == -1) {
                    microsbf_set_info(infostr,infostr_len,"Calculation cancelled during iterative method");
                    return;
                }
            }
        }
    }
    if (niterii == 0) {
        *p_nstop = 1;
        microsbf_set_info(infostr,infostr_len,"No valid BF illumination samples");
        return;
    }
    if (rig == 5) {
        int macro_status = 0;
        if (cuda_data::copy_device_to_host(
                &macro_status, bf_macro_buffers.status, sizeof(int),
                cuda_utils::TransferCategory::Diagnostic) != cudaSuccess ||
            macro_status != 0) {
            *p_nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "CUDA local-macro matrix inversion failed");
            return;
        }
    }
    {
        std::vector<CU_REAL> host_image_intensity;
        try {
            host_image_intensity.resize(6 * nimage);
        } catch (...) {
            *p_nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Unable to stage BF CUDA image result");
            return;
        }
        if (microscope_intensity_sqrt_wrapper(
                bf_image_buffers.image_intensity, 6 * nimage) != cudaSuccess ||
            cuda_data::copy_device_to_host(
                host_image_intensity.data(), bf_image_buffers.image_intensity,
                6 * nimage * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Result) != cudaSuccess) {
            *p_nstop = 1;
            microsbf_set_info(infostr, infostr_len,
                              "Unable to download BF CUDA image result");
            return;
        }
        COMPLEX *image_planes[6] = {
            Eimagex, Eimagey, Eimagez,
            Eimageincx, Eimageincy, Eimageincz};
        for (size_t component = 0; component < 6; ++component)
            for (size_t pixel = 0; pixel < nimage; ++pixel)
                image_planes[component][pixel] = COMPLEX(
                    REAL(host_image_intensity[component * nimage + pixel]), 0.0);
    }
    deltax = 2.0*pi/(REAL)nfft/(*deltakx);
    deltakxy = *deltakx / kval;
    for (int i = -half; i < half; ++i) {
        (xy)[i+half] = deltax*(REAL)i*(*gross);
        (kxy)[i+half] = *deltakx*(REAL)i/kval;
    }
    if (*nmat == 0) {
        const char *names[8] = {"imagebf.mat","imagebfx.mat","imagebfy.mat","imagebfz.mat",
            "imageincbf.mat","imageincbfx.mat","imageincbfy.mat","imageincbfz.mat"};
        FILE *f[8];
        FILE *fk=fopen("kxfourier.mat","w"), *fx=fopen("ximage.mat","w");
        FILE *fkx=fopen("kxmic.mat","w"), *fky=fopen("kymic.mat","w"), *fm=fopen("masquemic.mat","w");
        if (!fk || !fx || !fkx || !fky || !fm) {
            if (fk) fclose(fk);
            if (fx) fclose(fx);
            if (fkx) fclose(fkx);
            if (fky) fclose(fky);
            if (fm) fclose(fm);
            *p_nstop=1; microsbf_set_info(infostr,infostr_len,"Unable to open microscopy output file"); return;
        }
        const int output_count = *nside < 0 ? 4 : 8;
        for (int q=0;q<output_count;++q) if((f[q]=fopen(names[q],"w"))==NULL) {
            for(int t=0;t<q;++t) fclose(f[t]);
            fclose(fk); fclose(fx); fclose(fkx); fclose(fky); fclose(fm);
            *p_nstop=1; microsbf_set_info(infostr,infostr_len,"Unable to open microscopy output file"); return;
        }
        if (*nside < 0) {
            FILE *fikx=fopen("kxincidentbf.mat","w"), *fiky=fopen("kyincidentbf.mat","w");
            if (fikx == NULL || fiky == NULL) {
                if (fikx) fclose(fikx);
                if (fiky) fclose(fiky);
                fclose(fk); fclose(fx); fclose(fkx); fclose(fky); fclose(fm);
                for (int q=0;q<output_count;++q) fclose(f[q]);
                *p_nstop=1; microsbf_set_info(infostr,infostr_len,"Unable to open microscopy output file"); return;
            }
            for (int di=-imaxmask;di<=imaxmask;++di) for (int dj=-imaxmask;dj<=imaxmask;++dj) {
                const REAL kxi=(REAL)di * *deltak;
                const REAL kyi=(REAL)dj * *deltak;
                const REAL na=kval * *numaperinc;
                if (kxi*kxi+kyi*kyi <= na*na) {
                    fprintf(fikx," % .16E\n",kxi/kval);
                    fprintf(fiky," % .16E\n",kyi/kval);
                }
            }
            fclose(fikx); fclose(fiky);
        }
        for(int i=-half;i<half;++i) fprintf(fx," % .16E\n",(xy)[i+half]);
        for(int i=-*imaxk0;i<=*imaxk0;++i) fprintf(fk," % .16E\n",*deltakx*(REAL)i/kval);
        for(int j=-imaxmask;j<=imaxmask;++j) for(int i=-imaxmask;i<=imaxmask;++i) {
            const size_t idx=(size_t)(imaxmask+i)+(size_t)(imaxmask+j)*(size_t)(2*imaxmask+1);
            const REAL kxi=(REAL)i* *deltak, kyi=(REAL)j* *deltak;
            if (j==0) fprintf(fkx," % .16E\n",kxi/kval);
            if (i==0) fprintf(fky," % .16E\n",kyi/kval);
            fprintf(fm," % .16E\n",std::abs((masque)[idx]));
        }
        for(size_t i=0;i<nimage;++i) {
            const COMPLEX ax=(Eimagex)[i], ay=(Eimagey)[i], az=(Eimagez)[i];
            const COMPLEX ix=(Eimageincx)[i], iy=(Eimageincy)[i], iz=(Eimageincz)[i];
            fprintf(f[0]," % .16E\n",sqrt(std::real(ax*ax+ay*ay+az*az)));
            fprintf(f[1]," % .16E\n",std::real(ax)); fprintf(f[2]," % .16E\n",std::real(ay)); fprintf(f[3]," % .16E\n",std::real(az));
            if (output_count == 8) {
                fprintf(f[4]," % .16E\n",sqrt(std::real(ix*ix+iy*iy+iz*iz)));
                fprintf(f[5]," % .16E\n",std::real(ix)); fprintf(f[6]," % .16E\n",std::real(iy)); fprintf(f[7]," % .16E\n",std::real(iz));
            }
        }
        fclose(fk);fclose(fx);fclose(fkx);fclose(fky);fclose(fm); for(int q=0;q<output_count;++q)fclose(f[q]);
    } else if (*nmat == 2) {
        int dims[4] = {nfft,nfft,0,0};
        int zero = 0;
        // The Fortran input ID is integer(ip); HDF5 receives integer(hid_t).
        const int64_t group_idmic_hdf5 = *group_idmic_in;
        char dataset[100];
        const int grid = 2*imaxmask+1;
        REAL *mask_real = (REAL *)malloc((size_t)nmask*(size_t)nmask*sizeof *mask_real);
        if (mask_real == NULL) {
            *p_nstop=1;
            microsbf_set_info(infostr,infostr_len,"Unable to allocate microscopy mask output");
            return;
        }
        memset(dataset,' ',sizeof dataset); memcpy(dataset,"x Image",7);
        hdf5write1d_i8_(&group_idmic_hdf5,dataset,xy,dims,(size_t)100);
        memset(dataset,' ',sizeof dataset);
        if (*nside < 0) memcpy(dataset,"Image bright field kz<0",23);
        else memcpy(dataset,"Image bright field kz>0",23);
        writehdf5mic_(Eimagex,Eimagey,Eimagez,nfft2d,imaxk0,Ediffkzpos,&zero,dataset,
                      &group_idmic_hdf5,(size_t)40);
        if (*nside > 0) {
            memset(dataset,' ',sizeof dataset); memcpy(dataset,"Image+incident bright field kz>0",32);
            writehdf5mic_(Eimageincx,Eimageincy,Eimageincz,nfft2d,imaxk0,Ediffkzpos,&zero,dataset,
                          &group_idmic_hdf5,(size_t)40);
        }
        for (int i=-imaxmask;i<=imaxmask;++i)
            (kxy)[i+imaxmask]=(REAL)i* *deltak/kval;
        dims[0]=grid;
        memset(dataset,' ',sizeof dataset); memcpy(dataset,"kxmic",5);
        hdf5write1d_i8_(&group_idmic_hdf5,dataset,kxy,dims,(size_t)100);
        memset(dataset,' ',sizeof dataset); memcpy(dataset,"kymic",5);
        hdf5write1d_i8_(&group_idmic_hdf5,dataset,kxy,dims,(size_t)100);
        for (size_t i=0;i<(size_t)nmask*(size_t)nmask;++i)
            mask_real[i]=std::abs((masque)[i]);
        dims[0]=grid*grid; dims[1]=nmask*nmask; dims[2]=1; dims[3]=1;
        memset(dataset,' ',sizeof dataset); memcpy(dataset,"masquemic",9);
        hdf5write1d_i8_(&group_idmic_hdf5,dataset,mask_real,dims,(size_t)100);
        free(mask_real);
    }
    *p_nstop = 0;
    (void)k02; (void)nfield; (void)deltakxy; (void)trope_len;
}
