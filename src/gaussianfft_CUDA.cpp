#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "gaussian_gkp_tables.h"
#include "kernel.h"

#include <climits>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <limits>
#include <mutex>
#include <vector>

namespace {
std::mutex gaussian_fft_gkp_mutex;
bool gaussian_fft_gkp_uploaded = false;
} // namespace

extern "C" void ifdda_reset_gaussianfft_gkp_state() {
    std::lock_guard<std::mutex> lock(gaussian_fft_gkp_mutex);
    gaussian_fft_gkp_uploaded = false;
}

// Full GPU path for gaussianchampfft: create the paraxial field, evaluate the
// rigorous correction, select the same thresholded pixels, then propagate
// through all z planes without materializing the first image on the CPU.
//-----------------------------------------------------------
extern "C" void ifdda_gaussianfft_cuda_(
    const int *nfft2d, const int *nx, const int *ny, const int *nz, const int *nmax,
    const int *dipole_count, const int *kkm, const int *jjm, const CU_REAL *dipole_xs,
    const CU_REAL *dipole_ys, const CU_REAL *xmin, const CU_REAL *ymin, const CU_REAL *z,
    const CU_REAL *aretecube, const CU_REAL *x0, const CU_REAL *y0, const CU_REAL *z0,
    const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *waist, const CU_REAL *k0,
    const CU_REAL *ss, const CU_REAL *pp, const COMPLEX *e0, const CU_REAL *tolerance,
    const int *circular, COMPLEX *dipole_field, int *nstop,
    const CU_REAL *dipole_xs_device = nullptr,
    const CU_REAL *dipole_ys_device = nullptr,
    CUCOMPLEX **resident_field = nullptr) {
    if (!nstop)
        return;
    *nstop = 1;
    if (resident_field)
        *resident_field = nullptr;
    if (!nfft2d || !nx || !ny || !nz || !nmax || !dipole_count || !kkm || !jjm ||
        (!dipole_xs && !dipole_xs_device) || (!dipole_ys && !dipole_ys_device) ||
        !xmin || !ymin || !z || !aretecube || !x0 || !y0 || !z0 || !theta || !phi ||
        !waist || !k0 || !ss || !pp || !e0 || !tolerance || !circular ||
        (!dipole_field && !resident_field) ||
        *nfft2d <= 0 || *nfft2d > INT_MAX / *nfft2d || *nx <= 0 || *ny <= 0 || *nz <= 0 ||
        *nmax <= 0 || *aretecube <= CU_REAL(0) || *waist <= CU_REAL(0) || *k0 <= CU_REAL(0) ||
        *tolerance <= CU_REAL(0) || (*circular != 0 && *circular != 1))
        return;

    const size_t plane = size_t(*nfft2d) * size_t(*nfft2d);
    const size_t plane_count = size_t(*nx) * size_t(*ny);
    if (size_t(*nx) > std::numeric_limits<size_t>::max() / size_t(*ny) ||
        plane_count > std::numeric_limits<size_t>::max() / size_t(*nz) ||
        plane_count * size_t(*nz) > size_t(*nmax) || *dipole_count <= 0 ||
        size_t(*dipole_count) > size_t(*nmax) || plane > size_t(std::numeric_limits<int>::max()))
        return;
    if (!initialize_cuda_device())
        return;

    // Geometry may already own these GPU coordinates. The lookup is keyed by
    // the original CPU arrays and preserves the compact dipole ordering.
    const CU_REAL *resolved_xs_device = dipole_xs_device;
    const CU_REAL *resolved_ys_device = dipole_ys_device;
    if ((!resolved_xs_device || !resolved_ys_device) &&
        dipole_xs && dipole_ys && z) {
        const CU_REAL *alias_x = nullptr;
        const CU_REAL *alias_y = nullptr;
        const CU_REAL *alias_z = nullptr;
        if (cuda_data::get_device_geometry_coordinate_aliases(
                dipole_xs, dipole_ys, z, size_t(*dipole_count),
                &alias_x, &alias_y, &alias_z)) {
            if (!resolved_xs_device)
                resolved_xs_device = alias_x;
            if (!resolved_ys_device)
                resolved_ys_device = alias_y;
        }
    }
    cuda_data::GaussianFftDeviceBuffers buffers{};
    const bool need_dipole_coordinates = !resolved_xs_device || !resolved_ys_device;
    if (!cuda_data::get_device_gaussian_fft_buffers(
            plane, size_t(*nmax), &buffers, need_dipole_coordinates))
        return;

    const char *stage = "upload dipole coordinates for pixel map";
    using cuda_utils::TransferCategory;
    cudaError_t status = cudaSuccess;
    if (!resolved_xs_device)
        status = cuda_data::copy_host_to_device(buffers.dipole_xs, dipole_xs,
                                                size_t(*dipole_count) * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    const CU_REAL *device_dipole_xs = resolved_xs_device ? resolved_xs_device : buffers.dipole_xs;
    const CU_REAL *device_dipole_ys = resolved_ys_device ? resolved_ys_device : buffers.dipole_ys;
    if (status == cudaSuccess && !resolved_ys_device)
        status = cuda_data::copy_host_to_device(buffers.dipole_ys, dipole_ys,
                                                size_t(*dipole_count) * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess) {
        stage = "build pixel-to-dipole map";
        status = gaussianfft_build_plane_map_wrapper(device_dipole_xs, device_dipole_ys,
                                                     buffers.plane_map, *nfft2d, *dipole_count,
                                                     *kkm, *jjm, *xmin, *ymin, *aretecube);
    }
    // Fortran gaussianfft.f90 computes Emodmax on the waist plane z=zgaus,
    // not on the first mesh plane z=zs(1). Keep that maximum for the blend.
    if (status == cudaSuccess) {
        stage = "build waist-plane coordinates";
        status = gaussianfft_build_coordinates_wrapper(buffers.xs, buffers.ys, buffers.zs, *nfft2d,
                                                       *nx, *ny, *xmin, *ymin, *z0, *aretecube);
    }
    const COMPLEX vacuum(CU_REAL(1), CU_REAL(0));
    if (status == cudaSuccess) {
        stage = "waist-plane paraxial field";
        status = incident_paraxial_gaussian_wrapper(
            buffers.xs, buffers.ys, buffers.zs, buffers.paraxial_field, int(plane), *e0, vacuum,
            *ss, *pp, *theta, *phi, *x0, *y0, *z0, *waist, *k0, *circular);
    }
    if (status == cudaSuccess) {
        stage = "waist-plane paraxial maximum";
        status = gaussianfft_max_intensity_wrapper(buffers.paraxial_field, buffers.max_partials,
                                                   buffers.max_intensity, int(plane));
    }
    if (status == cudaSuccess) {
        stage = "build first-plane coordinates";
        status = gaussianfft_build_coordinates_wrapper(buffers.xs, buffers.ys, buffers.zs, *nfft2d,
                                                       *nx, *ny, *xmin, *ymin, *z, *aretecube);
    }
    if (status == cudaSuccess) {
        stage = "first-plane paraxial field";
        status = incident_paraxial_gaussian_wrapper(
            buffers.xs, buffers.ys, buffers.zs, buffers.paraxial_field, int(plane), *e0, vacuum,
            *ss, *pp, *theta, *phi, *x0, *y0, *z0, *waist, *k0, *circular);
    }

    {
        std::lock_guard<std::mutex> lock(gaussian_fft_gkp_mutex);
        if (status == cudaSuccess && !gaussian_fft_gkp_uploaded) {
            stage = "GKP table upload";
            using cuda_utils::TransferCategory;
            status = cuda_data::copy_host_to_device(buffers.gkp_nodes, gp_x, sizeof(gp_x),
                                                    TransferCategory::IncidentField);
            if (status == cudaSuccess)
                status = cuda_data::copy_host_to_device(buffers.gkp_weights, gp_w, sizeof(gp_w),
                                                        TransferCategory::IncidentField);
            if (status == cudaSuccess)
                gaussian_fft_gkp_uploaded = true;
        }
    }
    if (status == cudaSuccess) {
        stage = "rigorous field";
        status = rigorous_gaussian_incident_wrapper(
            buffers.xs, buffers.ys, buffers.zs, int(plane), double(*x0), double(*y0), double(*z0),
            double(*theta), double(*phi), double(*waist), double(*k0), double(*ss), double(*pp),
            *e0, double(*tolerance), *circular, 0, buffers.gkp_nodes, buffers.gkp_weights,
            buffers.rigorous_field, buffers.convergence);
    }
    // gaussianchampcirc/gaussianchamp zero the rigorous tail when the
    // paraxial envelope is below the Fortran 1e-7 amplitude cutoff.
    // Apply only to the rigorous candidate: unselected pixels must retain
    // their ordinary paraxial value in the subsequent blend.
    if (status == cudaSuccess) {
        stage = "rigorous Fortran tail cutoff";
        status = rigorous_gaussian_fortran_cutoff_wrapper(
            buffers.xs, buffers.ys, buffers.zs, int(plane), double(*x0), double(*y0), double(*z0),
            double(*theta), double(*phi), double(*waist), double(*k0), double(*ss), double(*pp),
            *e0, *circular, buffers.rigorous_field);
    }
    if (status == cudaSuccess) {
        stage = "first-plane blend";
        status = gaussianfft_blend_first_plane_wrapper(
            buffers.paraxial_field, buffers.rigorous_field, buffers.max_intensity, *tolerance,
            *circular, buffers.image_planes, int(plane));
    }
    // The physical incident field is written directly into the persistent FF0;
    // secondary resident consumers (widefield) retain their distinct workspace.
    CUCOMPLEX *device_field = resident_field ? buffers.dipole_field
                                           : cuda_data::get_ff0(size_t(*nmax));
    if (!device_field)
        return;
    if (status == cudaSuccess)
        status = cudaMemset(device_field, 0, size_t(3) * size_t(*nmax) * sizeof(CUCOMPLEX));

    const cufftHandle plan = get_speckle_fft_plan(*nfft2d);
    if (status == cudaSuccess && plan == 0)
        status = cudaErrorUnknown;
    for (int layer = 0; status == cudaSuccess && layer < *nz; ++layer) {
        cufftResult fft_status =
            CUFFT_EXECUTE(plan, buffers.image_planes, buffers.image_planes, CUFFT_INVERSE);
        if (fft_status != CUFFT_SUCCESS) {
            status = cudaErrorUnknown;
            break;
        }
        status = gaussianfft_filter_wrapper(buffers.image_planes, *nfft2d, *aretecube, *k0,
                                            layer == 0 ? 1 : 0);
        if (status != cudaSuccess)
            break;
        fft_status = CUFFT_EXECUTE(plan, buffers.image_planes, buffers.image_planes, CUFFT_FORWARD);
        if (fft_status != CUFFT_SUCCESS) {
            status = cudaErrorUnknown;
            break;
        }
        status = gaussianfft_scatter_wrapper(buffers.image_planes, buffers.plane_map,
                                             device_field, *nfft2d, int(plane_count), layer,
                                             size_t(*nmax));
    }
    if (status == cudaSuccess && resident_field)
        *resident_field = device_field;
    else if (status == cudaSuccess)
        status = cuda_data::copy_device_complex_to_host(dipole_field, device_field,
                                                size_t(3) * size_t(*nmax),
                                                TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA gaussian FFT pipeline failed at %s: %s\n", stage,
                     cudaGetErrorString(status));
        return;
    }
    *nstop = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_gaussianfft_cuda_resident_grid_origin_(
    const int *nfft2d, const int *nx, const int *ny, const int *nz,
    const int *nmax, const int *grid_count, const REAL *grid_x,
    const REAL *grid_y, const REAL *grid_origin_x,
    const REAL *grid_origin_y, const REAL *grid_origin_z,
    const REAL *aretecube, const REAL *x0, const REAL *y0,
    const REAL *z0, const REAL *theta, const REAL *phi,
    const REAL *waist, const REAL *k0, const REAL *ss,
    const REAL *pp, const COMPLEX *e0, const REAL *tolerance,
    const int *circular, CUCOMPLEX **resident_field, int *status_out) {
    if (!status_out)
        return;
    *status_out = 1;
    if (!nfft2d || !nx || !ny || !nz || !nmax || !grid_count || !grid_x ||
        !grid_y || !grid_origin_x || !grid_origin_y || !grid_origin_z ||
        !aretecube || !x0 || !y0 || !z0 || !theta || !phi || !waist || !k0 ||
        !ss || !pp || !e0 || !tolerance || !circular || !resident_field ||
        *nfft2d <= 0 || *nx <= 0 || *ny <= 0 || *nz <= 0 || *nmax <= 0 ||
        *grid_count <= 0)
        return;

    const int kkm = (*nfft2d - *nx) / 2;
    const int jjm = (*nfft2d - *ny) / 2;
    ifdda_gaussianfft_cuda_(nfft2d, nx, ny, nz, nmax, grid_count, &kkm, &jjm,
        nullptr, nullptr, grid_origin_x, grid_origin_y, grid_origin_z,
        aretecube, x0, y0, z0, theta, phi, waist, k0, ss, pp, e0, tolerance,
        circular, nullptr, status_out, grid_x, grid_y, resident_field);
}

//-----------------------------------------------------------
extern "C" void gaussianchampfft_(
    const REAL *xs, const REAL *ys, const REAL *zs,
    const REAL *aretecube, const REAL *k0, const REAL *w0,
    const COMPLEX *E0, const REAL *ss, const REAL *pp,
    const REAL *theta, const REAL *phi, const REAL *xgaus,
    const REAL *ygaus, const REAL *zgaus, const char *beam,
    const int *ndipole, const int *nx, const int *ny, const int *nz,
    const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey,
    COMPLEX *imagez, COMPLEX *FF0, const REAL *tolinit,
    const IFDDA_PLAN_INT64 *plan2f, const IFDDA_PLAN_INT64 *plan2b, int *nstop,
    char *infostr, size_t beam_length, size_t infostr_length) {
    (void)nxm;
    (void)nym;
    (void)nzm;
    (void)imagex;
    (void)imagey;
    (void)imagez;
    (void)plan2f;
    (void)plan2b;
    if (!nstop)
        return;
    *nstop = 1;
    if (infostr && infostr_length)
        std::memset(infostr, ' ', infostr_length);

    const auto set_error = [infostr, infostr_length](const char *message) {
        if (infostr && infostr_length && message)
            std::memcpy(infostr, message,
                        std::min(infostr_length, std::strlen(message)));
    };
    if (!xs || !ys || !zs || !aretecube || !k0 || !w0 || !E0 || !ss || !pp ||
        !theta || !phi || !xgaus || !ygaus || !zgaus || !beam || !ndipole ||
        !nx || !ny || !nz || !nmax || !nfft2d || !FF0 || !tolinit ||
        !infostr || *ndipole <= 0 || *nx <= 0 || *ny <= 0 || *nz <= 0 ||
        *nmax <= 0 || *nfft2d <= 0 || *nfft2d > INT_MAX / *nfft2d) {
        set_error("invalid Gaussian FFT arguments");
        return;
    }

    const bool circular = beam_length >= 16 &&
                          std::strncmp(beam, "gfftwavecircular", 16) == 0;
    const bool supported_beam =
        (beam_length >= 14 && std::strncmp(beam, "gfftwavelinear", 14) == 0) || circular;
    if (!supported_beam) {
        set_error("unsupported beam for CUDA Gaussian FFT");
        return;
    }

    CU_REAL xmin = xs[0];
    CU_REAL ymin = ys[0];
    for (int i = 1; i < *ndipole; ++i) {
        xmin = std::min(xmin, xs[i]);
        ymin = std::min(ymin, ys[i]);
    }
    const int kkm = (*nfft2d - *nx) / 2;
    const int jjm = (*nfft2d - *ny) / 2;
    int circular_flag = circular ? 1 : 0;
    ifdda_gaussianfft_cuda_(
        nfft2d, nx, ny, nz, nmax, ndipole, &kkm, &jjm, xs, ys, &xmin, &ymin,
        zs, aretecube, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, E0,
        tolinit, &circular_flag, FF0, nstop);
    if (*nstop != 0)
        set_error("CUDA Gaussian FFT pipeline failed");
}
