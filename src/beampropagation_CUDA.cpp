#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "gaussian_gkp_tables.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace {
std::mutex bpm_gkp_mutex;
bool bpm_gkp_uploaded = false;

//-----------------------------------------------------------
bool acquire_beampropagation_fft_plan(int nfft2d, cufftHandle* plan) {
    cuda_data::MultilayerFftPlans plans{};
    if (!plan || !cuda_data::get_device_multilayer_fft_plans(nfft2d, nfft2d, &plans))
        return false;
    *plan = plans.image;
    return *plan != 0;
}

//-----------------------------------------------------------
cudaError_t blend_rigorous_gaussian_background(CU_INT *device_map, CUCOMPLEX *device_images,
                                               int nfft2d, int nx, int ny, int kkm, int jjm,
                                               CU_REAL xmin, CU_REAL ymin, CU_REAL z,
                                               CU_REAL aretecube, CUCOMPLEX incident, CU_REAL ss,
                                               CU_REAL pp, CU_REAL theta, CU_REAL phi, CU_REAL x0,
                                               CU_REAL y0, CU_REAL z0, CU_REAL waist, CU_REAL k0,
                                               bool circular) {
    const size_t plane = size_t(nfft2d) * size_t(nfft2d);
    cuda_data::BeamPropagationGaussianBuffers buffers{};
    if (!cuda_data::get_device_beam_propagation_gaussian_buffers(plane, &buffers))
        return cudaErrorMemoryAllocation;

    cudaError_t status = gaussianfft_build_coordinates_wrapper(
        buffers.xs, buffers.ys, buffers.zs, nfft2d, nx, ny, xmin, ymin, z, aretecube);
    const COMPLEX vacuum(CU_REAL(1), CU_REAL(0));
    const COMPLEX incident_host(incident.x, incident.y);
    if (status == cudaSuccess)
        status = incident_paraxial_gaussian_wrapper(
            buffers.xs, buffers.ys, buffers.zs, buffers.paraxial_field, int(plane), incident_host,
            vacuum, ss, pp, theta, phi, x0, y0, z0, waist, k0, circular ? 1 : 0);
    {
        std::lock_guard<std::mutex> lock(bpm_gkp_mutex);
        if (status == cudaSuccess && !bpm_gkp_uploaded) {
            status = cuda_data::copy_host_to_device(buffers.gkp_nodes, gp_x, sizeof(gp_x),
                                                    cuda_utils::TransferCategory::IncidentField);
            if (status == cudaSuccess)
                status =
                    cuda_data::copy_host_to_device(buffers.gkp_weights, gp_w, sizeof(gp_w),
                                                   cuda_utils::TransferCategory::IncidentField);
            if (status == cudaSuccess)
                bpm_gkp_uploaded = true;
        }
    }
    if (status == cudaSuccess)
        status = rigorous_gaussian_incident_wrapper(
            buffers.xs, buffers.ys, buffers.zs, int(plane), double(x0), double(y0), double(z0),
            double(theta), double(phi), double(waist), double(k0), double(ss), double(pp), incident_host,
            1.0e-3, circular ? 1 : 0, 0, buffers.gkp_nodes, buffers.gkp_weights,
            buffers.rigorous_field, buffers.convergence);
    if (status == cudaSuccess)
        status = gaussianfft_max_intensity_wrapper(buffers.paraxial_field, buffers.max_partials,
                                                   buffers.max_intensity, int(plane));
    if (status == cudaSuccess)
        status = beampropagation_blend_gaussian_background_wrapper(
            device_map, buffers.paraxial_field, buffers.rigorous_field, buffers.max_intensity,
            CU_REAL(1.0e-3), device_images, int(plane));
    return status;
}
} // namespace

//-----------------------------------------------------------
extern "C" void ifdda_beampropagation_initialize_plane_cuda_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CUCOMPLEX *epsilon,
    const CUCOMPLEX *incident, CUCOMPLEX *image_x, CUCOMPLEX *image_y, CUCOMPLEX *image_z,
    int *first_plane_map, CUCOMPLEX *local_field, const int *nx, const int *ny, const int *nfft2d,
    const int *nmax, const int *niso, const int *ndipole, const int *kkm, const int *jjm,
    const CU_REAL *xmin, const CU_REAL *ymin, const CU_REAL *first_z, const CU_REAL *aretecube,
    const CU_REAL *k0, const CUCOMPLEX *E0, const CU_REAL *ss, const CU_REAL *pp,
    const CU_REAL *theta_degrees, const CU_REAL *phi_degrees, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const CU_REAL *w0, const int *beam_type,
    int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!xs || !ys || !zs || !epsilon || !incident || !image_x || !image_y || !image_z ||
        !first_plane_map || !local_field || !nx || !ny || !nfft2d || !nmax || !niso || !ndipole ||
        !kkm || !jjm || !xmin || !ymin || !first_z || !aretecube || !k0 || !E0 || !ss || !pp ||
        !theta_degrees || !phi_degrees || !xgaus || !ygaus || !zgaus || !w0 || !beam_type ||
        *nx <= 0 || *ny <= 0 || *nfft2d <= 0 || *nmax <= 0 || *niso <= 0 || *ndipole <= 0 ||
        *nfft2d > std::numeric_limits<int>::max() / *nfft2d) {
        std::fprintf(stderr, "Beam propagation CUDA initialization: invalid input\n");
        return;
    }
    if (!initialize_cuda_device()) {
        std::fprintf(stderr,
                     "Beam propagation CUDA initialization: device initialization failed\n");
        return;
    }

    size_t plane = 0, image_count = 0, epsilon_count = 0;
    size_t vector_count = 0;
    if (!cuda_utils::checked_multiply(size_t(*nfft2d), size_t(*nfft2d), &plane) ||
        !cuda_utils::checked_multiply(plane, size_t(3), &image_count) ||
        !cuda_utils::checked_multiply(size_t(*nmax), size_t(*niso), &epsilon_count) ||
        !cuda_utils::checked_multiply(epsilon_count, size_t(*niso), &epsilon_count) ||
        !cuda_utils::checked_multiply(size_t(*nmax), size_t(3), &vector_count) ||
        image_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        epsilon_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        vector_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        plane > std::numeric_limits<size_t>::max() / sizeof(int) ||
        size_t(*ndipole) > std::numeric_limits<size_t>::max() / sizeof(CU_REAL)) {
        std::fprintf(stderr, "Beam propagation CUDA initialization: array size overflow\n");
        return;
    }

    std::vector<CUCOMPLEX> images(image_count);
    CU_REAL *device_xs = nullptr;
    CU_REAL *device_ys = nullptr;
    CU_REAL *device_zs = nullptr;
    CUCOMPLEX *device_epsilon = nullptr;
    CUCOMPLEX *device_incident = nullptr;
    CUCOMPLEX *device_images = nullptr;
    CU_INT *device_map = nullptr;
    CUCOMPLEX *device_local_field = nullptr;
    cuda_data::BeamPropagationBufferRequest request{};
    request.coordinate_count = size_t(*ndipole);
    request.epsilon_count = epsilon_count;
    request.incident_count = vector_count;
    request.image_count = image_count;
    request.map_count = plane;
    request.local_field_count = vector_count;
    cuda_data::BeamPropagationDeviceBuffers buffers{};
    if (!cuda_data::get_device_beam_propagation_buffers(request, &buffers))
        return;
    device_xs = buffers.xs;
    device_ys = buffers.ys;
    device_zs = buffers.zs;
    device_epsilon = buffers.epsilon;
    device_incident = buffers.incident;
    device_images = buffers.images;
    device_map = buffers.map;
    device_local_field = buffers.local_field;
    cudaError_t status = (device_xs && device_ys && device_zs && device_epsilon &&
                          device_incident && device_images && device_map && device_local_field)
                             ? cudaSuccess
                             : cudaErrorMemoryAllocation;
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_xs, xs, size_t(*ndipole) * sizeof(CU_REAL),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_ys, ys, size_t(*ndipole) * sizeof(CU_REAL),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_zs, zs, size_t(*ndipole) * sizeof(CU_REAL),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_epsilon, epsilon,
                                                epsilon_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_incident, incident,
                                                vector_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_local_field, local_field,
                                                vector_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    const bool rigorous_linear = *beam_type == 4;
    const bool rigorous_circular = *beam_type == 5;
    const int initialization_beam_type = rigorous_linear ? 2 : (rigorous_circular ? 3 : *beam_type);
    if (status == cudaSuccess)
        status = beampropagation_initialize_plane_wrapper(
            device_xs, device_ys, device_zs, device_epsilon, device_incident, device_images,
            device_map, device_local_field, *nx, *ny, *nfft2d, *nmax, *niso, *ndipole, *kkm, *jjm,
            *xmin, *ymin, *first_z, *aretecube, *k0, *E0, *ss, *pp, *theta_degrees, *phi_degrees,
            *xgaus, *ygaus, *zgaus, *w0, initialization_beam_type);
    if (status == cudaSuccess && (rigorous_linear || rigorous_circular))
        status = blend_rigorous_gaussian_background(
            device_map, device_images, *nfft2d, *nx, *ny, *kkm, *jjm, *xmin, *ymin, *first_z,
            *aretecube, *E0, *ss, *pp, *theta_degrees, *phi_degrees, *xgaus, *ygaus, *zgaus, *w0,
            *k0, rigorous_circular);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(images.data(), device_images,
                                                image_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(first_plane_map, device_map, plane * sizeof(int),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(local_field, device_local_field,
                                                vector_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);

    if (status != cudaSuccess) {
        std::fprintf(stderr, "Beam propagation CUDA initialization: %s\n",
                     cudaGetErrorString(status));
        return;
    }
    for (size_t i = 0; i < plane; ++i) {
        image_x[i] = images[i];
        image_y[i] = images[plane + i];
        image_z[i] = images[2 * plane + i];
    }
    *nstop = 0;
}

// Run first-plane initialization and all remaining BPM layers without
// bouncing the image planes or their map through host memory between calls.
//-----------------------------------------------------------
extern "C" void ifdda_beampropagation_run_cuda_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const COMPLEX *epsilon,
    const COMPLEX *incident, COMPLEX *local_field, const int *nx, const int *ny, const int *nz,
    const int *nfft2d, const int *nmax, const int *niso, const int *ndipole, const int *kkm,
    const int *jjm, const CU_REAL *xmin, const CU_REAL *ymin, const CU_REAL *first_z,
    const CU_REAL *aretecube, const CU_REAL *k0, const COMPLEX *E0, const CU_REAL *ss,
    const CU_REAL *pp, const CU_REAL *theta_degrees, const CU_REAL *phi_degrees,
    const CU_REAL *xgaus, const CU_REAL *ygaus, const CU_REAL *zgaus, const CU_REAL *w0,
    const int *beam_type, const int resident_mode, const int macro_mode,
    int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!xs || !ys || !zs || !epsilon || !incident || !local_field || !nx || !ny || !nz ||
        !nfft2d || !nmax || !niso || !ndipole || !kkm || !jjm || !xmin || !ymin || !first_z ||
        !aretecube || !k0 || !E0 || !ss || !pp || !theta_degrees || !phi_degrees || !xgaus ||
        !ygaus || !zgaus || !w0 || !beam_type || *nx <= 0 || *ny <= 0 || *nz <= 1 || *nfft2d <= 0 ||
        *nmax <= 0 || *niso <= 0 || *ndipole <= 0 ||
        (macro_mode != 0 && macro_mode != 1) ||
        *nfft2d > std::numeric_limits<int>::max() / *nfft2d)
        return;
    if (!initialize_cuda_device())
        return;

    const size_t plane = size_t(*nfft2d) * size_t(*nfft2d);
    const size_t vector_count = size_t(*nmax) * size_t(3);
    const size_t epsilon_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
    if (plane > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        vector_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        epsilon_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        size_t(*ndipole) > std::numeric_limits<size_t>::max() / sizeof(CU_REAL))
        return;

    CU_REAL *device_xs = nullptr;
    CU_REAL *device_ys = nullptr;
    CU_REAL *device_zs = nullptr;
    CUCOMPLEX *device_epsilon = nullptr;
    CUCOMPLEX *device_incident = nullptr;
    CUCOMPLEX *device_images = nullptr;
    CUCOMPLEX *device_local_field = nullptr;
    CU_INT *device_map = nullptr;
    cuda_data::BeamPropagationBufferRequest request{};
    request.coordinate_count = size_t(*ndipole);
    request.epsilon_count = epsilon_count;
    request.incident_count = vector_count;
    request.image_count = 3 * plane;
    request.map_count = plane;
    request.local_field_count = resident_mode ? 0 : vector_count;
    cuda_data::BeamPropagationDeviceBuffers buffers{};
    if (!cuda_data::get_device_beam_propagation_buffers(request, &buffers))
        return;
    device_xs = buffers.xs;
    device_ys = buffers.ys;
    device_zs = buffers.zs;
    device_epsilon = buffers.epsilon;
    device_incident = buffers.incident;
    device_images = buffers.images;
    if (resident_mode == 1)
        device_local_field = get_gpbicg_device_xi_buffer(local_field, vector_count);
    //-----------------------------------------------------------
    else if (resident_mode == 2) {
        cuda_data::ScalarDriverDeviceBuffers scalar_buffers{};
        device_local_field = cuda_data::get_device_scalar_driver_buffers(
                                 size_t(*ndipole), 0, size_t(*nmax), &scalar_buffers)
                                 ? scalar_buffers.vector_field
                                 : nullptr;
    } else
        device_local_field = buffers.local_field;
    if (std::getenv("IFDDA_TRACE_BPM_RESIDENT"))
        std::fprintf(stderr, "[BPM resident] mode=%d host key=%p GPU field=%p count=%zu\n",
                     resident_mode, static_cast<void *>(local_field),
                     static_cast<void *>(device_local_field), vector_count);
    device_map = buffers.map;
    cudaError_t status = (device_xs && device_ys && device_zs && device_epsilon &&
                          device_incident && device_images && device_local_field && device_map)
                             ? cudaSuccess
                             : cudaErrorMemoryAllocation;

    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_xs, xs, size_t(*ndipole) * sizeof(CU_REAL),
                                                cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_ys, ys, size_t(*ndipole) * sizeof(CU_REAL),
                                                cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_zs, zs, size_t(*ndipole) * sizeof(CU_REAL),
                                                cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_epsilon, epsilon,
                                                epsilon_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Material);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_incident, incident,
                                                vector_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::SolverVector);
    if (status == cudaSuccess)
        status = cudaMemset(device_local_field, 0, vector_count * sizeof(CUCOMPLEX));
    const bool rigorous_linear = *beam_type == 4;
    const bool rigorous_circular = *beam_type == 5;
    const CUCOMPLEX device_E0 = cuda_utils::device_value(
        cuda_utils::HostComplex(E0->real(), E0->imag()));
    const int initialization_beam_type = rigorous_linear ? 2 : (rigorous_circular ? 3 : *beam_type);
    if (status == cudaSuccess)
        status = beampropagation_initialize_plane_wrapper(
            device_xs, device_ys, device_zs, device_epsilon, device_incident, device_images,
            device_map, device_local_field, *nx, *ny, *nfft2d, *nmax, *niso, *ndipole, *kkm, *jjm,
            *xmin, *ymin, *first_z, *aretecube, *k0, device_E0, *ss, *pp, *theta_degrees, *phi_degrees,
            *xgaus, *ygaus, *zgaus, *w0, initialization_beam_type);
    if (status == cudaSuccess && (rigorous_linear || rigorous_circular))
        status = blend_rigorous_gaussian_background(
            device_map, device_images, *nfft2d, *nx, *ny, *kkm, *jjm, *xmin, *ymin, *first_z,
            *aretecube, device_E0, *ss, *pp, *theta_degrees, *phi_degrees, *xgaus, *ygaus, *zgaus, *w0,
            *k0, rigorous_circular);
    cufftHandle spectral_plan = 0;
    if (status == cudaSuccess && !acquire_beampropagation_fft_plan(*nfft2d, &spectral_plan))
        status = cudaErrorMemoryAllocation;
    if (status == cudaSuccess)
        status = beampropagation_spectral_wrapper(device_images, device_map, device_epsilon,
                                                  device_local_field, *nx, *ny, *nz, *nfft2d, *nmax,
                                                  *niso, *ndipole, spectral_plan, *aretecube, *k0,
                                                  macro_mode);
    if (status == cudaSuccess && resident_mode == 0)
        status = cuda_data::copy_device_to_host(local_field, device_local_field,
                                                vector_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Result);

    if (status == cudaSuccess && std::getenv("IFDDA_TRACE_BPM_RESIDENT")) {
        std::vector<CUCOMPLEX> diagnostic_field(vector_count);
        std::vector<CUCOMPLEX> diagnostic_images(3 * plane);
        std::vector<int> diagnostic_map(plane);
        status = cuda_data::copy_device_to_host(diagnostic_field.data(), device_local_field,
                                                vector_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Diagnostic);
        if (status == cudaSuccess)
            status = cuda_data::copy_device_to_host(diagnostic_images.data(), device_images,
                                                    3 * plane * sizeof(CUCOMPLEX),
                                                    cuda_utils::TransferCategory::Diagnostic);
        if (status == cudaSuccess)
            status = cuda_data::copy_device_to_host(diagnostic_map.data(), device_map,
                                                    plane * sizeof(int),
                                                    cuda_utils::TransferCategory::Diagnostic);
        if (status == cudaSuccess) {
            // Reproduce beampropagation.f90's ordered coordinate walk so the
            // resident path can be checked against its exact tabfft mapping.
            size_t next_dipole = 0;
            size_t map_mismatches = 0;
            const CU_REAL tolerance = *aretecube / CU_REAL(1000);
            for (int iy = 0; iy < *nfft2d; ++iy) {
                for (int ix = 0; ix < *nfft2d; ++ix) {
                    const size_t index = size_t(ix) + size_t(*nfft2d) * size_t(iy);
                    int expected = 0;
                    if (next_dipole < size_t(*ndipole)) {
                        const CU_REAL x = *xmin + CU_REAL(ix - *kkm) * *aretecube;
                        const CU_REAL y = *ymin + CU_REAL(iy - *jjm) * *aretecube;
                        if (std::abs(x - xs[next_dipole]) <= tolerance &&
                            std::abs(y - ys[next_dipole]) <= tolerance &&
                            std::abs(*first_z - zs[next_dipole]) <= tolerance) {
                            expected = int(next_dipole + 1);
                            ++next_dipole;
                        }
                    }
                    if (diagnostic_map[index] != expected)
                        ++map_mismatches;
                }
            }
            std::fprintf(
                stderr, "[BPM resident] ordered tabfft comparison: mismatches=%zu, mapped=%zu/%d\n",
                map_mismatches, next_dipole, *ndipole);
            std::fprintf(stderr, "[BPM resident] XI sample:");
            for (size_t i = 0; i < std::min<size_t>(12, vector_count); ++i)
                std::fprintf(stderr, " (% .16g,% .16g)", double(diagnostic_field[i].x),
                             double(diagnostic_field[i].y));
            const size_t legacy_sample =
                std::min<size_t>(size_t(*nmax - 1), vector_count > 12 ? vector_count - 12 : 0);
            std::fprintf(stderr, "\n[BPM resident] XI sample at mesh boundary %zu:", legacy_sample);
            for (size_t i = legacy_sample; i < std::min(vector_count, legacy_sample + 11); ++i)
                std::fprintf(stderr, " (% .16g,% .16g)", double(diagnostic_field[i].x),
                             double(diagnostic_field[i].y));
            std::fprintf(stderr, "\n[BPM resident] image abs sums:");
            for (size_t component = 0; component < 3; ++component) {
                double sum = 0.0;
                for (size_t i = 0; i < plane; ++i) {
                    const CUCOMPLEX value = diagnostic_images[component * plane + i];
                    sum += std::hypot(double(value.x), double(value.y));
                }
                std::fprintf(stderr, " %.16g", sum);
            }
            std::fprintf(stderr, "\n");
        }
    }

    if (status != cudaSuccess) {
        std::fprintf(stderr, "Beam propagation CUDA: %s\n", cudaGetErrorString(status));
        return;
    }
    if (resident_mode == 1)
        mark_gpbicg_device_xi_current(local_field, true);
    *nstop = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_beampropagation_propagate_cuda_(
    CUCOMPLEX *image_x, CUCOMPLEX *image_y, CUCOMPLEX *image_z, const int *first_plane_map,
    const CUCOMPLEX *epsilon, CUCOMPLEX *local_field, const int *nx, const int *ny, const int *nz,
    const int *nfft2d, const int *nmax, const int *niso, const int *ndipole,
    const CU_REAL *aretecube, const CU_REAL *k0, const int *convert_local_to_macro, int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!image_x || !image_y || !image_z || !first_plane_map || !epsilon || !local_field || !nx ||
        !ny || !nz || !nfft2d || !nmax || !niso || !ndipole || !aretecube || !k0 || *nx <= 0 ||
        *ny <= 0 || *nz <= 1 || *nfft2d <= 0 || *nmax <= 0 || *niso <= 0 || *ndipole <= 0 ||
        !convert_local_to_macro || *aretecube <= CU_REAL(0)) {
        std::fprintf(stderr, "Beam propagation CUDA: invalid input\n");
        return;
    }
    if (!initialize_cuda_device()) {
        std::fprintf(stderr, "Beam propagation CUDA: device initialization failed\n");
        return;
    }

    size_t plane = 0, image_count = 0, epsilon_count = 0, field_count = 0;
    if (!cuda_utils::checked_multiply(size_t(*nfft2d), size_t(*nfft2d), &plane) ||
        !cuda_utils::checked_multiply(plane, size_t(3), &image_count) ||
        !cuda_utils::checked_multiply(size_t(*nmax), size_t(*niso), &epsilon_count) ||
        !cuda_utils::checked_multiply(epsilon_count, size_t(*niso), &epsilon_count) ||
        !cuda_utils::checked_multiply(size_t(*nmax), size_t(3), &field_count) ||
        image_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        epsilon_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        field_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        plane > std::numeric_limits<size_t>::max() / sizeof(int)) {
        std::fprintf(stderr, "Beam propagation CUDA: array size overflow\n");
        return;
    }

    std::vector<CUCOMPLEX> packed_images(image_count);
    for (size_t i = 0; i < plane; ++i) {
        packed_images[i] = image_x[i];
        packed_images[plane + i] = image_y[i];
        packed_images[2 * plane + i] = image_z[i];
    }

    CUCOMPLEX *device_images = nullptr;
    CU_INT *device_map = nullptr;
    CUCOMPLEX *device_epsilon = nullptr;
    CUCOMPLEX *device_local_field = nullptr;
    cuda_data::BeamPropagationBufferRequest request{};
    request.epsilon_count = epsilon_count;
    request.image_count = image_count;
    request.map_count = plane;
    request.local_field_count = field_count;
    cuda_data::BeamPropagationDeviceBuffers buffers{};
    if (!cuda_data::get_device_beam_propagation_buffers(request, &buffers))
        return;
    device_images = buffers.images;
    device_map = buffers.map;
    device_epsilon = buffers.epsilon;
    device_local_field = buffers.local_field;
    cudaError_t status = (device_images && device_map && device_epsilon && device_local_field)
                             ? cudaSuccess
                             : cudaErrorMemoryAllocation;
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_images, packed_images.data(),
                                                image_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_map, first_plane_map, plane * sizeof(int),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_epsilon, epsilon,
                                                epsilon_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_local_field, local_field,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    cufftHandle spectral_plan = 0;
    if (status == cudaSuccess && !acquire_beampropagation_fft_plan(*nfft2d, &spectral_plan))
        status = cudaErrorMemoryAllocation;
    if (status == cudaSuccess)
        status = beampropagation_spectral_wrapper(
            device_images, device_map, device_epsilon, device_local_field, *nx, *ny, *nz, *nfft2d,
            *nmax, *niso, *ndipole, spectral_plan, *aretecube, *k0, *convert_local_to_macro);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(packed_images.data(), device_images,
                                                image_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(local_field, device_local_field,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);

    if (status != cudaSuccess) {
        std::fprintf(stderr, "Beam propagation CUDA: %s\n", cudaGetErrorString(status));
        return;
    }
    for (size_t i = 0; i < plane; ++i) {
        image_x[i] = packed_images[i];
        image_y[i] = packed_images[plane + i];
        image_z[i] = packed_images[2 * plane + i];
    }
    *nstop = 0;
}

//-----------------------------------------------------------
extern "C" void beampropagation_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
    const CU_REAL *aretecube, const CU_REAL *k0, const CU_REAL *w0,
    const COMPLEX *E0, const CU_REAL *ss, const CU_REAL *pp,
    const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char *beam,
    const COMPLEX *epsilon, const int *niso, const int *ndipole,
    const int *nx, const int *ny, const int *nz, const int *nxm,
    const int *nym, const int *nzm, const int *nmax, const int *nfft2d,
    COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF,
    const IFDDA_PLAN_INT64 *plan2f, const IFDDA_PLAN_INT64 *plan2b, int *nstop,
    char *infostr, const int *resident_mode, size_t beam_length,
    size_t infostr_length) {
    (void)nxm;
    (void)nym;
    (void)nzm;
    (void)imagex;
    (void)imagey;
    (void)imagez;
    (void)FF;
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
        !theta || !phi || !xgaus || !ygaus || !zgaus || !beam || !epsilon ||
        !niso || !ndipole || !nx || !ny || !nz || !nmax || !nfft2d ||
        !FF0 || !FFloc || !infostr || *ndipole <= 0 || *niso <= 0 ||
        *nx <= 0 || *ny <= 0 || *nz <= 0 || *nmax <= 0 || *nfft2d <= 0 ||
        *nfft2d > std::numeric_limits<int>::max() / *nfft2d) {
        set_error("invalid beam propagation arguments");
        return;
    }

    int beam_type = -1;
    if (beam_length >= 11 && std::strncmp(beam, "pwavelinear", 11) == 0)
        beam_type = 0;
    else if (beam_length >= 13 && std::strncmp(beam, "pwavecircular", 13) == 0)
        beam_type = 1;
    else if (beam_length >= 15 && std::strncmp(beam, "gparawavelinear", 15) == 0)
        beam_type = 2;
    else if (beam_length >= 17 && std::strncmp(beam, "gparawavecircular", 17) == 0)
        beam_type = 3;
    else if ((beam_length >= 11 && std::strncmp(beam, "gwavelinear", 11) == 0) ||
             (beam_length >= 14 && std::strncmp(beam, "gfftwavelinear", 14) == 0))
        beam_type = 4;
    else if ((beam_length >= 13 && std::strncmp(beam, "gwavecircular", 13) == 0) ||
             (beam_length >= 16 && std::strncmp(beam, "gfftwavecircular", 16) == 0))
        beam_type = 5;
    if (beam_type < 0) {
        set_error("unsupported beam for CUDA propagation");
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
    const int mode = resident_mode ? *resident_mode : 0;
    ifdda_beampropagation_run_cuda_(
        xs, ys, zs, epsilon, FF0, FFloc, nx, ny, nz, nfft2d, nmax, niso,
        ndipole, &kkm, &jjm, &xmin, &ymin, zs, aretecube, k0, E0, ss, pp,
        theta, phi, xgaus, ygaus, zgaus, w0, &beam_type, mode, 0, nstop);
    if (*nstop != 0)
        set_error("CUDA beam propagation failed");
}

extern "C" void ifdda_reset_beampropagation_gkp_state() {
    std::lock_guard<std::mutex> lock(bpm_gkp_mutex);
    bpm_gkp_uploaded = false;
}
