#include "cuda_data.h"
#define IFDDA_CUDA_CDMLIB_HOST_ABI
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CDMLIB_HOST_ABI
#include "cuda_utils.h"
#include "gaussian_gkp_tables.h"
#include "kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace {
using Complex = COMPLEX;
struct FortranComplex {
    REAL real;
    REAL imag;
};
FortranComplex to_fortran_complex(const COMPLEX& value) {
    return {static_cast<REAL>(value.real()),
            static_cast<REAL>(value.imag())};
}
COMPLEX to_host_complex(const FortranComplex& value) {
    return COMPLEX(value.real, value.imag);
}
constexpr REAL pi = REAL(3.1415926535897932384626433832795L);
std::mutex gaussian_gkp_table_mutex;
std::mutex rigorous_gaussian_workspace_mutex;
std::mutex plane_wave_derivative_mutex;
std::mutex dipole_derivative_mutex;
std::mutex paraxial_gaussian_mutex;
bool gaussian_gkp_tables_uploaded = false;

//-----------------------------------------------------------
CUCOMPLEX make_cuda_complex(REAL real, REAL imag) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real, imag);
#else
    return make_cuDoubleComplex(real, imag);
#endif
}

//-----------------------------------------------------------
bool near_value(REAL value, REAL reference) {
    return reference == REAL(0) ? value == REAL(0)
                                : std::abs((value - reference) / reference) <= REAL(1e-12);
}

//-----------------------------------------------------------
bool calculate_plane_wave_derivative(CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL k0,
                                      CUCOMPLEX amplitude, CU_REAL ss, CU_REAL pp,
                                      CU_REAL theta, CU_REAL phi, int selector,
                                      bool circular, FortranComplex* output) {
    if (!output || selector < 1 || selector > 4 || k0 <= CU_REAL(0) ||
        !initialize_cuda_device()) {
        return false;
    }

    // The legacy Fortran derivative routines may be called from OpenMP loops.
    // They share this persistent incident-field buffer, so serialize its use.
    std::lock_guard<std::mutex> lock(plane_wave_derivative_mutex);

    cuda_data::IncidentFieldBufferRequest request{};
    request.coordinate_count = 1;
    request.field_count = 9;
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_incident_field_buffers(request, &buffers) || !buffers.field) {
        return false;
    }
    if (!cuda_utils::check_cuda(
            incident_plane_wave_derivative_wrapper(buffers.field, x, y, z, k0,
                                                   amplitude, ss, pp, theta, phi,
                                                   selector, circular ? 1 : 0),
            "plane-wave derivative", "launch CUDA kernel")) {
        return false;
    }

    CUCOMPLEX host_output[9]{};
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(host_output, buffers.field,
                                           sizeof(host_output),
                                           cuda_utils::TransferCategory::IncidentField),
            "plane-wave derivative", "download derivative tensor")) {
        return false;
    }
    for (int i = 0; i < 9; ++i) {
        output[i] = {host_output[i].x, host_output[i].y};
    }
    return true;
}

//-----------------------------------------------------------
bool calculate_plane_wave_derivatives_batch(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    const int* tabdip, int coordinate_count, int dipole_count, CU_REAL k0,
    const FortranComplex* amplitude, CU_REAL ss, CU_REAL pp,
    CU_REAL theta, CU_REAL phi, int circular, FortranComplex* output) {
    if (!xs || !ys || !zs || !tabdip || !amplitude || !output ||
        coordinate_count <= 0 || dipole_count <= 0 || k0 <= CU_REAL(0) ||
        (circular != 0 && circular != 1) ||
        size_t(dipole_count) > size_t(INT_MAX) / 9 ||
        size_t(dipole_count) > std::numeric_limits<size_t>::max() / 9 / sizeof(CUCOMPLEX) ||
        !initialize_cuda_device())
        return false;

    const size_t count = size_t(dipole_count);
    std::vector<CU_REAL> point_x(count, CU_REAL(0));
    std::vector<CU_REAL> point_y(count, CU_REAL(0));
    std::vector<CU_REAL> point_z(count, CU_REAL(0));
    for (size_t point = 0; point < count; ++point) {
        const int one_based_index = tabdip[point];
        if (one_based_index == 0)
            continue;
        if (one_based_index < 0 || one_based_index > coordinate_count)
            return false;
        const size_t coordinate = size_t(one_based_index - 1);
        point_x[point] = xs[coordinate];
        point_y[point] = ys[coordinate];
        point_z[point] = zs[coordinate];
    }

    cuda_data::IncidentFieldBufferRequest request{};
    request.coordinate_count = count;
    request.field_count = count * 9;
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_incident_field_buffers(request, &buffers) ||
        !buffers.xs || !buffers.ys || !buffers.zs || !buffers.field)
        return false;

    const size_t coordinate_bytes = count * sizeof(CU_REAL);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.xs, point_x.data(), coordinate_bytes,
                cuda_utils::TransferCategory::Geometry),
            "plane-wave derivative batch", "upload x coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.ys, point_y.data(), coordinate_bytes,
                cuda_utils::TransferCategory::Geometry),
            "plane-wave derivative batch", "upload y coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.zs, point_z.data(), coordinate_bytes,
                cuda_utils::TransferCategory::Geometry),
            "plane-wave derivative batch", "upload z coordinates"))
        return false;

    if (!cuda_utils::check_cuda(
            incident_plane_wave_derivative_batch_wrapper(
                buffers.xs, buffers.ys, buffers.zs, buffers.field, dipole_count,
                k0, make_cuda_complex(amplitude->real, amplitude->imag), ss, pp,
                theta, phi, circular),
            "plane-wave derivative batch", "launch batched CUDA kernel"))
        return false;

    std::vector<CUCOMPLEX> host_derivatives(count * 9);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(
                host_derivatives.data(), buffers.field,
                host_derivatives.size() * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::IncidentField),
            "plane-wave derivative batch", "download derivative tensor"))
        return false;

    for (size_t value = 0; value < host_derivatives.size(); ++value)
        output[value] = {host_derivatives[value].x, host_derivatives[value].y};
    return true;
}
} // namespace

struct PlaneWaveGrid {
    const CU_INT* device_tabdip = nullptr;
    int nx = 0, ny = 0, nz = 0;
    int anchor_ix = 0, anchor_iy = 0, anchor_iz = 0;
    REAL anchor_x = 0, anchor_y = 0, anchor_z = 0;
    REAL cell_size = 0;
};

// The Fortran-compatible illumination wrappers do not receive tabdip or
// lattice dimensions explicitly. Recover the CPU-only descriptor published
// by the shared geometry stage. This avoids allocating/uploading GPU xyz
// arrays when the resident tabdip mapping determines every voxel position.
struct HostGridRequest {
    cuda_data::HostRegularGeometry geometry{};
    IfddaIncidentRegularGridSpec spec{};
};
static const IfddaIncidentRegularGridSpec* ifdda_fortran_grid_request(
    const REAL* xs, const REAL* ys, const REAL* zs,
    const int* count, HostGridRequest* request) {
    if (!request || !count ||
        !cuda_data::get_host_regular_geometry(
            xs, ys, zs, *count, &request->geometry))
        return nullptr;
    const auto& g = request->geometry;
    request->spec = {g.tabdip, &g.nx, &g.ny, &g.nz,
                     &g.compact_count, &g.cell_size};
    // ifdda_prepare_incident_regular_grid validates the complete map and
    // the compact coordinate ordering; irregular meshes automatically fall
    // back to explicit coordinates.
    return &request->spec;
}

static bool ifdda_prepare_incident_regular_grid(
    const REAL* xs, const REAL* ys, const REAL* zs, const int* tabdip,
    const int* nx, const int* ny, const int* nz, const int* grid_count,
    const REAL* cell_size, const int* count, PlaneWaveGrid* grid) {
    if (!xs || !ys || !zs || !tabdip || !nx || !ny || !nz ||
        !grid_count || !cell_size || !count || !grid ||
        *nx <= 0 || *ny <= 0 || *nz <= 0 || *count <= 0 ||
        *nx > INT_MAX / *ny || *nx * *ny > INT_MAX / *nz ||
        *grid_count != *nx * *ny * *nz || *grid_count < *count ||
        *cell_size <= REAL(0)) return false;
    int first_flat = -1, occupied = 0;
    for (int flat = 0; flat < *grid_count; ++flat) {
        if (!tabdip[flat]) continue;
        if (tabdip[flat] != ++occupied) return false;
        if (first_flat < 0) first_flat = flat;
    }
    if (occupied != *count || first_flat < 0) return false;
    PlaneWaveGrid candidate{};
    candidate.nx = *nx; candidate.ny = *ny; candidate.nz = *nz;
    candidate.cell_size = *cell_size;
    candidate.anchor_ix = first_flat % *nx;
    candidate.anchor_iy = (first_flat / *nx) % *ny;
    candidate.anchor_iz = first_flat / (*nx * *ny);
    candidate.anchor_x = xs[0]; candidate.anchor_y = ys[0];
    candidate.anchor_z = zs[0];
    const REAL epsilon = REAL(16) * std::numeric_limits<REAL>::epsilon();
    for (int flat = 0; flat < *grid_count; ++flat) {
        const int slot = tabdip[flat];
        if (!slot) continue;
        const int ix = flat % *nx;
        const int iy = (flat / *nx) % *ny;
        const int iz = flat / (*nx * *ny);
        const REAL expected[3] = {
            std::fma(REAL(ix - candidate.anchor_ix), candidate.cell_size, candidate.anchor_x),
            std::fma(REAL(iy - candidate.anchor_iy), candidate.cell_size, candidate.anchor_y),
            std::fma(REAL(iz - candidate.anchor_iz), candidate.cell_size, candidate.anchor_z)};
        const size_t compact = size_t(slot - 1);
        const REAL observed[3] = {xs[compact], ys[compact], zs[compact]};
        for (int axis = 0; axis < 3; ++axis) {
            const REAL scale = std::max(candidate.cell_size,
                std::max(std::abs(expected[axis]), std::abs(observed[axis])));
            if (std::abs(expected[axis] - observed[axis]) > epsilon * scale)
                return false;
        }
    }
    candidate.device_tabdip = cuda_data::get_tapdip(size_t(*grid_count));
    if (!candidate.device_tabdip) return false;
    *grid = candidate;
    return true;
}

static IfddaIncidentGrid ifdda_kernel_incident_grid(const PlaneWaveGrid& grid,
                                                     int compact_count) {
    return {grid.device_tabdip, grid.nx, grid.ny, grid.nx * grid.ny * grid.nz,
            compact_count, grid.anchor_ix, grid.anchor_iy, grid.anchor_iz,
            grid.anchor_x, grid.anchor_y, grid.anchor_z, grid.cell_size};
}

static const PlaneWaveGrid* ifdda_resolve_incident_grid(
    const IfddaIncidentRegularGridSpec* request,
    const REAL* xs, const REAL* ys, const REAL* zs,
    const int* count, PlaneWaveGrid* result) {
    if (!request) return nullptr;
    return ifdda_prepare_incident_regular_grid(
        xs, ys, zs, request->tabdip, request->nx, request->ny, request->nz,
        request->grid_count, request->cell_size, count, result) ? result : nullptr;
}

//-----------------------------------------------------------
extern "C" void ifdda_plane_wave_derivatives_cuda_(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, const int* tabdip,
    const int* coordinate_count, const int* dipole_count, const CU_REAL* k0,
    const COMPLEX* amplitude, const CU_REAL* ss, const CU_REAL* pp,
    const CU_REAL* theta, const CU_REAL* phi, const int* circular,
    COMPLEX* output, int* status_out) {
    if (status_out)
        *status_out = 1;
    if (!status_out || !coordinate_count || !dipole_count || !k0 || !ss || !pp ||
        !theta || !phi || !circular || !amplitude || !output || *dipole_count <= 0)
        return;
    FortranComplex f_amplitude = to_fortran_complex(*amplitude);
    std::vector<FortranComplex> f_output(size_t(*dipole_count) * 9);
    if (!calculate_plane_wave_derivatives_batch(
            xs, ys, zs, tabdip, *coordinate_count, *dipole_count, *k0,
            &f_amplitude, *ss, *pp, *theta, *phi, *circular, f_output.data()))
        return;
    for (size_t i = 0; i < f_output.size(); ++i) output[i] = to_host_complex(f_output[i]);
    *status_out = 0;
}

//-----------------------------------------------------------


//-----------------------------------------------------------
extern "C" void ondeplane_(
    REAL* x, REAL* y, REAL* z, REAL* k0,
    const COMPLEX* eps0, const COMPLEX* e0,
    REAL* ss, REAL* pp, REAL* theta, REAL* phi,
    COMPLEX* ex, COMPLEX* ey, COMPLEX* ez,
    int* nstop, char* infostr, std::size_t infostr_length) {
    if (!x || !y || !z || !k0 || !eps0 || !e0 || !ss || !pp || !theta || !phi ||
        !ex || !ey || !ez || !nstop || !infostr) {
        return;
    }
    std::fill(infostr, infostr + infostr_length, ' ');
    *nstop = 0;

    const REAL xs[1] = {*x};
    const REAL ys[1] = {*y};
    const REAL zs[1] = {*z};
    const int count = 1;
    const int circular = 0;
    const CU_REAL pp_value = *pp;
    const COMPLEX medium = *eps0;
    const COMPLEX amplitude = *e0;
    COMPLEX field[3]{};
    int cuda_status = 1;

    ifdda_plane_wave_field_cuda_execute(xs, ys, zs, &count, k0, &medium, &amplitude,
                                     ss, &pp_value, theta, phi, &circular, field,
                                     nullptr, &cuda_status);
    if (cuda_status != 0) {
        *nstop = cuda_status == 2 ? -1 : 1;
        const char* message = cuda_status == 2
                                 ? "problem in plane incident wave"
                                 : "CUDA plane-wave illumination failed";
        const std::size_t message_length =
            std::min(std::strlen(message), infostr_length);
        std::memcpy(infostr, message, message_length);
        return;
    }

    *ex = field[0];
    *ey = field[1];
    *ez = field[2];
}

//-----------------------------------------------------------
extern "C" void ondeplanekxky_(
    const REAL* x, const REAL* y, const REAL* z, const REAL* k0,
    const COMPLEX* e0, const REAL* ss, const REAL* pp,
    const REAL* k0x, const REAL* k0y,
    COMPLEX* ex, COMPLEX* ey, COMPLEX* ez,
    int* nstop, char* infostr, std::size_t infostr_length) {
    if (!x || !y || !z || !k0 || !e0 || !ss || !pp || !k0x || !k0y ||
        !ex || !ey || !ez || !nstop || !infostr) {
        return;
    }

    std::fill(infostr, infostr + infostr_length, ' ');
    *nstop = 0;
    if (*k0 <= CU_REAL(0)) {
        *nstop = -1;
        const char* message = "problem in plane incident wave";
        std::memcpy(infostr, message,
                    std::min(std::strlen(message), infostr_length));
        return;
    }

    const CU_REAL transverse_k = std::sqrt((*k0x) * (*k0x) + (*k0y) * (*k0y));
    if (transverse_k > *k0) {
        *nstop = -1;
        const char* message = "problem in plane incident wave";
        std::memcpy(infostr, message,
                    std::min(std::strlen(message), infostr_length));
        return;
    }

    const CU_REAL theta = std::asin(std::min(CU_REAL(1), transverse_k / *k0)) *
                          CU_REAL(180) / pi;
    const CU_REAL phi = transverse_k <= *k0 / CU_REAL(1000)
                            ? CU_REAL(0)
                            : std::atan2(*k0y, *k0x) * CU_REAL(180) / pi;
    const REAL xs[1] = {*x};
    const REAL ys[1] = {*y};
    const REAL zs[1] = {*z};
    const int count = 1;
    const int linear = 0;
    const COMPLEX vacuum_permittivity(REAL(1), REAL(0));
    const COMPLEX amplitude = *e0;
    COMPLEX field[3]{};
    int cuda_status = 1;

    ifdda_plane_wave_field_cuda_execute(xs, ys, zs, &count, k0, &vacuum_permittivity,
                                     &amplitude, ss, pp, &theta, &phi, &linear,
                                     field, nullptr, &cuda_status);
    if (cuda_status != 0) {
        *nstop = cuda_status == 2 ? -1 : 1;
        const char* message = cuda_status == 2
                                 ? "problem in plane incident wave"
                                 : "CUDA plane-wave illumination failed";
        std::memcpy(infostr, message,
                    std::min(std::strlen(message), infostr_length));
        return;
    }

    *ex = field[0];
    *ey = field[1];
    *ez = field[2];
}

//-----------------------------------------------------------
extern "C" void ondecirce_(
    const REAL* x, const REAL* y, const REAL* z, const REAL* k0,
    const COMPLEX* e0, const REAL* ss, const REAL* theta,
    const REAL* phi, COMPLEX* ex, COMPLEX* ey, COMPLEX* ez) {
    if (!x || !y || !z || !k0 || !e0 || !ss || !theta || !phi || !ex || !ey || !ez) {
        return;
    }

    const REAL xs[1] = {*x};
    const REAL ys[1] = {*y};
    const REAL zs[1] = {*z};
    const CU_REAL vacuum_epsilon = CU_REAL(1);
    const CU_REAL pp = CU_REAL(0);
    const int count = 1;
    const int circular = 1;
    const COMPLEX medium(vacuum_epsilon, REAL(0));
    const COMPLEX amplitude = *e0;
    COMPLEX field[3]{};
    int cuda_status = 1;

    ifdda_plane_wave_field_cuda_execute(xs, ys, zs, &count, k0, &medium, &amplitude,
                                     ss, &pp, theta, phi, &circular, field,
                                     nullptr, &cuda_status);
    if (cuda_status != 0) {
        return;
    }

    *ex = field[0];
    *ey = field[1];
    *ez = field[2];
}

//-----------------------------------------------------------
extern "C" void ondeplaned_(
    REAL* x, REAL* y, REAL* z, REAL* k0,
    const COMPLEX* amplitude, REAL* ss, REAL* pp,
    REAL* theta, REAL* phi, const int* selector,
    COMPLEX derivative[3][3]) {
    if (!x || !y || !z || !k0 || !amplitude || !ss || !pp || !theta || !phi ||
        !selector || !derivative) {
        return;
    }
    const CUCOMPLEX cuda_amplitude = make_cuda_complex(amplitude->real(), amplitude->imag());
    FortranComplex result[9]{};
    if (!calculate_plane_wave_derivative(*x, *y, *z, *k0, cuda_amplitude,
                                         *ss, *pp, *theta, *phi, *selector,
                                         false, result)) {
        std::fill(result, result + 9, FortranComplex{0, 0});
    }
    for (int i = 0; i < 9; ++i) derivative[i / 3][i % 3] = to_host_complex(result[i]);
}

//-----------------------------------------------------------
extern "C" void ondecirced_(
    const REAL* x, const REAL* y, const REAL* z, const REAL* k0,
    const COMPLEX* amplitude, const REAL* ss,
    const REAL* theta, const REAL* phi, const int* selector,
    COMPLEX derivative[3][3]) {
    if (!x || !y || !z || !k0 || !amplitude || !ss || !theta || !phi ||
        !selector || !derivative) {
        return;
    }
    const CUCOMPLEX cuda_amplitude = make_cuda_complex(amplitude->real(), amplitude->imag());
    FortranComplex result[9]{};
    if (!calculate_plane_wave_derivative(*x, *y, *z, *k0, cuda_amplitude,
                                         *ss, CU_REAL(0), *theta, *phi,
                                         *selector, true, result)) {
        std::fill(result, result + 9, FortranComplex{0, 0});
    }
    for (int i = 0; i < 9; ++i) derivative[i / 3][i % 3] = to_host_complex(result[i]);
}

//-----------------------------------------------------------
extern "C" void ondeplanedmulti_(
    REAL* x, REAL* y, REAL* z, REAL* k0,
    const COMPLEX amplitudes[10], const REAL ss[10], const REAL pp[10],
    const REAL theta[10], const REAL phi[10], const int* incidence_count,
    const int* selector, COMPLEX derivative[3][3], int* nstop,
    char* infostr, std::size_t infostr_length) {
    if (!x || !y || !z || !k0 || !amplitudes || !ss || !pp || !theta || !phi ||
        !incidence_count || !selector || !derivative || !nstop || !infostr) {
        return;
    }

    std::fill(infostr, infostr + infostr_length, ' ');
    FortranComplex total[9]{};
    *nstop = 0;
    if (*incidence_count < 0 || *incidence_count > 10 ||
        *selector < 1 || *selector > 4) {
        const char* message = *incidence_count > 10
                                  ? "too many plane wave"
                                  : "wrong value for test";
        *nstop = 1;
        std::memcpy(infostr, message,
                    std::min(std::strlen(message), infostr_length));
        return;
    }

    FortranComplex contribution[9]{};
    for (int incidence = 0; incidence < *incidence_count; ++incidence) {
        const CUCOMPLEX amplitude = make_cuda_complex(
            amplitudes[incidence].real(), amplitudes[incidence].imag());
        if (!calculate_plane_wave_derivative(
                *x, *y, *z, *k0, amplitude, ss[incidence], pp[incidence],
                theta[incidence], phi[incidence], *selector, false,
                contribution)) {
            *nstop = 1;
            const char* message = "CUDA plane-wave derivative failed";
            std::memcpy(infostr, message,
                        std::min(std::strlen(message), infostr_length));
            for (int i = 0; i < 9; ++i) derivative[i / 3][i % 3] = COMPLEX(0, 0);
            return;
        }

        if (*selector == 4) {
            for (int i = 0; i < 9; ++i) {
                total[i].real += contribution[i].real;
                total[i].imag += contribution[i].imag;
            }
        } else {
            const int first = 3 * (*selector - 1);
            for (int i = first; i < first + 3; ++i) {
                total[i].real += contribution[i].real;
                total[i].imag += contribution[i].imag;
            }
        }
    }
    for (int i = 0; i < 9; ++i) derivative[i / 3][i % 3] = to_host_complex(total[i]);
}

//-----------------------------------------------------------
void ifdda_rigorous_gaussian_field_cuda_execute(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count, const REAL *x0,
    const REAL *y0, const REAL *z0, const REAL *theta_degrees, const REAL *phi_degrees,
    const REAL *waist, const REAL *k0, const REAL *ss, const REAL *pp,
    const COMPLEX *e0, const REAL *tolerance, const int *circular, const int *nloin,
    COMPLEX *output, int *status_out, bool device_coordinates,
    CUCOMPLEX *external_device_output,
    const IfddaIncidentRegularGridSpec* grid_request,
    const CU_REAL* xs_device, const CU_REAL* ys_device,
    const CU_REAL* zs_device) {
    PlaneWaveGrid checked_grid{};
    const PlaneWaveGrid* regular_grid = ifdda_resolve_incident_grid(
        grid_request, xs, ys, zs, count, &checked_grid);
    if (!external_device_output && grid_request && count && *count > 0)
        external_device_output = cuda_data::get_ff0(size_t(*count));
    if (status_out)
        *status_out = 1;
    if ((device_coordinates ? (!xs_device || !ys_device || !zs_device)
                            : (!xs || !ys || !zs)) ||
        !count || !x0 || !y0 || !z0 || !theta_degrees || !phi_degrees ||
        !waist || !k0 || !ss || !pp || !e0 || !tolerance || !circular || !nloin ||
        (!output && !external_device_output) || !status_out || *count <= 0 ||
        *waist <= CU_REAL(0) || *k0 <= CU_REAL(0) ||
        *tolerance <= CU_REAL(0) || (*circular != 0 && *circular != 1) || *nloin < 0 ||
        *nloin > 2 || (!*circular && (*pp < CU_REAL(0) || *pp > CU_REAL(1))))
        return;
    if (!initialize_cuda_device())
        return;

    std::lock_guard<std::mutex> workspace_lock(rigorous_gaussian_workspace_mutex);
    cuda_data::RigorousGaussianDeviceBuffers buffers{};
    const size_t n = size_t(*count);
    if (!cuda_data::get_device_rigorous_gaussian_buffers(
            n, !device_coordinates && !regular_grid, !external_device_output, &buffers))
        return;
    using cuda_utils::TransferCategory;
    cudaError_t status = cudaSuccess;
    {
        std::lock_guard<std::mutex> lock(gaussian_gkp_table_mutex);
        if (!gaussian_gkp_tables_uploaded) {
            status = cuda_data::copy_host_to_device(buffers.gkp_nodes, gp_x, sizeof(gp_x),
                                                    TransferCategory::IncidentField);
            if (status == cudaSuccess)
                status = cuda_data::copy_host_to_device(buffers.gkp_weights, gp_w, sizeof(gp_w),
                                                        TransferCategory::IncidentField);
            if (status == cudaSuccess)
                gaussian_gkp_tables_uploaded = true;
        }
    }
    if (status == cudaSuccess && !device_coordinates && !regular_grid)
        status = cuda_data::copy_host_to_device(buffers.xs, xs, n * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess && !device_coordinates && !regular_grid)
        status = cuda_data::copy_host_to_device(buffers.ys, ys, n * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess && !device_coordinates && !regular_grid)
        status = cuda_data::copy_host_to_device(buffers.zs, zs, n * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    const CU_REAL *device_x = regular_grid ? nullptr : device_coordinates ? xs_device : buffers.xs;
    const CU_REAL *device_y = regular_grid ? nullptr : device_coordinates ? ys_device : buffers.ys;
    const CU_REAL *device_z = regular_grid ? nullptr : device_coordinates ? zs_device : buffers.zs;
    CUCOMPLEX *device_field = external_device_output ? external_device_output : buffers.field;
    if (status == cudaSuccess) {
        if (regular_grid)
            status = rigorous_gaussian_incident_grid_wrapper(
                ifdda_kernel_incident_grid(*regular_grid, *count),
                *x0, *y0, *z0, *theta_degrees, *phi_degrees, *waist, *k0,
                *ss, *pp, *e0, *tolerance, *circular, *nloin,
                buffers.gkp_nodes, buffers.gkp_weights, device_field, buffers.convergence);
        else
            status = rigorous_gaussian_incident_wrapper(
                device_x, device_y, device_z, *count, *x0, *y0, *z0, *theta_degrees, *phi_degrees,
                *waist, *k0, *ss, *pp, *e0, *tolerance, *circular, *nloin, buffers.gkp_nodes,
                buffers.gkp_weights, device_field, buffers.convergence);
    }
    if (status == cudaSuccess)
        if (regular_grid)
            status = rigorous_gaussian_fortran_cutoff_grid_wrapper(
                ifdda_kernel_incident_grid(*regular_grid, *count),
                *x0, *y0, *z0, *theta_degrees, *phi_degrees, *waist, *k0,
                *ss, *pp, *e0, *circular, device_field);
        else
            status = rigorous_gaussian_fortran_cutoff_wrapper(
                device_x, device_y, device_z, *count, *x0, *y0, *z0,
                *theta_degrees, *phi_degrees, *waist, *k0, *ss, *pp, *e0,
                *circular, device_field);
    const bool trace_convergence = std::getenv("IFDDA_TRACE_GAUSSIAN") != nullptr;
    if (status == cudaSuccess && trace_convergence) {
        std::vector<int> convergence(n, -1);
        status = cuda_data::copy_device_to_host(convergence.data(), buffers.convergence,
                                                n * sizeof(int), TransferCategory::Result);
        const size_t failed_quadratures = size_t(std::count_if(
            convergence.begin(), convergence.end(), [](int value) { return value != 0; }));
        // Fortran's Patterson routine returns its final-order estimate even when
        // epsrel is not reached, so this is diagnostic rather than fatal.
        if (status == cudaSuccess && failed_quadratures)
            std::fprintf(stderr,
                "[rigorous Gaussian CUDA] tolerance not reached at %zu/%zu dipoles\\n",
                failed_quadratures, n);
    }
    if (status == cudaSuccess && output)
        status = cuda_data::copy_device_complex_to_host(output, device_field, 3 * n,
                                                        TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA rigorous Gaussian illumination failed: %s\n",
                     cudaGetErrorString(status));
        *status_out = 2;
        return;
    }
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_rigorous_gaussian_field_cuda_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count,
    const REAL *x0, const REAL *y0, const REAL *z0,
    const REAL *theta_degrees, const REAL *phi_degrees,
    const REAL *waist, const REAL *k0, const REAL *ss, const REAL *pp,
    const COMPLEX *e0, const REAL *tolerance, const int *circular,
    const int *nloin, COMPLEX *output, int *status_out) {
    if (!count || *count < 0 || !e0 || !output) {
        if (status_out) *status_out = 1;
        return;
    }
    CUCOMPLEX *device_ff0 = cuda_data::get_ff0(size_t(*count));
    if (!device_ff0) {
        if (status_out) *status_out = 1;
        return;
    }
    HostGridRequest host_grid{};
    const auto* grid = ifdda_fortran_grid_request(xs, ys, zs, count, &host_grid);
    ifdda_rigorous_gaussian_field_cuda_execute(
        xs, ys, zs, count, x0, y0, z0, theta_degrees, phi_degrees, waist, k0,
        ss, pp, e0, tolerance, circular, nloin, nullptr, status_out, false,
        device_ff0, grid);
    if (status_out && *status_out == 0)
        if (cuda_data::copy_device_complex_to_host(output, device_ff0,
                3 * size_t(*count), cuda_utils::TransferCategory::IncidentField) != cudaSuccess)
            *status_out = 1;
}


//-----------------------------------------------------------
extern "C" void ifdda_rigorous_gaussian_field_cuda_device_coordinates_(
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const int *count, const REAL *x0,
    const REAL *y0, const REAL *z0, const REAL *theta_degrees,
    const REAL *phi_degrees, const REAL *waist, const REAL *k0,
    const REAL *ss, const REAL *pp, const COMPLEX *e0,
    const REAL *tolerance, const int *circular, const int *nloin,
    CUCOMPLEX *output_device, int *status_out) {
    ifdda_rigorous_gaussian_field_cuda_execute(
        nullptr, nullptr, nullptr, count, x0, y0, z0, theta_degrees,
        phi_degrees, waist, k0, ss, pp, e0, tolerance, circular, nloin,
        nullptr, status_out, true, output_device, nullptr,
        xs_device, ys_device, zs_device);
}

namespace {
//-----------------------------------------------------------
REAL sign_fortran(REAL magnitude, REAL sign_source) {
    return std::signbit(sign_source) ? -magnitude : magnitude;
}
} // namespace

//-----------------------------------------------------------
void ifdda_plane_wave_field_cuda_execute(
    const REAL *xs, const REAL *ys, const REAL *zs,
    const int *count, const REAL *k0,
    const COMPLEX *eps0, const COMPLEX *e0,
    const REAL *ss, const REAL *pp_input,
    const REAL *theta_deg, const REAL *phi_deg, const int *circular,
    COMPLEX *output, CUCOMPLEX **resident_output, int *status_out,
    const CU_REAL *xs_device, const CU_REAL *ys_device, const CU_REAL *zs_device,
    CUCOMPLEX *external_device_output,
    const IfddaIncidentRegularGridSpec *grid_request) {
    PlaneWaveGrid checked_grid{};
    const PlaneWaveGrid* regular_grid = ifdda_resolve_incident_grid(
        grid_request, xs, ys, zs, count, &checked_grid);
    if (!external_device_output && grid_request && count && *count > 0)
        external_device_output = cuda_data::get_ff0(size_t(*count));
    const bool device_coordinates = xs_device || ys_device || zs_device;
    if (resident_output)
        *resident_output = nullptr;
    if (status_out)
        *status_out = 1;
    if ((device_coordinates ? (!xs_device || !ys_device || !zs_device)
                            : (!xs || !ys || !zs)) ||
        !count || !k0 || !eps0 || !e0 || !ss || !pp_input || !theta_deg ||
        !phi_deg || !circular ||
        (!output && !resident_output && !external_device_output) ||
        !status_out || *count < 0)
        return;

    const REAL theta = *theta_deg * pi / REAL(180);
    const REAL phi = *phi_deg * pi / REAL(180);
    const REAL st = std::sin(theta), ct = std::cos(theta);
    const REAL sp = std::sin(phi), cp = std::cos(phi);
    const Complex incident = *e0;
    const Complex epsilon = *eps0;
    REAL kx = 0, ky = 0;
    Complex kz(0, 0), ex(0, 0), ey(0, 0), ez(0, 0);

    if (*circular) {
        const REAL us = REAL(1) / std::sqrt(REAL(2));
        const Complex e_s = incident / std::sqrt(REAL(2));
        const Complex e_p = incident * Complex(0, sign_fortran(us, *ss));
        kx = *k0 * st * cp;
        ky = *k0 * st * sp;
        kz = Complex(*k0 * ct, 0);
        ex = -e_s * sp + e_p * ct * cp;
        ey = e_s * cp + e_p * ct * sp;
        ez = -e_p * st;
    } else {
        REAL pp = *pp_input;
        REAL p = std::sqrt(std::max(REAL(0), pp));
        REAL s = std::sqrt(std::max(REAL(0), REAL(1) - pp));
        REAL phi_degrees = *phi_deg;
        if (near_value(pp, REAL(3))) {
            if (phi_degrees < 0)
                phi_degrees += REAL(360);
            pp = (sp * sp) / (ct * ct * cp * cp + sp * sp);
            if (phi_degrees < 90 && phi_degrees >= 0) {
                p = std::sqrt(pp);
                s = std::sqrt(1 - pp);
            } else if (phi_degrees < 180 && phi_degrees >= 90) {
                p = std::sqrt(pp);
                s = -std::sqrt(1 - pp);
            } else if (phi_degrees < 270 && phi_degrees >= 180) {
                p = -std::sqrt(pp);
                s = -std::sqrt(1 - pp);
            } else if (phi_degrees < 360 && phi_degrees >= 270) {
                p = -std::sqrt(pp);
                s = std::sqrt(1 - pp);
            }
        }
        if (near_value(pp, REAL(2))) {
            pp = (cp * cp) / (ct * ct * sp * sp + cp * cp);
            if (phi_degrees < 0)
                phi_degrees += REAL(360);
            if (phi_degrees < 90 && phi_degrees >= 0) {
                p = std::sqrt(pp);
                s = -std::sqrt(1 - pp);
            } else if (phi_degrees < 180 && phi_degrees >= 90) {
                p = -std::sqrt(pp);
                s = -std::sqrt(1 - pp);
            } else if (phi_degrees < 270 && phi_degrees >= 180) {
                p = -std::sqrt(pp);
                s = std::sqrt(1 - pp);
            } else if (phi_degrees < 360 && phi_degrees >= 270) {
                p = std::sqrt(pp);
                s = std::sqrt(1 - pp);
            }
        }
        if (!(pp >= 0 && pp <= 1)) {
            *status_out = 2;
            return;
        }

        const REAL direction = sign_fortran(REAL(1), theta);
        const Complex e_s = incident * (s * direction);
        const Complex e_p = incident * (p * direction);
        ex = -e_s * sp + e_p * ct * cp;
        ey = e_s * cp + e_p * ct * sp;
        const REAL eps_abs = std::abs(epsilon);
        if (eps_abs < REAL(1.0000001) && eps_abs > REAL(0.999999)) {
            kx = *k0 * st * cp;
            ky = *k0 * st * sp;
            kz = Complex(*k0 * ct, 0);
            ez = -e_p * st;
        } else {
            const Complex refractive = std::sqrt(epsilon);
            const REAL kr = std::real((*k0) * refractive);
            kx = kr * st * cp;
            ky = kr * st * sp;
            kz = std::sqrt((*k0) * (*k0) * epsilon - kx * kx - ky * ky);
            if (std::abs(kz) < REAL(1e-7) * (*k0)) {
                *status_out = 3;
                return;
            }
            ez = -(ex * kx + ey * ky) / kz;
        }
    }

    if (!initialize_cuda_device())
        return;
    const size_t n = size_t(*count);
    if (regular_grid) {
        if (!regular_grid->device_tabdip || !external_device_output)
            return;
        const cudaError_t grid_status = incident_plane_wave_grid_wrapper(
            regular_grid->device_tabdip, external_device_output,
            regular_grid->nx, regular_grid->ny, regular_grid->nz, *count,
            regular_grid->anchor_ix, regular_grid->anchor_iy, regular_grid->anchor_iz,
            regular_grid->anchor_x, regular_grid->anchor_y, regular_grid->anchor_z,
            regular_grid->cell_size, kx, ky, kz, ex, ey, ez);
        if (grid_status != cudaSuccess) {
            std::fprintf(stderr, "CUDA incident plane-wave grid failed: %s\n",
                         cudaGetErrorString(grid_status));
            return;
        }
        if (resident_output) *resident_output = external_device_output;
        if (output && cuda_data::copy_device_complex_to_host(
                output, external_device_output, 3 * n,
                cuda_utils::TransferCategory::IncidentField) != cudaSuccess)
            return;
        *status_out = 0;
        return;
    }
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    if (!device_coordinates || !external_device_output) {
        cuda_data::IncidentFieldBufferRequest request{};
        request.coordinate_count = device_coordinates ? 0 : n;
        request.field_count = external_device_output ? 0 : 3 * n;
        if (!cuda_data::get_device_incident_field_buffers(request, &buffers))
            return;
    }
    CU_REAL *host_x_device = device_coordinates ? nullptr : buffers.xs;
    CU_REAL *host_y_device = device_coordinates ? nullptr : buffers.ys;
    CU_REAL *host_z_device = device_coordinates ? nullptr : buffers.zs;
    const CU_REAL *dx = device_coordinates ? xs_device : host_x_device;
    const CU_REAL *dy = device_coordinates ? ys_device : host_y_device;
    const CU_REAL *dz = device_coordinates ? zs_device : host_z_device;
    CUCOMPLEX *df = external_device_output ? external_device_output : buffers.field;
    if (!dx || !dy || !dz || !df)
        return;
    using cuda_utils::TransferCategory;
    cudaError_t status = cudaSuccess;
    if (!device_coordinates)
        status = cuda_data::copy_host_to_device(
            host_x_device, xs, n * sizeof(CU_REAL), TransferCategory::Geometry);
    if (status == cudaSuccess && !device_coordinates)
        status = cuda_data::copy_host_to_device(
            host_y_device, ys, n * sizeof(CU_REAL), TransferCategory::Geometry);
    if (status == cudaSuccess && !device_coordinates)
        status = cuda_data::copy_host_to_device(
            host_z_device, zs, n * sizeof(CU_REAL), TransferCategory::Geometry);
    if (status == cudaSuccess)
        status =
            incident_plane_wave_wrapper(dx, dy, dz, df, *count, kx, ky,
                                        kz, ex, ey, ez);
    if (status == cudaSuccess && resident_output)
        *resident_output = df;
    if (status == cudaSuccess && output)
        status = cuda_data::copy_device_complex_to_host(output, df, 3 * n,
                                                        TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA incident plane wave failed: %s\n", cudaGetErrorString(status));
        return;
    }
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_plane_wave_field_cuda_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count,
    const REAL *k0, const COMPLEX *eps0, const COMPLEX *e0,
    const REAL *ss, const REAL *pp_input, const REAL *theta_deg,
    const REAL *phi_deg, const int *circular, COMPLEX *output,
    int *status_out) {
    if (!count || *count <= 0 || !eps0 || !e0 || !output) {
        if (status_out) *status_out = 1;
        return;
    }
    CUCOMPLEX* device_ff0 = cuda_data::get_ff0(size_t(*count));
    if (!device_ff0) {
        if (status_out) *status_out = 1;
        return;
    }
    HostGridRequest host_grid{};
    const auto* grid = ifdda_fortran_grid_request(xs, ys, zs, count, &host_grid);
    ifdda_plane_wave_field_cuda_execute(
        xs, ys, zs, count, k0, eps0, e0, ss, pp_input, theta_deg, phi_deg,
        circular, nullptr, nullptr, status_out,
        nullptr, nullptr, nullptr, device_ff0, grid);
    if (status_out && *status_out == 0 &&
        cuda_data::copy_device_complex_to_host(
            output, device_ff0, size_t(3) * size_t(*count),
            cuda_utils::TransferCategory::IncidentField) != cudaSuccess)
        *status_out = 1;
}

//-----------------------------------------------------------

//-----------------------------------------------------------
extern "C" void ifdda_plane_wave_field_cuda_resident_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count,
    const REAL *k0, const COMPLEX *eps0, const COMPLEX *e0,
    const REAL *ss, const REAL *pp_input, const REAL *theta_deg,
    const REAL *phi_deg, const int *circular, CUCOMPLEX **output_device,
    int *status_out) {
    if (!output_device) {
        if (status_out)
            *status_out = 1;
        return;
    }
    ifdda_plane_wave_field_cuda_execute(
        xs, ys, zs, count, k0, eps0, e0, ss, pp_input, theta_deg, phi_deg,
        circular, nullptr, output_device, status_out);
}

//-----------------------------------------------------------
extern "C" void ifdda_plane_wave_field_cuda_device_coordinates_(
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const int *count, const REAL *k0,
    const COMPLEX *eps0, const COMPLEX *e0, const REAL *ss,
    const REAL *pp_input, const REAL *theta_deg,
    const REAL *phi_deg, const int *circular,
    CUCOMPLEX *output_device, int *status_out) {
    ifdda_plane_wave_field_cuda_execute(
        nullptr, nullptr, nullptr, count, k0, eps0, e0, ss, pp_input,
        theta_deg, phi_deg, circular, nullptr, nullptr, status_out,
        xs_device, ys_device, zs_device, output_device);
}

//-----------------------------------------------------------
static bool pack_multi_plane_wave_parameters(
    CU_REAL k0, const Complex* amplitudes, const CU_REAL* pp_values,
    const CU_REAL* theta_values, const CU_REAL* phi_values, int plane_count,
    std::array<CUCOMPLEX, 60>* parameters) {
    if (!amplitudes || !pp_values || !theta_values || !phi_values ||
        !parameters || plane_count <= 0 || plane_count > 10)
        return false;
    parameters->fill(CUFFT_ZERO);
    for (int beam = 0; beam < plane_count; ++beam) {
        CU_REAL pp = pp_values[beam];
        const CU_REAL theta = theta_values[beam] * pi / CU_REAL(180);
        const CU_REAL phi = phi_values[beam] * pi / CU_REAL(180);
        const CU_REAL st = std::sin(theta), ct = std::cos(theta);
        const CU_REAL sp = std::sin(phi), cp = std::cos(phi);
        CU_REAL p = std::sqrt(std::max(CU_REAL(0), pp));
        CU_REAL s = std::sqrt(std::max(CU_REAL(0), CU_REAL(1) - pp));
        CU_REAL phi_degrees = phi_values[beam];
        if (near_value(pp, CU_REAL(3))) {
            if (phi_degrees < 0) phi_degrees += CU_REAL(360);
            pp = (sp * sp) / (ct * ct * cp * cp + sp * sp);
            if (phi_degrees < 90 && phi_degrees >= 0) { p = std::sqrt(pp); s = std::sqrt(1 - pp); }
            else if (phi_degrees < 180 && phi_degrees >= 90) { p = std::sqrt(pp); s = -std::sqrt(1 - pp); }
            else if (phi_degrees < 270 && phi_degrees >= 180) { p = -std::sqrt(pp); s = -std::sqrt(1 - pp); }
            else if (phi_degrees < 360 && phi_degrees >= 270) { p = -std::sqrt(pp); s = std::sqrt(1 - pp); }
        }
        if (near_value(pp, CU_REAL(2))) {
            pp = (cp * cp) / (ct * ct * sp * sp + cp * cp);
            if (phi_degrees < 0) phi_degrees += CU_REAL(360);
            if (phi_degrees < 90 && phi_degrees >= 0) { p = std::sqrt(pp); s = -std::sqrt(1 - pp); }
            else if (phi_degrees < 180 && phi_degrees >= 90) { p = -std::sqrt(pp); s = -std::sqrt(1 - pp); }
            else if (phi_degrees < 270 && phi_degrees >= 180) { p = -std::sqrt(pp); s = std::sqrt(1 - pp); }
            else if (phi_degrees < 360 && phi_degrees >= 270) { p = std::sqrt(pp); s = std::sqrt(1 - pp); }
        }
        if (!(pp >= 0 && pp <= 1)) return false;
        const CU_REAL direction = sign_fortran(CU_REAL(1), theta);
        const Complex es = amplitudes[beam] * (s * direction);
        const Complex ep = amplitudes[beam] * (p * direction);
        const Complex ex = -es * sp + ep * ct * cp;
        const Complex ey = es * cp + ep * ct * sp;
        const Complex ez = -ep * st;
        const size_t offset = size_t(6) * size_t(beam);
        (*parameters)[offset] = cuda_utils::device_value(Complex(k0 * st * cp, 0));
        (*parameters)[offset + 1] = cuda_utils::device_value(Complex(k0 * st * sp, 0));
        (*parameters)[offset + 2] = cuda_utils::device_value(Complex(k0 * ct, 0));
        (*parameters)[offset + 3] = cuda_utils::device_value(ex);
        (*parameters)[offset + 4] = cuda_utils::device_value(ey);
        (*parameters)[offset + 5] = cuda_utils::device_value(ez);
    }
    return true;
}

//-----------------------------------------------------------
void ifdda_multi_plane_wave_field_cuda_execute(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count,
    const REAL *k0, const COMPLEX *e0, const COMPLEX *e0m,
    const REAL *ssm, const REAL *ppm, const REAL *theta_deg,
    const REAL *phi_deg, const int *plane_count, COMPLEX *output,
    int *status_out, bool device_coordinates, CUCOMPLEX *external_device_output,
    const IfddaIncidentRegularGridSpec* grid_request,
    const CU_REAL* xs_device, const CU_REAL* ys_device,
    const CU_REAL* zs_device) {
    PlaneWaveGrid checked_grid{};
    const PlaneWaveGrid* regular_grid = ifdda_resolve_incident_grid(
        grid_request, xs, ys, zs, count, &checked_grid);
    if (!external_device_output && grid_request && count && *count > 0)
        external_device_output = cuda_data::get_ff0(size_t(*count));
    if (status_out)
        *status_out = 1;
    if ((device_coordinates ? (!xs_device || !ys_device || !zs_device)
                            : (!xs || !ys || !zs)) ||
        !count || !k0 || !e0 || !e0m || !ssm || !ppm || !theta_deg ||
        !phi_deg || !plane_count || (!output && !external_device_output) || !status_out ||
        *count < 0 || *plane_count < 0 ||
        *plane_count > 10)
        return;

    if (*plane_count == 0) {
        if (external_device_output && cudaMemset(external_device_output, 0,
            size_t(3) * size_t(*count) * sizeof(CUCOMPLEX)) != cudaSuccess)
            return;
        if (output)
            std::fill(output, output + size_t(3) * size_t(*count), COMPLEX{});
        *status_out = 0;
        return;
    }

    std::array<COMPLEX, 60> host_parameters{};
    const size_t host_parameter_count = size_t(6) * size_t(*plane_count);
    for (int b = 0; b < *plane_count; ++b) {
        CU_REAL pp = ppm[b];
        const CU_REAL theta = theta_deg[b] * pi / CU_REAL(180);
        const CU_REAL phi = phi_deg[b] * pi / CU_REAL(180);
        const CU_REAL st = std::sin(theta), ct = std::cos(theta);
        const CU_REAL sp = std::sin(phi), cp = std::cos(phi);
        CU_REAL p = std::sqrt(std::max(CU_REAL(0), pp));
        CU_REAL s = std::sqrt(std::max(CU_REAL(0), CU_REAL(1) - pp));
        CU_REAL phi_degrees = phi_deg[b];

        if (near_value(pp, CU_REAL(3))) {
            if (phi_degrees < 0)
                phi_degrees += CU_REAL(360);
            pp = (sp * sp) / (ct * ct * cp * cp + sp * sp);
            if (phi_degrees < 90 && phi_degrees >= 0) {
                p = std::sqrt(pp);
                s = std::sqrt(1 - pp);
            } else if (phi_degrees < 180 && phi_degrees >= 90) {
                p = std::sqrt(pp);
                s = -std::sqrt(1 - pp);
            } else if (phi_degrees < 270 && phi_degrees >= 180) {
                p = -std::sqrt(pp);
                s = -std::sqrt(1 - pp);
            } else if (phi_degrees < 360 && phi_degrees >= 270) {
                p = -std::sqrt(pp);
                s = std::sqrt(1 - pp);
            }
        }
        if (near_value(pp, CU_REAL(2))) {
            pp = (cp * cp) / (ct * ct * sp * sp + cp * cp);
            if (phi_degrees < 0)
                phi_degrees += CU_REAL(360);
            if (phi_degrees < 90 && phi_degrees >= 0) {
                p = std::sqrt(pp);
                s = -std::sqrt(1 - pp);
            } else if (phi_degrees < 180 && phi_degrees >= 90) {
                p = -std::sqrt(pp);
                s = -std::sqrt(1 - pp);
            } else if (phi_degrees < 270 && phi_degrees >= 180) {
                p = -std::sqrt(pp);
                s = std::sqrt(1 - pp);
            } else if (phi_degrees < 360 && phi_degrees >= 270) {
                p = std::sqrt(pp);
                s = std::sqrt(1 - pp);
            }
        }
        if (!(pp >= 0 && pp <= 1)) {
            *status_out = 2;
            return;
        }

        const Complex incident = e0m[b];
        const CU_REAL direction = sign_fortran(CU_REAL(1), theta);
        const Complex es = incident * (s * direction);
        const Complex ep = incident * (p * direction);
        const Complex ex = -es * sp + ep * ct * cp;
        const Complex ey = es * cp + ep * ct * sp;
        const Complex ez = -ep * st;
        const CU_REAL kx = *k0 * st * cp;
        const CU_REAL ky = *k0 * st * sp;
        const CU_REAL kz = *k0 * ct;
        const size_t offset = size_t(6) * size_t(b);
        host_parameters[offset] = COMPLEX(kx, 0);
        host_parameters[offset + 1] = COMPLEX(ky, 0);
        host_parameters[offset + 2] = COMPLEX(kz, 0);
        host_parameters[offset + 3] = ex;
        host_parameters[offset + 4] = ey;
        host_parameters[offset + 5] = ez;
    }

    if (!initialize_cuda_device())
        return;
    const size_t n = size_t(*count);
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    cuda_data::IncidentFieldBufferRequest request{};
    request.coordinate_count = (device_coordinates || regular_grid) ? 0 : n;
    request.field_count = external_device_output ? 0 : 3 * n;
    request.parameter_count = host_parameter_count;
    if (!cuda_data::get_device_incident_field_buffers(request, &buffers))
        return;
    CU_REAL *dx = buffers.xs;
    CU_REAL *dy = buffers.ys;
    CU_REAL *dz = buffers.zs;
    CUCOMPLEX *dp = buffers.parameters;
    CUCOMPLEX *df = buffers.field;
    if ((!device_coordinates && !regular_grid && (!dx || !dy || !dz)) || !dp ||
        (!external_device_output && !df))
        return;
    using cuda_utils::TransferCategory;
    const CU_REAL *device_x = regular_grid ? nullptr : device_coordinates ? xs_device : dx;
    const CU_REAL *device_y = regular_grid ? nullptr : device_coordinates ? ys_device : dy;
    const CU_REAL *device_z = regular_grid ? nullptr : device_coordinates ? zs_device : dz;
    CUCOMPLEX *device_field = external_device_output ? external_device_output : df;
    cudaError_t status = cudaSuccess;
    if (!device_coordinates && !regular_grid)
        status = cuda_data::copy_host_to_device(dx, xs, n * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess)
        if (!device_coordinates && !regular_grid)
            status = cuda_data::copy_host_to_device(dy, ys, n * sizeof(CU_REAL),
                                                    TransferCategory::Geometry);
    if (status == cudaSuccess && !device_coordinates && !regular_grid)
        status = cuda_data::copy_host_to_device(dz, zs, n * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_complex_to_device(dp, host_parameters.data(),
                                                        host_parameter_count,
                                                        TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = regular_grid ?
            incident_multi_plane_wave_grid_wrapper(
                ifdda_kernel_incident_grid(*regular_grid, *count), dp, device_field, *plane_count) :
            incident_multi_plane_wave_wrapper(device_x, device_y, device_z,
                                               dp, device_field, *count, *plane_count);
    if (status == cudaSuccess && output)
        status = cuda_data::copy_device_complex_to_host(output, device_field, 3 * n,
                                                        TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA multi-plane-wave illumination failed: %s\n",
                     cudaGetErrorString(status));
        return;
    }
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_multi_plane_wave_field_cuda_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count,
    const REAL *k0, const COMPLEX *e0, const COMPLEX *e0m,
    const REAL *ssm, const REAL *ppm, const REAL *theta_deg,
    const REAL *phi_deg, const int *plane_count, COMPLEX *output,
    int *status_out) {
    if (!count || *count < 0 || !plane_count || *plane_count < 0 ||
        *plane_count > 10 || !e0 || !output) {
        if (status_out) *status_out = 1;
        return;
    }
    CUCOMPLEX* device_ff0=cuda_data::get_ff0(size_t(*count));
    if (!device_ff0) {
        if (status_out) *status_out=1;
        return;
    }
    HostGridRequest host_grid{};
    const auto* grid = ifdda_fortran_grid_request(xs, ys, zs, count, &host_grid);
    ifdda_multi_plane_wave_field_cuda_execute(
        xs, ys, zs, count, k0, e0, e0m,
        ssm, ppm, theta_deg, phi_deg,
        plane_count, nullptr, status_out, false, device_ff0, grid);
    if (status_out && *status_out == 0)
        if (cuda_data::copy_device_complex_to_host(
                output, device_ff0, 3*size_t(*count),
                cuda_utils::TransferCategory::IncidentField)!=cudaSuccess)
            *status_out=1;
}


//-----------------------------------------------------------
extern "C" void ifdda_multi_plane_wave_derivatives_cuda_(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, const int* tabdip,
    const int* coordinate_count, const int* dipole_count, const CU_REAL* k0,
    const COMPLEX* amplitudes, const CU_REAL* pp_values,
    const CU_REAL* theta_values, const CU_REAL* phi_values, const int* plane_count,
    COMPLEX* output, int* status_out) {
    if (status_out)
        *status_out = 1;
    if (!xs || !ys || !zs || !tabdip || !coordinate_count || !dipole_count ||
        !k0 || !amplitudes || !pp_values || !theta_values || !phi_values ||
        !plane_count || !output || *coordinate_count <= 0 || *dipole_count <= 0 ||
        *plane_count <= 0 || *plane_count > 10 || !status_out ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / 9 / sizeof(CUCOMPLEX) ||
        !initialize_cuda_device())
        return;

    std::lock_guard<std::mutex> lock(plane_wave_derivative_mutex);
    const size_t count = size_t(*dipole_count);
    std::array<Complex, 10> host_amplitudes{};
    for (int beam = 0; beam < *plane_count; ++beam)
        host_amplitudes[size_t(beam)] = amplitudes[beam];
    std::vector<CU_REAL> point_x(count, CU_REAL(0));
    std::vector<CU_REAL> point_y(count, CU_REAL(0));
    std::vector<CU_REAL> point_z(count, CU_REAL(0));
    for (size_t point = 0; point < count; ++point) {
        const int index = tabdip[point];
        if (index == 0)
            continue;
        if (index < 0 || index > *coordinate_count)
            return;
        const size_t coordinate = size_t(index - 1);
        point_x[point] = xs[coordinate];
        point_y[point] = ys[coordinate];
        point_z[point] = zs[coordinate];
    }
    std::array<CUCOMPLEX, 60> host_parameters{};
    if (!pack_multi_plane_wave_parameters(*k0, host_amplitudes.data(), pp_values,
                                           theta_values, phi_values, *plane_count,
                                           &host_parameters))
        return;

    cuda_data::IncidentFieldBufferRequest request{};
    request.coordinate_count = count;
    request.field_count = count * 9;
    request.parameter_count = size_t(*plane_count) * 6;
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_incident_field_buffers(request, &buffers) ||
        !buffers.xs || !buffers.ys || !buffers.zs || !buffers.field || !buffers.parameters)
        return;
    const size_t coordinate_bytes = count * sizeof(CU_REAL);
    using cuda_utils::TransferCategory;
    if (!cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.xs, point_x.data(), coordinate_bytes, TransferCategory::Geometry),
            "multi-plane-wave derivative", "upload x coordinates") ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.ys, point_y.data(), coordinate_bytes, TransferCategory::Geometry),
            "multi-plane-wave derivative", "upload y coordinates") ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.zs, point_z.data(), coordinate_bytes, TransferCategory::Geometry),
            "multi-plane-wave derivative", "upload z coordinates") ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.parameters, host_parameters.data(), size_t(*plane_count) * 6 * sizeof(CUCOMPLEX),
            TransferCategory::IncidentField),
            "multi-plane-wave derivative", "upload plane parameters") ||
        !cuda_utils::check_cuda(incident_multi_plane_wave_derivative_batch_wrapper(
            buffers.xs, buffers.ys, buffers.zs, buffers.parameters, buffers.field,
            *dipole_count, *plane_count), "multi-plane-wave derivative", "launch CUDA kernel"))
        return;
    std::vector<CUCOMPLEX> host_derivatives(count * 9);
    if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
            host_derivatives.data(), buffers.field, host_derivatives.size() * sizeof(CUCOMPLEX),
            TransferCategory::IncidentField),
            "multi-plane-wave derivative", "download derivative tensor"))
        return;
    for (size_t i = 0; i < host_derivatives.size(); ++i)
        output[i] = COMPLEX(host_derivatives[i].x, host_derivatives[i].y);
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_paraxial_gaussian_derivatives_cuda_(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, const int* tabdip,
    const int* coordinate_count, const int* dipole_count, const CU_REAL* x0,
    const CU_REAL* y0, const CU_REAL* z0, const CU_REAL* theta_degrees,
    const CU_REAL* phi_degrees, const CU_REAL* waist, const CU_REAL* k0,
    const CU_REAL* ss, const CU_REAL* pp, const COMPLEX* amplitude,
    const int* circular, COMPLEX* output, int* status_out) {
    if (status_out)
        *status_out = 1;
    if (!xs || !ys || !zs || !tabdip || !coordinate_count || !dipole_count ||
        !x0 || !y0 || !z0 || !theta_degrees || !phi_degrees || !waist || !k0 ||
        !ss || !pp || !amplitude || !circular || !output || !status_out ||
        *coordinate_count <= 0 || *dipole_count <= 0 || *waist <= CU_REAL(0) ||
        *k0 <= CU_REAL(0) || (*circular != 0 && *circular != 1) ||
        (!*circular && (*pp < CU_REAL(0) || *pp > CU_REAL(1))) ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / 9 / sizeof(CUCOMPLEX) ||
        !initialize_cuda_device())
        return;

    std::lock_guard<std::mutex> lock(paraxial_gaussian_mutex);
    const CU_REAL theta = *theta_degrees * pi / CU_REAL(180);
    const CU_REAL phi = *phi_degrees * pi / CU_REAL(180);
    const CU_REAL vx = std::sin(theta) * std::cos(phi);
    const CU_REAL vy = std::sin(theta) * std::sin(phi);
    const CU_REAL vz = std::cos(theta);
    const CU_REAL machine_epsilon = PRODUITMATVECTFFT_SINGLE_PRECISION
                                       ? CU_REAL(1.0e-6) : CU_REAL(1.0e-12);
    CU_REAL rotation[9]{};
    if (vx * vx + vy * vy <= machine_epsilon) {
        rotation[0] = rotation[4] = rotation[8] = CU_REAL(1);
    } else {
        rotation[0] = CU_REAL(1) - vx * vx / (CU_REAL(1) + vz);
        rotation[3] = -vx * vy / (CU_REAL(1) + vz);
        rotation[6] = -vx;
        rotation[1] = rotation[3];
        rotation[4] = CU_REAL(1) - vy * vy / (CU_REAL(1) + vz);
        rotation[7] = -vy;
        rotation[2] = vx;
        rotation[5] = vy;
        rotation[8] = vz;
    }
    const Complex e0(amplitude->real(), amplitude->imag());
    const Complex e0s = *circular ? e0 / std::sqrt(CU_REAL(2))
                                  : e0 * std::sqrt(CU_REAL(1) - *pp);
    const Complex e0p = *circular
        ? e0 * Complex(CU_REAL(0), std::copysign(CU_REAL(1) / std::sqrt(CU_REAL(2)), *ss))
        : e0 * std::sqrt(*pp);
    const Complex e0x = -e0s * std::sin(phi) + e0p * std::cos(theta) * std::cos(phi);
    const Complex e0y = e0s * std::cos(phi) + e0p * std::cos(theta) * std::sin(phi);
    const Complex e0z = -e0p * std::sin(theta);
    const Complex e00x = rotation[0] * e0x + rotation[3] * e0y + rotation[6] * e0z;
    const Complex e00y = rotation[1] * e0x + rotation[4] * e0y + rotation[7] * e0z;
    const Complex e00z = rotation[2] * e0x + rotation[5] * e0y + rotation[8] * e0z;
    if (std::abs(e00z) >= machine_epsilon * std::abs(e0))
        return;

    const size_t count = size_t(*dipole_count);
    std::vector<CU_REAL> point_x(count, CU_REAL(0));
    std::vector<CU_REAL> point_y(count, CU_REAL(0));
    std::vector<CU_REAL> point_z(count, CU_REAL(0));
    for (size_t point = 0; point < count; ++point) {
        const int index = tabdip[point];
        if (index == 0)
            continue;
        if (index < 0 || index > *coordinate_count)
            return;
        point_x[point] = xs[index - 1];
        point_y[point] = ys[index - 1];
        point_z[point] = zs[index - 1];
    }

    cuda_data::IncidentFieldBufferRequest request{};
    request.coordinate_count = count;
    request.field_count = count * 9;
    request.real_parameter_count = 9;
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_incident_field_buffers(request, &buffers) ||
        !buffers.xs || !buffers.ys || !buffers.zs || !buffers.field ||
        !buffers.real_parameters)
        return;
    using cuda_utils::TransferCategory;
    const size_t coordinate_bytes = count * sizeof(CU_REAL);
    if (!cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.xs, point_x.data(), coordinate_bytes, TransferCategory::Geometry),
            "paraxial Gaussian derivative batch", "upload x coordinates") ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.ys, point_y.data(), coordinate_bytes, TransferCategory::Geometry),
            "paraxial Gaussian derivative batch", "upload y coordinates") ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.zs, point_z.data(), coordinate_bytes, TransferCategory::Geometry),
            "paraxial Gaussian derivative batch", "upload z coordinates") ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.real_parameters, rotation, sizeof(rotation), TransferCategory::IncidentField),
            "paraxial Gaussian derivative batch", "upload beam rotation") ||
        !cuda_utils::check_cuda(gaussianparad_derivative_batch_wrapper(
            buffers.xs, buffers.ys, buffers.zs, buffers.field, *dipole_count,
            *x0, *y0, *z0, *waist, *k0,
            make_cuda_complex(e00x.real(), e00x.imag()),
            make_cuda_complex(e00y.real(), e00y.imag()), buffers.real_parameters),
            "paraxial Gaussian derivative batch", "launch CUDA kernel"))
        return;

    std::vector<CUCOMPLEX> host_derivatives(count * 9);
    if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
            host_derivatives.data(), buffers.field,
            host_derivatives.size() * sizeof(CUCOMPLEX), TransferCategory::IncidentField),
            "paraxial Gaussian derivative batch", "download derivative tensor"))
        return;
    for (size_t value = 0; value < host_derivatives.size(); ++value)
        output[value] = COMPLEX(host_derivatives[value].x, host_derivatives[value].y);
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_multi_plane_wave_field_cuda_device_coordinates_(
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const int *count, const REAL *k0,
    const COMPLEX *e0, const COMPLEX *e0m, const REAL *ssm,
    const REAL *ppm, const REAL *theta_deg, const REAL *phi_deg,
    const int *plane_count, CUCOMPLEX *output_device, int *status_out) {
    ifdda_multi_plane_wave_field_cuda_execute(
        nullptr, nullptr, nullptr, count, k0, e0, e0m, ssm, ppm,
        theta_deg, phi_deg, plane_count, nullptr, status_out, true,
        output_device, nullptr, xs_device, ys_device, zs_device);
}

//-----------------------------------------------------------
void ifdda_paraxial_gaussian_field_cuda_execute(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count, const COMPLEX *e0,
    const COMPLEX *epsilon, const REAL *ss, const REAL *pp, const REAL *theta_deg,
    const REAL *phi_deg, const REAL *x0, const REAL *y0, const REAL *z0,
    const REAL *w0, const REAL *k0, const int *circular, COMPLEX *output,
    int *status_out, bool device_coordinates, CUCOMPLEX *external_device_output,
    const IfddaIncidentRegularGridSpec* grid_request,
    const CU_REAL* xs_device, const CU_REAL* ys_device,
    const CU_REAL* zs_device) {
    PlaneWaveGrid checked_grid{};
    const PlaneWaveGrid* regular_grid = ifdda_resolve_incident_grid(
        grid_request, xs, ys, zs, count, &checked_grid);
    if (!external_device_output && grid_request && count && *count > 0)
        external_device_output = cuda_data::get_ff0(size_t(*count));
    if (status_out)
        *status_out = 1;
    if ((device_coordinates ? (!xs_device || !ys_device || !zs_device)
                            : (!xs || !ys || !zs)) ||
        !count || !e0 || !epsilon || !ss || !pp || !theta_deg || !phi_deg ||
        !x0 || !y0 || !z0 || !w0 || !k0 || !circular ||
        (!output && !external_device_output) || !status_out || *count < 0 ||
        *circular < 0 || *circular > 1 || (!*circular && (*pp < 0 || *pp > 1)))
        return;
    if (!initialize_cuda_device())
        return;

    std::lock_guard<std::mutex> lock(paraxial_gaussian_mutex);

    const size_t n = size_t(*count);
    if (n > size_t(-1) / (3 * sizeof(CUCOMPLEX)))
        return;
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    if ((!device_coordinates && !regular_grid) || !external_device_output) {
        cuda_data::IncidentFieldBufferRequest request{};
        request.coordinate_count = (device_coordinates || regular_grid) ? 0 : n;
        request.field_count = external_device_output ? 0 : 3 * n;
        if (!cuda_data::get_device_incident_field_buffers(request, &buffers))
            return;
    }
    if ((!device_coordinates && !regular_grid && (!buffers.xs || !buffers.ys || !buffers.zs)) ||
        (!external_device_output && !buffers.field))
        return;

    const CU_REAL *device_x = regular_grid ? nullptr : device_coordinates ? xs_device : buffers.xs;
    const CU_REAL *device_y = regular_grid ? nullptr : device_coordinates ? ys_device : buffers.ys;
    const CU_REAL *device_z = regular_grid ? nullptr : device_coordinates ? zs_device : buffers.zs;
    CUCOMPLEX *device_field = external_device_output ? external_device_output : buffers.field;

    using cuda_utils::TransferCategory;
    cudaError_t status = cudaSuccess;
    if (!device_coordinates && !regular_grid)
        status = cuda_data::copy_host_to_device(buffers.xs, xs, n * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess)
        if (!device_coordinates && !regular_grid)
            status = cuda_data::copy_host_to_device(buffers.ys, ys, n * sizeof(CU_REAL),
                                                    TransferCategory::Geometry);
    if (status == cudaSuccess && !device_coordinates && !regular_grid)
        status = cuda_data::copy_host_to_device(buffers.zs, zs, n * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = regular_grid ?
            incident_paraxial_gaussian_grid_wrapper(
                ifdda_kernel_incident_grid(*regular_grid, *count), device_field,
                *e0, *epsilon, *ss, *pp, *theta_deg, *phi_deg,
                *x0, *y0, *z0, *w0, *k0, *circular) :
            incident_paraxial_gaussian_wrapper(device_x, device_y, device_z, device_field,
                                                *count, *e0, *epsilon, *ss, *pp,
                                                *theta_deg, *phi_deg, *x0, *y0, *z0, *w0, *k0,
                                                *circular);
    if (status == cudaSuccess && output)
        status = cuda_data::copy_device_complex_to_host(output, device_field, 3 * n,
                                                        TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA paraxial Gaussian illumination failed: %s\\n",
                     cudaGetErrorString(status));
        return;
    }
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_paraxial_gaussian_field_cuda_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count,
    const COMPLEX *e0, const COMPLEX *epsilon, const REAL *ss,
    const REAL *pp, const REAL *theta_deg, const REAL *phi_deg,
    const REAL *x0, const REAL *y0, const REAL *z0, const REAL *w0,
    const REAL *k0, const int *circular, COMPLEX *output, int *status_out) {
    if (!count || *count < 0 || !e0 || !epsilon || !output) {
        if (status_out) *status_out = 1;
        return;
    }
    CUCOMPLEX *device_ff0 = cuda_data::get_ff0(size_t(*count));
    if (!device_ff0) {
        if (status_out) *status_out = 1;
        return;
    }
    HostGridRequest host_grid{};
    const auto* grid = ifdda_fortran_grid_request(xs, ys, zs, count, &host_grid);
    ifdda_paraxial_gaussian_field_cuda_execute(
        xs, ys, zs, count, e0, epsilon, ss, pp, theta_deg, phi_deg, x0, y0,
        z0, w0, k0, circular, nullptr, status_out, false, device_ff0, grid);
    if (status_out && *status_out == 0)
        if (cuda_data::copy_device_complex_to_host(output, device_ff0,
                3 * size_t(*count), cuda_utils::TransferCategory::IncidentField) != cudaSuccess)
            *status_out = 1;
}


//-----------------------------------------------------------
extern "C" void ifdda_paraxial_gaussian_field_cuda_device_coordinates_(
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const int *count, const COMPLEX *e0,
    const COMPLEX *epsilon, const REAL *ss, const REAL *pp,
    const REAL *theta_deg, const REAL *phi_deg, const REAL *x0,
    const REAL *y0, const REAL *z0, const REAL *w0, const REAL *k0,
    const int *circular, CUCOMPLEX *output_device, int *status_out) {
    ifdda_paraxial_gaussian_field_cuda_execute(
        nullptr, nullptr, nullptr, count, e0, epsilon, ss, pp,
        theta_deg, phi_deg, x0, y0, z0, w0, k0, circular, nullptr,
        status_out, true, output_device, nullptr,
        xs_device, ys_device, zs_device);
}

//-----------------------------------------------------------
namespace {
void legacy_paraxial_gaussian_field(
    CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL x0, CU_REAL y0, CU_REAL z0,
    CU_REAL theta, CU_REAL phi, CU_REAL waist, CU_REAL k0, CU_REAL ss, CU_REAL pp,
    FortranComplex epsilon0, FortranComplex amplitude, bool circular,
    FortranComplex* ex, FortranComplex* ey, FortranComplex* ez,
    int* nstop, char* infostr, std::size_t infostr_length) {
    if (!ex || !ey || !ez || !nstop || !infostr)
        return;
    *ex = {CU_REAL(0), CU_REAL(0)};
    *ey = {CU_REAL(0), CU_REAL(0)};
    *ez = {CU_REAL(0), CU_REAL(0)};
    *nstop = -1;
    std::fill(infostr, infostr + infostr_length, ' ');
    if (!circular && (pp < CU_REAL(0) || pp > CU_REAL(1))) {
        const char* message = "problem in plane incident wave";
        std::memcpy(infostr, message, std::min(std::strlen(message), infostr_length));
        return;
    }
    const CU_REAL xs[1] = {x};
    const CU_REAL ys[1] = {y};
    const CU_REAL zs[1] = {z};
    const int count = 1;
    const int circular_flag = circular ? 1 : 0;
    const COMPLEX cuda_epsilon(epsilon0.real, epsilon0.imag);
    const COMPLEX cuda_amplitude(amplitude.real, amplitude.imag);
    COMPLEX output[3]{};
    int status = 1;
    ifdda_paraxial_gaussian_field_cuda_execute(
        xs, ys, zs, &count, &cuda_amplitude, &cuda_epsilon, &ss, &pp,
        &theta, &phi, &x0, &y0, &z0, &waist, &k0, &circular_flag,
        output, &status, false, nullptr);
    if (status != 0) {
        const char* message = "CUDA paraxial Gaussian illumination failed";
        std::memcpy(infostr, message, std::min(std::strlen(message), infostr_length));
        return;
    }
    *ex = {output[0].real(), output[0].imag()};
    *ey = {output[1].real(), output[1].imag()};
    *ez = {output[2].real(), output[2].imag()};
    *nstop = 0;
}
} // namespace

//-----------------------------------------------------------
extern "C" void gaussianparalinear_(
    const REAL* x, const REAL* y, const REAL* z,
    const REAL* x0, const REAL* y0, const REAL* z0,
    const REAL* theta, const REAL* phi, const REAL* waist,
    const REAL* k0, const REAL* ss, const REAL* pp,
    const COMPLEX* epsilon0, const COMPLEX* amplitude,
    COMPLEX* ex, COMPLEX* ey, COMPLEX* ez,
    int* nstop, char* infostr, std::size_t infostr_length) {
    if (!x || !y || !z || !x0 || !y0 || !z0 || !theta || !phi || !waist ||
        !k0 || !ss || !pp || !epsilon0 || !amplitude || !ex || !ey || !ez ||
        !nstop || !infostr)
        return;
    FortranComplex feps = to_fortran_complex(*epsilon0),
                   famp = to_fortran_complex(*amplitude), fx{}, fy{}, fz{};
    legacy_paraxial_gaussian_field(
        *x, *y, *z, *x0, *y0, *z0, *theta, *phi, *waist, *k0, *ss, *pp,
        feps, famp, false, &fx, &fy, &fz, nstop, infostr, infostr_length);
    *ex = to_host_complex(fx); *ey = to_host_complex(fy); *ez = to_host_complex(fz);
}

//-----------------------------------------------------------
extern "C" void gaussianparacirc_(
    const REAL* x, const REAL* y, const REAL* z,
    const REAL* x0, const REAL* y0, const REAL* z0,
    const REAL* theta, const REAL* phi, const REAL* waist,
    const REAL* k0, const REAL* handedness,
    const COMPLEX* epsilon0, const COMPLEX* amplitude,
    COMPLEX* ex, COMPLEX* ey, COMPLEX* ez,
    int* nstop, char* infostr, std::size_t infostr_length) {
    if (!x || !y || !z || !x0 || !y0 || !z0 || !theta || !phi || !waist ||
        !k0 || !handedness || !epsilon0 || !amplitude || !ex || !ey || !ez ||
        !nstop || !infostr)
        return;
    FortranComplex feps = to_fortran_complex(*epsilon0),
                   famp = to_fortran_complex(*amplitude), fx{}, fy{}, fz{};
    legacy_paraxial_gaussian_field(
        *x, *y, *z, *x0, *y0, *z0, *theta, *phi, *waist, *k0,
        *handedness, CU_REAL(0), feps, famp, true,
        &fx, &fy, &fz, nstop, infostr, infostr_length);
    *ex = to_host_complex(fx); *ey = to_host_complex(fy); *ez = to_host_complex(fz);
}

//-----------------------------------------------------------
namespace {
void rigorous_gaussian_point(
    const CU_REAL* x, const CU_REAL* y, const CU_REAL* z,
    const CU_REAL* x0, const CU_REAL* y0, const CU_REAL* z0,
    const CU_REAL* theta, const CU_REAL* phi, const CU_REAL* waist,
    const CU_REAL* k0, const CU_REAL* ss, const CU_REAL* pp,
    const FortranComplex* amplitude, const CU_REAL* tolerance,
    const int* nloin, const bool circular, FortranComplex* ex,
    FortranComplex* ey, FortranComplex* ez, int* nstop,
    char* infostr, const std::size_t infostr_length) {
    if (!ex || !ey || !ez || !nstop || !infostr)
        return;
    *ex = {CU_REAL(0), CU_REAL(0)};
    *ey = {CU_REAL(0), CU_REAL(0)};
    *ez = {CU_REAL(0), CU_REAL(0)};
    *nstop = 1;
    std::fill(infostr, infostr + infostr_length, ' ');
    if (!x || !y || !z || !x0 || !y0 || !z0 || !theta || !phi || !waist ||
        !k0 || !ss || !pp || !amplitude || !tolerance || !nloin)
        return;

    const int count = 1;
    const int circular_flag = circular ? 1 : 0;
    const COMPLEX e0(amplitude->real, amplitude->imag);
    COMPLEX field[3]{};
    int status = 1;
    ifdda_rigorous_gaussian_field_cuda_execute(
        x, y, z, &count, x0, y0, z0, theta, phi, waist, k0, ss, pp,
        &e0, tolerance, &circular_flag, nloin, field, &status, false, nullptr);
    if (status != 0) {
        const char* message = "CUDA rigorous Gaussian illumination failed";
        std::memcpy(infostr, message,
                    std::min(std::strlen(message), infostr_length));
        return;
    }
    *ex = {field[0].real(), field[0].imag()};
    *ey = {field[1].real(), field[1].imag()};
    *ez = {field[2].real(), field[2].imag()};
    *nstop = 0;
}
} // namespace

//-----------------------------------------------------------
extern "C" void gaussianchamp_(
    const REAL* x, const REAL* y, const REAL* z,
    const REAL* x0, const REAL* y0, const REAL* z0,
    const REAL* theta, const REAL* phi, const REAL* waist,
    const REAL* k0, const REAL* ss, const REAL* pp,
    const COMPLEX* amplitude, COMPLEX* ex,
    COMPLEX* ey, COMPLEX* ez, const REAL* tolerance,
    const int* nloin, int* nstop, char* infostr,
    const std::size_t infostr_length) {
    FortranComplex famp = to_fortran_complex(*amplitude), fx{}, fy{}, fz{};
    rigorous_gaussian_point(x, y, z, x0, y0, z0, theta, phi, waist,
                            k0, ss, pp, &famp, tolerance, nloin,
                            false, &fx, &fy, &fz, nstop, infostr,
                            infostr_length);
    *ex = to_host_complex(fx); *ey = to_host_complex(fy); *ez = to_host_complex(fz);
}

//-----------------------------------------------------------
extern "C" void gaussianchampcirc_(
    const REAL* x, const REAL* y, const REAL* z,
    const REAL* x0, const REAL* y0, const REAL* z0,
    const REAL* theta, const REAL* phi, const REAL* waist,
    const REAL* k0, const REAL* handedness,
    const COMPLEX* amplitude, COMPLEX* ex,
    COMPLEX* ey, COMPLEX* ez, const REAL* tolerance,
    const int* nloin, int* nstop, char* infostr,
    const std::size_t infostr_length) {
    const CU_REAL pp = CU_REAL(0);
    FortranComplex famp = to_fortran_complex(*amplitude), fx{}, fy{}, fz{};
    rigorous_gaussian_point(x, y, z, x0, y0, z0, theta, phi, waist,
                            k0, handedness, &pp, &famp, tolerance,
                            nloin, true, &fx, &fy, &fz, nstop, infostr,
                            infostr_length);
    *ex = to_host_complex(fx); *ey = to_host_complex(fy); *ez = to_host_complex(fz);
}

//-----------------------------------------------------------
namespace {
void gaussianparad_derivative(
    const CU_REAL* x, const CU_REAL* y, const CU_REAL* z,
    const CU_REAL* x0, const CU_REAL* y0, const CU_REAL* z0,
    const CU_REAL* theta_degrees, const CU_REAL* phi_degrees,
    const CU_REAL* waist, const CU_REAL* k0, const CU_REAL* ss,
    const CU_REAL* pp, const FortranComplex* amplitude, bool circular,
    FortranComplex derivative[9], int* nstop, char* infostr,
    std::size_t infostr_length) {
    if (!derivative || !nstop || !infostr)
        return;
    std::fill(derivative, derivative + 9, FortranComplex{CU_REAL(0), CU_REAL(0)});
    *nstop = -1;
    std::fill(infostr, infostr + infostr_length, ' ');
    if (!x || !y || !z || !x0 || !y0 || !z0 || !theta_degrees || !phi_degrees ||
        !waist || !k0 || !ss || !pp || !amplitude || *waist <= CU_REAL(0) ||
        *k0 <= CU_REAL(0) || (!circular && (*pp < CU_REAL(0) || *pp > CU_REAL(1))))
        return;

    const CU_REAL theta = *theta_degrees * pi / CU_REAL(180);
    const CU_REAL phi = *phi_degrees * pi / CU_REAL(180);
    const CU_REAL vx = std::sin(theta) * std::cos(phi);
    const CU_REAL vy = std::sin(theta) * std::sin(phi);
    const CU_REAL vz = std::cos(theta);
    const CU_REAL machine_epsilon = PRODUITMATVECTFFT_SINGLE_PRECISION
                                       ? CU_REAL(1.0e-6) : CU_REAL(1.0e-12);
    CU_REAL rotation[9]{};
    if (vx * vx + vy * vy <= machine_epsilon) {
        rotation[0] = rotation[4] = rotation[8] = CU_REAL(1);
    } else {
        rotation[0] = CU_REAL(1) - vx * vx / (CU_REAL(1) + vz);
        rotation[3] = -vx * vy / (CU_REAL(1) + vz);
        rotation[6] = -vx;
        rotation[1] = rotation[3];
        rotation[4] = CU_REAL(1) - vy * vy / (CU_REAL(1) + vz);
        rotation[7] = -vy;
        rotation[2] = vx;
        rotation[5] = vy;
        rotation[8] = vz;
    }
    const Complex e0(amplitude->real, amplitude->imag);
    const Complex e0s = circular ? e0 / std::sqrt(CU_REAL(2))
                                 : e0 * std::sqrt(CU_REAL(1) - *pp);
    const Complex e0p = circular
        ? e0 * Complex(CU_REAL(0), std::copysign(CU_REAL(1) / std::sqrt(CU_REAL(2)), *ss))
        : e0 * std::sqrt(*pp);
    const Complex e0x = -e0s * std::sin(phi) + e0p * std::cos(theta) * std::cos(phi);
    const Complex e0y = e0s * std::cos(phi) + e0p * std::cos(theta) * std::sin(phi);
    const Complex e0z = -e0p * std::sin(theta);
    const Complex e00x = rotation[0] * e0x + rotation[3] * e0y + rotation[6] * e0z;
    const Complex e00y = rotation[1] * e0x + rotation[4] * e0y + rotation[7] * e0z;
    const Complex e00z = rotation[2] * e0x + rotation[5] * e0y + rotation[8] * e0z;
    if (std::abs(e00z) >= machine_epsilon * std::abs(e0)) {
        const char* message = "probleme in paraxial gaussian beam";
        std::memcpy(infostr, message, std::min(std::strlen(message), infostr_length));
        return;
    }
    const CU_REAL rx = *x - *x0;
    const CU_REAL ry = *y - *y0;
    const CU_REAL rz = *z - *z0;
    const CU_REAL local_x = rotation[0] * rx + rotation[3] * ry + rotation[6] * rz;
    const CU_REAL local_y = rotation[1] * rx + rotation[4] * ry + rotation[7] * rz;
    const CU_REAL local_z = rotation[2] * rx + rotation[5] * ry + rotation[8] * rz;

    if (!initialize_cuda_device())
        return;
    cuda_data::IncidentFieldBufferRequest request{};
    request.coordinate_count = 9;
    request.field_count = 9;
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_incident_field_buffers(request, &buffers) ||
        !buffers.xs || !buffers.field)
        return;
    using cuda_utils::TransferCategory;
    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.xs, rotation, sizeof(rotation), TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = gaussianparad_derivative_wrapper(
            buffers.field, local_x, local_y, local_z, *waist, *k0,
            make_cuda_complex(e00x.real(), e00x.imag()),
            make_cuda_complex(e00y.real(), e00y.imag()), buffers.xs);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            derivative, buffers.field, sizeof(FortranComplex) * 9,
            TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        const char* message = "CUDA paraxial Gaussian derivative failed";
        std::memcpy(infostr, message, std::min(std::strlen(message), infostr_length));
        return;
    }
    *nstop = 0;
}
} // namespace

//-----------------------------------------------------------
extern "C" void gaussianparalineard_(
    const REAL* x, const REAL* y, const REAL* z,
    const REAL* x0, const REAL* y0, const REAL* z0,
    const REAL* theta, const REAL* phi, const REAL* waist,
    const REAL* k0, const REAL* ss, const REAL* pp,
    const COMPLEX* amplitude, COMPLEX derivative[9],
    int* nstop, char* infostr, std::size_t infostr_length) {
    FortranComplex famp = to_fortran_complex(*amplitude), fderivative[9]{};
    gaussianparad_derivative(x, y, z, x0, y0, z0, theta, phi, waist, k0,
                             ss, pp, &famp, false, fderivative, nstop,
                             infostr, infostr_length);
    for (int i = 0; i < 9; ++i) derivative[i] = to_host_complex(fderivative[i]);
}

//-----------------------------------------------------------
extern "C" void gaussianparacircd_(
    const REAL* x, const REAL* y, const REAL* z,
    const REAL* x0, const REAL* y0, const REAL* z0,
    const REAL* theta, const REAL* phi, const REAL* waist,
    const REAL* k0, const REAL* handedness,
    const COMPLEX* amplitude, COMPLEX derivative[9],
    int* nstop, char* infostr, std::size_t infostr_length) {
    const CU_REAL pp = CU_REAL(0);
    FortranComplex famp = to_fortran_complex(*amplitude), fderivative[9]{};
    gaussianparad_derivative(x, y, z, x0, y0, z0, theta, phi, waist, k0,
                             handedness, &pp, &famp, true, fderivative,
                             nstop, infostr, infostr_length);
    for (int i = 0; i < 9; ++i) derivative[i] = to_host_complex(fderivative[i]);
}

//-----------------------------------------------------------
namespace {
void rigorous_gaussian_derivative(
    const CU_REAL* x, const CU_REAL* y, const CU_REAL* z,
    const CU_REAL* x0, const CU_REAL* y0, const CU_REAL* z0,
    const CU_REAL* theta, const CU_REAL* phi, const CU_REAL* waist,
    const CU_REAL* k0, const CU_REAL* ss, const CU_REAL* pp,
    const FortranComplex* amplitude, const CU_REAL* tolerance,
    const int* nloin, bool circular, FortranComplex derivative[9]) {
    if (!derivative)
        return;
    std::fill(derivative, derivative + 9, FortranComplex{CU_REAL(0), CU_REAL(0)});
    if (!x || !y || !z || !x0 || !y0 || !z0 || !theta || !phi || !waist ||
        !k0 || !ss || !pp || !amplitude || !tolerance || !nloin ||
        *waist <= CU_REAL(0) || *k0 <= CU_REAL(0) || *tolerance <= CU_REAL(0) ||
        *nloin < 0 || *nloin > 2 || (!circular && (*pp < CU_REAL(0) || *pp > CU_REAL(1))))
        return;

    std::lock_guard<std::mutex> lock(rigorous_gaussian_workspace_mutex);
    if (!initialize_cuda_device())
        return;

    constexpr size_t point_count = 6;
    const CU_REAL derivative_scale = std::min(*waist, CU_REAL(1) / *k0);
    const CU_REAL step = std::cbrt(std::numeric_limits<CU_REAL>::epsilon()) * derivative_scale;
    const CU_REAL xs[point_count] = {*x + step, *x - step, *x, *x,
                                     *x, *x};
    const CU_REAL ys[point_count] = {*y, *y, *y + step, *y - step,
                                     *y, *y};
    const CU_REAL zs[point_count] = {*z, *z, *z, *z,
                                     *z + step, *z - step};
    cuda_data::RigorousGaussianDeviceBuffers buffers{};
    if (!cuda_data::get_device_rigorous_gaussian_buffers(point_count, &buffers))
        return;

    cudaError_t status = cudaSuccess;
    {
        std::lock_guard<std::mutex> table_lock(gaussian_gkp_table_mutex);
        if (!gaussian_gkp_tables_uploaded) {
            status = cuda_data::copy_host_to_device(
                buffers.gkp_nodes, gp_x, sizeof(gp_x),
                cuda_utils::TransferCategory::IncidentField);
            if (status == cudaSuccess)
                status = cuda_data::copy_host_to_device(
                    buffers.gkp_weights, gp_w, sizeof(gp_w),
                    cuda_utils::TransferCategory::IncidentField);
            if (status == cudaSuccess)
                gaussian_gkp_tables_uploaded = true;
        }
    }
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.xs, xs, sizeof(xs), cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.ys, ys, sizeof(ys), cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.zs, zs, sizeof(zs), cuda_utils::TransferCategory::Geometry);
    const COMPLEX e0(amplitude->real, amplitude->imag);
    const int circular_flag = circular ? 1 : 0;
    if (status == cudaSuccess)
        status = rigorous_gaussian_incident_wrapper(
            buffers.xs, buffers.ys, buffers.zs, int(point_count), *x0, *y0, *z0,
            *theta, *phi, *waist, *k0, *ss, *pp, e0, *tolerance, circular_flag,
            *nloin, buffers.gkp_nodes, buffers.gkp_weights, buffers.field,
            buffers.convergence);
    if (status == cudaSuccess)
        status = rigorous_gaussian_derivative_wrapper(buffers.field, step);
    CUCOMPLEX result[9]{};
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            result, buffers.field, sizeof(result), cuda_utils::TransferCategory::Result);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA rigorous Gaussian derivative failed: %s\n",
                     cudaGetErrorString(status));
        return;
    }
    for (int i = 0; i < 9; ++i)
        derivative[i] = {result[i].x, result[i].y};
}
} // namespace

//-----------------------------------------------------------
extern "C" void gaussianchampd_(
    const REAL* x, const REAL* y, const REAL* z,
    const REAL* x0, const REAL* y0, const REAL* z0,
    const REAL* theta, const REAL* phi, const REAL* waist,
    const REAL* k0, const REAL* ss, const REAL* pp,
    const COMPLEX* amplitude, COMPLEX derivative[9],
    const REAL* tolerance, const int* nloin) {
    FortranComplex famp = to_fortran_complex(*amplitude), fderivative[9]{};
    rigorous_gaussian_derivative(x, y, z, x0, y0, z0, theta, phi, waist,
                                 k0, ss, pp, &famp, tolerance, nloin,
                                 false, fderivative);
    for (int i = 0; i < 9; ++i) derivative[i] = to_host_complex(fderivative[i]);
}

//-----------------------------------------------------------
extern "C" void gaussianchampdcirc_(
    const REAL* x, const REAL* y, const REAL* z,
    const REAL* x0, const REAL* y0, const REAL* z0,
    const REAL* theta, const REAL* phi, const REAL* waist,
    const REAL* k0, const REAL* handedness,
    const COMPLEX* amplitude, COMPLEX derivative[9],
    const REAL* tolerance, const int* nloin) {
    const CU_REAL pp = CU_REAL(0);
    FortranComplex famp = to_fortran_complex(*amplitude), fderivative[9]{};
    rigorous_gaussian_derivative(x, y, z, x0, y0, z0, theta, phi, waist,
                                 k0, handedness, &pp, &famp, tolerance,
                                 nloin, true, fderivative);
    for (int i = 0; i < 9; ++i) derivative[i] = to_host_complex(fderivative[i]);
}

//-----------------------------------------------------------
extern "C" void ifdda_rigorous_gaussian_derivatives_cuda_(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, const int* tabdip,
    const int* coordinate_count, const int* dipole_count, const CU_REAL* x0,
    const CU_REAL* y0, const CU_REAL* z0, const CU_REAL* theta,
    const CU_REAL* phi, const CU_REAL* waist, const CU_REAL* k0,
    const CU_REAL* ss, const CU_REAL* pp, const COMPLEX* amplitude,
    const CU_REAL* tolerance, const int* nloin, const int* circular,
    COMPLEX* output, int* status_out) {
    if (status_out)
        *status_out = 1;
    if (!xs || !ys || !zs || !tabdip || !coordinate_count || !dipole_count ||
        !x0 || !y0 || !z0 || !theta || !phi || !waist || !k0 || !ss || !pp ||
        !amplitude || !tolerance || !nloin || !circular || !output || !status_out ||
        *coordinate_count <= 0 || *dipole_count <= 0 || *dipole_count > INT_MAX / 6 ||
        *waist <= CU_REAL(0) || *k0 <= CU_REAL(0) || *tolerance <= CU_REAL(0) ||
        *nloin < 0 || *nloin > 2 || (*circular != 0 && *circular != 1) ||
        (!*circular && (*pp < CU_REAL(0) || *pp > CU_REAL(1))) ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / 18 / sizeof(CUCOMPLEX) ||
        !initialize_cuda_device())
        return;

    const FortranComplex fortran_amplitude = to_fortran_complex(*amplitude);
    std::lock_guard<std::mutex> lock(rigorous_gaussian_workspace_mutex);
    const size_t point_count = size_t(*dipole_count);
    const size_t sample_count = point_count * 6;
    const CU_REAL derivative_scale = std::min(*waist, CU_REAL(1) / *k0);
    const CU_REAL step = std::cbrt(std::numeric_limits<CU_REAL>::epsilon()) * derivative_scale;
    std::vector<CU_REAL> sample_x(sample_count);
    std::vector<CU_REAL> sample_y(sample_count);
    std::vector<CU_REAL> sample_z(sample_count);
    for (size_t point = 0; point < point_count; ++point) {
        const int index = tabdip[point];
        CU_REAL x = CU_REAL(0), y = CU_REAL(0), z = CU_REAL(0);
        if (index != 0) {
            if (index < 0 || index > *coordinate_count)
                return;
            x = xs[index - 1];
            y = ys[index - 1];
            z = zs[index - 1];
        }
        const size_t base = point * 6;
        sample_x[base] = x + step;
        sample_x[base + 1] = x - step;
        sample_y[base + 2] = y + step;
        sample_y[base + 3] = y - step;
        sample_z[base + 4] = z + step;
        sample_z[base + 5] = z - step;
        sample_y[base] = sample_y[base + 1] = y;
        sample_x[base + 2] = sample_x[base + 3] = x;
        sample_z[base + 2] = sample_z[base + 3] = z;
        sample_x[base + 4] = sample_x[base + 5] = x;
        sample_y[base + 4] = sample_y[base + 5] = y;
    }

    cuda_data::RigorousGaussianDeviceBuffers gaussian_buffers{};
    cuda_data::IncidentFieldBufferRequest request{};
    request.field_count = point_count * 9;
    cuda_data::IncidentFieldDeviceBuffers derivative_buffers{};
    if (!cuda_data::get_device_rigorous_gaussian_buffers(sample_count, &gaussian_buffers) ||
        !cuda_data::get_device_incident_field_buffers(request, &derivative_buffers) ||
        !gaussian_buffers.xs || !gaussian_buffers.ys || !gaussian_buffers.zs ||
        !gaussian_buffers.field || !gaussian_buffers.gkp_nodes ||
        !gaussian_buffers.gkp_weights || !derivative_buffers.field)
        return;

    cudaError_t status = cudaSuccess;
    {
        std::lock_guard<std::mutex> table_lock(gaussian_gkp_table_mutex);
        if (!gaussian_gkp_tables_uploaded) {
            status = cuda_data::copy_host_to_device(
                gaussian_buffers.gkp_nodes, gp_x, sizeof(gp_x),
                cuda_utils::TransferCategory::IncidentField);
            if (status == cudaSuccess)
                status = cuda_data::copy_host_to_device(
                    gaussian_buffers.gkp_weights, gp_w, sizeof(gp_w),
                    cuda_utils::TransferCategory::IncidentField);
            if (status == cudaSuccess)
                gaussian_gkp_tables_uploaded = true;
        }
    }
    const size_t coordinate_bytes = sample_count * sizeof(CU_REAL);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            gaussian_buffers.xs, sample_x.data(), coordinate_bytes,
            cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            gaussian_buffers.ys, sample_y.data(), coordinate_bytes,
            cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            gaussian_buffers.zs, sample_z.data(), coordinate_bytes,
            cuda_utils::TransferCategory::Geometry);
    const COMPLEX e0(fortran_amplitude.real, fortran_amplitude.imag);
    if (status == cudaSuccess)
        status = rigorous_gaussian_incident_wrapper(
            gaussian_buffers.xs, gaussian_buffers.ys, gaussian_buffers.zs,
            int(sample_count), *x0, *y0, *z0, *theta, *phi, *waist, *k0,
            *ss, *pp, e0, *tolerance, *circular, *nloin,
            gaussian_buffers.gkp_nodes, gaussian_buffers.gkp_weights,
            gaussian_buffers.field, gaussian_buffers.convergence);
    if (status == cudaSuccess)
        status = rigorous_gaussian_derivative_batch_wrapper(
            gaussian_buffers.field, derivative_buffers.field,
            *dipole_count, step);
    std::vector<CUCOMPLEX> host_derivatives(point_count * 9);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            host_derivatives.data(), derivative_buffers.field,
            host_derivatives.size() * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA rigorous Gaussian derivative batch failed: %s\n",
                     cudaGetErrorString(status));
        return;
    }
    for (size_t value = 0; value < host_derivatives.size(); ++value)
        output[value] = COMPLEX(host_derivatives[value].x, host_derivatives[value].y);
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void dipoleincder_(
    const REAL* xdip, const REAL* ydip, const REAL* zdip,
    const REAL* theta, const REAL* phi,
    const REAL* x, const REAL* y, const REAL* z,
    const REAL* aretecube, const REAL* k0,
    const COMPLEX* e0, COMPLEX eder[9],
    int* test, int* nstop, char* infostr, std::size_t infostr_length) {
    if (!nstop || !infostr || !eder)
        return;
    std::fill(infostr, infostr + infostr_length, ' ');
    *nstop = 0;
    std::fill(eder, eder + 9, COMPLEX(0, 0));
    if (!xdip || !ydip || !zdip || !theta || !phi || !x || !y || !z ||
        !aretecube || !k0 || !e0 || !test || *test < 1 || *test > 3 ||
        !std::isfinite(*xdip) || !std::isfinite(*ydip) || !std::isfinite(*zdip) ||
        !std::isfinite(*x) || !std::isfinite(*y) || !std::isfinite(*z) ||
        !std::isfinite(*theta) || !std::isfinite(*phi) ||
        !std::isfinite(*aretecube) || !std::isfinite(*k0) ||
        *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0)) {
        *nstop = -1;
        constexpr char message[] = "invalid dipole derivative arguments";
        std::copy_n(message, std::min(sizeof(message) - 1, infostr_length), infostr);
        return;
    }
    if (!initialize_cuda_device()) {
        *nstop = -1;
        constexpr char message[] = "CUDA initialization failed";
        std::copy_n(message, std::min(sizeof(message) - 1, infostr_length), infostr);
        return;
    }

    std::lock_guard<std::mutex> lock(dipole_derivative_mutex);
    cuda_data::DipoleDerivativeDeviceBuffers buffers{};
    const CUCOMPLEX amplitude = make_cuda_complex(e0->real(), e0->imag());
    bool ok = cuda_data::get_device_dipole_derivative_buffers(&buffers) &&
        cuda_utils::check_cuda(dipole_incident_derivative_wrapper(
            buffers.derivative, *xdip, *ydip, *zdip, *theta, *phi, *x, *y, *z,
            *aretecube, *k0, amplitude, *test),
            "dipole derivative", "launch CUDA derivative");
    CUCOMPLEX result[3]{};
    if (ok) {
        ok = cuda_utils::check_cuda(cuda_data::copy_device_to_host(
            result, buffers.derivative, sizeof(result), cuda_utils::TransferCategory::Result),
            "dipole derivative", "download derivative result");
    }
    if (!ok) {
        *nstop = -1;
        constexpr char message[] = "CUDA dipole derivative failed";
        std::copy_n(message, std::min(sizeof(message) - 1, infostr_length), infostr);
        return;
    }
    const int column = *test - 1;
    for (int row = 0; row < 3; ++row)
        eder[row + 3 * column] = COMPLEX(result[row].x, result[row].y);
}

//-----------------------------------------------------------
extern "C" void ifdda_antenna_derivatives_cuda_(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, const int* tabdip,
    const int* coordinate_count, const int* dipole_count, const CU_REAL* xdip,
    const CU_REAL* ydip, const CU_REAL* zdip, const CU_REAL* theta,
    const CU_REAL* phi, const CU_REAL* aretecube, const CU_REAL* k0,
    const COMPLEX* e0, COMPLEX* output, int* status_out) {
    if (status_out)
        *status_out = 1;
    if (!xs || !ys || !zs || !tabdip || !coordinate_count || !dipole_count ||
        !xdip || !ydip || !zdip || !theta || !phi || !aretecube || !k0 ||
        !e0 || !output || !status_out || *coordinate_count <= 0 ||
        *dipole_count <= 0 || *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0) ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / 9 / sizeof(CUCOMPLEX) ||
        !initialize_cuda_device())
        return;

    const FortranComplex fortran_e0 = to_fortran_complex(*e0);
    std::lock_guard<std::mutex> lock(dipole_derivative_mutex);
    const size_t count = size_t(*dipole_count);
    std::vector<CU_REAL> point_x(count, CU_REAL(0));
    std::vector<CU_REAL> point_y(count, CU_REAL(0));
    std::vector<CU_REAL> point_z(count, CU_REAL(0));
    for (size_t point = 0; point < count; ++point) {
        const int index = tabdip[point];
        if (index == 0)
            continue;
        if (index < 0 || index > *coordinate_count)
            return;
        point_x[point] = xs[index - 1];
        point_y[point] = ys[index - 1];
        point_z[point] = zs[index - 1];
    }

    cuda_data::IncidentFieldBufferRequest request{};
    request.coordinate_count = count;
    request.field_count = count * 9;
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_incident_field_buffers(request, &buffers) ||
        !buffers.xs || !buffers.ys || !buffers.zs || !buffers.field)
        return;
    using cuda_utils::TransferCategory;
    const size_t coordinate_bytes = count * sizeof(CU_REAL);
    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.xs, point_x.data(), coordinate_bytes, TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.ys, point_y.data(), coordinate_bytes, TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.zs, point_z.data(), coordinate_bytes, TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = dipole_incident_derivative_batch_wrapper(
            buffers.xs, buffers.ys, buffers.zs, buffers.field, *dipole_count,
            *xdip, *ydip, *zdip, *theta, *phi, *aretecube, *k0,
            make_cuda_complex(fortran_e0.real, fortran_e0.imag));
    std::vector<CUCOMPLEX> host_derivatives(count * 9);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            host_derivatives.data(), buffers.field,
            host_derivatives.size() * sizeof(CUCOMPLEX), TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA antenna derivative batch failed: %s\n",
                     cudaGetErrorString(status));
        return;
    }
    for (size_t value = 0; value < host_derivatives.size(); ++value)
        output[value] = COMPLEX(host_derivatives[value].x, host_derivatives[value].y);
    *status_out = 0;
}

//-----------------------------------------------------------
void ifdda_antenna_field_cuda_execute(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count, const REAL *xdip,
    const REAL *ydip, const REAL *zdip, const REAL *aretecube, const REAL *k0,
    const COMPLEX *e0, const REAL *theta_deg, const REAL *phi_deg,
    const REAL *reference_x, const REAL *reference_y, const REAL *reference_z,
    COMPLEX *output, COMPLEX *reference_field, int *status_out,
    bool device_coordinates, CUCOMPLEX *external_device_output,
    const IfddaIncidentRegularGridSpec* grid_request,
    const CU_REAL* xs_device, const CU_REAL* ys_device,
    const CU_REAL* zs_device) {
    PlaneWaveGrid checked_grid{};
    const PlaneWaveGrid* regular_grid = ifdda_resolve_incident_grid(
        grid_request, xs, ys, zs, count, &checked_grid);
    if (!external_device_output && grid_request && count && *count > 0)
        external_device_output = cuda_data::get_ff0(size_t(*count));
    if (status_out)
        *status_out = 1;
    if ((device_coordinates ? (!xs_device || !ys_device || !zs_device)
                            : (!xs || !ys || !zs)) ||
        !count || !xdip || !ydip || !zdip || !aretecube || !k0 || !e0 ||
        !theta_deg || !phi_deg || !reference_x || !reference_y || !reference_z ||
        (!output && !external_device_output) ||
        !reference_field || !status_out || *count <= 0 || *aretecube <= 0 || *k0 <= 0 ||
        size_t(*count) > size_t(-1) / (3 * sizeof(CUCOMPLEX)))
        return;
    if (!initialize_cuda_device())
        return;

    const size_t n = size_t(*count);
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    cuda_data::IncidentFieldBufferRequest request{};
    request.antenna_coordinate_count = (device_coordinates || regular_grid) ? 0 : n;
    request.antenna_field_count = external_device_output ? 0 : 3 * n;
    request.antenna_reference_count = 3;
    if (!cuda_data::get_device_incident_field_buffers(request, &buffers))
        return;
    CU_REAL *dx = buffers.antenna_xs;
    CU_REAL *dy = buffers.antenna_ys;
    CU_REAL *dz = buffers.antenna_zs;
    CUCOMPLEX *df = buffers.antenna_field;
    CUCOMPLEX *dr = buffers.antenna_reference;
    if ((!device_coordinates && !regular_grid && (!dx || !dy || !dz)) ||
        (!external_device_output && !df) || !dr)
        return;

    using cuda_utils::TransferCategory;
    const CU_REAL *device_x = regular_grid ? nullptr : device_coordinates ? xs_device : dx;
    const CU_REAL *device_y = regular_grid ? nullptr : device_coordinates ? ys_device : dy;
    const CU_REAL *device_z = regular_grid ? nullptr : device_coordinates ? zs_device : dz;
    CUCOMPLEX *device_field = external_device_output ? external_device_output : df;
    cudaError_t status = cudaSuccess;
    if (!device_coordinates && !regular_grid)
        status = cuda_data::copy_host_to_device(dx, xs, n * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess)
        if (!device_coordinates && !regular_grid)
            status = cuda_data::copy_host_to_device(dy, ys, n * sizeof(CU_REAL),
                                                    TransferCategory::Geometry);
    if (status == cudaSuccess && !device_coordinates && !regular_grid)
        status = cuda_data::copy_host_to_device(dz, zs, n * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = regular_grid ?
            incident_antenna_grid_wrapper(
                ifdda_kernel_incident_grid(*regular_grid, *count),
                device_field, dr, *xdip, *ydip, *zdip, *aretecube,
                *k0, *e0, *theta_deg, *phi_deg,
                *reference_x, *reference_y, *reference_z) :
            incident_antenna_wrapper(device_x, device_y, device_z,
                                     device_field, dr, *count, *xdip, *ydip, *zdip,
                                     *aretecube, *k0, *e0, *theta_deg, *phi_deg, *reference_x,
                                     *reference_y, *reference_z);
    if (status == cudaSuccess && output)
        status = cuda_data::copy_device_complex_to_host(output, device_field, 3 * n,
                                                TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_complex_to_host(reference_field, dr, 3,
                                                TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA antenna illumination failed: %s\n", cudaGetErrorString(status));
        return;
    }
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_antenna_field_cuda_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count,
    const REAL *xdip, const REAL *ydip, const REAL *zdip,
    const REAL *aretecube, const REAL *k0, const COMPLEX *e0,
    const REAL *theta_deg, const REAL *phi_deg,
    const REAL *reference_x, const REAL *reference_y,
    const REAL *reference_z, COMPLEX *output,
    COMPLEX *reference_field, int *status_out) {
    if (!count || !output || *count <= 0) {
        if (status_out) *status_out = 1;
        return;
    }
    CUCOMPLEX *device_ff0 = cuda_data::get_ff0(size_t(*count));
    if (!device_ff0) {
        if (status_out) *status_out = 1;
        return;
    }
    HostGridRequest host_grid{};
    const auto* grid = ifdda_fortran_grid_request(xs, ys, zs, count, &host_grid);
    ifdda_antenna_field_cuda_execute(
        xs, ys, zs, count, xdip, ydip, zdip, aretecube, k0, e0, theta_deg,
        phi_deg, reference_x, reference_y, reference_z, nullptr, reference_field,
        status_out, false, device_ff0, grid);
    if (status_out && *status_out == 0)
        if (cuda_data::copy_device_complex_to_host(output, device_ff0,
                3 * size_t(*count), cuda_utils::TransferCategory::IncidentField) != cudaSuccess)
            *status_out = 1;
}


//-----------------------------------------------------------
extern "C" void ifdda_antenna_field_cuda_device_coordinates_(
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const int *count, const REAL *xdip,
    const REAL *ydip, const REAL *zdip, const REAL *aretecube,
    const REAL *k0, const COMPLEX *e0, const REAL *theta_deg,
    const REAL *phi_deg, const REAL *reference_x,
    const REAL *reference_y, const REAL *reference_z,
    COMPLEX *reference_field, int *status_out, CUCOMPLEX *output_device) {
    ifdda_antenna_field_cuda_execute(
        nullptr, nullptr, nullptr, count, xdip, ydip, zdip, aretecube,
        k0, e0, theta_deg, phi_deg, reference_x, reference_y, reference_z,
        nullptr, reference_field, status_out, true, output_device, nullptr,
        xs_device, ys_device, zs_device);
}

//-----------------------------------------------------------
extern "C" void dipoleinc_(
    const REAL *xdip, const REAL *ydip, const REAL *zdip,
    const REAL *theta, const REAL *phi,
    const REAL *x, const REAL *y, const REAL *z,
    const REAL *aretecube, const REAL *k0, const COMPLEX *e0,
    COMPLEX *ex, COMPLEX *ey, COMPLEX *ez,
    int *nstop, char *infostr, std::size_t infostr_length) {
    if (!nstop || !infostr)
        return;
    *nstop = 1;
    std::memset(infostr, ' ', infostr_length);
    if (!xdip || !ydip || !zdip || !theta || !phi || !x || !y || !z ||
        !aretecube || !k0 || !e0 || !ex || !ey || !ez)
        return;

    const int count = 1;
    const CU_REAL xs[1] = {*x};
    const CU_REAL ys[1] = {*y};
    const CU_REAL zs[1] = {*z};
    COMPLEX output[3]{};
    COMPLEX reference[3]{};
    int status = 1;
    ifdda_antenna_field_cuda_execute(xs, ys, zs, &count, xdip, ydip, zdip,
                                  aretecube, k0, e0, theta, phi, x, y, z,
                                  output, reference, &status, false, nullptr);
    if (status != 0) {
        const char *message = "CUDA dipole illumination failed";
        std::memcpy(infostr, message,
                    std::min(infostr_length, std::strlen(message)));
        return;
    }
    *ex = output[0];
    *ey = output[1];
    *ez = output[2];
    *nstop = 0;
}

//-----------------------------------------------------------
extern "C" void dipoleincp_batch_(
    const REAL *xdip, const REAL *ydip, const REAL *zdip,
    const COMPLEX *moment, const REAL *xs, const REAL *ys, const REAL *zs,
    const int *count, const REAL *aretecube, const REAL *k0, COMPLEX *output,
    int *nstop, char *infostr, std::size_t infostr_length) {
    if (!nstop || !infostr)
        return;
    *nstop = 1;
    std::memset(infostr, ' ', infostr_length);
    if (!xdip || !ydip || !zdip || !moment || !xs || !ys || !zs || !count ||
        !aretecube || !k0 || !output || *count <= 0 ||
        *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0))
        return;
    if (!initialize_cuda_device())
        return;

    CUCOMPLEX cuda_moment[3]{};
    for (int i = 0; i < 3; ++i)
        cuda_moment[i] = make_cuda_complex(moment[i].real(), moment[i].imag());

    const size_t field_bytes = size_t(3) * size_t(*count) * sizeof(CUCOMPLEX);
    CUCOMPLEX *device_field = static_cast<CUCOMPLEX *>(
        cuda_data::get_device_workspace_buffer_bytes("incident.dipole_batch.field", field_bytes));
    if (!device_field)
        return;

    const CU_REAL *device_xs = nullptr;
    const CU_REAL *device_ys = nullptr;
    const CU_REAL *device_zs = nullptr;
    bool resident = cuda_data::get_device_geometry_coordinate_aliases(
        xs, ys, zs, size_t(*count), &device_xs, &device_ys, &device_zs);
    cudaError_t status = cudaSuccess;
    if (!resident) {
        device_xs = static_cast<const CU_REAL *>(
            cuda_data::get_device_workspace_buffer_bytes(
                "incident.dipole_batch.xs", size_t(*count) * sizeof(CU_REAL)));
        device_ys = static_cast<const CU_REAL *>(
            cuda_data::get_device_workspace_buffer_bytes(
                "incident.dipole_batch.ys", size_t(*count) * sizeof(CU_REAL)));
        device_zs = static_cast<const CU_REAL *>(
            cuda_data::get_device_workspace_buffer_bytes(
                "incident.dipole_batch.zs", size_t(*count) * sizeof(CU_REAL)));
        if (!device_xs || !device_ys || !device_zs)
            return;
        status = cuda_data::copy_host_to_device(
            const_cast<CU_REAL *>(device_xs), xs, size_t(*count) * sizeof(CU_REAL),
            cuda_utils::TransferCategory::Geometry);
        if (status == cudaSuccess)
            status = cuda_data::copy_host_to_device(
                const_cast<CU_REAL *>(device_ys), ys, size_t(*count) * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Geometry);
        if (status == cudaSuccess)
            status = cuda_data::copy_host_to_device(
                const_cast<CU_REAL *>(device_zs), zs, size_t(*count) * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Geometry);
    }
    if (status == cudaSuccess)
        status = dipole_moment_field_batch_wrapper(
            device_xs, device_ys, device_zs, device_field, *count,
            *xdip, *ydip, *zdip, *aretecube, *k0,
            cuda_moment[0], cuda_moment[1], cuda_moment[2]);
    std::vector<CUCOMPLEX> host_output(size_t(3) * size_t(*count));
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            host_output.data(), device_field, field_bytes, cuda_utils::TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        const char *message = "CUDA batched dipole moment illumination failed";
        std::memcpy(infostr, message, std::min(infostr_length, std::strlen(message)));
        return;
    }
    for (size_t i = 0; i < host_output.size(); ++i)
        output[i] = COMPLEX(host_output[i].x, host_output[i].y);
    *nstop = 0;
}
//-------------------------------------------
extern "C" void dipoleincp_(
    const REAL *xdip, const REAL *ydip, const REAL *zdip,
    const COMPLEX *moment, const REAL *x, const REAL *y, const REAL *z,
    const REAL *aretecube, const REAL *k0,
    COMPLEX *ex, COMPLEX *ey, COMPLEX *ez,
    int *nstop, char *infostr, std::size_t infostr_length) {
    if (!nstop || !infostr)
        return;
    *nstop = 1;
    std::memset(infostr, ' ', infostr_length);
    if (!xdip || !ydip || !zdip || !moment || !x || !y || !z ||
        !aretecube || !k0 || !ex || !ey || !ez ||
        *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0))
        return;
    if (!initialize_cuda_device())
        return;

    cuda_data::IncidentFieldBufferRequest request{};
    request.antenna_coordinate_count = 1;
    request.antenna_field_count = 3;
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_incident_field_buffers(request, &buffers) ||
        !buffers.antenna_xs || !buffers.antenna_ys || !buffers.antenna_zs ||
        !buffers.antenna_field)
        return;

    CUCOMPLEX cuda_moment[3]{};
    for (int i = 0; i < 3; ++i)
        cuda_moment[i] = make_cuda_complex(moment[i].real(), moment[i].imag());
    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.antenna_xs, x, sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.antenna_ys, y, sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.antenna_zs, z, sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = dipole_moment_field_wrapper(
            buffers.antenna_xs, buffers.antenna_ys, buffers.antenna_zs,
            buffers.antenna_field, *xdip, *ydip, *zdip, *aretecube, *k0,
            cuda_moment[0], cuda_moment[1], cuda_moment[2]);
    CUCOMPLEX output[3]{};
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            output, buffers.antenna_field, sizeof(output),
            cuda_utils::TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        const char *message = "CUDA dipole moment illumination failed";
        std::memcpy(infostr, message,
                    std::min(infostr_length, std::strlen(message)));
        return;
    }
    *ex = COMPLEX(output[0].x, output[0].y);
    *ey = COMPLEX(output[1].x, output[1].y);
    *ez = COMPLEX(output[2].x, output[2].y);
    *nstop = 0;
}

extern "C" void ifdda_reset_incident_gaussian_gkp_state() {
    std::lock_guard<std::mutex> lock(gaussian_gkp_table_mutex);
    gaussian_gkp_tables_uploaded = false;
}
