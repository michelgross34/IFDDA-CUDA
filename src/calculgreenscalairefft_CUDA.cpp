/* CUDA implementation of the scalar Green tensor construction/FFT used by
 * greencalculscalairefft (nrig=8) and greencalculscalairefft2 (nrig=9).
 * Fortran arguments retain pass-by-address semantics. */
#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "gaussian_gkp_tables.h"
#include "kernel.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>

namespace {
CUCOMPLEX to_cuda_complex(const COMPLEX &value) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(static_cast<CU_REAL>(value.real()),
                               static_cast<CU_REAL>(value.imag()));
#else
    return make_cuDoubleComplex(static_cast<CU_REAL>(value.real()),
                                static_cast<CU_REAL>(value.imag()));
#endif
}

CUCOMPLEX *multilayer_image = nullptr;
CUCOMPLEX *multilayer_polarizability = nullptr;
CUCOMPLEX *multilayer_local_field = nullptr;
CU_INT *multilayer_plane_map = nullptr;
CUCOMPLEX *multilayer_history = nullptr;
size_t multilayer_image_capacity = 0;
size_t multilayer_polarizability_capacity = 0;
size_t multilayer_local_field_capacity = 0;
size_t multilayer_plane_map_capacity = 0;
size_t multilayer_history_capacity = 0;
CUCOMPLEX *multilayer_vector_images[3] = {nullptr, nullptr, nullptr};
CUCOMPLEX *multilayer_vector_alpha = nullptr;
CUCOMPLEX *multilayer_vector_local = nullptr;
CUCOMPLEX *multilayer_vector_polarization[3] = {nullptr, nullptr, nullptr};
CUCOMPLEX *multilayer_vector_field[3] = {nullptr, nullptr, nullptr};
CU_INT *multilayer_vector_map = nullptr;
CUCOMPLEX *multilayer_vector_history = nullptr;
size_t multilayer_vector_image_capacity[3] = {0, 0, 0};
size_t multilayer_vector_alpha_capacity = 0;
size_t multilayer_vector_local_capacity = 0;
size_t multilayer_vector_polarization_capacity[3] = {0, 0, 0};
size_t multilayer_vector_field_capacity[3] = {0, 0, 0};
size_t multilayer_vector_map_capacity = 0;
size_t multilayer_vector_history_capacity = 0;
CU_REAL *multilayer_first_plane_x = nullptr;
CU_REAL *multilayer_first_plane_y = nullptr;
CU_REAL *multilayer_first_plane_z = nullptr;
CUCOMPLEX *multilayer_first_plane_incident = nullptr;
size_t multilayer_first_plane_x_capacity = 0;
size_t multilayer_first_plane_y_capacity = 0;
size_t multilayer_first_plane_z_capacity = 0;
size_t multilayer_first_plane_incident_capacity = 0;
std::mutex multilayer_gkp_mutex;
bool multilayer_gkp_uploaded = false;
bool multilayer_first_plane_ready = false;
const COMPLEX *multilayer_first_plane_image_key = nullptr;
const COMPLEX *multilayer_first_plane_local_key = nullptr;
int multilayer_first_plane_nfft = 0;
int multilayer_first_plane_nmax = 0;
const COMPLEX *multilayer_result_local_key = nullptr;
const COMPLEX *multilayer_result_alpha_key = nullptr;
int multilayer_result_nmax = 0;
int multilayer_result_ndipole = 0;
bool multilayer_vector_first_plane_ready = false;
const COMPLEX *multilayer_vector_first_plane_image_keys[3] = {nullptr, nullptr, nullptr};
const COMPLEX *multilayer_vector_first_plane_local_key = nullptr;
int multilayer_vector_first_plane_nfft = 0;
int multilayer_vector_first_plane_nmax = 0;
const COMPLEX *multilayer_vector_result_local_key = nullptr;
const COMPLEX *multilayer_vector_result_alpha_key = nullptr;
int multilayer_vector_result_nmax = 0;
int multilayer_vector_result_ndipole = 0;

//-----------------------------------------------------------
template <typename T>
bool reserve_device(T *&pointer, size_t *capacity, size_t required, size_t element_size,
                    const char *buffer_name, const char *context) {
    if (required == 0 || element_size == 0 ||
        required > std::numeric_limits<size_t>::max() / element_size)
        return false;
    if (*capacity >= required && pointer != nullptr)
        return true;
    pointer = static_cast<T *>(
        cuda_data::get_device_workspace_buffer_bytes(buffer_name, required * element_size));
    if (!pointer) {
        std::fprintf(stderr, "%s: could not reserve device buffer '%s'\n", context, buffer_name);
        return false;
    }
    *capacity = required;
    return true;
}
} // namespace

//-----------------------------------------------------------
extern "C" int ifdda_generate_scalar_green_cuda_(const int *nx, const int *ny, const int *nz,
                                                 const int *nx2, const int *ny2, const int *nz2,
                                                 const CU_REAL *aretecube, const CU_REAL *k0,
                                                 const CUCOMPLEX *uincx, const CUCOMPLEX *uincy,
                                                 const CUCOMPLEX *uincz, const int *ntotalm,
                                                 const int *nrig, const COMPLEX *cpu_green_key) {
    constexpr const char *context = "CUDA scalar Green generation";
    if (!nx || !ny || !nz || !nx2 || !ny2 || !nz2 || !aretecube || !k0 || !uincx || !uincy ||
        !uincz || !ntotalm || !nrig || !cpu_green_key || *nx <= 0 || *ny <= 0 || *nz <= 0 ||
        *nx2 < *nx || *ny2 < *ny || *nz2 < *nz || *ntotalm <= 0 || (*nrig != 8 && *nrig != 9))
        return 1;

    size_t xy = 0, fft_count = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx2), size_t(*ny2), &xy) ||
        !cuda_utils::checked_multiply(xy, size_t(*nz2), &fft_count) ||
        fft_count > size_t(std::numeric_limits<int>::max()) || size_t(*ntotalm) < fft_count) {
        std::fprintf(stderr, "%s: invalid padded tensor size\n", context);
        return 2;
    }

    CUCOMPLEX *gpu_green = get_device_scalar_green_output(cpu_green_key, size_t(*ntotalm));
    if (!gpu_green) {
        std::fprintf(stderr, "%s: could not allocate/reuse device tensor\n", context);
        return 3;
    }

    const cudaError_t generation =
        generate_scalar_green_wrapper(gpu_green, *nx, *ny, *nz, *nx2, *ny2, *nz2, *ntotalm,
                                      *aretecube, *k0, *uincx, *uincy, *uincz, *nrig);
    if (!cuda_utils::check_cuda(generation, context, "generate tensor kernel"))
        return 4;

    const cufftHandle plan = get_fft_plan(*nx2, *ny2, *nz2);
    if (!plan) {
        std::fprintf(stderr, "%s: cuFFT plan creation failed\n", context);
        return 5;
    }
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(plan, gpu_green, gpu_green, CUFFT_INVERSE), context,
                                 "inverse cuFFT"))
        return 6;
    if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), context,
                                "synchronize tensor generation and FFT"))
        return 7;

    return 0;
}

namespace {
//-----------------------------------------------------------
void set_green_status(int *nstop, char *infostr, size_t infostr_len, int status, int nrig) {
    if (nstop)
        *nstop = status == 0 ? 0 : 1;
    if (!infostr || infostr_len == 0)
        return;
    std::memset(infostr, ' ', infostr_len);
    if (status != 0) {
        const char *message = nrig == 8 ? "CUDA scalar Green nrig=8 generation failed"
                                        : "CUDA scalar Green nrig=9 generation failed";
        const size_t length = std::min(std::strlen(message), infostr_len);
        std::memcpy(infostr, message, length);
    }
}

//-----------------------------------------------------------
void generate_scalar_green_fortran_abi(
    const int *nx, const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nz2, const int *nxy2, const int *nxm, const int *nym, const int *nzm,
    const CU_REAL *aretecube, const CU_REAL *k0, const CUCOMPLEX *uincx,
    const CUCOMPLEX *uincy, const CUCOMPLEX *uincz, COMPLEX *green,
    const long long *planb, const int *ntotalm, int *nstop, char *infostr,
    size_t infostr_len, int nrig) {
    (void)nxy2;
    (void)nxm;
    (void)nym;
    (void)nzm;
    (void)planb;
    if (!nx || !ny || !nz || !nx2 || !ny2 || !nz2 || !aretecube || !k0 || !uincx || !uincy ||
        !uincz || !green || !ntotalm || !nstop) {
        set_green_status(nstop, infostr, infostr_len, -1, nrig);
        return;
    }
    const int status = ifdda_generate_scalar_green_cuda_(
        nx, ny, nz, nx2, ny2, nz2, aretecube, k0, uincx, uincy, uincz, ntotalm, &nrig, green);
    set_green_status(nstop, infostr, infostr_len, status, nrig);
}
} // namespace

//-----------------------------------------------------------
extern "C" void greencalculscalairefft_(
    const int *nx, const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nz2, const int *nxy2, const int *nxm, const int *nym, const int *nzm,
    const CU_REAL *aretecube, const CU_REAL *k0, const CUCOMPLEX *uincx,
    const CUCOMPLEX *uincy, const CUCOMPLEX *uincz, COMPLEX *green,
    const long long *planb, const int *ntotalm, int *nstop, char *infostr,
    size_t infostr_len) {
    generate_scalar_green_fortran_abi(nx, ny, nz, nx2, ny2, nz2, nxy2, nxm, nym, nzm,
                                      aretecube, k0, uincx, uincy, uincz, green, planb,
                                      ntotalm, nstop, infostr, infostr_len, 8);
}

//-----------------------------------------------------------
extern "C" void greencalculscalairefft2_(
    const int *nx, const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nz2, const int *nxy2, const int *nxm, const int *nym, const int *nzm,
    const CU_REAL *aretecube, const CU_REAL *k0, const CUCOMPLEX *uincx,
    const CUCOMPLEX *uincy, const CUCOMPLEX *uincz, COMPLEX *green,
    const long long *planb, const int *ntotalm, int *nstop, char *infostr,
    size_t infostr_len) {
    generate_scalar_green_fortran_abi(nx, ny, nz, nx2, ny2, nz2, nxy2, nxm, nym, nzm,
                                      aretecube, k0, uincx, uincy, uincz, green, planb,
                                      ntotalm, nstop, infostr, infostr_len, 9);
}

// Builds the z=aretecube scalar Green slice used by multilayerbornsca.
// The transformed tensor remains resident for the layer loop that follows.
//-----------------------------------------------------------
extern "C" int ifdda_generate_multilayer_scalar_green_cuda_(COMPLEX *cpu_green, const int *nfft2d,
                                                            const CU_REAL *aretecube,
                                                            const CU_REAL *k0) {
    constexpr const char *context = "CUDA multilayer scalar Green";
    if (!cpu_green || !nfft2d || !aretecube || !k0 || *nfft2d <= 0 || *nfft2d > INT_MAX / 2 ||
        *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0))
        return 1;
    const size_t side = size_t(2) * size_t(*nfft2d);
    size_t count = 0;
    if (!cuda_utils::checked_multiply(side, side, &count) || count > size_t(INT_MAX))
        return 2;
    CUCOMPLEX *gpu_green = get_device_scalar_green_output(cpu_green, count);
    if (!gpu_green)
        return 3;
    if (!cuda_utils::check_cuda(generate_multilayer_scalar_green_wrapper(
                                    gpu_green, *nfft2d, int(count), *aretecube, *k0),
                                context, "generate Green slice"))
        return 4;
    const cufftHandle plan = get_fft_plan(int(side), int(side), 1);
    if (!plan)
        return 5;
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(plan, gpu_green, gpu_green, CUFFT_INVERSE), context,
                                 "inverse cuFFT"))
        return 6;
    (void)gpu_green;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_generate_multilayer_projected_green_cuda_(
    COMPLEX *cpu_green, const int *nfft2d, const CU_REAL *aretecube, const CU_REAL *k0,
    const COMPLEX *ux, const COMPLEX *uy, const COMPLEX *uz) {
    constexpr const char *context = "CUDA multilayer projected Green";
    if (!cpu_green || !nfft2d || !aretecube || !k0 || !ux || !uy || !uz || *nfft2d <= 0 ||
        *nfft2d > INT_MAX / 2 || *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0))
        return 1;
    const size_t side = size_t(2) * size_t(*nfft2d);
    size_t count = 0;
    if (!cuda_utils::checked_multiply(side, side, &count) || count > size_t(INT_MAX))
        return 2;
    CUCOMPLEX *gpu_green = get_device_scalar_green_output(cpu_green, count);
    if (!gpu_green)
        return 3;
    const CUCOMPLEX ux_cuda = to_cuda_complex(*ux);
    const CUCOMPLEX uy_cuda = to_cuda_complex(*uy);
    const CUCOMPLEX uz_cuda = to_cuda_complex(*uz);
    if (!cuda_utils::check_cuda(generate_multilayer_projected_green_wrapper(
                                    gpu_green, *nfft2d, int(count), *aretecube, *k0,
                                    ux_cuda, uy_cuda, uz_cuda),
                                context, "generate projected dyadic Green slice"))
        return 4;
    const cufftHandle plan = get_fft_plan(int(side), int(side), 1);
    if (!plan)
        return 5;
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(plan, gpu_green, gpu_green, CUFFT_INVERSE), context,
                                 "inverse cuFFT"))
        return 6;
    return 0;
}

// Generate the first scalar multilayer image and its initial local field on
// the GPU for plane-wave illumination. The coordinate arrays are copied once
// as geometry metadata; no per-pixel field samples are created on the CPU.
//-----------------------------------------------------------
extern "C" int ifdda_multilayer_scalar_initialize_plane_cuda_(
    COMPLEX *cpu_image, COMPLEX *cpu_local_field, const COMPLEX *cpu_incident_dipoles,
    const CU_REAL *cpu_xs, const CU_REAL *cpu_ys, const CU_REAL *cpu_zs, const int *ndipole,
    const int *nfft2d, const int *nmax, const int *kkm, const int *jjm, const CU_REAL *xmin,
    const CU_REAL *ymin, const CU_REAL *first_z, const CU_REAL *aretecube, const CU_REAL *k0,
    const COMPLEX *E0, const COMPLEX *eps0, const CU_REAL *w0, const CU_REAL *ss,
    const CU_REAL *pp, const CU_REAL *theta, const CU_REAL *phi, const int *beam_type,
    const int *conjugate_projection, const COMPLEX *ux, const COMPLEX *uy,
    const COMPLEX *uz) {
    constexpr const char *context = "CUDA multilayer scalar first plane";
    multilayer_result_local_key = nullptr;
    multilayer_result_alpha_key = nullptr;
    multilayer_result_nmax = 0;
    multilayer_result_ndipole = 0;
    if (!cpu_image || !cpu_local_field || !cpu_incident_dipoles || !cpu_xs || !cpu_ys || !cpu_zs ||
        !ndipole || !nfft2d || !nmax || !kkm || !jjm || !xmin || !ymin || !first_z || !aretecube ||
        !k0 || !E0 || !eps0 || !w0 || !ss || !pp || !theta || !phi || !beam_type ||
        !conjugate_projection || !ux || !uy || !uz || *ndipole <= 0 || *nfft2d <= 0 ||
        *nmax < *ndipole || *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0) || *beam_type < 0 ||
        *beam_type > 3 || ((*beam_type == 2 || *beam_type == 3) && *w0 <= CU_REAL(0)) ||
        (*conjugate_projection != 0 && *conjugate_projection != 1))
        return 1;

    const CUCOMPLEX E0_cuda = to_cuda_complex(*E0);
    const CUCOMPLEX eps0_cuda = to_cuda_complex(*eps0);
    const CUCOMPLEX ux_cuda = to_cuda_complex(*ux);
    const CUCOMPLEX uy_cuda = to_cuda_complex(*uy);
    const CUCOMPLEX uz_cuda = to_cuda_complex(*uz);

    const size_t plane = size_t(*nfft2d) * size_t(*nfft2d);
    const size_t local_count = size_t(3) * size_t(*nmax);
    if (!reserve_device(multilayer_image, &multilayer_image_capacity,
                        plane, sizeof(CUCOMPLEX), "multilayer.scalar.image", context) ||
        !reserve_device(multilayer_local_field,
                        &multilayer_local_field_capacity, local_count, sizeof(CUCOMPLEX),
                        "multilayer.scalar.local_field", context) ||
        !reserve_device(multilayer_plane_map,
                        &multilayer_plane_map_capacity, plane, sizeof(int),
                        "multilayer.scalar.plane_map", context) ||
        !reserve_device(multilayer_first_plane_x,
                        &multilayer_first_plane_x_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.x", context) ||
        !reserve_device(multilayer_first_plane_y,
                        &multilayer_first_plane_y_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.y", context) ||
        !reserve_device(multilayer_first_plane_z,
                        &multilayer_first_plane_z_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.z", context) ||
        !reserve_device(multilayer_first_plane_incident,
                        &multilayer_first_plane_incident_capacity, local_count, sizeof(CUCOMPLEX),
                        "multilayer.first_plane.incident", context))
        return 3;

    bool ok = cuda_utils::check_cuda(
                  cuda_data::copy_host_to_device(multilayer_first_plane_x, cpu_xs,
                                                 size_t(*ndipole) * sizeof(CU_REAL),
                                                 cuda_utils::TransferCategory::Geometry),
                  context, "upload x coordinates") &&
              cuda_utils::check_cuda(
                  cuda_data::copy_host_to_device(multilayer_first_plane_y, cpu_ys,
                                                 size_t(*ndipole) * sizeof(CU_REAL),
                                                 cuda_utils::TransferCategory::Geometry),
                  context, "upload y coordinates") &&
              cuda_utils::check_cuda(
                  cuda_data::copy_host_to_device(multilayer_first_plane_z, cpu_zs,
                                                 size_t(*ndipole) * sizeof(CU_REAL),
                                                 cuda_utils::TransferCategory::Geometry),
                  context, "upload z coordinates") &&
              cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                                         multilayer_first_plane_incident, cpu_incident_dipoles,
                                         local_count * sizeof(CUCOMPLEX),
                                         cuda_utils::TransferCategory::IncidentField),
                                     context, "upload incident dipole fields") &&
              cuda_utils::check_cuda(cudaMemset(multilayer_plane_map, 0, plane * sizeof(int)),
                                     context, "clear first-plane map") &&
              cuda_utils::check_cuda(
                  cudaMemset(multilayer_local_field, 0, local_count * sizeof(CUCOMPLEX)), context,
                  "clear local field");
    if (!ok)
        return 4;

    if (!cuda_utils::check_cuda(multilayer_scalar_map_first_plane_wrapper(
                                    multilayer_first_plane_x, multilayer_first_plane_y,
                                    multilayer_first_plane_z, multilayer_plane_map, *ndipole,
                                    *nfft2d, *kkm, *jjm, *xmin, *ymin, *first_z, *aretecube),
                                context, "map first-plane dipoles") ||
        !cuda_utils::check_cuda(multilayer_scalar_initialize_plane_wrapper(
                                    multilayer_plane_map, multilayer_first_plane_incident,
                                    multilayer_image, multilayer_local_field, *nfft2d, *nmax, *kkm,
                                    *jjm, *xmin, *ymin, *first_z, *aretecube, *k0, E0_cuda,
                                    eps0_cuda, *w0, *ss, *pp, *theta, *phi, *beam_type,
                                    *conjugate_projection, ux_cuda, uy_cuda, uz_cuda),
                                context, "initialize first-plane incident image"))
        return 5;

    multilayer_first_plane_ready = true;
    multilayer_first_plane_image_key = cpu_image;
    multilayer_first_plane_local_key = cpu_local_field;
    multilayer_first_plane_nfft = *nfft2d;
    multilayer_first_plane_nmax = *nmax;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_multilayer_scalar_initialize_rigorous_plane_cuda_(
    COMPLEX *cpu_image, COMPLEX *cpu_local_field, const COMPLEX *cpu_incident_dipoles,
    const CU_REAL *cpu_xs, const CU_REAL *cpu_ys, const CU_REAL *cpu_zs, const int *ndipole,
    const int *nfft2d, const int *nmax, const int *kkm, const int *jjm, const CU_REAL *xmin,
    const CU_REAL *ymin, const CU_REAL *first_z, const CU_REAL *aretecube, const CU_REAL *k0,
    const COMPLEX *E0, const CU_REAL *ss, const CU_REAL *pp, const CU_REAL *theta,
    const CU_REAL *phi, const CU_REAL *x0, const CU_REAL *y0, const CU_REAL *z0,
    const CU_REAL *w0, const CU_REAL *tolerance, const int *circular,
    const int *conjugate_projection, const COMPLEX *ux, const COMPLEX *uy,
    const COMPLEX *uz) {
    constexpr const char *context = "CUDA multilayer rigorous Gaussian first plane";
    multilayer_result_local_key = nullptr;
    multilayer_result_alpha_key = nullptr;
    multilayer_result_nmax = 0;
    multilayer_result_ndipole = 0;
    if (!cpu_image || !cpu_local_field || !cpu_incident_dipoles || !cpu_xs || !cpu_ys || !cpu_zs ||
        !ndipole || !nfft2d || !nmax || !kkm || !jjm || !xmin || !ymin || !first_z || !aretecube ||
        !k0 || !E0 || !ss || !pp || !theta || !phi || !x0 || !y0 || !z0 || !w0 || !tolerance ||
        !circular || !conjugate_projection || !ux || !uy || !uz || *ndipole <= 0 || *nfft2d <= 0 ||
        *nfft2d > INT_MAX / *nfft2d || *nmax < *ndipole || *aretecube <= CU_REAL(0) ||
        *k0 <= CU_REAL(0) || *w0 <= CU_REAL(0) || *tolerance <= CU_REAL(0) ||
        (*circular != 0 && *circular != 1) ||
        (*conjugate_projection != 0 && *conjugate_projection != 1))
        return 1;

    const CUCOMPLEX E0_cuda = to_cuda_complex(*E0);
    const CUCOMPLEX ux_cuda = to_cuda_complex(*ux);
    const CUCOMPLEX uy_cuda = to_cuda_complex(*uy);
    const CUCOMPLEX uz_cuda = to_cuda_complex(*uz);

    const size_t plane = size_t(*nfft2d) * size_t(*nfft2d);
    const size_t local_count = size_t(3) * size_t(*nmax);
    if (!initialize_cuda_device())
        return 2;
    cuda_data::GaussianFftDeviceBuffers gaussian{};
    if (!cuda_data::get_device_gaussian_fft_buffers(plane, plane, &gaussian))
        return 3;
    if (!reserve_device(multilayer_image, &multilayer_image_capacity,
                        plane, sizeof(CUCOMPLEX), "multilayer.scalar.image", context) ||
        !reserve_device(multilayer_local_field,
                        &multilayer_local_field_capacity, local_count, sizeof(CUCOMPLEX),
                        "multilayer.scalar.local_field", context) ||
        !reserve_device(multilayer_plane_map,
                        &multilayer_plane_map_capacity, plane, sizeof(int),
                        "multilayer.scalar.plane_map", context) ||
        !reserve_device(multilayer_first_plane_x,
                        &multilayer_first_plane_x_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.x", context) ||
        !reserve_device(multilayer_first_plane_y,
                        &multilayer_first_plane_y_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.y", context) ||
        !reserve_device(multilayer_first_plane_z,
                        &multilayer_first_plane_z_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.z", context) ||
        !reserve_device(multilayer_first_plane_incident,
                        &multilayer_first_plane_incident_capacity, local_count, sizeof(CUCOMPLEX),
                        "multilayer.first_plane.incident", context))
        return 3;

    bool ok = cuda_utils::check_cuda(
        cuda_data::copy_host_to_device(gaussian.dipole_xs, cpu_xs,
            size_t(*ndipole) * sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
        context, "upload dipole x coordinates");
    ok = ok && cuda_utils::check_cuda(
        cuda_data::copy_host_to_device(gaussian.dipole_ys, cpu_ys,
            size_t(*ndipole) * sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
        context, "upload dipole y coordinates");
    ok = ok && cuda_utils::check_cuda(
        cuda_data::copy_host_to_device(multilayer_first_plane_x, cpu_xs,
            size_t(*ndipole) * sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
        context, "upload first-plane x coordinates");
    ok = ok && cuda_utils::check_cuda(
        cuda_data::copy_host_to_device(multilayer_first_plane_y, cpu_ys,
            size_t(*ndipole) * sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
        context, "upload first-plane y coordinates");
    ok = ok && cuda_utils::check_cuda(
        cuda_data::copy_host_to_device(multilayer_first_plane_z, cpu_zs,
            size_t(*ndipole) * sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
        context, "upload first-plane z coordinates");
    ok = ok && cuda_utils::check_cuda(
        cuda_data::copy_host_to_device(multilayer_first_plane_incident, cpu_incident_dipoles,
            local_count * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::IncidentField),
        context, "upload incident dipole fields");
    if (!ok)
        return 4;

    {
        std::lock_guard<std::mutex> lock(multilayer_gkp_mutex);
        if (!multilayer_gkp_uploaded) {
            if (!cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                    gaussian.gkp_nodes, gp_x, sizeof(gp_x),
                    cuda_utils::TransferCategory::IncidentField), context, "upload GKP nodes") ||
                !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                    gaussian.gkp_weights, gp_w, sizeof(gp_w),
                    cuda_utils::TransferCategory::IncidentField), context, "upload GKP weights"))
                return 5;
            multilayer_gkp_uploaded = true;
        }
    }

    if (!cuda_utils::check_cuda(gaussianfft_build_coordinates_wrapper(
            gaussian.xs, gaussian.ys, gaussian.zs, *nfft2d,
            *nfft2d - 2 * *kkm, *nfft2d - 2 * *jjm,
            *xmin, *ymin, *first_z, *aretecube), context, "create plane coordinates"))
        return 6;
    if (!cuda_utils::check_cuda(rigorous_gaussian_incident_wrapper(
            gaussian.xs, gaussian.ys, gaussian.zs, int(plane), double(*x0), double(*y0),
            double(*z0), double(*theta), double(*phi), double(*w0), double(*k0),
            double(*ss), double(*pp), *E0, double(*tolerance), *circular, 0,
            gaussian.gkp_nodes, gaussian.gkp_weights, gaussian.rigorous_field,
            gaussian.convergence), context, "evaluate rigorous Gaussian field"))
        return 7;

    if (!cuda_utils::check_cuda(cudaMemset(multilayer_plane_map, 0, plane * sizeof(int)),
                                context, "clear plane map") ||
        !cuda_utils::check_cuda(cudaMemset(multilayer_local_field, 0,
                                           local_count * sizeof(CUCOMPLEX)),
                                context, "clear local field") ||
        !cuda_utils::check_cuda(multilayer_scalar_map_first_plane_wrapper(
                                    multilayer_first_plane_x, multilayer_first_plane_y,
                                    multilayer_first_plane_z, multilayer_plane_map, *ndipole,
                                    *nfft2d, *kkm, *jjm, *xmin, *ymin, *first_z, *aretecube),
                                context, "map first-plane dipoles") ||
        !cuda_utils::check_cuda(multilayer_scalar_initialize_rigorous_plane_wrapper(
                                    multilayer_plane_map, multilayer_first_plane_incident,
                                    gaussian.rigorous_field, multilayer_image, multilayer_local_field,
                                    *nfft2d, *nmax, *conjugate_projection,
                                    ux_cuda, uy_cuda, uz_cuda),
                                context, "initialize rigorous first plane"))
        return 8;

    multilayer_first_plane_ready = true;
    multilayer_first_plane_image_key = cpu_image;
    multilayer_first_plane_local_key = cpu_local_field;
    multilayer_first_plane_nfft = *nfft2d;
    multilayer_first_plane_nmax = *nmax;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_multilayer_vector_initialize_plane_cuda_(
    COMPLEX *cpu_image_x, COMPLEX *cpu_image_y, COMPLEX *cpu_image_z,
    COMPLEX *cpu_local_field, const COMPLEX *cpu_incident_dipoles, const CU_REAL *cpu_xs,
    const CU_REAL *cpu_ys, const CU_REAL *cpu_zs, const int *ndipole, const int *nfft2d,
    const int *nmax, const int *kkm, const int *jjm, const CU_REAL *xmin, const CU_REAL *ymin,
    const CU_REAL *first_z, const CU_REAL *aretecube, const CU_REAL *k0, const COMPLEX *E0,
    const COMPLEX *eps0, const CU_REAL *w0, const CU_REAL *ss, const CU_REAL *pp,
    const CU_REAL *theta, const CU_REAL *phi, const int *beam_type) {
    constexpr const char *context = "CUDA multilayer vector first plane";
    multilayer_vector_first_plane_ready = false;
    multilayer_vector_result_local_key = nullptr;
    multilayer_vector_result_alpha_key = nullptr;
    multilayer_vector_result_nmax = 0;
    multilayer_vector_result_ndipole = 0;
    if (!cpu_image_x || !cpu_image_y || !cpu_image_z || !cpu_local_field || !cpu_incident_dipoles ||
        !cpu_xs || !cpu_ys || !cpu_zs || !ndipole || !nfft2d || !nmax || !kkm || !jjm || !xmin ||
        !ymin || !first_z || !aretecube || !k0 || !E0 || !eps0 || !w0 || !ss || !pp || !theta ||
        !phi || !beam_type || *ndipole <= 0 || *nfft2d <= 0 || *nmax < *ndipole ||
        *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0) || *beam_type < 0 || *beam_type > 3 ||
        ((*beam_type == 2 || *beam_type == 3) && *w0 <= CU_REAL(0)))
        return 1;

    const CUCOMPLEX E0_cuda = to_cuda_complex(*E0);
    const CUCOMPLEX eps0_cuda = to_cuda_complex(*eps0);

    const size_t plane = size_t(*nfft2d) * size_t(*nfft2d);
    const size_t local_count = size_t(3) * size_t(*nmax);
    static const char *image_buffer_names[3] = {
        "multilayer.vector.image.x", "multilayer.vector.image.y", "multilayer.vector.image.z"};
    for (int c = 0; c < 3; ++c)
        if (!reserve_device(multilayer_vector_images[c],
                            &multilayer_vector_image_capacity[c], plane, sizeof(CUCOMPLEX),
                            image_buffer_names[c], context))
            return 3;
    if (!reserve_device(multilayer_vector_local,
                        &multilayer_vector_local_capacity, local_count, sizeof(CUCOMPLEX),
                        "multilayer.vector.local_field", context) ||
        !reserve_device(multilayer_vector_map,
                        &multilayer_vector_map_capacity, plane, sizeof(int),
                        "multilayer.vector.plane_map", context) ||
        !reserve_device(multilayer_first_plane_x,
                        &multilayer_first_plane_x_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.x", context) ||
        !reserve_device(multilayer_first_plane_y,
                        &multilayer_first_plane_y_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.y", context) ||
        !reserve_device(multilayer_first_plane_z,
                        &multilayer_first_plane_z_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.z", context) ||
        !reserve_device(multilayer_first_plane_incident,
                        &multilayer_first_plane_incident_capacity, local_count, sizeof(CUCOMPLEX),
                        "multilayer.first_plane.incident", context))
        return 3;

    bool ok = cuda_utils::check_cuda(
                  cuda_data::copy_host_to_device(multilayer_first_plane_x, cpu_xs,
                                                 size_t(*ndipole) * sizeof(CU_REAL),
                                                 cuda_utils::TransferCategory::Geometry),
                  context, "upload x coordinates") &&
              cuda_utils::check_cuda(
                  cuda_data::copy_host_to_device(multilayer_first_plane_y, cpu_ys,
                                                 size_t(*ndipole) * sizeof(CU_REAL),
                                                 cuda_utils::TransferCategory::Geometry),
                  context, "upload y coordinates") &&
              cuda_utils::check_cuda(
                  cuda_data::copy_host_to_device(multilayer_first_plane_z, cpu_zs,
                                                 size_t(*ndipole) * sizeof(CU_REAL),
                                                 cuda_utils::TransferCategory::Geometry),
                  context, "upload z coordinates") &&
              cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                                         multilayer_first_plane_incident, cpu_incident_dipoles,
                                         local_count * sizeof(CUCOMPLEX),
                                         cuda_utils::TransferCategory::IncidentField),
                                     context, "upload incident dipole fields") &&
              cuda_utils::check_cuda(cudaMemset(multilayer_vector_map, 0, plane * sizeof(int)),
                                     context, "clear first-plane map") &&
              cuda_utils::check_cuda(
                  cudaMemset(multilayer_vector_local, 0, local_count * sizeof(CUCOMPLEX)), context,
                  "clear local field");
    for (int c = 0; c < 3 && ok; ++c)
        ok = cuda_utils::check_cuda(
            cudaMemset(multilayer_vector_images[c], 0, plane * sizeof(CUCOMPLEX)), context,
            "clear incident image");
    if (!ok)
        return 4;

    if (!cuda_utils::check_cuda(multilayer_scalar_map_first_plane_wrapper(
                                    multilayer_first_plane_x, multilayer_first_plane_y,
                                    multilayer_first_plane_z, multilayer_vector_map, *ndipole,
                                    *nfft2d, *kkm, *jjm, *xmin, *ymin, *first_z, *aretecube),
                                context, "map first-plane dipoles") ||
        !cuda_utils::check_cuda(multilayer_vector_initialize_plane_wrapper(
                                    multilayer_vector_map, multilayer_first_plane_incident,
                                    multilayer_vector_images[0], multilayer_vector_images[1],
                                    multilayer_vector_images[2], multilayer_vector_local, *nfft2d,
                                    *nmax, *kkm, *jjm, *xmin, *ymin, *first_z, *aretecube, *k0,
                                    E0_cuda, eps0_cuda, *w0, *ss, *pp, *theta, *phi, *beam_type),
                                context, "initialize vector first plane"))
        return 5;

    multilayer_vector_first_plane_ready = true;
    multilayer_vector_first_plane_image_keys[0] = cpu_image_x;
    multilayer_vector_first_plane_image_keys[1] = cpu_image_y;
    multilayer_vector_first_plane_image_keys[2] = cpu_image_z;
    multilayer_vector_first_plane_local_key = cpu_local_field;
    multilayer_vector_first_plane_nfft = *nfft2d;
    multilayer_vector_first_plane_nmax = *nmax;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_multilayer_vector_initialize_rigorous_plane_cuda_(
    COMPLEX *cpu_image_x, COMPLEX *cpu_image_y, COMPLEX *cpu_image_z,
    COMPLEX *cpu_local_field, const COMPLEX *cpu_incident_dipoles,
    const CU_REAL *cpu_xs, const CU_REAL *cpu_ys, const CU_REAL *cpu_zs,
    const int *ndipole, const int *nfft2d, const int *nmax, const int *kkm,
    const int *jjm, const CU_REAL *xmin, const CU_REAL *ymin, const CU_REAL *first_z,
    const CU_REAL *aretecube, const CU_REAL *k0, const COMPLEX *E0, const CU_REAL *ss,
    const CU_REAL *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *x0,
    const CU_REAL *y0, const CU_REAL *z0, const CU_REAL *w0, const CU_REAL *tolerance,
    const int *circular) {
    constexpr const char *context = "CUDA multilayer vector rigorous Gaussian first plane";
    multilayer_vector_first_plane_ready = false;
    multilayer_vector_result_local_key = nullptr;
    multilayer_vector_result_alpha_key = nullptr;
    multilayer_vector_result_nmax = 0;
    multilayer_vector_result_ndipole = 0;
    if (!cpu_image_x || !cpu_image_y || !cpu_image_z || !cpu_local_field ||
        !cpu_incident_dipoles || !cpu_xs || !cpu_ys || !cpu_zs || !ndipole || !nfft2d ||
        !nmax || !kkm || !jjm || !xmin || !ymin || !first_z || !aretecube || !k0 || !E0 ||
        !ss || !pp || !theta || !phi || !x0 || !y0 || !z0 || !w0 || !tolerance || !circular ||
        *ndipole <= 0 || *nfft2d <= 0 || *nfft2d > INT_MAX / *nfft2d || *nmax < *ndipole ||
        *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0) || *w0 <= CU_REAL(0) ||
        *tolerance <= CU_REAL(0) || (*circular != 0 && *circular != 1))
        return 1;

    const CUCOMPLEX E0_cuda = to_cuda_complex(*E0);

    const size_t plane = size_t(*nfft2d) * size_t(*nfft2d);
    const size_t local_count = size_t(3) * size_t(*nmax);
    if (!initialize_cuda_device())
        return 2;
    cuda_data::GaussianFftDeviceBuffers gaussian{};
    if (!cuda_data::get_device_gaussian_fft_buffers(plane, plane, &gaussian))
        return 3;
    static const char *image_names[3] = {
        "multilayer.vector.image.x", "multilayer.vector.image.y", "multilayer.vector.image.z"};
    for (int component = 0; component < 3; ++component)
        if (!reserve_device(multilayer_vector_images[component],
                            &multilayer_vector_image_capacity[component], plane,
                            sizeof(CUCOMPLEX), image_names[component], context))
            return 3;
    if (!reserve_device(multilayer_vector_local,
                        &multilayer_vector_local_capacity, local_count, sizeof(CUCOMPLEX),
                        "multilayer.vector.local_field", context) ||
        !reserve_device(multilayer_vector_map,
                        &multilayer_vector_map_capacity, plane, sizeof(int),
                        "multilayer.vector.plane_map", context) ||
        !reserve_device(multilayer_first_plane_x,
                        &multilayer_first_plane_x_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.x", context) ||
        !reserve_device(multilayer_first_plane_y,
                        &multilayer_first_plane_y_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.y", context) ||
        !reserve_device(multilayer_first_plane_z,
                        &multilayer_first_plane_z_capacity, size_t(*ndipole), sizeof(CU_REAL),
                        "multilayer.first_plane.z", context) ||
        !reserve_device(multilayer_first_plane_incident,
                        &multilayer_first_plane_incident_capacity, local_count,
                        sizeof(CUCOMPLEX), "multilayer.first_plane.incident", context))
        return 3;

    bool ok = cuda_utils::check_cuda(cuda_data::copy_host_to_device(
        gaussian.dipole_xs, cpu_xs, size_t(*ndipole) * sizeof(CU_REAL),
        cuda_utils::TransferCategory::Geometry), context, "upload Gaussian dipole x") &&
        cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            gaussian.dipole_ys, cpu_ys, size_t(*ndipole) * sizeof(CU_REAL),
            cuda_utils::TransferCategory::Geometry), context, "upload Gaussian dipole y") &&
        cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            multilayer_first_plane_x, cpu_xs, size_t(*ndipole) * sizeof(CU_REAL),
            cuda_utils::TransferCategory::Geometry), context, "upload first-plane x") &&
        cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            multilayer_first_plane_y, cpu_ys, size_t(*ndipole) * sizeof(CU_REAL),
            cuda_utils::TransferCategory::Geometry), context, "upload first-plane y") &&
        cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            multilayer_first_plane_z, cpu_zs, size_t(*ndipole) * sizeof(CU_REAL),
            cuda_utils::TransferCategory::Geometry), context, "upload first-plane z") &&
        cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            multilayer_first_plane_incident, cpu_incident_dipoles,
            local_count * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::IncidentField),
            context, "upload incident dipole fields");
    if (!ok)
        return 4;
    {
        std::lock_guard<std::mutex> lock(multilayer_gkp_mutex);
        if (!multilayer_gkp_uploaded) {
            if (!cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                    gaussian.gkp_nodes, gp_x, sizeof(gp_x),
                    cuda_utils::TransferCategory::IncidentField), context, "upload GKP nodes") ||
                !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                    gaussian.gkp_weights, gp_w, sizeof(gp_w),
                    cuda_utils::TransferCategory::IncidentField), context, "upload GKP weights"))
                return 5;
            multilayer_gkp_uploaded = true;
        }
    }
    if (!cuda_utils::check_cuda(gaussianfft_build_coordinates_wrapper(
            gaussian.xs, gaussian.ys, gaussian.zs, *nfft2d,
            *nfft2d - 2 * *kkm, *nfft2d - 2 * *jjm, *xmin, *ymin, *first_z, *aretecube),
            context, "create rigorous vector plane coordinates"))
        return 6;
    if (!cuda_utils::check_cuda(rigorous_gaussian_incident_wrapper(
            gaussian.xs, gaussian.ys, gaussian.zs, int(plane), double(*x0), double(*y0),
            double(*z0), double(*theta), double(*phi), double(*w0), double(*k0),
            double(*ss), double(*pp), *E0, double(*tolerance), *circular, 0,
            gaussian.gkp_nodes, gaussian.gkp_weights, gaussian.rigorous_field,
            gaussian.convergence), context, "evaluate rigorous vector Gaussian field"))
        return 7;

    if (!cuda_utils::check_cuda(cudaMemset(multilayer_vector_map, 0, plane * sizeof(int)),
                                context, "clear vector plane map") ||
        !cuda_utils::check_cuda(cudaMemset(multilayer_vector_local, 0,
                                           local_count * sizeof(CUCOMPLEX)),
                                context, "clear vector local field"))
        return 8;
    for (int component = 0; component < 3; ++component)
        if (!cuda_utils::check_cuda(cudaMemset(multilayer_vector_images[component], 0,
                                               plane * sizeof(CUCOMPLEX)),
                                    context, "clear vector image"))
            return 8;
    if (!cuda_utils::check_cuda(multilayer_scalar_map_first_plane_wrapper(
            multilayer_first_plane_x, multilayer_first_plane_y, multilayer_first_plane_z,
            multilayer_vector_map, *ndipole, *nfft2d, *kkm, *jjm, *xmin, *ymin,
            *first_z, *aretecube), context, "map vector first-plane dipoles") ||
        !cuda_utils::check_cuda(multilayer_vector_initialize_rigorous_plane_wrapper(
            multilayer_vector_map, multilayer_first_plane_incident, gaussian.rigorous_field,
            multilayer_vector_images[0], multilayer_vector_images[1],
            multilayer_vector_images[2], multilayer_vector_local, *nfft2d, *nmax),
            context, "initialize rigorous vector first plane"))
        return 9;

    multilayer_vector_first_plane_ready = true;
    multilayer_vector_first_plane_image_keys[0] = cpu_image_x;
    multilayer_vector_first_plane_image_keys[1] = cpu_image_y;
    multilayer_vector_first_plane_image_keys[2] = cpu_image_z;
    multilayer_vector_first_plane_local_key = cpu_local_field;
    multilayer_vector_first_plane_nfft = *nfft2d;
    multilayer_vector_first_plane_nmax = *nmax;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_multilayerbornsca_layers_cuda_(
    COMPLEX *cpu_image, COMPLEX *cpu_local_field, const COMPLEX *cpu_polarizability,
    const int *cpu_plane_map, const COMPLEX *cpu_green_key, const int *nfft2d, const int *nx,
    const int *ny, const int *nz, const int *nmax, const int *niso, const int *reflection,
    const CU_REAL *deltakx, const CU_REAL *deltaky, const CU_REAL *aretecube, const CU_REAL *k0,
    const COMPLEX *ux, const COMPLEX *uy, const COMPLEX *uz) {
    constexpr const char *context = "CUDA multilayer scalar layers";
    multilayer_result_local_key = nullptr;
    multilayer_result_alpha_key = nullptr;
    multilayer_result_nmax = 0;
    multilayer_result_ndipole = 0;
    if (!cpu_image || !cpu_local_field || !cpu_polarizability || !cpu_plane_map || !cpu_green_key ||
        !nfft2d || !nx || !ny || !nz || !nmax || !niso || !reflection || !deltakx || !deltaky ||
        !aretecube || !k0 || !ux || !uy || !uz || *nfft2d <= 0 || *nx <= 0 || *ny <= 0 ||
        *nz <= 0 || *nmax <= 0 || (*niso != 1 && *niso != 3))
        return 1;

    const CUCOMPLEX ux_cuda = to_cuda_complex(*ux);
    const CUCOMPLEX uy_cuda = to_cuda_complex(*uy);
    const CUCOMPLEX uz_cuda = to_cuda_complex(*uz);

    const size_t fft_side = size_t(2) * size_t(*nfft2d);
    size_t fft_count = 0, image_count = 0, local_count = 0, alpha_count = 0, history_count = 0;
    if (!cuda_utils::checked_multiply(fft_side, fft_side, &fft_count) ||
        !cuda_utils::checked_multiply(size_t(*nfft2d), size_t(*nfft2d), &image_count) ||
        !cuda_utils::checked_multiply(size_t(3), size_t(*nmax), &local_count) ||
        !cuda_utils::checked_multiply(size_t(*nmax), size_t(*niso), &alpha_count) ||
        !cuda_utils::checked_multiply(alpha_count, size_t(*niso), &alpha_count) ||
        fft_count > size_t(std::numeric_limits<int>::max()) ||
        image_count > size_t(std::numeric_limits<int>::max()) ||
        size_t(*nmax) > size_t(std::numeric_limits<int>::max()))
        return 2;
    if (*reflection && !cuda_utils::checked_multiply(image_count, size_t(*nz), &history_count))
        return 2;

    CUCOMPLEX *gpu_green = get_device_scalar_green_output(cpu_green_key, fft_count);
    CUCOMPLEX *gpu_polarization = get_device_vectors(fft_count);
    if (!gpu_green || !gpu_polarization ||
        !reserve_device(multilayer_image, &multilayer_image_capacity,
                        image_count, sizeof(CUCOMPLEX), "multilayer.scalar.image", context) ||
        !reserve_device(multilayer_polarizability,
                        &multilayer_polarizability_capacity, alpha_count, sizeof(CUCOMPLEX),
                        "multilayer.scalar.polarizability", context) ||
        !reserve_device(multilayer_local_field,
                        &multilayer_local_field_capacity, local_count, sizeof(CUCOMPLEX),
                        "multilayer.scalar.local_field", context) ||
        !reserve_device(multilayer_plane_map,
                        &multilayer_plane_map_capacity, image_count, sizeof(int),
                        "multilayer.scalar.plane_map", context) ||
        (*reflection && !reserve_device(multilayer_history,
                                        &multilayer_history_capacity, history_count,
                                        sizeof(CUCOMPLEX), "multilayer.scalar.history", context)))
        return 3;

    const bool preinitialized =
        multilayer_first_plane_ready && multilayer_first_plane_image_key == cpu_image &&
        multilayer_first_plane_local_key == cpu_local_field &&
        multilayer_first_plane_nfft == *nfft2d && multilayer_first_plane_nmax == *nmax;
    bool ok =
        (preinitialized ||
         cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                                    multilayer_image, cpu_image, image_count * sizeof(CUCOMPLEX),
                                    cuda_utils::TransferCategory::IncidentField),
                                context, "upload incident image")) &&
        cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(multilayer_polarizability, cpu_polarizability,
                                           alpha_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Material),
            context, "upload scalar polarizability") &&
        (preinitialized ||
         cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                                    multilayer_plane_map, cpu_plane_map, image_count * sizeof(int),
                                    cuda_utils::TransferCategory::Geometry),
                                context, "upload plane dipole map")) &&
        (preinitialized || cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                                                      multilayer_local_field, cpu_local_field,
                                                      local_count * sizeof(CUCOMPLEX),
                                                      cuda_utils::TransferCategory::SolverVector),
                                                  context, "upload initial local field"));
    if (!ok)
        return 4;

    cuda_data::MultilayerFftPlans fft_plans{};
    if (!cuda_data::get_device_multilayer_fft_plans(int(fft_side), *nfft2d, &fft_plans))
        return 5;
    const cufftHandle layer_plan = fft_plans.layer;
    const cufftHandle image_plan = fft_plans.image;

    for (int layer = 1; layer < *nz && ok; ++layer) {
        ok = cuda_utils::check_cuda(multilayer_scalar_prepare_polarization_wrapper(
                                        multilayer_image, nullptr, multilayer_plane_map,
                                        multilayer_polarizability, gpu_polarization, *nfft2d, *nx,
                                        *ny, layer, int(fft_count)),
                                    context, "prepare layer polarization") &&
             cuda_utils::check_cufft(
                 CUFFT_EXECUTE(layer_plan, gpu_polarization, gpu_polarization, CUFFT_INVERSE),
                 context, "forward layer convolution input") &&
             cuda_utils::check_cuda(multilayer_scalar_multiply_green_wrapper(
                                        gpu_polarization, gpu_green, int(fft_count)),
                                    context, "multiply layer Green tensor") &&
             cuda_utils::check_cufft(
                 CUFFT_EXECUTE(layer_plan, gpu_polarization, gpu_polarization, CUFFT_FORWARD),
                 context, "inverse layer convolution") &&
             cuda_utils::check_cufft(
                 CUFFT_EXECUTE(image_plan, multilayer_image, multilayer_image, CUFFT_INVERSE),
                 context, "forward image propagation") &&
             cuda_utils::check_cuda(multilayer_scalar_propagate_wrapper(multilayer_image, *nfft2d,
                                                                        *deltakx, *deltaky, *k0,
                                                                        *aretecube),
                                    context, "propagate layer image") &&
             cuda_utils::check_cufft(
                 CUFFT_EXECUTE(image_plan, multilayer_image, multilayer_image, CUFFT_FORWARD),
                 context, "inverse image propagation") &&
             cuda_utils::check_cuda(multilayer_scalar_accumulate_wrapper(
                                        multilayer_image, gpu_polarization, multilayer_local_field,
                                        multilayer_plane_map, *nfft2d, *nx, *ny, layer, *nmax,
                                        int(image_count), ux_cuda, uy_cuda, uz_cuda,
                                        CU_REAL(1) / CU_REAL(image_count),
                                        CU_REAL(1) / CU_REAL(fft_count), 0),
                                    context, "accumulate layer fields") &&
             (!*reflection || cuda_utils::check_cuda(multilayer_scalar_save_plane_wrapper(
                                                         multilayer_image, multilayer_history,
                                                         int(image_count), layer, *nz),
                                                     context, "save forward layer field"));
    }

    if (ok && *reflection) {
        const CUCOMPLEX *top_field = multilayer_history + size_t(*nz - 1) * image_count;
        ok = cuda_utils::check_cuda(multilayer_scalar_prepare_polarization_wrapper(
                                        top_field, nullptr, multilayer_plane_map,
                                        multilayer_polarizability, gpu_polarization, *nfft2d, *nx,
                                        *ny, *nz - 1, int(fft_count)),
                                    context, "prepare transmission polarization") &&
             cuda_utils::check_cufft(
                 CUFFT_EXECUTE(layer_plan, gpu_polarization, gpu_polarization, CUFFT_INVERSE),
                 context, "forward transmission polarization") &&
             cuda_utils::check_cuda(multilayer_scalar_multiply_green_wrapper(
                                        gpu_polarization, gpu_green, int(fft_count)),
                                    context, "multiply transmission Green tensor") &&
             cuda_utils::check_cufft(
                 CUFFT_EXECUTE(layer_plan, gpu_polarization, gpu_polarization, CUFFT_FORWARD),
                 context, "inverse transmission convolution") &&
             cuda_utils::check_cuda(multilayer_scalar_accumulate_wrapper(
                                        multilayer_image, gpu_polarization, multilayer_local_field,
                                        multilayer_plane_map, *nfft2d, *nx, *ny, *nz - 2, *nmax,
                                        int(image_count), ux_cuda, uy_cuda, uz_cuda, CU_REAL(0),
                                        CU_REAL(1) / CU_REAL(fft_count), 1),
                                    context, "accumulate transmission reflection");
    }

    if (ok && *reflection) {
        for (int layer = *nz - 2; layer >= 1 && ok; --layer) {
            const CUCOMPLEX *forward_field = multilayer_history + size_t(layer) * image_count;
            ok = cuda_utils::check_cuda(multilayer_scalar_prepare_polarization_wrapper(
                                            forward_field, multilayer_image, multilayer_plane_map,
                                            multilayer_polarizability, gpu_polarization, *nfft2d,
                                            *nx, *ny, layer, int(fft_count)),
                                        context, "prepare reflected layer polarization") &&
                 cuda_utils::check_cufft(
                     CUFFT_EXECUTE(layer_plan, gpu_polarization, gpu_polarization, CUFFT_INVERSE),
                     context, "forward reflected polarization") &&
                 cuda_utils::check_cuda(multilayer_scalar_multiply_green_wrapper(
                                            gpu_polarization, gpu_green, int(fft_count)),
                                        context, "multiply reflected Green tensor") &&
                 cuda_utils::check_cufft(
                     CUFFT_EXECUTE(layer_plan, gpu_polarization, gpu_polarization, CUFFT_FORWARD),
                     context, "inverse reflected convolution") &&
                 cuda_utils::check_cufft(
                     CUFFT_EXECUTE(image_plan, multilayer_image, multilayer_image, CUFFT_INVERSE),
                     context, "forward reflected propagation") &&
                 cuda_utils::check_cuda(
                     multilayer_scalar_propagate_wrapper(multilayer_image, *nfft2d, *deltakx,
                                                         *deltaky, *k0, *aretecube),
                     context, "propagate reflected field") &&
                 cuda_utils::check_cufft(
                     CUFFT_EXECUTE(image_plan, multilayer_image, multilayer_image, CUFFT_FORWARD),
                     context, "inverse reflected propagation") &&
                 cuda_utils::check_cuda(multilayer_scalar_accumulate_wrapper(
                                            multilayer_image, gpu_polarization,
                                            multilayer_local_field, multilayer_plane_map, *nfft2d,
                                            *nx, *ny, layer, *nmax, int(image_count), ux_cuda, uy_cuda, uz_cuda,
                                            CU_REAL(1) / CU_REAL(image_count),
                                            CU_REAL(1) / CU_REAL(fft_count), 1),
                                        context, "accumulate reflected layer");
        }
    }

    if (ok) {
        // The Fortran caller consumes only the local-field vector after this
        // routine; the propagated image is scratch state and stays on device.
        ok = cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(cpu_local_field, multilayer_local_field,
                                           local_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Result),
            context, "download local fields");
    }
    multilayer_first_plane_ready = false;
    multilayer_first_plane_image_key = nullptr;
    multilayer_first_plane_local_key = nullptr;
    if (ok) {
        multilayer_result_local_key = cpu_local_field;
        multilayer_result_alpha_key = cpu_polarizability;
        multilayer_result_nmax = *nmax;
        multilayer_result_ndipole = *nmax;
    }
    return ok ? 0 : 6;
}

// Let the immediately following produitpolavect call reuse the device-local
// field and polarizability that this multilayer solve already prepared.
//-----------------------------------------------------------
extern "C" int ifdda_get_multilayer_resident_polarization_inputs_cuda(
    const COMPLEX *cpu_local_key, const COMPLEX *cpu_alpha_key, const int *niso,
    const int *nmax, const int *ndipoles, CUCOMPLEX **gpu_local, CUCOMPLEX **gpu_alpha) {
    if (!cpu_local_key || !cpu_alpha_key || !niso || !nmax || !ndipoles || !gpu_local ||
        !gpu_alpha || (*niso != 1 && *niso != 3))
        return 0;
    if (cpu_local_key == multilayer_result_local_key &&
        cpu_alpha_key == multilayer_result_alpha_key && *nmax == multilayer_result_nmax &&
        *ndipoles <= multilayer_result_ndipole && multilayer_local_field &&
        multilayer_polarizability) {
        *gpu_local = multilayer_local_field;
        *gpu_alpha = multilayer_polarizability;
        multilayer_result_local_key = nullptr;
        multilayer_result_alpha_key = nullptr;
        return 1;
    }
    if (cpu_local_key == multilayer_vector_result_local_key &&
        cpu_alpha_key == multilayer_vector_result_alpha_key &&
        *nmax == multilayer_vector_result_nmax && *ndipoles <= multilayer_vector_result_ndipole &&
        multilayer_vector_local && multilayer_vector_alpha) {
        *gpu_local = multilayer_vector_local;
        *gpu_alpha = multilayer_vector_alpha;
        multilayer_vector_result_local_key = nullptr;
        multilayer_vector_result_alpha_key = nullptr;
        return 1;
    }
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_generate_multilayer_dyadic_green_cuda_(
    COMPLEX *cpu_xx, COMPLEX *cpu_xy, COMPLEX *cpu_xz, COMPLEX *cpu_yy, COMPLEX *cpu_yz,
    COMPLEX *cpu_zz, const int *nfft2d, const CU_REAL *aretecube, const CU_REAL *k0) {
    constexpr const char *context = "CUDA multilayer dyadic Green";
    if (!cpu_xx || !cpu_xy || !cpu_xz || !cpu_yy || !cpu_yz || !cpu_zz || !nfft2d || !aretecube ||
        !k0 || *nfft2d <= 0 || *nfft2d > INT_MAX / 2 || *aretecube <= CU_REAL(0) ||
        *k0 <= CU_REAL(0))
        return 1;
    const size_t side = size_t(2) * size_t(*nfft2d);
    size_t count = 0;
    if (!cuda_utils::checked_multiply(side, side, &count) || count > size_t(INT_MAX))
        return 2;
    const COMPLEX *cpu_keys[6] = {cpu_xx, cpu_xy, cpu_xz, cpu_yy, cpu_yz, cpu_zz};
    CUCOMPLEX *packed_green = get_device_multilayer_green_output(cpu_keys, count);
    if (!packed_green)
        return 3;
    CUCOMPLEX *green[6];
    for (int i = 0; i < 6; ++i)
        green[i] = packed_green + size_t(i) * count;
    if (!cuda_utils::check_cuda(generate_multilayer_dyadic_green_wrapper(
                                    green[0], green[1], green[2], green[3], green[4], green[5],
                                    *nfft2d, int(count), *aretecube, *k0),
                                context, "generate dyadic tensor"))
        return 4;
    const cufftHandle plan = get_fft_plan(int(side), int(side), 1);
    if (!plan)
        return 5;
    for (int i = 0; i < 6; ++i)
        if (!cuda_utils::check_cufft(CUFFT_EXECUTE(plan, green[i], green[i], CUFFT_INVERSE),
                                     context, "transform dyadic component"))
            return 6;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_multilayerbornvect_layers_cuda_(
    COMPLEX *cpu_ix, COMPLEX *cpu_iy, COMPLEX *cpu_iz, COMPLEX *cpu_local,
    const COMPLEX *cpu_alpha, const int *cpu_map, COMPLEX *cpu_gxx, COMPLEX *cpu_gxy,
    COMPLEX *cpu_gxz, COMPLEX *cpu_gyy, COMPLEX *cpu_gyz, COMPLEX *cpu_gzz, const int *nfft,
    const int *nx, const int *ny, const int *nz, const int *nmax, const int *niso,
    const int *reflection,
    const CU_REAL *dkx, const CU_REAL *dky, const CU_REAL *spacing, const CU_REAL *k0) {
    constexpr const char *context = "CUDA multilayer vector layers";
    multilayer_vector_result_local_key = nullptr;
    multilayer_vector_result_alpha_key = nullptr;
    multilayer_vector_result_nmax = 0;
    multilayer_vector_result_ndipole = 0;
    if (!cpu_ix || !cpu_iy || !cpu_iz || !cpu_local || !cpu_alpha || !cpu_map || !cpu_gxx ||
        !cpu_gxy || !cpu_gxz || !cpu_gyy || !cpu_gyz || !cpu_gzz || !nfft || !nx || !ny || !nz ||
        !nmax || !niso || !reflection || !dkx || !dky || !spacing || !k0 || *nfft <= 0 ||
        *nx <= 0 || *ny <= 0 || *nz < 2 || *nmax <= 0 || (*niso != 1 && *niso != 3))
        return 1;
    const size_t n = size_t(*nfft), side = 2 * n, plane = n * n;
    size_t fft_count = 0, local_count = 0, alpha_count = 0, history_count = 0;
    if (!cuda_utils::checked_multiply(side, side, &fft_count) ||
        !cuda_utils::checked_multiply(size_t(3), size_t(*nmax), &local_count) ||
        !cuda_utils::checked_multiply(size_t(*nmax), size_t(*niso), &alpha_count) ||
        !cuda_utils::checked_multiply(alpha_count, size_t(*niso), &alpha_count) ||
        fft_count > size_t(INT_MAX) || plane > size_t(INT_MAX))
        return 2;
    if (*reflection && !cuda_utils::checked_multiply(plane, size_t(*nz), &history_count))
        return 2;
    const COMPLEX *cpu_keys[6] = {cpu_gxx, cpu_gxy, cpu_gxz, cpu_gyy, cpu_gyz, cpu_gzz};
    CUCOMPLEX *packed_green = get_device_multilayer_green_output(cpu_keys, fft_count);
    if (!packed_green)
        return 3;
    CUCOMPLEX *green[6];
    for (int i = 0; i < 6; ++i)
        green[i] = packed_green + size_t(i) * fft_count;
    for (int i = 0; i < 3; ++i) {
        static const char *image_buffer_names[3] = {
            "multilayer.vector.image.x", "multilayer.vector.image.y", "multilayer.vector.image.z"};
        static const char *polarization_buffer_names[3] = {"multilayer.vector.polarization.x",
                                                           "multilayer.vector.polarization.y",
                                                           "multilayer.vector.polarization.z"};
        static const char *field_buffer_names[3] = {
            "multilayer.vector.field.x", "multilayer.vector.field.y", "multilayer.vector.field.z"};
        if (!reserve_device(multilayer_vector_images[i],
                            &multilayer_vector_image_capacity[i], plane, sizeof(CUCOMPLEX),
                            image_buffer_names[i], context) ||
            !reserve_device(multilayer_vector_polarization[i],
                            &multilayer_vector_polarization_capacity[i], fft_count,
                            sizeof(CUCOMPLEX), polarization_buffer_names[i], context) ||
            !reserve_device(multilayer_vector_field[i],
                            &multilayer_vector_field_capacity[i], fft_count, sizeof(CUCOMPLEX),
                            field_buffer_names[i], context))
            return 3;
    }
    if (!reserve_device(multilayer_vector_alpha,
                        &multilayer_vector_alpha_capacity, alpha_count, sizeof(CUCOMPLEX),
                        "multilayer.vector.polarizability", context) ||
        !reserve_device(multilayer_vector_local,
                        &multilayer_vector_local_capacity, local_count, sizeof(CUCOMPLEX),
                        "multilayer.vector.local_field", context) ||
        !reserve_device(multilayer_vector_map,
                        &multilayer_vector_map_capacity, plane, sizeof(int),
                        "multilayer.vector.plane_map", context) ||
        (*reflection && !reserve_device(multilayer_vector_history,
                                        &multilayer_vector_history_capacity, history_count * 3,
                                        sizeof(CUCOMPLEX), "multilayer.vector.history", context)))
        return 3;
    bool ok = true;
    COMPLEX *cpu_images[3] = {cpu_ix, cpu_iy, cpu_iz};
    const bool preinitialized = multilayer_vector_first_plane_ready &&
                                multilayer_vector_first_plane_image_keys[0] == cpu_ix &&
                                multilayer_vector_first_plane_image_keys[1] == cpu_iy &&
                                multilayer_vector_first_plane_image_keys[2] == cpu_iz &&
                                multilayer_vector_first_plane_local_key == cpu_local &&
                                multilayer_vector_first_plane_nfft == *nfft &&
                                multilayer_vector_first_plane_nmax == *nmax;
    multilayer_vector_first_plane_ready = false;
    for (int c = 0; c < 3 && ok && !preinitialized; ++c)
        ok = cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(multilayer_vector_images[c], cpu_images[c],
                                           plane * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::IncidentField),
            context, "upload incident image");
    ok = ok &&
         cuda_utils::check_cuda(
             cuda_data::copy_host_to_device(multilayer_vector_alpha, cpu_alpha,
                                            alpha_count * sizeof(CUCOMPLEX),
                                            cuda_utils::TransferCategory::Material),
             context, "upload polarizability") &&
         (preinitialized ||
          cuda_utils::check_cuda(
              cuda_data::copy_host_to_device(multilayer_vector_map, cpu_map, plane * sizeof(int),
                                             cuda_utils::TransferCategory::Geometry),
              context, "upload plane map")) &&
         (preinitialized || cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                                                       multilayer_vector_local, cpu_local,
                                                       local_count * sizeof(CUCOMPLEX),
                                                       cuda_utils::TransferCategory::SolverVector),
                                                   context, "upload initial local field"));
    if (!ok)
        return 4;
    cuda_data::MultilayerFftPlans fft_plans{};
    if (!cuda_data::get_device_multilayer_fft_plans(int(side), *nfft, &fft_plans))
        return 5;
    const cufftHandle layer_plan = fft_plans.layer;
    const cufftHandle image_plan = fft_plans.image;
    for (int layer = 1; layer < *nz && ok; ++layer) {
        ok = cuda_utils::check_cuda(
            multilayer_vector_prepare_wrapper(
                multilayer_vector_images[0], multilayer_vector_images[1],
                multilayer_vector_images[2], multilayer_vector_map, multilayer_vector_alpha,
                multilayer_vector_polarization[0], multilayer_vector_polarization[1],
                multilayer_vector_polarization[2], *nfft, *nx, *ny, layer, int(fft_count)),
            context, "prepare vector polarization");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cufft(
                CUFFT_EXECUTE(layer_plan, multilayer_vector_polarization[c],
                              multilayer_vector_polarization[c], CUFFT_INVERSE),
                context, "forward polarization FFT");
        ok = ok &&
             cuda_utils::check_cuda(
                 multilayer_vector_product_wrapper(
                     multilayer_vector_polarization[0], multilayer_vector_polarization[1],
                     multilayer_vector_polarization[2], green[0], green[1], green[2], green[3],
                     green[4], green[5], multilayer_vector_field[0], multilayer_vector_field[1],
                     multilayer_vector_field[2], int(fft_count), 0),
                 context, "dyadic product");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cufft(CUFFT_EXECUTE(layer_plan, multilayer_vector_field[c],
                                                       multilayer_vector_field[c], CUFFT_FORWARD),
                                         context, "inverse convolution FFT");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cufft(CUFFT_EXECUTE(image_plan, multilayer_vector_images[c],
                                                       multilayer_vector_images[c], CUFFT_INVERSE),
                                         context, "forward image FFT");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cuda(
                multilayer_scalar_propagate_wrapper(multilayer_vector_images[c], *nfft, *dkx, *dky,
                                                    *k0, *spacing),
                context, "propagate layer image");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cufft(CUFFT_EXECUTE(image_plan, multilayer_vector_images[c],
                                                       multilayer_vector_images[c], CUFFT_FORWARD),
                                         context, "inverse image FFT");
        if (ok)
            ok = cuda_utils::check_cuda(
                multilayer_vector_accumulate_wrapper(
                    multilayer_vector_images[0], multilayer_vector_images[1],
                    multilayer_vector_images[2], multilayer_vector_field[0],
                    multilayer_vector_field[1], multilayer_vector_field[2], multilayer_vector_local,
                    multilayer_vector_map, *nfft, *nx, *ny, layer, *nmax, int(plane),
                    CU_REAL(1) / CU_REAL(plane), CU_REAL(1) / CU_REAL(fft_count), 0),
                context, "accumulate forward layer");
        if (ok && *reflection)
            for (int c = 0; c < 3 && ok; ++c)
                ok = cuda_utils::check_cuda(
                    cuda_utils::tracked_memcpy(multilayer_vector_history +
                                                   (size_t(c) * size_t(*nz) + size_t(layer)) *
                                                       plane,
                                               multilayer_vector_images[c],
                                               plane * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice),
                    context, "save forward field");
    }
    if (ok && *reflection) {
        const int top = *nz - 1;
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cuda(
                cuda_utils::tracked_memcpy(multilayer_vector_images[c],
                                           multilayer_vector_history +
                                               (size_t(c) * size_t(*nz) + size_t(top)) * plane,
                                           plane * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice),
                context, "restore top field");
        ok = ok &&
             cuda_utils::check_cuda(
                 multilayer_vector_prepare_wrapper(
                     multilayer_vector_images[0], multilayer_vector_images[1],
                     multilayer_vector_images[2], multilayer_vector_map, multilayer_vector_alpha,
                     multilayer_vector_polarization[0], multilayer_vector_polarization[1],
                     multilayer_vector_polarization[2], *nfft, *nx, *ny, top, int(fft_count)),
                 context, "prepare transmitted polarization");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cufft(
                CUFFT_EXECUTE(layer_plan, multilayer_vector_polarization[c],
                              multilayer_vector_polarization[c], CUFFT_INVERSE),
                context, "forward transmitted polarization FFT");
        ok = ok &&
             cuda_utils::check_cuda(
                 multilayer_vector_product_wrapper(
                     multilayer_vector_polarization[0], multilayer_vector_polarization[1],
                     multilayer_vector_polarization[2], green[0], green[1], green[2], green[3],
                     green[4], green[5], multilayer_vector_field[0], multilayer_vector_field[1],
                     multilayer_vector_field[2], int(fft_count), 1),
                 context, "transmission dyadic product");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cufft(CUFFT_EXECUTE(layer_plan, multilayer_vector_field[c],
                                                       multilayer_vector_field[c], CUFFT_FORWARD),
                                         context, "inverse transmission convolution FFT");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cuda(
                cuda_utils::tracked_memcpy(multilayer_vector_images[c], multilayer_vector_field[c],
                                           plane * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice),
                context, "copy transmission field to image");
        if (ok)
            ok = cuda_utils::check_cuda(
                multilayer_vector_accumulate_wrapper(
                    multilayer_vector_images[0], multilayer_vector_images[1],
                    multilayer_vector_images[2], multilayer_vector_field[0],
                    multilayer_vector_field[1], multilayer_vector_field[2], multilayer_vector_local,
                    multilayer_vector_map, *nfft, *nx, *ny, *nz - 2, *nmax, int(plane), CU_REAL(0),
                    CU_REAL(1) / CU_REAL(fft_count), 1),
                context, "accumulate transmitted field");
    }
    for (int layer = *nz - 2; *reflection && layer >= 1 && ok; --layer) {
        ok = cuda_utils::check_cuda(
            multilayer_vector_prepare_reflected_wrapper(
                multilayer_vector_history + (size_t(0) * size_t(*nz) + size_t(layer)) * plane,
                multilayer_vector_history + (size_t(1) * size_t(*nz) + size_t(layer)) * plane,
                multilayer_vector_history + (size_t(2) * size_t(*nz) + size_t(layer)) * plane,
                multilayer_vector_images[0], multilayer_vector_images[1],
                multilayer_vector_images[2], multilayer_vector_map, multilayer_vector_alpha,
                multilayer_vector_polarization[0], multilayer_vector_polarization[1],
                multilayer_vector_polarization[2], *nfft, *nx, *ny, layer, int(fft_count)),
            context, "prepare forward plus reflected polarization");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cufft(
                CUFFT_EXECUTE(layer_plan, multilayer_vector_polarization[c],
                              multilayer_vector_polarization[c], CUFFT_INVERSE),
                context, "forward reflected polarization FFT");
        ok = ok &&
             cuda_utils::check_cuda(
                 multilayer_vector_product_wrapper(
                     multilayer_vector_polarization[0], multilayer_vector_polarization[1],
                     multilayer_vector_polarization[2], green[0], green[1], green[2], green[3],
                     green[4], green[5], multilayer_vector_field[0], multilayer_vector_field[1],
                     multilayer_vector_field[2], int(fft_count), 1),
                 context, "reflected dyadic product");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cufft(CUFFT_EXECUTE(layer_plan, multilayer_vector_field[c],
                                                       multilayer_vector_field[c], CUFFT_FORWARD),
                                         context, "inverse reflected convolution FFT");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cufft(CUFFT_EXECUTE(image_plan, multilayer_vector_images[c],
                                                       multilayer_vector_images[c], CUFFT_INVERSE),
                                         context, "forward reflected image FFT");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cuda(
                multilayer_scalar_propagate_wrapper(multilayer_vector_images[c], *nfft, *dkx, *dky,
                                                    *k0, *spacing),
                context, "propagate reflected image");
        for (int c = 0; c < 3 && ok; ++c)
            ok = cuda_utils::check_cufft(CUFFT_EXECUTE(image_plan, multilayer_vector_images[c],
                                                       multilayer_vector_images[c], CUFFT_FORWARD),
                                         context, "inverse reflected image FFT");
        if (ok)
            ok = cuda_utils::check_cuda(
                multilayer_vector_accumulate_wrapper(
                    multilayer_vector_images[0], multilayer_vector_images[1],
                    multilayer_vector_images[2], multilayer_vector_field[0],
                    multilayer_vector_field[1], multilayer_vector_field[2], multilayer_vector_local,
                    multilayer_vector_map, *nfft, *nx, *ny, layer - 1, *nmax, int(plane),
                    CU_REAL(1) / CU_REAL(plane), CU_REAL(1) / CU_REAL(fft_count), 1),
                context, "accumulate reflected field");
    }
    if (ok)
        ok = cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(cpu_local, multilayer_vector_local,
                                           local_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Result),
            context, "download local field");
    multilayer_vector_first_plane_ready = false;
    if (ok) {
        multilayer_vector_result_local_key = cpu_local;
        multilayer_vector_result_alpha_key = cpu_alpha;
        multilayer_vector_result_nmax = *nmax;
        multilayer_vector_result_ndipole = *nmax;
    }
    return ok ? 0 : 6;
}

