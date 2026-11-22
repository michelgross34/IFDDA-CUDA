/* CUDA/cuFFT replacement for cdmlib/produitfftmatvectscalaire.f90.
 * Keep the original Fortran argument order and pass-by-address ABI. */
#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_solver_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>

//-----------------------------------------------------------
extern "C" bool produitfftmatvectscalaire_cuda(
    const CUCOMPLEX *device_green, CUCOMPLEX *device_work, const CU_INT *device_tabdip,
    const CUCOMPLEX *device_ffx, CUCOMPLEX *device_ffb, int nx, int ny, int nz, int nx2,
    int ny2, int nz2, int nxy2, int ntotal, int nmax) {
    size_t physical_xy = 0;
    size_t physical_count = 0;
    size_t padded_xy = 0;
    size_t padded_count = 0;
    if (!device_green || !device_work || !device_tabdip || !device_ffx || !device_ffb || nx <= 0 ||
        ny <= 0 || nz <= 0 || nx2 < nx || ny2 < ny || nz2 < nz || nxy2 <= 0 || ntotal <= 0 ||
        nmax <= 0 || !cuda_utils::checked_multiply(size_t(nx), size_t(ny), &physical_xy) ||
        !cuda_utils::checked_multiply(physical_xy, size_t(nz), &physical_count) ||
        !cuda_utils::checked_multiply(size_t(nx2), size_t(ny2), &padded_xy) ||
        !cuda_utils::checked_multiply(padded_xy, size_t(nz2), &padded_count) ||
        physical_count > size_t(INT_MAX) || padded_count != size_t(ntotal) ||
        padded_xy != size_t(nxy2)) {
        std::fprintf(stderr, "produitfftmatvectscalaire CUDA: invalid GPU buffers or dimensions\n");
        return false;
    }

    const cufftHandle device_plan = get_fft_plan(nx2, ny2, nz2);
    if (!device_plan ||
        !cuda_utils::check_cuda(cudaMemset(device_work, 0, padded_count * sizeof(CUCOMPLEX)),
                                "produitfftmatvectscalaire CUDA", "clear FFT workspace") ||
        !cuda_utils::check_cuda(gather_scalar_product_wrapper(device_ffx, device_tabdip, device_work,
                                                              nx, ny, nz, nx2, nxy2, nmax),
                                "produitfftmatvectscalaire CUDA", "gather input") ||
        !cuda_utils::check_cufft(CUFFT_EXECUTE(device_plan, device_work, device_work, CUFFT_INVERSE),
                                 "produitfftmatvectscalaire CUDA", "inverse cuFFT") ||
        !cuda_utils::check_cuda(multiply_scalar_green_wrapper(device_green, device_work, ntotal),
                                "produitfftmatvectscalaire CUDA", "multiply Green tensor") ||
        !cuda_utils::check_cufft(CUFFT_EXECUTE(device_plan, device_work, device_work, CUFFT_FORWARD),
                                 "produitfftmatvectscalaire CUDA", "forward cuFFT") ||
        !cuda_utils::check_cuda(scatter_scalar_product_wrapper(device_work, device_tabdip, device_ffb,
                                                               nx, ny, nz, nx2, nxy2, nmax, ntotal),
                                "produitfftmatvectscalaire CUDA", "scatter output"))
        return false;

    return true;
}

//-----------------------------------------------------------
extern "C" void produitfftmatvectscalaire_(const COMPLEX *green, COMPLEX *work,
                                           const int *tabdip, const int *ntotalm, const int *ntotal,
                                           const int *nmax, const int *nxm, const int *nym,
                                           const int *nzm, const int *nx, const int *ny,
                                           const int *nz, const int *nx2, const int *ny2,
                                           const int *nxy2, const int *nz2, const COMPLEX *ffx,
                                           COMPLEX *ffb, const long long *planf,
                                           const long long *planb) {
    (void)work;
    (void)planf;
    (void)planb;
    const auto fail = [](const char *message) {
        std::fprintf(stderr, "produitfftmatvectscalaire CUDA: %s\n", message);
        std::fflush(stderr);
        std::abort();
    };

    if (!green || !tabdip || !ntotalm || !ntotal || !nmax || !nxm || !nym || !nzm || !nx || !ny ||
        !nz || !nx2 || !ny2 || !nxy2 || !nz2 || !ffx || !ffb || *ntotalm < *ntotal ||
        *ntotal <= 0 || *nmax <= 0 || *nx <= 0 || *ny <= 0 || *nz <= 0 || *nx2 < *nx ||
        *ny2 < *ny || *nz2 < *nz || *nxy2 != *nx2 * *ny2)
        fail("invalid Fortran arguments or dimensions");

    const size_t physical_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
    const size_t padded_count = size_t(*nx2) * size_t(*ny2) * size_t(*nz2);
    const size_t dipole_map_count = size_t(*nxm) * size_t(*nym) * size_t(*nzm);
    if (physical_count > size_t(std::numeric_limits<int>::max()) ||
        padded_count != size_t(*ntotal) || dipole_map_count < physical_count)
        fail("inconsistent dimensions");

    CUCOMPLEX *gpu_work = get_device_auxiliary_vector(padded_count);
    CUCOMPLEX *gpu_green = get_device_scalar_green(green, size_t(*ntotalm));
    CU_INT *gpu_tabdip = get_device_tabdip(tabdip, physical_count);
    CUCOMPLEX *gpu_ffx = nullptr, *gpu_ffb = nullptr;
    if (!gpu_work || !gpu_green || !gpu_tabdip ||
        !get_ffx_ffb(ffx, ffb, size_t(*nmax), &gpu_ffx, &gpu_ffb))
        fail("could not prepare persistent GPU arrays");

    if (!produitfftmatvectscalaire_cuda(gpu_green, gpu_work, gpu_tabdip, gpu_ffx, gpu_ffb, *nx,
                                        *ny, *nz, *nx2, *ny2, *nz2, *nxy2, *ntotal, *nmax) ||
        !copy_ffb_to_cpu(ffb, gpu_ffb, size_t(*nmax)))
        fail("GPU scalar matrix-vector product failed");

    static bool reported = false;
    if (!reported) {
        std::fprintf(stderr,
                     "[produitfftmatvectscalaire] CUDA/cuFFT backend active (%s precision)\n",
                     sizeof(CU_REAL) == sizeof(float) ? "single" : "double");
        reported = true;
    }
}

//-----------------------------------------------------------
extern "C" void produitfftmatvectoptscalaire_(
    const COMPLEX *green, COMPLEX *work, const int *ntotalm, const int *ntotal,
    const int *nmax, const int *nxm, const int *nym, const int *nzm, const int *nx,
    const int *ny, const int *nz, const int *nx2, const int *ny2, const int *nxy2,
    const int *nz2, const COMPLEX *ffx, COMPLEX *ffb, const long long *planf,
    const long long *planb) {
    (void)work;
    (void)planf;
    (void)planb;
    const auto fail = [](const char *message) {
        std::fprintf(stderr, "produitfftmatvectoptscalaire CUDA: %s\n", message);
        std::fflush(stderr);
        std::abort();
    };

    if (!green || !ntotalm || !ntotal || !nmax || !nxm || !nym || !nzm || !nx || !ny ||
        !nz || !nx2 || !ny2 || !nxy2 || !nz2 || !ffx || !ffb || *ntotalm < *ntotal ||
        *ntotal <= 0 || *nmax <= 0 || *nx <= 0 || *ny <= 0 || *nz <= 0 || *nx2 < *nx ||
        *ny2 < *ny || *nz2 < *nz || *nx2 > INT_MAX / *ny2 ||
        *nx2 * *ny2 != *nxy2 || *nx > INT_MAX / *ny || *nx * *ny > INT_MAX / *nz)
        fail("invalid Fortran arguments or dimensions");

    const size_t physical_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
    const size_t padded_count = size_t(*nx2) * size_t(*ny2) * size_t(*nz2);
    size_t output_count = 0;
    if (physical_count > size_t(INT_MAX) || physical_count > size_t(*nmax) ||
        padded_count != size_t(*ntotal) ||
        !cuda_utils::checked_multiply(size_t(*nmax), size_t(3), &output_count))
        fail("inconsistent Fortran dimensions");

    if (!initialize_cuda_device())
        fail("CUDA device initialization failed");
    CUCOMPLEX *gpu_work = get_device_auxiliary_vector(padded_count);
    CUCOMPLEX *gpu_green = get_device_scalar_green(green, size_t(*ntotalm));
    CUCOMPLEX *gpu_ffx = nullptr;
    CUCOMPLEX *gpu_ffb = nullptr;
    if (!gpu_work || !gpu_green ||
        !get_ffx_ffb(ffx, ffb, output_count, &gpu_ffx, &gpu_ffb))
        fail("could not acquire persistent GPU buffers");

    const cufftHandle fft_plan = get_fft_plan(*nx2, *ny2, *nz2);
    if (!fft_plan ||
        !cuda_utils::check_cuda(cudaMemset(gpu_work, 0, padded_count * sizeof(CUCOMPLEX)),
                                "produitfftmatvectoptscalaire CUDA", "clear FFT workspace") ||
        !cuda_utils::check_cuda(
            gather_scalar_product_dense_wrapper(gpu_ffx, gpu_work, *nx, *ny, *nz, *nx2,
                                                *nxy2),
            "produitfftmatvectoptscalaire CUDA", "gather input") ||
        !cuda_utils::check_cufft(CUFFT_EXECUTE(fft_plan, gpu_work, gpu_work, CUFFT_INVERSE),
                                 "produitfftmatvectoptscalaire CUDA", "inverse cuFFT") ||
        !cuda_utils::check_cuda(multiply_scalar_green_wrapper(gpu_green, gpu_work, *ntotal),
                                "produitfftmatvectoptscalaire CUDA", "multiply Green tensor") ||
        !cuda_utils::check_cufft(CUFFT_EXECUTE(fft_plan, gpu_work, gpu_work, CUFFT_FORWARD),
                                 "produitfftmatvectoptscalaire CUDA", "forward cuFFT") ||
        !cuda_utils::check_cuda(
            scatter_scalar_product_dense_wrapper(gpu_work, gpu_ffb, *nx, *ny, *nz, *nx2,
                                                 *nxy2, *ntotal),
            "produitfftmatvectoptscalaire CUDA", "update scattered field") ||
        !copy_ffb_to_cpu(ffb, gpu_ffb, output_count))
        fail("GPU scalar optimized matrix-vector product failed");

    static bool reported = false;
    if (!reported) {
        std::fprintf(stderr,
                     "[produitfftmatvectoptscalaire] CUDA/cuFFT backend active (%s precision)\n",
                     sizeof(CU_REAL) == sizeof(float) ? "single" : "double");
        reported = true;
    }
}
