/*
 * Readable CUDA/cuFFT version of cdmlib/produitfftmatvect3.f90.
 *
 * Precision switch: set this to 1 for cuComplex/float, or 0 for
 * cuDoubleComplex/double. The Fortran ABI must use the same precision.
 *
 * The routine keeps the Fortran argument order and pass-by-address rules.
 * planb and planf are accepted for ABI compatibility; this implementation
 * creates its own cuFFT plan because FFTW plan handles are not cuFFT handles.
 *
 * The CUDA target uses this implementation for produitfftmatvect3_ and for
 * the historical produitfftmatvect3fftopt_ entry point used by inverserig.
 */
#ifndef PRODUITMATVECTFFT_SINGLE_PRECISION
#define PRODUITMATVECTFFT_SINGLE_PRECISION 0
#endif

#include <cuda_runtime.h>
#include "cuda_interfaces.h"
#include <cufft.h>

#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "cdmlib_cpp_abi.hpp"
#include "cuda_data.h"
#include "cuda_solver_interfaces.h"
#include "cuda_utils.h"

namespace {
using cuda_utils::check_cuda;
using cuda_utils::check_cufft;
using cuda_utils::checked_multiply;

} // namespace

//------------------------------------------

/* Fortran routine: produitfftmatvect3(FFTTENSORxx,...,FFB,planb,planf).
 * All scalar arguments are passed by address, and arrays are contiguous. */
//-----------------------------------------------------------
extern "C" bool produitfftmatvect_cuda(
    GreenTensorDeviceView device_green, CUCOMPLEX *device_vectors, const CU_INT *device_tabdip,
    const CUCOMPLEX *device_ffx, CUCOMPLEX *device_ffb, int nx, int ny, int nz, int nx2, int ny2,
    int nz2, int nxy2, int ntotal, int nmax, size_t tensor_count) {
    size_t physical_xy = 0;
    size_t physical_count = 0;
    size_t padded_xy = 0;
    size_t padded_count = 0;
    size_t vector_count = 0;
    if (!device_green || !device_vectors || !device_tabdip || !device_ffx || !device_ffb ||
        nx <= 0 || ny <= 0 || nz <= 0 || nx2 < nx || ny2 < ny ||
        nz2 < nz || nxy2 <= 0 || ntotal <= 0 || nmax <= 0 || tensor_count == 0 ||
        !checked_multiply(size_t(nx), size_t(ny), &physical_xy) ||
        !checked_multiply(physical_xy, size_t(nz), &physical_count) ||
        !checked_multiply(size_t(nx2), size_t(ny2), &padded_xy) ||
        !checked_multiply(padded_xy, size_t(nz2), &padded_count) ||
        !checked_multiply(padded_count, size_t(3), &vector_count) ||
        physical_count > size_t(INT_MAX) || padded_count != size_t(ntotal) ||
        padded_xy != size_t(nxy2) || vector_count > size_t(INT_MAX)) {
        std::fprintf(stderr, "produitmatvectfft CUDA: invalid GPU buffers or dimensions\n");
        return false;
    }

    const cufftHandle fft_plan = get_fft_plan(nx2, ny2, nz2);
    if (!fft_plan) {
        std::fprintf(stderr, "produitmatvectfft CUDA: cannot get cached cuFFT plan\n");
        return false;
    }

    if (!check_cuda(zero_vectors_wrapper(device_vectors, int(vector_count)),
                    "produitmatvectfft CUDA", "zero FFT vectors"))
        return false;

    if (!check_cuda(gather_incident_wrapper(device_ffx, device_tabdip, device_vectors, nx, ny, nz,
                                            nx2, nxy2, ntotal, nmax),
                    "produitmatvectfft CUDA", "gather incident field"))
        return false;

    for (int component = 0; component < 3; ++component) {
        CUCOMPLEX *device_vector = device_vectors + size_t(component) * padded_count;
        if (!check_cufft(CUFFT_EXECUTE(fft_plan, device_vector, device_vector, CUFFT_INVERSE),
                         "produitmatvectfft CUDA", "inverse FFT") ||
            cudaGetLastError() != cudaSuccess)
            return false;
    }

    if (!check_cuda(multiply_green_tensor_wrapper(device_green, device_vectors, nx, ny, nz, nx2, ny2,
                                                  nz2, tensor_count),
                    "produitmatvectfft CUDA", "multiply Green tensor"))
        return false;

    for (int component = 0; component < 3; ++component) {
        CUCOMPLEX *device_vector = device_vectors + size_t(component) * padded_count;
        if (!check_cufft(CUFFT_EXECUTE(fft_plan, device_vector, device_vector, CUFFT_FORWARD),
                         "produitmatvectfft CUDA", "forward FFT") ||
            cudaGetLastError() != cudaSuccess)
            return false;
    }

    if (!check_cuda(update_output_wrapper(device_vectors, device_tabdip, device_ffb, nx, ny, nz, nx2,
                                          nxy2, nmax, ntotal),
                    "produitmatvectfft CUDA", "update scattered field") ||
        !check_cuda(cudaDeviceSynchronize(), "produitmatvectfft CUDA", "finish CUDA calculation"))
        return false;

    return true;
}

//------------------------------------------

static bool produitfftmatvect3_impl(
    const COMPLEX *tensor_xx, const COMPLEX *tensor_xy, const COMPLEX *tensor_xz,
    const COMPLEX *tensor_yy, const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
    const int *cpu_tabdip, const int *ntotalm, const int *ntotal, const int *nmax,
    const int *ndipole, const int *nxm, const int *nym, const int *nzm, const int *nx,
    const int *ny, const int *nz, const int *nx2, const int *ny2, const int *nxy2,
    const int *nz2, const CUCOMPLEX *device_ffx, CUCOMPLEX *device_ffb,
    const long long *planb, const long long *planf, GreenTensorDeviceView resident_green = {},
    const CU_INT *device_tabdip = nullptr, CUCOMPLEX *device_vectx = nullptr,
    CUCOMPLEX *device_vecty = nullptr, CUCOMPLEX *device_vectz = nullptr) {
    (void)ntotalm;
    (void)ndipole;
    (void)planb;
    (void)planf;

    if ((!resident_green && (!tensor_xx || !tensor_xy || !tensor_xz || !tensor_yy ||
                             !tensor_yz || !tensor_zz)) ||
        (!device_tabdip && !cpu_tabdip) || !nmax || !nxm || !nym || !nzm || !nx || !ny ||
        !nz || !nx2 || !ny2 || !nxy2 || !nz2 || !ntotal || !device_ffx || !device_ffb ||
        *nxm <= 0 || *nym <= 0 || *nzm <= 0 || *nx <= 0 || *ny <= 0 || *nz <= 0 ||
        *nx >= INT_MAX || *ny >= INT_MAX || *nz >= INT_MAX || *nxm < *nx || *nym < *ny ||
        *nzm < *nz || *nx2 < *nx || *ny2 < *ny || *nz2 < *nz || *nmax <= 0 || *ntotal <= 0 ||
        *nxy2 <= 0) {
        std::fprintf(stderr, "produitmatvectfft CUDA: invalid Fortran arguments\n");
        return false;
    }

    size_t physical_xy = 0;
    size_t physical_count = 0;
    size_t padded_xy = 0;
    size_t padded_count = 0;
    size_t tensor_plane = 0;
    size_t tensor_count = 0;
    size_t vector_count = 0;
    size_t output_count = 0;
    if (!checked_multiply(size_t(*nx), size_t(*ny), &physical_xy) ||
        !checked_multiply(physical_xy, size_t(*nz), &physical_count) ||
        !checked_multiply(size_t(*nx2), size_t(*ny2), &padded_xy) ||
        !checked_multiply(padded_xy, size_t(*nz2), &padded_count) ||
        !checked_multiply(size_t(*nxm) + 1, size_t(*nym) + 1, &tensor_plane) ||
        !checked_multiply(tensor_plane, size_t(*nzm) + 1, &tensor_count) ||
        !checked_multiply(padded_count, size_t(3), &vector_count) ||
        !checked_multiply(size_t(*nmax), size_t(3), &output_count) ||
        physical_count > size_t(INT_MAX) || padded_count != size_t(*ntotal) ||
        padded_xy != size_t(*nxy2) || tensor_count == 0 || vector_count > size_t(INT_MAX) ||
        output_count > size_t(INT_MAX)) {
        std::fprintf(stderr, "produitmatvectfft CUDA: dimensions overflow or mismatch\n");
        return false;
    }

    if ((device_vectx || device_vecty || device_vectz) &&
        (!device_vectx || !device_vecty || !device_vectz)) {
        std::fprintf(stderr, "produitmatvectfft CUDA: incomplete device vector views\n");
        return false;
    }

    if (!device_vectx) {
        device_vectx = get_device_vectors(vector_count);
        if (!device_vectx)
            return false;
        device_vecty = device_vectx + padded_count;
        device_vectz = device_vecty + padded_count;
    }

    const COMPLEX *cpu_tensors[6] = {tensor_xx, tensor_xy, tensor_xz,
                                     tensor_yy, tensor_yz, tensor_zz};
    const GreenTensorDeviceView device_green = resident_green
        ? resident_green : green_view(find_ready_device_green(cpu_tensors, tensor_count));
    const CU_INT *gpu_tabdip = device_tabdip ? device_tabdip : get_device_tabdip(cpu_tabdip, physical_count);
    if (!device_green || !gpu_tabdip) {
        std::fprintf(stderr, "produitmatvectfft CUDA: cannot get Green/table buffers on GPU\n");
        return false;
    }

    return produitfftmatvect_cuda(device_green, device_vectx, gpu_tabdip, device_ffx, device_ffb,
                                  *nx, *ny, *nz, *nx2, *ny2, *nz2,
                                  *nxy2, *ntotal, *nmax, tensor_count);
}

//------------------------------------------

extern "C" void produitfftmatvect3_(const COMPLEX *tensor_xx, const COMPLEX *tensor_xy,
                                    const COMPLEX *tensor_xz, const COMPLEX *tensor_yy,
                                    const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
                                    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
                                    const int *tabdip, const int *ntotalm, const int *ntotal,
                                    const int *nmax, const int *ndipole, const int *nxm,
                                    const int *nym, const int *nzm, const int *nx, const int *ny,
                                    const int *nz, const int *nx2, const int *ny2, const int *nxy2,
                                    const int *nz2, const COMPLEX *ffx, COMPLEX *ffb,
                                    const long long *planb, const long long *planf) {
    size_t output_count = 0;
    CUCOMPLEX *device_ffx = nullptr, *device_ffb = nullptr;
    if (!nmax || *nmax <= 0 || !checked_multiply(size_t(*nmax), size_t(3), &output_count) ||
        !get_ffx_ffb(ffx, ffb, output_count, &device_ffx, &device_ffb)) {
        std::fprintf(stderr, "produitmatvectfft CUDA: cannot get host FFX/FFB buffers\n");
        return;
    }
    (void)vectx;
    (void)vecty;
    (void)vectz;
    if (produitfftmatvect3_impl(tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz, tensor_zz,
                                tabdip, ntotalm, ntotal, nmax, ndipole, nxm,
                                nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, device_ffx, device_ffb,
                                planb, planf) &&
        !copy_ffb_to_cpu(ffb, device_ffb, output_count))
        std::fprintf(stderr, "produitmatvectfft CUDA: cannot publish FFB to host\n");
}

// Internal inverse-driver entry: ffx and ffb are device pointers, so the
// regular CPU staging buffers and D2H publication are bypassed.
//-----------------------------------------------------------
extern "C" void produitfftmatvect3device_(
    const CUCOMPLEX *tensor_xx, const CUCOMPLEX *tensor_xy, const CUCOMPLEX *tensor_xz,
    const CUCOMPLEX *tensor_yy, const CUCOMPLEX *tensor_yz, const CUCOMPLEX *tensor_zz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz, const CU_INT *tabdip, const int *ntotalm,
    const int *ntotal, const int *nmax, const int *ndipole, const int *nxm, const int *nym,
    const int *nzm, const int *nx, const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nxy2, const int *nz2, const CUCOMPLEX *device_ffx, CUCOMPLEX *device_ffb,
    const long long *planb, const long long *planf) {
    (void)ntotalm;
    (void)ndipole;
    (void)planb;
    (void)planf;
    if (!nxm || !nym || !nzm || !nx || !ny || !nz || !nx2 || !ny2 || !nz2 || !nxy2 ||
        !ntotal || !nmax || !tensor_xx || !tabdip || !device_ffx || !device_ffb) {
        std::fprintf(stderr, "produitmatvectfft CUDA: invalid device entry arguments\n");
        return;
    }
    size_t tensor_plane = 0;
    size_t tensor_count = 0;
    if (!checked_multiply(size_t(*nxm) + 1, size_t(*nym) + 1, &tensor_plane) ||
        !checked_multiply(tensor_plane, size_t(*nzm) + 1, &tensor_count)) {
        std::fprintf(stderr, "produitmatvectfft CUDA: tensor dimensions overflow\n");
        return;
    }
    size_t vector_count = 0;
    if (!checked_multiply(size_t(*ntotal), size_t(3), &vector_count) ||
        vector_count > size_t(INT_MAX)) {
        std::fprintf(stderr, "produitmatvectfft CUDA: FFT scratch dimensions overflow\n");
        return;
    }
    // The legacy callers may pass three independent host work arrays. cuFFT
    // needs one contiguous device allocation, so obtain it at this adapter.
    CUCOMPLEX *device_vectors = get_device_vectors(vector_count);
    if (!device_vectors) {
        std::fprintf(stderr, "produitmatvectfft CUDA: cannot allocate GPU FFT scratch\n");
        return;
    }
    (void)vectx;
    (void)vecty;
    (void)vectz;
    const GreenTensorDeviceView green{{tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz, tensor_zz}};
    (void)produitfftmatvect_cuda(green, device_vectors, tabdip, device_ffx, device_ffb, *nx,
                                *ny, *nz, *nx2, *ny2, *nz2, *nxy2, *ntotal, *nmax,
                                tensor_count);
}

extern "C" void produitfftmatvect3host_(
    const CUCOMPLEX *tensor_xx, const CUCOMPLEX *tensor_xy, const CUCOMPLEX *tensor_xz,
    const CUCOMPLEX *tensor_yy, const CUCOMPLEX *tensor_yz, const CUCOMPLEX *tensor_zz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz, const int *cpu_tabdip,
    const int *ntotalm, const int *ntotal, const int *nmax, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm, const int *nx, const int *ny,
    const int *nz, const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const CUCOMPLEX *device_ffx, CUCOMPLEX *device_ffb,
    const long long *planb, const long long *planf) {
    if (!cpu_tabdip || !nx || !ny || !nz || *nx <= 0 || *ny <= 0 || *nz <= 0)
        return;
    const size_t map_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
    const CU_INT *gpu_tabdip = get_device_tabdip(cpu_tabdip, map_count);
    if (!gpu_tabdip)
        return;
    produitfftmatvect3device_(
        tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz, tensor_zz,
        vectx, vecty, vectz, gpu_tabdip, ntotalm, ntotal, nmax, ndipole,
        nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,
        device_ffx, device_ffb, planb, planf);
}

// The non-optimized Fortran inverse driver calls this historical wrapper
// name. Keep its Fortran argument convention while routing the operation
// through the same persistent CUDA/cuFFT implementation.
//-----------------------------------------------------------
extern "C" void produitfftmatvect3fftopt_(
    const COMPLEX *tensor_xx, const COMPLEX *tensor_xy, const COMPLEX *tensor_xz,
    const COMPLEX *tensor_yy, const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz, const int *tabdip, const int *ntotalm,
    const int *ntotal, const int *nmax, const int *ndipole, const int *nxm, const int *nym,
    const int *nzm, const int *nx, const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nxy2, const int *nz2, const COMPLEX *ffx, COMPLEX *ffb, const long long *plan_xf,
    const long long *plan_xb, const long long *plan_yf, const long long *plan_yb,
    const long long *plan_zf, const long long *plan_zb) {
    static bool path_reported = false;
    if (!path_reported) {
        std::fprintf(stderr,
                     "[produitfftmatvect3fftopt] CUDA/cuFFT backend active (%s precision)\n",
#if PRODUITMATVECTFFT_SINGLE_PRECISION
                     "single"
#else
                     "double"
#endif
        );
        path_reported = true;
    }
    produitfftmatvect3_(tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz, tensor_zz, vectx,
                        vecty, vectz, tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny,
                        nz, nx2, ny2, nxy2, nz2, ffx, ffb, plan_xb, plan_xf);
    (void)plan_yf;
    (void)plan_yb;
    (void)plan_zf;
    (void)plan_zb;
}

#undef CUFFT_TRANSFORM
#undef CUFFT_EXECUTE
#undef CUFFT_ZERO
#undef CUFFT_REAL
#undef CUFFT_ADD
#undef CUFFT_SUB
#undef CUFFT_MUL
