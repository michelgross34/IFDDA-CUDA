/*
 * CUDA/cuFFT version of cdmlib/produitfftmatvectopt.f90.
 *
 * This keeps the Fortran argument order and pass-by-address convention.
 * FFTW plan handles are accepted for ABI compatibility; the CUDA version
 * keeps and reuses its own cuFFT plan through cuda_data.cpp.
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
#include <limits>

#include "cdmlib_cpp_precision.hpp"
#include "cuda_data.h"
#include "cuda_solver_interfaces.h"
#include "cuda_utils.h"

namespace {

using cuda_utils::check_cuda;
using cuda_utils::check_cufft;
using cuda_utils::checked_multiply;

} // namespace

//------------------------------------------

extern "C" bool produitfftmatvectopt_cuda(GreenTensorDeviceView device_green,
                                          CUCOMPLEX* device_vectors,
                                          const CUCOMPLEX* device_ffx,
                                          CUCOMPLEX* device_ffb,
                                          int nx,
                                          int ny,
                                          int nz,
                                          int nx2,
                                          int ny2,
                                          int nz2,
                                          int nxy2,
                                          int ntotal,
                                          int nmax,
                                          size_t tensor_count) {
    size_t physical_xy = 0;
    size_t physical_count = 0;
    size_t padded_xy = 0;
    size_t padded_count = 0;
    size_t vector_count = 0;
    if (!device_green || !device_vectors || !device_ffx ||
        !device_ffb || nx <= 0 || ny <= 0 || nz <= 0 || nx2 < nx || ny2 < ny || nz2 < nz ||
        nxy2 <= 0 || ntotal <= 0 || nmax <= 0 || tensor_count == 0 ||
        !checked_multiply(size_t(nx), size_t(ny), &physical_xy) ||
        !checked_multiply(physical_xy, size_t(nz), &physical_count) ||
        !checked_multiply(size_t(nx2), size_t(ny2), &padded_xy) ||
        !checked_multiply(padded_xy, size_t(nz2), &padded_count) ||
        !checked_multiply(padded_count, size_t(3), &vector_count) ||
        physical_count > size_t(INT_MAX) || padded_count != size_t(ntotal) ||
        padded_xy != size_t(nxy2) || vector_count > size_t(INT_MAX)) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: invalid GPU buffers or dimensions\n");
        return false;
    }

    const cufftHandle fft_plan = get_fft_plan(nx2, ny2, nz2);
    if (!fft_plan) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: cannot get cached cuFFT plan\n");
        return false;
    }

    if (!check_cuda(zero_vectors_wrapper(device_vectors, int(vector_count)),
                    "produitfftmatvectopt CUDA",
                    "zero FFT vectors"))
        return false;

    if (!check_cuda(gather_incident_regular_wrapper(
                        device_ffx, device_vectors, nx, ny, nz, nx2, nxy2, ntotal, nmax),
                    "produitfftmatvectopt CUDA",
                    "gather incident field"))
        return false;

    for (int component = 0; component < 3; ++component) {
        CUCOMPLEX* device_vector = device_vectors + size_t(component) * padded_count;
        if (!check_cufft(CUFFT_EXECUTE(fft_plan, device_vector, device_vector, CUFFT_INVERSE),
                         "produitfftmatvectopt CUDA",
                         "inverse FFT") ||
            cudaGetLastError() != cudaSuccess)
            return false;
    }

    if (!check_cuda(multiply_green_tensor_opt_wrapper(
                        device_green, device_vectors, nx, ny, nz, nx2, ny2, nz2, tensor_count),
                    "produitfftmatvectopt CUDA",
                    "multiply Green tensor"))
        return false;

    for (int component = 0; component < 3; ++component) {
        CUCOMPLEX* device_vector = device_vectors + size_t(component) * padded_count;
        if (!check_cufft(CUFFT_EXECUTE(fft_plan, device_vector, device_vector, CUFFT_FORWARD),
                         "produitfftmatvectopt CUDA",
                         "forward FFT") ||
            cudaGetLastError() != cudaSuccess)
            return false;
    }

    if (!check_cuda(update_output_regular_wrapper(
                        device_vectors, device_ffb, nx, ny, nz, nx2, nxy2, nmax, ntotal),
                    "produitfftmatvectopt CUDA",
                    "update scattered field") ||
        !check_cuda(
            cudaDeviceSynchronize(), "produitfftmatvectopt CUDA", "finish CUDA calculation"))
        return false;

    return true;
}

//------------------------------------------

extern "C" void produitfftmatvectopt_(const COMPLEX* tensor_xx,
                                      const COMPLEX* tensor_xy,
                                      const COMPLEX* tensor_xz,
                                      const COMPLEX* tensor_yy,
                                      const COMPLEX* tensor_yz,
                                      const COMPLEX* tensor_zz,
                                      COMPLEX* vectx,
                                      COMPLEX* vecty,
                                      COMPLEX* vectz,
                                      const int* ntotalm,
                                      const int* ntotal,
                                      const int* nmax,
                                      const int* nxm,
                                      const int* nym,
                                      const int* nzm,
                                      const int* nx,
                                      const int* ny,
                                      const int* nz,
                                      const int* nx2,
                                      const int* ny2,
                                      const int* nxy2,
                                      const int* nz2,
                                      const COMPLEX* ffx,
                                      COMPLEX* ffb,
                                      const long long* planf,
                                      const long long* planb) {
    (void)vectx;
    (void)vecty;
    (void)vectz;
    (void)ntotalm;
    (void)planf;
    (void)planb;
    if (!tensor_xx || !tensor_xy || !tensor_xz || !tensor_yy || !tensor_yz || !tensor_zz ||
        !ntotal || !nmax || !nxm || !nym || !nzm || !nx || !ny || !nz || !nx2 || !ny2 ||
        !nxy2 || !nz2 || !ffx || !ffb || *nxm < 0 || *nym < 0 || *nzm < 0 || *nx <= 0 ||
        *ny <= 0 || *nz <= 0 || *nx2 < *nx || *ny2 < *ny || *nz2 < *nz || *nmax <= 0 ||
        *ntotal <= 0 || *nx >= INT_MAX || *ny >= INT_MAX || *nz >= INT_MAX) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: invalid Fortran arguments\n");
        return;
    }

    size_t padded_xy = 0;
    size_t padded_count = 0;
    size_t tensor_plane = 0;
    size_t tensor_count = 0;
    size_t vector_count = 0;
    size_t output_count = 0;
    if (!checked_multiply(size_t(*nx2), size_t(*ny2), &padded_xy) ||
        !checked_multiply(padded_xy, size_t(*nz2), &padded_count) ||
        !checked_multiply(size_t(*nxm) + 1, size_t(*nym) + 1, &tensor_plane) ||
        !checked_multiply(tensor_plane, size_t(*nzm) + 1, &tensor_count) ||
        !checked_multiply(padded_count, size_t(3), &vector_count) ||
        !checked_multiply(size_t(*nmax), size_t(3), &output_count) ||
        padded_count != size_t(*ntotal) || padded_xy != size_t(*nxy2) || tensor_count == 0 ||
        vector_count > size_t(INT_MAX) || output_count > size_t(INT_MAX)) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: dimensions overflow or mismatch\n");
        return;
    }

    const COMPLEX* host_tensors[6] = {
        tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz, tensor_zz};
    // Products never allocate or upload Green. The inverse driver or GPU
    // generator has already established six resident device components.
    const GreenTensorDevicePointers device_green =
        find_ready_device_green(host_tensors, tensor_count);
    CUCOMPLEX* device_vectors = get_device_vectors(vector_count);
    CUCOMPLEX *device_ffx = nullptr, *device_ffb = nullptr;
    if (!device_green || !device_vectors ||
        !get_ffx_ffb(ffx, ffb, output_count, &device_ffx, &device_ffb)) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: cannot prepare GPU buffers\n");
        return;
    }
    if (produitfftmatvectopt_cuda(green_view(device_green), device_vectors, device_ffx, device_ffb, *nx, *ny,
                                  *nz, *nx2, *ny2, *nz2, *nxy2, *ntotal, *nmax, tensor_count) &&
        !copy_ffb_to_cpu(ffb, device_ffb, output_count))
        std::fprintf(stderr, "produitfftmatvectopt CUDA: cannot publish FFB to host\n");
}

//-----------------------------------------------------------
extern "C" void produitfftmatvectoptdevice_(const CUCOMPLEX* tensor_xx,
                                            const CUCOMPLEX* tensor_xy,
                                            const CUCOMPLEX* tensor_xz,
                                            const CUCOMPLEX* tensor_yy,
                                            const CUCOMPLEX* tensor_yz,
                                            const CUCOMPLEX* tensor_zz,
                                            CUCOMPLEX* device_vectx,
                                            CUCOMPLEX* device_vecty,
                                            CUCOMPLEX* device_vectz,
                                            const int* ntotalm,
                                            const int* ntotal,
                                            const int* nmax,
                                            const int* nxm,
                                            const int* nym,
                                            const int* nzm,
                                            const int* nx,
                                            const int* ny,
                                            const int* nz,
                                            const int* nx2,
                                            const int* ny2,
                                            const int* nxy2,
                                            const int* nz2,
                                            const CUCOMPLEX* device_ffx,
                                            CUCOMPLEX* device_ffb,
                                            const long long* planf,
                                            const long long* planb) {
    (void)device_vectx;
    (void)device_vecty;
    (void)device_vectz;
    (void)ntotalm;
    (void)planf;
    (void)planb;
    if (!tensor_xx || !tensor_xy || !tensor_xz || !tensor_yy || !tensor_yz || !tensor_zz ||
        !ntotal || !nmax || !nxm || !nym || !nzm || !nx || !ny || !nz || !nx2 ||
        !ny2 || !nxy2 || !nz2 || !device_ffx || !device_ffb || *ntotal <= 0 || *nmax <= 0 ||
        *nxm < 0 || *nym < 0 || *nzm < 0) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: invalid device arguments\n");
        return;
    }
    size_t vector_count = 0;
    size_t tensor_plane = 0;
    size_t tensor_count = 0;
    if (!checked_multiply(size_t(*ntotal), size_t(3), &vector_count) ||
        !checked_multiply(size_t(*nxm) + 1, size_t(*nym) + 1, &tensor_plane) ||
        !checked_multiply(tensor_plane, size_t(*nzm) + 1, &tensor_count) ||
        vector_count > size_t(INT_MAX) || tensor_count == 0) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: device dimensions overflow\n");
        return;
    }
    CUCOMPLEX* device_vectors = get_device_vectors(vector_count);
    if (!device_vectors) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: cannot get GPU scratch\n");
        return;
    }
    const GreenTensorDeviceView green{{tensor_xx, tensor_xy, tensor_xz, tensor_yy, tensor_yz, tensor_zz}};
    (void)produitfftmatvectopt_cuda(green, device_vectors, device_ffx, device_ffb, *nx, *ny,
                                    *nz, *nx2, *ny2, *nz2, *nxy2, *ntotal, *nmax, tensor_count);
}

//-----------------------------------------------------------
// Wide-field and other fully resident callers already own a generated Green
// tensor bundle and both field vectors on the device. Avoid staging any of
// those arrays through the CPU-facing Fortran ABI.
extern "C" void produitfftmatvectoptresident_(GreenTensorDeviceView device_green,
                                              CUCOMPLEX* device_ffx,
                                              CUCOMPLEX* device_ffb,
                                              const int* ntotalm,
                                              const int* ntotal,
                                              const int* nmax,
                                              const int* nxm,
                                              const int* nym,
                                              const int* nzm,
                                              const int* nx,
                                              const int* ny,
                                              const int* nz,
                                              const int* nx2,
                                              const int* ny2,
                                              const int* nxy2,
                                              const int* nz2,
                                              const long long* planf,
                                              const long long* planb) {
    (void)ntotalm;
    (void)planf;
    (void)planb;
    if (!device_green || !device_ffx || !device_ffb || !ntotal || !nmax || !nxm || !nym ||
        !nzm || !nx || !ny || !nz || !nx2 || !ny2 || !nxy2 || !nz2 || *ntotal <= 0 ||
        *nmax <= 0 || *nxm < 0 || *nym < 0 || *nzm < 0) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: invalid resident arguments\n");
        return;
    }
    size_t vector_count = 0;
    size_t tensor_plane = 0;
    size_t tensor_count = 0;
    if (!checked_multiply(size_t(*ntotal), size_t(3), &vector_count) ||
        !checked_multiply(size_t(*nxm) + 1, size_t(*nym) + 1, &tensor_plane) ||
        !checked_multiply(tensor_plane, size_t(*nzm) + 1, &tensor_count) ||
        vector_count > size_t(INT_MAX) || tensor_count == 0) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: resident dimensions overflow\n");
        return;
    }
    CUCOMPLEX* device_vectors = get_device_vectors(vector_count);
    if (!device_vectors) {
        std::fprintf(stderr, "produitfftmatvectopt CUDA: cannot get GPU scratch\n");
        return;
    }
    (void)produitfftmatvectopt_cuda(device_green, device_vectors, device_ffx, device_ffb, *nx, *ny,
                                    *nz, *nx2, *ny2, *nz2, *nxy2, *ntotal, *nmax, tensor_count);
}

// Entry point used by inverserigopt.f90. Its six FFTW plan arguments are
// retained for ABI compatibility; the CUDA implementation owns a cuFFT plan.
//-----------------------------------------------------------
extern "C" void produitfftmatvectoptfftopt_(const COMPLEX* tensor_xx,
                                            const COMPLEX* tensor_xy,
                                            const COMPLEX* tensor_xz,
                                            const COMPLEX* tensor_yy,
                                            const COMPLEX* tensor_yz,
                                            const COMPLEX* tensor_zz,
                                            COMPLEX* vectx,
                                            COMPLEX* vecty,
                                            COMPLEX* vectz,
                                            const int* ntotalm,
                                            const int* ntotal,
                                            const int* nmax,
                                            const int* nxm,
                                            const int* nym,
                                            const int* nzm,
                                            const int* nx,
                                            const int* ny,
                                            const int* nz,
                                            const int* nx2,
                                            const int* ny2,
                                            const int* nxy2,
                                            const int* nz2,
                                            const COMPLEX* ffx,
                                            COMPLEX* ffb,
                                            const long long* plan_xf,
                                            const long long* plan_xb,
                                            const long long* plan_yf,
                                            const long long* plan_yb,
                                            const long long* plan_zf,
                                            const long long* plan_zb) {
    static bool path_reported = false;
    if (!path_reported) {
        std::fprintf(stderr,
                     "[produitfftmatvectoptfftopt] CUDA/cuFFT backend active (%s precision)\n",
#if PRODUITMATVECTFFT_SINGLE_PRECISION
                     "single"
#else
                     "double"
#endif
        );
        path_reported = true;
    }

    (void)plan_yf;
    (void)plan_yb;
    (void)plan_zf;
    (void)plan_zb;
    produitfftmatvectopt_(tensor_xx,
                          tensor_xy,
                          tensor_xz,
                          tensor_yy,
                          tensor_yz,
                          tensor_zz,
                          vectx,
                          vecty,
                          vectz,
                          ntotalm,
                          ntotal,
                          nmax,
                          nxm,
                          nym,
                          nzm,
                          nx,
                          ny,
                          nz,
                          nx2,
                          ny2,
                          nxy2,
                          nz2,
                          ffx,
                          ffb,
                          plan_xf,
                          plan_xb);
}

#undef CUFFT_TRANSFORM
#undef CUFFT_EXECUTE
#undef CUFFT_ZERO
#undef CUFFT_REAL
#undef CUFFT_ADD
#undef CUFFT_SUB
#undef CUFFT_MUL
