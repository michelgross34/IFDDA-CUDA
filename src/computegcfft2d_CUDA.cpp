#include "cdmlib_cpp_precision.hpp"
#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"
/* C99 port of cdmlib/computegcfft2d.f90 (column-major field arrays). */
#include <cmath>
#include <climits>
#include <stddef.h>

static REAL abs2(COMPLEX z)
{
    return std::real(z)*std::real(z) + std::imag(z)*std::imag(z);
}

static bool computegcfft2d_cuda(
    const int *imax, const REAL *deltakx, const REAL *deltaky,
    const REAL *k0, const REAL *i0, const REAL *theta, const REAL *phi,
    const int *nfft2d, const COMPLEX *ediffkzpos, const COMPLEX *ediffkzneg,
    REAL *gasym, REAL *cscai)
{
    if (!imax || !deltakx || !deltaky || !k0 || !i0 || !theta || !phi ||
        !nfft2d || !ediffkzpos || !ediffkzneg || !gasym || !cscai ||
        *imax < 0 || *nfft2d <= 0 || *k0 <= REAL(0) || *i0 <= REAL(0))
        return false;

    const size_t nfft = static_cast<size_t>(*nfft2d);
    const size_t plane = nfft * nfft;
    const size_t side = static_cast<size_t>(2) * static_cast<size_t>(*imax) + 1;
    if (nfft > SIZE_MAX / nfft || side > SIZE_MAX / side ||
        plane > SIZE_MAX / 3 || side * side > static_cast<size_t>(INT_MAX))
        return false;
    const size_t sample_count = side * side;
    if (side > nfft)
        return false;

    if (!initialize_cuda_device())
        return false;

    CUCOMPLEX *device_positive = nullptr;
    CUCOMPLEX *device_negative = nullptr;
    CU_REAL *device_positive_output = nullptr;
    CU_REAL *device_negative_output = nullptr;
    CU_REAL *device_kx = nullptr;
    CU_REAL *device_angular = nullptr;
    CU_REAL *device_theta = nullptr;
    CU_REAL *device_phi = nullptr;
    CU_REAL *device_weights = nullptr;
    CU_REAL *device_weighted_asymmetry = nullptr;
    CU_REAL *device_ones = nullptr;
    bool ok = false;

    const size_t complex_bytes = 3 * plane * sizeof(CUCOMPLEX);
    const size_t sample_bytes = sample_count * sizeof(CU_REAL);
    const size_t angle_bytes = 2 * sizeof(CU_REAL);
    if (cudaMalloc(&device_positive, complex_bytes) != cudaSuccess ||
        cudaMalloc(&device_negative, complex_bytes) != cudaSuccess ||
        cudaMalloc(&device_positive_output, sample_bytes) != cudaSuccess ||
        cudaMalloc(&device_negative_output, sample_bytes) != cudaSuccess ||
        cudaMalloc(&device_kx, side * sizeof(CU_REAL)) != cudaSuccess ||
        cudaMalloc(&device_angular, angle_bytes) != cudaSuccess ||
        cudaMalloc(&device_theta, angle_bytes) != cudaSuccess ||
        cudaMalloc(&device_phi, angle_bytes) != cudaSuccess ||
        cudaMalloc(&device_weights, sample_bytes) != cudaSuccess ||
        cudaMalloc(&device_weighted_asymmetry, sample_bytes) != cudaSuccess ||
        cudaMalloc(&device_ones, sample_bytes) != cudaSuccess)
        goto cleanup;

    if (cuda_data::copy_host_to_device(device_positive, ediffkzpos, complex_bytes,
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_host_to_device(device_negative, ediffkzneg, complex_bytes,
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        quick_diffraction_postprocess_wrapper(
            device_positive, device_negative, *nfft2d, *imax, *deltakx, *deltaky, *k0,
            *theta * REAL(3.141592653589793238462643383279502884) / REAL(180),
            *phi * REAL(3.141592653589793238462643383279502884) / REAL(180), REAL(1),
            1, 1, device_positive_output, device_negative_output, device_kx, device_angular,
            device_theta, device_phi, device_weights, device_weighted_asymmetry, device_ones) !=
            cudaSuccess)
        goto cleanup;

    {
        cublasHandle_t handle = cuda_utils::get_cublas_handle();
        if (!handle)
            goto cleanup;
        CU_REAL scattering = REAL(0), asymmetry = REAL(0);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        if (cublasSdot(handle, static_cast<int>(sample_count), device_ones, 1,
                      device_weights, 1, &scattering) != CUBLAS_STATUS_SUCCESS ||
            cublasSdot(handle, static_cast<int>(sample_count), device_ones, 1,
                       device_weighted_asymmetry, 1, &asymmetry) != CUBLAS_STATUS_SUCCESS)
#else
        if (cublasDdot(handle, static_cast<int>(sample_count), device_ones, 1,
                      device_weights, 1, &scattering) != CUBLAS_STATUS_SUCCESS ||
            cublasDdot(handle, static_cast<int>(sample_count), device_ones, 1,
                       device_weighted_asymmetry, 1, &asymmetry) != CUBLAS_STATUS_SUCCESS)
#endif
            goto cleanup;
        if (!(scattering > REAL(0)) || !std::isfinite(static_cast<double>(scattering)) ||
            !std::isfinite(static_cast<double>(asymmetry)))
            goto cleanup;
        *gasym = asymmetry / scattering;
        *cscai = scattering * *k0 * *k0 * *k0 / *i0;
        ok = std::isfinite(static_cast<double>(*gasym)) &&
             std::isfinite(static_cast<double>(*cscai));
    }

cleanup:
    if (device_ones) cudaFree(device_ones);
    if (device_weighted_asymmetry) cudaFree(device_weighted_asymmetry);
    if (device_weights) cudaFree(device_weights);
    if (device_phi) cudaFree(device_phi);
    if (device_theta) cudaFree(device_theta);
    if (device_angular) cudaFree(device_angular);
    if (device_kx) cudaFree(device_kx);
    if (device_negative_output) cudaFree(device_negative_output);
    if (device_positive_output) cudaFree(device_positive_output);
    if (device_negative) cudaFree(device_negative);
    if (device_positive) cudaFree(device_positive);
    return ok;
}

extern "C" void computegcfft2d_(const int *imax, const REAL *deltakx,
                     const REAL *deltaky, const REAL *k0,
                     const REAL *i0, const REAL *theta,
                     const REAL *phi, const int *nfft2d,
                     const COMPLEX *ediffkzpos,
                     const COMPLEX *ediffkzneg,
                     REAL *gasym, REAL *cscai)
{
    if (computegcfft2d_cuda(imax, deltakx, deltaky, k0, i0, theta, phi, nfft2d,
                            ediffkzpos, ediffkzneg, gasym, cscai))
        return;

    const REAL pi = 3.141592653589793238462643383279502884;
    const REAL k02 = *k0 * *k0;
    const REAL k03 = k02 * *k0;
    const REAL t = *theta * pi / 180.0;
    const REAL p = *phi * pi / 180.0;
    REAL csca = 0.0, asym = 0.0;
    const size_t plane = (size_t)*nfft2d * (size_t)*nfft2d;

    for (int i = -*imax; i <= *imax; ++i) {
        for (int j = -*imax; j <= *imax; ++j) {
            const REAL kx = (REAL)i * *deltakx;
            const REAL ky = (REAL)j * *deltaky;
            const REAL kz2 = k02 - kx*kx - ky*ky;
            if (kz2 > 0.0) {
                const REAL kz = std::sqrt(kz2);
                const int ii = *imax + i;
                const int jj = *imax + j;
                const size_t base = (size_t)ii + (size_t)*nfft2d * (size_t)jj;
                REAL emod = 0.0;
                for (int c = 0; c < 3; ++c) {
                    emod += abs2(ediffkzpos[base + plane*(size_t)c]);
                }
                const REAL weight = *deltakx * *deltaky * emod / kz;
                csca += weight;
                asym += weight * (kx*std::sin(t)*std::cos(p) + ky*std::sin(t)*std::sin(p) + kz*std::cos(t)) / *k0;

                emod = 0.0;
                for (int c = 0; c < 3; ++c) {
                    emod += abs2(ediffkzneg[base + plane*(size_t)c]);
                }
                const REAL neg_weight = *deltakx * *deltaky * emod / kz;
                csca += neg_weight;
                asym += neg_weight * (kx*std::sin(t)*std::cos(p) + ky*std::sin(t)*std::sin(p) - kz*std::cos(t)) / *k0;
            }
        }
    }
    *gasym = asym / csca;
    *cscai = csca * k03 / *i0;
}
