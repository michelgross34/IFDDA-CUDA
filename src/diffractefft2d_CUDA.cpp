#include "cdmlib_cpp_precision.hpp"
#include <cstdint>
#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cmath>
#include <limits>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#define I (COMPLEX(0.0, 1.0))

static void set_info(char *out, size_t length, const char *message)
{
    size_t n = strlen(message);
    if (n > length) n = length;
    memset(out, ' ', length);
    memcpy(out, message, n);
}

extern "C" void diffractefft2d_(const int *nx, const int *ny, const int *nz,
                     const int *nxm, const int *nym, const int *nzm,
                     const int *nfft2d, int *tabfft2, const REAL *k0,
                     const REAL *xs, const REAL *ys, const REAL *zs,
                     const REAL *aretecube, COMPLEX *Eloinx,
                     COMPLEX *Eloiny, COMPLEX *Eloinz,
                     const COMPLEX *FF, int *imax, REAL *deltakx,
                     REAL *deltaky, COMPLEX *Ediffkzpos,
                     COMPLEX *Ediffkzneg, const int64_t *plan2f,
                     const int64_t *plan2b, int *nstop, char *infostr,
                     size_t infostr_len)
{
    int i, j, k, ii, jj, kk, indice, nfft2d2;
    REAL var1, var2, kx, ky, kz;
    COMPLEX ctmp, ctmp1;
    const int nfft = *nfft2d;
    const size_t plane = (size_t)nfft * (size_t)nfft;
    (void)nxm; (void)nym; (void)nzm; (void)plan2f; (void)plan2b;

    printf("Quick method with FFT2 %d\n", nfft);
    *deltakx = 2.0 * 3.14159265358979323846 / ((REAL)nfft * *aretecube);
    *deltaky = *deltakx;
    nfft2d2 = nfft / 2;
    var1 = (xs[0] + (REAL)nfft2d2 * *aretecube) * *deltakx;
    var2 = (ys[0] + (REAL)nfft2d2 * *aretecube) * *deltaky;

    if (nfft > 16384) {
        *nstop = 1;
        set_info(infostr, infostr_len, "nfft2d too large");
        return;
    }
    if (*deltakx >= *k0) {
        *nstop = 1;
        set_info(infostr, infostr_len, "In FFT Poynting nfft2d too small");
        return;
    }

    for (i = 0; i < nfft; ++i)
        tabfft2[i] = (i - nfft2d2 >= 0) ? i - nfft2d2 + 1 : nfft2d2 + i + 1;

    *imax = (int)floor(*k0 / *deltakx + 0.5) + 1;
    printf("deltakx %.16g m-1\nNumber of point in NA %d\n",
           *deltakx, 2 * *imax + 1);
    if (2 * *imax + 1 > nfft || 2 * *imax + 1 < 7) {
        printf("Size %d %d\n", 2 * *imax + 1, nfft);
        *nstop = 1;
        set_info(infostr, infostr_len, "In FFT diffract nfft2d too small");
        return;
    }

    size_t dipole_count = 0;
    size_t field_count = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx), size_t(*ny), &dipole_count) ||
        !cuda_utils::checked_multiply(dipole_count, size_t(*nz), &dipole_count) ||
        !cuda_utils::checked_multiply(dipole_count, size_t(3), &field_count) ||
        dipole_count > size_t(std::numeric_limits<int>::max())) {
        *nstop = 1;
        set_info(infostr, infostr_len, "diffraction dimensions overflow");
        return;
    }
    cuda_data::QuickDiffractionFftDeviceBuffers fft_buffers{};
    if (!cuda_data::get_device_quick_diffraction_fft_buffers(dipole_count, nfft,
                                                              &fft_buffers)) {
        *nstop = 1;
        set_info(infostr, infostr_len, "Cannot acquire cuFFT buffers for diffraction");
        return;
    }
    const CUCOMPLEX *device_field = get_device_result_vector_alias(FF, field_count);
    if (!device_field) {
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(fft_buffers.dipole_field, FF,
                    field_count * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result),
                "quick diffraction", "upload CPU field")) {
            *nstop = 1;
            set_info(infostr, infostr_len, "Cannot upload field for cuFFT diffraction");
            return;
        }
        device_field = fft_buffers.dipole_field;
    }
    if (!cuda_utils::check_cuda(
            quick_diffraction_build_fft_map_wrapper(fft_buffers.fft_map, nfft),
            "quick diffraction", "build cuFFT map")) {
        *nstop = 1;
        set_info(infostr, infostr_len, "Cannot build cuFFT diffraction map");
        return;
    }

    std::fill_n(Ediffkzpos, 3 * plane, COMPLEX{});
    std::fill_n(Ediffkzneg, 3 * plane, COMPLEX{});
    for (k = 0; k < *nz; ++k) {
        std::fill_n(Eloinx, plane, COMPLEX{});
        std::fill_n(Eloiny, plane, COMPLEX{});
        std::fill_n(Eloinz, plane, COMPLEX{});
        if (!cuda_utils::check_cuda(
                quick_diffraction_gather_wrapper(device_field, fft_buffers.fft_images,
                    fft_buffers.fft_map, *nx, *ny, int(size_t(*nx) * size_t(*ny)),
                    nfft, k),
                "quick diffraction", "pack cuFFT planes") ||
            !cuda_utils::check_cufft(
                CUFFT_EXECUTE(fft_buffers.fft_plan, fft_buffers.fft_images,
                              fft_buffers.fft_images, CUFFT_FORWARD),
                "quick diffraction", "transform cuFFT planes") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                Eloinx, fft_buffers.fft_images, plane * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::Result),
                "quick diffraction", "download x spectrum") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                Eloiny, fft_buffers.fft_images + plane, plane * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::Result),
                "quick diffraction", "download y spectrum") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                Eloinz, fft_buffers.fft_images + 2 * plane, plane * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::Result),
                "quick diffraction", "download z spectrum")) {
            *nstop = 1;
            set_info(infostr, infostr_len, "cuFFT diffraction failed");
            return;
        }
        kk = *nx * *ny * k;
        for (i = -*imax; i <= *imax; ++i) {
            for (j = -*imax; j <= *imax; ++j) {
                ii = *imax + i;
                jj = *imax + j;
                kx = (REAL)i * *deltakx;
                ky = (REAL)j * *deltaky;
                if (*k0 * *k0 - kx * kx - ky * ky > 0.0) {
                    kz = sqrt(*k0 * *k0 - kx * kx - ky * ky);
                    indice = (tabfft2[i + nfft2d2] - 1)
                           + nfft * (tabfft2[j + nfft2d2] - 1);
                    ctmp = (kx * Eloinx[indice] + ky * Eloiny[indice]
                            + kz * Eloinz[indice]) / *k0;
                    ctmp1 = std::exp(-I * kz * zs[kk])
                          * std::exp(-I * (var1 * (REAL)i + var2 * (REAL)j));
                    Ediffkzpos[ii + nfft * jj] += (Eloinx[indice] - kx * ctmp / *k0) * ctmp1;
                    Ediffkzpos[ii + nfft * jj + plane] += (Eloiny[indice] - ky * ctmp / *k0) * ctmp1;
                    Ediffkzpos[ii + nfft * jj + 2 * plane] += (Eloinz[indice] - kz * ctmp / *k0) * ctmp1;
                    ctmp = (kx * Eloinx[indice] + ky * Eloiny[indice]
                            - kz * Eloinz[indice]) / *k0;
                    ctmp1 = std::exp(I * kz * zs[kk])
                          * std::exp(-I * (var1 * (REAL)i + var2 * (REAL)j));
                    Ediffkzneg[ii + nfft * jj] += (Eloinx[indice] - kx * ctmp / *k0) * ctmp1;
                    Ediffkzneg[ii + nfft * jj + plane] += (Eloiny[indice] - ky * ctmp / *k0) * ctmp1;
                    Ediffkzneg[ii + nfft * jj + 2 * plane] += (Eloinz[indice] + kz * ctmp / *k0) * ctmp1;
                }
            }
        }
    }
    (void)nstop;
}

