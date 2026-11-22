#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

//-----------------------------------------------------------
void set_info(char *infostr, const char *message) {
    if (!infostr)
        return;
    std::memset(infostr, ' ', 64);
    if (message)
        std::memcpy(infostr, message, std::min(size_t(64), std::strlen(message)));
}

//-----------------------------------------------------------
bool rounded_int(CU_REAL value, int *result) {
    if (!result || !std::isfinite(double(value)) ||
        value < CU_REAL(std::numeric_limits<int>::min()) ||
        value > CU_REAL(std::numeric_limits<int>::max()))
        return false;
    *result = int(std::round(value));
    return true;
}

//-----------------------------------------------------------
bool fail(int *nstop, char *infostr, const char *message) {
    if (nstop)
        *nstop = 1;
    set_info(infostr, message);
    return false;
}

} // namespace

//-----------------------------------------------------------
extern "C" void micromasque_(CUCOMPLEX *mask, const int *nmasque, int *imaxinc, int *imaxk0,
                              int *nbitermic, const int *ntypemic, const int *nfft2d,
                              CU_REAL *deltak, CU_REAL *deltakx, CU_REAL *numaper,
                              CU_REAL *numaperinc, CU_REAL *numaperinc2, CU_REAL *kcnax,
                              CU_REAL *kcnay, CU_REAL *aretecube, CU_REAL *k0, const CU_REAL *xs,
                              const CU_REAL *ys, const int *nxm, const int *nym, const int *nzm,
                              const int *nbsphere, const int *nquicklens, int *nstop, char *infostr) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!mask || !nmasque || !imaxinc || !imaxk0 || !nbitermic || !ntypemic || !nfft2d ||
        !deltak || !deltakx || !numaper || !numaperinc || !numaperinc2 || !kcnax || !kcnay ||
        !aretecube || !k0 || !xs || !ys || !nxm || !nym || !nzm || !nbsphere || !nquicklens ||
        *nmasque <= 0 || *nfft2d <= 0 || *nbsphere <= 0 || *nxm <= 0 || *nym <= 0 || *nzm <= 0 ||
        size_t(*nbsphere) > size_t(*nxm) * size_t(*nym) * size_t(*nzm) ||
        *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0)) {
        set_info(infostr, "Invalid microscopy mask arguments");
        return;
    }

    set_info(infostr, "");
    *imaxinc = 0;
    *imaxk0 = 0;
    *nbitermic = 0;
    const CU_REAL zero = CU_REAL(0);
    const CU_REAL one = CU_REAL(1);
    const CU_REAL pi = CU_REAL(3.141592653589793238462643383279502884);
    const CU_REAL twopi = CU_REAL(2) * pi;
    const CU_REAL numaperk = *k0 * *numaperinc;
    const CU_REAL numaperk2 = *k0 * *numaperinc2;
    if ((*ntypemic == 1 || *ntypemic == 6 || *ntypemic == 7) &&
        (*numaperinc >= one || *numaperinc <= zero)) {
        fail(nstop, infostr, "NA inc strictly between 0 and 1");
        return;
    }
    if ((*ntypemic == 2 || *ntypemic == 5) && *numaperinc2 >= *numaperinc) {
        fail(nstop, infostr, "NA inc 2 larger than NA inc!");
        return;
    }
    if ((*ntypemic == 2 || *ntypemic == 5) && *numaperinc2 < zero) {
        fail(nstop, infostr, "NA inc 2 smaller than 0");
        return;
    }
    if (*ntypemic == 3) {
        const CU_REAL pupil_offset = std::sqrt(*kcnax * *kcnax + *kcnay * *kcnay);
        if (pupil_offset + *numaperinc >= CU_REAL(1.000000001)) {
            fail(nstop, infostr, "pupil larger than 1");
            return;
        }
        if (*numaperinc <= zero) {
            fail(nstop, infostr, "NA inc strictly larger than 0");
            return;
        }
        if (std::abs(*kcnax) >= one) {
            fail(nstop, infostr, "X position of the pupil larger than 1");
            return;
        }
        if (std::abs(*kcnay) >= one) {
            fail(nstop, infostr, "Y position of the pupil larger than 1");
            return;
        }
    }
    if (*ntypemic != 1 && *ntypemic != 2 && *ntypemic != 3 && *ntypemic != 5 &&
        *ntypemic != 6 && *ntypemic != 7) {
        fail(nstop, infostr, "Unsupported microscopy mask type");
        return;
    }

    cuda_data::MicroscopeMaskDeviceBuffers buffers{};
    const size_t coordinate_count = size_t(*nbsphere);
    if (size_t(*nmasque) > std::numeric_limits<size_t>::max() / size_t(*nmasque)) {
        fail(nstop, infostr, "Microscopy mask size overflow");
        return;
    }
    const size_t mask_capacity = size_t(*nmasque) * size_t(*nmasque);
    if (
        !cuda_data::get_device_microscope_mask_buffers(coordinate_count, mask_capacity, &buffers) ||
        cuda_data::copy_host_to_device(buffers.xs, xs, coordinate_count * sizeof(CU_REAL),
                                       cuda_utils::TransferCategory::Geometry) != cudaSuccess ||
        cuda_data::copy_host_to_device(buffers.ys, ys, coordinate_count * sizeof(CU_REAL),
                                       cuda_utils::TransferCategory::Geometry) != cudaSuccess ||
        !cuda_utils::check_cuda(microscope_mask_extrema_wrapper(
                                    buffers.xs, buffers.ys, *nbsphere, buffers.extrema),
                                "micromasque CUDA", "reduce object extents")) {
        set_info(infostr, "Unable to prepare CUDA microscopy mask buffers");
        return;
    }

    CU_REAL extrema[4]{};
    if (cuda_data::copy_device_to_host(extrema, buffers.extrema, sizeof(extrema),
                                       cuda_utils::TransferCategory::Diagnostic) != cudaSuccess) {
        set_info(infostr, "Unable to read CUDA microscopy mask extents");
        return;
    }
    const CU_REAL deltakm = pi / std::max(extrema[1] - extrema[0], extrema[3] - extrema[2]);
    int imul = 0;
    if (*nquicklens == 1) {
        *deltakx = twopi / (*aretecube * CU_REAL(*nfft2d));
        if (*deltakx >= numaperk) {
            fail(nstop, infostr, "In FFT lens nfft2d too small:Increase size of FFT 1");
            return;
        }
        if (!rounded_int(deltakm / *deltakx, &imul) || imul == 0) {
            fail(nstop, infostr, "In FFT lens nfft2d too small:Increase size of FFT 2");
            return;
        }
        for (int attempt = 0; attempt < 100000; ++attempt) {
            *deltak = *deltakx * CU_REAL(imul);
            if (!rounded_int(numaperk / *deltak, imaxinc)) {
                fail(nstop, infostr, "Invalid incident numerical aperture");
                return;
            }
            ++(*imaxinc);
            if (*imaxinc > 3 &&
                ((*ntypemic != 2 && *ntypemic != 5) ||
                 *deltak <= (numaperk - numaperk2) / CU_REAL(2)))
                break;
            --imul;
            if (imul == 0) {
                fail(nstop, infostr, "In FFT lens nfft2d too small:Increase size of FFT 3");
                return;
            }
        }
    } else {
        int k = 0;
        for (int attempt = 0; attempt < 100000; ++attempt) {
            *deltakx = twopi / (CU_REAL(*nfft2d) * *aretecube) / std::ldexp(one, k);
            if (!rounded_int(numaperk / *deltakx, imaxk0)) {
                fail(nstop, infostr, "Invalid diffracted numerical aperture");
                return;
            }
            ++(*imaxk0);
            if (*imaxk0 > 5)
                break;
            ++k;
        }
        if (!rounded_int(deltakm / *deltakx, &imul)) {
            fail(nstop, infostr, "Invalid incident numerical aperture");
            return;
        }
        for (int attempt = 0; attempt < 100000; ++attempt) {
            if (imul == 0) {
                ++k;
                *deltakx = twopi / (CU_REAL(*nfft2d) * *aretecube) / std::ldexp(one, k);
                if (!rounded_int(numaperk / *deltakx, imaxk0)) {
                    fail(nstop, infostr, "Invalid diffracted numerical aperture");
                    return;
                }
                ++(*imaxk0);
                if (!rounded_int(deltakm / *deltakx, &imul)) {
                    fail(nstop, infostr, "Invalid incident numerical aperture");
                    return;
                }
                continue;
            }
            *deltak = *deltakx * CU_REAL(imul);
            if (!rounded_int(numaperk / *deltak, imaxinc)) {
                fail(nstop, infostr, "Invalid incident numerical aperture");
                return;
            }
            ++(*imaxinc);
            if (*imaxinc > 4 &&
                ((*ntypemic != 2 && *ntypemic != 5) ||
                 *deltak <= (numaperk - numaperk2) / CU_REAL(2)))
                break;
            --imul;
        }
    }

    if (*ntypemic == 3) {
        const CU_REAL pupil_offset = std::sqrt(*kcnax * *kcnax + *kcnay * *kcnay);
        if (!rounded_int((pupil_offset + *numaperinc) * *k0 / *deltak, imaxinc)) {
            fail(nstop, infostr, "Invalid Schieren pupil dimensions");
            return;
        }
        ++(*imaxinc);
    }
    if (*imaxinc > (std::numeric_limits<int>::max() - 1) / 2) {
        fail(nstop, infostr, "Microscopy mask dimensions overflow");
        return;
    }
    const int sizemask = 2 * *imaxinc + 1;
    if (sizemask > *nmasque) {
        fail(nstop, infostr, "Too many illumination=too many time");
        return;
    }
    const size_t compact_count = size_t(sizemask) * size_t(sizemask);
    CUCOMPLEX *host_mask_end = mask + compact_count;
    std::fill(mask, host_mask_end, CUFFT_ZERO);
    const CU_REAL schieren_center_x = *kcnax * *k0;
    const CU_REAL schieren_center_y = *kcnay * *k0;
    if (!cuda_utils::check_cuda(microscope_mask_fill_wrapper(
                                    buffers.mask, buffers.selected_count, *imaxinc, sizemask,
                                    *ntypemic, *deltak, numaperk, numaperk2, schieren_center_x,
                                    schieren_center_y),
                                "micromasque CUDA", "fill illumination mask") ||
        cuda_data::copy_device_to_host(mask, buffers.mask, compact_count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(nbitermic, buffers.selected_count, sizeof(int),
                                       cuda_utils::TransferCategory::Diagnostic) != cudaSuccess) {
        set_info(infostr, "Unable to compute CUDA microscopy mask");
        return;
    }
    if (*nbitermic <= 8) {
        fail(nstop, infostr, "Not enough illumination: increase size FFT");
        return;
    }
    *nstop = 0;
}
