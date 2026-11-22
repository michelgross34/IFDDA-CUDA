#include "cdmlib_cpp_abi.hpp"
#include "cuda_interfaces.h"
#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>




namespace {

//-----------------------------------------------------------
int positive_modulo(int value, int modulus)
{
    const int remainder = value % modulus;
    return remainder < 0 ? remainder + modulus : remainder;
}

//-----------------------------------------------------------
void fill_speckle_phase_grid(int nfft, REAL k0, REAL aperture,
    REAL spacing, int seed, COMPLEX *phases)
{
    const int half = nfft / 2;
    const REAL kmax = k0 * aperture;
    const REAL deltak = REAL(6.283185307179586476925286766559005768) /
        (REAL(nfft) * spacing);
    const REAL cutoff = kmax * kmax * REAL(0.999);

    random_bridge_init(seed);
    for (int nkx = -half; nkx < half; ++nkx)
    {
        const int ix = positive_modulo(nkx, nfft);
        const REAL kx = deltak * REAL(nkx);
        for (int nky = -half; nky < half; ++nky)
        {
            const REAL ky = deltak * REAL(nky);
            if (kx * kx + ky * ky > cutoff)
            {
                continue;
            }

            const REAL random = random_bridge_next();
            const REAL angle = REAL(6.283185307179586476925286766559005768) * random;
            const int iy = positive_modulo(nky, nfft);
            const size_t index = size_t(ix) + size_t(nfft) * size_t(iy);
            phases[index] = COMPLEX(std::cos(angle), std::sin(angle));
        }
    }
}

//-----------------------------------------------------------
void fill_speckle_uniform_grid(int nfft, REAL k0, REAL aperture,
    REAL spacing, int seed, std::vector<CU_REAL>* uniforms)
{
    const int half = nfft / 2;
    const REAL kmax = k0 * aperture;
    const REAL deltak = REAL(6.283185307179586476925286766559005768) /
        (REAL(nfft) * spacing);
    const REAL cutoff = kmax * kmax * REAL(0.999);

    random_bridge_init(seed);
    for (int nkx = -half; nkx < half; ++nkx)
    {
        const int ix = positive_modulo(nkx, nfft);
        const REAL kx = deltak * REAL(nkx);
        for (int nky = -half; nky < half; ++nky)
        {
            const REAL ky = deltak * REAL(nky);
            if (kx * kx + ky * ky > cutoff)
                continue;

            const REAL random = random_bridge_next();
            const int iy = positive_modulo(nky, nfft);
            const size_t index = size_t(ix) + size_t(nfft) * size_t(iy);
            (*uniforms)[index] = static_cast<CU_REAL>(random);
        }
    }
}

} // namespace

//-----------------------------------------------------------
extern "C" void ifdda_prepare_speckle_phases_(
    const int *nfft, const int *nfft_max, const REAL *k0,
    const REAL *numaperil, const REAL *spacing, const int *speckseed,
    const int *bidirectional, COMPLEX *phase_positive,
    COMPLEX *phase_negative)
{
    if (nfft == nullptr || nfft_max == nullptr || k0 == nullptr ||
        numaperil == nullptr || spacing == nullptr || speckseed == nullptr ||
        bidirectional == nullptr || phase_positive == nullptr ||
        phase_negative == nullptr || *nfft <= 0 || (*nfft & 1) != 0 ||
        *nfft_max < *nfft || *k0 <= REAL(0) || *numaperil < REAL(0) ||
        *spacing <= REAL(0) ||
        size_t(*nfft_max) > std::numeric_limits<size_t>::max() / size_t(*nfft_max))
    {
        std::fprintf(stderr, "Speckle phase generation: invalid arguments\n");
        std::abort();
    }

    const size_t output_count = size_t(*nfft_max) * size_t(*nfft_max);
    std::fill_n(phase_positive, output_count, COMPLEX(0, 0));
    std::fill_n(phase_negative, output_count, COMPLEX(0, 0));
    fill_speckle_phase_grid(*nfft, *k0, *numaperil, *spacing,
        *speckseed, phase_positive);

    if (*bidirectional != 0)
    {
        const int reverse_seed = static_cast<int>(
            static_cast<unsigned int>(*speckseed) + 1U);
        fill_speckle_phase_grid(*nfft, *k0, *numaperil, *spacing,
            reverse_seed, phase_negative);
    }
}

//-----------------------------------------------------------
extern "C" int ifdda_prepare_speckle_phases_device_(
    const int *nfft, const int *nfft_max, const CU_REAL *k0,
    const CU_REAL *numaperil, const CU_REAL *spacing, const int *speckseed,
    const int *bidirectional, const CUCOMPLEX **phase_positive,
    const CUCOMPLEX **phase_negative)
{
    if (!nfft || !nfft_max || !k0 || !numaperil || !spacing || !speckseed ||
        !bidirectional || !phase_positive || !phase_negative || *nfft <= 0 ||
        *nfft_max < *nfft || *k0 <= CU_REAL(0) ||
        *numaperil < CU_REAL(0) || *spacing <= CU_REAL(0) ||
        (*bidirectional != 0 && *bidirectional != 1) ||
        size_t(*nfft_max) > std::numeric_limits<size_t>::max() / size_t(*nfft_max))
        return 1;

    const size_t phase_count = size_t(*nfft_max) * size_t(*nfft_max);
    std::vector<CU_REAL> uniforms_positive;
    std::vector<CU_REAL> uniforms_negative;
    try
    {
        uniforms_positive.assign(phase_count, CU_REAL(-1));
        if (*bidirectional)
            uniforms_negative.assign(phase_count, CU_REAL(-1));
    }
    catch (...)
    {
        return 2;
    }

    fill_speckle_uniform_grid(*nfft, REAL(*k0), REAL(*numaperil),
        REAL(*spacing), *speckseed, &uniforms_positive);
    if (*bidirectional)
    {
        const int reverse_seed = static_cast<int>(
            static_cast<unsigned int>(*speckseed) + 1U);
        fill_speckle_uniform_grid(*nfft, REAL(*k0), REAL(*numaperil),
            REAL(*spacing), reverse_seed, &uniforms_negative);
    }

    if (!initialize_cuda_device())
        return 3;
    cuda_data::SpecklePhaseDeviceBuffers buffers{};
    if (!cuda_data::get_device_speckle_phase_buffers(
            phase_count, *bidirectional != 0, &buffers))
        return 4;

    using cuda_utils::TransferCategory;
    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.uniforms_positive, uniforms_positive.data(),
        phase_count * sizeof(CU_REAL), TransferCategory::IncidentField);
    if (status == cudaSuccess && *bidirectional)
        status = cuda_data::copy_host_to_device(
            buffers.uniforms_negative, uniforms_negative.data(),
            phase_count * sizeof(CU_REAL), TransferCategory::IncidentField);
    constexpr CU_REAL two_pi = CU_REAL(6.283185307179586476925286766559005768L);
    if (status == cudaSuccess)
        status = speckle_uniforms_to_phase_wrapper(
            buffers.uniforms_positive, buffers.phase_positive, phase_count, two_pi);
    if (status == cudaSuccess && *bidirectional)
        status = speckle_uniforms_to_phase_wrapper(
            buffers.uniforms_negative, buffers.phase_negative, phase_count, two_pi);
    if (status != cudaSuccess)
        return 5;

    *phase_positive = buffers.phase_positive;
    *phase_negative = *bidirectional ? buffers.phase_negative : nullptr;
    return 0;
}
