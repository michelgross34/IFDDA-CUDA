#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {
struct FortranComplex {
    CU_REAL real;
    CU_REAL imag;
};

//-----------------------------------------------------------
bool finite_value(CU_REAL value) {
    return std::isfinite(static_cast<double>(value));
}
} // namespace

//-----------------------------------------------------------
extern "C" void callbhmie_(const CU_REAL *refmed, const FortranComplex *epsilon,
                            const CU_REAL *radius, const CU_REAL *wavelength,
                            CU_REAL *extinction, CU_REAL *absorption,
                            CU_REAL *scattering, CU_REAL *asymmetry) {
    if (!refmed || !epsilon || !radius || !wavelength || !extinction || !absorption ||
        !scattering || !asymmetry || *refmed <= CU_REAL(0) || *radius <= CU_REAL(0) ||
        *wavelength <= CU_REAL(0))
        return;

    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795);
    const CU_REAL eps_magnitude = std::hypot(epsilon->real, epsilon->imag);
    const CU_REAL sqrt_real = std::sqrt(std::max(CU_REAL(0),
        (eps_magnitude + epsilon->real) / CU_REAL(2)));
    const CU_REAL sqrt_imag = std::copysign(std::sqrt(std::max(CU_REAL(0),
        (eps_magnitude - epsilon->real) / CU_REAL(2))), epsilon->imag);
    const CU_REAL relative_real = sqrt_real / *refmed;
    const CU_REAL relative_imag = sqrt_imag / *refmed;
    const CU_REAL x = CU_REAL(2) * pi * *radius * *refmed / *wavelength;
    if (!finite_value(x) || !finite_value(relative_real) || !finite_value(relative_imag))
        return;

    const CU_REAL x_stop = x + CU_REAL(4) * std::pow(x, CU_REAL(0.3333)) + CU_REAL(2);
    const CU_REAL y_magnitude = x * std::hypot(relative_real, relative_imag);
    const int max_order = 15000;
    const int nmx = static_cast<int>(std::max(x_stop, y_magnitude)) + 15;
    if (nmx < 2 || nmx > max_order)
        return;

    cuda_data::MieReferenceDeviceBuffers buffers{};
    if (!cuda_data::get_device_mie_reference_buffers(size_t(max_order), &buffers))
        return;
    if (!cuda_utils::check_cuda(mie_efficiencies_wrapper(
            buffers.recurrence, relative_real, relative_imag, x, *radius,
            buffers.outputs, max_order), "Mie reference", "launch efficiencies kernel") ||
        !cuda_utils::check_cuda(cudaDeviceSynchronize(), "Mie reference", "synchronize kernel"))
        return;

    CU_REAL result[5] = {};
    if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
            result, buffers.outputs, sizeof(result), cuda_utils::TransferCategory::Result),
            "Mie reference", "copy efficiencies"))
        return;
    if (result[4] != CU_REAL(0))
        return;
    *extinction = result[0];
    *absorption = result[1];
    *scattering = result[2];
    *asymmetry = result[3];
}
