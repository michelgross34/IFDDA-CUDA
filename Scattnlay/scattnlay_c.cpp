#include "scattnlay_c.h"
#include "concentric_mie.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {

void write_error(char* message, std::size_t capacity, const char* details) {
    if (message && capacity > 0) {
        std::snprintf(message, capacity, "%s", details);
    }
}

int calculate(int count, double wavelength, const double* radii, const double* real,
              const double* imag, bool use_index, double* sections,
              char* message, std::size_t capacity) {
    write_error(message, capacity, "");
    if (sections) std::fill_n(sections, 3, 0.0);
    if (count <= 0 || !radii || !real || !imag || !sections) {
        write_error(message, capacity, "expected a positive layer count and non-null arrays");
        return IFDDA_SCATTNLAY_INVALID_INPUT;
    }
    try {
        const std::vector<double> layer_radii(radii, radii + count);
        std::vector<std::complex<double>> permittivities;
        permittivities.reserve(count);
        for (int layer = 0; layer < count; ++layer) {
            const std::complex<double> material(real[layer], imag[layer]);
            if (use_index && (!std::isfinite(material.real()) || !std::isfinite(material.imag()) ||
                              material.real() < 0 || material.imag() < 0)) {
                throw std::invalid_argument("expected nonnegative, finite refractive-index components");
            }
            permittivities.push_back(use_index ? material * material : material);
        }
        const auto result = ifdda_scattnlay::cross_sections(wavelength, layer_radii, permittivities);
        sections[0] = result.extinction;
        sections[1] = result.scattering;
        sections[2] = result.absorption;
        return IFDDA_SCATTNLAY_OK;
    } catch (const std::invalid_argument& error) {
        write_error(message, capacity, error.what());
        return IFDDA_SCATTNLAY_INVALID_INPUT;
    } catch (const std::exception& error) {
        write_error(message, capacity, error.what());
    } catch (...) {
        write_error(message, capacity, "unknown Scattnlay failure");
    }
    return IFDDA_SCATTNLAY_CALCULATION_FAILED;
}

}

extern "C" int ifdda_scattnlay_sections_eps_v1(
    int layer_count, double wavelength_m, const double* radii_m,
    const double* eps_real, const double* eps_imag, double sections_m2[3],
    char* error_message, size_t error_capacity) {
    return calculate(layer_count, wavelength_m, radii_m, eps_real, eps_imag, false,
                     sections_m2, error_message, error_capacity);
}

extern "C" int ifdda_scattnlay_sections_index_v1(
    int layer_count, double wavelength_m, const double* radii_m,
    const double* index_real, const double* index_imag, double sections_m2[3],
    char* error_message, size_t error_capacity) {
    return calculate(layer_count, wavelength_m, radii_m, index_real, index_imag, true,
                     sections_m2, error_message, error_capacity);
}
