#include "concentric_mie.hpp"
#include "upstream/nmie-basic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace ifdda_scattnlay {

CrossSections cross_sections(double wavelength, const std::vector<double>& radii,
                             const std::vector<std::complex<double>>& permittivities) {
    if (!std::isfinite(wavelength) || wavelength <= 0 ||
        radii.empty() || radii.size() != permittivities.size()) {
        throw std::invalid_argument("expected a positive wavelength and one permittivity per layer");
    }

    std::vector<double> sizes;
    std::vector<std::complex<double>> indices;
    sizes.reserve(radii.size());
    indices.reserve(radii.size());
    bool lossless = true;
    bool has_contrast = false;
    double previous = 0;
    for (std::size_t i = 0; i < radii.size(); ++i) {
        const double radius = radii[i];
        const auto epsilon = permittivities[i];
        if (!std::isfinite(radius) || radius <= previous ||
            !std::isfinite(epsilon.real()) || !std::isfinite(epsilon.imag()) ||
            epsilon.imag() < 0 || epsilon == std::complex<double>{}) {
            throw std::invalid_argument("expected increasing radii and finite, passive, nonzero permittivities");
        }
        previous = radius;
        lossless = lossless && epsilon.imag() == 0;
        has_contrast = has_contrast || epsilon != std::complex<double>(1.0);
        const double size = (radius / wavelength) * (2 * std::numbers::pi);
        const auto index = std::sqrt(std::complex<double>(epsilon.real(), std::abs(epsilon.imag())));
        // Scattnlay estimates its multipole order using integer indices.
        constexpr double order_limit = std::numeric_limits<int>::max() / 4.0;
        if (!(size > 0 && size < order_limit && size * std::abs(index) < order_limit)) {
            throw std::invalid_argument("size parameter outside the supported range");
        }
        sizes.push_back(size);
        indices.push_back(index);
    }

    const double area = std::numbers::pi * radii.back() * radii.back();
    if (!std::isfinite(area) || area <= 0) {
        throw std::invalid_argument("outer geometric area outside the supported range");
    }
    if (!has_contrast) return {};
    nmie::MultiLayerMie<double> mie;
    mie.SetLayersSize(sizes);
    mie.SetLayersIndex(indices);
    mie.RunMieCalculation();
    const CrossSections result{area * mie.GetQext(), area * mie.GetQsca(), area * mie.GetQabs()};
    const double tolerance = 128 * std::numeric_limits<double>::epsilon() *
                             std::max(result.extinction, result.scattering);
    if (!std::isfinite(result.extinction) || !std::isfinite(result.scattering) ||
        !std::isfinite(result.absorption) || result.extinction < 0 || result.scattering < 0 ||
        result.absorption < -tolerance || (lossless && result.absorption > tolerance)) {
        throw std::runtime_error("Scattnlay produced invalid cross sections");
    }
    return {result.extinction, result.scattering,
            lossless ? 0.0 : std::max(0.0, result.absorption)};
}

}
