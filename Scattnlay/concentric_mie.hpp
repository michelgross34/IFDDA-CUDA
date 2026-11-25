#ifndef IFDDA_CONCENTRIC_MIE_HPP
#define IFDDA_CONCENTRIC_MIE_HPP

#include <complex>
#include <vector>

namespace ifdda_scattnlay {

struct CrossSections {
    double extinction, scattering, absorption;
};

// Wavelength and interface radii in metres, from the core outwards.
// Relative permittivities of nonmagnetic layers in vacuum. Results in m².
CrossSections cross_sections(double wavelength, const std::vector<double>& radii,
                            const std::vector<std::complex<double>>& permittivities);

}

#endif
