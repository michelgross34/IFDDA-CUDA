#include "concentric_mie.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
double parse(const char* argument) {
    std::size_t used = 0;
    const double value = std::stod(argument, &used);
    if (argument[used] != '\0') {
        throw std::invalid_argument("expected a number");
    }
    return value;
}
}

int main(int argc, char** argv) {
    char program[] = "ifdda_scattnlay_sections";
    char wavelength[] = "31.41592653589793";
    char r1[] = "2.5", e1[] = "2.25", i1[] = "0";
    char r2[] = "5", e2[] = "1.25", i2[] = "0";
    char* defaults[] = {program, wavelength, r1, e1, i1, r2, e2, i2, nullptr};
    if (argc == 1) {
        argc = 8;
        argv = defaults;
    }
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: ifdda_scattnlay_sections [wavelength_nm radius_nm eps_re eps_im"
                     " [radius_nm eps_re eps_im ...]]\n"
                     "Defaults (no arguments): 31.41592653589793 2.5 2.25 0 5 1.25 0\n";
        return 0;
    }
    if (argc < 5 || (argc - 2) % 3 != 0) {
        std::cerr << "Usage: ifdda_scattnlay_sections wavelength_nm radius_nm eps_re eps_im"
                     " [radius_nm eps_re eps_im ...]\n";
        return 2;
    }
    try {
        std::vector<double> radii;
        std::vector<std::complex<double>> permittivities;
        for (int arg = 2; arg < argc; arg += 3) {
            radii.push_back(1e-9 * parse(argv[arg]));
            permittivities.emplace_back(parse(argv[arg + 1]), parse(argv[arg + 2]));
        }
        const auto result = ifdda_scattnlay::cross_sections(1e-9 * parse(argv[1]), radii, permittivities);
        std::cout << std::scientific << std::setprecision(16)
                  << "Cext (Scattnlay) = " << result.extinction << " m2\n"
                  << "Csca (Scattnlay) = " << result.scattering << " m2\n"
                  << "Cabs (Scattnlay) = " << result.absorption << " m2\n";
    } catch (const std::exception& error) {
        std::cerr << "Scattnlay: " << error.what() << '\n';
        return 1;
    }
}
