#include "mishchenko_c.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void usage(std::ostream& output) {
    output << "Usage:\n"
           << "  ifdda_mishchenko_sections\n"
           << "  ifdda_mishchenko_sections spheroid wavelength_nm transverse_nm axial_nm mr mi theta_deg\n"
           << "  ifdda_mishchenko_sections ellipsoid wavelength_nm a_nm b_nm c_nm mr mi theta_deg\n"
           << "  ifdda_mishchenko_sections cylinder wavelength_nm radius_nm height_nm mr mi theta_deg\n"
           << "  ifdda_mishchenko_sections capsule wavelength_nm radius_nm straight_length_nm mr mi theta_deg\n"
           << "  Append --precision auto|dp|qp (default auto).\n"
           << "  Append [Ep_re Ep_im Es_re Es_im] to specify polarization.\n"
           << "  Without arguments, run spheroid 630 140 280 1.5 0.02 27 with E_theta polarization.\n";
}

double number(const char* argument) {
    std::size_t used = 0;
    const double value = std::stod(argument, &used);
    if (argument[used] != '\0' || !std::isfinite(value)) {
        throw std::invalid_argument("expected a finite number");
    }
    return value;
}

double length(const char* argument) {
    const double value = number(argument);
    if (value <= 0) throw std::invalid_argument("lengths must be positive");
    return value;
}

struct Geometry {
    int shape;
    double transverse;
    double axial;
};

Geometry geometry(std::string_view shape, const char* const* dimensions) {
    const double a = length(dimensions[0]);
    if (shape == "capsule") {
        const double straight = number(dimensions[1]);
        if (straight < 0) throw std::invalid_argument("capsule straight length must be nonnegative");
        return {IFDDA_MISHCHENKO_CAPSULE, a, straight / 2};
    }
    const double b = length(dimensions[1]);
    if (shape == "cylinder") return {IFDDA_MISHCHENKO_CYLINDER, a, b / 2};
    if (shape == "spheroid") return {IFDDA_MISHCHENKO_SPHEROID, a, b};

    const double c = length(dimensions[2]);
    if (a == b) return {IFDDA_MISHCHENKO_SPHEROID, a, c};
    if (a == c) return {IFDDA_MISHCHENKO_SPHEROID, a, b};
    if (b == c) return {IFDDA_MISHCHENKO_SPHEROID, b, a};
    throw std::invalid_argument("a triaxial ellipsoid is outside the axisymmetric solver");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        usage(std::cout);
        return 0;
    }
    int precision=0;
    if (argc>=3 && std::string_view(argv[argc-2])=="--precision") {
        const std::string_view mode=argv[argc-1];
        if(mode=="dp") precision=1;
        else if(mode=="qp") precision=2;
        else if(mode!="auto") { usage(std::cerr); return 1; }
        argc-=2;
    }
    constexpr std::array<const char*, 8> default_arguments{
        "ifdda_mishchenko_sections", "spheroid", "630", "140", "280", "1.5", "0.02", "27"
    };
    const char* const* arguments = argv;
    if (argc == 1) {
        arguments = default_arguments.data();
        argc = static_cast<int>(default_arguments.size());
    }
    const std::string_view shape = argc >= 2 ? arguments[1] : "";
    const int required = shape == "ellipsoid" ? 9 : 8;
    if ((shape != "spheroid" && shape != "ellipsoid" && shape != "cylinder" && shape != "capsule") ||
        (argc != required && argc != required + 4)) {
        usage(std::cerr);
        return IFDDA_MISHCHENKO_INVALID_INPUT;
    }

    try {
        const double wavelength = length(arguments[2]);
        const double wavelength_m = wavelength * 1e-9;
        const double area = wavelength_m * wavelength_m;
        if (!std::isfinite(area) || area <= 0) {
            throw std::invalid_argument("wavelength outside the output range");
        }
        const Geometry object = geometry(shape, arguments + 3);
        const double transverse = object.transverse / wavelength;
        const double axial = object.axial / wavelength;
        const bool cylinder = object.shape == IFDDA_MISHCHENKO_CYLINDER;
        const bool capsule = object.shape == IFDDA_MISHCHENKO_CAPSULE;
        const double volume_axis = capsule ? transverse + 1.5 * axial : (cylinder ? 1.5 * axial : axial);
        const double volume_radius = std::cbrt(transverse * transverse * volume_axis);
        const double longest = capsule ? transverse + axial :
            (cylinder ? std::hypot(transverse, axial) : std::max(transverse, axial));
        const double ratio = capsule ? axial / transverse : transverse / axial;
        const double mr = number(arguments[required - 3]);
        const double mi = number(arguments[required - 2]);
        const double theta = number(arguments[required - 1]);

        std::array<double, 4> polarization{1, 0, 0, 0};
        if (argc > required) {
            for (std::size_t i = 0; i < polarization.size(); ++i) {
                polarization[i] = number(arguments[required + i]);
            }
        }
        double sections[3], variation;
        int order, quadrature, precision_used;
        const int status = ifdda_mishchenko_v2(object.shape, volume_radius, ratio,
            longest, mr, mi, theta, polarization[0], polarization[1], polarization[2],
            polarization[3], precision, sections, &order, &quadrature, &variation, &precision_used);
        if (status != IFDDA_MISHCHENKO_OK) {
            std::cerr << "Mishchenko: "
                      << (status == IFDDA_MISHCHENKO_CUDA_ERROR ? "CUDA backend error\n" :
                          status == IFDDA_MISHCHENKO_NO_CONVERGENCE
                          ? "cross sections did not converge\n" : "invalid or unsupported input\n");
            return status;
        }
        std::cout << std::scientific << std::setprecision(16)
                  << "Cext (T-matrix) = " << sections[0] * area << " m2\n"
                  << "Csca (T-matrix) = " << sections[1] * area << " m2\n"
                  << "Cabs (T-matrix) = " << sections[2] * area << " m2\n"
                  << "Order = " << order << ", surface quadrature = " << quadrature
                  << ", relative variation = " << variation << '\n'
                  << "Backend = " << ifdda_mishchenko_backend() << '\n'
                  << "T-matrix precision = " << (precision_used==2 ? "qp" : "dp") << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Mishchenko: " << error.what() << '\n';
        return IFDDA_MISHCHENKO_INVALID_INPUT;
    }
}
