#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"
#include "scattnlay_reference.hpp"
#include "tmatrix_reference.hpp"

#ifdef IFDDA_WITH_SCATTNLAY
#include "../Scattnlay/concentric_mie.hpp"
#include <exception>
#endif
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <string>
#endif
#if defined(_WIN32) || defined(IFDDA_LINK_MISHCHENKO)
#include <array>
#include <limits>
#include <mutex>
#endif

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

extern "C" void ifdda_report_scattnlay_(const int* layers, const REAL* radii,
    const COMPLEX* permittivities, const REAL* wavelength, const REAL* dda_ext,
    const REAL* dda_sca, const REAL* dda_abs) {
#ifdef IFDDA_WITH_SCATTNLAY
    if (*layers <= 0) return;
    try {
        const std::vector<double> layer_radii(radii, radii + *layers);
        const std::vector<std::complex<double>> epsilon(permittivities, permittivities + *layers);
        const auto result = ifdda_scattnlay::cross_sections(*wavelength, layer_radii, epsilon);
        const char* labels[] = {"Cext", "Csca", "Cabs"};
        const double references[] = {result.extinction, result.scattering, result.absorption};
        const double dda[] = {*dda_ext, *dda_sca, *dda_abs};
        std::printf(" Mie multilayer (Scattnlay): %d layer(s)\n", *layers);
        for (int i = 0; i < 3; ++i) {
            std::printf(" %s (Scattnlay) = %26.16E m2\n", labels[i], references[i]);
            if (references[i] > 0) {
                std::printf(" %s DDA error vs Scattnlay (%%) = %+.8g\n",
                            labels[i], 100 * (dda[i] - references[i]) / references[i]);
            }
        }
    } catch (const std::exception& error) {
        std::printf(" Mie multilayer (Scattnlay) unavailable: %s.\n", error.what());
    }
#else
    std::puts(" Mie multilayer (Scattnlay) unavailable in this build.");
#endif
    std::fflush(stdout);
}

#if defined(IFDDA_LINK_MISHCHENKO)
extern "C" int ifdda_mishchenko_v1(int, double, double, double, double, double, double,
    double, double, double, double, double*, int*, int*, double*);
#endif

#if defined(_WIN32) || defined(IFDDA_LINK_MISHCHENKO)
namespace {
#if defined(_WIN32)
using ReferenceFunction = int (*)(int, double, double, double, double, double, double,
    double, double, double, double, double*, int*, int*, double*);
#endif
using Vector = std::array<double,3>;
constexpr double reference_pi = 3.14159265358979323846;

double dot(const Vector& a, const Vector& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

Vector cross(const Vector& a, const Vector& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}

std::complex<double> project(const Vector& basis, const COMPLEX* field) {
    return basis[0]*std::complex<double>(field[0]) + basis[1]*std::complex<double>(field[1]) +
           basis[2]*std::complex<double>(field[2]);
}

#if defined(_WIN32)
ReferenceFunction load_reference() {
    wchar_t path[32768];
    const DWORD length = GetModuleFileNameW(nullptr,path,static_cast<DWORD>(sizeof(path)/sizeof(path[0])));
    if (!length || length >= sizeof(path)/sizeof(path[0])) return nullptr;
    std::wstring dll(path,length);
    dll.resize(dll.find_last_of(L"\\/")+1);
    dll += L"ifdda_mishchenko.dll";
    const HMODULE module = LoadLibraryExW(dll.c_str(),nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module) return nullptr;
    const auto function = reinterpret_cast<ReferenceFunction>(GetProcAddress(module,"ifdda_mishchenko_v1"));
    if (!function) FreeLibrary(module);
    return function;
}
#endif
} // namespace
#endif

extern "C" void ifdda_report_tmatrix_(const int* shape, const REAL* dimensions,
    const REAL* rotation, const REAL* wavelength, const COMPLEX* epsilon,
    const REAL* theta, const REAL* phi, const COMPLEX* incident,
    const REAL* dda_ext, const REAL* dda_sca, const REAL* dda_abs) {
#if defined(_WIN32) || defined(IFDDA_LINK_MISHCHENKO)
    if (!std::isfinite(*wavelength) || *wavelength <= 0) return;
    const Vector d{dimensions[0]/double(*wavelength), dimensions[1]/double(*wavelength),
                   dimensions[2]/double(*wavelength)};
    if (!(d[0] > 0 && d[1] > 0)) return;
    int symmetry_axis = 2;
    double transverse = d[0], longitudinal = d[1]/2;
    if (*shape == -1) {
        if (!(d[2] > 0)) return;
        const auto equal = [](double a, double b) {
            return std::abs(a-b) <= 4*std::numeric_limits<REAL>::epsilon()*std::max(a,b);
        };
        const bool xy = equal(d[0],d[1]);
        const bool xz = equal(d[0],d[2]);
        const bool yz = equal(d[1],d[2]);
        if (int(xy) + int(xz) + int(yz) != 1) return;
        symmetry_axis = xy ? 2 : (xz ? 1 : 0);
        longitudinal = d[symmetry_axis];
        transverse = (d[(symmetry_axis+1)%3]+d[(symmetry_axis+2)%3])/2;
    } else if (*shape != -2) {
        return;
    }
    const bool cylinder = *shape == -2;
    const double ratio = transverse/longitudinal;
    // Equal-volume radius and distance from the center to the farthest point.
    const double radius = std::cbrt((cylinder ? 1.5 : 1.0)*transverse*transverse*longitudinal);
    const double longest = cylinder ? std::hypot(transverse,longitudinal) : std::max(transverse,longitudinal);

#if defined(_WIN32)
    static const auto reference = load_reference();
    if (!reference) {
        std::puts(" T-matrix (Mishchenko) unavailable: ifdda_mishchenko.dll could not be loaded beside the executable.");
        return;
    }
#else
    constexpr auto reference = &ifdda_mishchenko_v1;
#endif

    const double cp = std::cos(rotation[0]), sp = std::sin(rotation[0]);
    const double ct = std::cos(rotation[1]), st = std::sin(rotation[1]);
    const double cs = std::cos(rotation[2]), ss = std::sin(rotation[2]);
    // Columns of the body-to-laboratory rotation in object_all.cpp.
    const Vector axes[] = {
        {cs*cp-ss*ct*sp, ss*cp+cs*ct*sp, st*sp},
        {-cs*sp-ss*ct*cp, -ss*sp+cs*ct*cp, st*cp},
        {ss*st, -cs*st, ct}
    };
    const double t = *theta*reference_pi/180, p = *phi*reference_pi/180;
    const Vector k{std::sin(t)*std::cos(p),std::sin(t)*std::sin(p),std::cos(t)};
    const Vector& axis = axes[symmetry_axis];
    Vector s = cross(axis,k);
    const double s_norm = std::sqrt(dot(s,s));
    if (s_norm > 1.e-12) for (double& component : s) component /= s_norm;
    else s = {-std::sin(p),std::cos(p),0};
    const Vector p_basis = cross(s,k);
    const auto ep = project(p_basis,incident), es = project(s,incident);
    const double angle = std::atan2(s_norm,dot(axis,k))*180/reference_pi;
    const auto index = std::sqrt(std::complex<double>(*epsilon));
    double sections[3], variation = 0;
    int order = 0, quadrature = 0;
    // The original Fortran uses COMMON blocks shared across calls.
    static std::mutex reference_mutex;
    int status;
    {
        const std::lock_guard<std::mutex> lock(reference_mutex);
        status = reference(*shape,radius,ratio,longest,index.real(),index.imag(),angle,
            ep.real(),ep.imag(),es.real(),es.imag(),sections,&order,&quadrature,&variation);
    }
    if (status != 0) {
        std::printf(" T-matrix (Mishchenko) unavailable: %s.\n",
            status == 2 ? "cross sections did not converge" : "input outside the supported range");
        return;
    }

    const char* labels[] = {"Cext","Csca","Cabs"};
    const double dda[] = {*dda_ext,*dda_sca,*dda_abs};
    const double area = double(*wavelength)*double(*wavelength);
    std::printf(" T-matrix (Mishchenko): N=%d, surface quadrature=%d, relative variation=%.3g\n",
        order,quadrature,variation);
    for (int i=0; i<3; ++i) {
        const double value = sections[i]*area;
        std::printf(" %s (T-matrix) = %26.16E m2\n",labels[i],value);
        if (value > 0) std::printf(" %s DDA error vs T-matrix (%%) = %+.8g\n",
            labels[i],100*(dda[i]-value)/value);
    }
#endif
}
