#ifndef IFDDA_CUDA_MOD_PRECISION_H
#define IFDDA_CUDA_MOD_PRECISION_H

#include <complex>
#include <cstdint>
#include <limits>

#ifdef USE_SIMP
using wp = float;
using ip = std::int32_t;
using precision_complex = std::complex<float>;
constexpr wp zero = 0.0f;
constexpr wp one = 1.0f;
constexpr wp two = 2.0f;
constexpr wp pi = 3.14159265358979323846f;
#else
using wp = double;
using ip = std::int32_t;
using precision_complex = std::complex<double>;
constexpr wp zero = 0.0;
constexpr wp one = 1.0;
constexpr wp two = 2.0;
constexpr wp pi = 3.14159265358979323846;
#endif

constexpr wp twopi = two * pi;
constexpr wp c = static_cast<wp>(299792458.0);
constexpr wp quatpieps0 = one / (c * c * static_cast<wp>(1.0e-7));
constexpr wp degtorad = pi / static_cast<wp>(180.0);
constexpr wp radtodeg = static_cast<wp>(180.0) / pi;
constexpr wp nmtom = static_cast<wp>(1.0e-9);
constexpr wp nmtosurf = one;
constexpr precision_complex icomp = precision_complex(zero, one);
constexpr precision_complex ZZERO = precision_complex(zero, zero);
constexpr precision_complex ZONE = precision_complex(one, zero);

#ifdef USE_SIMP
#define IFDDA_FMT_REAL "%22.15E"
#else
#define IFDDA_FMT_REAL "%22.15E"
#endif

#endif
