#ifndef IFDDA_CDMLIB_CPP_PRECISION_HPP
#define IFDDA_CDMLIB_CPP_PRECISION_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef USE_SIMP
using REAL = float;
using COMPLEX = std::complex<float>;
constexpr REAL CDMLIB_EPSILON = std::numeric_limits<float>::epsilon();
constexpr REAL CDMLIB_MAX = std::numeric_limits<float>::max();
constexpr REAL CDMLIB_MIN = std::numeric_limits<float>::min();
#else
using REAL = double;
using COMPLEX = std::complex<double>;
constexpr REAL CDMLIB_EPSILON = std::numeric_limits<double>::epsilon();
constexpr REAL CDMLIB_MAX = std::numeric_limits<double>::max();
constexpr REAL CDMLIB_MIN = std::numeric_limits<double>::min();
#endif

//-------------------------------------------------
/* Keep mixed real/complex scalar expressions in the selected REAL precision.
   This provides the usual C99 complex arithmetic convenience for std::complex. */
template <class T,
          class U,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline std::complex<T> operator+(const std::complex<T> &lhs, U rhs) {
    return lhs + static_cast<T>(rhs);
}

//-------------------------------------------------
template <class U,
          class T,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline std::complex<T> operator+(U lhs, const std::complex<T> &rhs) {
    return static_cast<T>(lhs) + rhs;
}

//-------------------------------------------------
template <class T,
          class U,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline std::complex<T> operator-(const std::complex<T> &lhs, U rhs) {
    return lhs - static_cast<T>(rhs);
}

//-------------------------------------------------
template <class U,
          class T,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline std::complex<T> operator-(U lhs, const std::complex<T> &rhs) {
    return static_cast<T>(lhs) - rhs;
}

//-------------------------------------------------
template <class T,
          class U,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline std::complex<T> operator*(const std::complex<T> &lhs, U rhs) {
    return lhs * static_cast<T>(rhs);
}

//-------------------------------------------------
template <class U,
          class T,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline std::complex<T> operator*(U lhs, const std::complex<T> &rhs) {
    return static_cast<T>(lhs) * rhs;
}

//-------------------------------------------------
template <class T,
          class U,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline std::complex<T> operator/(const std::complex<T> &lhs, U rhs) {
    return lhs / static_cast<T>(rhs);
}

//-------------------------------------------------
template <class U,
          class T,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline std::complex<T> operator/(U lhs, const std::complex<T> &rhs) {
    return static_cast<T>(lhs) / rhs;
}

//-------------------------------------------------
template <class T,
          class U,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline bool operator==(const std::complex<T> &lhs, U rhs) {
    return lhs.real() == static_cast<T>(rhs) && lhs.imag() == T{};
}

//-------------------------------------------------
template <class U,
          class T,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline bool operator==(U lhs, const std::complex<T> &rhs) {
    return rhs == lhs;
}

//-------------------------------------------------
template <class T,
          class U,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline bool operator!=(const std::complex<T> &lhs, U rhs) {
    return !(lhs == rhs);
}

//-------------------------------------------------
template <class U,
          class T,
          std::enable_if_t<std::is_arithmetic_v<U> && !std::is_same_v<T, std::decay_t<U>>, int> = 0>
inline bool operator!=(U lhs, const std::complex<T> &rhs) {
    return !(rhs == lhs);
}

#endif
