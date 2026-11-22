#ifndef IFDDA_GFORTRAN_RNG_CPP_HPP
#define IFDDA_GFORTRAN_RNG_CPP_HPP

#include "cdmlib_cpp_abi.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>

/* Keep one interface for RNG callers. MinGW uses GNU Fortran's exact state;
 * MSVC routes the same calls through the portable bridge implementation. */
#if defined(__MINGW32__) || defined(__MINGW64__)
struct ifdda_gfc_dim1 {
    std::int64_t stride;
    std::int64_t lower_bound;
    std::int64_t upper_bound;
};

struct ifdda_gfc_array_i4 {
    void *base_addr;
    std::int64_t offset;
    std::uint64_t elem_len;
    unsigned char dtype[8];
    std::int64_t span;
    ifdda_gfc_dim1 dim[1];
};

extern "C" {
void _gfortran_random_seed_i4(int *size, ifdda_gfc_array_i4 *put, ifdda_gfc_array_i4 *get);
#ifdef USE_SIMP
void _gfortran_random_r4(float *value);
#else
void _gfortran_random_r8(double *value);
#endif
}
#else
extern "C" {
void random_bridge_init(int seed);
void random_bridge_reset(void);
REAL random_bridge_next(void);
}
#endif

//-------------------------------------------------
static inline int ifdda_random_seed_all_i4(std::int32_t value) {
#if defined(__MINGW32__) || defined(__MINGW64__)
    int size = 0;
    _gfortran_random_seed_i4(&size, nullptr, nullptr);
    if (size <= 0)
        return -1;
    auto *seed = static_cast<std::int32_t *>(
        std::malloc(static_cast<std::size_t>(size) * sizeof(std::int32_t)));
    if (seed == nullptr)
        return -1;
    for (int i = 0; i < size; ++i)
        seed[i] = value;
    ifdda_gfc_array_i4 desc{};
    desc.base_addr = seed;
    desc.offset = -1;
    desc.elem_len = sizeof(*seed);
    desc.span = sizeof(*seed);
    desc.dtype[4] = 1;
    desc.dtype[5] = 1;
    desc.dim[0].stride = 1;
    desc.dim[0].lower_bound = 1;
    desc.dim[0].upper_bound = size;
    _gfortran_random_seed_i4(nullptr, &desc, nullptr);
    std::free(seed);
    return 0;
#else
    random_bridge_init(static_cast<int>(value));
    return 0;
#endif
}

//-------------------------------------------------
static inline REAL ifdda_random_next() {
#if defined(__MINGW32__) || defined(__MINGW64__)
    REAL value;
#ifdef USE_SIMP
    _gfortran_random_r4(&value);
#else
    _gfortran_random_r8(&value);
#endif
    return value;
#else
    return random_bridge_next();
#endif
}

//-------------------------------------------------
static inline REAL ifdda_random_next_r8() {
    return ifdda_random_next();
}

//-------------------------------------------------
static inline void ifdda_random_seed_reset() {
#if defined(__MINGW32__) || defined(__MINGW64__)
    _gfortran_random_seed_i4(nullptr, nullptr, nullptr);
#else
    random_bridge_reset();
#endif
}

#endif
