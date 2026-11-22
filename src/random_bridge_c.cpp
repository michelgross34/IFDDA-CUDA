#include "cdmlib_cpp_abi.hpp"
#include "cuda_interfaces.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define I (COMPLEX(0.0, 1.0))
#if defined(__MINGW32__) || defined(__MINGW64__)
#include "gfortran_rng_cpp.hpp"
#else
#include <cstdint>
#include <chrono>

/* Keep the portable non-MinGW stream used by the C99 fallback. */
static uint64_t bridge_state = 1;
static uint64_t bridge_stream_state[2] = {1, 1};

static uint64_t bridge_splitmix64_next(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static REAL bridge_uniform(uint64_t *state)
{
    return (REAL)((bridge_splitmix64_next(state) >> 11) * 0x1.0p-53);
}
#endif

#if defined(__MINGW32__) || defined(__MINGW64__)
static int stream_size;
static int32_t *stream_seed[2];

static void seed_desc(ifdda_gfc_array_i4 *desc, int32_t *data)
{
    memset(desc, 0, sizeof(*desc));
    desc->base_addr = data;
    desc->offset = -1;
    desc->elem_len = sizeof(*data);
    desc->span = sizeof(*data);
    desc->dtype[4] = 1;
    desc->dtype[5] = 1;
    desc->dim[0].stride = 1;
    desc->dim[0].lower_bound = 1;
    desc->dim[0].upper_bound = stream_size;
}
#endif

extern "C" void random_bridge_init(int seed)
{
#if defined(__MINGW32__) || defined(__MINGW64__)
    int size = 0;
    _gfortran_random_seed_i4(&size, NULL, NULL);
    if (size <= 0) return;
    int32_t *values = (int32_t *)malloc((size_t)size * sizeof(*values));
    if (!values) return;
    for (int i = 0; i < size; ++i) values[i] = (int32_t)seed;
    ifdda_gfc_array_i4 desc;
    memset(&desc, 0, sizeof desc);
    desc.base_addr = values;
    desc.offset = -1;
    desc.elem_len = sizeof(*values);
    desc.span = sizeof(*values);
    desc.dtype[4] = 1; desc.dtype[5] = 1;
    desc.dim[0].stride = 1;
    desc.dim[0].lower_bound = 1;
    desc.dim[0].upper_bound = size;
    _gfortran_random_seed_i4(NULL, &desc, NULL);
    free(values);
#else
    bridge_state = (uint64_t)(uint32_t)seed;
#endif
}

extern "C" void random_bridge_reset(void)
{
#if defined(__MINGW32__) || defined(__MINGW64__)
    _gfortran_random_seed_i4(NULL, NULL, NULL);
#else
    const uint64_t ticks = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    bridge_state = ticks ^ static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(&bridge_state));
    if (bridge_state == 0) bridge_state = 1;
#endif
}

extern "C" REAL random_bridge_next(void)
{
#if defined(__MINGW32__) || defined(__MINGW64__)
    return ifdda_random_next_r8();
#else
    return bridge_uniform(&bridge_state);
#endif
}

extern "C" void random_bridge_stream_init(int channel, int seed)
{
#if defined(__MINGW32__) || defined(__MINGW64__)
    int size = 0;
    _gfortran_random_seed_i4(&size, NULL, NULL);
    if (size <= 0 || channel < 0 || channel > 1) return;
    if (size != stream_size) {
        for (int i = 0; i < 2; ++i) { free(stream_seed[i]); stream_seed[i] = NULL; }
        stream_size = size;
    }
    if (!stream_seed[channel]) stream_seed[channel] = (int32_t *)malloc((size_t)size * sizeof(int32_t));
    if (!stream_seed[channel]) return;
    for (int i = 0; i < size; ++i) stream_seed[channel][i] = (int32_t)seed;
    ifdda_gfc_array_i4 desc;
    seed_desc(&desc, stream_seed[channel]);
    _gfortran_random_seed_i4(NULL, &desc, NULL);
#else
    if (channel >= 0 && channel < 2)
        bridge_stream_state[channel] = (uint64_t)(uint32_t)seed;
#endif
}

extern "C" REAL random_bridge_stream_next(int channel)
{
#if defined(__MINGW32__) || defined(__MINGW64__)
    if (channel < 0 || channel > 1 || !stream_seed[channel] || stream_size <= 0) return 0.0;
    ifdda_gfc_array_i4 desc;
    seed_desc(&desc, stream_seed[channel]);
    _gfortran_random_seed_i4(NULL, &desc, NULL);
    REAL value = ifdda_random_next_r8();
    _gfortran_random_seed_i4(NULL, NULL, &desc);
    return value;
#else
    if (channel < 0 || channel > 1) return (REAL)0;
    return bridge_uniform(&bridge_stream_state[channel]);
#endif
}
