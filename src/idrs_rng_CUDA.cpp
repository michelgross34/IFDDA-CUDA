#include "cdmlib_cpp_abi.hpp"
#include "cuda_interfaces.h"
#include "cuda_utils.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>





//-----------------------------------------------------------
extern "C" void idrs_fill_random_(const int *ndim, const int *ns,
                                   CUCOMPLEX *values)
{
    if (ndim == nullptr || ns == nullptr || values == nullptr ||
        *ndim <= 0 || *ns <= 0)
    {
        std::fprintf(stderr, "IDRS random initialization: invalid dimensions\n");
        std::abort();
    }

    int seed = 0;
    char seed_text[33] = {};
    const char *seed_environment = std::getenv("IFDDA_IDRS_SEED");
    const bool has_seed = seed_environment != nullptr &&
        std::strlen(seed_environment) <= 32;

    if (has_seed)
    {
        std::strncpy(seed_text, seed_environment, sizeof(seed_text) - 1);
        char *end = nullptr;
        errno = 0;
        const long parsed = std::strtol(seed_text, &end, 10);
        while (end != nullptr && (*end == ' ' || *end == '\t' ||
               *end == '\n' || *end == '\r' || *end == '\f' || *end == '\v'))
        {
            ++end;
        }
        if (errno == ERANGE || end == seed_text || end == nullptr || *end != '\0' ||
            parsed < INT_MIN || parsed > INT_MAX)
        {
            std::fprintf(stderr,
                "Invalid IFDDA_IDRS_SEED: expected a decimal integer\n");
            std::abort();
        }
        seed = static_cast<int>(parsed);
        random_bridge_init(seed);
    }
    else
    {
        random_bridge_reset();
    }

    const size_t count = static_cast<size_t>(*ndim) * static_cast<size_t>(*ns);
    for (size_t index = 0; index < count; ++index)
    {
        const REAL alea_imag = random_bridge_next();
        const REAL alea_real = random_bridge_next();
        values[index] = cuda_utils::device_value(
            cuda_utils::HostComplex(alea_real, alea_imag));
    }
}
