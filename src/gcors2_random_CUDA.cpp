#include "cdmlib_cpp_abi.hpp"
#include "cuda_interfaces.h"

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <climits>



//-----------------------------------------------------------
extern "C" void gcors2_random_values(int n, REAL *random_imag,
                                     REAL *random_real)
{
    if (n < 0 || (n > 0 && (random_imag == nullptr || random_real == nullptr)))
    {
        std::fprintf(stderr, "GCORS2 random stream: invalid output buffers\n");
        std::abort();
    }

    // Match cdmlib/gcors2.f90: only reset the GNU Fortran random stream
    // when IFDDA_GCORS2_SEED is explicitly provided.  No reset in the
    // unseeded case, which retains the original shared RNG state.
    const char *seed_text = std::getenv("IFDDA_GCORS2_SEED");
    if (seed_text) {
        errno = 0;
        char *end = nullptr;
        const long parsed = std::strtol(seed_text, &end, 10);
        if (errno != 0 || end == seed_text || parsed < INT_MIN ||
            parsed > INT_MAX) {
            std::fprintf(stderr, "Invalid IFDDA_GCORS2_SEED: expected a decimal integer\n");
            std::abort();
        }
        while (*end == ' ' || *end == '\t') ++end;
        if (*end != '\0') {
            std::fprintf(stderr, "Invalid IFDDA_GCORS2_SEED: expected a decimal integer\n");
            std::abort();
        }
        random_bridge_init(static_cast<int>(parsed));
    }

    for (int index = 0; index < n; ++index)
    {
        random_imag[index] = random_bridge_next();
        random_real[index] = random_bridge_next();
    }
}
