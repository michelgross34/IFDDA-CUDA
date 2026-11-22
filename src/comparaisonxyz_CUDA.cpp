#include "kernel.h"

//-----------------------------------------------------------
extern "C" int comparaison_(
    const CU_REAL *x,
    const CU_REAL *y,
    const CU_REAL *z,
    const CU_REAL *x0,
    const CU_REAL *y0,
    const CU_REAL *z0,
    const CU_REAL *lambda)
{
    if (!x || !y || !z || !x0 || !y0 || !z0 || !lambda)
    {
        return 0;
    }

    const CU_REAL tolerance = *lambda / CU_REAL(1000);
    return *x >= *x0 - tolerance && *x <= *x0 + tolerance &&
                   *y >= *y0 - tolerance && *y <= *y0 + tolerance &&
                   *z >= *z0 - tolerance && *z <= *z0 + tolerance
               ? 1
               : 0;
}
