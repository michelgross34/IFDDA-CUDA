#include "cdmlib_cpp_abi.hpp"
#include "cuda_interfaces.h"

#include <cmath>

//-----------------------------------------------------------
extern "C" void deltakroutine_(const REAL *kxinc, const REAL *kyinc,
    const REAL *deltakx, const REAL *deltaky, const REAL *numaper,
    int *i, int *j)
{
    (void)numaper;
    *i = static_cast<int>(std::round(static_cast<double>(*kxinc / *deltakx)));
    *j = static_cast<int>(std::round(static_cast<double>(*kyinc / *deltaky)));
}
