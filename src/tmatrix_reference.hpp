#ifndef IFDDA_TMATRIX_REFERENCE_HPP
#define IFDDA_TMATRIX_REFERENCE_HPP

#include "cdmlib_cpp_precision.hpp"

extern "C" void ifdda_report_tmatrix_(const int* shape, const REAL* dimensions,
    const REAL* rotation, const REAL* wavelength, const COMPLEX* epsilon,
    const REAL* theta, const REAL* phi, const COMPLEX* incident,
    const REAL* dda_ext, const REAL* dda_sca, const REAL* dda_abs);

#endif
