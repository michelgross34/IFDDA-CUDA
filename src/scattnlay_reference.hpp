#ifndef IFDDA_SCATTNLAY_REFERENCE_HPP
#define IFDDA_SCATTNLAY_REFERENCE_HPP

#include "cdmlib_cpp_precision.hpp"

extern "C" void ifdda_report_scattnlay_(const int* layers, const REAL* radii,
    const COMPLEX* permittivities, const REAL* wavelength, const REAL* dda_ext,
    const REAL* dda_sca, const REAL* dda_abs);

#endif
