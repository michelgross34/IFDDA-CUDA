#ifndef IFDDA_CUDA_POLARIZABILITY_INTERFACES_H
#define IFDDA_CUDA_POLARIZABILITY_INTERFACES_H

#include "kernel.h"

// Kept solely for the two historical C ABI entry points.
// All internal CPU calculations use COMPLEX; GPU buffers use CUCOMPLEX.
struct IfddaCudaPolarizabilityComplex {
    CU_REAL real;
    CU_REAL imag;
};

#endif // IFDDA_CUDA_POLARIZABILITY_INTERFACES_H
