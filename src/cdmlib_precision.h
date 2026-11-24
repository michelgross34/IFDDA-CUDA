#ifndef IFDDA_CDMLIB_PRECISION_C99_H
#define IFDDA_CDMLIB_PRECISION_C99_H

#include <complex.h>
#include <float.h>

/* Shared scalar types for every C99 translation unit. */
#ifdef USE_SIMP
#define REAL float
#define COMPLEX float complex
#define CDMLIB_EPSILON FLT_EPSILON
#define CDMLIB_MAX FLT_MAX
#define CDMLIB_MIN FLT_MIN
#else
#define REAL double
#define COMPLEX double complex
#define CDMLIB_EPSILON DBL_EPSILON
#define CDMLIB_MAX DBL_MAX
#define CDMLIB_MIN DBL_MIN
#endif

#endif
