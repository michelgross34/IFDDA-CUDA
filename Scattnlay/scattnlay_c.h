#ifndef IFDDA_SCATTNLAY_C_H
#define IFDDA_SCATTNLAY_C_H

#include <stddef.h>

#if defined(_WIN32)
#  if defined(IFDDA_SCATTNLAY_EXPORTS)
#    define IFDDA_SCATTNLAY_API __declspec(dllexport)
#  else
#    define IFDDA_SCATTNLAY_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define IFDDA_SCATTNLAY_API __attribute__((visibility("default")))
#else
#  define IFDDA_SCATTNLAY_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum ifdda_scattnlay_status {
    IFDDA_SCATTNLAY_OK = 0,
    IFDDA_SCATTNLAY_INVALID_INPUT = 1,
    IFDDA_SCATTNLAY_CALCULATION_FAILED = 2
};

/* Lengths in metres. Input arrays have layer_count elements, core first.
 * sections_m2 receives Cext, Csca, Cabs, or zeros on failure.
 * error_message may be NULL. A supplied buffer of positive capacity is
 * NUL-terminated. Input and output buffers must not overlap. */
IFDDA_SCATTNLAY_API int ifdda_scattnlay_sections_eps_v1(
    int layer_count, double wavelength_m, const double* radii_m,
    const double* eps_real, const double* eps_imag, double sections_m2[3],
    char* error_message, size_t error_capacity);

/* Same contract, using relative refractive indices n + i*k. */
IFDDA_SCATTNLAY_API int ifdda_scattnlay_sections_index_v1(
    int layer_count, double wavelength_m, const double* radii_m,
    const double* index_real, const double* index_imag, double sections_m2[3],
    char* error_message, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
