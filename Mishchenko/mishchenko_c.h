#ifndef IFDDA_MISHCHENKO_C_H
#define IFDDA_MISHCHENKO_C_H

#ifdef _WIN32
#define IFDDA_MISHCHENKO_API __declspec(dllimport)
#else
#define IFDDA_MISHCHENKO_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum ifdda_mishchenko_shape {
    IFDDA_MISHCHENKO_SPHEROID = -1,
    IFDDA_MISHCHENKO_CYLINDER = -2,
    IFDDA_MISHCHENKO_CAPSULE = -4
};

enum ifdda_mishchenko_status {
    IFDDA_MISHCHENKO_OK = 0,
    IFDDA_MISHCHENKO_INVALID_INPUT = 1,
    IFDDA_MISHCHENKO_NO_CONVERGENCE = 2,
    IFDDA_MISHCHENKO_CUDA_ERROR = 3
};

/* 0 = serial CPU, 1 = OpenMP CPU, 2 = OpenMP + CUDA FP64. */
IFDDA_MISHCHENKO_API int ifdda_mishchenko_backend(void);

/* Length inputs are in wavelengths. Section outputs are in wavelength squared.
 * The capsule ratio is straight length / diameter, including zero for a sphere.
 * The capsule longest is radius + half the straight length.
 * theta_deg is measured from the symmetry axis to the incident wave vector.
 * ep and es are the complex E_theta and E_phi components in the particle frame.
 * Caller supplies all output pointers. Calls to the same library must be serialized. */
IFDDA_MISHCHENKO_API int ifdda_mishchenko_v1(
    int shape, double volume_radius, double ratio, double longest,
    double mr, double mi, double theta_deg,
    double ep_real, double ep_imag, double es_real, double es_imag,
    double sections[3], int* order, int* quadrature, double* variation);

#ifdef __cplusplus
}
#endif

#endif
