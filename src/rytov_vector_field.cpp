/* Vector Rytov regularization: parallel component exponential, transverse
 * scattered component linear. First-order expansion is Eref + DeltaE.
 * The previous componentwise exp(DeltaE_j/Eref_j) was singular when any
 * reference component was almost zero, even for a strong vector reference. */
#include "cdmlib_cpp_abi.hpp"

#include <math.h>
#include <stddef.h>

#ifdef USE_SIMP
#define RYTOV_ABS std::abs
#define RYTOV_EXP std::exp
#define RYTOV_REAL std::real
#define RYTOV_IMAG std::imag
#define RYTOV_LOG std::log
#define RYTOV_SQRT std::sqrt
#else
#define RYTOV_ABS std::abs
#define RYTOV_EXP std::exp
#define RYTOV_REAL std::real
#define RYTOV_IMAG std::imag
#define RYTOV_LOG std::log
#define RYTOV_SQRT std::sqrt
#endif

//-------------------------------------------------
extern "C" int cdmlib_rytov_vector_field_c99(const COMPLEX *eref,
                                             const COMPLEX *delta,
                                             COMPLEX *result,
                                             int nb) {
    for (int n = 0; n < nb; ++n) {
        const int k = 3 * n;
        REAL en2 = (REAL)0;
        for (int j = 0; j < 3; ++j) {
            const REAL mag = RYTOV_ABS(eref[k + j]);
            en2 += mag * mag;
        }
        if (!isfinite(en2))
            return 1;
        if (en2 <= CDMLIB_MIN) {
            for (int j = 0; j < 3; ++j)
                result[k + j] = eref[k + j] + delta[k + j];
        } else {
            COMPLEX q = (COMPLEX)0;
            for (int j = 0; j < 3; ++j)
                q += std::conj(eref[k + j]) * delta[k + j];
            q /= en2;
            if (!isfinite(RYTOV_REAL(q)) || !isfinite(RYTOV_IMAG(q)))
                return 1;
            /* Downstream field magnitudes are squared.  If the Rytov
             * exponential would exceed sqrt(max), use the finite first-order
             * Born limit and retain the scattered correction rather than
             * generating an enormous field whose later norm becomes Inf. */
            const REAL safe_limit = RYTOV_LOG(RYTOV_SQRT((REAL)CDMLIB_MAX)) -
                                    RYTOV_LOG(fmax((REAL)1, RYTOV_SQRT(en2))) - (REAL)2;
            if (RYTOV_REAL(q) > safe_limit) {
                for (int j = 0; j < 3; ++j)
                    result[k + j] = eref[k + j] + delta[k + j];
            } else {
                const COMPLEX nonlinear = RYTOV_EXP(q) - (COMPLEX)1 - q;
                for (int j = 0; j < 3; ++j)
                    result[k + j] = eref[k + j] + delta[k + j] + eref[k + j] * nonlinear;
            }
        }
        for (int j = 0; j < 3; ++j)
            if (!isfinite(RYTOV_REAL(result[k + j])) || !isfinite(RYTOV_IMAG(result[k + j])))
                return 1;
    }
    return 0;
}

//-------------------------------------------------
/* Fortran ABI entry point kept for callers that use the original procedure
 * name. This is a native C99 implementation; it does not forward to Fortran. */
extern "C" void rytov_vector_field_(
    const COMPLEX *eref, const COMPLEX *delta, COMPLEX *result, const int *nb, int *nstop) {
    if (nstop == NULL)
        return;
    *nstop = 1;
    if (nb == NULL || *nb < 0 || (eref == NULL && *nb != 0) || (delta == NULL && *nb != 0) ||
        (result == NULL && *nb != 0))
        return;
    *nstop = cdmlib_rytov_vector_field_c99(eref, delta, result, *nb);
}
