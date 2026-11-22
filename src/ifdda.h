#ifndef IFDDA_CUDA_IFDDA_H
#define IFDDA_CUDA_IFDDA_H

#include "cdmlib_cpp_precision.hpp"
#include "ifdda_micros_abi_typed.hpp"
#include "ifdda_solver_abi_typed.hpp"

#include <cstddef>

/*
 * C interfaces declared in CUDA/cdmlib.f90 under
 * IFDDA_CUDA_INVERSE_DRIVER.  The typed solver and microscopy declarations
 * are kept in the shared ABI headers above; these declarations cover the
 * orchestration helpers used directly by the translated entry point.
 */
extern "C" {
void ifdda_set_initial_guess_mode_all_cuda_(int mode);
void ifdda_reset_cuda_transfer_stats();
void ifdda_print_cuda_transfer_stats();
void ifdda_renormalized_born_all_cuda_(const COMPLEX* incident,
                                   const COMPLEX* polarizability,
                                   COMPLEX* local_field,
                                   COMPLEX* polarized_field,
                                   const int* nmax,
                                   const int* niso,
                                   const int* ndipoles,
                                   int* nstop);
void ifdda_born_all_cuda_(const COMPLEX* incident,
                      const COMPLEX* epsilon,
                      const COMPLEX* polarizability,
                      COMPLEX* local_field,
                      COMPLEX* polarized_field,
                      const int* nmax,
                      const int* niso,
                      const int* ndipoles,
                      const REAL* aretecube,
                      const REAL* k0,
                      int* nstop);
void ifdda_vector_copy_all_cuda_(
    const COMPLEX* input, COMPLEX* output, const int* count, const int* negate, int* nstop);
void ifdda_vector_zero_all_cuda_(COMPLEX* output, const int* count, int* nstop);
void ifdda_vector_negate_all_cuda_(COMPLEX* values, const int* count, int* nstop);
void ifdda_rytov_macro_prepare_all_cuda_(const COMPLEX* incident,
                                     const COMPLEX* epsilon,
                                     const COMPLEX* polarizability,
                                     const COMPLEX* background_epsilon,
                                     COMPLEX* macroscopic_local_field,
                                     COMPLEX* delta_field,
                                     COMPLEX* negative_polarized_field,
                                     const int* nmax,
                                     const int* niso,
                                     const int* ndipoles,
                                     const REAL* aretecube,
                                     const REAL* k0,
                                     int* nstop);
}

#endif
