#include "kernel.h"
#include "cuda_interfaces.h"

#include <cstddef>
#include <cstdint>

// The CUDA-only target currently builds without USE_HDF5. Match the historical
// no-HDF5 behavior: these host-side output hooks leave the calculation intact.

//-----------------------------------------------------------
extern "C" void hdf5write_(const int *file_id, const char *dataset_name,
                             const CU_REAL *value, const int *dimensions,
                             size_t dataset_length) {
    (void)file_id;
    (void)dataset_name;
    (void)value;
    (void)dimensions;
    (void)dataset_length;
}

//-----------------------------------------------------------
extern "C" void hdf5write_int_(const int *file_id, const char *dataset_name,
                                 const int *value, const int *dimensions,
                                 size_t dataset_length) {
    (void)file_id;
    (void)dataset_name;
    (void)value;
    (void)dimensions;
    (void)dataset_length;
}

//-----------------------------------------------------------
extern "C" void hdf5write1d_(const int *file_id, const char *dataset_name,
                               const CU_REAL *data, const int *dimensions,
                               size_t dataset_length) {
    (void)file_id;
    (void)dataset_name;
    (void)data;
    (void)dimensions;
    (void)dataset_length;
}

//-----------------------------------------------------------
// HDF5 group IDs are 64-bit. A distinct entry point checks this ABI.
extern "C" void hdf5write1d_i8_(const std::int64_t *file_id, const char *dataset_name,
                                  const CU_REAL *data, const int *dimensions,
                                  size_t dataset_length) {
    (void)file_id;
    (void)dataset_name;
    (void)data;
    (void)dimensions;
    (void)dataset_length;
}

//-----------------------------------------------------------
extern "C" void writehdf5mic_(const COMPLEX *ex, const COMPLEX *ey,
                                const COMPLEX *ez, const int *nfft2d,
                                const int *imaxk0, COMPLEX *ediff,
                                const int *ntest, const char *name,
                                const std::int64_t *group_idmic, size_t name_length) {
    (void)ex;
    (void)ey;
    (void)ez;
    (void)nfft2d;
    (void)imaxk0;
    (void)ediff;
    (void)ntest;
    (void)name;
    (void)group_idmic;
    (void)name_length;
}
