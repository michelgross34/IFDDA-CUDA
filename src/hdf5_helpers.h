#include "cdmlib_precision.h"
#ifndef IFDDA_HDF5_HELPERS_C99_H
#define IFDDA_HDF5_HELPERS_C99_H

#include <stddef.h>
#include <stdint.h>

/* IDs are returned in this order: file, Option, Object, Far Field,
   Microscopy, Optical Force, Near Field, Version of the code. */
int cdmlib_hdf5_create_groups_c99(const char *filename, size_t filename_len, int64_t ids[8]);
void cdmlib_hdf5_close_groups_c99(int64_t ids[8]);
int cdmlib_hdf5_write1d_c99(
    int64_t group_id, const char *dataset, const REAL *data, int32_t count, int32_t memory_size);

#endif
