#ifndef PRODUITMATVECTFFT_KERNEL_H
#define PRODUITMATVECTFFT_KERNEL_H

#ifndef PRODUITMATVECTFFT_SINGLE_PRECISION
#define PRODUITMATVECTFFT_SINGLE_PRECISION 0
#endif

#include <cuda_runtime.h>
#include <cufft.h>
#include <stddef.h>
#ifdef __cplusplus
#include "cdmlib_cpp_precision.hpp"
#endif

typedef cudaError_t (*ifdda_host_download_callback)(void* host_destination,
                                                    const void* device_source,
                                                    size_t bytes);

#if defined(_WIN32) && defined(__CUDACC__)
// NVCC builds and exports the wrapper DLL; regular C++ imports its wrappers.
#define PRODUITMATVECTFFT_KERNEL_API __declspec(dllexport)
#elif defined(_WIN32)
#define PRODUITMATVECTFFT_KERNEL_API __declspec(dllimport)
#else
// NVCC may otherwise mark some host-callable CUDA wrappers as ELF HIDDEN.
// They must remain exported from libifdda_cuda_kernels_{double,float}.so.
#define PRODUITMATVECTFFT_KERNEL_API __attribute__((visibility("default")))
#endif

// Route optional DLL diagnostics through the executable's centralized
// cuda_data host/device transfer layer.
extern "C" PRODUITMATVECTFFT_KERNEL_API void
ifdda_set_host_download_callback(ifdda_host_download_callback callback);

#if PRODUITMATVECTFFT_SINGLE_PRECISION
typedef cuComplex CUCOMPLEX;
typedef float CU_REAL;
#define CUFFT_TRANSFORM CUFFT_C2C
#define CUFFT_EXECUTE cufftExecC2C
#define CUFFT_ZERO make_cuFloatComplex(0.0f, 0.0f)
#define CUFFT_REAL(value) make_cuFloatComplex((value), 0.0f)
#define CUFFT_ADD(a, b) cuCaddf((a), (b))
#define CUFFT_SUB(a, b) cuCsubf((a), (b))
#define CUFFT_MUL(a, b) cuCmulf((a), (b))
#define CUFFT_DIV(a, b) cuCdivf((a), (b))
#define CUFFT_CONJ(a) cuConjf((a))
#else
typedef cuDoubleComplex CUCOMPLEX;
typedef double CU_REAL;
#define CUFFT_TRANSFORM CUFFT_Z2Z
#define CUFFT_EXECUTE cufftExecZ2Z
#define CUFFT_ZERO make_cuDoubleComplex(0.0, 0.0)
#define CUFFT_REAL(value) make_cuDoubleComplex((value), 0.0)
#define CUFFT_ADD(a, b) cuCadd((a), (b))
#define CUFFT_SUB(a, b) cuCsub((a), (b))
#define CUFFT_MUL(a, b) cuCmul((a), (b))
#define CUFFT_DIV(a, b) cuCdiv((a), (b))
#define CUFFT_CONJ(a) cuConj((a))
#endif

// Six independent device allocations. Never derive a component with pointer
// arithmetic: each component must fit within one Windows CUDA allocation.
struct GreenTensorDevicePointers {
    CUCOMPLEX* data[6]{};
    CUCOMPLEX* operator[](size_t i) const { return data[i]; }
    explicit operator bool() const {
        for (int i = 0; i < 6; ++i)
            if (!data[i]) return false;
        return true;
    }
};
struct GreenTensorDeviceView {
    const CUCOMPLEX* data[6]{};
    const CUCOMPLEX* operator[](size_t i) const { return data[i]; }
    explicit operator bool() const {
        for (int i = 0; i < 6; ++i)
            if (!data[i]) return false;
        return true;
    }
};
inline GreenTensorDeviceView green_view(const GreenTensorDevicePointers& writable) {
    GreenTensorDeviceView result{};
    for (int i = 0; i < 6; ++i) result.data[i] = writable.data[i];
    return result;
}

// tabdip in CPU memory is int*. Its GPU copy uses CU_INT* so a
// host pointer cannot accidentally be passed to a CUDA map kernel.
typedef struct { int dat; } CU_INT;
// Device map and scalar geometry metadata for regular incident-field kernels.
// tabdip is full-grid -> one-based compact slot; no CPU complex pointers occur.
struct IfddaIncidentGrid {
    const CU_INT* tabdip;
    int nx, ny, grid_count, dipole_count;
    int anchor_ix, anchor_iy, anchor_iz;
    REAL anchor_x, anchor_y, anchor_z, cell_size;
};
#ifdef __cplusplus
static_assert(sizeof(REAL) == sizeof(CU_REAL),
              "Host REAL and device CU_REAL must use the same precision");
#endif
#ifdef __cplusplus
static_assert(sizeof(CU_INT) == sizeof(int), "CU_INT must match a Fortran INTEGER");
static_assert(offsetof(CU_INT, dat) == 0, "CU_INT must have no leading padding");
#endif

// Host-callable wrappers. The .cpp/.cu caller only sees ordinary C++ calls;
// kernel launch syntax stays in kernel.cu.
#ifdef __cplusplus
extern "C" {
#endif

PRODUITMATVECTFFT_KERNEL_API cudaError_t zero_vectors_wrapper(CUCOMPLEX* vectors, int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t mie_efficiencies_wrapper(CUCOMPLEX* recurrence,
                                                                  CU_REAL relative_index_real,
                                                                  CU_REAL relative_index_imag,
                                                                  CU_REAL x,
                                                                  CU_REAL radius,
                                                                  CU_REAL* outputs,
                                                                  int max_order);

PRODUITMATVECTFFT_KERNEL_API cudaError_t cube_geometry_wrapper(CU_REAL* xs,
                                                               CU_REAL* ys,
                                                               CU_REAL* zs,
                                                               CU_INT* dipole_map,
                                                               int subdivisions,
                                                               CU_REAL side,
                                                               CU_REAL cell_size,
                                                               CU_REAL center_x,
                                                               CU_REAL center_y,
                                                               CU_REAL center_z);
// Occupancy-only cube generator; full-grid coordinates are reconstructed on CPU
// solely for the existing Fortran-compatible output arrays.
PRODUITMATVECTFFT_KERNEL_API cudaError_t cube_dipole_map_wrapper(
    CU_INT* dipole_map, int subdivisions);
PRODUITMATVECTFFT_KERNEL_API cudaError_t full_grid_dipole_map_wrapper(
    CU_INT* dipole_map, int grid_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t cuboid_dipole_map_wrapper(
    CU_INT* dipole_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL side_x, CU_REAL side_y, CU_REAL side_z);
PRODUITMATVECTFFT_KERNEL_API cudaError_t multi_sphere_dipole_map_wrapper(
    CU_INT* sphere_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL x_min, CU_REAL y_min, CU_REAL z_min,
    const CU_REAL* sphere_parameters, int sphere_count, int fixed_sphere_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t concentric_sphere_dipole_map_wrapper(
    CU_INT* layer_map, int subdivisions, CU_REAL cell_size,
    CU_REAL center, const CU_REAL* radii, int sphere_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t widefield_build_grid_wrapper(CU_REAL* grid_x,
                                                                      CU_REAL* grid_y,
                                                                      CU_REAL* grid_z,
                                                                      int nx,
                                                                      int ny,
                                                                      int nz,
                                                                      int nxmp,
                                                                      int nymp,
                                                                      int nzmp,
                                                                      CU_REAL xmin,
                                                                      CU_REAL ymin,
                                                                      CU_REAL zmin,
                                                                      CU_REAL cell_size);
PRODUITMATVECTFFT_KERNEL_API cudaError_t widefield_map_dipoles_wrapper(const CU_REAL* dipole_x,
                                                                       const CU_REAL* dipole_y,
                                                                       const CU_REAL* dipole_z,
                                                                       const CU_REAL* grid_x,
                                                                       const CU_REAL* grid_y,
                                                                       const CU_REAL* grid_z,
                                                                       CU_INT* dipole_map,
                                                                       int dipole_count,
                                                                       int nx,
                                                                       int ny,
                                                                       int nz,
                                                                       CU_REAL cell_size,
                                                                       CU_REAL tolerance);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
widefield_scatter_dipoles_wrapper(const CU_INT* grid_to_dipole,
                                  int grid_count,
                                  int dipole_count,
                                  int isotropy,
                                  int preserve_incident_field,
                                  const CUCOMPLEX* dipole_scattered,
                                  const CUCOMPLEX* dipole_incident,
                                  const CUCOMPLEX* dipole_epsilon,
                                  CUCOMPLEX* xi,
                                  CUCOMPLEX* xr,
                                  CUCOMPLEX* polarization);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
widefield_pack_dipole_epsilon_wrapper(const CUCOMPLEX* fortran_epsilon,
                                      CUCOMPLEX* packed_epsilon,
                                      int dipole_count,
                                      int nmax,
                                      int isotropy);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
widefield_restore_material_local_field_wrapper(const CU_INT* grid_to_dipole,
                                               int grid_count,
                                               int dipole_count,
                                               int isotropy,
                                               const CUCOMPLEX* dipole_epsilon,
                                               const CUCOMPLEX* dipole_local_field,
                                               CUCOMPLEX* grid_field);
PRODUITMATVECTFFT_KERNEL_API cudaError_t widefield_incident_magnitude_wrapper(
    const CUCOMPLEX* grid_field, CU_REAL* magnitude, int grid_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t dipole_inside_find_wrapper(const CU_REAL* xs,
                                                                    const CU_REAL* ys,
                                                                    const CU_REAL* zs,
                                                                    int count,
                                                                    CU_REAL xdip,
                                                                    CU_REAL ydip,
                                                                    CU_REAL zdip,
                                                                    CU_REAL cell_size,
                                                                    CU_INT* first_match);
PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_mask_extrema_wrapper(const CU_REAL* xs,
                                                                         const CU_REAL* ys,
                                                                         int count,
                                                                         CU_REAL* extrema);
PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_mask_fill_wrapper(CUCOMPLEX* mask,
                                                                      CU_INT* selected_count,
                                                                      int imaxinc,
                                                                      int sizemask,
                                                                      int mask_type,
                                                                      CU_REAL deltak,
                                                                      CU_REAL numaperk,
                                                                      CU_REAL numaperk2,
                                                                      CU_REAL kcnax,
                                                                      CU_REAL kcnay);
PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_copy_scale_wrapper(const CUCOMPLEX* source,
                                                                       CUCOMPLEX* destination,
                                                                       size_t count,
                                                                       CU_REAL scale);
PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_axes_wrapper(
    CU_REAL* kxy, CU_REAL* xy, int count, int half, CU_REAL delta_k, CU_REAL gross, CU_REAL k0);
PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_identity_dipole_map_wrapper(CU_INT* dipole_map,
                                                                                int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_project_scalar_rhs_wrapper(
    const CUCOMPLEX* incident, CUCOMPLEX* rhs, int count, CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
microscope_expand_scalar_fields_wrapper(CUCOMPLEX* xi,
                                        const CUCOMPLEX* polarization,
                                        const CUCOMPLEX* scalar_local,
                                        CUCOMPLEX* scattered,
                                        CUCOMPLEX* local,
                                        int count,
                                        CUCOMPLEX ux,
                                        CUCOMPLEX uy,
                                        CUCOMPLEX uz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t abcchi_profile_wrapper(CUCOMPLEX* epsilon_output,
                                                                CUCOMPLEX* contrast_output,
                                                                CUCOMPLEX epsilon_value,
                                                                CUCOMPLEX epsilon_background,
                                                                CU_REAL contrast_scale,
                                                                int profile_type,
                                                                int i,
                                                                int j,
                                                                int k,
                                                                int nx,
                                                                int ny,
                                                                int nz,
                                                                int neps,
                                                                int isotropy);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
pml_sphere_material_profile_wrapper(const CU_REAL* xs,
                                    const CU_REAL* ys,
                                    const CU_REAL* zs,
                                    const CU_REAL* material,
                                    CUCOMPLEX* epsilon_output,
                                    CUCOMPLEX* contrast_output,
                                    int count,
                                    int nx,
                                    int ny,
                                    int nz,
                                    int material_extent,
                                    CU_REAL radius,
                                    CU_REAL xmin,
                                    CU_REAL spacing,
                                    CU_REAL epsilon_imag,
                                    CUCOMPLEX epsilon_value,
                                    CUCOMPLEX epsilon_background,
                                    CU_REAL contrast_scale,
                                    int profile_type,
                                    int profile_extent);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
pml_cuboid_material_profile_wrapper(const CU_REAL* xs,
                                    const CU_REAL* ys,
                                    const CU_REAL* zs,
                                    const CU_REAL* material,
                                    CUCOMPLEX* epsilon_output,
                                    CUCOMPLEX* contrast_output,
                                    int nx,
                                    int ny,
                                    int nz,
                                    int material_nx,
                                    int material_ny,
                                    int material_nz,
                                    CU_REAL side_x,
                                    CU_REAL side_y,
                                    CU_REAL side_z,
                                    CU_REAL xmin,
                                    CU_REAL ymin,
                                    CU_REAL zmin,
                                    CU_REAL spacing,
                                    CU_REAL epsilon_imag,
                                    CUCOMPLEX epsilon_value,
                                    CUCOMPLEX epsilon_background,
                                    CU_REAL contrast_scale,
                                    int profile_type,
                                    int profile_extent);

// dipole_map may be nullptr to initialize material tensors only.
PRODUITMATVECTFFT_KERNEL_API cudaError_t initialize_polarizability_arrays_wrapper(
    CU_INT* dipole_map, CUCOMPLEX* polarizability, CUCOMPLEX* epsilon, int dipole_count, int isotropy);
PRODUITMATVECTFFT_KERNEL_API cudaError_t sphere_material_from_cpu_tabdip_wrapper(
    const CU_INT* tabdip, const CU_INT* inside_map,
    CUCOMPLEX* polarizability, CUCOMPLEX* epsilon,
    const CUCOMPLEX* tensor_alpha, const CUCOMPLEX* tensor_epsilon,
    CUCOMPLEX scalar_alpha, CUCOMPLEX scalar_epsilon,
    int grid_count, int nmax, int compact_mesh, int anisotropic);
PRODUITMATVECTFFT_KERNEL_API cudaError_t labeled_material_from_cpu_tabdip_wrapper(
    const CU_INT* tabdip, const CU_INT* material_labels,
    CUCOMPLEX* polarizability, CUCOMPLEX* epsilon,
    const CUCOMPLEX* label_alpha, const CUCOMPLEX* label_epsilon,
    int grid_count, int nmax, int label_count, int anisotropic);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scatter_cell_material_from_cpu_tabdip_wrapper(
    const CU_INT* tabdip, const CU_INT* inside_map,
    const CUCOMPLEX* cell_epsilon, const CUCOMPLEX* cell_alpha,
    CUCOMPLEX* output_epsilon, CUCOMPLEX* output_alpha,
    int grid_count, int nmax);
PRODUITMATVECTFFT_KERNEL_API cudaError_t arbitrary_tensor_material_batch_wrapper(
    const CUCOMPLEX* cell_epsilon, CUCOMPLEX* output_epsilon,
    CUCOMPLEX* output_alpha, int count, int nmax, CUCOMPLEX epsilon0,
    CU_REAL radius, CU_REAL k0, int method);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scalar_polarizability_wrapper(CUCOMPLEX* result,
                                                                       CU_REAL* sldr,
                                                                       CU_REAL radius,
                                                                       CUCOMPLEX epsilon,
                                                                       CUCOMPLEX epsilon0,
                                                                       CU_REAL k0,
                                                                       int discretized,
                                                                       int method);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
scalar_polarizability_batch_wrapper(const CUCOMPLEX* epsilon,
                                    const CU_INT* inside,
                                    CUCOMPLEX* values,
                                    CU_REAL* sldr,
                                    int count,
                                    CU_REAL radius,
                                    CUCOMPLEX epsilon0,
                                    CU_REAL k0,
                                    int discretized,
                                    int method,
                                    CU_REAL initial_sldr);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
rerange_isotropic_polarizability_wrapper(const CUCOMPLEX* epsilon,
                                         CUCOMPLEX* polarizability,
                                         int count,
                                         CUCOMPLEX epsilon0,
                                         CU_REAL scale);

PRODUITMATVECTFFT_KERNEL_API cudaError_t material_interpolation_wrapper(
    CU_REAL* result, CU_REAL refractive_index, CU_REAL extinction_coefficient);
PRODUITMATVECTFFT_KERNEL_API cudaError_t tensor_polarizability_wrapper(const CUCOMPLEX* epsilon,
                                                                       CUCOMPLEX* result,
                                                                       CU_INT* status,
                                                                       CUCOMPLEX epsilon0,
                                                                       CU_REAL radius,
                                                                       CU_REAL k0,
                                                                       int discretized,
                                                                       int method,
                                                                       int invert_result);

PRODUITMATVECTFFT_KERNEL_API cudaError_t cuboid_geometry_wrapper(CU_REAL* xs,
                                                                 CU_REAL* ys,
                                                                 CU_REAL* zs,
                                                                 CU_INT* dipole_map,
                                                                 int nx,
                                                                 int ny,
                                                                 int nz,
                                                                 CU_REAL cell_size,
                                                                 CU_REAL side_x,
                                                                 CU_REAL side_y,
                                                                 CU_REAL side_z,
                                                                 CU_REAL center_x,
                                                                 CU_REAL center_y,
                                                                 CU_REAL center_z);

PRODUITMATVECTFFT_KERNEL_API cudaError_t rotated_cuboid_geometry_wrapper(CU_REAL* xs,
                                                                         CU_REAL* ys,
                                                                         CU_REAL* zs,
                                                                         CU_INT* inside_map,
                                                                         int nx,
                                                                         int ny,
                                                                         int nz,
                                                                         CU_REAL cell_size,
                                                                         CU_REAL extent_x,
                                                                         CU_REAL extent_y,
                                                                         CU_REAL extent_z,
                                                                         CU_REAL side_x,
                                                                         CU_REAL side_y,
                                                                         CU_REAL side_z,
                                                                         CU_REAL center_x,
                                                                         CU_REAL center_y,
                                                                         CU_REAL center_z,
                                                                         CU_REAL r0,
                                                                         CU_REAL r1,
                                                                         CU_REAL r2,
                                                                         CU_REAL r3,
                                                                         CU_REAL r4,
                                                                         CU_REAL r5,
                                                                         CU_REAL r6,
                                                                         CU_REAL r7,
                                                                         CU_REAL r8);
PRODUITMATVECTFFT_KERNEL_API cudaError_t rotated_cuboid_dipole_map_wrapper(
    CU_INT* dipole_map, int nx, int ny, int nz,
    CU_REAL cell_size, CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
    CU_REAL side_x, CU_REAL side_y, CU_REAL side_z,
    CU_REAL r0, CU_REAL r1, CU_REAL r2, CU_REAL r3, CU_REAL r4,
    CU_REAL r5, CU_REAL r6, CU_REAL r7, CU_REAL r8);

PRODUITMATVECTFFT_KERNEL_API cudaError_t sphere_geometry_wrapper(CU_REAL* xs,
                                                                 CU_REAL* ys,
                                                                 CU_REAL* zs,
                                                                 CU_INT* inside_map,
                                                                 int subdivisions,
                                                                 CU_REAL radius,
                                                                 CU_REAL cell_size,
                                                                 CU_REAL center_x,
                                                                 CU_REAL center_y,
                                                                 CU_REAL center_z);
// Occupancy-only sphere generator: coordinates are computed in each GPU thread.
PRODUITMATVECTFFT_KERNEL_API cudaError_t sphere_dipole_map_wrapper(
    CU_INT* dipole_map, int subdivisions, CU_REAL radius, CU_REAL cell_size);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
sphere_geometry_compaction_workspace_wrapper(int count, size_t* temporary_bytes);
PRODUITMATVECTFFT_KERNEL_API cudaError_t sphere_geometry_compact_wrapper(const CU_REAL* mesh_x,
                                                                         const CU_REAL* mesh_y,
                                                                         const CU_REAL* mesh_z,
                                                                         const CU_INT* inside_map,
                                                                         int count,
                                                                         CU_INT* prefix,
                                                                         CU_REAL* packed_x,
                                                                         CU_REAL* packed_y,
                                                                         CU_REAL* packed_z,
                                                                         CU_INT* packed_count,
                                                                         void* temporary,
                                                                         size_t temporary_bytes);

PRODUITMATVECTFFT_KERNEL_API cudaError_t ellipsoid_geometry_wrapper(CU_REAL* xs,
                                                                    CU_REAL* ys,
                                                                    CU_REAL* zs,
                                                                    CU_INT* inside_map,
                                                                    int nx,
                                                                    int ny,
                                                                    int nz,
                                                                    CU_REAL cell_size,
                                                                    CU_REAL extent_x,
                                                                    CU_REAL extent_y,
                                                                    CU_REAL extent_z,
                                                                    CU_REAL radius_a,
                                                                    CU_REAL radius_b,
                                                                    CU_REAL radius_c,
                                                                    CU_REAL m0,
                                                                    CU_REAL m1,
                                                                    CU_REAL m2,
                                                                    CU_REAL m3,
                                                                    CU_REAL m4,
                                                                    CU_REAL m5,
                                                                    CU_REAL m6,
                                                                    CU_REAL m7,
                                                                    CU_REAL m8);
PRODUITMATVECTFFT_KERNEL_API cudaError_t ellipsoid_dipole_map_wrapper(
    CU_INT* inside_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
    CU_REAL radius_a, CU_REAL radius_b, CU_REAL radius_c,
    CU_REAL m0, CU_REAL m1, CU_REAL m2, CU_REAL m3, CU_REAL m4,
    CU_REAL m5, CU_REAL m6, CU_REAL m7, CU_REAL m8);

PRODUITMATVECTFFT_KERNEL_API cudaError_t cylinder_geometry_wrapper(CU_REAL* xs,
                                                                   CU_REAL* ys,
                                                                   CU_REAL* zs,
                                                                   CU_INT* inside_map,
                                                                   int nx,
                                                                   int ny,
                                                                   int nz,
                                                                   CU_REAL cell_size,
                                                                   CU_REAL extent_x,
                                                                   CU_REAL extent_y,
                                                                   CU_REAL extent_z,
                                                                   CU_REAL radius,
                                                                   CU_REAL height,
                                                                   CU_REAL m0,
                                                                   CU_REAL m1,
                                                                   CU_REAL m2,
                                                                   CU_REAL m3,
                                                                   CU_REAL m4,
                                                                   CU_REAL m5,
                                                                   CU_REAL m6,
                                                                   CU_REAL m7,
                                                                   CU_REAL m8);
PRODUITMATVECTFFT_KERNEL_API cudaError_t cylinder_dipole_map_wrapper(
    CU_INT* inside_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
    CU_REAL radius, CU_REAL height,
    CU_REAL m0, CU_REAL m1, CU_REAL m2, CU_REAL m3, CU_REAL m4,
    CU_REAL m5, CU_REAL m6, CU_REAL m7, CU_REAL m8);

PRODUITMATVECTFFT_KERNEL_API cudaError_t
inhomogeneous_material_spectrum_wrapper(CUCOMPLEX* spectrum,
                                        const CU_REAL* phases,
                                        int nx,
                                        int ny,
                                        int nz,
                                        CU_REAL dkx,
                                        CU_REAL dky,
                                        CU_REAL dkz,
                                        CU_REAL coefficient1,
                                        CU_REAL coefficient2,
                                        CU_REAL coefficient3);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
inhomogeneous_material_finalize_wrapper(const CUCOMPLEX* spectrum,
                                        CU_REAL* real_values,
                                        CU_REAL* magnitude_squares,
                                        int count,
                                        CU_REAL epsilon_real,
                                        CU_REAL epsilon_imag);
PRODUITMATVECTFFT_KERNEL_API cudaError_t fill_real_ones_wrapper(CU_REAL* values, int count);

PRODUITMATVECTFFT_KERNEL_API cudaError_t
multi_sphere_geometry_wrapper(CU_REAL* xs,
                              CU_REAL* ys,
                              CU_REAL* zs,
                              CU_INT* sphere_map,
                              int nx,
                              int ny,
                              int nz,
                              CU_REAL cell_size,
                              CU_REAL x_min,
                              CU_REAL y_min,
                              CU_REAL z_min,
                              const CU_REAL* sphere_parameters,
                              int sphere_count,
                              int fixed_sphere_count);

PRODUITMATVECTFFT_KERNEL_API cudaError_t concentric_sphere_geometry_wrapper(CU_REAL* xs,
                                                                            CU_REAL* ys,
                                                                            CU_REAL* zs,
                                                                            CU_INT* layer_map,
                                                                            int subdivisions,
                                                                            CU_REAL cell_size,
                                                                            CU_REAL center,
                                                                            const CU_REAL* radii,
                                                                            int sphere_count);

PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_plane_wave_wrapper(const CU_REAL* xs,
                                                                     const CU_REAL* ys,
                                                                     const CU_REAL* zs,
                                                                     CUCOMPLEX* field,
                                                                     int count,
                                                                     REAL kx,
                                                                     REAL ky,
                                                                     COMPLEX kz,
                                                                     COMPLEX ex0,
                                                                     COMPLEX ey0,
                                                                     COMPLEX ez0);
// Full-grid index -> compact index is supplied by the geometry's tabdip.
// Coordinates are reconstructed in the kernel from an occupied anchor cell.
PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_plane_wave_grid_wrapper(
    const CU_INT* tabdip, CUCOMPLEX* field, int nx, int ny, int nz,
    int dipole_count, int anchor_ix, int anchor_iy, int anchor_iz,
    REAL anchor_x, REAL anchor_y, REAL anchor_z, REAL cell_size,
    REAL kx, REAL ky, COMPLEX kz,
    COMPLEX ex0, COMPLEX ey0, COMPLEX ez0);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
incident_plane_wave_derivative_wrapper(CUCOMPLEX* field_derivative,
                                       CU_REAL x,
                                       CU_REAL y,
                                       CU_REAL z,
                                       CU_REAL k0,
                                       CUCOMPLEX amplitude,
                                       CU_REAL ss,
                                       CU_REAL pp,
                                       CU_REAL theta_degrees,
                                       CU_REAL phi_degrees,
                                       int derivative_selector,
                                       int circular);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
incident_plane_wave_derivative_batch_wrapper(const CU_REAL* xs,
                                             const CU_REAL* ys,
                                             const CU_REAL* zs,
                                             CUCOMPLEX* field_derivatives,
                                             int count,
                                             CU_REAL k0,
                                             CUCOMPLEX amplitude,
                                             CU_REAL ss,
                                             CU_REAL pp,
                                             CU_REAL theta_degrees,
                                             CU_REAL phi_degrees,
                                             int circular);

PRODUITMATVECTFFT_KERNEL_API cudaError_t
incident_multi_plane_wave_wrapper(const CU_REAL* xs,
                                  const CU_REAL* ys,
                                  const CU_REAL* zs,
                                  const CUCOMPLEX* plane_parameters,
                                  CUCOMPLEX* field,
                                  int count,
                                  int plane_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_multi_plane_wave_grid_wrapper(
    IfddaIncidentGrid grid, const CUCOMPLEX* plane_parameters,
    CUCOMPLEX* field, int plane_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
incident_multi_plane_wave_derivative_batch_wrapper(const CU_REAL* xs,
                                                   const CU_REAL* ys,
                                                   const CU_REAL* zs,
                                                   const CUCOMPLEX* plane_parameters,
                                                   CUCOMPLEX* field_derivatives,
                                                   int count,
                                                   int plane_count);

PRODUITMATVECTFFT_KERNEL_API cudaError_t speckle_spectrum_wrapper(const CUCOMPLEX* random_phase,
                                                                  CUCOMPLEX* spectrum,
                                                                  int nfft,
                                                                  CU_REAL k0,
                                                                  CU_REAL deltak,
                                                                  CU_REAL kmax,
                                                                  CU_REAL psi,
                                                                  COMPLEX e0,
                                                                  CU_REAL x0,
                                                                  CU_REAL y0,
                                                                  CU_REAL z0,
                                                                  CU_REAL z_plane,
                                                                  int direction);
PRODUITMATVECTFFT_KERNEL_API cudaError_t speckle_uniforms_to_phase_wrapper(const CU_REAL* uniforms,
                                                                           CUCOMPLEX* phases,
                                                                           size_t count,
                                                                           CU_REAL two_pi);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
diffraction_incident_spectrum_wrapper(CUCOMPLEX* spectrum,
                                      const CUCOMPLEX* plane_parameters,
                                      const CUCOMPLEX* random_phases,
                                      int nfft,
                                      int imax,
                                      int mode,
                                      int plane_count,
                                      CU_REAL k0,
                                      CU_REAL deltakx,
                                      CU_REAL deltaky,
                                      CU_REAL kmax_squared,
                                      CU_REAL psi,
                                      CU_REAL x0,
                                      CU_REAL y0,
                                      CU_REAL z0,
                                      CUCOMPLEX e0);
PRODUITMATVECTFFT_KERNEL_API cudaError_t diffraction_incident_fft_coordinates_wrapper(
    CU_REAL* xs, CU_REAL* ys, CU_REAL* zs, int nfft, CU_REAL deltax);
PRODUITMATVECTFFT_KERNEL_API cudaError_t diffraction_incident_scale_wrapper(CUCOMPLEX* field,
                                                                            int count,
                                                                            CU_REAL scale);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
diffraction_incident_crop_wrapper(const CUCOMPLEX* fft_spectrum,
                                  CUCOMPLEX* spectrum,
                                  int nfft,
                                  int imax,
                                  CU_REAL deltakx,
                                  CU_REAL deltaky,
                                  CU_REAL numerical_aperture);
PRODUITMATVECTFFT_KERNEL_API cudaError_t diffraction_incident_interleaved_to_planes_wrapper(
    const CUCOMPLEX* interleaved, CUCOMPLEX* planes, int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
speckle_flux_normalization_wrapper(CU_REAL* normalization,
                                   int nfft,
                                   CU_REAL k0,
                                   CU_REAL deltak,
                                   CU_REAL kmax,
                                   CU_REAL psi,
                                   COMPLEX e0,
                                   CU_REAL power,
                                   CU_REAL flux_multiplier);
PRODUITMATVECTFFT_KERNEL_API cudaError_t speckle_extract_plane_wrapper(const CUCOMPLEX* spectrum,
                                                                       CUCOMPLEX* field,
                                                                       int nx,
                                                                       int ny,
                                                                       int nfft,
                                                                       int z_index,
                                                                       CU_REAL factor,
                                                                       int add_to_field);
PRODUITMATVECTFFT_KERNEL_API cudaError_t speckle_scale_field_wrapper(CUCOMPLEX* field,
                                                                     int count,
                                                                     const CU_REAL* normalization);

PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_paraxial_gaussian_wrapper(const CU_REAL* xs,
                                                                            const CU_REAL* ys,
                                                                            const CU_REAL* zs,
                                                                            CUCOMPLEX* field,
                                                                            int count,
                                                                            COMPLEX e0,
                                                                            COMPLEX epsilon,
                                                                            CU_REAL ss,
                                                                            CU_REAL pp,
                                                                            CU_REAL theta_degrees,
                                                                            CU_REAL phi_degrees,
                                                                            CU_REAL x0,
                                                                            CU_REAL y0,
                                                                            CU_REAL z0,
                                                                            CU_REAL w0,
                                                                            CU_REAL k0,
                                                                            int circular);
PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_paraxial_gaussian_grid_wrapper(
    IfddaIncidentGrid grid, CUCOMPLEX* field, COMPLEX e0, COMPLEX epsilon,
    REAL ss, REAL pp, REAL theta_degrees, REAL phi_degrees,
    REAL x0, REAL y0, REAL z0, REAL w0, REAL k0, int circular);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gaussianparad_derivative_wrapper(CUCOMPLEX* derivative,
                                                                          CU_REAL x,
                                                                          CU_REAL y,
                                                                          CU_REAL z,
                                                                          CU_REAL waist,
                                                                          CU_REAL k0,
                                                                          CUCOMPLEX e0x,
                                                                          CUCOMPLEX e0y,
                                                                          const CU_REAL* rotation);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
gaussianparad_derivative_batch_wrapper(const CU_REAL* xs,
                                       const CU_REAL* ys,
                                       const CU_REAL* zs,
                                       CUCOMPLEX* derivatives,
                                       int count,
                                       CU_REAL x0,
                                       CU_REAL y0,
                                       CU_REAL z0,
                                       CU_REAL waist,
                                       CU_REAL k0,
                                       CUCOMPLEX e0x,
                                       CUCOMPLEX e0y,
                                       const CU_REAL* rotation);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gaussian_power_wrapper(CUCOMPLEX* field,
                                                                CU_REAL* irradiance,
                                                                CU_REAL waist,
                                                                CU_REAL power);
PRODUITMATVECTFFT_KERNEL_API cudaError_t rigorous_gaussian_power_wrapper(
    CUCOMPLEX* field, CU_REAL* irradiance, CU_REAL waist, CU_REAL k0, CU_REAL power);

PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_antenna_wrapper(const CU_REAL* xs,
                                                                  const CU_REAL* ys,
                                                                  const CU_REAL* zs,
                                                                  CUCOMPLEX* field,
                                                                  CUCOMPLEX* reference_field,
                                                                  int count,
                                                                  CU_REAL xdip,
                                                                  CU_REAL ydip,
                                                                  CU_REAL zdip,
                                                                  CU_REAL aretecube,
                                                                  CU_REAL k0,
                                                                  COMPLEX e0,
                                                                  CU_REAL theta_degrees,
                                                                  CU_REAL phi_degrees,
                                                                  CU_REAL reference_x,
                                                                  CU_REAL reference_y,
                                                                  CU_REAL reference_z);
PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_antenna_grid_wrapper(
    IfddaIncidentGrid grid, CUCOMPLEX* field, CUCOMPLEX* reference_field,
    REAL xdip, REAL ydip, REAL zdip, REAL aretecube, REAL k0,
    COMPLEX e0, REAL theta_degrees, REAL phi_degrees,
    REAL reference_x, REAL reference_y, REAL reference_z);
PRODUITMATVECTFFT_KERNEL_API cudaError_t dipole_moment_field_wrapper(const CU_REAL* xs,
                                                                     const CU_REAL* ys,
                                                                     const CU_REAL* zs,
                                                                     CUCOMPLEX* field,
                                                                     CU_REAL xdip,
                                                                     CU_REAL ydip,
                                                                     CU_REAL zdip,
                                                                     CU_REAL aretecube,
                                                                     CU_REAL k0,
                                                                     CUCOMPLEX moment_x,
                                                                     CUCOMPLEX moment_y,
                                                                     CUCOMPLEX moment_z);
PRODUITMATVECTFFT_KERNEL_API cudaError_t dipole_moment_field_batch_wrapper(const CU_REAL* xs,
                                                                           const CU_REAL* ys,
                                                                           const CU_REAL* zs,
                                                                           CUCOMPLEX* field,
                                                                           int count,
                                                                           CU_REAL xdip,
                                                                           CU_REAL ydip,
                                                                           CU_REAL zdip,
                                                                           CU_REAL aretecube,
                                                                           CU_REAL k0,
                                                                           CUCOMPLEX moment_x,
                                                                           CUCOMPLEX moment_y,
                                                                           CUCOMPLEX moment_z);
PRODUITMATVECTFFT_KERNEL_API cudaError_t dipole_incident_derivative_wrapper(CUCOMPLEX* derivative,
                                                                            CU_REAL xdip,
                                                                            CU_REAL ydip,
                                                                            CU_REAL zdip,
                                                                            CU_REAL theta_degrees,
                                                                            CU_REAL phi_degrees,
                                                                            CU_REAL x,
                                                                            CU_REAL y,
                                                                            CU_REAL z,
                                                                            CU_REAL aretecube,
                                                                            CU_REAL k0,
                                                                            CUCOMPLEX e0,
                                                                            int derivative_axis);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
dipole_incident_derivative_batch_wrapper(const CU_REAL* xs,
                                         const CU_REAL* ys,
                                         const CU_REAL* zs,
                                         CUCOMPLEX* derivatives,
                                         int count,
                                         CU_REAL xdip,
                                         CU_REAL ydip,
                                         CU_REAL zdip,
                                         CU_REAL theta_degrees,
                                         CU_REAL phi_degrees,
                                         CU_REAL aretecube,
                                         CU_REAL k0,
                                         CUCOMPLEX e0);

PRODUITMATVECTFFT_KERNEL_API cudaError_t polarization_inplace_wrapper(
    const CUCOMPLEX* polarizability, CUCOMPLEX* field, int niso, int nmax, int ndipoles);

PRODUITMATVECTFFT_KERNEL_API cudaError_t local_macro_to_local_wrapper(const CUCOMPLEX* epsilon,
                                                                      CUCOMPLEX* field,
                                                                      int niso,
                                                                      int nmax,
                                                                      int ndipoles,
                                                                      CU_REAL aretecube,
                                                                      CU_REAL k0);
PRODUITMATVECTFFT_KERNEL_API cudaError_t local_macro_batch_wrapper(const CUCOMPLEX* epsilon,
                                                                   const CUCOMPLEX* input_field,
                                                                   CUCOMPLEX* output_field,
                                                                   CU_REAL* field_magnitude,
                                                                   CU_INT* status,
                                                                   int niso,
                                                                   int nmax,
                                                                   int ndipoles,
                                                                   CU_REAL aretecube,
                                                                   CU_REAL k0,
                                                                   int nsens);

PRODUITMATVECTFFT_KERNEL_API cudaError_t complex_vector_scale_wrapper(const CUCOMPLEX* input,
                                                                      CUCOMPLEX* output,
                                                                      CUCOMPLEX scale,
                                                                      int count);

PRODUITMATVECTFFT_KERNEL_API cudaError_t rytov_macro_delta_wrapper(const CUCOMPLEX* incident,
                                                                   const CUCOMPLEX* epsilon,
                                                                   CUCOMPLEX* delta,
                                                                   CUCOMPLEX background_epsilon,
                                                                   int niso,
                                                                   int nmax,
                                                                   int ndipoles);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
rytov_reference_local_wrapper(const CUCOMPLEX* reference_field,
                              const CUCOMPLEX* delta_field,
                              CUCOMPLEX* local_field,
                              int ndipoles);

PRODUITMATVECTFFT_KERNEL_API cudaError_t gather_incident_wrapper(const CUCOMPLEX* incident,
                                                                 const CU_INT* dipole_map,
                                                                 CUCOMPLEX* vectors,
                                                                 int nx,
                                                                 int ny,
                                                                 int nz,
                                                                 int nx2,
                                                                 int nxy2,
                                                                 int ntotal,
                                                                 int nmax);

PRODUITMATVECTFFT_KERNEL_API cudaError_t multiply_green_tensor_wrapper(GreenTensorDeviceView green,
                                                                       CUCOMPLEX* vectors,
                                                                       int nx,
                                                                       int ny,
                                                                       int nz,
                                                                       int nx2,
                                                                       int ny2,
                                                                       int nz2,
                                                                       size_t tensor_count);

PRODUITMATVECTFFT_KERNEL_API cudaError_t update_output_wrapper(const CUCOMPLEX* vectors,
                                                               const CU_INT* dipole_map,
                                                               CUCOMPLEX* output,
                                                               int nx,
                                                               int ny,
                                                               int nz,
                                                               int nx2,
                                                               int nxy2,
                                                               int nmax,
                                                               int ntotal);

PRODUITMATVECTFFT_KERNEL_API cudaError_t gather_incident_regular_wrapper(const CUCOMPLEX* incident,
                                                                         CUCOMPLEX* vectors,
                                                                         int nx,
                                                                         int ny,
                                                                         int nz,
                                                                         int nx2,
                                                                         int nxy2,
                                                                         int ntotal,
                                                                         int nmax);

PRODUITMATVECTFFT_KERNEL_API cudaError_t multiply_green_tensor_opt_wrapper(GreenTensorDeviceView green,
                                                                           CUCOMPLEX* vectors,
                                                                           int nx,
                                                                           int ny,
                                                                           int nz,
                                                                           int nx2,
                                                                           int ny2,
                                                                           int nz2,
                                                                           size_t tensor_count);

PRODUITMATVECTFFT_KERNEL_API cudaError_t update_output_regular_wrapper(const CUCOMPLEX* vectors,
                                                                       CUCOMPLEX* output,
                                                                       int nx,
                                                                       int ny,
                                                                       int nz,
                                                                       int nx2,
                                                                       int nxy2,
                                                                       int nmax,
                                                                       int ntotal);

PRODUITMATVECTFFT_KERNEL_API cudaError_t gather_scalar_product_wrapper(const CUCOMPLEX* input,
                                                                       const CU_INT* dipole_map,
                                                                       CUCOMPLEX* work,
                                                                       int nx,
                                                                       int ny,
                                                                       int nz,
                                                                       int nx2,
                                                                       int nxy2,
                                                                       int nmax);
PRODUITMATVECTFFT_KERNEL_API cudaError_t multiply_scalar_green_wrapper(const CUCOMPLEX* green,
                                                                       CUCOMPLEX* work,
                                                                       int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scatter_scalar_product_wrapper(const CUCOMPLEX* work,
                                                                        const CU_INT* dipole_map,
                                                                        CUCOMPLEX* output,
                                                                        int nx,
                                                                        int ny,
                                                                        int nz,
                                                                        int nx2,
                                                                        int nxy2,
                                                                        int nmax,
                                                                        int ntotal);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gather_scalar_product_dense_wrapper(
    const CUCOMPLEX* input, CUCOMPLEX* work, int nx, int ny, int nz, int nx2, int nxy2);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scatter_scalar_product_dense_wrapper(const CUCOMPLEX* work,
                                                                              CUCOMPLEX* output,
                                                                              int nx,
                                                                              int ny,
                                                                              int nz,
                                                                              int nx2,
                                                                              int nxy2,
                                                                              int ntotal);

PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicg_save_initial_wrapper(const CUCOMPLEX* xi,
                                                                     CUCOMPLEX* work,
                                                                     int lda,
                                                                     int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
gpbicg_initialize_wrapper(const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicg2_initialize_wrapper(
    const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
gpbicg_prepare_direction_wrapper(CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX beta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
gpbicg_save_xr_column_wrapper(const CUCOMPLEX* xr, CUCOMPLEX* work, int lda, int n, int column);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
gpbicg_update_stage2_wrapper(CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX alpha, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicg_update_stage3_wrapper(CUCOMPLEX* work,
                                                                      CUCOMPLEX alpha,
                                                                      CUCOMPLEX beta,
                                                                      CUCOMPLEX eta,
                                                                      CUCOMPLEX dzeta,
                                                                      int lda,
                                                                      int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicg_update_stage4_wrapper(CUCOMPLEX* work,
                                                                      CUCOMPLEX beta,
                                                                      int lda,
                                                                      int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicg_publish_solution_wrapper(CUCOMPLEX* xi,
                                                                         const CUCOMPLEX* work,
                                                                         int lda,
                                                                         int n);

PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgplus_initialize_wrapper(
    const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgplus_prepare_direction_wrapper(
    CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX beta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgplus_update_ap_wrapper(
    const CUCOMPLEX* xr, CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX beta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgplus_update_stage2_wrapper(CUCOMPLEX* xi,
                                                                          CUCOMPLEX* work,
                                                                          CUCOMPLEX alpha,
                                                                          CUCOMPLEX beta,
                                                                          CUCOMPLEX eta,
                                                                          CUCOMPLEX dzeta,
                                                                          int lda,
                                                                          int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgplus_update_stage3_wrapper(CUCOMPLEX* work,
                                                                          const CUCOMPLEX* xr,
                                                                          CUCOMPLEX alpha,
                                                                          CUCOMPLEX eta,
                                                                          CUCOMPLEX dzeta,
                                                                          int lda,
                                                                          int n);

PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgsafe_initialize_wrapper(
    const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgsafe_prepare_direction_wrapper(
    CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX beta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgsafe_update_ap_wrapper(
    const CUCOMPLEX* xr, CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX beta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgsafe_update_stage2_wrapper(
    CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX beta, CUCOMPLEX eta, CUCOMPLEX dzeta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgsafe_update_stage3_wrapper(CUCOMPLEX* work,
                                                                          const CUCOMPLEX* xr,
                                                                          CUCOMPLEX alpha,
                                                                          CUCOMPLEX eta,
                                                                          CUCOMPLEX dzeta,
                                                                          int lda,
                                                                          int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgar_initialize_wrapper(
    const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* xi, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
gpbicgar_update_ap_wrapper(const CUCOMPLEX* xr, CUCOMPLEX* work, CUCOMPLEX beta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgar_update_u_wrapper(
    CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX beta, CUCOMPLEX eta, CUCOMPLEX dzeta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgar_update_stage_wrapper(CUCOMPLEX* work,
                                                                       const CUCOMPLEX* xr,
                                                                       CUCOMPLEX alpha,
                                                                       CUCOMPLEX eta,
                                                                       CUCOMPLEX dzeta,
                                                                       int lda,
                                                                       int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgar_publish_residual_wrapper(CUCOMPLEX* xi,
                                                                           const CUCOMPLEX* work,
                                                                           int lda,
                                                                           int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgar_copy_ar_wrapper(const CUCOMPLEX* xr,
                                                                  CUCOMPLEX* work,
                                                                  int lda,
                                                                  int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicor_initialize_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicor_first_direction_wrapper(
    CUCOMPLEX* xi, CUCOMPLEX* work, const CUCOMPLEX* xr, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicor_update_directions_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, CUCOMPLEX* work, CUCOMPLEX beta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicor_prepare_stage_wrapper(
    CUCOMPLEX* xi, CUCOMPLEX* xr, CUCOMPLEX* work, CUCOMPLEX alpha, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicor_finish_stage_wrapper(CUCOMPLEX* ffloc,
                                                                      CUCOMPLEX* work,
                                                                      const CUCOMPLEX* xi,
                                                                      const CUCOMPLEX* xr,
                                                                      CUCOMPLEX alpha,
                                                                      CUCOMPLEX beta,
                                                                      CUCOMPLEX eta,
                                                                      CUCOMPLEX dzeta,
                                                                      int lda,
                                                                      int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicor_publish_solution_wrapper(CUCOMPLEX* xi,
                                                                          const CUCOMPLEX* ffloc,
                                                                          int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicor_publish_residual_wrapper(CUCOMPLEX* xi,
                                                                          const CUCOMPLEX* work,
                                                                          int lda,
                                                                          int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t cors_save_initial_wrapper(const CUCOMPLEX* xi,
                                                                   CUCOMPLEX* work,
                                                                   int lda,
                                                                   int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t cors_initialize_residual_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t cors_initialize_direction_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t cors_update_direction_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, CUCOMPLEX* work, CUCOMPLEX beta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
cors_update_stage_wrapper(CUCOMPLEX* work, const CUCOMPLEX* xr, CUCOMPLEX alpha, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t cors_publish_solution_wrapper(CUCOMPLEX* xi,
                                                                       const CUCOMPLEX* work,
                                                                       int lda,
                                                                       int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t cors_publish_residual_wrapper(CUCOMPLEX* xi,
                                                                       const CUCOMPLEX* work,
                                                                       int lda,
                                                                       int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t tfqmr_initialize_residual_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t tfqmr_initialize_vectors_wrapper(const CUCOMPLEX* xr,
                                                                          CUCOMPLEX* work,
                                                                          int lda,
                                                                          int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
tfqmr_prepare_uplus_wrapper(CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX alpha, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t tfqmr_update_w_wrapper(CUCOMPLEX* work,
                                                                CUCOMPLEX alpha,
                                                                int lda,
                                                                int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t tfqmr_update_halfstep_wrapper(
    CUCOMPLEX* x, CUCOMPLEX* work, CUCOMPLEX d_coefficient, CUCOMPLEX eta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t tfqmr_continue_wrapper(CUCOMPLEX* xi,
                                                                const CUCOMPLEX* xr,
                                                                CUCOMPLEX* work,
                                                                CUCOMPLEX beta,
                                                                int odd_step,
                                                                int lda,
                                                                int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t tfqmr_end_halfstep_wrapper(
    CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX beta, int odd_step, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t bicgstarplus_save_solution_wrapper(const CUCOMPLEX* xi,
                                                                            CUCOMPLEX* work,
                                                                            int lda,
                                                                            int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t bicgstarplus_initialize_residual_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t bicgstarplus_update_direction_wrapper(CUCOMPLEX* xi,
                                                                               CUCOMPLEX* work,
                                                                               CUCOMPLEX alpha,
                                                                               CUCOMPLEX beta,
                                                                               CUCOMPLEX eta,
                                                                               CUCOMPLEX dzeta,
                                                                               int lda,
                                                                               int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t bicgstarplus_update_residual_wrapper(CUCOMPLEX* xi,
                                                                              const CUCOMPLEX* xr,
                                                                              CUCOMPLEX* work,
                                                                              CUCOMPLEX alpha,
                                                                              CUCOMPLEX beta,
                                                                              CUCOMPLEX eta,
                                                                              CUCOMPLEX dzeta,
                                                                              int lda,
                                                                              int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
bicgstarplus_publish_residual_wrapper(CUCOMPLEX* xi, const CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t zcg_save_solution_wrapper(const CUCOMPLEX* xi,
                                                                   CUCOMPLEX* work,
                                                                   int lda,
                                                                   int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t zcg_initialize_residual_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t zcg_save_initial_w_wrapper(const CUCOMPLEX* xr,
                                                                    CUCOMPLEX* work,
                                                                    int lda,
                                                                    int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t zcg_step_wrapper(CUCOMPLEX* work,
                                                          CUCOMPLEX alpha,
                                                          int lda,
                                                          int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
zcg_update_directions_wrapper(const CUCOMPLEX* xr, CUCOMPLEX* work, CUCOMPLEX beta, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t zcg_publish_solution_wrapper(CUCOMPLEX* xi,
                                                                      const CUCOMPLEX* work,
                                                                      int lda,
                                                                      int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t zcg_publish_residual_wrapper(CUCOMPLEX* xi,
                                                                      const CUCOMPLEX* work,
                                                                      int lda,
                                                                      int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pimzbicgstab_save_rhs_wrapper(const CUCOMPLEX* b,
                                                                       CUCOMPLEX* work,
                                                                       int lda,
                                                                       int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pimzbicgstab_initialize_residual_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pimzbicgstab_prepare_direction_wrapper(
    CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX beta, CUCOMPLEX omega, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pimzbicgstab_prepare_s_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, CUCOMPLEX* work, CUCOMPLEX alpha, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pimzbicgstab_update_xr_wrapper(
    CUCOMPLEX* x, CUCOMPLEX* work, CUCOMPLEX alpha, CUCOMPLEX omega, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pimzbicgstab_soft_breakdown_wrapper(CUCOMPLEX* x,
                                                                             const CUCOMPLEX* work,
                                                                             int lda,
                                                                             int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pimzbicgstab_publish_vector_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* work, int lda, int column, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t qmrbicgstab_initialize_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t qmrbicgstab_update_p_wrapper(
    CUCOMPLEX* xi, CUCOMPLEX* work, CUCOMPLEX beta, CUCOMPLEX omega, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t qmrbicgstab_prepare_s_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, CUCOMPLEX* work, CUCOMPLEX alpha, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t qmrbicgstab_prepare_x_wrapper(CUCOMPLEX* x,
                                                                       CUCOMPLEX* work,
                                                                       CUCOMPLEX alpha,
                                                                       CUCOMPLEX theta,
                                                                       CUCOMPLEX eta,
                                                                       CUCOMPLEX etat,
                                                                       int lda,
                                                                       int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t qmrbicgstab_save_t_wrapper(const CUCOMPLEX* xr,
                                                                    CUCOMPLEX* work,
                                                                    int lda,
                                                                    int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t qmrbicgstab_update_xr_wrapper(CUCOMPLEX* work,
                                                                       CUCOMPLEX omega,
                                                                       int lda,
                                                                       int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t qmrbicgstab_finalize_x_wrapper(CUCOMPLEX* x,
                                                                        CUCOMPLEX* work,
                                                                        CUCOMPLEX omega,
                                                                        CUCOMPLEX eta,
                                                                        CUCOMPLEX thetat,
                                                                        CUCOMPLEX etat,
                                                                        int lda,
                                                                        int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t qmrbicgstab_publish_x_wrapper(CUCOMPLEX* xi,
                                                                       const CUCOMPLEX* x,
                                                                       int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
idrs_initialize_wrapper(const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
idrs_copy_column_wrapper(CUCOMPLEX* work, int dst_col, int src_col, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
idrs_copy_xr_wrapper(CUCOMPLEX* work, const CUCOMPLEX* xr, int dst_col, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
idrs_scale_copy_wrapper(CUCOMPLEX* work, int dst_col, int src_col, CUCOMPLEX scale, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
idrs_axpy_wrapper(CUCOMPLEX* work, int dst_col, int src_col, CUCOMPLEX scale, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
idrs_normalize_basis_wrapper(CUCOMPLEX* work, int column, CU_REAL norm, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t idrs_initial_pair_wrapper(
    CUCOMPLEX* work, CUCOMPLEX* x, CUCOMPLEX omega, int is, int ns, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t idrs_apply_pair_wrapper(CUCOMPLEX* work,
                                                                 CUCOMPLEX* x,
                                                                 int lda,
                                                                 int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t idrs_negate_xr_wrapper(CUCOMPLEX* work,
                                                                const CUCOMPLEX* xr,
                                                                int lda,
                                                                int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t idrs_scale_xr_wrapper(
    CUCOMPLEX* work, const CUCOMPLEX* xr, int dst_col, CUCOMPLEX scale, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t idrs_shift_qe_wrapper(CUCOMPLEX* work,
                                                               int ns,
                                                               int lda,
                                                               int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
idrs_copy_work_to_xi_wrapper(CUCOMPLEX* xi, const CUCOMPLEX* work, int column, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gpbicgstabl_combine_wrapper(CUCOMPLEX* work,
                                                                     CUCOMPLEX* xsol,
                                                                     int lda,
                                                                     int n,
                                                                     int dst_col,
                                                                     const int* source_cols,
                                                                     const CUCOMPLEX* coefficients,
                                                                     int term_count,
                                                                     int add_to_destination);
// Solves a small dense system in existing device buffers. Matrix layout is
// row-major; rhs is replaced by the solution. info_device is zero on success.
PRODUITMATVECTFFT_KERNEL_API cudaError_t solve_small_dense_system_device_wrapper(
    CUCOMPLEX* matrix_device, CUCOMPLEX* rhs_device, int order, CU_INT* info_device);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pimzqmr_initialize_wrapper(
    const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pimzqmr_shadow_wrapper(CUCOMPLEX* xi,
                                                                CUCOMPLEX* work,
                                                                int lda,
                                                                int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t ssbicgsafe2_initialize_wrapper(
    const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gcors2_initialize_residual_wrapper(
    CUCOMPLEX* xi, const CUCOMPLEX* xr, const CUCOMPLEX* b, CUCOMPLEX* work, int lda, int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gcors2_setup_wrapper(CUCOMPLEX* xi,
                                                              const CUCOMPLEX* xr,
                                                              CUCOMPLEX* work,
                                                              const CUCOMPLEX* random_values,
                                                              int lda,
                                                              int n);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gcors2_publish_solution_wrapper(CUCOMPLEX* xi,
                                                                         const CUCOMPLEX* xsol,
                                                                         int n);

// Chan preconditioner application: gather/scatter the three field components
// between the physical dipole layout, 2-D FFT planes, and batched z systems.
PRODUITMATVECTFFT_KERNEL_API cudaError_t
chan_preconditioner_pack_fft_wrapper(const CUCOMPLEX* field,
                                     const CU_INT* dipole_map,
                                     CUCOMPLEX* fft_planes,
                                     int nx,
                                     int ny,
                                     int nz,
                                     int nxm,
                                     int nym,
                                     int nzm,
                                     int nmax);
PRODUITMATVECTFFT_KERNEL_API cudaError_t chan_preconditioner_fft_to_rhs_wrapper(
    const CUCOMPLEX* fft_planes, CUCOMPLEX* rhs, int nx, int ny, int nz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t chan_preconditioner_rhs_to_fft_wrapper(
    const CUCOMPLEX* rhs, CUCOMPLEX* fft_planes, int nx, int ny, int nz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
chan_preconditioner_unpack_fft_wrapper(const CUCOMPLEX* fft_planes,
                                       const CU_INT* dipole_map,
                                       CUCOMPLEX* field,
                                       int nx,
                                       int ny,
                                       int nz,
                                       int nxm,
                                       int nym,
                                       int nzm,
                                       int nmax,
                                       CU_REAL normalization);
PRODUITMATVECTFFT_KERNEL_API cudaError_t chan_preconditioner_build_blocks_wrapper(
    const CUCOMPLEX* chan_vectors, CUCOMPLEX* matrices, int nx, int ny, int nz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t chan_preconditioner_pack_matrices_wrapper(
    const CUCOMPLEX* source, CUCOMPLEX* packed, int nz, int nzm, int batch_count);
// Construct cuBLAS batched pointer arrays and dense dipole maps directly on GPU.
// Neither operation requires a host-side staging array or a PCIe transfer.
PRODUITMATVECTFFT_KERNEL_API cudaError_t chan_preconditioner_init_batch_ptrs_wrapper(
    CUCOMPLEX* matrices, CUCOMPLEX* rhs, CUCOMPLEX** matrix_ptrs,
    CUCOMPLEX** rhs_ptrs, size_t matrix_stride, size_t rhs_stride, int batch_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t chan_preconditioner_identity_map_wrapper(
    CU_INT* identity_map, int map_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t chan_preconditioner_green_tensor_wrapper(
    CUCOMPLEX* green_components, CU_REAL aretecube, CU_REAL k0, int nx, int ny, int nz);

PRODUITMATVECTFFT_KERNEL_API cudaError_t chan_preconditioner_mean_polarizability_wrapper(
    const CUCOMPLEX* polarizabilities, int count, CUCOMPLEX* sum, CU_INT* nonzero_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
chan_preconditioner_finalize_mean_polarizability_wrapper(CUCOMPLEX* sum, const CU_INT* nonzero_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
chan_preconditioner_interpolate_wrapper(const CUCOMPLEX* green_components,
                                        CUCOMPLEX* fft_components,
                                        const CUCOMPLEX* mean_polarizability,
                                        int nx,
                                        int ny,
                                        int nz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t chan_preconditioner_transpose_vectors_wrapper(
    const CUCOMPLEX* fft_components, CUCOMPLEX* chan_vectors, int nx, int ny, int nz);

// Scalar-approximation Chan preconditioner kernels. Layouts follow the
// Fortran preconditionneurscal arrays, with one complex field component.
PRODUITMATVECTFFT_KERNEL_API cudaError_t
scalar_driver_project_incident_wrapper(const CUCOMPLEX* vector_incident,
                                       CUCOMPLEX* scalar_incident,
                                       CUCOMPLEX uincx,
                                       CUCOMPLEX uincy,
                                       CUCOMPLEX uincz,
                                       int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scalar_driver_apply_polarizability_wrapper(
    const CUCOMPLEX* input, const CUCOMPLEX* polarizability, CUCOMPLEX* output, int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
scalar_driver_exponential_guess_wrapper(const CUCOMPLEX* incident,
                                        const CUCOMPLEX* green_correction,
                                        CUCOMPLEX* initial_guess,
                                        int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scalar_driver_copy_solution_wrapper(
    const CUCOMPLEX* solution, CUCOMPLEX* local_solution, int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scalar_driver_residual_wrapper(CUCOMPLEX* product,
                                                                        const CUCOMPLEX* rhs,
                                                                        const CUCOMPLEX* solution,
                                                                        CUCOMPLEX* scalar_solution,
                                                                        int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
scalar_driver_expand_fields_wrapper(const CUCOMPLEX* solution,
                                    const CUCOMPLEX* local_solution,
                                    CUCOMPLEX* vector_field,
                                    CUCOMPLEX* vector_local_field,
                                    CUCOMPLEX uincx,
                                    CUCOMPLEX uincy,
                                    CUCOMPLEX uincz,
                                    int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scalar_chan_build_wrapper(CUCOMPLEX* channel,
                                                                   CUCOMPLEX mean_polarizability,
                                                                   CUCOMPLEX uincx,
                                                                   CUCOMPLEX uincy,
                                                                   CUCOMPLEX uincz,
                                                                   CU_REAL aretecube,
                                                                   CU_REAL k0,
                                                                   int nrig,
                                                                   int nx,
                                                                   int ny,
                                                                   int nz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t generate_scalar_green_wrapper(CUCOMPLEX* green,
                                                                       int nx,
                                                                       int ny,
                                                                       int nz,
                                                                       int nx2,
                                                                       int ny2,
                                                                       int nz2,
                                                                       int tensor_count,
                                                                       CU_REAL aretecube,
                                                                       CU_REAL k0,
                                                                       CUCOMPLEX uincx,
                                                                       CUCOMPLEX uincy,
                                                                       CUCOMPLEX uincz,
                                                                       int nrig);
PRODUITMATVECTFFT_KERNEL_API cudaError_t generate_multilayer_scalar_green_wrapper(
    CUCOMPLEX* green, int nfft2d, int count, CU_REAL aretecube, CU_REAL k0);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
generate_multilayer_projected_green_wrapper(CUCOMPLEX* green,
                                            int nfft2d,
                                            int count,
                                            CU_REAL aretecube,
                                            CU_REAL k0,
                                            CUCOMPLEX uincx,
                                            CUCOMPLEX uincy,
                                            CUCOMPLEX uincz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
generate_multilayer_dyadic_green_wrapper(CUCOMPLEX* green_xx,
                                         CUCOMPLEX* green_xy,
                                         CUCOMPLEX* green_xz,
                                         CUCOMPLEX* green_yy,
                                         CUCOMPLEX* green_yz,
                                         CUCOMPLEX* green_zz,
                                         int nfft2d,
                                         int count,
                                         CU_REAL aretecube,
                                         CU_REAL k0);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
generate_vector_green_tensor_wrapper(GreenTensorDevicePointers green,
                                     GreenTensorDevicePointers spatial,
                                     GreenTensorDevicePointers spectra,
                                     int nx,
                                     int ny,
                                     int nz,
                                     int nxm,
                                     int nym,
                                     int nzm,
                                     int nx2,
                                     int ny2,
                                     int nz2,
                                     cufftHandle plan,
                                     REAL aretecube,
                                     REAL k0,
                                     COMPLEX eps0,
                                     int medium,
                                     int nquad,
                                     int mode);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_vector_prepare_wrapper(const CUCOMPLEX* image_x,
                                  const CUCOMPLEX* image_y,
                                  const CUCOMPLEX* image_z,
                                  const CU_INT* plane_map,
                                  const CUCOMPLEX* polarizability,
                                  CUCOMPLEX* polarization_x,
                                  CUCOMPLEX* polarization_y,
                                  CUCOMPLEX* polarization_z,
                                  int nfft2d,
                                  int nx,
                                  int ny,
                                  int layer,
                                  int plane_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_vector_prepare_reflected_wrapper(const CUCOMPLEX* forward_x,
                                            const CUCOMPLEX* forward_y,
                                            const CUCOMPLEX* forward_z,
                                            const CUCOMPLEX* reflected_x,
                                            const CUCOMPLEX* reflected_y,
                                            const CUCOMPLEX* reflected_z,
                                            const CU_INT* plane_map,
                                            const CUCOMPLEX* polarizability,
                                            CUCOMPLEX* polarization_x,
                                            CUCOMPLEX* polarization_y,
                                            CUCOMPLEX* polarization_z,
                                            int nfft2d,
                                            int nx,
                                            int ny,
                                            int layer,
                                            int plane_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_vector_product_wrapper(const CUCOMPLEX* polarization_x,
                                  const CUCOMPLEX* polarization_y,
                                  const CUCOMPLEX* polarization_z,
                                  const CUCOMPLEX* green_xx,
                                  const CUCOMPLEX* green_xy,
                                  const CUCOMPLEX* green_xz,
                                  const CUCOMPLEX* green_yy,
                                  const CUCOMPLEX* green_yz,
                                  const CUCOMPLEX* green_zz,
                                  CUCOMPLEX* field_x,
                                  CUCOMPLEX* field_y,
                                  CUCOMPLEX* field_z,
                                  int count,
                                  int reflected);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_vector_accumulate_wrapper(CUCOMPLEX* image_x,
                                     CUCOMPLEX* image_y,
                                     CUCOMPLEX* image_z,
                                     const CUCOMPLEX* field_x,
                                     const CUCOMPLEX* field_y,
                                     const CUCOMPLEX* field_z,
                                     CUCOMPLEX* local_field,
                                     const CU_INT* plane_map,
                                     int nfft2d,
                                     int nx,
                                     int ny,
                                     int layer,
                                     int nmax,
                                     int plane_count,
                                     CU_REAL image_scale,
                                     CU_REAL field_scale,
                                     int add_to_local);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_scalar_prepare_polarization_wrapper(const CUCOMPLEX* image,
                                               const CUCOMPLEX* add_image,
                                               const CU_INT* plane_map,
                                               const CUCOMPLEX* polarizability,
                                               CUCOMPLEX* polarization,
                                               int nfft2d,
                                               int nx,
                                               int ny,
                                               int layer,
                                               int plane_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_scalar_map_first_plane_wrapper(const CU_REAL* xs,
                                          const CU_REAL* ys,
                                          const CU_REAL* zs,
                                          CU_INT* plane_map,
                                          int ndipole,
                                          int nfft2d,
                                          int kkm,
                                          int jjm,
                                          CU_REAL xmin,
                                          CU_REAL ymin,
                                          CU_REAL first_z,
                                          CU_REAL aretecube);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_scalar_initialize_plane_wrapper(const CU_INT* plane_map,
                                           const CUCOMPLEX* incident_dipoles,
                                           CUCOMPLEX* image,
                                           CUCOMPLEX* local_field,
                                           int nfft2d,
                                           int nmax,
                                           int kkm,
                                           int jjm,
                                           CU_REAL xmin,
                                           CU_REAL ymin,
                                           CU_REAL first_z,
                                           CU_REAL aretecube,
                                           CU_REAL k0,
                                           CUCOMPLEX E0,
                                           CUCOMPLEX eps0,
                                           CU_REAL w0,
                                           CU_REAL ss,
                                           CU_REAL pp,
                                           CU_REAL theta_degrees,
                                           CU_REAL phi_degrees,
                                           int beam_type,
                                           int conjugate_projection,
                                           CUCOMPLEX ux,
                                           CUCOMPLEX uy,
                                           CUCOMPLEX uz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_scalar_initialize_rigorous_plane_wrapper(const CU_INT* plane_map,
                                                    const CUCOMPLEX* incident_dipoles,
                                                    const CUCOMPLEX* rigorous_grid,
                                                    CUCOMPLEX* image,
                                                    CUCOMPLEX* local_field,
                                                    int nfft2d,
                                                    int nmax,
                                                    int conjugate_projection,
                                                    CUCOMPLEX ux,
                                                    CUCOMPLEX uy,
                                                    CUCOMPLEX uz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_vector_initialize_plane_wrapper(const CU_INT* plane_map,
                                           const CUCOMPLEX* incident_dipoles,
                                           CUCOMPLEX* image_x,
                                           CUCOMPLEX* image_y,
                                           CUCOMPLEX* image_z,
                                           CUCOMPLEX* local_field,
                                           int nfft2d,
                                           int nmax,
                                           int kkm,
                                           int jjm,
                                           CU_REAL xmin,
                                           CU_REAL ymin,
                                           CU_REAL first_z,
                                           CU_REAL aretecube,
                                           CU_REAL k0,
                                           CUCOMPLEX E0,
                                           CUCOMPLEX eps0,
                                           CU_REAL w0,
                                           CU_REAL ss,
                                           CU_REAL pp,
                                           CU_REAL theta_degrees,
                                           CU_REAL phi_degrees,
                                           int beam_type);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_vector_initialize_rigorous_plane_wrapper(const CU_INT* plane_map,
                                                    const CUCOMPLEX* incident_dipoles,
                                                    const CUCOMPLEX* rigorous_grid,
                                                    CUCOMPLEX* image_x,
                                                    CUCOMPLEX* image_y,
                                                    CUCOMPLEX* image_z,
                                                    CUCOMPLEX* local_field,
                                                    int nfft2d,
                                                    int nmax);
PRODUITMATVECTFFT_KERNEL_API cudaError_t multilayer_scalar_multiply_green_wrapper(
    CUCOMPLEX* polarization, const CUCOMPLEX* green, int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t multilayer_scalar_propagate_wrapper(
    CUCOMPLEX* image, int nfft2d, CU_REAL deltakx, CU_REAL deltaky, CU_REAL k0, CU_REAL aretecube);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_scalar_accumulate_wrapper(CUCOMPLEX* image,
                                     const CUCOMPLEX* scattered,
                                     CUCOMPLEX* local_field,
                                     const CU_INT* plane_map,
                                     int nfft2d,
                                     int nx,
                                     int ny,
                                     int layer,
                                     int nmax,
                                     int plane_count,
                                     CUCOMPLEX ux,
                                     CUCOMPLEX uy,
                                     CUCOMPLEX uz,
                                     CU_REAL image_scale,
                                     CU_REAL scattered_scale,
                                     int add_to_local);
PRODUITMATVECTFFT_KERNEL_API cudaError_t multilayer_scalar_save_plane_wrapper(
    const CUCOMPLEX* image, CUCOMPLEX* history, int image_count, int layer, int layer_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t rytov_vector_field_wrapper(const CUCOMPLEX* eref,
                                                                    const CUCOMPLEX* delta,
                                                                    CUCOMPLEX* result,
                                                                    CU_INT* error_flag,
                                                                    int dipole_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
beampropagation_spectral_wrapper(CUCOMPLEX* image_planes,
                                 const CU_INT* first_plane_map,
                                 const CUCOMPLEX* epsilon,
                                 CUCOMPLEX* local_field,
                                 int nx,
                                 int ny,
                                 int nz,
                                 int nfft2d,
                                 int nmax,
                                 int niso,
                                 int ndipole,
                                 cufftHandle plan,
                                 CU_REAL aretecube,
                                 CU_REAL k0,
                                 int convert_local_to_macro);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gaussianfft_filter_wrapper(
    CUCOMPLEX* image_planes, int nfft2d, CU_REAL aretecube, CU_REAL k0, int first_plane);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gaussianfft_build_coordinates_wrapper(CU_REAL* xs,
                                                                               CU_REAL* ys,
                                                                               CU_REAL* zs,
                                                                               int nfft2d,
                                                                               int nx,
                                                                               int ny,
                                                                               CU_REAL xmin,
                                                                               CU_REAL ymin,
                                                                               CU_REAL z,
                                                                               CU_REAL aretecube);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
gaussianfft_build_plane_map_wrapper(const CU_REAL* dipole_xs,
                                    const CU_REAL* dipole_ys,
                                    CU_INT* plane_map,
                                    int nfft2d,
                                    int dipole_count,
                                    int kkm,
                                    int jjm,
                                    CU_REAL xmin,
                                    CU_REAL ymin,
                                    CU_REAL aretecube);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gaussianfft_max_intensity_wrapper(const CUCOMPLEX* field,
                                                                           CU_REAL* partials,
                                                                           CU_REAL* maximum,
                                                                           int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
gaussianfft_blend_first_plane_wrapper(const CUCOMPLEX* paraxial,
                                      const CUCOMPLEX* rigorous,
                                      const CU_REAL* max_intensity,
                                      CU_REAL tolerance,
                                      int circular,
                                      CUCOMPLEX* image_planes,
                                      int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
beampropagation_blend_gaussian_background_wrapper(const CU_INT* first_plane_map,
                                                  const CUCOMPLEX* paraxial,
                                                  const CUCOMPLEX* rigorous,
                                                  const CU_REAL* max_intensity,
                                                  CU_REAL threshold,
                                                  CUCOMPLEX* image_planes,
                                                  int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t gaussianfft_scatter_wrapper(const CUCOMPLEX* image_planes,
                                                                     const CU_INT* plane_map,
                                                                     CUCOMPLEX* dipole_field,
                                                                     int nfft2d,
                                                                     int plane_count,
                                                                     int layer,
                                                                     size_t mesh_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
beampropagation_initialize_plane_wrapper(const CU_REAL* xs,
                                         const CU_REAL* ys,
                                         const CU_REAL* zs,
                                         const CUCOMPLEX* epsilon,
                                         const CUCOMPLEX* incident,
                                         CUCOMPLEX* image_planes,
                                         CU_INT* first_plane_map,
                                         CUCOMPLEX* local_field,
                                         int nx,
                                         int ny,
                                         int nfft2d,
                                         int nmax,
                                         int niso,
                                         int ndipole,
                                         int kkm,
                                         int jjm,
                                         CU_REAL xmin,
                                         CU_REAL ymin,
                                         CU_REAL first_z,
                                         CU_REAL aretecube,
                                         CU_REAL k0,
                                         CUCOMPLEX E0,
                                         CU_REAL ss,
                                         CU_REAL pp,
                                         CU_REAL theta_degrees,
                                         CU_REAL phi_degrees,
                                         CU_REAL xgaus,
                                         CU_REAL ygaus,
                                         CU_REAL zgaus,
                                         CU_REAL w0,
                                         int beam_type);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scalar_chan_transpose_wrapper(
    const CUCOMPLEX* fft_channel, CUCOMPLEX* chan_vectors, int nx, int ny, int nz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scalar_chan_build_blocks_wrapper(
    const CUCOMPLEX* chan_vectors, CUCOMPLEX* matrices, int nx, int ny, int nz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
scalar_preconditioner_gather_wrapper(const CUCOMPLEX* field,
                                     const CU_INT* dipole_map,
                                     CUCOMPLEX* fft_planes,
                                     int nx,
                                     int ny,
                                     int nz,
                                     int nxm,
                                     int nym,
                                     int nzm);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scalar_preconditioner_fft_to_rhs_wrapper(
    const CUCOMPLEX* fft_planes, CUCOMPLEX* rhs, int nx, int ny, int nz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scalar_preconditioner_rhs_to_fft_wrapper(
    const CUCOMPLEX* rhs, CUCOMPLEX* fft_planes, int nx, int ny, int nz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t scalar_preconditioner_matvec_wrapper(
    const CUCOMPLEX* matrices, const CUCOMPLEX* rhs, CUCOMPLEX* output, int nx, int ny, int nz);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
scalar_preconditioner_scatter_wrapper(const CUCOMPLEX* fft_planes,
                                      const CU_INT* dipole_map,
                                      CUCOMPLEX* field,
                                      int nx,
                                      int ny,
                                      int nz,
                                      int nxm,
                                      int nym,
                                      int nzm,
                                      CU_REAL normalization);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
optical_force_density_wrapper(const CUCOMPLEX* polarization,
                              const CUCOMPLEX* field_derivatives,
                              CU_REAL* force_x,
                              CU_REAL* force_y,
                              CU_REAL* force_z,
                              int dipole_count,
                              int mesh_size,
                              CU_REAL scale);
PRODUITMATVECTFFT_KERNEL_API cudaError_t optical_force_fill_reduction_ones_wrapper(CU_REAL* ones,
                                                                                   int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t optical_force_group_sum_wrapper(const CU_INT* object_ids,
                                                                         const CU_REAL* force_x,
                                                                         const CU_REAL* force_y,
                                                                         const CU_REAL* force_z,
                                                                         CU_REAL* object_sums,
                                                                         int dipole_count,
                                                                         int object_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t optical_torque_group_sum_wrapper(const CU_INT* object_ids,
                                                                          const CU_REAL* torque_x,
                                                                          const CU_REAL* torque_y,
                                                                          const CU_REAL* torque_z,
                                                                          CU_REAL* object_sums,
                                                                          int dipole_count,
                                                                          int object_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t optical_field_derivative_wrapper(const CUCOMPLEX* field,
                                                                          CUCOMPLEX* derivative,
                                                                          int nx,
                                                                          int ny,
                                                                          int nz,
                                                                          int axis,
                                                                          CU_REAL spacing);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
optical_force_gather_fft_wrapper(const CUCOMPLEX* polarization,
                                 const CU_INT* dipole_map,
                                 CUCOMPLEX* fft_vectors,
                                 int nx,
                                 int ny,
                                 int nz,
                                 int nx2,
                                 int ny2,
                                 int nz2,
                                 int padded_count,
                                 int use_map);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
optical_force_build_green_derivative_wrapper(CUCOMPLEX* green_derivative,
                                             int nx,
                                             int ny,
                                             int nz,
                                             int nx2,
                                             int ny2,
                                             int nz2,
                                             int nxy2,
                                             int padded_count,
                                             int axis,
                                             CU_REAL spacing,
                                             CU_REAL k0);
PRODUITMATVECTFFT_KERNEL_API cudaError_t optical_force_derivative_spectral_product_wrapper(
    const CUCOMPLEX* fft_vectors, CUCOMPLEX* green_derivative, int padded_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
optical_force_extract_green_derivative_wrapper(const CUCOMPLEX* green_derivative,
                                               CUCOMPLEX* output,
                                               int nx,
                                               int ny,
                                               int nz,
                                               int nx2,
                                               int nxy2,
                                               int padded_count,
                                               int physical_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
optical_torque_density_wrapper(const CUCOMPLEX* polarization,
                               const CUCOMPLEX* local_field,
                               const CU_REAL* xs,
                               const CU_REAL* ys,
                               const CU_REAL* zs,
                               const CU_REAL* force_x,
                               const CU_REAL* force_y,
                               const CU_REAL* force_z,
                               CU_REAL* torque_x,
                               CU_REAL* torque_y,
                               CU_REAL* torque_z,
                               int dipole_count,
                               CU_REAL k03,
                               CU_REAL center_x,
                               CU_REAL center_y,
                               CU_REAL center_z,
                               CU_REAL scale);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pml_build_mask_candidates_wrapper(const CU_REAL* xs,
                                                                           const CU_REAL* ys,
                                                                           const CU_REAL* zs,
                                                                           CU_REAL* mask,
                                                                           CU_REAL* z_candidates,
                                                                           int dipole_count,
                                                                           CU_REAL xmin,
                                                                           CU_REAL xmax,
                                                                           CU_REAL ymin,
                                                                           CU_REAL ymax,
                                                                           CU_REAL zmin,
                                                                           CU_REAL zmax,
                                                                           CU_REAL aretecube,
                                                                           CU_REAL lneps,
                                                                           CU_REAL lpml,
                                                                           CU_REAL dpml);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pml_reduce_zinf_wrapper(const CU_REAL* z_candidates,
                                                                 CU_REAL* zinf,
                                                                 int dipole_count,
                                                                 CU_REAL fallback_zinf);
PRODUITMATVECTFFT_KERNEL_API cudaError_t pml_apply_field_wrapper(const CU_REAL* mask,
                                                                 CUCOMPLEX* field,
                                                                 int dipole_count,
                                                                 CU_REAL field_scale);
PRODUITMATVECTFFT_KERNEL_API cudaError_t rigorous_gaussian_incident_wrapper(const CU_REAL* xs,
                                                                            const CU_REAL* ys,
                                                                            const CU_REAL* zs,
                                                                            int count,
                                                                            double x0,
                                                                            double y0,
                                                                            double z0,
                                                                            double theta_degrees,
                                                                            double phi_degrees,
                                                                            double waist,
                                                                            double k0,
                                                                            double ss,
                                                                            double pp,
                                                                            COMPLEX e0,
                                                                            double tolerance,
                                                                            int circular,
                                                                            int nloin,
                                                                            const double* nodes,
                                                                            const double* weights,
                                                                            CUCOMPLEX* field,
                                                                            CU_INT* stop);
PRODUITMATVECTFFT_KERNEL_API cudaError_t rigorous_gaussian_incident_grid_wrapper(
    IfddaIncidentGrid grid, double x0, double y0, double z0,
    double theta_degrees, double phi_degrees, double waist, double k0,
    double ss, double pp, COMPLEX e0, double tolerance, int circular, int nloin,
    const double* nodes, const double* weights, CUCOMPLEX* field, CU_INT* stop);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
rigorous_gaussian_fortran_cutoff_wrapper(const CU_REAL* xs,
                                         const CU_REAL* ys,
                                         const CU_REAL* zs,
                                         int count,
                                         double x0,
                                         double y0,
                                         double z0,
                                         double theta_degrees,
                                         double phi_degrees,
                                         double waist,
                                         double k0,
                                         double ss,
                                         double pp,
                                         COMPLEX e0,
                                         int circular,
                                         CUCOMPLEX* field);
PRODUITMATVECTFFT_KERNEL_API cudaError_t rigorous_gaussian_fortran_cutoff_grid_wrapper(
    IfddaIncidentGrid grid, double x0, double y0, double z0,
    double theta_degrees, double phi_degrees, double waist, double k0,
    double ss, double pp, COMPLEX e0, int circular, CUCOMPLEX* field);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
rigorous_gaussian_derivative_wrapper(CUCOMPLEX* six_point_fields, CU_REAL step);
PRODUITMATVECTFFT_KERNEL_API cudaError_t rigorous_gaussian_derivative_batch_wrapper(
    const CUCOMPLEX* six_point_fields, CUCOMPLEX* derivatives, int point_count, CU_REAL step);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
direct_scattering_angles_wrapper(const CUCOMPLEX* field,
                                 const CU_REAL* xs,
                                 const CU_REAL* ys,
                                 const CU_REAL* zs,
                                 int dipole_count,
                                 int ntheta,
                                 int nphi,
                                 CU_REAL k0,
                                 CU_REAL theta0,
                                 CU_REAL phi0,
                                 CU_REAL dtheta,
                                 CU_REAL dphi,
                                 CU_REAL const_poynting,
                                 CU_REAL* poynting,
                                 CU_REAL* theta_values,
                                 CU_REAL* phi_values,
                                 CU_REAL* weighted_scattering,
                                 CU_REAL* weighted_asymmetry,
                                 CU_REAL* reduction_ones);
PRODUITMATVECTFFT_KERNEL_API cudaError_t quick_energy_incident_wave_wrapper(CUCOMPLEX* incident,
                                                                            int nfft,
                                                                            CU_REAL spacing,
                                                                            CU_REAL scale,
                                                                            CU_REAL kx,
                                                                            CU_REAL ky,
                                                                            CUCOMPLEX kz,
                                                                            CUCOMPLEX ex0,
                                                                            CUCOMPLEX ey0,
                                                                            CUCOMPLEX ez0);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
quick_energy_finalize_fields_wrapper(CUCOMPLEX* positive_field,
                                     CUCOMPLEX* negative_field,
                                     int nfft,
                                     int imax,
                                     CU_REAL deltakx,
                                     CU_REAL deltaky,
                                     CU_REAL k0);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
quick_energy_flux_wrapper(const CUCOMPLEX* incident_spectrum,
                          const CUCOMPLEX* positive_field,
                          const CUCOMPLEX* negative_field,
                          int nfft,
                          int imax,
                          CU_REAL deltakx,
                          CU_REAL deltaky,
                          CU_REAL k0,
                          CU_REAL* flux_incident,
                          CU_REAL* flux_reflected,
                          CU_REAL* flux_transmitted,
                          CU_REAL* energy_inside_na,
                          CU_REAL* energy_total,
                          CU_REAL* reduction_ones);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_energy_spectrum_wrapper(CUCOMPLEX* incident,
                                 CUCOMPLEX* positive_incident,
                                 CUCOMPLEX* negative_incident,
                                 const CU_REAL* phase_positive,
                                 const CU_REAL* phase_negative,
                                 int nfft,
                                 int imax,
                                 CU_REAL k0,
                                 CU_REAL deltakx,
                                 CU_REAL deltaky,
                                 CU_REAL kmax_squared,
                                 CU_REAL psi,
                                 CU_REAL x0,
                                 CU_REAL y0,
                                 CU_REAL z0,
                                 CUCOMPLEX e0,
                                 int nside,
                                 int beam_kind);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_energy_flux_wrapper(const CUCOMPLEX* incident_positive,
                             const CUCOMPLEX* incident_negative,
                             const CUCOMPLEX* scattered_positive,
                             const CUCOMPLEX* scattered_negative,
                             int nfft,
                             int imax,
                             CU_REAL deltakx,
                             CU_REAL deltaky,
                             CU_REAL k0,
                             CU_REAL kmax_squared,
                             CU_REAL* flux_incident,
                             CU_REAL* flux_reflected,
                             CU_REAL* flux_transmitted);
PRODUITMATVECTFFT_KERNEL_API cudaError_t confocal_fluorescence_zero_wrapper(CU_REAL* intensity,
                                                                            size_t count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t confocal_fluorescence_accumulate_wrapper(
    const CUCOMPLEX* field_planes, CU_REAL* intensity, size_t plane_size, size_t plane_index);
PRODUITMATVECTFFT_KERNEL_API cudaError_t confocal_fluorescence_square_wrapper(CU_REAL* intensity,
                                                                              size_t count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_intensity_sqrt_wrapper(CU_REAL* intensity,
                                                                           size_t count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t confocal_fluorescence_extract_plane_wrapper(
    const CU_REAL* intensity, CUCOMPLEX* output, size_t plane_size, size_t plane_index);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_fluorescence_phase_wrapper(CUCOMPLEX* field_planes,
                                    int fft_size,
                                    CU_REAL aretecube,
                                    CU_REAL kxinc,
                                    CU_REAL kyinc,
                                    CU_REAL kzinc,
                                    CU_REAL lens_z);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_fluorescence_rotate_wrapper(CUCOMPLEX* field_planes,
                                     int fft_size,
                                     int max_index,
                                     CU_REAL delta_kx,
                                     CU_REAL delta_ky,
                                     CU_REAL k0,
                                     CU_REAL gross,
                                     int side,
                                     CU_REAL aperture);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_fluorescence_build_wrapper(const CUCOMPLEX* diffraction,
                                    CUCOMPLEX* spectrum_planes,
                                    int fft_size,
                                    int max_index,
                                    CU_REAL delta_kx,
                                    CU_REAL delta_ky,
                                    CU_REAL k0,
                                    CU_REAL aperture,
                                    CU_REAL lens_z,
                                    int side);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_direct_diffraction_wrapper(const CUCOMPLEX* dipole_field,
                                    const CU_REAL* xs,
                                    const CU_REAL* ys,
                                    const CU_REAL* zs,
                                    int dipole_count,
                                    int nfft,
                                    int max_index,
                                    int side,
                                    CU_REAL k0,
                                    CU_REAL aperture,
                                    CU_REAL delta_kx,
                                    CU_REAL delta_ky,
                                    CUCOMPLEX* diffraction);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
microscope_expand_packed_field_wrapper(const CUCOMPLEX* packed_field,
                                       const CU_INT* dipole_map,
                                       CUCOMPLEX* dense_interleaved_field,
                                       int cell_count,
                                       int dipole_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t direct_scattering_spectrum_wrapper(const CUCOMPLEX* field,
                                                                            const CU_REAL* xs,
                                                                            const CU_REAL* ys,
                                                                            const CU_REAL* zs,
                                                                            int dipole_count,
                                                                            int ll,
                                                                            int np,
                                                                            CU_REAL k0,
                                                                            CU_REAL delta,
                                                                            CU_REAL const_poynting,
                                                                            CU_REAL* positive,
                                                                            CU_REAL* negative,
                                                                            CU_REAL* kx_values);
PRODUITMATVECTFFT_KERNEL_API cudaError_t quick_diffraction_gather_wrapper(const CUCOMPLEX* field,
                                                                          CUCOMPLEX* images,
                                                                          const CU_INT* fft_map,
                                                                          int nx,
                                                                          int ny,
                                                                          int nxny,
                                                                          int nfft,
                                                                          int plane_index);
PRODUITMATVECTFFT_KERNEL_API cudaError_t quick_diffraction_build_fft_map_wrapper(CU_INT* fft_map,
                                                                                 int nfft);
PRODUITMATVECTFFT_KERNEL_API cudaError_t fourier_to_image_rearrange_wrapper(CUCOMPLEX* fields,
                                                                            int nfft,
                                                                            CU_REAL scale);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
bright_field_prepare_fourier_wrapper(const CUCOMPLEX* diffraction,
                                     CUCOMPLEX* fourier_fields,
                                     int nfft,
                                     int max_index,
                                     CU_REAL delta_kx,
                                     CU_REAL delta_ky,
                                     CU_REAL k0,
                                     CU_REAL aperture,
                                     CU_REAL lens_z,
                                     CU_REAL gross,
                                     int side,
                                     int microscope_type,
                                     CU_REAL aperture_inner,
                                     int incident_index_x,
                                     int incident_index_y,
                                     CU_REAL incident_scale,
                                     CUCOMPLEX incident_x,
                                     CUCOMPLEX incident_y,
                                     CUCOMPLEX incident_z);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
bright_field_image_accumulate_wrapper(const CUCOMPLEX* fourier_fields,
                                      CU_REAL* image_intensity,
                                      size_t plane_size,
                                      int first_component,
                                      int component_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
microsconfocal_detector_prepare_wrapper(const CUCOMPLEX* diffraction,
                                        CUCOMPLEX* fourier_fields,
                                        int nfft,
                                        int max_index,
                                        CU_REAL k0,
                                        CU_REAL aperture,
                                        CU_REAL delta_kx,
                                        CU_REAL delta_ky,
                                        CU_REAL gross,
                                        CU_REAL lens_z,
                                        int side,
                                        int incident_index_x,
                                        int incident_index_y,
                                        CU_REAL incident_scale,
                                        CUCOMPLEX incident_x,
                                        CUCOMPLEX incident_y,
                                        CUCOMPLEX incident_z,
                                        CU_INT* device_status);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
microsconfocal_detector_accumulate_wrapper(const CUCOMPLEX* fourier_fields,
                                           CUCOMPLEX* image_stack,
                                           int nfft,
                                           int plane_index,
                                           int plane_count,
                                           CU_REAL pixel_size,
                                           CU_REAL kx_incident,
                                           CU_REAL ky_incident,
                                           CU_REAL kz_incident,
                                           CU_REAL lens_z);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
microsconfocal_detector_zero_wrapper(CUCOMPLEX* image_stack, size_t value_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
quick_diffraction_accumulate_wrapper(const CUCOMPLEX* images,
                                     const CU_INT* fft_map,
                                     const CU_REAL* zs,
                                     int nxny,
                                     int nfft,
                                     int imax,
                                     int plane_index,
                                     CU_REAL k0,
                                     CU_REAL deltak,
                                     CU_REAL phase_x,
                                     CU_REAL phase_y,
                                     CUCOMPLEX* positive,
                                     CUCOMPLEX* negative);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
quick_diffraction_postprocess_wrapper(const CUCOMPLEX* positive_field,
                                      const CUCOMPLEX* negative_field,
                                      int nfft,
                                      int imax,
                                      CU_REAL deltakx,
                                      CU_REAL deltaky,
                                      CU_REAL k0,
                                      CU_REAL theta0,
                                      CU_REAL phi0,
                                      CU_REAL const_poynting,
                                      int ntheta,
                                      int nphi,
                                      CU_REAL* positive_spectrum,
                                      CU_REAL* negative_spectrum,
                                      CU_REAL* kx_values,
                                      CU_REAL* angular_poynting,
                                      CU_REAL* theta_values,
                                      CU_REAL* phi_values,
                                      CU_REAL* weighted_scattering,
                                      CU_REAL* weighted_asymmetry,
                                      CU_REAL* reduction_ones);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
arbitrary_illumination_interpolate_wrapper(const CU_REAL* xs,
                                           const CU_REAL* ys,
                                           const CU_REAL* zs,
                                           int dipole_count,
                                           const CUCOMPLEX* grid_x,
                                           const CUCOMPLEX* grid_y,
                                           const CUCOMPLEX* grid_z,
                                           int nx,
                                           int ny,
                                           int nz,
                                           CU_REAL xmin,
                                           CU_REAL ymin,
                                           CU_REAL zmin,
                                           CU_REAL dx,
                                           CU_REAL dy,
                                           CU_REAL dz,
                                           CUCOMPLEX* output);
PRODUITMATVECTFFT_KERNEL_API cudaError_t arbitrary_derivative_grid_pack_wrapper(
    const CUCOMPLEX* interleaved, CUCOMPLEX* component_major, size_t site_count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
negative_kz_diffraction_spectrum_wrapper(const CUCOMPLEX* random_phases,
                                         CUCOMPLEX* spectrum,
                                         int nfft,
                                         int imax,
                                         CU_REAL deltakx,
                                         CU_REAL deltaky,
                                         CU_REAL k0,
                                         CU_REAL numerical_aperture,
                                         CU_REAL polarization_angle,
                                         CU_REAL x0,
                                         CU_REAL y0,
                                         CU_REAL z0,
                                         CUCOMPLEX incident_amplitude,
                                         int speckle);
PRODUITMATVECTFFT_KERNEL_API cudaError_t speckle_embed_negative_kz_phase_wrapper(
    const CUCOMPLEX* source_phases, CUCOMPLEX* dense_phases, int source_size, int nfft, int imax);
PRODUITMATVECTFFT_KERNEL_API cudaError_t lens_diffraction_clear_wrapper(CUCOMPLEX* spectrum,
                                                                        int count);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
lens_diffraction_pack_layer_wrapper(const CUCOMPLEX* green_field,
                                    CUCOMPLEX* layer_spectrum,
                                    const CU_INT* fft_indices,
                                    int nx,
                                    int ny,
                                    int nfft,
                                    int layer);
PRODUITMATVECTFFT_KERNEL_API cudaError_t
lens_diffraction_accumulate_wrapper(const CUCOMPLEX* layer_spectrum,
                                    CUCOMPLEX* output,
                                    const CU_INT* fft_indices,
                                    int nfft,
                                    int imax,
                                    CU_REAL deltakx,
                                    CU_REAL deltaky,
                                    CU_REAL k0,
                                    CU_REAL numerical_aperture,
                                    CU_REAL dside,
                                    CU_REAL var1,
                                    CU_REAL var2,
                                    CU_REAL z);
PRODUITMATVECTFFT_KERNEL_API cudaError_t quicklens_direct_sum_wrapper(const CUCOMPLEX* field,
                                                                      const CU_REAL* xs,
                                                                      const CU_REAL* ys,
                                                                      const CU_REAL* zs,
                                                                      const CUCOMPLEX* incident,
                                                                      CUCOMPLEX* scattered,
                                                                      CUCOMPLEX* combined,
                                                                      int dipole_count,
                                                                      int nfft,
                                                                      int half,
                                                                      int imax,
                                                                      CU_REAL dkx,
                                                                      CU_REAL k0,
                                                                      CU_REAL na,
                                                                      int side,
                                                                      CU_REAL zlens);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // PRODUITMATVECTFFT_KERNEL_H
