#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "cuda_solver_host_adapter.h"
#include "kernel.h"

#include <cstdio>
#include <cstring>


namespace {

//-----------------------------------------------------------
CUCOMPLEX complex_value(CU_REAL real, CU_REAL imag) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real, imag);
#else
    return make_cuDoubleComplex(real, imag);
#endif
}

//-----------------------------------------------------------
} // namespace

//-----------------------------------------------------------
extern "C" void rerangechamppola_(
    COMPLEX *field_local, COMPLEX *epsilon, COMPLEX *polarizability,
    const int *isotropy, const CU_REAL *cell_size, const CU_REAL *k0,
    COMPLEX *field, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    CU_REAL *xs_wide, CU_REAL *ys_wide, CU_REAL *zs_wide,
    const int *dipole_map, const int *multi_map, CU_REAL *pml_mask,
    int *nx, int *ny, int *nz, const int *nxm, const int *nym,
    const int *nzm, const int *dipole_count, int *active_count,
    const int *nmax, const char *geometry_type, const char *method,
    CU_REAL *sldr, int *near_field_domain,
    size_t geometry_type_length, size_t method_length) {
    (void)dipole_map;
    (void)multi_map;
    (void)sldr;
    (void)geometry_type_length;

    if (!field_local || !epsilon || !polarizability || !isotropy ||
        !cell_size || !k0 || !field || !xs || !ys || !zs || !xs_wide ||
        !ys_wide || !zs_wide || !pml_mask || !nx || !ny || !nz || !nxm ||
        !nym || !nzm || !dipole_count || !active_count || !nmax ||
        !geometry_type || !method || !near_field_domain || *dipole_count <= 0 ||
        *nmax <= 0 || *dipole_count > *nmax || *isotropy != 1 && *isotropy != 3 ||
        *cell_size <= CU_REAL(0)) {
        std::fprintf(stderr, "rerangechamppola CUDA: invalid arguments\n");
        return;
    }

    *active_count = 0;
    *near_field_domain = 1;
    for (int i = 0; i < *dipole_count; ++i) {
        xs_wide[i] = xs[i];
        ys_wide[i] = ys[i];
        zs_wide[i] = zs[i];
    }

    const bool isotropic = std::memcmp(geometry_type, "iso", 3) == 0;
    const CU_REAL scale = (*cell_size) * (*cell_size) * (*cell_size) /
                          (CU_REAL(4) * CU_REAL(3.1415926535897932384626433832795));
    const COMPLEX epsilon0(CU_REAL(1), CU_REAL(0));
    COMPLEX epsilon_tensor[9];
    COMPLEX polarizability_tensor[9];
    const int discretized = 1;
    const int invert_result = 1;

    for (int i = 0; i < *dipole_count; ++i) {
        if (pml_mask[i] < CU_REAL(0.999999))
            continue;

        const int destination = *active_count;
        ++*active_count;
        xs[destination] = xs[i];
        ys[destination] = ys[i];
        zs[destination] = zs[i];
        field_local[3 * destination] = field_local[3 * i];
        field_local[3 * destination + 1] = field_local[3 * i + 1];
        field_local[3 * destination + 2] = field_local[3 * i + 2];

        if (isotropic) {
            epsilon[destination] = epsilon[i];
        } else {
            for (int column = 0; column < 3; ++column) {
                for (int row = 0; row < 3; ++row) {
                    const size_t source_index = size_t(i) + size_t(*nmax) *
                        size_t(row + 3 * column);
                    const size_t destination_index = size_t(destination) + size_t(*nmax) *
                        size_t(row + 3 * column);
                    epsilon_tensor[row + 3 * column] = epsilon[source_index];
                    epsilon[destination_index] = epsilon[source_index];
                }
            }

            polaepstens_cuda(cell_size, epsilon_tensor, &epsilon0, k0,
                          &discretized, method, &invert_result,
                          polarizability_tensor, method_length);
            for (int component = 0; component < 9; ++component) {
                const size_t destination_index = size_t(destination) + size_t(*nmax) *
                    size_t(component);
                polarizability[destination_index] = polarizability_tensor[component];
            }
        }
    }

    if (*active_count <= 0) {
        std::fprintf(stderr, "rerangechamppola CUDA: no dipoles remain after PML mask\n");
        return;
    }

    const auto extent = [cell_size, xs, ys, zs, active_count](int axis) {
        const CU_REAL first = axis == 0 ? xs[0] : (axis == 1 ? ys[0] : zs[0]);
        const CU_REAL last = axis == 0 ? xs[*active_count - 1] :
                             (axis == 1 ? ys[*active_count - 1] : zs[*active_count - 1]);
        return int((last - first) / *cell_size + CU_REAL(0.5)) + 1;
    };
    *nx = extent(0);
    *ny = extent(1);
    *nz = extent(2);

    std::printf(" Restriction %d %d %d %d\n", *nx, *ny, *nz, *active_count);
    std::printf(" Restriction x %.9g %.9g\n", double(xs[0]),
                double(xs[*active_count - 1]));
    std::printf(" Restriction y %.9g %.9g\n", double(ys[0]),
                double(ys[*active_count - 1]));
    std::printf(" Restriction z %.9g %.9g\n", double(zs[0]),
                double(zs[*active_count - 1]));

    if (size_t(*nx) * size_t(*ny) * size_t(*nz) != size_t(*active_count)) {
        std::fprintf(stderr, "rerangechamppola CUDA: compacted extent does not match dipole count\n");
        return;
    }

    if (isotropic) {
        if (!initialize_cuda_device())
            return;
        cuda_data::PolarizabilityInitDeviceBuffers buffers{};
        if (!cuda_data::get_device_polarizability_init_buffers(
                0, size_t(*active_count), &buffers))
            return;
        cudaError_t status = cuda_data::copy_host_complex_to_device(
            buffers.epsilon, epsilon, size_t(*active_count),
            cuda_utils::TransferCategory::Material);
        if (status == cudaSuccess)
            status = rerange_isotropic_polarizability_wrapper(
                buffers.epsilon, buffers.polarizability, *active_count,
                complex_value(CU_REAL(1), CU_REAL(0)), scale);
        if (status == cudaSuccess)
            status = cuda_data::copy_device_complex_to_host(
                polarizability, buffers.polarizability, size_t(*active_count),
                cuda_utils::TransferCategory::Material);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "rerangechamppola CUDA: isotropic kernel failed: %s\n",
                         cudaGetErrorString(status));
            return;
        }
    }

    std::printf(" reduction %d %d\n", *active_count, *dipole_count);
    // All three public arrays are CPU-resident COMPLEX. No reinterpretation
    // of the host arrays as CUDA pointers is needed for the product.
    produitpolavect_(polarizability, isotropy, nmax, active_count,
                     field_local, field);
    (void)nxm;
    (void)nym;
    (void)nzm;
}
