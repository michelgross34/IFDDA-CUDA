#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cstdio>
#include <cstring>

//-----------------------------------------------------------
extern "C" void abcchi_(const int* profile_type, const int* i, const int* j, const int* k,
                         const int* nx, const int* ny, const int* nz, const int* npml,
                         CUCOMPLEX* epsilon, const int* isotropy, const int* nmax,
                         const int* dipole_count, const CUCOMPLEX* epsilon_value,
                         const CUCOMPLEX* epsilon_background, const CUCOMPLEX* epsilon_anisotropic,
                         const CU_REAL* k0, const char* method, const CU_REAL* contrast_scale,
                         const CU_REAL* cell_size, const int* boundary_count,
                         const int* pml_layers, const int* neps, const int* dipole_map,
                         CUCOMPLEX* contrast, std::size_t method_length) {
    (void)npml;
    (void)epsilon_anisotropic;
    (void)k0;
    (void)method;
    (void)cell_size;
    (void)boundary_count;
    (void)pml_layers;
    (void)dipole_map;
    (void)method_length;
    if (!profile_type || !i || !j || !k || !nx || !ny || !nz || !epsilon || !isotropy ||
        !nmax || !dipole_count || !epsilon_value || !epsilon_background || !contrast_scale ||
        !neps || !contrast || *nmax <= 0 || *dipole_count <= 0 ||
        (*isotropy != 1 && *isotropy != 3) ||
        *dipole_count > *nmax || *i <= 0 || *j <= 0 || *k <= 0 || *i > *nx || *j > *ny || *k > *nz) {
        std::fprintf(stderr, "[abcchi CUDA] invalid Fortran arguments\n");
        return;
    }
    if (!initialize_cuda_device()) {
        std::fprintf(stderr, "[abcchi CUDA] CUDA device initialization failed\n");
        return;
    }
    cuda_data::AbcChiDeviceBuffers buffers{};
    if (!cuda_data::get_device_abcchi_buffers(&buffers) ||
        !cuda_utils::check_cuda(abcchi_profile_wrapper(
                                    buffers.epsilon_output, buffers.contrast_output,
                                    *epsilon_value, *epsilon_background, *contrast_scale,
                                    *profile_type, *i, *j, *k, *nx, *ny, *nz, *neps, *isotropy),
                                "abcchi CUDA", "compute boundary material")) {
        std::fprintf(stderr, "[abcchi CUDA] unable to run boundary material kernel\n");
        return;
    }
    CUCOMPLEX result[9]{};
    CUCOMPLEX contrast_result{};
    if (cuda_data::copy_device_to_host(result, buffers.epsilon_output, sizeof(result),
                                       cuda_utils::TransferCategory::Material) != cudaSuccess ||
        cuda_data::copy_device_to_host(&contrast_result, buffers.contrast_output,
                                       sizeof(contrast_result),
                                       cuda_utils::TransferCategory::Material) != cudaSuccess) {
        std::fprintf(stderr, "[abcchi CUDA] unable to copy boundary material result\n");
        return;
    }
    const std::size_t base = std::size_t(*dipole_count - 1);
    epsilon[base] = result[0];
    if (*isotropy == 3) {
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                const std::size_t offset = base + std::size_t(row) * std::size_t(*nmax) +
                    std::size_t(column) * std::size_t(*nmax) * std::size_t(*isotropy);
                epsilon[offset] = result[row + 3 * column];
            }
        }
    }
    *contrast = contrast_result;
}
