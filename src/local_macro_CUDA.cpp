#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cstdio>
#include <complex>
#include <limits>
#include <vector>

//-----------------------------------------------------------
extern "C" void ifdda_local_macro_batch_cuda_(
    const CUCOMPLEX *epsilon, const CUCOMPLEX *input_field,
    CUCOMPLEX *output_x, CUCOMPLEX *output_y, CUCOMPLEX *output_z,
    CU_REAL *field_magnitude, const int *nmax, const int *niso,
    const int *ndipoles, const CU_REAL *aretecube, const CU_REAL *k0,
    const int *nsens, int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!epsilon || !input_field || !output_x || !output_y || !output_z ||
        !field_magnitude || !nmax || !niso || !ndipoles || !aretecube ||
        !k0 || !nsens || *nmax <= 0 || *ndipoles <= 0 || *ndipoles > *nmax ||
        (*niso != 1 && *niso != 3) || (*nsens != 1 && *nsens != -1) ||
        *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0))
        return;

    const size_t dipoles = static_cast<size_t>(*ndipoles);
    const size_t isotropy = static_cast<size_t>(*niso);
    const size_t maximum = static_cast<size_t>(*nmax);
    if (maximum > std::numeric_limits<size_t>::max() / isotropy / isotropy ||
        dipoles > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        maximum * isotropy * isotropy >
            std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return;
    if (!initialize_cuda_device())
        return;

    const size_t tensor_count = maximum * isotropy * isotropy;
    const size_t field_count = 3 * dipoles;
    const size_t tensor_bytes = tensor_count * sizeof(CUCOMPLEX);
    const size_t field_bytes = field_count * sizeof(CUCOMPLEX);
    cuda_data::LocalMacroDeviceBuffers buffers{};
    if (!cuda_data::get_device_local_macro_buffers(tensor_count, dipoles, &buffers))
        return;

    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.epsilon, epsilon, tensor_bytes, cuda_utils::TransferCategory::Material);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.input_field, input_field, field_bytes,
            cuda_utils::TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = cudaMemset(buffers.status, 0, sizeof(int));
    if (status == cudaSuccess)
        status = local_macro_batch_wrapper(
            buffers.epsilon, buffers.input_field, buffers.output_field,
            buffers.field_magnitude, buffers.status, *niso, *nmax, *ndipoles,
            *aretecube, *k0, *nsens);

    int kernel_status = 1;
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            &kernel_status, buffers.status, sizeof(int),
            cuda_utils::TransferCategory::Result);
    std::vector<CUCOMPLEX> output;
    if (status == cudaSuccess && kernel_status == 0)
        output.resize(field_count);
    if (status == cudaSuccess && kernel_status == 0)
        status = cuda_data::copy_device_to_host(
            output.data(), buffers.output_field, field_bytes,
            cuda_utils::TransferCategory::Result);
    if (status == cudaSuccess && kernel_status == 0)
        status = cuda_data::copy_device_to_host(
            field_magnitude, buffers.field_magnitude, dipoles * sizeof(CU_REAL),
            cuda_utils::TransferCategory::Result);

    if (status != cudaSuccess || kernel_status != 0) {
        std::fprintf(stderr, "CUDA local-macro conversion failed: %s\n",
                     status == cudaSuccess ? "singular local matrix"
                                           : cudaGetErrorString(status));
        return;
    }
    for (size_t i = 0; i < dipoles; ++i) {
        output_x[i] = output[3 * i];
        output_y[i] = output[3 * i + 1];
        output_z[i] = output[3 * i + 2];
    }
    *nstop = 0;
}

//-----------------------------------------------------------
extern "C" void local_macro_(std::complex<CU_REAL> *output,
    const std::complex<CU_REAL> *input,
    const std::complex<CU_REAL> *epsilon,
    const CU_REAL *aretecube, const CU_REAL *k0, const int *nsens)
{
    if (!output || !input || !epsilon || !aretecube || !k0 || !nsens)
        return;

    CUCOMPLEX device_epsilon[9];
    CUCOMPLEX device_input[3];
    CUCOMPLEX output_x[1];
    CUCOMPLEX output_y[1];
    CUCOMPLEX output_z[1];
    CU_REAL magnitude[1]{};
    for (int i = 0; i < 9; ++i) {
        device_epsilon[i] = CUCOMPLEX{epsilon[i].real(), epsilon[i].imag()};
    }
    for (int i = 0; i < 3; ++i) {
        device_input[i] = CUCOMPLEX{input[i].real(), input[i].imag()};
    }

    const int nmax = 1;
    const int niso = 3;
    const int ndipoles = 1;
    int status = 1;
    ifdda_local_macro_batch_cuda_(
        device_epsilon, device_input, output_x, output_y, output_z, magnitude,
        &nmax, &niso, &ndipoles, aretecube, k0, nsens, &status);
    if (status != 0) {
        std::fprintf(stderr, "CUDA local_macro_ failed for one dipole\\n");
        return;
    }

    output[0] = std::complex<CU_REAL>(output_x[0].x, output_x[0].y);
    output[1] = std::complex<CU_REAL>(output_y[0].x, output_y[0].y);
    output[2] = std::complex<CU_REAL>(output_z[0].x, output_z[0].y);
}
