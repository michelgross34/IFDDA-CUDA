#include "cuda_data.h"
#include "cuda_utils.h"

#include <cstdio>
#include <cstdlib>
#include <complex>
#include <limits>

//-----------------------------------------------------------
extern "C" int ifdda_compute_cross_sections_cuda_(
    const COMPLEX *incident, const COMPLEX *polarization, const COMPLEX *local_field,
    const int *dipole_count, const REAL *k0, const REAL *incident_intensity,
    REAL *extinction, REAL *absorption, REAL *scattering) {
    if (!incident || !polarization || !local_field || !dipole_count || !k0 || !incident_intensity ||
        !extinction || !absorption || !scattering || *dipole_count <= 0 ||
        *dipole_count > std::numeric_limits<int>::max() / 3 || *incident_intensity == CU_REAL(0))
        return 1;

    const int count = 3 * *dipole_count;
    const size_t packed_count = size_t(count) * 3;
    CUCOMPLEX *device = get_device_result_vectors(packed_count);
    cublasHandle_t blas = cuda_utils::get_cublas_handle();
    if (!device || !blas)
        return 2;

    const size_t bytes = size_t(count) * sizeof(CUCOMPLEX);
    const CUCOMPLEX *device_incident = get_device_result_vector_alias(incident, size_t(count));
    const CUCOMPLEX *device_polarization =
        get_device_result_vector_alias(polarization, size_t(count));
    const CUCOMPLEX *device_local = get_device_result_vector_alias(local_field, size_t(count));
    if (!device_incident) {
        if (cuda_data::copy_host_to_device(device, incident, bytes,
                                           cuda_utils::TransferCategory::IncidentField) !=
            cudaSuccess)
            return 3;
        device_incident = device;
    }
    if (!device_polarization) {
        if (cuda_data::copy_host_to_device(device + count, polarization, bytes,
                                           cuda_utils::TransferCategory::SolverVector) !=
            cudaSuccess)
            return 3;
        device_polarization = device + count;
    }
    if (!device_local) {
        if (cuda_data::copy_host_to_device(device + 2 * count, local_field, bytes,
                                           cuda_utils::TransferCategory::Result) != cudaSuccess)
            return 3;
        device_local = device + 2 * count;
    }

    CUCOMPLEX extinction_dot{}, absorption_dot{}, norm_dot{};
    CU_REAL polarization_norm = CU_REAL(0);
    if (!cuda_utils::dotc(blas, count, device_incident, device_polarization, &extinction_dot,
                          "cross section CUDA extinction") ||
        !cuda_utils::dotc(blas, count, device_local, device_polarization, &absorption_dot,
                          "cross section CUDA absorption") ||
        !cuda_utils::dotc(blas, count, device_polarization, device_polarization, &norm_dot,
                          "cross section CUDA polarization norm"))
        return 4;

    polarization_norm = norm_dot.x;
    const CU_REAL pi = CU_REAL(3.14159265358979323846);
    const CU_REAL scale = CU_REAL(4) * pi * *k0 / *incident_intensity;
    *extinction = scale * extinction_dot.y;
    *absorption = scale * (absorption_dot.y -
                           CU_REAL(2) / CU_REAL(3) * (*k0) * (*k0) * (*k0) * polarization_norm);
    *scattering = *extinction - *absorption;
    if (*absorption <= *extinction * REAL(1.0e-12))
        *absorption = REAL(0);
    const char *trace = std::getenv("IFDDA_TRACE_CROSS_SECTIONS");
    if (trace && trace[0] == '1' && trace[1] == '\0') {
        double host_ext = 0.0, host_abs = 0.0, host_norm = 0.0;
        for (int i = 0; i < count; ++i) {
            host_ext += std::imag(std::conj(incident[i]) * polarization[i]);
            host_abs += std::imag(std::conj(local_field[i]) * polarization[i]);
            host_norm += std::norm(polarization[i]);
        }
        std::fprintf(stdout, "CDMLIB_CROSS_SECTION_HOST %.17g %.17g %.17g\n",
                     host_ext, host_abs, host_norm);
        std::fprintf(stdout,
                     "CDMLIB_CROSS_SECTION_RAW_CUDA %.17g %.17g %.17g %.17g\n",
                     static_cast<double>(extinction_dot.y),
                     static_cast<double>(absorption_dot.y),
                     static_cast<double>(CU_REAL(2) / CU_REAL(3) * (*k0) * (*k0) *
                                         (*k0) * polarization_norm),
                     static_cast<double>(absorption_dot.y -
                                         CU_REAL(2) / CU_REAL(3) * (*k0) * (*k0) *
                                         (*k0) * polarization_norm));
    }
    return 0;
}
