#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>

namespace {
using Complex = std::complex<CU_REAL>;
constexpr CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);

//-----------------------------------------------------------
CU_REAL signed_unit(CU_REAL value) {
    return std::signbit(value) ? CU_REAL(-1) : CU_REAL(1);
}

//-----------------------------------------------------------
bool same_value(CU_REAL a, CU_REAL b) {
    return std::abs(a - b) <=
           CU_REAL(1e-12) * (std::abs(b) > CU_REAL(1) ? std::abs(b) : CU_REAL(1));
}

//-----------------------------------------------------------
bool plane_wave_components(CU_REAL k0, CUCOMPLEX epsilon_value, CUCOMPLEX e0_value,
                           CU_REAL ss_value, CU_REAL pp_value, CU_REAL theta_degrees,
                           CU_REAL phi_degrees, bool circular, CU_REAL *kx, CU_REAL *ky,
                           CUCOMPLEX *kz, CUCOMPLEX *ex, CUCOMPLEX *ey, CUCOMPLEX *ez) {
    if (!kx || !ky || !kz || !ex || !ey || !ez)
        return false;
    const CU_REAL theta = theta_degrees * pi / CU_REAL(180);
    const CU_REAL phi = phi_degrees * pi / CU_REAL(180);
    const CU_REAL st = std::sin(theta), ct = std::cos(theta);
    const CU_REAL sp = std::sin(phi), cp = std::cos(phi);
    const Complex e0 = cuda_utils::host_value(e0_value);

    if (circular) {
        const CU_REAL us = CU_REAL(1) / std::sqrt(CU_REAL(2));
        const Complex es = e0 / std::sqrt(CU_REAL(2));
        const Complex ep = e0 * Complex(0, std::signbit(ss_value) ? -us : us);
        *kx = k0 * st * cp;
        *ky = k0 * st * sp;
        *kz = cuda_utils::device_value(Complex(k0 * ct, 0));
        *ex = cuda_utils::device_value(-es * sp + ep * ct * cp);
        *ey = cuda_utils::device_value(es * cp + ep * ct * sp);
        *ez = cuda_utils::device_value(-ep * st);
        return true;
    }

    CU_REAL pp = pp_value;
    CU_REAL p = std::sqrt(pp);
    CU_REAL s = std::sqrt(CU_REAL(1) - pp);
    if (same_value(pp, CU_REAL(3))) {
        CU_REAL phi_deg = phi_degrees;
        if (phi_deg < 0)
            phi_deg += CU_REAL(360);
        pp = (sp * sp) / (ct * ct * cp * cp + sp * sp);
        if (phi_deg < 90 && phi_deg >= 0) {
            p = std::sqrt(pp);
            s = std::sqrt(1 - pp);
        } else if (phi_deg < 180 && phi_deg >= 90) {
            p = std::sqrt(pp);
            s = -std::sqrt(1 - pp);
        } else if (phi_deg < 270 && phi_deg >= 180) {
            p = -std::sqrt(pp);
            s = -std::sqrt(1 - pp);
        } else if (phi_deg < 360 && phi_deg >= 270) {
            p = -std::sqrt(pp);
            s = std::sqrt(1 - pp);
        }
    }
    if (same_value(pp, CU_REAL(2))) {
        pp = (cp * cp) / (ct * ct * sp * sp + cp * cp);
        CU_REAL phi_deg = phi_degrees;
        if (phi_deg < 0)
            phi_deg += CU_REAL(360);
        if (phi_deg < 90 && phi_deg >= 0) {
            p = std::sqrt(pp);
            s = -std::sqrt(1 - pp);
        } else if (phi_deg < 180 && phi_deg >= 90) {
            p = -std::sqrt(pp);
            s = -std::sqrt(1 - pp);
        } else if (phi_deg < 270 && phi_deg >= 180) {
            p = -std::sqrt(pp);
            s = std::sqrt(1 - pp);
        } else if (phi_deg < 360 && phi_deg >= 270) {
            p = std::sqrt(pp);
            s = std::sqrt(1 - pp);
        }
    }
    if (!(pp >= 0 && pp <= 1))
        return false;

    const CU_REAL direction = signed_unit(theta_degrees);
    const Complex es = e0 * (s * direction);
    const Complex ep = e0 * (p * direction);
    const Complex ex0 = -es * sp + ep * ct * cp;
    const Complex ey0 = es * cp + ep * ct * sp;
    const Complex epsilon = cuda_utils::host_value(epsilon_value);
    const CU_REAL eps_abs = std::abs(epsilon);
    if (eps_abs < CU_REAL(1.0000001) && eps_abs > CU_REAL(0.999999)) {
        *kx = k0 * st * cp;
        *ky = k0 * st * sp;
        *kz = cuda_utils::device_value(Complex(k0 * ct, 0));
        *ex = cuda_utils::device_value(ex0);
        *ey = cuda_utils::device_value(ey0);
        *ez = cuda_utils::device_value(-ep * st);
    } else {
        const Complex refractive = std::sqrt(epsilon);
        const CU_REAL kr = std::real(k0 * refractive);
        *kx = kr * st * cp;
        *ky = kr * st * sp;
        const Complex gamma = std::sqrt(k0 * k0 * epsilon - *kx * *kx - *ky * *ky);
        if (std::abs(gamma) < CU_REAL(1e-7) * k0)
            return false;
        *kz = cuda_utils::device_value(gamma);
        *ex = cuda_utils::device_value(ex0);
        *ey = cuda_utils::device_value(ey0);
        *ez = cuda_utils::device_value(-(ex0 * *kx + ey0 * *ky) / gamma);
    }
    return true;
}

//-----------------------------------------------------------
bool reduce_sum(cublasHandle_t handle, int count, const CU_REAL *values, const CU_REAL *ones,
                CU_REAL *result) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return cuda_utils::check_cublas(cublasSdot(handle, count, values, 1, ones, 1, result),
                                    "energy conservation", "reduce GPU flux");
#else
    return cuda_utils::check_cublas(cublasDdot(handle, count, values, 1, ones, 1, result),
                                    "energy conservation", "reduce GPU flux");
#endif
}
} // namespace

//-----------------------------------------------------------
extern "C" void ifdda_energy_conservation_plane_cuda_(
    const int *nx, const int *ny, const int *nz, const int *nfft, int *fft_map, const CU_REAL *k0,
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *spacing,
    const CUCOMPLEX *epsilon, const CUCOMPLEX *e0, const CU_REAL *ss, const CU_REAL *pp,
    const CU_REAL *theta, const CU_REAL *phi, const int *circular, const CUCOMPLEX *field,
    int *imax_out, CU_REAL *dkx_out, CU_REAL *dky_out, CU_REAL *efficiency, CU_REAL *reflectivity,
    CU_REAL *transmissivity, CU_REAL *incident_flux_w, CU_REAL *reflected_flux_w,
    CU_REAL *transmitted_flux_w, CU_REAL *energy_outside_percent, int *status_out) {
    if (status_out)
        *status_out = 1;
    if (!nx || !ny || !nz || !nfft || !fft_map || !k0 || !xs || !ys || !zs || !spacing ||
        !epsilon || !e0 || !ss || !pp || !theta || !phi || !circular || !field || !imax_out ||
        !dkx_out || !dky_out || !efficiency || !reflectivity || !transmissivity || !status_out ||
        !incident_flux_w || !reflected_flux_w || !transmitted_flux_w || !energy_outside_percent ||
        *nx <= 0 || *ny <= 0 || *nz <= 0 || *nfft < 8 || *nfft > 16384 || (*nfft & 1) != 0 ||
        *k0 <= CU_REAL(0) || *spacing <= CU_REAL(0) || (*circular != 0 && *circular != 1))
        return;

    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t nxny = size_t(*nx) * size_t(*ny);
    if (nxny > max_size / size_t(*nz))
        return;
    const size_t dipoles = nxny * size_t(*nz);
    const size_t nfft_size = size_t(*nfft);
    if (nfft_size > max_size / nfft_size)
        return;
    const size_t plane = nfft_size * nfft_size;
    if (dipoles > size_t(std::numeric_limits<int>::max()) ||
        plane > size_t(std::numeric_limits<int>::max()))
        return;

    const CU_REAL dkx = CU_REAL(2) * pi / (CU_REAL(*nfft) * *spacing);
    const CU_REAL dky = dkx;
    if (dkx >= *k0)
        return;
    const int imax = int(std::floor(*k0 / dkx + CU_REAL(0.5))) + 1;
    if (2 * imax + 1 > *nfft || 2 * imax + 1 < 7)
        return;
    const size_t real_bytes = dipoles * sizeof(CU_REAL);
    const size_t field_bytes = 3 * dipoles * sizeof(CUCOMPLEX);
    if (!initialize_cuda_device())
        return;

    cuda_data::EnergyConservationDeviceBuffers buffers{};
    if (!cuda_data::get_device_energy_conservation_buffers(dipoles, plane, nfft_size, &buffers))
        return;
    for (int i = 0; i < *nfft; ++i)
        fft_map[i] = i - *nfft / 2 >= 0 ? i - *nfft / 2 + 1 : *nfft / 2 + i + 1;

    using cuda_utils::TransferCategory;
    const CUCOMPLEX* resident_field = get_device_result_vector_alias(field, 3 * dipoles);
    const CUCOMPLEX* device_field = resident_field ? resident_field : buffers.dipole_field;
    cudaError_t status = resident_field ? cudaSuccess :
        cuda_data::copy_host_to_device(buffers.dipole_field, field, field_bytes,
                                       TransferCategory::Result);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(buffers.z_planes, zs, real_bytes,
                                                TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = quick_diffraction_build_fft_map_wrapper(buffers.fft_map, *nfft);
    if (status != cudaSuccess)
        return;

    if (cudaMemset(buffers.positive_field, 0, 3 * plane * sizeof(CUCOMPLEX)) != cudaSuccess ||
        cudaMemset(buffers.negative_field, 0, 3 * plane * sizeof(CUCOMPLEX)) != cudaSuccess)
        return;

    const cufftHandle plan = get_speckle_fft_plan(*nfft);
    if (!plan)
        return;
    const CU_REAL phase_x = (xs[0] + CU_REAL(*nfft / 2) * *spacing) * dkx;
    const CU_REAL phase_y = (ys[0] + CU_REAL(*nfft / 2) * *spacing) * dky;
    for (int k = 0; k < *nz; ++k) {
        status = quick_diffraction_gather_wrapper(device_field, buffers.fft_images,
                                                  buffers.fft_map, *nx, *ny, int(nxny), *nfft, k);
        if (status != cudaSuccess)
            return;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const cufftResult fft_status =
            cufftExecC2C(plan, buffers.fft_images, buffers.fft_images, CUFFT_FORWARD);
#else
        const cufftResult fft_status =
            cufftExecZ2Z(plan, buffers.fft_images, buffers.fft_images, CUFFT_FORWARD);
#endif
        if (fft_status != CUFFT_SUCCESS)
            return;
        status = quick_diffraction_accumulate_wrapper(
            buffers.fft_images, buffers.fft_map, buffers.z_planes, int(nxny), *nfft, imax, k, *k0,
            dkx, phase_x, phase_y, buffers.positive_field, buffers.negative_field);
        if (status != cudaSuccess)
            return;
    }

    CU_REAL kx = CU_REAL(0), ky = CU_REAL(0);
    CUCOMPLEX kz = CUFFT_ZERO, ex = CUFFT_ZERO, ey = CUFFT_ZERO, ez = CUFFT_ZERO;
    if (!plane_wave_components(*k0, *epsilon, *e0, *ss, *pp, *theta, *phi, *circular != 0, &kx, &ky,
                               &kz, &ex, &ey, &ez))
        return;
    const CU_REAL scale = CU_REAL(1) / dkx / dky / CU_REAL(plane);
    status = quick_energy_incident_wave_wrapper(buffers.fft_images, *nfft, *spacing, scale, kx, ky,
                                                kz, ex, ey, ez);
    if (status != cudaSuccess)
        return;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    cufftResult fft_status =
        cufftExecC2C(plan, buffers.fft_images, buffers.fft_images, CUFFT_FORWARD);
#else
    cufftResult fft_status =
        cufftExecZ2Z(plan, buffers.fft_images, buffers.fft_images, CUFFT_FORWARD);
#endif
    if (fft_status != CUFFT_SUCCESS)
        return;
    status = cudaMemcpy(buffers.incident_spectrum, buffers.fft_images,
                        3 * plane * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess)
        return;

    status = quick_energy_finalize_fields_wrapper(buffers.positive_field, buffers.negative_field,
                                                  *nfft, imax, dkx, dky, *k0);
    if (status != cudaSuccess)
        return;
    status = quick_energy_flux_wrapper(
        buffers.incident_spectrum, buffers.positive_field, buffers.negative_field, *nfft, imax, dkx,
        dky, *k0, buffers.flux_incident, buffers.flux_reflected, buffers.flux_transmitted,
        buffers.energy_inside_na, buffers.energy_total, buffers.reduction_ones);
    if (status != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        return;

    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle)
        return;
    CU_REAL flux_incident = CU_REAL(0), flux_reflected = CU_REAL(0);
    CU_REAL flux_transmitted = CU_REAL(0), energy_inside = CU_REAL(0);
    CU_REAL energy_total = CU_REAL(0);
    if (!reduce_sum(handle, int(plane), buffers.flux_incident, buffers.reduction_ones,
                    &flux_incident) ||
        !reduce_sum(handle, int(plane), buffers.flux_reflected, buffers.reduction_ones,
                    &flux_reflected) ||
        !reduce_sum(handle, int(plane), buffers.flux_transmitted, buffers.reduction_ones,
                    &flux_transmitted) ||
        !reduce_sum(handle, int(plane), buffers.energy_inside_na, buffers.reduction_ones,
                    &energy_inside) ||
        !reduce_sum(handle, int(plane), buffers.energy_total, buffers.reduction_ones,
                    &energy_total) ||
        !(flux_incident > CU_REAL(0)) || !(energy_total > CU_REAL(0)))
        return;

    *energy_outside_percent = std::abs(energy_inside - energy_total) / energy_total * CU_REAL(100);
    *imax_out = imax;
    *dkx_out = dkx;
    *dky_out = dky;
    *efficiency = (flux_reflected + flux_transmitted) / flux_incident;
    *reflectivity = flux_reflected / flux_incident;
    *transmissivity = flux_transmitted / flux_incident;
    const CU_REAL flux_scale = CU_REAL(4) * pi * pi * dkx * dky /
                               (*k0 * CU_REAL(8) * pi * CU_REAL(1e-7) * CU_REAL(299792458));
    *incident_flux_w = flux_incident * flux_scale;
    *reflected_flux_w = flux_reflected * flux_scale;
    *transmitted_flux_w = flux_transmitted * flux_scale;
    *status_out = *energy_outside_percent >= CU_REAL(1) ? 2 : 0;
}
