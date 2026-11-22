#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "gaussian_gkp_tables.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

using HostComplex = std::complex<CU_REAL>;




namespace {
constexpr CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);
std::mutex gaussian_fft_mutex;
bool gaussian_gkp_uploaded = false;

//-----------------------------------------------------------
void report_error(int* nstop, char* infostr, size_t length, const char* message) {
    if (nstop)
        *nstop = 1;
    if (!infostr)
        return;
    std::memset(infostr, ' ', length);
    std::memcpy(infostr, message, std::min(length, std::strlen(message)));
}

//-----------------------------------------------------------
CUCOMPLEX device_value(const HostComplex& value) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(value.real(), value.imag());
#else
    return make_cuDoubleComplex(value.real(), value.imag());
#endif
}

//-----------------------------------------------------------
bool beam_is(const char* beam, const char* name, size_t length) {
    return std::memcmp(beam, name, length) == 0;
}

//-----------------------------------------------------------
int nearest_integer(CU_REAL value) {
    return int(std::round(value));
}

//-----------------------------------------------------------
bool plane_wave_components(CU_REAL k0, HostComplex amplitude, CU_REAL ss0,
                           CU_REAL pp0, CU_REAL kx, CU_REAL ky, bool circular,
                           HostComplex* ex, HostComplex* ey, HostComplex* ez) {
    const CU_REAL kp2 = kx * kx + ky * ky;
    if (!ex || !ey || !ez || k0 <= CU_REAL(0) || kp2 > k0 * k0)
        return false;
    if (!circular && pp0 != CU_REAL(2) && pp0 != CU_REAL(3) &&
        (pp0 < CU_REAL(0) || pp0 > CU_REAL(1)))
        return false;
    const CU_REAL kz = std::sqrt(k0 * k0 - kp2);
    const CU_REAL phi = kp2 <= k0 * k0 / CU_REAL(1000000)
                            ? CU_REAL(0) : std::atan2(ky, kx);
    const CU_REAL theta = std::asin(std::sqrt(kp2) / k0);
    CU_REAL s = std::sqrt(std::max(CU_REAL(0), CU_REAL(1) - pp0));
    CU_REAL p = std::sqrt(std::max(CU_REAL(0), pp0));

    const bool y_polarization = pp0 == CU_REAL(3);
    const bool x_polarization = pp0 == CU_REAL(2);
    if (x_polarization || y_polarization) {
        const CU_REAL cosine = std::cos(phi);
        const CU_REAL sine = std::sin(phi);
        if (y_polarization)
            pp0 = sine * sine / (std::cos(theta) * std::cos(theta) * cosine * cosine + sine * sine);
        else
            pp0 = cosine * cosine / (std::cos(theta) * std::cos(theta) * sine * sine + cosine * cosine);
        p = std::sqrt(std::max(CU_REAL(0), pp0));
        s = std::sqrt(std::max(CU_REAL(0), CU_REAL(1) - pp0));
        CU_REAL phi_degrees = phi * CU_REAL(180) / pi;
        if (phi_degrees < CU_REAL(0))
            phi_degrees += CU_REAL(360);
        if (y_polarization) {
            if (phi_degrees < CU_REAL(90)) {}
            else if (phi_degrees < CU_REAL(180)) s = -s;
            else if (phi_degrees < CU_REAL(270)) { p = -p; s = -s; }
            else p = -p;
        } else {
            if (phi_degrees < CU_REAL(90)) s = -s;
            else if (phi_degrees < CU_REAL(180)) { p = -p; s = -s; }
            else if (phi_degrees < CU_REAL(270)) p = -p;
        }
    }

    HostComplex e0s;
    HostComplex e0p;
    if (circular) {
        const CU_REAL unit = CU_REAL(1) / std::sqrt(CU_REAL(2));
        e0s = amplitude * unit;
        e0p = amplitude * HostComplex(0, std::copysign(unit, ss0));
    } else {
        e0s = amplitude * s;
        e0p = amplitude * p;
    }

    if (kz >= CU_REAL(0.99999) * k0) {
        if (circular) {
            *ex = e0p;
            *ey = e0s;
        } else {
            *ex = e0p;
            *ey = e0s;
        }
        *ez = HostComplex(0, 0);
        return true;
    }

    const CU_REAL transverse = std::sqrt(kp2);
    if (transverse == CU_REAL(0))
        return false;
    CU_REAL nsx, nsy, npx, npy, npz;
    if (circular) {
        nsx = ky / transverse;
        nsy = -kx / transverse;
        npx = -kz * nsy;
        npy = kz * nsx;
        npz = kx * nsy - ky * nsx;
    } else {
        nsx = -ky / transverse;
        nsy = kx / transverse;
        npx = kz * nsy;
        npy = -kz * nsx;
        npz = -kx * nsy + ky * nsx;
    }
    const CU_REAL norm = std::sqrt(npx * npx + npy * npy + npz * npz);
    *ex = e0s * (nsx / norm) + e0p * (npx / norm);
    *ey = e0s * (nsy / norm) + e0p * (npy / norm);
    *ez = e0p * (npz / norm);
    return true;
}

//-----------------------------------------------------------
bool run_gaussian_fft(int nfft, int imax, CU_REAL k0, HostComplex amplitude,
                      CU_REAL ss, CU_REAL pp, CU_REAL theta, CU_REAL phi,
                      CU_REAL x0, CU_REAL y0, CU_REAL z0, CU_REAL waist,
                      CU_REAL tolerance, CU_REAL deltakx, CU_REAL deltaky,
                      CU_REAL numerical_aperture, bool circular, bool rigorous,
                      HostComplex* elo_in_x, HostComplex* elo_in_y,
                      HostComplex* elo_in_z, HostComplex* output,
                      int* nstop, char* infostr, size_t infostr_length) {
    if (!nstop || !output || !elo_in_x || !elo_in_y || !elo_in_z)
        return false;
    if (!initialize_cuda_device()) {
        report_error(nstop, infostr, infostr_length, "CUDA initialization failed");
        return false;
    }
    std::lock_guard<std::mutex> lock(gaussian_fft_mutex);
    const size_t plane = size_t(nfft) * size_t(nfft);
    if (plane > size_t(std::numeric_limits<int>::max() / 3)) {
        report_error(nstop, infostr, infostr_length, "Gaussian FFT plane is too large");
        return false;
    }
    cuda_data::DiffractionIncidentDeviceBuffers buffers{};
    if (!cuda_data::get_device_diffraction_incident_buffers(
            plane, 1, true, false, &buffers)) {
        report_error(nstop, infostr, infostr_length,
                     "could not acquire CUDA Gaussian FFT buffers");
        return false;
    }
    const CU_REAL deltax = CU_REAL(2) * pi / (CU_REAL(nfft) * deltakx);
    const COMPLEX e0 = amplitude;
    const COMPLEX epsilon0(CU_REAL(1), CU_REAL(0));
    cudaError_t status = diffraction_incident_fft_coordinates_wrapper(
        buffers.xs, buffers.ys, buffers.zs, nfft, deltax);
    if (status == cudaSuccess)
        status = incident_paraxial_gaussian_wrapper(
            buffers.xs, buffers.ys, buffers.zs, buffers.paraxial_field,
            int(plane), e0, epsilon0, ss, pp, theta, phi, x0, y0, z0,
            waist, k0, circular ? 1 : 0);

    if (status == cudaSuccess && rigorous) {
        if (!gaussian_gkp_uploaded) {
            status = cuda_data::copy_host_to_device(
                buffers.gkp_nodes, gp_x, sizeof(gp_x),
                cuda_utils::TransferCategory::IncidentField);
            if (status == cudaSuccess)
                status = cuda_data::copy_host_to_device(
                    buffers.gkp_weights, gp_w, sizeof(gp_w),
                    cuda_utils::TransferCategory::IncidentField);
            if (status == cudaSuccess)
                gaussian_gkp_uploaded = true;
        }
        if (status == cudaSuccess)
            status = rigorous_gaussian_incident_wrapper(
                buffers.xs, buffers.ys, buffers.zs, int(plane),
                double(x0), double(y0), double(z0), double(theta), double(phi),
                double(waist), double(k0), double(ss), double(pp), e0,
                double(tolerance), circular ? 1 : 0, 0, buffers.gkp_nodes,
                buffers.gkp_weights, buffers.rigorous_field, buffers.convergence);
    }
    if (status == cudaSuccess && rigorous)
        status = gaussianfft_max_intensity_wrapper(
            buffers.paraxial_field, buffers.max_partials,
            buffers.max_intensity, int(plane));
    if (status == cudaSuccess && !rigorous)
        status = cudaMemset(buffers.max_intensity, 0, sizeof(CU_REAL));
    if (status == cudaSuccess) {
        const CU_REAL blend_tolerance = rigorous
            ? (circular ? tolerance / CU_REAL(100) : tolerance / CU_REAL(10))
            : CU_REAL(0);
        status = gaussianfft_blend_first_plane_wrapper(
            buffers.paraxial_field,
            rigorous ? buffers.rigorous_field : buffers.paraxial_field,
            buffers.max_intensity, blend_tolerance, circular ? 1 : 0,
            buffers.fft_planes, int(plane));
    }
    if (status == cudaSuccess)
        status = diffraction_incident_scale_wrapper(
            buffers.fft_planes, int(3 * plane),
            CU_REAL(1) / (deltakx * deltakx * CU_REAL(plane)));
    if (status == cudaSuccess) {
        const cufftResult fft_status = CUFFT_EXECUTE(
            buffers.fft_plan, buffers.fft_planes, buffers.fft_planes, CUFFT_FORWARD);
        if (fft_status != CUFFT_SUCCESS)
            status = cudaErrorUnknown;
    }
    if (status == cudaSuccess)
        status = diffraction_incident_crop_wrapper(
            buffers.fft_planes, buffers.spectrum, nfft, imax, deltakx,
            deltaky, numerical_aperture);

    std::vector<CUCOMPLEX> host_fft(3 * plane);
    std::vector<CUCOMPLEX> host_spectrum(3 * plane);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            host_fft.data(), buffers.fft_planes,
            host_fft.size() * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            host_spectrum.data(), buffers.spectrum,
            host_spectrum.size() * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA Gaussian incident FFT failed: %s\n",
                     cudaGetErrorString(status));
        report_error(nstop, infostr, infostr_length,
                     "CUDA Gaussian incident FFT failed");
        return false;
    }
    for (size_t i = 0; i < plane; ++i) {
        elo_in_x[i] = HostComplex(host_fft[i].x, host_fft[i].y);
        elo_in_y[i] = HostComplex(host_fft[plane + i].x, host_fft[plane + i].y);
        elo_in_z[i] = HostComplex(host_fft[2 * plane + i].x, host_fft[2 * plane + i].y);
    }
    for (size_t i = 0; i < 3 * plane; ++i)
        output[i] = HostComplex(host_spectrum[i].x, host_spectrum[i].y);
    *nstop = 0;
    return true;
}
//-----------------------------------------------------------
bool run_antenna_fft(int nfft, int imax, CU_REAL k0, HostComplex amplitude,
                     CU_REAL theta, CU_REAL phi, CU_REAL xdip, CU_REAL ydip,
                     CU_REAL zdip, CU_REAL aretecube, CU_REAL deltakx,
                     CU_REAL deltaky, CU_REAL numerical_aperture,
                     HostComplex* elo_in_x, HostComplex* elo_in_y,
                     HostComplex* elo_in_z, HostComplex* output,
                     int* nstop, char* infostr, size_t infostr_length) {
    if (!nstop || !output || !elo_in_x || !elo_in_y || !elo_in_z)
        return false;
    if (!initialize_cuda_device()) {
        report_error(nstop, infostr, infostr_length, "CUDA initialization failed");
        return false;
    }
    std::lock_guard<std::mutex> lock(gaussian_fft_mutex);
    const size_t plane = size_t(nfft) * size_t(nfft);
    cuda_data::DiffractionIncidentDeviceBuffers buffers{};
    if (!cuda_data::get_device_diffraction_incident_buffers(
            plane, 1, false, true, &buffers)) {
        report_error(nstop, infostr, infostr_length,
                     "could not acquire CUDA antenna FFT buffers");
        return false;
    }
    const CU_REAL deltax = CU_REAL(2) * pi / (CU_REAL(nfft) * deltakx);
    cudaError_t status = diffraction_incident_fft_coordinates_wrapper(
        buffers.xs, buffers.ys, buffers.zs, nfft, deltax);
    if (status == cudaSuccess)
        status = incident_antenna_wrapper(
            buffers.xs, buffers.ys, buffers.zs, buffers.antenna_field,
            buffers.antenna_reference, int(plane), xdip, ydip, zdip,
            aretecube, k0, amplitude, theta, phi,
            CU_REAL(0), CU_REAL(0), CU_REAL(0));
    if (status == cudaSuccess)
        status = diffraction_incident_interleaved_to_planes_wrapper(
            buffers.antenna_field, buffers.fft_planes, int(plane));
    if (status == cudaSuccess)
        status = diffraction_incident_scale_wrapper(
            buffers.fft_planes, int(3 * plane),
            CU_REAL(1) / (deltakx * deltakx * CU_REAL(plane)));
    if (status == cudaSuccess) {
        const cufftResult fft_status = CUFFT_EXECUTE(
            buffers.fft_plan, buffers.fft_planes, buffers.fft_planes, CUFFT_FORWARD);
        if (fft_status != CUFFT_SUCCESS)
            status = cudaErrorUnknown;
    }
    if (status == cudaSuccess)
        status = diffraction_incident_crop_wrapper(
            buffers.fft_planes, buffers.spectrum, nfft, imax,
            deltakx, deltaky, numerical_aperture);
    std::vector<CUCOMPLEX> host_fft(3 * plane);
    std::vector<CUCOMPLEX> host_spectrum(3 * plane);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            host_fft.data(), buffers.fft_planes,
            host_fft.size() * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            host_spectrum.data(), buffers.spectrum,
            host_spectrum.size() * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA antenna incident FFT failed: %s\n",
                     cudaGetErrorString(status));
        report_error(nstop, infostr, infostr_length,
                     "CUDA antenna incident FFT failed");
        return false;
    }
    for (size_t i = 0; i < plane; ++i) {
        elo_in_x[i] = HostComplex(host_fft[i].x, host_fft[i].y);
        elo_in_y[i] = HostComplex(host_fft[plane + i].x, host_fft[plane + i].y);
        elo_in_z[i] = HostComplex(host_fft[2 * plane + i].x, host_fft[2 * plane + i].y);
    }
    for (size_t i = 0; i < 3 * plane; ++i)
        output[i] = HostComplex(host_spectrum[i].x, host_spectrum[i].y);
    *nstop = 0;
    return true;
}
} // namespace

//-----------------------------------------------------------
extern "C" void diffractefft2dinc_(
    const int* speckseed, const int* nfft2d, const CU_REAL* k0, HostComplex* E0,
    CU_REAL* ss, CU_REAL* pp, CU_REAL* theta, CU_REAL* phi,
    const CU_REAL* thetam, const CU_REAL* phim, const CU_REAL* ppm,
    const CU_REAL* ssm, const HostComplex* E0m, const int* nbinc,
    const CU_REAL* xdip, const CU_REAL* ydip, const CU_REAL* zdip,
    const CU_REAL* xgaus, const CU_REAL* ygaus, const CU_REAL* zgaus,
    const CU_REAL* numaperil, const CU_REAL* w0, const CU_REAL* aretecube,
    const CU_REAL* tol, HostComplex* Eloinx, HostComplex* Eloiny,
    HostComplex* Eloinz, int* imax, CU_REAL* deltakx, CU_REAL* deltaky,
    HostComplex* Ediffkzpos, const CU_REAL* numaper, const char* beam,
    const long long* plan2f, const long long* plan2b, int* nstop,
    char* infostr, size_t infostr_length) {
    if (!nstop || !infostr)
        return;
    *nstop = 1;
    std::memset(infostr, ' ', infostr_length);
    if (!speckseed || !nfft2d || !k0 || !E0 || !ss || !pp || !theta || !phi ||
        !thetam || !phim || !ppm || !ssm || !E0m || !nbinc || !xdip || !ydip ||
        !zdip || !xgaus || !ygaus || !zgaus || !numaperil || !w0 || !aretecube ||
        !tol || !Eloinx || !Eloiny || !Eloinz || !imax || !deltakx || !deltaky ||
        !Ediffkzpos || !numaper || !beam || !plan2f || !plan2b || *nfft2d <= 0 ||
        *nfft2d > 16384 || *imax < 0 || 2 * *imax + 1 > *nfft2d ||
        *k0 <= CU_REAL(0) || *deltakx <= CU_REAL(0) || *deltaky <= CU_REAL(0)) {
        report_error(nstop, infostr, infostr_length, "invalid diffractefft2dinc arguments");
        return;
    }

    const bool plane = beam_is(beam, "pwavelinear", 11);
    const bool circular = beam_is(beam, "pwavecircular", 13);
    const bool multi = beam_is(beam, "wavelinearmulti", 15);
    const bool speckle = beam_is(beam, "speckle", 7) || beam_is(beam, "demispeckle", 11);
    const bool confocal = beam_is(beam, "confocal", 8) || beam_is(beam, "demiconfocal", 12);
    const bool gaussian_linear_rigorous = beam_is(beam, "gwavelinear", 11) ||
                                          beam_is(beam, "gfftwavelinear", 14);
    const bool gaussian_circular_rigorous = beam_is(beam, "gwavecircular", 13) ||
                                            beam_is(beam, "gfftwavecircular", 16);
    const bool gaussian_linear_paraxial = beam_is(beam, "gparawavelinear", 15);
    const bool gaussian_circular_paraxial = beam_is(beam, "gparawavecircular", 17);
    const bool gaussian = gaussian_linear_rigorous || gaussian_circular_rigorous ||
                         gaussian_linear_paraxial || gaussian_circular_paraxial;

    if (gaussian) {
        if (*deltakx >= *numaper) {
            report_error(nstop, infostr, infostr_length,
                         "In FFT lens inc nfft2d too small");
            return;
        }
        if (std::abs(*deltakx - *deltaky) >
            CU_REAL(1e-12) * std::max(*deltakx, *deltaky)) {
            report_error(nstop, infostr, infostr_length,
                         "Gaussian CUDA path requires equal x/y frequency spacing");
            return;
        }
        const bool circular_gaussian = gaussian_circular_rigorous ||
                                       gaussian_circular_paraxial;
        const bool rigorous_gaussian = gaussian_linear_rigorous ||
                                       gaussian_circular_rigorous;
        run_gaussian_fft(*nfft2d, *imax, *k0, *E0, *ss, *pp, *theta, *phi,
                         *xgaus, *ygaus, *zgaus, *w0, *tol, *deltakx,
                         *deltaky, *numaper, circular_gaussian, rigorous_gaussian,
                         Eloinx, Eloiny, Eloinz, Ediffkzpos, nstop, infostr,
                         infostr_length);
        return;
    }
    const bool antenna = beam_is(beam, "antenna", 7);

    if (antenna) {
        run_antenna_fft(*nfft2d, *imax, *k0, *E0, *theta, *phi,
                        *xdip, *ydip, *zdip, *aretecube, *deltakx,
                        *deltaky, *numaper, Eloinx, Eloiny, Eloinz,
                        Ediffkzpos, nstop, infostr, infostr_length);
        return;
    }
    if (!plane && !circular && !multi && !speckle && !confocal) {
        report_error(nstop, infostr, infostr_length, "unsupported incident diffraction beam");
        return;
    }

    if (multi && (*nbinc < 0 || *nbinc > 10)) {
        report_error(nstop, infostr, infostr_length, "too many plane waves");
        return;
    }
    if (*deltakx >= *numaper) {
        report_error(nstop, infostr, infostr_length,
                     "In FFT lens inc nfft2d too small");
        return;
    }
    if (std::abs(*deltakx - *deltaky) > CU_REAL(1e-12) * std::max(*deltakx, *deltaky)) {
        report_error(nstop, infostr, infostr_length,
                     "spectral CUDA path requires equal x/y frequency spacing");
        return;
    }
    if (!initialize_cuda_device()) {
        report_error(nstop, infostr, infostr_length, "CUDA initialization failed");
        return;
    }

    const int nfft = *nfft2d;
    const size_t plane_size = size_t(nfft) * size_t(nfft);
    const size_t output_bytes = 3 * plane_size * sizeof(CUCOMPLEX);
    const size_t width = size_t(2 * *imax + 1);
    const size_t random_count = width * width;
    int mode = plane || circular ? 0 : multi ? 1 : speckle ? 2 : 3;
    const int plane_count = multi ? *nbinc : (plane || circular ? 1 : 0);
    cuda_data::DiffractionIncidentDeviceBuffers buffers{};
    if (!cuda_data::get_device_diffraction_incident_buffers(
            plane_size, size_t(std::max(1, plane_count)), false, false, &buffers)) {
        report_error(nstop, infostr, infostr_length,
                     "could not acquire CUDA diffraction illumination buffers");
        return;
    }

    std::vector<CUCOMPLEX> host_parameters(
        5 * size_t(std::max(1, plane_count)), CUFFT_ZERO);
    const int wave_count = plane_count;
    for (int wave = 0; wave < wave_count; ++wave) {
        const CU_REAL wave_theta = multi ? thetam[wave] : *theta;
        const CU_REAL wave_phi = multi ? phim[wave] : *phi;
        const CU_REAL polarization = multi ? ppm[wave] : *pp;
        const CU_REAL s_component = multi ? ssm[wave] : *ss;
        const HostComplex amplitude = multi ? E0m[wave] : *E0;
        const CU_REAL radians_theta = wave_theta * pi / CU_REAL(180);
        const CU_REAL radians_phi = wave_phi * pi / CU_REAL(180);
        const CU_REAL kx_inc = std::sin(radians_theta) * std::cos(radians_phi) * *k0;
        const CU_REAL ky_inc = std::sin(radians_theta) * std::sin(radians_phi) * *k0;
        const int ix = nearest_integer(kx_inc / *deltakx);
        const int iy = nearest_integer(ky_inc / *deltaky);
        const CU_REAL kx = *deltakx * CU_REAL(ix);
        const CU_REAL ky = *deltaky * CU_REAL(iy);
        if (ix < -*imax || ix > *imax || iy < -*imax || iy > *imax) {
            report_error(nstop, infostr, infostr_length,
                         "incident plane wave lies outside the selected FFT aperture");
            return;
        }
        HostComplex ex, ey, ez;
        if (!plane_wave_components(*k0, amplitude, s_component, polarization,
                                   kx, ky, circular, &ex, &ey, &ez)) {
            report_error(nstop, infostr, infostr_length,
                         "invalid polarization for incident plane wave");
            return;
        }
        const size_t base = size_t(wave) * 5;
        host_parameters[base] = device_value(HostComplex(CU_REAL(ix), CU_REAL(iy)));
        const CU_REAL inverse_dk_squared = CU_REAL(1) / (*deltakx * *deltakx);
        host_parameters[base + 1] = device_value(ex * inverse_dk_squared);
        host_parameters[base + 2] = device_value(ey * inverse_dk_squared);
        host_parameters[base + 3] = device_value(ez * inverse_dk_squared);
    }

    std::vector<CUCOMPLEX> host_random(plane_size, device_value(HostComplex(1, 0)));
    if (speckle) {
        random_bridge_init(*speckseed);
        const CU_REAL kmax = *k0 * *numaperil;
        for (int i = -*imax; i <= *imax; ++i) {
            for (int j = -*imax; j <= *imax; ++j) {
                const CU_REAL kx = CU_REAL(i) * *deltakx;
                const CU_REAL ky = CU_REAL(j) * *deltaky;
                if (kx * kx + ky * ky <= kmax * kmax * CU_REAL(0.999)) {
                    const CU_REAL angle = CU_REAL(2) * pi * random_bridge_next();
                    const size_t index = size_t(i + *imax) + width * size_t(j + *imax);
                    host_random[index] = device_value(
                        HostComplex(std::cos(angle), std::sin(angle)));
                }
            }
        }
    }

    using cuda_utils::TransferCategory;
    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.plane_parameters, host_parameters.data(),
        host_parameters.size() * sizeof(CUCOMPLEX), TransferCategory::Control);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.random_phases, host_random.data(),
            host_random.size() * sizeof(CUCOMPLEX), TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = diffraction_incident_spectrum_wrapper(
            buffers.spectrum, buffers.plane_parameters, buffers.random_phases,
            nfft, *imax, mode, plane_count, *k0, *deltakx, *deltaky,
            *k0 * *numaperil * *k0 * *numaperil,
            *pp * pi / CU_REAL(180), *xgaus, *ygaus, *zgaus,
            device_value(*E0));
    std::vector<CUCOMPLEX> host_output(3 * plane_size);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            host_output.data(), buffers.spectrum, output_bytes, TransferCategory::Result);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA incident diffraction failed: %s\n",
                     cudaGetErrorString(status));
        report_error(nstop, infostr, infostr_length, "CUDA incident diffraction failed");
        return;
    }
    for (size_t i = 0; i < 3 * plane_size; ++i)
        Ediffkzpos[i] = HostComplex(host_output[i].x, host_output[i].y);

    if (multi && wave_count > 0) {
        *theta = thetam[wave_count - 1];
        *phi = phim[wave_count - 1];
        *ss = ssm[wave_count - 1];
        *pp = ppm[wave_count - 1];
        *E0 = E0m[wave_count - 1];
    }
    (void)Eloinx;
    (void)Eloiny;
    (void)Eloinz;
    (void)numaper;
    (void)aretecube;
    (void)w0;
    (void)tol;
    (void)plan2f;
    (void)plan2b;
    *nstop = 0;
}
