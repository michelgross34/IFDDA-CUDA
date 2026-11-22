#include "cuda_data.h"
#include "cuda_utils.h"
#include "gaussian_gkp_tables.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

using HostComplex = std::complex<CU_REAL>;

namespace {
std::mutex energy_gaussian_tables_mutex;
bool energy_gaussian_tables_uploaded = false;
} // namespace

namespace {

//-----------------------------------------------------------
void set_error(int* nstop, char* infostr, size_t length, const char* message) {
    if (nstop)
        *nstop = 1;
    if (!infostr || length == 0)
        return;
    std::memset(infostr, ' ', length);
    std::memcpy(infostr, message, std::min(length, std::strlen(message)));
}

//-----------------------------------------------------------
bool beam_is(const char beam[64], const char* name, size_t length) {
    return std::memcmp(beam, name, length) == 0;
}

//-----------------------------------------------------------
HostComplex scaled(HostComplex value, CU_REAL factor) {
    return HostComplex(value.real() * factor, value.imag() * factor);
}

//-----------------------------------------------------------
bool plane_polarization(HostComplex amplitude, CU_REAL ss, CU_REAL pp,
                        CU_REAL theta_degrees, CU_REAL phi_degrees, bool circular,
                        CUCOMPLEX* ex, CUCOMPLEX* ey, CUCOMPLEX* ez) {
    if (!ex || !ey || !ez || !std::isfinite(double(theta_degrees)) ||
        !std::isfinite(double(phi_degrees)) || !std::isfinite(double(pp)))
        return false;
    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);
    const CU_REAL theta = theta_degrees * pi / CU_REAL(180);
    const CU_REAL phi = phi_degrees * pi / CU_REAL(180);
    const CU_REAL sin_theta = std::sin(theta);
    const CU_REAL cos_theta = std::cos(theta);
    const CU_REAL sin_phi = std::sin(phi);
    const CU_REAL cos_phi = std::cos(phi);
    const CU_REAL direction = theta_degrees < CU_REAL(0) ? CU_REAL(-1) : CU_REAL(1);

    HostComplex e0s{};
    HostComplex e0p{};
    if (circular) {
        const CU_REAL inverse_sqrt_two = CU_REAL(1) / std::sqrt(CU_REAL(2));
        const CU_REAL handedness = ss < CU_REAL(0) ? CU_REAL(-1) : CU_REAL(1);
        e0s = scaled(amplitude, inverse_sqrt_two);
        e0p = HostComplex(-amplitude.imag() * inverse_sqrt_two * handedness,
                          amplitude.real() * inverse_sqrt_two * handedness);
    } else {
        CU_REAL p_fraction = pp;
        CU_REAL p = CU_REAL(0);
        CU_REAL s = CU_REAL(0);
        if (std::abs(pp - CU_REAL(3)) <= CU_REAL(3e-12)) {
            CU_REAL angle = phi_degrees;
            if (angle < CU_REAL(0))
                angle += CU_REAL(360);
            p_fraction = (sin_phi * sin_phi) /
                (cos_theta * cos_theta * cos_phi * cos_phi + sin_phi * sin_phi);
            p = std::sqrt(p_fraction);
            s = std::sqrt(CU_REAL(1) - p_fraction);
            if (angle >= CU_REAL(90) && angle < CU_REAL(180))
                s = -s;
            else if (angle >= CU_REAL(180) && angle < CU_REAL(270)) {
                p = -p;
                s = -s;
            } else if (angle >= CU_REAL(270) && angle < CU_REAL(360)) {
                p = -p;
            }
        } else if (std::abs(pp - CU_REAL(2)) <= CU_REAL(2e-12)) {
            CU_REAL angle = phi_degrees;
            p_fraction = (cos_phi * cos_phi) /
                (cos_theta * cos_theta * sin_phi * sin_phi + cos_phi * cos_phi);
            if (angle < CU_REAL(0))
                angle += CU_REAL(360);
            p = std::sqrt(p_fraction);
            s = std::sqrt(CU_REAL(1) - p_fraction);
            if (angle >= CU_REAL(0) && angle < CU_REAL(90))
                s = -s;
            else if (angle >= CU_REAL(90) && angle < CU_REAL(180)) {
                p = -p;
                s = -s;
            } else if (angle >= CU_REAL(180) && angle < CU_REAL(270)) {
                p = -p;
            }
        } else {
            if (pp < CU_REAL(0) || pp > CU_REAL(1))
                return false;
            p = std::sqrt(pp);
            s = std::sqrt(CU_REAL(1) - pp);
        }
        e0s = scaled(amplitude, s * direction);
        e0p = scaled(amplitude, p * direction);
    }

    const HostComplex hx = -sin_phi * e0s + cos_theta * cos_phi * e0p;
    const HostComplex hy = cos_phi * e0s + cos_theta * sin_phi * e0p;
    const HostComplex hz = -sin_theta * e0p;
    *ex = cuda_utils::device_value(hx);
    *ey = cuda_utils::device_value(hy);
    *ez = cuda_utils::device_value(hz);
    return true;
}

//-----------------------------------------------------------
bool check_cuda_status(cudaError_t status, const char* operation) {
    return cuda_utils::check_cuda(status, "FFT energy diffraction", operation);
}

//-----------------------------------------------------------
bool run_plane_wave_spectrum(cuda_data::DiffractionEnergyDeviceBuffers& buffers,
                             const int nfft, const int beam_count,
                             const CU_REAL* theta, const CU_REAL* phi,
                             const CU_REAL* pp, const CU_REAL* ss,
                             const HostComplex* amplitudes, const CU_REAL k0,
                             const CU_REAL pixel, const CU_REAL scale,
                             cublasHandle_t handle) {
    const size_t plane = size_t(nfft) * size_t(nfft);
    const int complex_count = int(3 * plane);
    if (!check_cuda_status(zero_vectors_wrapper(buffers.incident_spectrum, complex_count),
                           "clear incident spectrum"))
        return false;

    const CUCOMPLEX one = cuda_utils::device_value(HostComplex(CU_REAL(1), CU_REAL(0)));
    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);
    const CU_REAL dkx = CU_REAL(2) * pi / (CU_REAL(nfft) * pixel);
    const CU_REAL dky = dkx;
    for (int mode = 0; mode < beam_count; ++mode) {
        CUCOMPLEX ex0{}, ey0{}, ez0{};
        const bool circular = pp == nullptr;
        const CU_REAL theta_radians = theta[mode] * pi / CU_REAL(180);
        const CU_REAL phi_radians = phi[mode] * pi / CU_REAL(180);
        const CU_REAL kx_incident = k0 * std::sin(theta_radians) * std::cos(phi_radians);
        const CU_REAL ky_incident = k0 * std::sin(theta_radians) * std::sin(phi_radians);
        // diffractefft2dinc snaps every plane wave to the nearest Fourier bin
        // before evaluating its polarization and field. Sampling the original
        // continuous k would leak one mode over the entire FFT plane.
        const CU_REAL kx = dkx * std::round(kx_incident / dkx);
        const CU_REAL ky = dky * std::round(ky_incident / dky);
        const CU_REAL theta_snapped = std::asin(std::min(
            CU_REAL(1), std::sqrt(kx * kx + ky * ky) / k0)) * CU_REAL(180) / pi;
        const CU_REAL phi_snapped = std::atan2(ky, kx) * CU_REAL(180) / pi;
        if (!plane_polarization(amplitudes[mode], ss[mode],
                                circular ? CU_REAL(0) : pp[mode],
                                theta_snapped, phi_snapped, circular,
                                &ex0, &ey0, &ez0))
            return false;

        const CU_REAL radicand = k0 * k0 - kx * kx - ky * ky;
        const CU_REAL kz = radicand > CU_REAL(0) ? std::sqrt(radicand) : CU_REAL(0);
        if (!check_cuda_status(quick_energy_incident_wave_wrapper(
                                   buffers.layer_spectrum, nfft, pixel, scale,
                                   kx, ky, cuda_utils::device_value(HostComplex(kz, CU_REAL(0))),
                                   ex0, ey0, ez0),
                               "generate plane-wave spectrum"))
            return false;
        if (!cuda_utils::axpy(handle, complex_count, one, buffers.layer_spectrum,
                              buffers.incident_spectrum, "accumulate plane-wave modes"))
            return false;
    }
    return cuda_utils::check_cufft(
        CUFFT_EXECUTE(buffers.fft_plan, buffers.incident_spectrum,
                      buffers.incident_spectrum, CUFFT_FORWARD),
        "FFT energy diffraction", "transform incident field");
}

//-----------------------------------------------------------
bool publish_spectrum(const CUCOMPLEX* device_source, HostComplex* ex,
                      HostComplex* ey, HostComplex* ez, size_t plane) {
    std::vector<CUCOMPLEX> host(3 * plane);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(host.data(), device_source,
                host.size() * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result),
            "FFT energy diffraction", "copy incident spectrum to CPU"))
        return false;
    for (size_t i = 0; i < plane; ++i) {
        ex[i] = cuda_utils::host_value(host[i]);
        ey[i] = cuda_utils::host_value(host[plane + i]);
        ez[i] = cuda_utils::host_value(host[2 * plane + i]);
    }
    return true;
}

//-----------------------------------------------------------
} // namespace

//-----------------------------------------------------------
extern "C" void diffractefft2denergie_(
    const int* nx, const int* ny, const int* nz, const int* nxm, const int* nym,
    const int* nzm, const int* nfft2d, int* tabfft2, const CU_REAL* k0,
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, const HostComplex* E0,
    const CU_REAL* ss, const CU_REAL* pp, const CU_REAL* theta, const CU_REAL* phi,
    const CU_REAL* thetam, const CU_REAL* phim, const CU_REAL* ppm, const CU_REAL* ssm,
    const HostComplex* E0m, const int* nbinc, const CU_REAL* xdip, const CU_REAL* ydip,
    const CU_REAL* zdip, const CU_REAL* xgaus, const CU_REAL* ygaus, const CU_REAL* zgaus,
    const CU_REAL* w0, const CU_REAL* aretecube, const CU_REAL* tol,
    HostComplex* Eloinx, HostComplex* Eloiny, HostComplex* Eloinz, const HostComplex* FF,
    int* imax, CU_REAL* deltakx, CU_REAL* deltaky, HostComplex* Ediffkzpos,
    HostComplex* Ediffkzneg, const char beam[64], CU_REAL* efficacite,
    CU_REAL* efficaciteref, CU_REAL* efficacitetrans, const int* nsectionsca,
    const int* nquickdiffracte, const long long* plan2f, const long long* plan2b,
    int* nstop, char infostr[64], size_t infostr_len) {
    (void)nxm;
    (void)nym;
    (void)nzm;
    (void)plan2f;
    (void)plan2b;
    (void)xdip;
    (void)ydip;
    (void)zdip;
    (void)xgaus;
    (void)ygaus;
    (void)zgaus;
    (void)w0;
    (void)tol;
    if (nstop)
        *nstop = 0;
    if (infostr && infostr_len)
        std::memset(infostr, ' ', infostr_len);

    if (!nx || !ny || !nz || !nfft2d || !tabfft2 || !k0 || !xs || !ys || !zs ||
        !E0 || !ss || !pp || !theta || !phi || !thetam || !phim || !ppm || !ssm ||
        !E0m || !nbinc || !aretecube || !Eloinx || !Eloiny || !Eloinz || !FF ||
        !imax || !deltakx || !deltaky || !Ediffkzpos || !Ediffkzneg || !beam ||
        !efficacite || !efficaciteref || !efficacitetrans || !nsectionsca ||
        !nquickdiffracte || !nstop || !infostr || *nx <= 0 || *ny <= 0 || *nz <= 0 ||
        *nfft2d < 8 || *nfft2d > 16384 || *k0 <= CU_REAL(0) || *aretecube <= CU_REAL(0)) {
        set_error(nstop, infostr, infostr_len, "Invalid FFT energy diffraction arguments");
        return;
    }

    const int nfft = *nfft2d;
    const int half = nfft / 2;
    const size_t plane = size_t(nfft) * size_t(nfft);
    size_t mesh_count = 0;
    size_t nxny = 0;
    if (plane > size_t(std::numeric_limits<int>::max()) / 3 ||
        size_t(*nx) > std::numeric_limits<size_t>::max() / size_t(*ny) ||
        !cuda_utils::checked_multiply(size_t(*nx), size_t(*ny), &nxny) ||
        !cuda_utils::checked_multiply(nxny, size_t(*nz), &mesh_count) ||
        mesh_count > size_t(std::numeric_limits<int>::max()) / 3) {
        set_error(nstop, infostr, infostr_len, "FFT energy diffraction dimensions overflow");
        return;
    }

    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);
    *deltakx = CU_REAL(2) * pi / (CU_REAL(nfft) * *aretecube);
    *deltaky = *deltakx;
    if (*deltakx >= *k0) {
        set_error(nstop, infostr, infostr_len, "In FFT Poynting nfft2d too small");
        return;
    }
    *imax = int(std::floor(*k0 / *deltakx + CU_REAL(0.5))) + 1;
    if (2 * *imax + 1 > nfft || 2 * *imax + 1 < 7) {
        set_error(nstop, infostr, infostr_len, "In FFT diffract nfft2d too small");
        return;
    }
    for (int i = 0; i < nfft; ++i)
        tabfft2[i] = i - half >= 0 ? i - half + 1 : half + i + 1;

    const bool plane_beam = beam_is(beam, "pwavelinear", 11) ||
        beam_is(beam, "pwavecircular", 13) || beam_is(beam, "wavelinearmulti", 15);
    const bool antenna_beam = beam_is(beam, "antenna", 7);
    const bool paraxial_linear = beam_is(beam, "gparawavelinear", 15);
    const bool paraxial_circular = beam_is(beam, "gparawavecircular", 17);
    const bool rigorous_linear = beam_is(beam, "gwavelinear", 11) ||
        beam_is(beam, "gfftwavelinear", 14);
    const bool rigorous_circular = beam_is(beam, "gwavecircular", 13) ||
        beam_is(beam, "gfftwavecircular", 16);
    if (!plane_beam && !antenna_beam && !paraxial_linear && !paraxial_circular &&
        !rigorous_linear && !rigorous_circular) {
        set_error(nstop, infostr, infostr_len,
                  "CUDA FFT energy does not support this illumination");
        return;
    }

    if (!initialize_cuda_device()) {
        set_error(nstop, infostr, infostr_len, "CUDA device initialization failed");
        return;
    }
    cuda_data::DiffractionEnergyDeviceBuffers buffers{};
    if (!cuda_data::get_device_diffraction_energy_buffers(plane, mesh_count, &buffers)) {
        set_error(nstop, infostr, infostr_len, "Could not acquire diffraction energy buffers");
        return;
    }

    const size_t spectrum_bytes = 3 * plane * sizeof(CUCOMPLEX);
    const size_t field_count = 3 * mesh_count;
    const CUCOMPLEX* resident_field = get_device_result_vector_alias(FF, field_count);
    const CUCOMPLEX* device_green_field = resident_field
        ? resident_field : buffers.green_field;
    if (!resident_field) {
        std::vector<CUCOMPLEX> host_ff(field_count);
        for (size_t i = 0; i < field_count; ++i)
            host_ff[i] = cuda_utils::device_value(FF[i]);
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(buffers.green_field, host_ff.data(),
                    host_ff.size() * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::SolverVector),
                "FFT energy diffraction", "upload solved field")) {
            set_error(nstop, infostr, infostr_len, "Could not upload solved field to CUDA");
            return;
        }
    }
    if (!device_green_field) {
        set_error(nstop, infostr, infostr_len, "Could not upload solved field to CUDA");
        return;
    }

    std::vector<CU_REAL> host_z(mesh_count);
    for (size_t i = 0; i < mesh_count; ++i)
        host_z[i] = zs[i];
    std::vector<int> host_fft_map(nfft);
    for (int i = 0; i < nfft; ++i)
        host_fft_map[size_t(i)] = tabfft2[i];
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.z_coordinates, host_z.data(),
                host_z.size() * sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
            "FFT energy diffraction", "upload layer coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.fft_indices, host_fft_map.data(),
                host_fft_map.size() * sizeof(int), cuda_utils::TransferCategory::Control),
            "FFT energy diffraction", "upload FFT index map")) {
        set_error(nstop, infostr, infostr_len, "Could not upload diffraction geometry");
        return;
    }

    if (*nsectionsca * *nquickdiffracte == 0) {
        if (!check_cuda_status(zero_vectors_wrapper(buffers.positive_spectrum, int(3 * plane)),
                               "clear positive spectrum") ||
            !check_cuda_status(zero_vectors_wrapper(buffers.negative_spectrum, int(3 * plane)),
                               "clear negative spectrum")) {
            set_error(nstop, infostr, infostr_len, "Could not clear diffraction spectra");
            return;
        }
        const CU_REAL var1 = (xs[0] + CU_REAL(half) * *aretecube) * *deltakx;
        const CU_REAL var2 = (ys[0] + CU_REAL(half) * *aretecube) * *deltaky;
        for (int layer = 0; layer < *nz; ++layer) {
            if (!check_cuda_status(quick_diffraction_gather_wrapper(
                    device_green_field, buffers.layer_spectrum, buffers.fft_indices,
                    *nx, *ny, int(nxny), nfft, layer), "pack solved-field layer") ||
                !cuda_utils::check_cufft(
                    CUFFT_EXECUTE(buffers.fft_plan, buffers.layer_spectrum,
                                  buffers.layer_spectrum, CUFFT_FORWARD),
                    "FFT energy diffraction", "transform solved-field layer") ||
                !check_cuda_status(quick_diffraction_accumulate_wrapper(
                    buffers.layer_spectrum, buffers.fft_indices, buffers.z_coordinates,
                    int(nxny), nfft, *imax, layer, *k0, *deltakx,
                    var1, var2, buffers.positive_spectrum, buffers.negative_spectrum),
                    "accumulate scattered spectrum")) {
                set_error(nstop, infostr, infostr_len, "CUDA scattered-field diffraction failed");
                return;
            }
        }
        if (!check_cuda_status(quick_energy_finalize_fields_wrapper(
                buffers.positive_spectrum, buffers.negative_spectrum,
                nfft, *imax, *deltakx, *deltaky, *k0), "normalize diffraction spectra")) {
            set_error(nstop, infostr, infostr_len, "CUDA diffraction normalization failed");
            return;
        }
    } else {
        std::vector<CUCOMPLEX> host_positive(3 * plane);
        std::vector<CUCOMPLEX> host_negative(3 * plane);
        for (size_t i = 0; i < 3 * plane; ++i) {
            host_positive[i] = cuda_utils::device_value(Ediffkzpos[i]);
            host_negative[i] = cuda_utils::device_value(Ediffkzneg[i]);
        }
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(buffers.positive_spectrum, host_positive.data(),
                    spectrum_bytes, cuda_utils::TransferCategory::SolverVector),
                "FFT energy diffraction", "upload positive spectrum") ||
            !cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(buffers.negative_spectrum, host_negative.data(),
                    spectrum_bytes, cuda_utils::TransferCategory::SolverVector),
                "FFT energy diffraction", "upload negative spectrum") ||
            !check_cuda_status(quick_energy_finalize_fields_wrapper(
                buffers.positive_spectrum, buffers.negative_spectrum,
                nfft, *imax, *deltakx, *deltaky, *k0), "normalize existing spectra")) {
            set_error(nstop, infostr, infostr_len, "Could not prepare diffraction spectra");
            return;
        }
    }

    const CU_REAL pixel = CU_REAL(2) * pi / (CU_REAL(nfft) * *deltakx);
    const CU_REAL scale = CU_REAL(1) /
        (*deltakx * *deltaky * CU_REAL(plane));
    const bool multi = beam_is(beam, "wavelinearmulti", 15);
    const bool circular = beam_is(beam, "pwavecircular", 13);
    const int mode_count = multi ? std::clamp(*nbinc, 1, 10) : 1;
    std::vector<CU_REAL> mode_theta(static_cast<size_t>(mode_count), CU_REAL(0));
    std::vector<CU_REAL> mode_phi(static_cast<size_t>(mode_count), CU_REAL(0));
    std::vector<CU_REAL> mode_pp(static_cast<size_t>(mode_count), CU_REAL(0));
    std::vector<CU_REAL> mode_ss(static_cast<size_t>(mode_count), CU_REAL(0));
    std::vector<HostComplex> mode_amplitude(static_cast<size_t>(mode_count));
    for (int mode = 0; mode < mode_count; ++mode) {
        mode_theta[size_t(mode)] = multi ? thetam[mode] : *theta;
        mode_phi[size_t(mode)] = multi ? phim[mode] : *phi;
        mode_pp[size_t(mode)] = circular ? CU_REAL(0) : (multi ? ppm[mode] : *pp);
        mode_ss[size_t(mode)] = multi ? ssm[mode] : *ss;
        mode_amplitude[size_t(mode)] = multi ? E0m[mode] : *E0;
    }
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    bool incident_ok = false;
    if (plane_beam) {
        incident_ok = handle && run_plane_wave_spectrum(
            buffers, nfft, mode_count, mode_theta.data(), mode_phi.data(),
            circular ? nullptr : mode_pp.data(), mode_ss.data(),
            mode_amplitude.data(), *k0, pixel, scale, handle);
    } else if (antenna_beam) {
        cuda_data::DiffractionIncidentDeviceBuffers antenna_buffers{};
        if (!xdip || !ydip || !zdip ||
            !cuda_data::get_device_diffraction_incident_buffers(
                plane, 1, false, true, &antenna_buffers)) {
            set_error(nstop, infostr, infostr_len,
                "Could not acquire antenna illumination buffers");
            return;
        }
        const CU_REAL origin = -CU_REAL(half) * pixel;
        const CU_REAL zero = CU_REAL(0);
        cudaError_t status = gaussianfft_build_coordinates_wrapper(
            antenna_buffers.xs, antenna_buffers.ys, antenna_buffers.zs,
            nfft, nfft, nfft, origin, origin, zero, pixel);
        if (status == cudaSuccess) {
            status = incident_antenna_wrapper(
                antenna_buffers.xs, antenna_buffers.ys, antenna_buffers.zs,
                antenna_buffers.antenna_field, antenna_buffers.antenna_reference,
                int(plane), *xdip, *ydip, *zdip, *aretecube, *k0,
                *E0, *theta, *phi, zero, zero, zero);
        }
        if (status == cudaSuccess) {
            status = diffraction_incident_interleaved_to_planes_wrapper(
                antenna_buffers.antenna_field, buffers.incident_spectrum, int(plane));
        }
        if (status == cudaSuccess) {
            status = complex_vector_scale_wrapper(
                buffers.incident_spectrum, buffers.incident_spectrum,
                cuda_utils::device_value(HostComplex(scale, CU_REAL(0))),
                int(3 * plane));
        }
        if (status == cudaSuccess) {
            status = cuda_utils::check_cufft(
                CUFFT_EXECUTE(buffers.fft_plan, buffers.incident_spectrum,
                    buffers.incident_spectrum, CUFFT_FORWARD),
                "FFT energy diffraction", "transform antenna incident field")
                ? cudaSuccess : cudaErrorUnknown;
        }
        incident_ok = status == cudaSuccess;
        if (!incident_ok)
            cuda_utils::check_cuda(status, "FFT energy diffraction",
                "generate antenna incident spectrum");
    } else if (paraxial_linear || paraxial_circular) {
        cuda_data::DiffractionGaussianDeviceBuffers gaussian_buffers{};
        if (!cuda_data::get_device_diffraction_gaussian_buffers(
                plane, false, &gaussian_buffers)) {
            set_error(nstop, infostr, infostr_len,
                "Could not acquire paraxial Gaussian buffers");
            return;
        }
        const CU_REAL origin = -CU_REAL(half) * pixel;
        const CU_REAL zero = CU_REAL(0);
        const COMPLEX vacuum(CU_REAL(1), CU_REAL(0));
        cudaError_t status = gaussianfft_build_coordinates_wrapper(
            gaussian_buffers.x_coordinates, gaussian_buffers.y_coordinates,
            gaussian_buffers.z_coordinates, nfft, nfft, nfft,
            origin, origin, zero, pixel);
        if (status == cudaSuccess) {
            status = incident_paraxial_gaussian_wrapper(
                gaussian_buffers.x_coordinates, gaussian_buffers.y_coordinates,
                gaussian_buffers.z_coordinates, gaussian_buffers.paraxial_field,
                int(plane), *E0, vacuum,
                *ss, *pp, *theta, *phi, *xgaus, *ygaus, *zgaus,
                *w0, *k0, paraxial_circular ? 1 : 0);
        }
        if (status == cudaSuccess) {
            status = complex_vector_scale_wrapper(
                gaussian_buffers.paraxial_field, gaussian_buffers.paraxial_field,
                cuda_utils::device_value(HostComplex(scale, CU_REAL(0))),
                int(3 * plane));
        }
        if (status == cudaSuccess) {
            status = diffraction_incident_interleaved_to_planes_wrapper(
                gaussian_buffers.paraxial_field, buffers.incident_spectrum, int(plane));
        }
        if (status == cudaSuccess) {
            status = cuda_utils::check_cufft(
                CUFFT_EXECUTE(buffers.fft_plan, buffers.incident_spectrum,
                    buffers.incident_spectrum, CUFFT_FORWARD),
                "FFT energy diffraction", "transform paraxial Gaussian incident field")
                ? cudaSuccess : cudaErrorUnknown;
        }
        incident_ok = status == cudaSuccess;
        if (!incident_ok)
            cuda_utils::check_cuda(status, "FFT energy diffraction",
                "generate paraxial Gaussian incident spectrum");
    } else {
        cuda_data::DiffractionGaussianDeviceBuffers gaussian_buffers{};
        if (!cuda_data::get_device_diffraction_gaussian_buffers(
                plane, true, &gaussian_buffers)) {
            set_error(nstop, infostr, infostr_len,
                "Could not acquire rigorous Gaussian buffers");
            return;
        }
        const CU_REAL origin = -CU_REAL(half) * pixel;
        const CU_REAL zero = CU_REAL(0);
        const COMPLEX vacuum(CU_REAL(1), CU_REAL(0));
        cudaError_t status = gaussianfft_build_coordinates_wrapper(
            gaussian_buffers.x_coordinates, gaussian_buffers.y_coordinates,
            gaussian_buffers.z_coordinates,
            nfft, nfft, nfft, origin, origin, zero, pixel);
        if (status == cudaSuccess) {
            status = incident_paraxial_gaussian_wrapper(
                gaussian_buffers.x_coordinates, gaussian_buffers.y_coordinates,
                gaussian_buffers.z_coordinates,
                gaussian_buffers.paraxial_field, int(plane),
                *E0, vacuum, *ss, *pp, *theta, *phi,
                *xgaus, *ygaus, *zgaus, *w0, *k0,
                rigorous_circular ? 1 : 0);
        }
        if (status == cudaSuccess) {
            status = gaussianfft_max_intensity_wrapper(
                gaussian_buffers.paraxial_field, gaussian_buffers.maximum_partials,
                gaussian_buffers.maximum_intensity, int(plane));
        }
        {
            std::lock_guard<std::mutex> lock(energy_gaussian_tables_mutex);
            if (status == cudaSuccess && !energy_gaussian_tables_uploaded) {
                status = cuda_data::copy_host_to_device(
                    gaussian_buffers.gkp_nodes, gp_x, sizeof(gp_x),
                    cuda_utils::TransferCategory::IncidentField);
                if (status == cudaSuccess) {
                    status = cuda_data::copy_host_to_device(
                        gaussian_buffers.gkp_weights, gp_w, sizeof(gp_w),
                        cuda_utils::TransferCategory::IncidentField);
                }
                if (status == cudaSuccess)
                    energy_gaussian_tables_uploaded = true;
            }
        }
        if (status == cudaSuccess) {
            status = rigorous_gaussian_incident_wrapper(
                gaussian_buffers.x_coordinates, gaussian_buffers.y_coordinates,
                gaussian_buffers.z_coordinates,
                int(plane), double(*xgaus), double(*ygaus), double(*zgaus),
                double(*theta), double(*phi), double(*w0), double(*k0),
                double(*ss), double(*pp), *E0,
                double(*tol), rigorous_circular ? 1 : 0, 0,
                gaussian_buffers.gkp_nodes, gaussian_buffers.gkp_weights,
                gaussian_buffers.rigorous_field, gaussian_buffers.convergence);
        }
        if (status == cudaSuccess) {
            status = gaussianfft_blend_first_plane_wrapper(
                gaussian_buffers.paraxial_field, gaussian_buffers.rigorous_field,
                gaussian_buffers.maximum_intensity, *tol, 1,
                buffers.incident_spectrum, int(plane));
        }
        if (status == cudaSuccess) {
            status = complex_vector_scale_wrapper(
                buffers.incident_spectrum, buffers.incident_spectrum,
                cuda_utils::device_value(HostComplex(scale, CU_REAL(0))),
                int(3 * plane));
        }
        if (status == cudaSuccess) {
            status = cuda_utils::check_cufft(
                CUFFT_EXECUTE(buffers.fft_plan, buffers.incident_spectrum,
                    buffers.incident_spectrum, CUFFT_FORWARD),
                "FFT energy diffraction", "transform rigorous Gaussian incident field")
                ? cudaSuccess : cudaErrorUnknown;
        }
        incident_ok = status == cudaSuccess;
        if (!incident_ok)
            cuda_utils::check_cuda(status, "FFT energy diffraction",
                "generate rigorous Gaussian incident spectrum");
    }
    if (!incident_ok) {
        set_error(nstop, infostr, infostr_len, "CUDA incident spectrum generation failed");
        return;
    }

    const size_t partial_count = plane;
    CU_REAL sums[5]{};
    if (!check_cuda_status(quick_energy_flux_wrapper(
            buffers.incident_spectrum, buffers.positive_spectrum,
            buffers.negative_spectrum, nfft, *imax, *deltakx, *deltaky, *k0,
            buffers.reduction_partials,
            buffers.reduction_partials + partial_count,
            buffers.reduction_partials + 2 * partial_count,
            buffers.reduction_partials + 3 * partial_count,
            buffers.reduction_partials + 4 * partial_count,
            buffers.reduction_partials + 5 * partial_count), "compute energy flux") ||
        !cuda_utils::sum_nonnegative(handle, int(partial_count),
            buffers.reduction_partials, &sums[0],
            "incident flux reduction") ||
        !cuda_utils::sum_nonnegative(handle, int(partial_count),
            buffers.reduction_partials + partial_count, &sums[1],
            "reflected flux reduction") ||
        !cuda_utils::sum_nonnegative(handle, int(partial_count),
            buffers.reduction_partials + 2 * partial_count, &sums[2],
            "transmitted flux reduction") ||
        !cuda_utils::sum_nonnegative(handle, int(partial_count),
            buffers.reduction_partials + 3 * partial_count, &sums[3],
            "inside-aperture energy reduction") ||
        !cuda_utils::sum_nonnegative(handle, int(partial_count),
            buffers.reduction_partials + 4 * partial_count, &sums[4],
            "total energy reduction")) {
        set_error(nstop, infostr, infostr_len, "CUDA energy reduction failed");
        return;
    }

    if (sums[0] == 0.0 || sums[4] == 0.0) {
        set_error(nstop, infostr, infostr_len, "Incident energy is zero");
        return;
    }
    const CU_REAL flux_scale = CU_REAL(4) * pi * pi * *deltaky * *deltakx /
        (*k0 * CU_REAL(8) * pi * CU_REAL(1.0e-7) * CU_REAL(299792458.0));
    std::printf("Incident flux         : %.16g W\n", double(sums[0] * flux_scale));
    std::printf("Reflected flux        : %.16g W\n", double(sums[1] * flux_scale));
    std::printf("Transmitted flux      : %.16g W\n", double(sums[2] * flux_scale));
    std::printf("Total flux            : %.16g W\n",
                double((sums[1] + sums[2]) * flux_scale));
    *efficacite = CU_REAL((sums[1] + sums[2]) / sums[0]);
    *efficaciteref = CU_REAL(sums[1] / sums[0]);
    *efficacitetrans = CU_REAL(sums[2] / sums[0]);
    const double outside_percent = double(std::abs(sums[3] - sums[4]) / sums[4] * CU_REAL(100));
    std::printf("Energy outside the NA (%%): %.16g\n", outside_percent);
    if (outside_percent >= 1.0)
        set_error(nstop, infostr, infostr_len,
                  "Energy: Beam too inclined or waist or nfft too small");

    std::vector<CUCOMPLEX> host_positive(3 * plane);
    std::vector<CUCOMPLEX> host_negative(3 * plane);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(host_positive.data(), buffers.positive_spectrum,
                spectrum_bytes, cuda_utils::TransferCategory::Result),
            "FFT energy diffraction", "download positive spectrum") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(host_negative.data(), buffers.negative_spectrum,
                spectrum_bytes, cuda_utils::TransferCategory::Result),
            "FFT energy diffraction", "download negative spectrum") ||
        !publish_spectrum(buffers.incident_spectrum, Eloinx, Eloiny, Eloinz, plane)) {
        set_error(nstop, infostr, infostr_len, "Could not publish energy spectra");
        return;
    }
    for (size_t i = 0; i < 3 * plane; ++i) {
        Ediffkzpos[i] = cuda_utils::host_value(host_positive[i]);
        Ediffkzneg[i] = cuda_utils::host_value(host_negative[i]);
    }
}
