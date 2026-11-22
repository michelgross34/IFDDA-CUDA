#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <vector>

using HostComplex = std::complex<CU_REAL>;






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
bool beam_is(const char beam[64], const char* value, size_t length) {
    return std::memcmp(beam, value, length) == 0;
}

//-----------------------------------------------------------
bool cuda_ok(cudaError_t status, const char* operation) {
    return cuda_utils::check_cuda(status, "confocal FFT energy", operation);
}

//-----------------------------------------------------------
bool make_incident_phases(const char beam[64], int seed, int width,
                          std::vector<CU_REAL>* positive,
                          std::vector<CU_REAL>* negative,
                          int* beam_kind) {
    if (!positive || !negative || !beam_kind)
        return false;
    positive->assign(size_t(width) * size_t(width), CU_REAL(0));
    negative->assign(size_t(width) * size_t(width), CU_REAL(0));
    const bool confocal = beam_is(beam, "confocal", 8);
    const bool demiconfocal = beam_is(beam, "demiconfocal", 12);
    const bool speckle = beam_is(beam, "speckle", 7);
    const bool demispeckle = beam_is(beam, "demispeckle", 11);
    if (!confocal && !demiconfocal && !speckle && !demispeckle)
        return false;
    *beam_kind = (demiconfocal || demispeckle) ? 1 : 0;
    if (demispeckle) {
        random_bridge_stream_init(0, seed);
        random_bridge_stream_init(1, seed + 1);
    } else if (speckle) {
        random_bridge_init(seed);
    }
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            const size_t index = size_t(i) + size_t(width) * size_t(j);
            if (speckle) {
                (*positive)[index] = random_bridge_next();
                (*negative)[index] = random_bridge_next();
            } else if (demispeckle) {
                (*positive)[index] = random_bridge_stream_next(0);
                (void)random_bridge_stream_next(1);
            }
        }
    }
    return true;
}

//-----------------------------------------------------------
bool publish_component_spectra(const cuda_data::DiffractionEnergyDeviceBuffers& buffers,
                               HostComplex* incident_x, HostComplex* incident_y,
                               HostComplex* incident_z, HostComplex* scattered_positive,
                               HostComplex* scattered_negative, size_t plane) {
    std::vector<CUCOMPLEX> host_incident(3 * plane);
    std::vector<CUCOMPLEX> host_positive(3 * plane);
    std::vector<CUCOMPLEX> host_negative(3 * plane);
    if (!cuda_ok(cuda_data::copy_device_to_host(host_incident.data(), buffers.incident_spectrum,
            host_incident.size() * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result),
            "copy incident spectrum to CPU") ||
        !cuda_ok(cuda_data::copy_device_to_host(host_positive.data(), buffers.positive_spectrum,
            host_positive.size() * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result),
            "copy positive scattered spectrum to CPU") ||
        !cuda_ok(cuda_data::copy_device_to_host(host_negative.data(), buffers.negative_spectrum,
            host_negative.size() * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result),
            "copy negative scattered spectrum to CPU"))
        return false;
    for (size_t i = 0; i < plane; ++i) {
        incident_x[i] = cuda_utils::host_value(host_incident[i]);
        incident_y[i] = cuda_utils::host_value(host_incident[plane + i]);
        incident_z[i] = cuda_utils::host_value(host_incident[2 * plane + i]);
        for (size_t component = 0; component < 3; ++component) {
            scattered_positive[component * plane + i] =
                cuda_utils::host_value(host_positive[component * plane + i]);
            scattered_negative[component * plane + i] =
                cuda_utils::host_value(host_negative[component * plane + i]);
        }
    }
    return true;
}

} // namespace

//-----------------------------------------------------------
extern "C" void diffractefft2denergieconfocal_(
    const int* speckseed, const CU_REAL* numaperil,
    const int* nx, const int* ny, const int* nz,
    const int* nxm, const int* nym, const int* nzm,
    const int* nfft2d, int* tabfft2, const CU_REAL* k0,
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    const HostComplex* E0, const CU_REAL* ss, const CU_REAL* pp,
    const CU_REAL* theta, const CU_REAL* phi,
    const CU_REAL* xgaus, const CU_REAL* ygaus, const CU_REAL* zgaus,
    const CU_REAL* w0, const CU_REAL* aretecube,
    HostComplex* Eloinx, HostComplex* Eloiny, HostComplex* Eloinz,
    const HostComplex* FF, int* imax, CU_REAL* deltakx, CU_REAL* deltaky,
    HostComplex* Ediffkzpos, HostComplex* Ediffkzneg,
    const char beam[64], CU_REAL* efficacite, CU_REAL* efficaciteref,
    CU_REAL* efficacitetrans, const int* nsectionsca,
    const int* nquickdiffracte, const long long* plan2f,
    const long long* plan2b, const int* nside, int* nstop,
    char infostr[64], size_t infostr_len) {
    (void)ss;
    (void)theta;
    (void)phi;
    (void)w0;
    (void)plan2f;
    (void)plan2b;
    if (nstop)
        *nstop = 0;
    if (infostr && infostr_len)
        std::memset(infostr, ' ', infostr_len);
    if (!speckseed || !numaperil || !nx || !ny || !nz || !nxm || !nym || !nzm ||
        !nfft2d || !tabfft2 || !k0 || !xs || !ys || !zs || !E0 || !pp ||
        !xgaus || !ygaus || !zgaus || !aretecube || !Eloinx || !Eloiny || !Eloinz ||
        !FF || !imax || !deltakx || !deltaky || !Ediffkzpos || !Ediffkzneg || !beam ||
        !efficacite || !efficaciteref || !efficacitetrans || !nsectionsca ||
        !nquickdiffracte || !nside || !nstop || !infostr ||
        *nx <= 0 || *ny <= 0 || *nz <= 0 || *nxm < *nx || *nym < *ny || *nzm < *nz ||
        *nfft2d < 8 || *nfft2d > 16384 || *k0 <= CU_REAL(0) ||
        *aretecube <= CU_REAL(0) || (*nside != 1 && *nside != -1)) {
        set_error(nstop, infostr, infostr_len, "Invalid confocal energy arguments");
        return;
    }

    size_t nxny = 0;
    size_t mesh_count = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx), size_t(*ny), &nxny) ||
        !cuda_utils::checked_multiply(nxny, size_t(*nz), &mesh_count) ||
        mesh_count > size_t(std::numeric_limits<int>::max()) / 3) {
        set_error(nstop, infostr, infostr_len, "Confocal energy dimensions overflow");
        return;
    }
    const int nfft = *nfft2d;
    const int half = nfft / 2;
    const size_t plane = size_t(nfft) * size_t(nfft);
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

    if (!initialize_cuda_device()) {
        set_error(nstop, infostr, infostr_len, "CUDA device initialization failed");
        return;
    }
    cuda_data::DiffractionEnergyDeviceBuffers buffers{};
    if (!cuda_data::get_device_diffraction_energy_buffers(plane, mesh_count, &buffers)) {
        set_error(nstop, infostr, infostr_len, "Could not acquire confocal energy buffers");
        return;
    }

    const size_t field_value_count = 3 * mesh_count;
    const CUCOMPLEX* resident_field =
        get_device_result_vector_alias(FF, field_value_count);
    const CUCOMPLEX* device_green_field = resident_field
        ? resident_field : buffers.green_field;
    std::vector<CUCOMPLEX> host_field;
    std::vector<CU_REAL> host_zs(mesh_count);
    std::vector<int> host_map(static_cast<size_t>(nfft));
    if (!resident_field)
        host_field.resize(field_value_count);
    for (size_t i = 0; i < mesh_count; ++i) {
        host_zs[i] = zs[i];
        if (!resident_field)
            host_field[i] = cuda_utils::device_value(FF[i]);
    }
    for (int i = 0; i < nfft; ++i)
        host_map[size_t(i)] = tabfft2[i];
    const size_t field_bytes = field_value_count * sizeof(CUCOMPLEX);
    if ((!resident_field &&
         !cuda_ok(cuda_data::copy_host_to_device(buffers.green_field, host_field.data(), field_bytes,
                 cuda_utils::TransferCategory::SolverVector), "upload scattered field")) ||
        !cuda_ok(cuda_data::copy_host_to_device(buffers.z_coordinates, host_zs.data(),
            host_zs.size() * sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
            "upload z coordinates") ||
        !cuda_ok(cuda_data::copy_host_to_device(buffers.fft_indices, host_map.data(),
            host_map.size() * sizeof(int), cuda_utils::TransferCategory::Control),
            "upload FFT index map")) {
        set_error(nstop, infostr, infostr_len, "Could not upload confocal energy inputs");
        return;
    }

    if (*nsectionsca * *nquickdiffracte == 0) {
        const int spectrum_count = int(3 * plane);
        if (!cuda_ok(zero_vectors_wrapper(buffers.positive_spectrum, spectrum_count),
                     "clear positive scattered spectrum") ||
            !cuda_ok(zero_vectors_wrapper(buffers.negative_spectrum, spectrum_count),
                     "clear negative scattered spectrum")) {
            set_error(nstop, infostr, infostr_len, "Could not clear scattered spectra");
            return;
        }
        const CU_REAL phase_x = (xs[0] + CU_REAL(half) * *aretecube) * *deltakx;
        const CU_REAL phase_y = (ys[0] + CU_REAL(half) * *aretecube) * *deltaky;
        for (int layer = 0; layer < *nz; ++layer) {
            if (!cuda_ok(quick_diffraction_gather_wrapper(device_green_field,
                    buffers.layer_spectrum, buffers.fft_indices, *nx, *ny,
                    int(nxny), nfft, layer), "pack scattered-field layer") ||
                !cuda_utils::check_cufft(CUFFT_EXECUTE(buffers.fft_plan,
                    buffers.layer_spectrum, buffers.layer_spectrum, CUFFT_FORWARD),
                    "confocal FFT energy", "transform scattered-field layer") ||
                !cuda_ok(quick_diffraction_accumulate_wrapper(buffers.layer_spectrum,
                    buffers.fft_indices, buffers.z_coordinates, int(nxny), nfft,
                    *imax, layer, *k0, *deltakx, phase_x, phase_y,
                    buffers.positive_spectrum, buffers.negative_spectrum),
                    "accumulate scattered-field spectrum")) {
                set_error(nstop, infostr, infostr_len, "CUDA scattered-field diffraction failed");
                return;
            }
        }
    } else {
        std::vector<CUCOMPLEX> positive_host(3 * plane), negative_host(3 * plane);
        for (size_t i = 0; i < 3 * plane; ++i) {
            positive_host[i] = cuda_utils::device_value(Ediffkzpos[i]);
            negative_host[i] = cuda_utils::device_value(Ediffkzneg[i]);
        }
        if (!cuda_ok(cuda_data::copy_host_to_device(buffers.positive_spectrum,
                positive_host.data(), positive_host.size() * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::SolverVector), "upload positive spectrum") ||
            !cuda_ok(cuda_data::copy_host_to_device(buffers.negative_spectrum,
                negative_host.data(), negative_host.size() * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::SolverVector), "upload negative spectrum")) {
            set_error(nstop, infostr, infostr_len, "Could not upload scattered spectra");
            return;
        }
    }
    if (!cuda_ok(quick_energy_finalize_fields_wrapper(buffers.positive_spectrum,
            buffers.negative_spectrum, nfft, *imax, *deltakx, *deltaky, *k0),
            "normalize scattered spectra")) {
        set_error(nstop, infostr, infostr_len, "CUDA diffraction normalization failed");
        return;
    }

    std::vector<CU_REAL> phases_positive, phases_negative;
    int beam_kind = 0;
    const int width = 2 * *imax + 1;
    if (!make_incident_phases(beam, *speckseed, width,
            &phases_positive, &phases_negative, &beam_kind)) {
        set_error(nstop, infostr, infostr_len, "Unsupported confocal energy illumination");
        return;
    }
    if (!cuda_ok(cuda_data::copy_host_to_device(buffers.phase_positive,
            phases_positive.data(), phases_positive.size() * sizeof(CU_REAL),
            cuda_utils::TransferCategory::Control), "upload positive random phases") ||
        !cuda_ok(cuda_data::copy_host_to_device(buffers.phase_negative,
            phases_negative.data(), phases_negative.size() * sizeof(CU_REAL),
            cuda_utils::TransferCategory::Control), "upload negative random phases")) {
        set_error(nstop, infostr, infostr_len, "Could not upload confocal phase arrays");
        return;
    }
    const CU_REAL psi = *pp * pi / CU_REAL(180);
    const CU_REAL kmax = *k0 * *numaperil;
    if (!cuda_ok(confocal_energy_spectrum_wrapper(buffers.incident_spectrum,
            buffers.incident_positive_spectrum, buffers.incident_negative_spectrum,
            buffers.phase_positive, buffers.phase_negative, nfft, *imax, *k0,
            *deltakx, *deltaky, kmax * kmax, psi, *xgaus, *ygaus, *zgaus,
            cuda_utils::device_value(*E0), *nside, beam_kind),
            "generate confocal incident spectrum")) {
        set_error(nstop, infostr, infostr_len, "CUDA confocal illumination failed");
        return;
    }

    CU_REAL* partials = buffers.reduction_partials;
    if (!cuda_ok(confocal_energy_flux_wrapper(buffers.incident_positive_spectrum,
            buffers.incident_negative_spectrum, buffers.positive_spectrum,
            buffers.negative_spectrum, nfft, *imax, *deltakx, *deltaky,
            *k0, kmax * kmax, partials, partials + plane, partials + 2 * plane),
            "compute confocal energy flux")) {
        set_error(nstop, infostr, infostr_len, "CUDA confocal energy flux failed");
        return;
    }
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    CU_REAL fluxes[3]{};
    if (!handle ||
        !cuda_utils::sum_nonnegative(handle, int(plane), partials, &fluxes[0],
                                     "incident confocal flux reduction") ||
        !cuda_utils::sum_nonnegative(handle, int(plane), partials + plane, &fluxes[1],
                                     "reflected confocal flux reduction") ||
        !cuda_utils::sum_nonnegative(handle, int(plane), partials + 2 * plane, &fluxes[2],
                                     "transmitted confocal flux reduction") || fluxes[0] == CU_REAL(0)) {
        set_error(nstop, infostr, infostr_len, "CUDA confocal energy reduction failed");
        return;
    }
    const CU_REAL flux_scale = CU_REAL(4) * pi * pi * *deltaky * *deltakx /
        (*k0 * CU_REAL(8) * pi * CU_REAL(1.0e-7) * CU_REAL(299792458.0));
    std::printf("Incident flux         : %.16g W\nReflected flux        : %.16g W\n"
                "Transmitted flux      : %.16g W\nTotal flux            : %.16g W\n",
                double(fluxes[0] * flux_scale), double(fluxes[1] * flux_scale),
                double(fluxes[2] * flux_scale), double((fluxes[1] + fluxes[2]) * flux_scale));
    *efficacite = (fluxes[1] + fluxes[2]) / fluxes[0];
    *efficaciteref = fluxes[1] / fluxes[0];
    *efficacitetrans = fluxes[2] / fluxes[0];

    if (!publish_component_spectra(buffers, Eloinx, Eloiny, Eloinz,
            Ediffkzpos, Ediffkzneg, plane))
        set_error(nstop, infostr, infostr_len, "Could not publish confocal spectra");
}
