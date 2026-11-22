#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"
#include "microsconfocal_CUDA.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace {

std::mutex confocal_coordinate_mutex;
const REAL* confocal_xs_source = nullptr;
const REAL* confocal_ys_source = nullptr;
const REAL* confocal_zs_source = nullptr;
CU_REAL* confocal_xs_device = nullptr;
CU_REAL* confocal_ys_device = nullptr;
CU_REAL* confocal_zs_device = nullptr;
size_t confocal_coordinate_count = 0;

//-----------------------------------------------------------
void set_info(char* destination, std::size_t length, const char* message)
{
    if (destination == nullptr || length == 0)
        return;

    std::memset(destination, ' ', length);
    std::memcpy(destination, message, std::min(length, std::strlen(message)));
}

//-----------------------------------------------------------
cudaError_t upload_confocal_coordinates(
    CU_REAL* device_xs,
    CU_REAL* device_ys,
    CU_REAL* device_zs,
    const REAL* host_xs,
    const REAL* host_ys,
    const REAL* host_zs,
    size_t count)
{
    if (!device_xs || !device_ys || !device_zs ||
        !host_xs || !host_ys || !host_zs || count == 0)
        return cudaErrorInvalidValue;

    std::lock_guard<std::mutex> lock(confocal_coordinate_mutex);
    if (confocal_xs_source == host_xs && confocal_ys_source == host_ys &&
        confocal_zs_source == host_zs && confocal_xs_device == device_xs &&
        confocal_ys_device == device_ys && confocal_zs_device == device_zs &&
        confocal_coordinate_count == count)
        return cudaSuccess;

    using cuda_utils::TransferCategory;
    cudaError_t status = cuda_data::copy_host_to_device(
        device_xs, host_xs, count * sizeof(REAL), TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            device_ys, host_ys, count * sizeof(REAL), TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            device_zs, host_zs, count * sizeof(REAL), TransferCategory::Geometry);
    if (status == cudaSuccess)
    {
        confocal_xs_source = host_xs;
        confocal_ys_source = host_ys;
        confocal_zs_source = host_zs;
        confocal_xs_device = device_xs;
        confocal_ys_device = device_ys;
        confocal_zs_device = device_zs;
        confocal_coordinate_count = count;
    }
    return status;
}

} // namespace

//-----------------------------------------------------------
extern "C" int microsconfocal_incident_field_c99(
    const int* nbsphere,
    const REAL* k0,
    const REAL* kxinc,
    const REAL* kyinc,
    const REAL* psi,
    const COMPLEX* E0,
    const REAL* xs,
    const REAL* ys,
    const REAL* zs,
    COMPLEX* FF0,
    COMPLEX* XI,
    COMPLEX* Ax,
    COMPLEX* Ay,
    COMPLEX* Az,
    REAL* kzinc)
{
    if (nbsphere == nullptr || k0 == nullptr || kxinc == nullptr ||
        kyinc == nullptr || psi == nullptr || E0 == nullptr || xs == nullptr ||
        ys == nullptr || zs == nullptr || FF0 == nullptr || XI == nullptr ||
        Ax == nullptr || Ay == nullptr || Az == nullptr || kzinc == nullptr ||
        *nbsphere < 0 || *k0 <= REAL(0) ||
        size_t(*nbsphere) > std::numeric_limits<size_t>::max() / 3u)
        return 1;

    const REAL k02 = *k0 * *k0;
    const REAL kp2 = *kxinc * *kxinc + *kyinc * *kyinc;
    if (kp2 > k02)
        return 1;

    *kzinc = std::sqrt(k02 - kp2);
    const REAL norm_p = std::sqrt(*kzinc * *kzinc + *kxinc * *kxinc);
    const REAL norm_s = std::sqrt(*kzinc * *kzinc + *kyinc * *kyinc);
    if (norm_p == REAL(0) || norm_s == REAL(0))
        return 1;

    const REAL cos_psi = std::cos(*psi);
    const REAL sin_psi = std::sin(*psi);
    *Ax = *E0 * (cos_psi * *kzinc / norm_p);
    *Ay = *E0 * (sin_psi * *kzinc / norm_s);
    *Az = *E0 * (cos_psi * (-*kxinc / norm_p) +
                  sin_psi * (-*kyinc / norm_s));

    const size_t count = size_t(*nbsphere);
    if (count == 0)
        return 0;

    if (!initialize_cuda_device())
        return 2;

    cuda_data::IncidentFieldBufferRequest request{};
    request.coordinate_count = count;
    request.field_count = 3u * count;
    cuda_data::IncidentFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_incident_field_buffers(request, &buffers) ||
        buffers.xs == nullptr || buffers.ys == nullptr || buffers.zs == nullptr ||
        buffers.field == nullptr)
        return 3;

    using cuda_utils::TransferCategory;
    cudaError_t status = upload_confocal_coordinates(
        buffers.xs, buffers.ys, buffers.zs, xs, ys, zs, count);

    if (status == cudaSuccess)
        status = incident_plane_wave_wrapper(
            buffers.xs, buffers.ys, buffers.zs, buffers.field, *nbsphere,
            *kxinc, *kyinc, COMPLEX(*kzinc, REAL(0)),
            *Ax, *Ay, *Az);

    const bool publish_incident_alias = take_device_only_result_vector_request(FF0);
    const bool incident_alias_registered = status == cudaSuccess && publish_incident_alias &&
        register_gpbicg_device_b_alias(FF0, buffers.field, 3u * count);
    if (status == cudaSuccess && publish_incident_alias && !incident_alias_registered)
        status = cudaErrorUnknown;
    if (status == cudaSuccess && !publish_incident_alias)
        status = cuda_data::copy_device_complex_to_host(
            FF0, buffers.field, 3u * count, TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        if (incident_alias_registered)
            release_gpbicg_device_b_alias(FF0);
        return 6;
    }

    if (!publish_incident_alias)
        std::copy(FF0, FF0 + 3u * count, XI);
    return 0;
}

//-----------------------------------------------------------
extern "C" int microsconfocal_validate_c99(
    const int* nfft2d,
    const REAL* nmtom,
    const REAL* numaperk,
    const REAL* deltak,
    const int* nside,
    const REAL* stepzin,
    const REAL* zin,
    const REAL* zsupin,
    REAL* stepz,
    REAL* zinf,
    REAL* zsup,
    int* nplanz,
    int* nplanze,
    int* nstop,
    char* infostr,
    std::size_t infostr_len)
{
    if (nfft2d == nullptr || nmtom == nullptr || numaperk == nullptr ||
        deltak == nullptr || nside == nullptr || stepzin == nullptr ||
        zin == nullptr || zsupin == nullptr || stepz == nullptr ||
        zinf == nullptr || zsup == nullptr || nplanz == nullptr ||
        nplanze == nullptr || nstop == nullptr)
        return 1;

    *nstop = 0;
    *zinf = *zin * *nmtom;
    *zsup = *zsupin * *nmtom;
    *stepz = std::abs(*stepzin);

    if (*zinf > *zsup)
    {
        set_info(infostr, infostr_len,
            "lower bound larger than the upper bound in z-stack");
        *nstop = 1;
        return 1;
    }

    if (*stepz <= REAL(1.0e-9) || *zsup == *zinf)
    {
        *zsup = *zinf;
        *stepz = REAL(0);
        *nplanz = 1;
        *nplanze = 1;
    }
    else
    {
        const REAL planes = (*zsup - *zinf) / *stepz;
        if (!std::isfinite(planes) ||
            planes > REAL(INT32_MAX) - REAL(1))
        {
            set_info(infostr, infostr_len, "too many plane in the z-stack");
            *nstop = 1;
            return 1;
        }
        *nplanz = int(std::floor(planes + REAL(0.5))) + 1;
        *nplanze = *nplanz / 2;
    }

    if (*nplanz > *nfft2d)
    {
        set_info(infostr, infostr_len, "too many plane in the z-stack");
        *nstop = 1;
        return 1;
    }
    if (*nfft2d > 16384)
    {
        set_info(infostr, infostr_len, "nfft2d too large");
        *nstop = 1;
        return 1;
    }
    if (*deltak >= *numaperk)
    {
        set_info(infostr, infostr_len, "In FFT lens nfft2d too small");
        *nstop = 1;
        return 1;
    }
    if (*nside != 1 && *nside != -1)
    {
        set_info(infostr, infostr_len, "problem with side");
        *nstop = -1;
        return 1;
    }

    return 0;
}

//-----------------------------------------------------------
extern "C" int microsconfocal_classical_diffraction_device_c99(
    const int* nfft2d,
    const int* imaxk0,
    const int* nside,
    const int* nbsphere,
    const REAL* k0,
    const REAL* numaperk,
    const REAL* deltakx,
    const REAL* deltaky,
    const REAL* xs,
    const REAL* ys,
    const REAL* zs,
    const COMPLEX* field,
    const CUCOMPLEX** diffraction)
{
    if (nfft2d == nullptr || imaxk0 == nullptr || nside == nullptr ||
        nbsphere == nullptr || k0 == nullptr || numaperk == nullptr ||
        deltakx == nullptr || deltaky == nullptr || xs == nullptr ||
        ys == nullptr || zs == nullptr || field == nullptr ||
        diffraction == nullptr || *nfft2d <= 0 || *nfft2d > 16384 ||
        *imaxk0 < 0 || 2LL * *imaxk0 + 1 > *nfft2d ||
        *nbsphere <= 0 || *k0 <= REAL(0) || *numaperk <= REAL(0) ||
        *deltakx <= REAL(0) || *deltaky <= REAL(0) ||
        (*nside != 1 && *nside != -1))
        return 1;

    const size_t dipole_count = size_t(*nbsphere);
    const size_t fft_size = size_t(*nfft2d);
    const size_t plane_size = fft_size * fft_size;
    if (dipole_count > std::numeric_limits<size_t>::max() / 3u ||
        plane_size > std::numeric_limits<size_t>::max() / 3u)
        return 2;
    *diffraction = nullptr;

    if (!initialize_cuda_device())
        return 3;

    cuda_data::ConfocalMicroscopeDeviceBuffers buffers{};
    if (!cuda_data::get_device_confocal_microscope_buffers(
            dipole_count, 3u * plane_size, 0, &buffers) ||
        buffers.dipole_field == nullptr || buffers.xs == nullptr ||
        buffers.ys == nullptr || buffers.zs == nullptr ||
        buffers.diffraction == nullptr)
        return 4;

    using cuda_utils::TransferCategory;
    const size_t field_count = 3u * dipole_count;
    const CUCOMPLEX* resident_field =
        get_device_result_vector_alias(field, field_count);
    cudaError_t status = cudaSuccess;
    if (resident_field) {
        status = cuda_data::copy_device_to_device(
            buffers.dipole_field, resident_field,
            field_count * sizeof(CUCOMPLEX));
    } else {
        status = cuda_data::copy_host_complex_to_device(
            buffers.dipole_field, field, field_count, TransferCategory::SolverVector);
    }
    if (status == cudaSuccess)
        status = upload_confocal_coordinates(
            buffers.xs, buffers.ys, buffers.zs, xs, ys, zs, dipole_count);
    if (status == cudaSuccess)
        status = confocal_direct_diffraction_wrapper(
            buffers.dipole_field, buffers.xs, buffers.ys, buffers.zs,
            *nbsphere, *nfft2d, *imaxk0, *nside, *k0, *numaperk,
            *deltakx, *deltaky, buffers.diffraction);
    if (status != cudaSuccess)
        return 6;
    *diffraction = buffers.diffraction;
    return 0;
}

//-------------------------------------------
extern "C" int microsconfocal_classical_diffraction_c99(
    const int* nfft2d, const int* imaxk0, const int* nside,
    const int* nbsphere, const REAL* k0, const REAL* numaperk,
    const REAL* deltakx, const REAL* deltaky, const REAL* xs,
    const REAL* ys, const REAL* zs, const COMPLEX* field,
    COMPLEX* diffraction) {
    if (!nfft2d || *nfft2d <= 0 || !diffraction)
        return 1;
    const CUCOMPLEX* device_diffraction = nullptr;
    const int status = microsconfocal_classical_diffraction_device_c99(
        nfft2d, imaxk0, nside, nbsphere, k0, numaperk, deltakx, deltaky,
        xs, ys, zs, field, &device_diffraction);
    if (status != 0)
        return status;
    const size_t plane_size = size_t(*nfft2d) * size_t(*nfft2d);
    return cuda_data::copy_device_to_host(
        diffraction, device_diffraction, 3u * plane_size * sizeof(CUCOMPLEX),
        cuda_utils::TransferCategory::Result) == cudaSuccess ? 0 : 6;
}
