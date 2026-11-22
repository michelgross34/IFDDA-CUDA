#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"
#include "microsdf_fields_CUDA.h"

#include <complex>
#include <cstddef>
#include <limits>
#include <mutex>
#include <vector>

namespace {

std::mutex microsdf_incident_mutex;
const CU_REAL* microsdf_incident_x_source = nullptr;
const CU_REAL* microsdf_incident_y_source = nullptr;
const CU_REAL* microsdf_incident_z_source = nullptr;
CU_REAL* microsdf_incident_x_device = nullptr;
CU_REAL* microsdf_incident_y_device = nullptr;
CU_REAL* microsdf_incident_z_device = nullptr;
size_t microsdf_incident_coordinate_count = 0;

//-----------------------------------------------------------
bool pack_complex(const std::complex<CU_REAL>* source,
    size_t count, std::vector<CUCOMPLEX>* packed) {
    if (!source || !packed) {
        return false;
    }
    packed->resize(count);
    for (size_t i = 0; i < count; ++i) {
        (*packed)[i].x = source[i].real();
        (*packed)[i].y = source[i].imag();
    }
    return true;
}

//-----------------------------------------------------------
void unpack_complex(const std::vector<CUCOMPLEX>& packed,
    size_t offset, size_t count, std::complex<CU_REAL>* destination) {
    for (size_t i = 0; i < count; ++i) {
        destination[i] = std::complex<CU_REAL>(
            packed[offset + i].x, packed[offset + i].y);
    }
}

} // namespace

//-----------------------------------------------------------
extern "C" int ifdda_microsdf_plane_wave_device_c99(
    const void* host_incident_key,
    const CU_REAL* xs,
    const CU_REAL* ys,
    const CU_REAL* zs,
    size_t dipole_count,
    CU_REAL k0,
    CU_REAL kx,
    CU_REAL ky,
    std::complex<CU_REAL> amplitude,
    CU_REAL pp,
    cuda_data::MicroscopeIncidentDeviceBuffers* buffers) {
    if (!host_incident_key || !xs || !ys || !zs || !buffers || dipole_count == 0 ||
        dipole_count > size_t(std::numeric_limits<int>::max()) ||
        dipole_count > std::numeric_limits<size_t>::max() / 6u ||
        k0 <= CU_REAL(0) || pp < CU_REAL(0) || pp > CU_REAL(1))
        return 1;

    const CU_REAL k02 = k0 * k0;
    const CU_REAL kp2 = kx * kx + ky * ky;
    if (kp2 > k02)
        return 2;
    const CU_REAL kz = std::sqrt(k02 - kp2);

    CU_REAL s = std::sqrt(CU_REAL(1) - pp);
    CU_REAL p = std::sqrt(pp);

    std::complex<CU_REAL> ex0;
    std::complex<CU_REAL> ey0;
    std::complex<CU_REAL> ez0;
    if (kz >= CU_REAL(0.99999) * k0) {
        ex0 = amplitude * p;
        ey0 = amplitude * s;
        ez0 = std::complex<CU_REAL>(CU_REAL(0), CU_REAL(0));
    } else {
        const CU_REAL transverse = std::sqrt(kx * kx + ky * ky);
        if (transverse == CU_REAL(0))
            return 3;
        const CU_REAL nsx = -ky / transverse;
        const CU_REAL nsy = kx / transverse;
        CU_REAL npx = kz * nsy;
        CU_REAL npy = -kz * nsx;
        CU_REAL npz = -kx * nsy + ky * nsx;
        const CU_REAL np = std::sqrt(npx * npx + npy * npy + npz * npz);
        if (np == CU_REAL(0))
            return 3;
        npx /= np;
        npy /= np;
        npz /= np;
        ex0 = amplitude * (s * nsx + p * npx);
        ey0 = amplitude * (s * nsy + p * npy);
        ez0 = amplitude * (p * npz);
    }

    if (!initialize_cuda_device())
        return 4;
    cuda_data::MicroscopeIncidentDeviceBuffers incident_buffers{};
    if (!cuda_data::get_device_microscope_incident_buffers(dipole_count, &incident_buffers) ||
        !incident_buffers.xs || !incident_buffers.ys || !incident_buffers.zs)
        return 5;

    {
        std::lock_guard<std::mutex> lock(microsdf_incident_mutex);
        const bool same_source = microsdf_incident_x_source == xs &&
            microsdf_incident_y_source == ys && microsdf_incident_z_source == zs &&
            microsdf_incident_x_device == incident_buffers.xs &&
            microsdf_incident_y_device == incident_buffers.ys &&
            microsdf_incident_z_device == incident_buffers.zs &&
            microsdf_incident_coordinate_count == dipole_count;
        if (!same_source) {
            using cuda_utils::TransferCategory;
            cudaError_t copy_status = cuda_data::copy_host_to_device(
                incident_buffers.xs, xs, dipole_count * sizeof(CU_REAL),
                TransferCategory::Geometry);
            if (copy_status == cudaSuccess)
                copy_status = cuda_data::copy_host_to_device(
                    incident_buffers.ys, ys, dipole_count * sizeof(CU_REAL),
                    TransferCategory::Geometry);
            if (copy_status == cudaSuccess)
                copy_status = cuda_data::copy_host_to_device(
                    incident_buffers.zs, zs, dipole_count * sizeof(CU_REAL),
                    TransferCategory::Geometry);
            if (copy_status != cudaSuccess)
                return 6;
            microsdf_incident_x_source = xs;
            microsdf_incident_y_source = ys;
            microsdf_incident_z_source = zs;
            microsdf_incident_x_device = incident_buffers.xs;
            microsdf_incident_y_device = incident_buffers.ys;
            microsdf_incident_z_device = incident_buffers.zs;
            microsdf_incident_coordinate_count = dipole_count;
        }
    }

    cudaError_t status = incident_plane_wave_wrapper(
        incident_buffers.xs, incident_buffers.ys, incident_buffers.zs,
        incident_buffers.incident, int(dipole_count), kx, ky,
        COMPLEX(kz, REAL(0)), ex0, ey0, ez0);
    if (status == cudaSuccess)
        status = incident_plane_wave_wrapper(
            incident_buffers.xs, incident_buffers.ys, incident_buffers.zs,
            incident_buffers.initial, int(dipole_count), kx, ky,
            COMPLEX(kz, REAL(0)), ex0, ey0, ez0);
    if (!cuda_utils::check_cuda(status, "dark-field incident field", "launch plane-wave kernels"))
        return 8;
    if (!register_gpbicg_device_b_alias(
            host_incident_key, incident_buffers.incident, 3u * dipole_count))
        return 9;
    *buffers = incident_buffers;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_microsdf_expand_scalar_fields_device_c99(
    const CUCOMPLEX* device_polarization,
    const CUCOMPLEX* device_scalar_local,
    std::complex<CU_REAL> ux,
    std::complex<CU_REAL> uy,
    std::complex<CU_REAL> uz,
    size_t dipole_count,
    cuda_data::DarkFieldMicroscopeDeviceBuffers* buffers) {
    if (!device_polarization || !device_scalar_local || !buffers || dipole_count == 0 ||
        dipole_count > size_t(std::numeric_limits<int>::max()) ||
        dipole_count > std::numeric_limits<size_t>::max() / 6u)
        return 1;

    cuda_data::DarkFieldMicroscopeDeviceBuffers acquired{};
    if (!cuda_data::get_device_dark_field_microscope_buffers(
            0, dipole_count, dipole_count, 0, 0, 0,
            6u * dipole_count, dipole_count, 0, &acquired) ||
        !acquired.scalar_solution || !acquired.solver_fields)
        return 2;

    const auto to_cuda = [](std::complex<CU_REAL> value) {
        return CUCOMPLEX{value.real(), value.imag()};
    };
    if (!cuda_utils::check_cuda(
            microscope_expand_scalar_fields_wrapper(
                acquired.scalar_solution, device_polarization, device_scalar_local,
                acquired.solver_fields, acquired.solver_fields + 3u * dipole_count,
                int(dipole_count), to_cuda(ux), to_cuda(uy), to_cuda(uz)),
            "dark-field scalar field expansion", "launch CUDA kernel"))
        return 3;

    *buffers = acquired;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_microsdf_expand_scalar_fields_c99(
    const std::complex<CU_REAL>* polarization,
    const std::complex<CU_REAL>* scalar_local,
    const std::complex<CU_REAL>* ux,
    const std::complex<CU_REAL>* uy,
    const std::complex<CU_REAL>* uz,
    size_t dipole_count,
    std::complex<CU_REAL>* xi,
    std::complex<CU_REAL>* scattered,
    std::complex<CU_REAL>* local) {
    if (!polarization || !scalar_local || !ux || !uy || !uz || !xi ||
        !scattered || !local || dipole_count == 0 ||
        dipole_count > size_t(std::numeric_limits<int>::max()) ||
        dipole_count > std::numeric_limits<size_t>::max() / 6u) {
        return 1;
    }

    try {
        cuda_data::DarkFieldMicroscopeDeviceBuffers buffers{};
        if (!cuda_data::get_device_dark_field_microscope_buffers(
                0, dipole_count, dipole_count, 0, 0, 0,
                6u * dipole_count, dipole_count, 0, &buffers)) {
            return 2;
        }

        std::vector<CUCOMPLEX> host_polarization;
        std::vector<CUCOMPLEX> host_scalar_local;
        std::vector<CUCOMPLEX> host_xi(dipole_count);
        std::vector<CUCOMPLEX> host_fields(6u * dipole_count);
        if (!pack_complex(polarization, dipole_count, &host_polarization) ||
            !pack_complex(scalar_local, dipole_count, &host_scalar_local)) {
            return 3;
        }

        if (cuda_data::copy_host_to_device(
                buffers.polarization, host_polarization.data(),
                dipole_count * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::Material) != cudaSuccess ||
            cuda_data::copy_host_to_device(
                buffers.scalar_local, host_scalar_local.data(),
                dipole_count * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::SolverVector) != cudaSuccess) {
            return 4;
        }

        if (ifdda_microsdf_expand_scalar_fields_device_c99(
                buffers.polarization, buffers.scalar_local, *ux, *uy, *uz,
                dipole_count, &buffers) != 0)
            return 5;

        if (cuda_data::copy_device_to_host(
                host_xi.data(), buffers.scalar_solution,
                dipole_count * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::Result) != cudaSuccess ||
            cuda_data::copy_device_to_host(
                host_fields.data(), buffers.solver_fields,
                6u * dipole_count * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::Result) != cudaSuccess) {
            return 6;
        }

        unpack_complex(host_xi, 0, dipole_count, xi);
        unpack_complex(host_fields, 0, 3u * dipole_count, scattered);
        unpack_complex(host_fields, 3u * dipole_count,
            3u * dipole_count, local);
        return 0;
    } catch (...) {
        return 7;
    }
}
