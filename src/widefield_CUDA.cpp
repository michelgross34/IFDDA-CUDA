#include "cuda_data.h"
#include "widefield_CUDA.h"

#include "cuda_utils.h"
#include "kernel.h"

#include <climits>
#include <cstddef>
#include <limits>
#include <vector>

//-----------------------------------------------------------
extern "C" int ifdda_widefield_generate_green_c99(
    const void* const host_tensor_keys[6],
    int nx, int ny, int nz, int nxm, int nym, int nzm,
    int nx2, int ny2, int nz2, CU_REAL cell_size, CU_REAL k0, int nquad,
    GreenTensorDevicePointers* device_green) {
    if (device_green)
        *device_green = {};
    if (!host_tensor_keys || nx <= 0 || ny <= 0 || nz <= 0 || nxm < nx ||
        nym < ny || nzm < nz || nx2 < 2 * nx || ny2 < 2 * ny || nz2 < 2 * nz ||
        cell_size <= CU_REAL(0) || k0 <= CU_REAL(0) || nquad < 0 || nquad > 5 ||
        !device_green)
        return 1;

    size_t xy = 0;
    size_t tensor_count = 0;
    size_t fft_xy = 0;
    size_t fft_count = 0;
    if (!cuda_utils::checked_multiply(size_t(nxm) + 1u, size_t(nym) + 1u, &xy) ||
        !cuda_utils::checked_multiply(xy, size_t(nzm) + 1u, &tensor_count) ||
        !cuda_utils::checked_multiply(size_t(nx2), size_t(ny2), &fft_xy) ||
        !cuda_utils::checked_multiply(fft_xy, size_t(nz2), &fft_count))
        return 2;

    // This grid may differ from the main solver Green tensor. Keep the main
    // six-component allocation intact and use a separate fixed widefield one.
    GreenTensorDevicePointers green = cuda_data::get_widefield_green(tensor_count);
    cuda_data::GreenTensorDeviceBuffers scratch{};
    if (!green ||
        !cuda_data::get_device_vector_green_buffers(tensor_count, fft_count, &scratch) ||
        !scratch.spatial || !scratch.spectra.data[0] || !scratch.spectra.data[1] ||
        !scratch.spectra.data[2]) {
        (void)cuda_data::release_device_vector_green_buffers();
        return 3;
    }

    const cufftHandle fft_plan = get_fft_plan(nx2, ny2, nz2);
    if (!fft_plan) {
        (void)cuda_data::release_device_vector_green_buffers();
        return 3;
    }
    const cudaError_t status = generate_vector_green_tensor_wrapper(
        green, scratch.spatial, scratch.spectra, nx, ny, nz,
        nxm, nym, nzm, nx2, ny2, nz2, fft_plan, cell_size, k0,
        COMPLEX(REAL(1), REAL(0)), 0, nquad, 0);
    const bool scratch_released = cuda_data::release_device_vector_green_buffers();
    if (!cuda_utils::check_cuda(status, "wide-field Green generation",
            "generate dyadic tensor and cuFFT") || !scratch_released)
        return 4;
    *device_green = green;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_widefield_prepare_grid_c99(
    const CU_REAL* dipole_x, const CU_REAL* dipole_y, const CU_REAL* dipole_z,
    int dipole_count, int nx, int ny, int nz,
    int nxmp, int nymp, int nzmp,
    CU_REAL xmin, CU_REAL ymin, CU_REAL zmin,
    CU_REAL cell_size, CU_REAL comparison_tolerance,
    cuda_data::WideFieldDeviceBuffers* buffers) {
    if (!dipole_x || !dipole_y || !dipole_z || !buffers || dipole_count <= 0 ||
        nx <= 0 || ny <= 0 || nz <= 0 || nxmp < 0 || nymp < 0 || nzmp < 0 ||
        cell_size <= CU_REAL(0) || comparison_tolerance < CU_REAL(0) ||
        nx > INT_MAX / ny || nx * ny > INT_MAX / nz)
        return 1;

    const size_t grid_count = size_t(nx) * size_t(ny) * size_t(nz);
    if (grid_count > size_t(INT_MAX) ||
        grid_count > std::numeric_limits<size_t>::max() / (3u * sizeof(CUCOMPLEX)))
        return 2;

    if (!initialize_cuda_device())
        return 3;

    const CU_REAL* resident_dipole_x = nullptr;
    const CU_REAL* resident_dipole_y = nullptr;
    const CU_REAL* resident_dipole_z = nullptr;
    const bool coordinates_are_resident =
        cuda_data::get_device_geometry_coordinate_aliases(
            dipole_x, dipole_y, dipole_z, size_t(dipole_count),
            &resident_dipole_x, &resident_dipole_y, &resident_dipole_z);

    cuda_data::WideFieldBufferRequest request{};
    request.grid_count = grid_count;
    request.dipole_count = coordinates_are_resident ? 0u : size_t(dipole_count);
    cuda_data::WideFieldDeviceBuffers acquired{};
    if (!cuda_data::get_device_widefield_buffers(request, &acquired) ||
        (!coordinates_are_resident &&
         (!acquired.dipole_x || !acquired.dipole_y || !acquired.dipole_z)) ||
        !acquired.grid_x || !acquired.grid_y || !acquired.grid_z ||
        !acquired.dipole_map)
        return 4;

    using cuda_utils::TransferCategory;
    const CU_REAL* map_dipole_x = coordinates_are_resident ?
        resident_dipole_x : acquired.dipole_x;
    const CU_REAL* map_dipole_y = coordinates_are_resident ?
        resident_dipole_y : acquired.dipole_y;
    const CU_REAL* map_dipole_z = coordinates_are_resident ?
        resident_dipole_z : acquired.dipole_z;
    cudaError_t status = cudaSuccess;
    if (!coordinates_are_resident) {
        const size_t dipole_bytes = size_t(dipole_count) * sizeof(CU_REAL);
        status = cuda_data::copy_host_to_device(
            acquired.dipole_x, dipole_x, dipole_bytes, TransferCategory::Geometry);
        if (status == cudaSuccess)
            status = cuda_data::copy_host_to_device(
                acquired.dipole_y, dipole_y, dipole_bytes, TransferCategory::Geometry);
        if (status == cudaSuccess)
            status = cuda_data::copy_host_to_device(
                acquired.dipole_z, dipole_z, dipole_bytes, TransferCategory::Geometry);
    }
    if (status == cudaSuccess)
        status = widefield_build_grid_wrapper(
            acquired.grid_x, acquired.grid_y, acquired.grid_z, nx, ny, nz,
            nxmp, nymp, nzmp, xmin, ymin, zmin, cell_size);
    if (status == cudaSuccess)
        status = widefield_map_dipoles_wrapper(
            map_dipole_x, map_dipole_y, map_dipole_z,
            acquired.grid_x, acquired.grid_y, acquired.grid_z,
            acquired.dipole_map, dipole_count, nx, ny, nz, cell_size,
            comparison_tolerance);
    if (status != cudaSuccess) {
        cuda_utils::check_cuda(status, "wide-field setup", "build grid and map dipoles");
        return 5;
    }

    *buffers = acquired;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_widefield_scatter_dipoles_c99(
    const std::complex<CU_REAL>* dipole_scattered,
    const std::complex<CU_REAL>* dipole_incident,
    const std::complex<CU_REAL>* dipole_epsilon,
    int dipole_count, int grid_count, int isotropy,
    int preserve_incident_field,
    cuda_data::WideFieldDeviceBuffers* buffers) {
    if (!dipole_scattered || (!preserve_incident_field && !dipole_incident) ||
        !dipole_epsilon || !buffers ||
        dipole_count <= 0 || grid_count <= 0 ||
        (isotropy != 1 && isotropy != 3) ||
        (preserve_incident_field != 0 && preserve_incident_field != 1))
        return 1;

    const size_t dipoles = size_t(dipole_count);
    const size_t grids = size_t(grid_count);
    const size_t tensor_components = size_t(isotropy) * size_t(isotropy);
    if (dipoles > std::numeric_limits<size_t>::max() / 3u ||
        dipoles > std::numeric_limits<size_t>::max() / tensor_components ||
        grids > std::numeric_limits<size_t>::max() / tensor_components ||
        grids > size_t(INT_MAX) || dipoles > size_t(INT_MAX))
        return 2;

    if (!initialize_cuda_device())
        return 3;

    cuda_data::WideFieldBufferRequest request{};
    request.grid_count = grids;
    request.dipole_count = dipoles;
    request.dipole_field_count = 3u * dipoles;
    request.dipole_incident_count = preserve_incident_field ? 0u : 3u * dipoles;
    request.dipole_material_count = tensor_components * dipoles;
    request.grid_material_count = tensor_components * grids;
    cuda_data::WideFieldDeviceBuffers acquired{};
    if (!cuda_data::get_device_widefield_buffers(request, &acquired) ||
        !acquired.dipole_scattered ||
        (!preserve_incident_field && !acquired.dipole_incident) ||
        !acquired.dipole_epsilon || !acquired.dipole_map ||
        !acquired.initial_field || !acquired.scattered_field ||
        !acquired.polarization)
        return 4;

    std::vector<CUCOMPLEX> scattered;
    std::vector<CUCOMPLEX> incident;
    std::vector<CUCOMPLEX> epsilon;
    try {
        scattered.resize(3u * dipoles);
        if (!preserve_incident_field)
            incident.resize(3u * dipoles);
        epsilon.resize(tensor_components * dipoles);
    } catch (...) {
        return 5;
    }
    for (size_t i = 0; i < scattered.size(); ++i) {
        scattered[i].x = dipole_scattered[i].real();
        scattered[i].y = dipole_scattered[i].imag();
        if (!preserve_incident_field) {
            incident[i].x = dipole_incident[i].real();
            incident[i].y = dipole_incident[i].imag();
        }
    }
    for (size_t i = 0; i < epsilon.size(); ++i) {
        epsilon[i].x = dipole_epsilon[i].real();
        epsilon[i].y = dipole_epsilon[i].imag();
    }

    using cuda_utils::TransferCategory;
    cudaError_t status = cuda_data::copy_host_to_device(
        acquired.dipole_scattered, scattered.data(),
        scattered.size() * sizeof(CUCOMPLEX), TransferCategory::SolverVector);
    if (status == cudaSuccess && !preserve_incident_field)
        status = cuda_data::copy_host_to_device(
            acquired.dipole_incident, incident.data(),
            incident.size() * sizeof(CUCOMPLEX), TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            acquired.dipole_epsilon, epsilon.data(),
            epsilon.size() * sizeof(CUCOMPLEX), TransferCategory::Material);
    if (status == cudaSuccess)
        status = widefield_scatter_dipoles_wrapper(
            acquired.dipole_map, grid_count, dipole_count, isotropy,
            preserve_incident_field, acquired.dipole_scattered,
            acquired.dipole_incident, acquired.dipole_epsilon,
            acquired.initial_field, acquired.scattered_field,
            acquired.polarization);
    if (status != cudaSuccess) {
        cuda_utils::check_cuda(status, "wide-field setup", "scatter dipole data");
        return 6;
    }

    *buffers = acquired;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_widefield_scatter_dipoles_host_c99(
    const std::complex<CU_REAL>* dipole_scattered,
    const std::complex<CU_REAL>* dipole_incident,
    const std::complex<CU_REAL>* fortran_epsilon, int nmax,
    int dipole_count, int grid_count, int isotropy,
    int preserve_incident_field,
    cuda_data::WideFieldDeviceBuffers* buffers) {
    if (!dipole_scattered || (!preserve_incident_field && !dipole_incident) ||
        !fortran_epsilon || !buffers || nmax < dipole_count ||
        dipole_count <= 0 || grid_count <= 0 ||
        (isotropy != 1 && isotropy != 3) ||
        (preserve_incident_field != 0 && preserve_incident_field != 1))
        return 1;

    const size_t dipoles = size_t(dipole_count);
    const size_t grids = size_t(grid_count);
    const size_t tensor_components = size_t(isotropy) * size_t(isotropy);
    if (dipoles > std::numeric_limits<size_t>::max() / 3u ||
        size_t(nmax) > std::numeric_limits<size_t>::max() / tensor_components ||
        dipoles > std::numeric_limits<size_t>::max() / tensor_components ||
        grids > std::numeric_limits<size_t>::max() / tensor_components ||
        grids > size_t(INT_MAX) || dipoles > size_t(INT_MAX))
        return 2;

    if (!initialize_cuda_device())
        return 3;

    cuda_data::WideFieldBufferRequest request{};
    request.grid_count = grids;
    request.dipole_count = dipoles;
    request.dipole_field_count = 3u * dipoles;
    request.dipole_incident_count = preserve_incident_field ? 0u : 3u * dipoles;
    request.dipole_material_count = tensor_components * dipoles;
    request.dipole_material_source_count = tensor_components * size_t(nmax);
    request.grid_material_count = tensor_components * grids;
    cuda_data::WideFieldDeviceBuffers acquired{};
    if (!cuda_data::get_device_widefield_buffers(request, &acquired) ||
        !acquired.dipole_scattered ||
        (!preserve_incident_field && !acquired.dipole_incident) ||
        !acquired.dipole_epsilon || !acquired.dipole_epsilon_source ||
        !acquired.dipole_map || !acquired.initial_field ||
        !acquired.scattered_field || !acquired.polarization)
        return 4;

    using cuda_utils::TransferCategory;
    cudaError_t status = cuda_data::copy_host_complex_to_device(
        acquired.dipole_scattered, dipole_scattered, 3u * dipoles,
        TransferCategory::SolverVector);
    if (status == cudaSuccess && !preserve_incident_field)
        status = cuda_data::copy_host_complex_to_device(
            acquired.dipole_incident, dipole_incident, 3u * dipoles,
            TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_complex_to_device(
            acquired.dipole_epsilon_source, fortran_epsilon,
            request.dipole_material_source_count, TransferCategory::Material);
    if (status == cudaSuccess)
        status = widefield_pack_dipole_epsilon_wrapper(
            acquired.dipole_epsilon_source, acquired.dipole_epsilon,
            dipole_count, nmax, isotropy);
    if (status == cudaSuccess)
        status = widefield_scatter_dipoles_wrapper(
            acquired.dipole_map, grid_count, dipole_count, isotropy,
            preserve_incident_field, acquired.dipole_scattered,
            acquired.dipole_incident, acquired.dipole_epsilon,
            acquired.initial_field, acquired.scattered_field,
            acquired.polarization);
    if (status != cudaSuccess) {
        cuda_utils::check_cuda(status, "wide-field setup",
                               "upload and scatter dipole data");
        return 5;
    }

    *buffers = acquired;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_widefield_scatter_dipoles_device_c99(
    const CUCOMPLEX* dipole_scattered_device,
    const std::complex<CU_REAL>* fortran_epsilon, int nmax,
    int dipole_count, int grid_count, int isotropy,
    cuda_data::WideFieldDeviceBuffers* buffers) {
    if (!dipole_scattered_device || !fortran_epsilon || !buffers ||
        nmax < dipole_count ||
        dipole_count <= 0 || grid_count <= 0 ||
        (isotropy != 1 && isotropy != 3))
        return 1;

    const size_t dipoles = size_t(dipole_count);
    const size_t grids = size_t(grid_count);
    const size_t tensor_components = size_t(isotropy) * size_t(isotropy);
    if (dipoles > std::numeric_limits<size_t>::max() / tensor_components ||
        grids > std::numeric_limits<size_t>::max() / tensor_components ||
        grids > size_t(INT_MAX) || dipoles > size_t(INT_MAX))
        return 2;

    if (!initialize_cuda_device())
        return 3;

    cuda_data::WideFieldBufferRequest request{};
    request.grid_count = grids;
    request.dipole_count = dipoles;
    request.dipole_material_count = tensor_components * dipoles;
    request.dipole_material_source_count = tensor_components * size_t(nmax);
    request.grid_material_count = tensor_components * grids;
    cuda_data::WideFieldDeviceBuffers acquired{};
    if (!cuda_data::get_device_widefield_buffers(request, &acquired) ||
        !acquired.dipole_epsilon || !acquired.dipole_epsilon_source ||
        !acquired.dipole_map ||
        !acquired.initial_field || !acquired.scattered_field ||
        !acquired.polarization)
        return 4;

    using cuda_utils::TransferCategory;
    cudaError_t status = cuda_data::copy_host_complex_to_device(
        acquired.dipole_epsilon_source, fortran_epsilon,
        request.dipole_material_source_count,
        TransferCategory::Material);
    if (status == cudaSuccess)
        status = widefield_pack_dipole_epsilon_wrapper(
            acquired.dipole_epsilon_source, acquired.dipole_epsilon,
            dipole_count, nmax, isotropy);
    if (status == cudaSuccess)
        status = widefield_scatter_dipoles_wrapper(
            acquired.dipole_map, grid_count, dipole_count, isotropy, 1,
            dipole_scattered_device, nullptr, acquired.dipole_epsilon,
            acquired.initial_field, acquired.scattered_field,
            acquired.polarization);
    if (status != cudaSuccess) {
        cuda_utils::check_cuda(status, "wide-field setup",
                               "scatter resident dipole field");
        return 5;
    }

    *buffers = acquired;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_widefield_plane_wave_c99(
    int grid_count,
    CU_REAL kx,
    CU_REAL ky,
    CU_REAL kz,
    std::complex<CU_REAL> ex0,
    std::complex<CU_REAL> ey0,
    std::complex<CU_REAL> ez0,
    int dipole_count,
    cuda_data::WideFieldDeviceBuffers* buffers) {
    if (!buffers || grid_count <= 0 || dipole_count <= 0 ||
        size_t(grid_count) > std::numeric_limits<size_t>::max() / 3u)
        return 1;

    if (!initialize_cuda_device())
        return 2;

    cuda_data::WideFieldBufferRequest request{};
    request.grid_count = size_t(grid_count);
    request.dipole_count = size_t(dipole_count);
    cuda_data::WideFieldDeviceBuffers acquired{};
    if (!cuda_data::get_device_widefield_buffers(request, &acquired) ||
        !acquired.grid_x || !acquired.grid_y || !acquired.grid_z || !acquired.incident)
        return 3;

    const cudaError_t status = incident_plane_wave_wrapper(
        acquired.grid_x, acquired.grid_y, acquired.grid_z, acquired.incident,
        grid_count, kx, ky, COMPLEX(kz, REAL(0)), ex0, ey0, ez0);
    if (status != cudaSuccess) {
        cuda_utils::check_cuda(status, "wide-field incident field", "launch plane-wave kernel");
        return 4;
    }

    *buffers = acquired;
    return 0;
}
