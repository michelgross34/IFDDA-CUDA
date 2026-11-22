#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

//-----------------------------------------------------------



namespace {

//-----------------------------------------------------------
void set_sphere_error(char* destination, size_t length, int* nstop,
                      const char* message) {
    if (destination) {
        std::memset(destination, ' ', length);
        if (message && length)
            std::memcpy(destination, message, std::min(length, std::strlen(message)));
    }
    if (nstop)
        *nstop = 1;
}

//-----------------------------------------------------------
bool write_sphere_mesh_files(int n, int occupied,
                             const CU_REAL* axis_x, const CU_REAL* axis_y,
                             const CU_REAL* axis_z, const CU_REAL* mesh_x,
                             const CU_REAL* mesh_y, const CU_REAL* mesh_z) {
    std::ofstream x_axis("x.mat");
    std::ofstream y_axis("y.mat");
    std::ofstream z_axis("z.mat");
    std::ofstream x_mesh("xc.mat");
    std::ofstream y_mesh("yc.mat");
    std::ofstream z_mesh("zc.mat");
    if (!x_axis || !y_axis || !z_axis || !x_mesh || !y_mesh || !z_mesh)
        return false;

    for (int k = 0; k < n; ++k)
        x_axis << axis_x[size_t(k)] << '\n';
    for (int j = 0; j < n; ++j)
        y_axis << axis_y[size_t(j) * size_t(n)] << '\n';
    for (int i = 0; i < n; ++i)
        z_axis << axis_z[size_t(i) * size_t(n) * size_t(n)] << '\n';
    for (int i = 0; i < occupied; ++i) {
        x_mesh << mesh_x[size_t(i)] << '\n';
        y_mesh << mesh_y[size_t(i)] << '\n';
        z_mesh << mesh_z[size_t(i)] << '\n';
    }
    return bool(x_axis && y_axis && z_axis && x_mesh && y_mesh && z_mesh);
}

//-----------------------------------------------------------
bool write_compact_sphere_mesh_files(int n, int occupied, CU_REAL radius,
                                     CU_REAL cell_size, CU_REAL center_x,
                                     CU_REAL center_y, CU_REAL center_z,
                                     const CU_REAL* packed_x,
                                     const CU_REAL* packed_y,
                                     const CU_REAL* packed_z) {
    std::ofstream x_axis("x.mat");
    std::ofstream y_axis("y.mat");
    std::ofstream z_axis("z.mat");
    std::ofstream x_mesh("xc.mat");
    std::ofstream y_mesh("yc.mat");
    std::ofstream z_mesh("zc.mat");
    if (!x_axis || !y_axis || !z_axis || !x_mesh || !y_mesh || !z_mesh)
        return false;

    for (int k = 0; k < n; ++k)
        x_axis << -radius + cell_size * (CU_REAL(k) + CU_REAL(0.5)) + center_x << '\n';
    for (int j = 0; j < n; ++j)
        y_axis << -radius + cell_size * (CU_REAL(j) + CU_REAL(0.5)) + center_y << '\n';
    for (int i = 0; i < n; ++i)
        z_axis << -radius + cell_size * (CU_REAL(i) + CU_REAL(0.5)) + center_z << '\n';
    for (int i = 0; i < occupied; ++i) {
        x_mesh << packed_x[size_t(i)] << '\n';
        y_mesh << packed_y[size_t(i)] << '\n';
        z_mesh << packed_z[size_t(i)] << '\n';
    }
    return bool(x_axis && y_axis && z_axis && x_mesh && y_mesh && z_mesh);
}

//-----------------------------------------------------------
int method_uses_mutable_sldr(const char* method, size_t method_length) {
    return method && method_length >= 2 && method[0] == 'L' && method[1] == 'R';
}

//-----------------------------------------------------------
// Material arrays keep the Fortran host ABI, but are initialized and populated
// on the device using the tabdip map built by the CPU. The geometry's device
// occupancy map is reused; tensor components are spaced by nmax.
bool fill_sphere_materials_gpu(
    const int* tabdip, size_t grid, int nmax, int niso, bool compact,
    bool anisotropic, const CUCOMPLEX* eps, const CUCOMPLEX* epsani,
    CUCOMPLEX scalar_alpha, const CUCOMPLEX* tensor_alpha,
    CUCOMPLEX* epsilon, CUCOMPLEX* polarisa) {
    const size_t tensor_count = size_t(nmax) * size_t(niso) * size_t(niso);
    cuda_data::CubeGeometryDeviceBuffers geometry{};
    cuda_data::PolarizabilityInitDeviceBuffers materials{};
    if (!cuda_data::get_device_cube_geometry_buffers(grid, &geometry) ||
        !cuda_data::get_device_polarizability_init_buffers(
            size_t(nmax), tensor_count, &materials))
        return false;

    // get_device_tabdip always copies the latest CPU map to GPU, even when
    // a previous allocation can be reused.
    const CU_INT* device_tabdip = get_device_tabdip(tabdip, grid);
    if (!device_tabdip)
        return false;

    if (!cuda_utils::check_cuda(
            initialize_polarizability_arrays_wrapper(
                materials.dipole_map, materials.polarizability,
                materials.epsilon, nmax, niso),
            "sphere material CUDA", "initialize all material slots"))
        return false;

    const CUCOMPLEX* device_tensor_alpha = nullptr;
    const CUCOMPLEX* device_tensor_epsilon = nullptr;
    if (anisotropic) {
        cuda_data::TensorPolarizabilityDeviceBuffers tensors{};
        if (!cuda_data::get_device_tensor_polarizability_buffers(&tensors))
            return false;
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(
                    tensors.value, tensor_alpha, 9 * sizeof(CUCOMPLEX),
                    cuda_utils::TransferCategory::Material),
                "sphere material CUDA", "upload tensor polarizability") ||
            !cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(
                    tensors.epsilon, epsani, 9 * sizeof(CUCOMPLEX),
                    cuda_utils::TransferCategory::Material),
                "sphere material CUDA", "upload tensor permittivity"))
            return false;
        device_tensor_alpha = tensors.value;
        device_tensor_epsilon = tensors.epsilon;
    }

    if (!cuda_utils::check_cuda(
            sphere_material_from_cpu_tabdip_wrapper(
                device_tabdip, geometry.dipole_map,
                materials.polarizability, materials.epsilon,
                device_tensor_alpha, device_tensor_epsilon,
                scalar_alpha, *eps, int(grid), nmax, int(compact), int(anisotropic)),
            "sphere material CUDA", "assign occupied sphere cells"))
        return false;

    const size_t material_bytes = tensor_count * sizeof(CUCOMPLEX);
    const auto transfer_start = std::chrono::steady_clock::now();
    const bool polarizability_copied = cuda_utils::check_cuda(
        cuda_data::copy_device_to_host(
            polarisa, materials.polarizability, material_bytes,
            cuda_utils::TransferCategory::Material),
        "sphere material CUDA", "download polarizability");
    const auto polarizability_end = std::chrono::steady_clock::now();
    if (!polarizability_copied)
        return false;

    const bool permittivity_copied = cuda_utils::check_cuda(
        cuda_data::copy_device_to_host(
            epsilon, materials.epsilon, material_bytes,
            cuda_utils::TransferCategory::Material),
        "sphere material CUDA", "download permittivity");
    const auto permittivity_end = std::chrono::steady_clock::now();
    if (std::getenv("IFDDA_SPHERE_BENCH")) {
        const double polarizability_ms = std::chrono::duration<double, std::milli>(
            polarizability_end - transfer_start).count();
        const double permittivity_ms = std::chrono::duration<double, std::milli>(
            permittivity_end - polarizability_end).count();
        std::fprintf(stderr,
                     "IFDDA_SPHERE_MATERIAL_D2H_MS polarisa=%.6f epsilon=%.6f bytes_each=%zu\n",
                     polarizability_ms, permittivity_ms, material_bytes);
    }
    return permittivity_copied;
}

//-----------------------------------------------------------
// MDDA/PML uses a larger, uniformly filled box (na=1). Only tabdip, geometry
// and the small material lookup table are constructed on the host; the final
// full-grid epsilon and polarisa arrays are assigned with the CUDA label kernel.
bool fill_sphere_pml_geometry(const COMPLEX* eps, const COMPLEX* eps0,
                              CU_REAL* lnrbc, CU_REAL* lcos, int ntypeabc,
                              CU_REAL* xs, CU_REAL* ys, CU_REAL* zs,
                              const CU_REAL* k0, CU_REAL* aretecube,
                              int* tabdip, int* pmlwrk, int n,
                              int nmax, int* nbsphere, int* ndipole,
                              int niso, CU_REAL radius, CU_REAL xg,
                              CU_REAL yg, CU_REAL zg, int nmat,
                              COMPLEX* epsilon, COMPLEX* polarisa) {
    if (niso != 1 || ntypeabc < 0 || ntypeabc > 6)
        return false;
    const size_t grid = size_t(n) * size_t(n) * size_t(n);
    CU_REAL lpml = CU_REAL(0);
    CU_REAL box_width = CU_REAL(2) * radius;
    int nrbc = 0, npml = 0;
    initnrbc_(k0, lcos, lnrbc, &lpml, &box_width, &n,
              &nrbc, &npml, pmlwrk, aretecube);
    if (!(*aretecube > CU_REAL(0)) || nrbc < 0 || npml < nrbc)
        return false;

    using HostComplex = std::complex<CU_REAL>;
    const HostComplex inside_epsilon = *eps;
    const HostComplex background = *eps0;
    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795);
    const CU_REAL scale = (*aretecube) * (*aretecube) * (*aretecube) /
                          (CU_REAL(4) * pi);
    const auto alpha = [background, scale](HostComplex v) {
        return (v - background) * scale;
    };

    const int m = nrbc + 1;
    if (m <= 0 || m > 256)
        return false;
    const size_t max_product = size_t(m) * size_t(m) * size_t(m);
    if (max_product > size_t(std::numeric_limits<int>::max() - 2))
        return false;
    const int label_count = ntypeabc == 0 ? 2 : int(max_product) + 2;
    std::vector<COMPLEX> label_eps(static_cast<size_t>(label_count));
    std::vector<COMPLEX> label_alpha(static_cast<size_t>(label_count));
    label_eps[0] = inside_epsilon;
    label_alpha[0] = alpha(inside_epsilon);
    label_eps[1] = HostComplex(1, 0);
    label_alpha[1] = alpha(HostComplex(1, 0));

    if (ntypeabc != 0) {
        const CU_REAL profile_alpha = CU_REAL((ntypeabc + 1) / 2);
        const CU_REAL denominator = ntypeabc % 2 != 0
            ? std::erf(profile_alpha) : std::tanh(profile_alpha);
        const HostComplex index_background = std::sqrt(background);
        for (size_t product = 1; product <= max_product; ++product) {
            const CU_REAL argument = profile_alpha *
                (CU_REAL(2) * CU_REAL(product) / CU_REAL(max_product) - CU_REAL(1));
            const CU_REAL modulation = (ntypeabc % 2 != 0
                 ? std::erf(argument) : std::tanh(argument)) / denominator;
            const HostComplex index = (index_background + HostComplex(1, 0)) /
                CU_REAL(2) + (HostComplex(1, 0) - index_background) *
                (modulation / CU_REAL(2));
            const HostComplex material = index * index;
            label_eps[product + 1] = material;
            label_alpha[product + 1] = alpha(material);
        }
    }

    std::vector<int> labels(grid);
    const CU_REAL box_start = -box_width / CU_REAL(2);
    for (size_t cell = 0; cell < grid; ++cell) {
        const int k = int(cell % size_t(n));
        const int j = int((cell / size_t(n)) % size_t(n));
        const int i = int(cell / (size_t(n) * size_t(n)));
        const CU_REAL x = box_start + *aretecube * (CU_REAL(k) + CU_REAL(0.5));
        const CU_REAL y = box_start + *aretecube * (CU_REAL(j) + CU_REAL(0.5));
        const CU_REAL z = box_start + *aretecube * (CU_REAL(i) + CU_REAL(0.5));
        xs[cell] = x + xg;
        ys[cell] = y + yg;
        zs[cell] = z + zg;
        tabdip[cell] = int(cell) + 1;
        // Match Fortran's sqrt(x*x+y*y+z*z) < rayon, including cells
        // extremely close to the sphere boundary. Squared comparison
        // changes the material classification on the PML grid.
        if (std::sqrt(x*x + y*y + z*z) < radius) {
            labels[cell] = 1;
        } else {
            labels[cell] = 2;
            if (ntypeabc > 0) {
                const int ii = std::min(i + 1, std::min(n - i, m));
                const int jj = std::min(j + 1, std::min(n - j, m));
                const int kk = std::min(k + 1, std::min(n - k, m));
                if (ii <= nrbc || jj <= nrbc || kk <= nrbc)
                    labels[cell] = 2 + ii * jj * kk;
            }
        }
    }
    std::fill(tabdip + grid, tabdip + size_t(nmax), 0);
    *ndipole = int(grid);
    *nbsphere = int(grid);
    if (ifdda_fill_labeled_geometry_materials_cuda(
            tabdip, grid, labels.data(), nmax, niso, label_count, 1,
            label_eps.data(), label_alpha.data(), epsilon, polarisa) != 0)
        return false;
    return nmat != 0 || write_sphere_mesh_files(
        n, int(grid), xs, ys, zs, xs, ys, zs);
}

} // namespace

//-----------------------------------------------------------
extern "C" int ifdda_fill_homogeneous_geometry_materials_cuda(
    const int* tabdip, size_t grid, int nmax, int niso,
    int compact, int anisotropic, const CUCOMPLEX* eps,
    const CUCOMPLEX* epsani, CUCOMPLEX scalar_alpha,
    const CUCOMPLEX* tensor_alpha, CUCOMPLEX* epsilon,
    CUCOMPLEX* polarisa) {
    return fill_sphere_materials_gpu(
        tabdip, grid, nmax, niso, compact != 0, anisotropic != 0,
        eps, epsani, scalar_alpha, tensor_alpha, epsilon, polarisa)
        ? 0 : 1;
}

// Host-typed variant of the same GPU material calculation. Unlike the
// historical packed ABI it does not pass CUCOMPLEX* host arrays to callers.
extern "C" int ifdda_fill_homogeneous_geometry_materials_host_cuda(
    const int* tabdip, size_t grid, int nmax, int niso,
    int compact, int anisotropic, const COMPLEX* eps,
    const COMPLEX* epsani, COMPLEX scalar_alpha,
    const COMPLEX* tensor_alpha, COMPLEX* epsilon, COMPLEX* polarisa) {
    if (!tabdip || !eps || !epsilon || !polarisa || grid == 0 ||
        nmax <= 0 || grid > size_t(nmax) || (niso != 1 && niso != 3) ||
        (anisotropic && (!epsani || !tensor_alpha)))
        return 1;
    const size_t tensor_count = size_t(nmax) * size_t(niso) * size_t(niso);
    cuda_data::CubeGeometryDeviceBuffers geometry{};
    cuda_data::PolarizabilityInitDeviceBuffers materials{};
    if (!cuda_data::get_device_cube_geometry_buffers(grid, &geometry) ||
        !cuda_data::get_device_polarizability_init_buffers(size_t(nmax),
                                                            tensor_count, &materials))
        return 2;
    const CU_INT* device_tabdip = get_device_tabdip(tabdip, grid);
    if (!device_tabdip || !cuda_utils::check_cuda(
            initialize_polarizability_arrays_wrapper(materials.dipole_map,
                materials.polarizability, materials.epsilon, nmax, niso),
            "typed host material CUDA", "initialize material slots"))
        return 3;
    const CUCOMPLEX* d_alpha = nullptr;
    const CUCOMPLEX* d_eps = nullptr;
    if (anisotropic) {
        cuda_data::TensorPolarizabilityDeviceBuffers tensors{};
        if (!cuda_data::get_device_tensor_polarizability_buffers(&tensors))
            return 4;
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_complex_to_device(tensors.value, tensor_alpha, 9,
                                                       cuda_utils::TransferCategory::Material),
                "typed host material CUDA", "upload alpha tensor") ||
            !cuda_utils::check_cuda(
                cuda_data::copy_host_complex_to_device(tensors.epsilon, epsani, 9,
                                                       cuda_utils::TransferCategory::Material),
                "typed host material CUDA", "upload epsilon tensor"))
            return 5;
        d_alpha = tensors.value;
        d_eps = tensors.epsilon;
    }
    if (!cuda_utils::check_cuda(
            sphere_material_from_cpu_tabdip_wrapper(device_tabdip,
                geometry.dipole_map, materials.polarizability, materials.epsilon,
                d_alpha, d_eps, cuda_utils::device_value(scalar_alpha),
                cuda_utils::device_value(*eps), int(grid), nmax,
                compact, anisotropic),
            "typed host material CUDA", "assign material"))
        return 6;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(polarisa, materials.polarizability,
                tensor_count, cuda_utils::TransferCategory::Material),
            "typed host material CUDA", "download polarizability") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(epsilon, materials.epsilon,
                tensor_count, cuda_utils::TransferCategory::Material),
            "typed host material CUDA", "download permittivity"))
        return 7;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_fill_labeled_geometry_materials_cuda(
    const int* tabdip, size_t grid, const int* material_labels,
    int nmax, int niso, int label_count, int vacuum_background,
    const COMPLEX* label_epsilon, const COMPLEX* label_alpha,
    COMPLEX* epsilon, COMPLEX* polarisa) {
    if (!tabdip || !material_labels || !label_epsilon || !label_alpha ||
        !epsilon || !polarisa || grid == 0 || grid > size_t(nmax) ||
        grid > size_t(std::numeric_limits<int>::max()) || label_count <= 0 ||
        (vacuum_background != 0 && vacuum_background != 1) ||
        (niso != 1 && niso != 3))
        return 1;

    struct TemporaryDeviceAllocation {
        void* ptr = nullptr;
        ~TemporaryDeviceAllocation() { if (ptr) cudaFree(ptr); }
    } device_epsilon_lookup, device_alpha_lookup;
    struct TemporaryIntegerDeviceAllocation {
        CU_INT* ptr = nullptr;
        ~TemporaryIntegerDeviceAllocation() { if (ptr) cudaFree(ptr); }
    } device_labels;
    const size_t label_bytes = size_t(nmax) * sizeof(CU_INT);
    const size_t lookup_bytes = size_t(label_count) * size_t(niso == 3 ? 9 : 1) *
                                sizeof(CUCOMPLEX);
    const auto allocate = [](TemporaryDeviceAllocation& buffer, size_t bytes,
                             const char* purpose) {
        return cuda_utils::check_cuda(cudaMalloc(&buffer.ptr, bytes),
                                      "labeled geometry material CUDA", purpose);
    };
    if (!cuda_utils::check_cuda(cudaMalloc(&device_labels.ptr, label_bytes),
                                "labeled geometry material CUDA", "allocate material labels") ||
        !allocate(device_epsilon_lookup, lookup_bytes, "allocate permittivity lookup") ||
        !allocate(device_alpha_lookup, lookup_bytes, "allocate polarizability lookup"))
        return 2;

    const CU_INT* device_tabdip = get_device_tabdip(tabdip, grid);
    cuda_data::PolarizabilityInitDeviceBuffers materials{};
    const size_t tensor_count = size_t(nmax) * size_t(niso) * size_t(niso);
    if (!device_tabdip || !cuda_data::get_device_polarizability_init_buffers(
                              size_t(nmax), tensor_count, &materials))
        return 3;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(device_labels.ptr, material_labels,
                                           label_bytes, cuda_utils::TransferCategory::Material),
            "labeled geometry material CUDA", "upload material IDs") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_complex_to_device(
                static_cast<CUCOMPLEX*>(device_epsilon_lookup.ptr),
                label_epsilon, size_t(label_count) * size_t(niso == 3 ? 9 : 1),
                cuda_utils::TransferCategory::Material),
            "labeled geometry material CUDA", "upload permittivity table") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_complex_to_device(
                static_cast<CUCOMPLEX*>(device_alpha_lookup.ptr),
                label_alpha, size_t(label_count) * size_t(niso == 3 ? 9 : 1),
                cuda_utils::TransferCategory::Material),
            "labeled geometry material CUDA", "upload polarizability table"))
        return 4;

    if (!cuda_utils::check_cuda(
            initialize_polarizability_arrays_wrapper(
                materials.dipole_map, materials.polarizability,
                materials.epsilon, nmax, niso),
            "labeled geometry material CUDA", "initialize output arrays"))
        return 5;
    // Some legacy stochastic geometries zero-filled every empty epsilon slot
    // instead of using vacuum (= identity). Preserve their exact host ABI.
    const size_t output_bytes = tensor_count * sizeof(CUCOMPLEX);
    if (!vacuum_background && !cuda_utils::check_cuda(
            cudaMemset(materials.epsilon, 0, output_bytes),
            "labeled geometry material CUDA", "zero background permittivity"))
        return 5;
    if (!cuda_utils::check_cuda(
            labeled_material_from_cpu_tabdip_wrapper(
                device_tabdip, device_labels.ptr,
                materials.polarizability, materials.epsilon,
                static_cast<const CUCOMPLEX*>(device_alpha_lookup.ptr),
                static_cast<const CUCOMPLEX*>(device_epsilon_lookup.ptr),
                int(grid), nmax, label_count, int(niso == 3)),
            "labeled geometry material CUDA", "assign labeled cells"))
        return 5;

    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(
                polarisa, materials.polarizability, tensor_count,
                cuda_utils::TransferCategory::Material),
            "labeled geometry material CUDA", "download polarizability") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(
                epsilon, materials.epsilon, tensor_count,
                cuda_utils::TransferCategory::Material),
            "labeled geometry material CUDA", "download permittivity"))
        return 6;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_scatter_gpu_cell_materials_cuda(
    const int* tabdip, size_t grid, int nmax,
    const CU_INT* device_inside_map,
    const CUCOMPLEX* device_cell_epsilon,
    const CUCOMPLEX* device_cell_alpha,
    COMPLEX* epsilon, COMPLEX* polarisa) {
    if (!tabdip || !device_inside_map || !device_cell_epsilon ||
        !device_cell_alpha || !epsilon || !polarisa || nmax <= 0 ||
        grid == 0 || grid > size_t(nmax) ||
        grid > size_t(std::numeric_limits<int>::max()))
        return 1;

    const CU_INT* device_tabdip = get_device_tabdip(tabdip, grid);
    cuda_data::PolarizabilityInitDeviceBuffers outputs{};
    if (!device_tabdip || !cuda_data::get_device_polarizability_init_buffers(
            size_t(nmax), size_t(nmax), &outputs))
        return 2;
    const size_t bytes = size_t(nmax) * sizeof(CUCOMPLEX);
    if (!cuda_utils::check_cuda(
            cudaMemset(outputs.epsilon, 0, bytes),
            "inhomogeneous material scatter", "zero epsilon output") ||
        !cuda_utils::check_cuda(
            cudaMemset(outputs.polarizability, 0, bytes),
            "inhomogeneous material scatter", "zero alpha output"))
        return 3;
    if (!cuda_utils::check_cuda(
            scatter_cell_material_from_cpu_tabdip_wrapper(
                device_tabdip, device_inside_map, device_cell_epsilon,
                device_cell_alpha, outputs.epsilon, outputs.polarizability,
                int(grid), nmax),
            "inhomogeneous material scatter", "scatter GPU cell materials"))
        return 4;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(
                epsilon, outputs.epsilon, size_t(nmax),
                cuda_utils::TransferCategory::Material),
            "inhomogeneous material scatter", "download epsilon") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(
                polarisa, outputs.polarizability, size_t(nmax),
                cuda_utils::TransferCategory::Material),
            "inhomogeneous material scatter", "download alpha"))
        return 5;
    return 0;
}

//-----------------------------------------------------------
extern "C" void objetsphere_legacy_cuda_(
    const char* trope, const COMPLEX* eps, const COMPLEX* epsani,
    const COMPLEX* eps0, CU_REAL* lnrbc, CU_REAL* lcos, int* ntypeabc,
    CU_REAL* xs, CU_REAL* ys, CU_REAL* zs, const CU_REAL* k0,
    CU_REAL* aretecube, int* tabdip, int* tabmulti, int* pmlwrk,
    const int* nnnr, const int* nmax, int* nbsphere, int* ndipole,
    int* nx, int* ny, int* nz, const char* methode, const int* na,
    COMPLEX* epsilon, COMPLEX* polarisa, const int* niso,
    CU_REAL* rayon, CU_REAL* xg, CU_REAL* yg, CU_REAL* zg,
    CU_REAL* sldr, const int* nmat, const int64_t* file_id,
    const int64_t* group_iddip, char* infostr, int* nstop,
    size_t trope_length, size_t methode_length, size_t infostr_length) {

    const auto sphere_benchmark_start = std::chrono::steady_clock::now();

    (void)tabmulti;
    (void)file_id;
    (void)group_iddip;
    (void)trope_length;

    if (!trope || !eps || !epsani || !eps0 || !xs || !ys || !zs || !k0 ||
        !aretecube || !tabdip || !nnnr || !nmax || !nbsphere || !ndipole ||
        !nx || !ny || !nz || !methode || !na || !epsilon || !polarisa ||
        !niso || !rayon || !xg || !yg || !zg || !sldr || !nmat ||
        !infostr || !nstop) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "invalid sphere geometry arguments");
        return;
    }

    *nstop = 0;
    *nbsphere = 0;
    *ndipole = 0;
    set_sphere_error(infostr, infostr_length, nstop, "");
    *nstop = 0;

    if (trope_length < 3 || methode_length < 2 || *nnnr <= 0 || *nmax <= 0 ||
        (*niso != 1 && *niso != 3) || (*na != -1 && *na != 0 && *na != 1) ||
        (*nmat != 0 && *nmat != 1) || *rayon <= CU_REAL(0) || *k0 <= CU_REAL(0)) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "unsupported sphere mode or invalid dimensions");
        return;
    }
    if (std::memcmp(trope, "iso", 3) != 0 && std::memcmp(trope, "ani", 3) != 0) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "invalid sphere isotropy");
        return;
    }
    if (std::memcmp(trope, "ani", 3) == 0 && *niso != 3) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "anisotropic sphere requires niso=3");
        return;
    }
    if (std::memcmp(methode, "PS", 2) == 0) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "PS sphere polarizability is not ported to CUDA yet");
        return;
    }

    const size_t n = size_t(*nnnr);
    if (n > size_t(std::numeric_limits<int>::max()) / n ||
        n * n > size_t(std::numeric_limits<int>::max()) / n) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "sphere grid size overflow");
        return;
    }
    const size_t grid = n * n * n;
    if (grid > size_t(*nmax)) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "nmax parameter too small: increase nxm nym nzm");
        return;
    }

    *rayon *= CU_REAL(1.0e-9);
    *xg *= CU_REAL(1.0e-9);
    *yg *= CU_REAL(1.0e-9);
    *zg *= CU_REAL(1.0e-9);
    *nx = *nnnr;
    *ny = *nnnr;
    *nz = *nnnr;
    *aretecube = CU_REAL(2) * *rayon / CU_REAL(*nnnr);

    if (*na == 1) {
        if (*niso != 1 || std::memcmp(trope, "iso", 3) != 0) {
            set_sphere_error(infostr, infostr_length, nstop,
                             "Anisotropic sphere MDDA requires a separate validation");
            return;
        }
        if (!fill_sphere_pml_geometry(
                eps, eps0, lnrbc, lcos, *ntypeabc, xs, ys, zs,
                k0, aretecube, tabdip, pmlwrk, *nnnr, *nmax,
                nbsphere, ndipole, *niso, *rayon, *xg, *yg, *zg,
                *nmat, epsilon, polarisa))
            set_sphere_error(infostr, infostr_length, nstop,
                             "CUDA sphere MDDA geometry/material initialization failed");
        return;
    }

    std::vector<int> inside(grid);
    const bool compact_mesh = *na == -1;
    int packed_geometry_count = 0;

    // The first CUDA API call can initialize the context. Account for this
    // startup cost separately when benchmarking the geometry kernels.
    if (std::getenv("IFDDA_SPHERE_BENCH")) {
        const auto init_start = std::chrono::steady_clock::now();
        const cudaError_t init_status = cudaFree(nullptr);
        const auto init_end = std::chrono::steady_clock::now();
        const double init_ms = std::chrono::duration<double, std::milli>(
            init_end - init_start).count();
        std::fprintf(stderr, "IFDDA_SPHERE_CUDA_INIT_MS %.6f\n", init_ms);
        if (!cuda_utils::check_cuda(init_status, "sphere benchmark", "initialize CUDA context")) {
            set_sphere_error(infostr, infostr_length, nstop,
                             "CUDA context initialization failed");
            return;
        }
    }

    const auto geometry_gpu_start = std::chrono::steady_clock::now();
    // calcul de inside_data sur le gpu
    int geometry_status;
    if (compact_mesh) {

        geometry_status = ifdda_sphere_geometry_compact_cuda(
                *nnnr, *rayon, *aretecube, *xg, *yg, *zg,
                xs, ys, zs, inside.data(),
                xs, ys, zs, &packed_geometry_count);

    }else {

        geometry_status = ifdda_sphere_geometry_cuda(
                *nnnr, *rayon, *aretecube, *xg, *yg, *zg,
                xs, ys, zs, inside.data());
    }
    const auto geometry_gpu_end = std::chrono::steady_clock::now();
    if (std::getenv("IFDDA_SPHERE_BENCH")) {
        const double geometry_ms = std::chrono::duration<double, std::milli>(
            geometry_gpu_end - geometry_gpu_start).count();
        std::fprintf(stderr,
                     "IFDDA_SPHERE_GPU_BEFORE_TABDIP_MS %.6f grid=%zu compact=%d\n",
                     geometry_ms, grid, int(compact_mesh));
    }
    if (geometry_status != 0) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "CUDA sphere geometry kernel failed");
        return;
    }

    size_t occupied = 0;
    size_t packed = 0;
    const int discretized = 1;
    const int invert_result = 1;
    const bool anisotropic = std::memcmp(trope, "ani", 3) == 0;
    CU_REAL local_sldr = *sldr;
    COMPLEX scalar_alpha{};
    COMPLEX tensor_alpha[9]{};

    if (!anisotropic && !method_uses_mutable_sldr(methode, methode_length))
        poladiff_cuda(aretecube, eps, eps0, k0, &discretized, methode,
                  &local_sldr, &scalar_alpha, methode_length);
    if (anisotropic)
        polaepstens_cuda(aretecube, epsani, eps0, k0, &discretized, methode,
                     &invert_result, tensor_alpha, methode_length);

    // The CPU builds only tabdip and the two counts. The GPU will populate
    // epsilon and polarisa using this map and the resident occupancy grid.
    // Every grid entry is assigned in the loop. Clear only any larger nmax tail.
    const auto tabdip_cpu_start = std::chrono::steady_clock::now();
    std::fill(tabdip + grid, tabdip + size_t(*nmax), 0);
    for (size_t cell = 0; cell < grid; ++cell) {
        if (compact_mesh) {
            if (!inside[cell]) {
                tabdip[cell] = 0;
                continue;
            }
            tabdip[cell] = int(++packed);
        } else {
            tabdip[cell] = int(cell + 1);
            ++occupied;
        }
    }

    *ndipole = int(grid);
    *nbsphere = int(compact_mesh ? packed : occupied);
    const auto tabdip_cpu_end = std::chrono::steady_clock::now();
    if (std::getenv("IFDDA_SPHERE_BENCH")) {
        const double tabdip_ms = std::chrono::duration<double, std::milli>(
            tabdip_cpu_end - tabdip_cpu_start).count();
        std::fprintf(stderr, "IFDDA_SPHERE_TABDIP_CPU_MS %.6f grid=%zu packed=%zu\n",
                     tabdip_ms, grid, packed);
    }
    if (compact_mesh && packed != size_t(packed_geometry_count)) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "GPU sphere compaction count does not match the material map");
        return;
    }

    // In the LR method the first occupied cell sets sldr to 0.2. Since a
    // homogeneous sphere shares one polarizability, the value is calculated
    // once and broadcast on the GPU (equivalent to the former per-cell calls).
    if (!anisotropic && method_uses_mutable_sldr(methode, methode_length) &&
        *nbsphere > 0)
        poladiff_cuda(aretecube, eps, eps0, k0, &discretized, methode,
                  &local_sldr, &scalar_alpha, methode_length);

    const auto materials_gpu_start = std::chrono::steady_clock::now();
    COMPLEX tensor_eps_host[9]{};
    for (int component = 0; component < 9; ++component) {
        tensor_eps_host[component] = epsani[component];
    }
    const COMPLEX scalar_eps_host = *eps;
    const bool materials_ok =
        ifdda_fill_homogeneous_geometry_materials_host_cuda(
            tabdip, grid, *nmax, *niso, int(compact_mesh), int(anisotropic),
            &scalar_eps_host, tensor_eps_host,
            scalar_alpha, tensor_alpha,
            epsilon, polarisa) == 0;
    const auto materials_gpu_end = std::chrono::steady_clock::now();
    if (std::getenv("IFDDA_SPHERE_BENCH")) {
        const double materials_ms = std::chrono::duration<double, std::milli>(
            materials_gpu_end - materials_gpu_start).count();
        std::fprintf(stderr,
                     "IFDDA_SPHERE_GPU_AFTER_TABDIP_MS %.6f grid=%zu compact=%d\n",
                     materials_ms, grid, int(compact_mesh));
    }
    if (!materials_ok) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "CUDA sphere material initialization failed");
        return;
    }
    const bool mesh_written = *nmat != 0 ||
        (compact_mesh
             ? write_compact_sphere_mesh_files(*nnnr, *nbsphere, *rayon,
                                               *aretecube, *xg, *yg, *zg,
                                               xs, ys, zs)
             : write_sphere_mesh_files(*nnnr, *nbsphere, xs, ys, zs,
                                       xs, ys, zs));
    if (!mesh_written) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "could not write sphere mesh files");
        return;
    }
    *sldr = local_sldr;
    if (std::getenv("IFDDA_SPHERE_BENCH")) {
        const auto sphere_benchmark_end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(sphere_benchmark_end - sphere_benchmark_start).count();
        std::fprintf(stderr, "IFDDA_SPHERE_TIME_MS %.6f\n", elapsed_ms);
    }
}
