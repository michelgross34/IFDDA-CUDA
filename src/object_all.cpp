#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

// Geometry IDs are aligned with the shared main0.cpp/main0.f90 test drivers.
enum class GeometryId : int {
    ConcentricSphere = 0,
    Sphere = 1,
    Cube = 2,
    Cuboid1 = 3,
    Cuboid2 = 4,
    InhomoSphere = 5,
    InhomoCuboid1 = 6,
    InhomoCuboid2 = 7,
    Ellipsoid = 8,
    NSpheres = 9,
    Cylinder = 10,
    RandomSphere1 = 11,
    RandomSphere2 = 12,
    Arbitrary = 13,
    RandomMulti = 14
};

void object_error(char* info, size_t info_length, int* stop, const char* message) {
    if (info) {
        std::memset(info, ' ', info_length);
        if (message)
            std::memcpy(info, message, std::min(info_length, std::strlen(message)));
    }
    if (stop)
        *stop = 1;
}

int object_polarizability_method_code(const char* method) {
    if (!method) return 0;
    const char c0 = method[0], c1 = method[1];
    if (c0 == 'C' && c1 == 'M') return 1;
    if (c0 == 'R' && c1 == 'R') return 2;
    if (c0 == 'P' && c1 == 'S') return 3;
    if (c0 == 'G' && c1 == 'B') return 4;
    if (c0 == 'L' && c1 == 'A') return 5;
    if (c0 == 'L' && c1 == 'R') return 6;
    if (c0 == 'L' && c1 == 'S') return 7;
    if (c0 == 'F' && c1 == 'G') return 8;
    return 0;
}

void object_set_info(char* info, const char* message) {
    if (!info) return;
    std::memset(info,' ',64);
    if (message) std::memcpy(info,message,std::min(size_t(64),std::strlen(message)));
}

std::string object_fortran_string(const char* value,size_t length) {
    size_t end=0;
    while (end<length && value[end]!='\0') ++end;
    while (end && value[end-1]==' ') --end;
    return std::string(value,end);
}

bool object_read_complex(std::istream& stream,COMPLEX* value) {
    COMPLEX parsed{};
    if (!value || !(stream >> parsed)) return false;
    *value=parsed;
    return true;
}

size_t object_array_offset(int nmax,int niso,int dipole,int row,int column) {
    return size_t(dipole)+size_t(nmax)*size_t(row+niso*column);
}

bool write_sphere_mesh(int n, int occupied, CU_REAL radius, CU_REAL spacing,
                       CU_REAL cx, CU_REAL cy, CU_REAL cz,
                       const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs) {
    std::ofstream fx("x.mat"), fy("y.mat"), fz("z.mat");
    std::ofstream fcx("xc.mat"), fcy("yc.mat"), fcz("zc.mat");
    if (!fx || !fy || !fz || !fcx || !fcy || !fcz)
        return false;
    for (int p = 0; p < n; ++p) {
        const CU_REAL offset = -radius + spacing * (CU_REAL(p) + CU_REAL(0.5));
        fx << offset + cx << '\n';
        fy << offset + cy << '\n';
        fz << offset + cz << '\n';
    }
    for (int p = 0; p < occupied; ++p) {
        fcx << xs[p] << '\n';
        fcy << ys[p] << '\n';
        fcz << zs[p] << '\n';
    }
    return bool(fx && fy && fz && fcx && fcy && fcz);
}

bool write_rectangular_mesh(int nx, int ny, int nz, int occupied,
                            const CU_REAL* mesh_x, const CU_REAL* mesh_y,
                            const CU_REAL* mesh_z, const CU_REAL* xs,
                            const CU_REAL* ys, const CU_REAL* zs) {
    const char* paths[6] = {"x.mat", "y.mat", "z.mat", "xc.mat", "yc.mat", "zc.mat"};
    const CU_REAL* values[6] = {mesh_x, mesh_y, mesh_z, xs, ys, zs};
    const int counts[6] = {nx, ny, nz, occupied, occupied, occupied};
    const int strides[6] = {1, nx, nx * ny, 1, 1, 1};
    for (int a = 0; a < 6; ++a) {
        FILE* f = std::fopen(paths[a], "w");
        if (!f) return false;
        for (int p = 0; p < counts[a]; ++p)
            std::fprintf(f, " %.17g\n", double(values[a][size_t(p) * size_t(strides[a])]));
        if (std::fclose(f) != 0) return false;
    }
    return true;
}

// The four main device arrays are owned by cuda_data's explicit persistent
// pointers. Other temporary kernel arrays use the named scratch workspace.
// The public Fortran-compatible outputs still have the original nmax stride;
// on GPU each material tensor component uses the compact nbsphere stride.
bool fill_homogeneous_materials_impl(const int* host_tabdip, size_t grid, int nmax,
                           int nbsphere, int niso, bool compact, bool anisotropic,
                           const COMPLEX* eps, const COMPLEX* epsani,
                           COMPLEX scalar_alpha, const COMPLEX tensor_alpha[9],
                           COMPLEX* host_epsilon, COMPLEX* host_polarisa) {
    if (nbsphere <= 0)
        return false;
    const size_t components = size_t(niso) * size_t(niso);
    const size_t packed_count = size_t(nbsphere) * components;
    CUCOMPLEX* device_polarisa = cuda_data::get_polarisa(packed_count);
    CUCOMPLEX* device_epsilon = cuda_data::get_epsilon(packed_count);
    const CU_INT* device_map = cuda_data::get_dipole_map(grid);
    const CU_INT* device_tabdip = cuda_data::get_tapdip(grid);
    if (!device_polarisa || !device_epsilon || !device_map || !device_tabdip)
        return false;
    if (!cuda_utils::check_cuda(
            initialize_polarizability_arrays_wrapper(
                nullptr, device_polarisa, device_epsilon, nbsphere, niso),
            "object_all", "initialize compact material tensors"))
        return false;

    const CUCOMPLEX* d_alpha = nullptr;
    const CUCOMPLEX* d_eps = nullptr;
    if (anisotropic) {
        cuda_data::TensorPolarizabilityDeviceBuffers tensors{};
        if (!cuda_data::get_device_tensor_polarizability_buffers(&tensors) ||
            !cuda_utils::check_cuda(
                cuda_data::copy_host_complex_to_device(
                    tensors.value, tensor_alpha, 9,
                    cuda_utils::TransferCategory::Material),
                "object_all", "upload anisotropic polarizability") ||
            !cuda_utils::check_cuda(
                cuda_data::copy_host_complex_to_device(
                    tensors.epsilon, epsani, 9,
                    cuda_utils::TransferCategory::Material),
                "object_all", "upload anisotropic permittivity"))
            return false;
        d_alpha = tensors.value;
        d_eps = tensors.epsilon;
    }

    if (!cuda_utils::check_cuda(
            sphere_material_from_cpu_tabdip_wrapper(
                device_tabdip, device_map, device_polarisa, device_epsilon,
                d_alpha, d_eps, cuda_utils::device_value(scalar_alpha),
                cuda_utils::device_value(*eps),
                int(grid), nbsphere, int(compact), int(anisotropic)),
            "object_all", "fill compact material tensors"))
        return false;

    std::vector<COMPLEX> compact_alpha(packed_count);
    std::vector<COMPLEX> compact_eps(packed_count);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(
                compact_alpha.data(), device_polarisa, packed_count,
                cuda_utils::TransferCategory::Material),
            "object_all", "publish host polarizability") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(
                compact_eps.data(), device_epsilon, packed_count,
                cuda_utils::TransferCategory::Material),
            "object_all", "publish host permittivity"))
        return false;

    for (size_t component = 0; component < components; ++component) {
        const size_t host_offset = component * size_t(nmax);
        const size_t packed_offset = component * size_t(nbsphere);
        const COMPLEX identity = component % 4 == 0 ? COMPLEX(1, 0) : COMPLEX(0, 0);
        std::fill_n(host_polarisa + host_offset, size_t(nmax), COMPLEX(0, 0));
        std::fill_n(host_epsilon + host_offset, size_t(nmax), identity);
        std::copy_n(compact_alpha.data() + packed_offset, size_t(nbsphere),
                    host_polarisa + host_offset);
        std::copy_n(compact_eps.data() + packed_offset, size_t(nbsphere),
                    host_epsilon + host_offset);
    }
    return true;
}

// Material families are indexed by the CPU-constructed compact dipole index.
// Keep the same four persistent GPU arrays and a compact tensor stride.
bool fill_labeled_materials_impl(const int* host_tabdip, const int* host_labels,
                            size_t grid, int nmax, int nbsphere, int niso,
                            int label_count, bool vacuum_background,
                            const COMPLEX* label_epsilon,
                            const COMPLEX* label_alpha,
                            COMPLEX* host_epsilon, COMPLEX* host_polarisa) {
    if (!host_tabdip || !host_labels || !label_epsilon || !label_alpha ||
        !host_epsilon || !host_polarisa || nmax < nbsphere || nbsphere <= 0 ||
        label_count <= 0 || (niso != 1 && niso != 3) ||
        grid > size_t(std::numeric_limits<int>::max())) return false;
    const size_t components = size_t(niso) * size_t(niso);
    const size_t packed_count = size_t(nbsphere) * components;
    const size_t lookup_count = size_t(label_count) * components;
    CUCOMPLEX* device_polarisa = cuda_data::get_polarisa(packed_count);
    CUCOMPLEX* device_epsilon = cuda_data::get_epsilon(packed_count);
    const CU_INT* device_tabdip = cuda_data::get_tapdip(grid);
    CU_INT* device_labels = static_cast<CU_INT*>(cuda_data::get_device_workspace_buffer_bytes(
        "object_all.material_labels", size_t(nbsphere)*sizeof(CU_INT)));
    CUCOMPLEX* device_eps_lookup = static_cast<CUCOMPLEX*>(
        cuda_data::get_device_workspace_buffer_bytes(
            "object_all.eps_lookup", lookup_count*sizeof(CUCOMPLEX)));
    CUCOMPLEX* device_alpha_lookup = static_cast<CUCOMPLEX*>(
        cuda_data::get_device_workspace_buffer_bytes(
            "object_all.alpha_lookup", lookup_count*sizeof(CUCOMPLEX)));
    if (!device_polarisa || !device_epsilon || !device_tabdip ||
        !device_labels || !device_eps_lookup || !device_alpha_lookup) return false;
    std::vector<CU_INT> labels(static_cast<size_t>(nbsphere));
    for (int i = 0; i < nbsphere; ++i) labels[size_t(i)].dat = host_labels[i];
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(device_labels, labels.data(),
                labels.size()*sizeof(CU_INT), cuda_utils::TransferCategory::Material),
            "object_all", "upload compact material labels") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_complex_to_device(device_eps_lookup,
                label_epsilon, lookup_count, cuda_utils::TransferCategory::Material),
            "object_all", "upload labeled permittivity") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_complex_to_device(device_alpha_lookup,
                label_alpha, lookup_count, cuda_utils::TransferCategory::Material),
            "object_all", "upload labeled polarizability") ||
        !cuda_utils::check_cuda(
            initialize_polarizability_arrays_wrapper(
                nullptr, device_polarisa, device_epsilon, nbsphere, niso),
            "object_all", "initialize compact labeled tensors")) return false;
    if (!vacuum_background && !cuda_utils::check_cuda(
            cudaMemset(device_epsilon, 0, packed_count*sizeof(CUCOMPLEX)),
            "object_all", "zero labeled vacuum background")) return false;
    if (!cuda_utils::check_cuda(
            labeled_material_from_cpu_tabdip_wrapper(
                device_tabdip, device_labels, device_polarisa, device_epsilon,
                device_alpha_lookup, device_eps_lookup, int(grid), nbsphere,
                label_count, niso == 3),
            "object_all", "fill compact labeled tensors")) return false;
    std::vector<COMPLEX> packed_alpha(packed_count), packed_eps(packed_count);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(packed_alpha.data(),
                device_polarisa, packed_count, cuda_utils::TransferCategory::Material),
            "object_all", "download labeled polarizability") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(packed_eps.data(),
                device_epsilon, packed_count, cuda_utils::TransferCategory::Material),
            "object_all", "download labeled permittivity")) return false;
    for (size_t component = 0; component < components; ++component) {
        const size_t offset = component * size_t(nmax);
        const size_t packed_offset = component * size_t(nbsphere);
        const COMPLEX background = vacuum_background && component%4 == 0
            ? COMPLEX(1,0) : COMPLEX(0,0);
        std::fill_n(host_polarisa+offset, size_t(nmax), COMPLEX(0,0));
        std::fill_n(host_epsilon+offset, size_t(nmax), background);
        std::copy_n(packed_alpha.data()+packed_offset, size_t(nbsphere),
                    host_polarisa+offset);
        std::copy_n(packed_eps.data()+packed_offset, size_t(nbsphere),
                    host_epsilon+offset);
    }
    return true;
}

// Keep inhomogeneous material results in the four persistent CUDA buffers.
// Existing CPU output arrays retain their Fortran nmax stride, while the GPU
// values are packed with nbsphere as the tensor stride.
bool retain_inhomogeneous_materials_impl(const int* host_tabdip, size_t grid,
                                    int nmax, int nbsphere, int niso,
                                    const CU_INT* device_inside,
                                    const CUCOMPLEX* device_cell_epsilon,
                                    const CUCOMPLEX* device_cell_alpha,
                                    COMPLEX* host_epsilon,
                                    COMPLEX* host_polarisa) {
    if (!host_tabdip || !host_epsilon || !host_polarisa ||
        nbsphere <= 0 || nmax < nbsphere || (niso != 1 && niso != 3))
        return false;
    const size_t components = size_t(niso)*size_t(niso);
    const size_t packed_count = size_t(nbsphere)*components;
    CU_INT* d_tabdip = cuda_data::get_tapdip(grid);
    CUCOMPLEX* d_alpha = cuda_data::get_polarisa(packed_count);
    CUCOMPLEX* d_eps = cuda_data::get_epsilon(packed_count);
    if (!d_tabdip || !d_alpha || !d_eps) return false;
    if (niso == 1 && device_inside && device_cell_epsilon && device_cell_alpha) {
        if (!cuda_utils::check_cuda(
                cudaMemset(d_alpha, 0, packed_count*sizeof(CUCOMPLEX)),
                "object_all", "zero compact alpha") ||
            !cuda_utils::check_cuda(
                cudaMemset(d_eps, 0, packed_count*sizeof(CUCOMPLEX)),
                "object_all", "zero compact epsilon") ||
            !cuda_utils::check_cuda(
                scatter_cell_material_from_cpu_tabdip_wrapper(
                    d_tabdip, device_inside, device_cell_epsilon,
                    device_cell_alpha, d_eps, d_alpha, int(grid), nbsphere),
                "object_all", "retain compact inhomogeneous materials") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                host_epsilon, d_eps, size_t(nbsphere),
                cuda_utils::TransferCategory::Material),
                "object_all", "download compact inhomogeneous epsilon") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                host_polarisa, d_alpha, size_t(nbsphere),
                cuda_utils::TransferCategory::Material),
                "object_all", "download compact inhomogeneous alpha"))
            return false;
        // The previous GPU scatter cleared its nmax-sized output. Retain that
        // exact CPU representation without allocating a second GPU output.
        std::fill_n(host_epsilon + nbsphere, size_t(nmax-nbsphere), COMPLEX(0,0));
        std::fill_n(host_polarisa + nbsphere, size_t(nmax-nbsphere), COMPLEX(0,0));
        return true;
    }
    std::vector<COMPLEX> packed_alpha(packed_count), packed_eps(packed_count);
    for (size_t component = 0; component < components; ++component) {
        std::copy_n(host_polarisa + component*size_t(nmax), size_t(nbsphere),
                    packed_alpha.data()+component*size_t(nbsphere));
        std::copy_n(host_epsilon + component*size_t(nmax), size_t(nbsphere),
                    packed_eps.data()+component*size_t(nbsphere));
    }
    return cuda_utils::check_cuda(cuda_data::copy_host_complex_to_device(
               d_alpha,packed_alpha.data(),packed_count,
               cuda_utils::TransferCategory::Material),
               "object_all","retain compact CPU polarizability") &&
           cuda_utils::check_cuda(cuda_data::copy_host_complex_to_device(
               d_eps,packed_eps.data(),packed_count,
               cuda_utils::TransferCategory::Material),
               "object_all","retain compact CPU permittivity");
}

} // namespace

// One entry for both stages of geometry assembly. A geometry-specific CUDA
// wrapper produces the map; the CPU stage may follow later when shape-specific
// lattice coordinates and labels are available. All shapes use these stages.
struct ObjectAllRegularLattice {
    int nx = 0, ny = 0, nz = 0;
    CU_REAL cell_size = CU_REAL(0);
    CU_REAL origin_x = CU_REAL(0), origin_y = CU_REAL(0), origin_z = CU_REAL(0);
    CU_REAL shift_x = CU_REAL(0), shift_y = CU_REAL(0), shift_z = CU_REAL(0);
    bool one_based = false;
};

bool write_rectangular_mesh_regular(const ObjectAllRegularLattice& l, int occupied,
                                    const CU_REAL* xs, const CU_REAL* ys,
                                    const CU_REAL* zs) {
    const char* paths[6] = {"x.mat", "y.mat", "z.mat", "xc.mat", "yc.mat", "zc.mat"};
    const int dimensions[3] = {l.nx, l.ny, l.nz};
    const CU_REAL origins[3] = {l.origin_x, l.origin_y, l.origin_z};
    const CU_REAL shifts[3] = {l.shift_x, l.shift_y, l.shift_z};
    const CU_REAL* compact[3] = {xs, ys, zs};
    for (int axis = 0; axis < 3; ++axis) {
        FILE* file = std::fopen(paths[axis], "w");
        if (!file) return false;
        for (int i = 0; i < dimensions[axis]; ++i) {
            const CU_REAL coordinate = std::fma(
                CU_REAL(i) + (l.one_based ? CU_REAL(1) : CU_REAL(0.5)),
                l.cell_size, origins[axis]) + shifts[axis];
            std::fprintf(file, " %.17g\n", double(coordinate));
        }
        if (std::fclose(file) != 0) return false;
        file = std::fopen(paths[axis + 3], "w");
        if (!file) return false;
        for (int i = 0; i < occupied; ++i)
            std::fprintf(file, " %.17g\n", double(compact[axis][i]));
        if (std::fclose(file) != 0) return false;
    }
    return true;
}

struct ObjectAllCpuStage {
    int nmax;
    bool compact;
    const CU_REAL* lattice_x;
    const CU_REAL* lattice_y;
    const CU_REAL* lattice_z;
    int* tabdip;
    int* labels;
    CU_REAL* xs;
    CU_REAL* ys;
    CU_REAL* zs;
    int* nbsphere;
    int* ndipole;
    int max_label;
    bool require_filled_grid;
    CU_REAL shift_x;
    CU_REAL shift_y;
    CU_REAL shift_z;
    const ObjectAllRegularLattice* regular_lattice = nullptr;
};

enum class ObjectAllMaterialKind {
    Homogeneous, Labeled, Inhomogeneous, PackedHost, PackedDevice, ArbitraryTensor
};

// A single explicit parameter block keeps the shared GPU material stage in
// ifdda_object_all_cuda. Only one of the three material families is used.
struct ObjectAllMaterialStage {
    ObjectAllMaterialKind kind;
    const int* host_tabdip = nullptr;
    const int* host_labels = nullptr;
    int nmax = 0;
    int nbsphere = 0;
    int niso = 0;
    int label_count = 0;
    bool compact = false;
    bool anisotropic = false;
    bool vacuum_background = false;
    const COMPLEX* eps = nullptr;
    const COMPLEX* epsani = nullptr;
    COMPLEX scalar_alpha{};
    const COMPLEX* tensor_alpha = nullptr;
    const COMPLEX* label_epsilon = nullptr;
    const COMPLEX* label_alpha = nullptr;
    const CU_INT* device_inside = nullptr;
    const CUCOMPLEX* device_cell_epsilon = nullptr;
    const CUCOMPLEX* device_cell_alpha = nullptr;
    COMPLEX* host_epsilon = nullptr;
    COMPLEX* host_polarisa = nullptr;
    const COMPLEX* source_epsilon = nullptr;
    const COMPLEX* source_polarisa = nullptr;
    COMPLEX background_epsilon{};
    CU_REAL cell_size = CU_REAL(0);
    CU_REAL wavenumber = CU_REAL(0);
    int method_code = 0;
};

bool ifdda_object_all_cuda(int geometry, size_t grid, cudaError_t launch_status,
                           std::vector<int>& cpu_map,
                           const CU_INT* existing_device_map = nullptr,
                           bool full_grid_cpu = false,
                           const ObjectAllCpuStage* cpu_stage = nullptr,
                           bool cpu_map_ready = false,
                           const ObjectAllMaterialStage* material_stage = nullptr) {
    if (geometry < int(GeometryId::ConcentricSphere) ||
        geometry > int(GeometryId::RandomMulti) || !grid ||
        grid > size_t(std::numeric_limits<int>::max()) ||
        !cuda_utils::check_cuda(launch_status, "object_all", "geometry map kernel"))
        return false;
    if (!cpu_map_ready) {
        CU_INT* device_map = cuda_data::get_dipole_map(grid);
        if (!device_map) return false;
        // PML material kernels own an auxiliary occupancy map.
        if (existing_device_map && existing_device_map != device_map &&
            !cuda_utils::check_cuda(
                cudaMemcpy(device_map, existing_device_map, grid*sizeof(CU_INT),
                           cudaMemcpyDeviceToDevice),
                "object_all", "publish PML geometry map"))
            return false;
        if (full_grid_cpu) {
            // PML includes its absorbing shell; no extra device download.
            cpu_map.assign(grid, 1);
        } else {
            static_assert(sizeof(CU_INT) == sizeof(int),
                          "CU_INT and int layouts differ");
            static_assert(offsetof(CU_INT, dat) == 0,
                          "CU_INT must hold int at offset zero");
            cpu_map.resize(grid);
            if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                    cpu_map.data(), device_map, grid*sizeof(CU_INT),
                    cuda_utils::TransferCategory::Geometry),
                    "object_all", "download geometry map"))
                return false;
        }
    }
    if (cpu_stage) {
        const ObjectAllCpuStage& s = *cpu_stage;
        const bool implicit = s.regular_lattice != nullptr;
        if (s.nmax <= 0 || cpu_map.size() != grid || grid > size_t(s.nmax) ||
            (!implicit && (!s.lattice_x || !s.lattice_y || !s.lattice_z)) ||
            !s.tabdip ||
            !s.xs || !s.ys || !s.zs || !s.nbsphere || !s.ndipole ||
            s.max_label < 1)
            return false;
        if (implicit) {
            const auto& l = *s.regular_lattice;
            if (l.nx <= 0 || l.ny <= 0 || l.nz <= 0 ||
                l.nx > INT_MAX / l.ny ||
                l.nx * l.ny > INT_MAX / l.nz ||
                size_t(l.nx) * size_t(l.ny) * size_t(l.nz) != grid ||
                l.cell_size <= CU_REAL(0))
                return false;
        }
        std::fill_n(s.tabdip, size_t(s.nmax), 0);
        if (s.labels) std::fill_n(s.labels, size_t(s.nmax), 0);
        int selected = 0;
        for (size_t flat = 0; flat < grid; ++flat) {
            const int material = cpu_map[flat];
            if (material < 0 || material > s.max_label ||
                (s.require_filled_grid && material == 0)) return false;
            if (s.compact && material == 0) continue;
            s.tabdip[flat] = ++selected;
            const size_t destination = size_t(selected - 1);
            if (implicit) {
                const auto& l = *s.regular_lattice;
                const size_t row = size_t(l.nx), plane = row * size_t(l.ny);
                const CU_REAL off = l.one_based ? CU_REAL(1) : CU_REAL(0.5);
                s.xs[destination] = std::fma(CU_REAL(flat % row) + off,
                                            l.cell_size, l.origin_x) + l.shift_x;
                s.ys[destination] = std::fma(CU_REAL((flat / row) % size_t(l.ny)) + off,
                                            l.cell_size, l.origin_y) + l.shift_y;
                s.zs[destination] = std::fma(CU_REAL(flat / plane) + off,
                                            l.cell_size, l.origin_z) + l.shift_z;
            } else {
                s.xs[destination] = s.lattice_x[flat] + s.shift_x;
                s.ys[destination] = s.lattice_y[flat] + s.shift_y;
                s.zs[destination] = s.lattice_z[flat] + s.shift_z;
            }
            if (s.labels && material) s.labels[destination] = material;
        }
        *s.nbsphere = selected;
        *s.ndipole = int(grid);
        // Reserve the whole Fortran/FFT extent before the first use.
        if (!get_tapdip(s.tabdip, size_t(s.nmax))) return false;
        if (implicit && selected > 0) {
            const auto& l = *s.regular_lattice;
            if (!cuda_data::register_host_regular_geometry({
                    s.xs, s.ys, s.zs, s.tabdip, l.nx, l.ny, l.nz,
                    selected, l.cell_size}))
                return false;
        }
    }
    if (!material_stage) return true;
    const ObjectAllMaterialStage& m = *material_stage;
    switch (m.kind) {
    case ObjectAllMaterialKind::Homogeneous:
        return fill_homogeneous_materials_impl(
            m.host_tabdip, grid, m.nmax, m.nbsphere, m.niso, m.compact,
            m.anisotropic, m.eps, m.epsani, m.scalar_alpha,
            m.tensor_alpha, m.host_epsilon, m.host_polarisa);
    case ObjectAllMaterialKind::Labeled:
        return fill_labeled_materials_impl(
            m.host_tabdip, m.host_labels, grid, m.nmax, m.nbsphere,
            m.niso, m.label_count, m.vacuum_background,
            m.label_epsilon, m.label_alpha,
            m.host_epsilon, m.host_polarisa);
    case ObjectAllMaterialKind::Inhomogeneous:
        return retain_inhomogeneous_materials_impl(
            m.host_tabdip, grid, m.nmax, m.nbsphere, m.niso,
            m.device_inside, m.device_cell_epsilon, m.device_cell_alpha,
            m.host_epsilon, m.host_polarisa);
    case ObjectAllMaterialKind::ArbitraryTensor: {
        // The file supplies one complete 3x3 tensor per voxel. The CUDA
        // material kernel writes directly to the compact persistent arrays.
        if (m.niso != 3 || m.nbsphere != int(grid) ||
            m.nmax < m.nbsphere || !m.source_epsilon ||
            !m.host_epsilon || !m.host_polarisa ||
            grid > std::numeric_limits<size_t>::max() / 9 / sizeof(CUCOMPLEX))
            return false;
        const size_t packed_count = grid * 9;
        CUCOMPLEX* d_alpha = cuda_data::get_polarisa(packed_count);
        CUCOMPLEX* d_eps = cuda_data::get_epsilon(packed_count);
        CUCOMPLEX* device_cells = static_cast<CUCOMPLEX*>(
            cuda_data::get_device_workspace_buffer_bytes(
                "object_all.arbitrary.input_tensor", packed_count*sizeof(CUCOMPLEX)));
        if (!d_alpha || !d_eps || !device_cells ||
            !cuda_utils::check_cuda(cuda_data::copy_host_complex_to_device(
                device_cells, m.source_epsilon, packed_count,
                cuda_utils::TransferCategory::Material),
                "object_all", "upload arbitrary tensor epsilon") ||
            !cuda_utils::check_cuda(initialize_polarizability_arrays_wrapper(
                nullptr, d_alpha, d_eps, m.nbsphere, m.niso),
                "object_all", "initialize compact arbitrary tensors") ||
            !cuda_utils::check_cuda(arbitrary_tensor_material_batch_wrapper(
                device_cells, d_eps, d_alpha, m.nbsphere, m.nbsphere,
                cuda_utils::device_value(m.background_epsilon),
                m.cell_size, m.wavenumber, m.method_code),
                "object_all", "compute compact arbitrary tensor alpha"))
            return false;
        std::vector<COMPLEX> packed_alpha(packed_count), packed_eps(packed_count);
        if (!cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                packed_alpha.data(), d_alpha, packed_count,
                cuda_utils::TransferCategory::Material),
                "object_all", "publish arbitrary tensor alpha") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                packed_eps.data(), d_eps, packed_count,
                cuda_utils::TransferCategory::Material),
                "object_all", "publish arbitrary tensor epsilon"))
            return false;
        for (int component = 0; component < 9; ++component) {
            const size_t source = size_t(component)*grid;
            const size_t target = size_t(component)*size_t(m.nmax);
            std::copy_n(packed_alpha.data()+source, grid, m.host_polarisa+target);
            std::copy_n(packed_eps.data()+source, grid, m.host_epsilon+target);
        }
        return true;
    }
    case ObjectAllMaterialKind::PackedHost:
    case ObjectAllMaterialKind::PackedDevice: {
        if (m.nmax < m.nbsphere || m.nbsphere <= 0 ||
            (m.niso != 1 && m.niso != 3) ||
            !cuda_data::get_tapdip(grid)) return false;
        const size_t components = size_t(m.niso)*size_t(m.niso);
        const size_t packed_count = size_t(m.nbsphere)*components;
        CUCOMPLEX* d_alpha = cuda_data::get_polarisa(packed_count);
        CUCOMPLEX* d_eps = cuda_data::get_epsilon(packed_count);
        if (!d_alpha || !d_eps) return false;
        if (m.kind == ObjectAllMaterialKind::PackedDevice) {
            // The inhomogeneous cuboid GPU batch currently has scalar values.
            if (m.niso != 1 || !m.device_cell_epsilon || !m.device_cell_alpha ||
                !m.host_epsilon || !m.host_polarisa)
                return false;
            if (!cuda_utils::check_cuda(
                       cudaMemcpy(d_alpha, m.device_cell_alpha,
                           packed_count*sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice),
                       "object_all", "publish resident voxel polarizability") ||
                !cuda_utils::check_cuda(
                       cudaMemcpy(d_eps, m.device_cell_epsilon,
                           packed_count*sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice),
                       "object_all", "publish resident voxel permittivity") ||
                !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                       m.host_polarisa, d_alpha, packed_count,
                       cuda_utils::TransferCategory::Material),
                       "object_all", "download resident voxel polarizability") ||
                !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                       m.host_epsilon, d_eps, packed_count,
                       cuda_utils::TransferCategory::Material),
                       "object_all", "download resident voxel permittivity"))
                return false;
            std::fill_n(m.host_polarisa + m.nbsphere,
                        size_t(m.nmax-m.nbsphere), COMPLEX(0,0));
            std::fill_n(m.host_epsilon + m.nbsphere,
                        size_t(m.nmax-m.nbsphere), COMPLEX(0,0));
            return true;
        }
        if (!m.source_polarisa || !m.source_epsilon) return false;
        std::vector<COMPLEX> packed_alpha(packed_count), packed_eps(packed_count);
        for (size_t component=0; component<components; ++component) {
            std::copy_n(m.source_polarisa+component*size_t(m.nmax),
                        size_t(m.nbsphere),
                        packed_alpha.data()+component*size_t(m.nbsphere));
            std::copy_n(m.source_epsilon+component*size_t(m.nmax),
                        size_t(m.nbsphere),
                        packed_eps.data()+component*size_t(m.nbsphere));
        }
        return cuda_utils::check_cuda(cuda_data::copy_host_complex_to_device(
                   d_alpha, packed_alpha.data(), packed_count,
                   cuda_utils::TransferCategory::Material),
                   "object_all", "publish CPU voxel polarizability") &&
               cuda_utils::check_cuda(cuda_data::copy_host_complex_to_device(
                   d_eps, packed_eps.data(), packed_count,
                   cuda_utils::TransferCategory::Material),
                   "object_all", "publish CPU voxel permittivity");
    }
    }
    return false;
}

// Fortran's tabdip is 1-based on the full grid; active coordinates and
// material labels are compacted in exactly the same order for every shape.
// The caller supplies its CPU lattice coordinates, whose rounding formula
// may depend on its historical Fortran geometry implementation.
bool object_all_build_cpu_dipoles(
    int geometry, std::vector<int>& map, int nmax, bool compact,
    const CU_REAL* lattice_x, const CU_REAL* lattice_y,
    const CU_REAL* lattice_z, int* tabdip, int* labels,
    CU_REAL* xs, CU_REAL* ys, CU_REAL* zs,
    int* nbsphere, int* ndipole, int max_label = 1,
    bool require_filled_grid = false,
    CU_REAL shift_x = CU_REAL(0), CU_REAL shift_y = CU_REAL(0),
    CU_REAL shift_z = CU_REAL(0)) {
    const ObjectAllCpuStage cpu_stage{
        nmax, compact, lattice_x, lattice_y, lattice_z,
        tabdip, labels, xs, ys, zs, nbsphere, ndipole,
        max_label, require_filled_grid, shift_x, shift_y, shift_z
    };
    return ifdda_object_all_cuda(geometry, map.size(), cudaSuccess,
                                  map, nullptr, false, &cpu_stage, true);
}

// Regular lattices need no three full-grid host arrays. Reconstruct only the
// compact Fortran ABI coordinates after CPU tabdip construction, preserving
// the per-shape FMA/centering formula and the original voxel ordering.
bool object_all_build_regular_cpu_dipoles(
    int geometry, std::vector<int>& map, int nmax, bool compact,
    const ObjectAllRegularLattice& lattice, int* tabdip,
    CU_REAL* xs, CU_REAL* ys, CU_REAL* zs,
    int* nbsphere, int* ndipole, bool require_filled_grid = false) {
    ObjectAllCpuStage cpu_stage{
        nmax, compact, nullptr, nullptr, nullptr, tabdip, nullptr,
        xs, ys, zs, nbsphere, ndipole, 1, require_filled_grid,
        CU_REAL(0), CU_REAL(0), CU_REAL(0), &lattice};
    return ifdda_object_all_cuda(geometry, map.size(), cudaSuccess,
                                 map, nullptr, false, &cpu_stage, true);
}

// The geometry-specific CPU code prepares the small material parameters;
// all material families enter the same CUDA object pipeline for publication.
bool fill_homogeneous_materials(
    int geometry, const int* host_tabdip, size_t grid, int nmax,
    int nbsphere, int niso, bool compact, bool anisotropic,
    const COMPLEX* eps, const COMPLEX* epsani,
    COMPLEX scalar_alpha, const COMPLEX tensor_alpha[9],
    COMPLEX* host_epsilon, COMPLEX* host_polarisa) {
    ObjectAllMaterialStage m{ObjectAllMaterialKind::Homogeneous};
    m.host_tabdip = host_tabdip;
    m.nmax = nmax;
    m.nbsphere = nbsphere;
    m.niso = niso;
    m.compact = compact;
    m.anisotropic = anisotropic;
    m.eps = eps;
    m.epsani = epsani;
    m.scalar_alpha = scalar_alpha;
    m.tensor_alpha = tensor_alpha;
    m.host_epsilon = host_epsilon;
    m.host_polarisa = host_polarisa;
    std::vector<int> unused_map;
    return ifdda_object_all_cuda(geometry, grid, cudaSuccess, unused_map,
                                  nullptr, false, nullptr, true, &m);
}

bool fill_labeled_materials(
    int geometry, const int* host_tabdip, const int* host_labels,
    size_t grid, int nmax, int nbsphere, int niso,
    int label_count, bool vacuum_background,
    const COMPLEX* label_epsilon, const COMPLEX* label_alpha,
    COMPLEX* host_epsilon, COMPLEX* host_polarisa) {
    ObjectAllMaterialStage m{ObjectAllMaterialKind::Labeled};
    m.host_tabdip = host_tabdip;
    m.host_labels = host_labels;
    m.nmax = nmax;
    m.nbsphere = nbsphere;
    m.niso = niso;
    m.label_count = label_count;
    m.vacuum_background = vacuum_background;
    m.label_epsilon = label_epsilon;
    m.label_alpha = label_alpha;
    m.host_epsilon = host_epsilon;
    m.host_polarisa = host_polarisa;
    std::vector<int> unused_map;
    return ifdda_object_all_cuda(geometry, grid, cudaSuccess, unused_map,
                                  nullptr, false, nullptr, true, &m);
}

bool retain_inhomogeneous_materials(
    int geometry, const int* host_tabdip, size_t grid, int nmax,
    int nbsphere, int niso, const CU_INT* device_inside,
    const CUCOMPLEX* device_cell_epsilon,
    const CUCOMPLEX* device_cell_alpha,
    COMPLEX* host_epsilon, COMPLEX* host_polarisa) {
    ObjectAllMaterialStage m{ObjectAllMaterialKind::Inhomogeneous};
    m.host_tabdip = host_tabdip;
    m.nmax = nmax;
    m.nbsphere = nbsphere;
    m.niso = niso;
    m.device_inside = device_inside;
    m.device_cell_epsilon = device_cell_epsilon;
    m.device_cell_alpha = device_cell_alpha;
    m.host_epsilon = host_epsilon;
    m.host_polarisa = host_polarisa;
    std::vector<int> unused_map;
    return ifdda_object_all_cuda(geometry, grid, cudaSuccess, unused_map,
                                  nullptr, false, nullptr, true, &m);
}

bool publish_compact_materials(
    int geometry, size_t grid, int nmax, int nbsphere, int niso,
    COMPLEX* host_polarisa, COMPLEX* host_epsilon,
    const CUCOMPLEX* device_polarisa = nullptr,
    const CUCOMPLEX* device_epsilon = nullptr) {
    ObjectAllMaterialStage m{
        device_polarisa ? ObjectAllMaterialKind::PackedDevice
                        : ObjectAllMaterialKind::PackedHost
    };
    m.nmax = nmax;
    m.nbsphere = nbsphere;
    m.niso = niso;
    m.source_polarisa = host_polarisa;
    m.source_epsilon = host_epsilon;
    m.host_polarisa = host_polarisa;
    m.host_epsilon = host_epsilon;
    m.device_cell_alpha = device_polarisa;
    m.device_cell_epsilon = device_epsilon;
    std::vector<int> unused_map;
    return ifdda_object_all_cuda(geometry, grid, cudaSuccess, unused_map,
                                  nullptr, false, nullptr, true, &m);
}

bool compute_arbitrary_tensor_materials(
    int geometry, size_t grid, int nmax, const COMPLEX* cell_epsilon,
    COMPLEX background_epsilon, CU_REAL cell_size, CU_REAL wavenumber,
    int method_code, COMPLEX* host_epsilon, COMPLEX* host_polarisa) {
    ObjectAllMaterialStage m{ObjectAllMaterialKind::ArbitraryTensor};
    m.nmax = nmax;
    m.nbsphere = int(grid);
    m.niso = 3;
    m.source_epsilon = cell_epsilon;
    m.background_epsilon = background_epsilon;
    m.cell_size = cell_size;
    m.wavenumber = wavenumber;
    m.method_code = method_code;
    m.host_epsilon = host_epsilon;
    m.host_polarisa = host_polarisa;
    std::vector<int> unused_map;
    return ifdda_object_all_cuda(geometry, grid, cudaSuccess, unused_map,
                                  nullptr, false, nullptr, true, &m);
}

extern "C" void ifdda_object_all_sphere_cuda(
    int geometry, const char* trope, const COMPLEX* eps, const COMPLEX* epsani,
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
    if (geometry != static_cast<int>(GeometryId::Sphere)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: geometry not implemented");
        return;
    }
    if (!trope || !eps || !epsani || !eps0 || !xs || !ys || !zs || !k0 ||
        !aretecube || !tabdip || !nnnr || !nmax || !nbsphere || !ndipole ||
        !nx || !ny || !nz || !methode || !na || !epsilon || !polarisa ||
        !niso || !rayon || !xg || !yg || !zg || !sldr || !nmat ||
        !infostr || !nstop) {
        object_error(infostr, infostr_length, nstop, "object_all: invalid arguments");
        return;
    }

    // Keep the established absorbing PML/MDDA calculation until its dedicated
    // material profile is migrated. It is still selected only for the sphere.
    if (*na == 1 || (methode_length >= 2 && std::memcmp(methode, "PS", 2) == 0)) {
        objetsphere_legacy_cuda_(
            trope, eps, epsani, eps0, lnrbc, lcos, ntypeabc, xs, ys, zs, k0,
            aretecube, tabdip, tabmulti, pmlwrk, nnnr, nmax, nbsphere,
            ndipole, nx, ny, nz, methode, na, epsilon, polarisa, niso,
            rayon, xg, yg, zg, sldr, nmat, file_id, group_iddip, infostr,
            nstop, trope_length, methode_length, infostr_length);
        return;
    }

    *nbsphere = 0;
    *ndipole = 0;
    object_error(infostr, infostr_length, nstop, "");
    *nstop = 0;
    const bool anisotropic = trope_length >= 3 && std::memcmp(trope, "ani", 3) == 0;
    const bool isotropic = trope_length >= 3 && std::memcmp(trope, "iso", 3) == 0;
    if ((!isotropic && !anisotropic) || methode_length < 2 || *nnnr <= 0 ||
        *nmax <= 0 || (*niso != 1 && *niso != 3) ||
        (anisotropic && *niso != 3) || (*na != -1 && *na != 0) ||
        (*nmat != 0 && *nmat != 1) || *rayon <= CU_REAL(0) || *k0 <= CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop, "object_all: invalid sphere mode");
        return;
    }
    const size_t n = size_t(*nnnr);
    if (n > size_t(std::numeric_limits<int>::max()) / n ||
        n * n > size_t(std::numeric_limits<int>::max()) / n ||
        n * n * n > size_t(*nmax)) {
        object_error(infostr, infostr_length, nstop, "object_all: invalid sphere grid");
        return;
    }
    const size_t grid = n * n * n;
    const bool compact = *na == -1;
    *rayon *= CU_REAL(1.0e-9);
    *xg *= CU_REAL(1.0e-9);
    *yg *= CU_REAL(1.0e-9);
    *zg *= CU_REAL(1.0e-9);
    *nx = *ny = *nz = *nnnr;
    *aretecube = CU_REAL(2) * *rayon / CU_REAL(*nnnr);

    // 1. The sphere kernel computes occupancy; the shared stage downloads it.
    CU_INT* d_map = cuda_data::get_dipole_map(grid);
    std::vector<int> host_map;
    if (!d_map || !ifdda_object_all_cuda(geometry, grid,
            sphere_dipole_map_wrapper(d_map, *nnnr, *rayon, *aretecube),
            host_map)) {
        object_error(infostr, infostr_length, nstop, "object_all: sphere kernel failed");
        return;
    }
    // 2. Reconstruct coordinates only for the Fortran compact output slots,
    // without materializing three full-grid CPU arrays (nor GPU xyz buffers).
    const ObjectAllRegularLattice lattice{
        *nnnr, *nnnr, *nnnr, *aretecube, -*rayon, -*rayon, -*rayon,
        *xg, *yg, *zg, false};
    if (!object_all_build_regular_cpu_dipoles(
             geometry, host_map, *nmax, compact, lattice, tabdip,
             xs, ys, zs, nbsphere, ndipole)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: sphere CPU map failed");
        return;
    }

    // 3. Small homogeneous material parameters are computed once on CPU;
    // the per-dipole arrays are initialized in compact persistent GPU storage.
    const int discretized = 1;
    const int invert_result = 1;
    CU_REAL local_sldr = *sldr;
    COMPLEX scalar_alpha{};
    COMPLEX tensor_alpha[9]{};
    if (anisotropic)
        polaepstens_cuda(aretecube, epsani, eps0, k0, &discretized, methode,
                         &invert_result, tensor_alpha, methode_length);
    else
        poladiff_cuda(aretecube, eps, eps0, k0, &discretized, methode,
                      &local_sldr, &scalar_alpha, methode_length);

    if (!fill_homogeneous_materials(geometry, tabdip, grid, *nmax, *nbsphere, *niso,
                               compact, anisotropic, eps, epsani, scalar_alpha,
                               tensor_alpha, epsilon, polarisa)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: compact GPU materials failed");
        return;
    }
    if (*nmat == 0 && !write_sphere_mesh(*nnnr, *nbsphere, *rayon, *aretecube,
                                         *xg, *yg, *zg, xs, ys, zs)) {
        object_error(infostr, infostr_length, nstop, "object_all: mesh output failed");
        return;
    }
    *sldr = local_sldr;
}

extern "C" void ifdda_object_all_nspheres_cuda(
    int geometry, 
    const char *trope, const COMPLEX *epss, const COMPLEX *epsanis,
    const int *numbersphere, const int *numberspheremax, CU_REAL *xg, CU_REAL *yg,
    CU_REAL *zg, CU_REAL *rayons, const COMPLEX *eps0, CU_REAL *xs,
    CU_REAL *ys, CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube, int *tabdip,
    int *tabnbs, const int *nnnr, const int *nmax, int *nbsphere, int *ndipole,
    int *nx, int *ny, int *nz, const char *methode, const int *na,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip, char *infostr,
    int *nstop, std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length) {
    if (geometry != int(GeometryId::NSpheres)) { object_error(infostr,infostr_length,nstop,"Unsupported geometry"); return; }
    (void)trope_length;
    if (!trope || !epss || !epsanis || !numbersphere || !numberspheremax || !xg ||
        !yg || !zg || !rayons || !eps0 || !xs || !ys || !zs || !k0 ||
        !aretecube || !tabdip || !tabnbs || !nnnr || !nmax || !nbsphere ||
        !ndipole || !nx || !ny || !nz || !methode || !na || !epsilon ||
        !polarisa || !niso || !sldr || !nmat || !file_id || !group_iddip ||
        !infostr || !nstop || *numbersphere <= 0 || *numbersphere > *numberspheremax ||
        *nnnr <= 0 || *nmax <= 0 || (*niso != 1 && *niso != 3) ||
        (*na != -1 && *na != 0)) {
        object_error(infostr, infostr_length, nstop,
                  "Invalid multiple-sphere geometry arguments");
        return;
    }

    *nstop = 0;
    *nbsphere = 0;
    *ndipole = 0;
    const CU_REAL nanometre = CU_REAL(1e-9);
    CU_REAL xmin = std::numeric_limits<CU_REAL>::max();
    CU_REAL xmax = -xmin;
    CU_REAL ymin = xmin;
    CU_REAL ymax = -xmin;
    CU_REAL zmin = xmin;
    CU_REAL zmax = -xmin;
    for (int sphere = 0; sphere < *numbersphere; ++sphere) {
        rayons[sphere] *= nanometre;
        xg[sphere] *= nanometre;
        yg[sphere] *= nanometre;
        zg[sphere] *= nanometre;
        if (rayons[sphere] == CU_REAL(0)) {
            object_error(infostr, infostr_length, nstop, "object nspheres: radius=0!");
            return;
        }
        xmin = std::min(xmin, xg[sphere] - rayons[sphere]);
        xmax = std::max(xmax, xg[sphere] + rayons[sphere]);
        ymin = std::min(ymin, yg[sphere] - rayons[sphere]);
        ymax = std::max(ymax, yg[sphere] + rayons[sphere]);
        zmin = std::min(zmin, zg[sphere] - rayons[sphere]);
        zmax = std::max(zmax, zg[sphere] + rayons[sphere]);
    }

    *aretecube = std::max({xmax - xmin, ymax - ymin, zmax - zmin}) /
                 CU_REAL(*nnnr);
    if (*aretecube <= CU_REAL(0) || !std::isfinite(double(*aretecube))) {
        object_error(infostr, infostr_length, nstop, "Invalid multiple-sphere mesh size");
        return;
    }
    *nx = int(std::floor((xmax - xmin) / *aretecube + CU_REAL(0.5)));
    *ny = int(std::floor((ymax - ymin) / *aretecube + CU_REAL(0.5)));
    *nz = int(std::floor((zmax - zmin) / *aretecube + CU_REAL(0.5)));
    if (*nx <= 0 || *ny <= 0 || *nz <= 0 || *nx > *nmax / *ny ||
        *nx * *ny > *nmax / *nz) {
        object_error(infostr, infostr_length, nstop,
                  "nmax parameter too small: increase nxm nym nzm");
        return;
    }
    const int mesh_count = *nx * *ny * *nz;
    // CUDA creates the sphere occupancy map below. Preserve its source;
    // CPU handles the Fortran one-based tabdip and destination material IDs.
    std::fill(tabdip, tabdip + *nmax, 0);
    std::fill(tabnbs, tabnbs + *nmax, 0);

    std::vector<CU_REAL> parameters(size_t(*numbersphere) * 4);
    std::vector<CU_REAL> mesh_x(static_cast<size_t>(mesh_count));
    std::vector<CU_REAL> mesh_y(static_cast<size_t>(mesh_count));
    std::vector<CU_REAL> mesh_z(static_cast<size_t>(mesh_count));
    for (int sphere = 0; sphere < *numbersphere; ++sphere) {
        parameters[4*sphere] = xg[sphere];
        parameters[4*sphere+1] = yg[sphere];
        parameters[4*sphere+2] = zg[sphere];
        parameters[4*sphere+3] = rayons[sphere];
    }
    CU_INT* device_map = cuda_data::get_dipole_map(size_t(mesh_count));
    CU_REAL* device_parameters = static_cast<CU_REAL*>(
        cuda_data::get_device_workspace_buffer_bytes(
            "object_all.nspheres.parameters", parameters.size()*sizeof(CU_REAL)));
    std::vector<int> sphere_map;
    if (!device_map || !device_parameters ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(device_parameters, parameters.data(),
                parameters.size()*sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
            "object_all", "upload sphere parameters") ||
        !ifdda_object_all_cuda(geometry, size_t(mesh_count),
            multi_sphere_dipole_map_wrapper(device_map, *nx, *ny, *nz,
                *aretecube, xmin, ymin, zmin, device_parameters, *numbersphere,
                *numbersphere), sphere_map)) {
        object_error(infostr, infostr_length, nstop, "CUDA multiple-sphere geometry failed");
        return;
    }
    for (int flat = 0; flat < mesh_count; ++flat) {
        const int k = flat % *nx;
        const int j = (flat / *nx) % *ny;
        const int i = flat / (*nx * *ny);
        mesh_x[size_t(flat)] = xmin + (CU_REAL(k)+CU_REAL(0.5))* *aretecube;
        mesh_y[size_t(flat)] = ymin + (CU_REAL(j)+CU_REAL(0.5))* *aretecube;
        mesh_z[size_t(flat)] = zmin + (CU_REAL(i)+CU_REAL(0.5))* *aretecube;
    }

    const bool all_grid_sites = *na == 0;
    const bool anisotropic = std::memcmp(trope, "iso", 3) != 0;
    const int discretized = 1;
    const int invert_result = 1;
    const bool updates_sldr = methode[0] == 'L' && methode[1] == 'R';
    // Stateful LR updates sldr for every occupied site: retain its CPU path.
    const bool use_gpu_materials = anisotropic || !updates_sldr;
    if (!use_gpu_materials)
        polaepsto0_(polarisa, epsilon, tabdip, niso, nmax);
    std::vector<COMPLEX> scalar_polar(static_cast<size_t>(*numbersphere));
    std::vector<COMPLEX> tensor_polar(static_cast<size_t>(*numbersphere) * 9);
    if (!anisotropic && !updates_sldr) {
        for (int sphere = 0; sphere < *numbersphere; ++sphere)
            poladiff_cuda(&*aretecube, epss + sphere, eps0, k0, &discretized, methode,
                      sldr, scalar_polar.data() + sphere, methode_length);
    } else if (anisotropic) {
        for (int sphere = 0; sphere < *numbersphere; ++sphere)
            polaepstens_cuda(&*aretecube, epsanis + size_t(sphere) * 9, eps0, k0,
                         &discretized, methode, &invert_result,
                         tensor_polar.data() + size_t(sphere) * 9, methode_length);
    }

    if (!object_all_build_cpu_dipoles(geometry, sphere_map, *nmax,
             !all_grid_sites, mesh_x.data(), mesh_y.data(), mesh_z.data(),
             tabdip, tabnbs, xs, ys, zs, nbsphere, ndipole, *numbersphere)) {
        object_error(infostr, infostr_length, nstop,
                     "CUDA multiple-sphere CPU map invalid");
        return;
    }
    if (!use_gpu_materials) {
        for (int flat = 0; flat < mesh_count; ++flat) {
            const int selected = sphere_map[size_t(flat)];
            if (selected == 0) continue;
            const size_t destination = size_t(tabdip[flat] - 1);
            const int material_sphere = selected - 1;
            COMPLEX value{};
            poladiff_cuda(&*aretecube, epss + material_sphere, eps0, k0,
                      &discretized, methode, sldr, &value, methode_length);
            polarisa[destination] = value;
            epsilon[destination] = epss[material_sphere];
        }
    }

    if (use_gpu_materials) {
        const size_t per_material = anisotropic ? 9 : 1;
        std::vector<COMPLEX> host_epsilon(size_t(*numbersphere) * per_material);
        std::vector<COMPLEX> host_alpha(size_t(*numbersphere) * per_material);
        for (int sphere = 0; sphere < *numbersphere; ++sphere) {
            for (size_t component = 0; component < per_material; ++component) {
                const size_t destination = size_t(sphere) * per_material + component;
                const COMPLEX eps_value = anisotropic ? epsanis[destination]
                                                             : epss[sphere];
                const COMPLEX alpha_value = anisotropic ? tensor_polar[destination]
                                                               : scalar_polar[sphere];
                host_epsilon[destination] = eps_value;
                host_alpha[destination] = alpha_value;
            }
        }
        if (!fill_labeled_materials(geometry,
                tabdip, tabnbs, size_t(mesh_count), *nmax, *nbsphere,
                *niso, *numbersphere, true, host_epsilon.data(),
                host_alpha.data(), epsilon, polarisa)) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA multiple-sphere material initialization failed");
            return;
        }
    }

    if (!use_gpu_materials && *nbsphere > 0) {
        // Stateful LR computes each occupied dipole on CPU. Publish the
        // resulting compact values through the same persistent GPU stage.
        if (!publish_compact_materials(geometry, size_t(mesh_count), *nmax,
                *nbsphere, 1, polarisa, epsilon)) {
            object_error(infostr, infostr_length, nstop,
                         "CUDA multiple-sphere LR material publishing failed");
            return;
        }
    }

    if (*nmat == 0 && !write_rectangular_mesh(*nx, *ny, *nz, *nbsphere,
                                        mesh_x.data(), mesh_y.data(), mesh_z.data(),
                                        xs, ys, zs)) {
        object_error(infostr, infostr_length, nstop, "Unable to write multiple-sphere mesh files");
        return;
    }
    if (*nmat == 2) {
        const int dimensions[2] = {*nbsphere, *nmax};
        hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dimensions, 17);
    }
    (void)file_id;
}

extern "C" void ifdda_object_all_arbitrary_cuda(
    int geometry, 
    const char* object_type, COMPLEX* epsilon_value, COMPLEX* epsilon_anisotropic,
    const COMPLEX* epsilon_background, CU_REAL* xs, CU_REAL* ys, CU_REAL* zs,
    const CU_REAL* k0, CU_REAL* cell_size, int* dipole_map, const int* random_count,
    const int* nmax, int* dipole_count, int* ndipole, int* nx, int* ny, int* nz,
    const char* polarizability_method, const char* filename, int* particle_count,
    COMPLEX* epsilon, COMPLEX* polarizability, const int* isotropy,
    CU_REAL* sldr, const int* output_mode, const int64_t* file_id, const int64_t* group_id,
    char* infostr, int* nstop, std::size_t object_type_length,
    std::size_t method_length, std::size_t filename_length,
    std::size_t infostr_length) {
    if (geometry != int(GeometryId::Arbitrary)) { object_set_info(infostr,"Unsupported geometry"); if(nstop) *nstop=1; return; }
    (void)object_type_length;
    (void)infostr_length;
    if (nstop)
        *nstop = 1;
    object_set_info(infostr, "Invalid arbitrary-object arguments");
    if (!object_type || !epsilon_value || !epsilon_anisotropic || !epsilon_background || !xs ||
        !ys || !zs || !k0 || !cell_size || !dipole_map || !random_count || !nmax ||
        !dipole_count || !ndipole || !nx || !ny || !nz || !polarizability_method || !filename ||
        !particle_count || !epsilon || !polarizability || !isotropy || !sldr || !output_mode ||
        !file_id || !group_id || !nstop || !infostr || *nmax <= 0 || *random_count < 0 ||
        (*isotropy != 1 && *isotropy != 3) || *output_mode < 0 || *output_mode > 2)
        return;

    *dipole_count = 0;
    *ndipole = 0;
    if (*particle_count == -1)
        *particle_count = 0;
    const std::size_t isotropy_count = std::size_t(*isotropy);
    if (std::size_t(*nmax) > std::numeric_limits<std::size_t>::max() /
                                  (isotropy_count * isotropy_count)) {
        object_set_info(infostr, "Arbitrary-object array size overflow");
        return;
    }

    polaepsto0_(polarizability, epsilon, dipole_map, isotropy, nmax);

    const std::string path = object_fortran_string(filename, filename_length);
    std::ifstream input(path);
    if (!input) {
        object_set_info(infostr, "arbitrary object: name of file does not exist");
        return;
    }
    if (!(input >> *nx >> *ny >> *nz >> *cell_size) || *nx <= 0 || *ny <= 0 || *nz <= 0 ||
        *nx > INT_MAX / *ny || *nx * *ny > INT_MAX / *nz || *cell_size <= CU_REAL(0)) {
        object_set_info(infostr, "Invalid arbitrary-object dimensions or cell size");
        return;
    }
    const int count = *nx * *ny * *nz;
    if (count > *nmax) {
        object_set_info(infostr, "nmax parameter too small: increase nxm nym nzm");
        return;
    }
    for (int z = 0; z < *nz; ++z) {
        for (int y = 0; y < *ny; ++y) {
            for (int x = 0; x < *nx; ++x) {
                const int index = x + *nx * (y + *ny * z);
                if (!(input >> xs[index] >> ys[index] >> zs[index])) {
                    object_set_info(infostr, "Invalid arbitrary-object coordinate data");
                    return;
                }
            }
        }
    }

    const int discretized = 1;
    const int invert = 1;
    const COMPLEX epsilon0 = *epsilon_background;
    const bool isotropic_material = std::strncmp(object_type, "iso", 3) == 0;
    const std::string method = object_fortran_string(polarizability_method, method_length);
    char method_chars[2] = {'R', 'R'};
    if (method.size() > 0) method_chars[0] = method[0];
    if (method.size() > 1) method_chars[1] = method[1];
    const int method_code = object_polarizability_method_code(method_chars);
    // LR mutates sldr between cells. Preserve its original ordered CPU loop.
    const bool gpu_scalar = isotropic_material && *isotropy == 1 &&
                            method_code != 0 && method_code != 6;
    const bool gpu_tensor = !isotropic_material && *isotropy == 3;
    std::vector<COMPLEX> cell_epsilon;
    if (gpu_scalar)
        cell_epsilon.resize(std::size_t(count));
    else if (gpu_tensor)
        cell_epsilon.resize(std::size_t(count) * 9);
    for (int index = 0; index < count; ++index) {
        dipole_map[index] = index + 1;
        const std::size_t diagonal = object_array_offset(*nmax, *isotropy, index, 0, 0);
        if (isotropic_material) {
            COMPLEX object_epsilon{};
            if (!object_read_complex(input, &object_epsilon)) {
                object_set_info(infostr, "Invalid arbitrary-object permittivity data");
                return;
            }
            *epsilon_value = object_epsilon;
            if (gpu_scalar) {
                cell_epsilon[std::size_t(index)] = object_epsilon;
            } else {
                epsilon[diagonal] = object_epsilon;
                COMPLEX alpha{};
                poladiff_cuda(cell_size, &object_epsilon, &epsilon0, k0, &discretized,
                          method_chars, sldr, &alpha, sizeof(method_chars));
                polarizability[diagonal] = alpha;
            }
        } else {
            COMPLEX tensor[9]{};
            for (int element = 0; element < 9; ++element) {
                if (!object_read_complex(input, &tensor[element])) {
                    object_set_info(infostr, "Invalid arbitrary-object tensor permittivity data");
                    return;
                }
                if (gpu_tensor)
                    cell_epsilon[std::size_t(index) * 9 + std::size_t(element)] = tensor[element];
                else
                    epsilon[object_array_offset(*nmax, *isotropy, index, element % 3, element / 3)] =
                        tensor[element];
            }
            for (int element = 0; element < 9; ++element)
                epsilon_anisotropic[element] = tensor[element];
            if (!gpu_tensor) {
                COMPLEX alpha[9]{};
                polaepstens_cuda(cell_size, tensor, &epsilon0, k0, &discretized,
                             method_chars, &invert, alpha, sizeof(method_chars));
                for (int element = 0; element < 9; ++element)
                    polarizability[object_array_offset(*nmax, *isotropy, index,
                                                 element % 3, element / 3)] =
                        alpha[element];
            }
        }
    }

    // Finalize the common geometry and full nmax GPU tabdip before any
    // material-scatter wrapper uses tabdip. The arbitrary file is full-grid.
    CU_INT* device_map = cuda_data::get_dipole_map(size_t(count));
    std::vector<int> file_map;
    if (!device_map || !ifdda_object_all_cuda(geometry, size_t(count),
            full_grid_dipole_map_wrapper(device_map,count),file_map) ||
        !object_all_build_cpu_dipoles(geometry, file_map, *nmax, false,
            xs, ys, zs, dipole_map, nullptr, xs, ys, zs,
            dipole_count, ndipole, 1, true)) {
        object_set_info(infostr,"Cannot initialize arbitrary dipole_map");
        return;
    }

    // Now the input materials may be uploaded and processed on GPU.
    cuda_data::PolarizabilityBatchDeviceBuffers buffers{};
    if (gpu_scalar) {
        const std::vector<int> inside(std::size_t(count), 1);
        if (!cuda_data::get_device_polarizability_batch_buffers(
                std::size_t(count), &buffers) ||
            !cuda_utils::check_cuda(cuda_data::copy_host_complex_to_device(
                buffers.epsilon, cell_epsilon.data(),
                std::size_t(count),
                cuda_utils::TransferCategory::Material),
                "arbitrary GPU materials", "upload scalar permittivity") ||
            !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                buffers.inside, inside.data(),
                std::size_t(count) * sizeof(int),
                cuda_utils::TransferCategory::Geometry),
                "arbitrary GPU materials", "upload occupancy") ||
            !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                buffers.sldr, sldr, sizeof(CU_REAL),
                cuda_utils::TransferCategory::Material),
                "arbitrary GPU materials", "upload parameter") ||
            !cuda_utils::check_cuda(scalar_polarizability_batch_wrapper(
                buffers.epsilon, buffers.inside, buffers.values, buffers.sldr,
                count, *cell_size, cuda_utils::device_value(*epsilon_background), *k0, discretized,
                method_code, *sldr),
                "arbitrary GPU materials", "compute scalar batch")) {
            object_set_info(infostr, "Arbitrary GPU scalar material calculation failed");
            return;
        }
    } else if (gpu_tensor) {
        // The shared material stage now performs the tensor GPU calculation,
        // compact publication and Fortran-strided host material download.
        if (!compute_arbitrary_tensor_materials(geometry, size_t(count), *nmax,
                cell_epsilon.data(), *epsilon_background,
                *cell_size, *k0, method_code, epsilon, polarizability)) {
            object_set_info(infostr,"Arbitrary GPU tensor material calculation failed");
            return;
        }
    }


    if (gpu_scalar || (!gpu_scalar && !gpu_tensor)) {
        if (!retain_inhomogeneous_materials(geometry,dipole_map,size_t(count),*nmax,
                count,*isotropy,
                gpu_scalar ? buffers.inside : nullptr,
                gpu_scalar ? buffers.epsilon : nullptr,
                gpu_scalar ? buffers.values : nullptr,
                epsilon,polarizability)) {
            object_set_info(infostr,"Cannot retain arbitrary compact materials");
            return;
        }
    }
    if (gpu_scalar) {
        // Preserve the original arbitrary-file vacuum tail on the CPU.
        for (int i = count; i < *nmax; ++i)
            epsilon[i] = {CU_REAL(1), CU_REAL(0)};
    }

    *nstop = 0;
    object_set_info(infostr, "");
    std::fprintf(stdout, "You choose an arbitrary object that you define\n");
    std::fprintf(stdout, "The code assumes that you define correctly\n");
    std::fprintf(stdout, "a cubic lattice for your object\n");

    if (*output_mode == 0) {
        std::ofstream mesh_x("x.mat"), mesh_y("y.mat"), mesh_z("z.mat");
        std::ofstream points_x("xc.mat"), points_y("yc.mat"), points_z("zc.mat");
        if (!mesh_x || !mesh_y || !mesh_z || !points_x || !points_y || !points_z) {
            *nstop = 1;
            object_set_info(infostr, "Unable to write arbitrary-object coordinate files");
            return;
        }
        for (int z = 0; z < *nz; ++z) {
            for (int y = 0; y < *ny; ++y) {
                for (int x = 0; x < *nx; ++x) {
                    const int index = x + *nx * (y + *ny * z);
                    points_x << xs[index] << '\n';
                    points_y << ys[index] << '\n';
                    points_z << zs[index] << '\n';
                    if (y == 0 && x == 0) mesh_z << zs[0] + CU_REAL(z) * *cell_size << '\n';
                    if (z == 0 && x == 0) mesh_y << ys[0] + CU_REAL(y) * *cell_size << '\n';
                    if (y == 0 && z == 0) mesh_x << xs[0] + CU_REAL(x) * *cell_size << '\n';
                }
            }
        }
    } else if (*output_mode == 2) {
        int dimensions[4] = {count, *nmax, 0, 0};
        hdf5write1d_i8_(group_id, "Dipole position x", xs, dimensions, 100);
        hdf5write1d_i8_(group_id, "Dipole position y", ys, dimensions, 100);
        hdf5write1d_i8_(group_id, "Dipole position z", zs, dimensions, 100);
    }
}

extern "C" void ifdda_object_all_inhomocuboid1_cuda(
    int geometry, 
    COMPLEX *eps, COMPLEX *eps0, CU_REAL *lnrbc,
    CU_REAL *lcos, int *ntypeabc, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, int *tabmulti,
    int *pmlwrk, const int *nnnr, const int *nmax, int *nbsphere,
    int *ndipole, int *nx, int *ny, int *nz, const int *nxm,
    const int *nym, const int *nzm, const char *methode,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso,
    CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez, CU_REAL *xg,
    CU_REAL *yg, CU_REAL *zg, CU_REAL *lc, const CU_REAL *hc,
    const int *ng, COMPLEX *epsb, const int *na, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip,
    char *infostr, int *nstop, std::size_t methode_length,
    std::size_t infostr_length) {
    if (geometry != int(GeometryId::InhomoCuboid1)) { object_error(infostr,infostr_length,nstop,"Unsupported geometry"); return; }
    if (!eps || !eps0 || !lnrbc || !lcos || !ntypeabc || !xs || !ys ||
        !zs || !k0 || !aretecube || !tabdip || !tabmulti || !pmlwrk ||
        !nnnr || !nmax || !nbsphere || !ndipole || !nx || !ny || !nz ||
        !nxm || !nym || !nzm || !methode || !epsilon || !polarisa ||
        !niso || !sidex || !sidey || !sidez || !xg || !yg || !zg || !lc ||
        !hc || !ng || !epsb || !na || !sldr || !nmat || !file_id ||
        !group_iddip || !infostr || !nstop || *nmax <= 0 || *nnnr <= 0 ||
        (*niso != 1 && *niso != 3)) {
        object_error(infostr, infostr_length, nstop,
                  "Invalid inhomogeneous cuboid arguments");
        return;
    }

    if (*na == 1) {
        *nstop = 0;
        std::memset(infostr, ' ', infostr_length);
        *nbsphere = 0;
        *ndipole = 0;
        if (*lc <= CU_REAL(0) || *hc <= CU_REAL(0)) {
            object_error(infostr, infostr_length, nstop,
                      *lc <= CU_REAL(0) ? "coherence length equal to zero" :
                                          "standard deviation equal to zero");
            return;
        }
        constexpr CU_REAL nanometre = CU_REAL(1e-9);
        *sidex *= nanometre;
        *sidey *= nanometre;
        *sidez *= nanometre;
        *xg *= nanometre;
        *yg *= nanometre;
        *zg *= nanometre;
        *lc *= nanometre;
        if (*sidex <= CU_REAL(0) || *sidey <= CU_REAL(0) ||
            *sidez <= CU_REAL(0)) {
            object_error(infostr, infostr_length, nstop,
                      "object cuboid: zero side length");
            return;
        }
        const CU_REAL base_side = std::max({*sidex, *sidey, *sidez});
        CU_REAL padded_side = base_side;
        CU_REAL lpml = CU_REAL(0);
        int nrbc = 0;
        int npml = 0;
        initnrbc_(k0, lcos, lnrbc, &lpml, &padded_side, nnnr,
                  &nrbc, &npml, pmlwrk, aretecube);
        const CU_REAL padding = padded_side - base_side;
        const CU_REAL padded_x = *sidex + padding;
        const CU_REAL padded_y = *sidey + padding;
        const CU_REAL padded_z = *sidez + padding;
        const auto fortran_nint = [](CU_REAL value) {
            return static_cast<int>(std::floor(value + CU_REAL(0.5)));
        };
        *nx = fortran_nint(padded_x / *aretecube);
        *ny = fortran_nint(padded_y / *aretecube);
        *nz = fortran_nint(padded_z / *aretecube);
        const int material_nx = fortran_nint(*sidex / *aretecube);
        const int material_ny = fortran_nint(*sidey / *aretecube);
        const int material_nz = fortran_nint(*sidez / *aretecube);
        if (*nx <= 0 || *ny <= 0 || *nz <= 0 || material_nx <= 0 ||
            material_ny <= 0 || material_nz <= 0 || *nx > *nxm ||
            *ny > *nym || *nz > *nzm ||
            *nx > std::numeric_limits<int>::max() / *ny ||
            *nx * *ny > std::numeric_limits<int>::max() / *nz ||
            material_nx > std::numeric_limits<int>::max() / material_ny ||
            material_nx * material_ny >
                std::numeric_limits<int>::max() / material_nz) {
            object_error(infostr, infostr_length, nstop,
                      "Dimension Problem of the Box : Box too small!");
            return;
        }
        const int count = *nx * *ny * *nz;
        const int material_count = material_nx * material_ny * material_nz;
        if (count > *nmax || material_count > *nmax) {
            object_error(infostr, infostr_length, nstop,
                      "nmax parameter too small: increase nxm nym nzm");
            return;
        }
        const std::size_t tensor_count = std::size_t(*nmax) *
            std::size_t(*niso) * std::size_t(*niso);
        std::fill_n(tabdip, *nmax, 0);
        std::fill_n(epsilon, tensor_count, COMPLEX{});
        std::fill_n(polarisa, tensor_count, COMPLEX{});
        std::vector<CU_REAL> phases(static_cast<std::size_t>(material_count));
        std::vector<CU_REAL> material(static_cast<std::size_t>(material_count));
        random_bridge_init(*ng);
        for (CU_REAL &phase : phases)
            phase = random_bridge_next();
        CU_REAL mean = CU_REAL(0);
        CU_REAL deviation = CU_REAL(0);
        if (ifdda_correlated_material_field_cuda(
                material_nx, material_ny, material_nz, *aretecube, *lc,
                *hc, eps->real(), eps->imag(), phases.data(), material.data(),
                &mean, &deviation) != 0) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA correlated cuboid material generation failed");
            return;
        }
        if (!initialize_cuda_device()) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML cuboid device initialization failed");
            return;
        }
        cuda_data::PmlDeviceBuffers device_buffers{};
        if (!cuda_data::get_device_pml_buffers(std::size_t(count),
                &device_buffers, std::size_t(material_count))) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML cuboid buffers allocation failed");
            return;
        }
        if (!cuda_utils::check_cuda(cuboid_geometry_wrapper(
                device_buffers.xs, device_buffers.ys, device_buffers.zs,
                device_buffers.inside_map, *nx, *ny, *nz, *aretecube,
                padded_x, padded_y, padded_z, CU_REAL(0), CU_REAL(0),
                CU_REAL(0)), "PML cuboid geometry", "generate grid") ||
            !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                device_buffers.material, material.data(),
                std::size_t(material_count) * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Material),
                "PML cuboid material", "upload correlated field")) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML cuboid setup failed");
            return;
        }
        if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                xs, device_buffers.xs, std::size_t(count) * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Geometry),
                "PML cuboid geometry", "download x coordinates") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                ys, device_buffers.ys, std::size_t(count) * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Geometry),
                "PML cuboid geometry", "download y coordinates") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                zs, device_buffers.zs, std::size_t(count) * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Geometry),
                "PML cuboid geometry", "download z coordinates")) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML cuboid geometry transfer failed");
            return;
        }
        CU_REAL xmin = std::numeric_limits<CU_REAL>::max();
        CU_REAL ymin = std::numeric_limits<CU_REAL>::max();
        CU_REAL zmin = std::numeric_limits<CU_REAL>::max();
        for (int ix = 0; ix < *nx; ++ix) {
            const CU_REAL value = *aretecube * (CU_REAL(ix) + CU_REAL(0.5)) -
                                  padded_x / CU_REAL(2);
            if (std::abs(value) < *sidex / CU_REAL(2))
                xmin = std::min(xmin, value);
        }
        for (int iy = 0; iy < *ny; ++iy) {
            const CU_REAL value = *aretecube * (CU_REAL(iy) + CU_REAL(0.5)) -
                                  padded_y / CU_REAL(2);
            if (std::abs(value) < *sidey / CU_REAL(2))
                ymin = std::min(ymin, value);
        }
        for (int iz = 0; iz < *nz; ++iz) {
            const CU_REAL value = *aretecube * (CU_REAL(iz) + CU_REAL(0.5)) -
                                  padded_z / CU_REAL(2);
            if (std::abs(value) < *sidez / CU_REAL(2))
                zmin = std::min(zmin, value);
        }
        const CU_REAL contrast_scale = (*aretecube * *aretecube * *aretecube) /
            (CU_REAL(4) * CU_REAL(3.1415926535897932384626433832795));
        const COMPLEX epsilon_value(CU_REAL(1), CU_REAL(0));
        const COMPLEX epsilon_background = *eps0;
        if (!cuda_utils::check_cuda(pml_cuboid_material_profile_wrapper(
                device_buffers.xs, device_buffers.ys, device_buffers.zs,
                device_buffers.material, device_buffers.epsilon_output,
                device_buffers.contrast_output, *nx, *ny, *nz,
                material_nx, material_ny, material_nz, *sidex, *sidey,
                *sidez, xmin, ymin, zmin, *aretecube, eps->imag(),
                cuda_utils::device_value(epsilon_value), cuda_utils::device_value(epsilon_background), contrast_scale,
                *ntypeabc, nrbc), "PML cuboid material", "build profile")) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML cuboid material profile failed");
            return;
        }
        std::vector<COMPLEX> epsilon_values(static_cast<std::size_t>(count));
        std::vector<COMPLEX> contrast_values(static_cast<std::size_t>(count));
        if (!cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                epsilon_values.data(), device_buffers.epsilon_output,
                std::size_t(count),
                cuda_utils::TransferCategory::Material),
                "PML cuboid material", "download epsilon") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                contrast_values.data(), device_buffers.contrast_output,
                std::size_t(count),
                cuda_utils::TransferCategory::Material),
                "PML cuboid material", "download contrast")) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML cuboid result transfer failed");
            return;
        }
        for (int flat = 0; flat < count; ++flat) {
            const CU_REAL x = xs[flat];
            const CU_REAL y = ys[flat];
            const CU_REAL z = zs[flat];
            const bool inside = std::abs(x) < *sidex / CU_REAL(2) &&
                std::abs(y) < *sidey / CU_REAL(2) &&
                std::abs(z) < *sidez / CU_REAL(2);
            const COMPLEX local_epsilon = epsilon_values[std::size_t(flat)];
            epsilon[std::size_t(flat)] = local_epsilon;
            polarisa[std::size_t(flat)] = contrast_values[std::size_t(flat)];
            xs[flat] = x + *xg;
            ys[flat] = y + *yg;
            zs[flat] = z + *zg;
            if (inside) {
                *eps = local_epsilon;
                epsb[flat] = local_epsilon;
            }
        }
        std::vector<int> pml_cpu_map;
        if (!ifdda_object_all_cuda(geometry, size_t(count), cudaSuccess,
                pml_cpu_map, device_buffers.inside_map, true) ||
            !object_all_build_cpu_dipoles(geometry, pml_cpu_map, *nmax,
                false, xs, ys, zs, tabdip, nullptr, xs, ys, zs,
                nbsphere, ndipole, 1, true) ||
            !retain_inhomogeneous_materials(geometry,tabdip,size_t(count),*nmax,*nbsphere,
                *niso,device_buffers.inside_map,device_buffers.epsilon_output,
                device_buffers.contrast_output,epsilon,polarisa)) {
            object_error(infostr,infostr_length,nstop,
                         "Cannot retain compact PML cuboid materials");
            return;
        }
        std::printf(" average espilon   : %.17g\n standard deviation: %.17g\n",
                    static_cast<double>(mean), static_cast<double>(deviation));
        std::vector<CU_REAL> shifted_mesh_x,shifted_mesh_y,shifted_mesh_z;
        if (*nmat == 0) {
            shifted_mesh_x.resize(size_t(count));
            shifted_mesh_y.resize(size_t(count));
            shifted_mesh_z.resize(size_t(count));
            for (int p=0;p<count;++p) {
                shifted_mesh_x[size_t(p)] = xs[p] - *xg;
                shifted_mesh_y[size_t(p)] = ys[p] - *yg;
                shifted_mesh_z[size_t(p)] = zs[p] - *zg;
            }
        }
        if (*nmat == 0 &&
            !write_rectangular_mesh(*nx,*ny,*nz,count,
                shifted_mesh_x.data(),shifted_mesh_y.data(),shifted_mesh_z.data(),
                xs,ys,zs)) {
            object_error(infostr, infostr_length, nstop,
                      "Unable to write inhomogeneous cuboid PML mesh files");
            return;
        }
        if (*nmat == 2) {
            const int dimensions[2] = {*nbsphere, *nmax};
            hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dimensions, 17);
            hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dimensions, 17);
            hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dimensions, 17);
        }
        std::printf(" Cuboid inhomogeneous PML::nbsphere= %d\n", *nbsphere);
        (void)file_id;
        (void)tabmulti;
        (void)sldr;
        return;
    }
    if (*na != -1 && *na != 0) {
        object_error(infostr, infostr_length, nstop,
                  "na should be equal to -1 or 0 or 1");
        return;
    }

    *nstop = 0;
    std::memset(infostr, ' ', infostr_length);
    *nbsphere = 0;
    *ndipole = 0;
    if (*lc <= CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "coherence length equal to zero");
        return;
    }
    if (*hc <= CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "standard deviation equal to zero");
        return;
    }
    if (*sidex <= CU_REAL(0) || *sidey <= CU_REAL(0) ||
        *sidez <= CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "object cuboid: zero side length");
        return;
    }

    constexpr CU_REAL nanometre = CU_REAL(1e-9);
    const CU_REAL side_x = *sidex * nanometre;
    const CU_REAL side_y = *sidey * nanometre;
    const CU_REAL side_z = *sidez * nanometre;
    const CU_REAL max_side = std::max({side_x, side_y, side_z});
    const CU_REAL cell_size = max_side / CU_REAL(*nnnr);
    const auto fortran_nint = [](CU_REAL value) {
        return static_cast<int>(std::floor(value + CU_REAL(0.5)));
    };
    *nx = fortran_nint(side_x / cell_size);
    *ny = fortran_nint(side_y / cell_size);
    *nz = fortran_nint(side_z / cell_size);
    if (*nx <= 0 || *ny <= 0 || *nz <= 0 || *nx > *nxm || *ny > *nym ||
        *nz > *nzm || *nx > std::numeric_limits<int>::max() / *ny ||
        *nx * *ny > std::numeric_limits<int>::max() / *nz) {
        object_error(infostr, infostr_length, nstop,
                  "Dimension Problem of the Box : Box too small!");
        return;
    }
    const int count = *nx * *ny * *nz;
    if (count > *nmax) {
        object_error(infostr, infostr_length, nstop,
                  "nmax parameter too small: increase nxm nym nzm");
        return;
    }

    *aretecube = cell_size;
    *sidex = side_x;
    *sidey = side_y;
    *sidez = side_z;
    *xg *= nanometre;
    *yg *= nanometre;
    *zg *= nanometre;
    *lc *= nanometre;

    const std::size_t tensor_count = std::size_t(*nmax) *
                                     std::size_t(*niso) * std::size_t(*niso);
    std::fill_n(tabdip, *nmax, 0);
    std::fill_n(epsilon, tensor_count, COMPLEX{});
    std::fill_n(polarisa, tensor_count, COMPLEX{});

    std::vector<CU_REAL> mesh_x(static_cast<size_t>(count));
    std::vector<CU_REAL> mesh_y(static_cast<size_t>(count));
    std::vector<CU_REAL> mesh_z(static_cast<size_t>(count));
    std::vector<int> inside_map;
    CU_INT* device_map = cuda_data::get_dipole_map(size_t(count));
    if (!device_map || !ifdda_object_all_cuda(geometry, size_t(count),
            cuboid_dipole_map_wrapper(device_map,*nx,*ny,*nz,
                cell_size,side_x,side_y,side_z), inside_map)) {
        object_error(infostr,infostr_length,nstop,
            "CUDA inhomogeneous cuboid geometry generation failed");
        return;
    }

    // The historical cuboid kernel encodes an occupied voxel as flat+1.
    // The common CPU map builder expects binary occupancy for this geometry.
    for (int& occupied : inside_map) occupied = occupied != 0 ? 1 : 0;
    for (int p=0;p<count;++p) {
        const int k=p%*nx,j=(p / *nx)%*ny,i=p/(*nx * *ny);
        mesh_x[size_t(p)] = cell_size*(CU_REAL(k)+CU_REAL(0.5)) - side_x/CU_REAL(2);
        mesh_y[size_t(p)] = cell_size*(CU_REAL(j)+CU_REAL(0.5)) - side_y/CU_REAL(2);
        mesh_z[size_t(p)] = cell_size*(CU_REAL(i)+CU_REAL(0.5)) - side_z/CU_REAL(2);
    }

    std::vector<CU_REAL> phases(static_cast<std::size_t>(count));
    std::vector<CU_REAL> material_real(static_cast<std::size_t>(count));
    random_bridge_init(*ng);
    for (CU_REAL &phase : phases)
        phase = random_bridge_next();
    CU_REAL mean = CU_REAL(0);
    CU_REAL standard_deviation = CU_REAL(0);
    const int material_status = ifdda_correlated_material_field_cuda(
        *nx, *ny, *nz, cell_size, *lc, *hc, eps->real(), eps->imag(),
        phases.data(), material_real.data(), &mean, &standard_deviation);
    if (material_status != 0) {
        object_error(infostr, infostr_length, nstop,
                  "CUDA correlated cuboid material generation failed");
        return;
    }
    for (int flat = 0; flat < count; ++flat) {
        epsb[flat] = COMPLEX(material_real[std::size_t(flat)],
                                    eps->imag());
    }
    std::printf(" average espilon   : %.17g\n standard deviation: %.17g\n",
                static_cast<double>(mean),
                static_cast<double>(standard_deviation));

    if (*na == 0)
        *nbsphere = count;
    const int discretized = 1;
    const int method_code = object_polarizability_method_code(methode);
    const bool use_gpu_scatter = method_code != 0 && *niso == 1;
    std::vector<COMPLEX> batch_epsilon(static_cast<std::size_t>(count));
    std::vector<COMPLEX> batch_alpha(static_cast<std::size_t>(count), COMPLEX{});
    for (int flat = 0; flat < count; ++flat)
        batch_epsilon[std::size_t(flat)] = COMPLEX(
            epsb[flat].real(), epsb[flat].imag());
    cuda_data::PolarizabilityBatchDeviceBuffers batch_buffers{};
    if (method_code != 0) {
        if (!cuda_data::get_device_polarizability_batch_buffers(
                std::size_t(count), &batch_buffers) ||
            !cuda_utils::check_cuda(cuda_data::copy_host_complex_to_device(
                batch_buffers.epsilon, batch_epsilon.data(), std::size_t(count),
                cuda_utils::TransferCategory::Material),
                "inhomogeneous cuboid polarizability", "upload epsilon batch") ||
            !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                batch_buffers.inside, inside_map.data(), std::size_t(count) * sizeof(int),
                cuda_utils::TransferCategory::Geometry),
                "inhomogeneous cuboid polarizability", "upload inside map") ||
            !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                batch_buffers.sldr, sldr, sizeof(CU_REAL),
                cuda_utils::TransferCategory::Material),
                "inhomogeneous cuboid polarizability", "upload polarizability parameter") ||
            !cuda_utils::check_cuda(scalar_polarizability_batch_wrapper(
                batch_buffers.epsilon, batch_buffers.inside, batch_buffers.values,
                batch_buffers.sldr, count, *aretecube,
                cuda_utils::device_value(*eps0), *k0,
                discretized, method_code, *sldr),
                "inhomogeneous cuboid polarizability", "compute alpha batch on GPU") ||
            (!use_gpu_scatter && !cuda_utils::check_cuda(
                cuda_data::copy_device_complex_to_host(
                    batch_alpha.data(), batch_buffers.values,
                    std::size_t(count),
                    cuda_utils::TransferCategory::Material),
                "inhomogeneous cuboid polarizability", "download alpha batch")) ||
            (method_code == 6 && !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                sldr, batch_buffers.sldr, sizeof(CU_REAL),
                cuda_utils::TransferCategory::Material),
                "inhomogeneous cuboid polarizability", "download polarizability parameter"))) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA inhomogeneous cuboid polarizability batch failed");
            return;
        }
    }
    std::vector<CU_REAL> field_x(static_cast<std::size_t>(count));
    std::vector<CU_REAL> field_y(static_cast<std::size_t>(count));
    std::vector<CU_REAL> field_z(static_cast<std::size_t>(count));
    for (int flat = 0; flat < count; ++flat) {
        const CU_REAL x = mesh_x[std::size_t(flat)] + *xg;
        const CU_REAL y = mesh_y[std::size_t(flat)] + *yg;
        const CU_REAL z = mesh_z[std::size_t(flat)] + *zg;
        field_x[std::size_t(flat)] = x;
        field_y[std::size_t(flat)] = y;
        field_z[std::size_t(flat)] = z;
    }
    if (!object_all_build_cpu_dipoles(geometry, inside_map, *nmax,
             *na == -1, field_x.data(), field_y.data(), field_z.data(),
             tabdip, nullptr, xs, ys, zs, nbsphere, ndipole)) {
        object_error(infostr, infostr_length, nstop,
                     "CUDA inhomogeneous cuboid CPU map invalid");
        return;
    }
    for (int flat = 0; flat < count; ++flat) {
        if (!inside_map[size_t(flat)]) continue;
        *eps = epsb[flat];
        if (!use_gpu_scatter) {
            const std::size_t index = size_t(tabdip[flat] - 1);
            epsilon[index] = *eps;
            polarisa[index] = batch_alpha[std::size_t(flat)];
        }
    }
    if (*na == 0)
        *nbsphere = count;
    if (!retain_inhomogeneous_materials(geometry,tabdip,size_t(count),*nmax,
            *nbsphere,*niso,
            use_gpu_scatter ? batch_buffers.inside : nullptr,
            use_gpu_scatter ? batch_buffers.epsilon : nullptr,
            use_gpu_scatter ? batch_buffers.values : nullptr,
            epsilon,polarisa)) {
        object_error(infostr,infostr_length,nstop,
                     "Cannot retain compact inhomogeneous cuboid materials");
        return;
    }

    if (*nmat == 0 &&
        !write_rectangular_mesh(*nx, *ny, *nz, *nbsphere,
                              field_x.data(), field_y.data(), field_z.data(),
                              xs, ys, zs)) {
        object_error(infostr, infostr_length, nstop,
                  "Unable to write inhomogeneous cuboid mesh files");
        return;
    }
    if (*nmat == 2) {
        const int dimensions[2] = {*nbsphere, *nmax};
        hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dimensions, 17);
    }
    std::printf(" Cuboid inhomogeneous::nbsphere= %d\n", *nbsphere);
    (void)lnrbc;
    (void)lcos;
    (void)ntypeabc;
    (void)tabmulti;
    (void)pmlwrk;
    (void)file_id;
}

extern "C" void ifdda_object_all_inhomosphere_cuda(
    int geometry, 
    COMPLEX *eps, const COMPLEX *eps0, CU_REAL *lnrbc,
    CU_REAL *lcos, int *ntypeabc, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, int *tabmulti,
    int *pmlwrk, const int *nnnr, const int *nmax, int *nbsphere,
    int *ndipole, int *nx, int *ny, int *nz, const char *methode,
    const int *na, COMPLEX *epsilon, COMPLEX *polarisa,
    const int *niso, CU_REAL *rayon, CU_REAL *lc, CU_REAL *hc,
    const int *ng, COMPLEX *epsb, CU_REAL *sldr, const int *nmat,
    const int64_t *file_id, const int64_t *group_iddip, char *infostr, int *nstop,
    std::size_t methode_length, std::size_t infostr_length) {
    if (geometry != int(GeometryId::InhomoSphere)) { object_error(infostr,infostr_length,nstop,"Unsupported geometry"); return; }
    if (!eps || !eps0 || !lnrbc || !lcos || !ntypeabc || !xs || !ys ||
        !zs || !k0 || !aretecube || !tabdip || !tabmulti || !pmlwrk ||
        !nnnr || !nmax || !nbsphere || !ndipole || !nx || !ny || !nz ||
        !methode || !na || !epsilon || !polarisa || !niso || !rayon ||
        !lc || !hc || !ng || !epsb || !sldr || !nmat || !file_id ||
        !group_iddip || !infostr || !nstop || *nnnr <= 0 || *nmax <= 0 ||
        (*niso != 1 && *niso != 3)) {
        object_error(infostr, infostr_length, nstop,
                  "Invalid inhomogeneous sphere arguments");
        return;
    }
    if (*na == 1) {
        *nstop = 0;
        std::memset(infostr, ' ', infostr_length);
        *nbsphere = 0;
        *ndipole = 0;
        if (*lc <= CU_REAL(0) || *hc <= CU_REAL(0) ||
            *rayon <= CU_REAL(1e-12)) {
            object_error(infostr, infostr_length, nstop,
                      "invalid inhomogeneous sphere material or radius");
            return;
        }
        constexpr CU_REAL nanometre = CU_REAL(1e-9);
        *rayon *= nanometre;
        *lc *= nanometre;
        *nx = *nnnr;
        *ny = *nnnr;
        *nz = *nnnr;
        CU_REAL object_size = CU_REAL(2) * *rayon;
        CU_REAL lpml = CU_REAL(0);
        int nrbc = 0;
        int npml = 0;
        initnrbc_(k0, lcos, lnrbc, &lpml, &object_size, nnnr,
                  &nrbc, &npml, pmlwrk, aretecube);
        if (*aretecube <= CU_REAL(0) || *nmax < 0 ||
            *nnnr > std::numeric_limits<int>::max() / *nnnr ||
            *nnnr * *nnnr > std::numeric_limits<int>::max() / *nnnr) {
            object_error(infostr, infostr_length, nstop,
                      "invalid PML sphere grid dimensions");
            return;
        }
        const int count = *nnnr * *nnnr * *nnnr;
        if (count > *nmax) {
            object_error(infostr, infostr_length, nstop,
                      "nmax parameter too small: increase nxm nym nzm");
            return;
        }
        // Match the reference FFT domain: all interior lattice samples must
        // be addressable, including the last fractional-radius interval.
        const int nnx = static_cast<int>(std::ceil(CU_REAL(2) * *rayon / *aretecube));
        if (nnx <= 0 || std::size_t(nnx) * nnx * nnx > std::size_t(*nmax)) {
            object_error(infostr, infostr_length, nstop,
                      "invalid PML sphere material dimensions");
            return;
        }
        const int material_count = nnx * nnx * nnx;
        const CU_REAL base_epsilon_real = eps->real();
        const CU_REAL base_epsilon_imag = eps->imag();
        const CU_REAL background_real = eps0->real();
        const CU_REAL background_imag = eps0->imag();
        const std::size_t tensor_count = std::size_t(*nmax) *
                                         std::size_t(*niso) * std::size_t(*niso);
        std::fill_n(tabdip, *nmax, 0);
        std::fill_n(tabmulti, *nmax, 0);
        std::fill_n(epsilon, tensor_count, COMPLEX{});
        std::fill_n(polarisa, tensor_count, COMPLEX{});
        if (!initialize_cuda_device()) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML sphere device initialization failed");
            return;
        }
        cuda_data::PmlDeviceBuffers device_buffers{};
        if (!cuda_data::get_device_pml_buffers(static_cast<std::size_t>(count),
                &device_buffers, static_cast<std::size_t>(material_count))) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML sphere buffer allocation failed");
            return;
        }
        if (!cuda_utils::check_cuda(cube_geometry_wrapper(
                device_buffers.xs, device_buffers.ys, device_buffers.zs,
                device_buffers.inside_map, *nnnr, object_size, *aretecube,
                CU_REAL(0), CU_REAL(0), CU_REAL(0)),
                "PML sphere geometry", "launch coordinate generation")) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML sphere geometry generation failed");
            return;
        }
        std::vector<CU_REAL> phases(static_cast<std::size_t>(material_count), CU_REAL(0));
        std::vector<CU_REAL> material(static_cast<std::size_t>(material_count), CU_REAL(0));
        random_bridge_init(*ng);
        for (CU_REAL &phase : phases)
            phase = random_bridge_next();
        CU_REAL mean = CU_REAL(0);
        CU_REAL deviation = CU_REAL(0);
        if (ifdda_correlated_material_field_cuda(nnx, nnx, nnx, *aretecube,
                *lc, *hc, base_epsilon_real, base_epsilon_imag,
                phases.data(), material.data(), &mean, &deviation) != 0) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML sphere material generation failed");
            return;
        }
        if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                xs, device_buffers.xs, std::size_t(count) * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Geometry),
                "PML sphere geometry", "download x coordinates") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                ys, device_buffers.ys, std::size_t(count) * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Geometry),
                "PML sphere geometry", "download y coordinates") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                zs, device_buffers.zs, std::size_t(count) * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Geometry),
                "PML sphere geometry", "download z coordinates") ||
            !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                device_buffers.material, material.data(),
                std::size_t(material_count) * sizeof(CU_REAL),
                cuda_utils::TransferCategory::Material),
                "PML sphere material", "upload correlated material")) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML sphere data transfer failed");
            return;
        }
        CU_REAL xmin = std::numeric_limits<CU_REAL>::max();
        const CU_REAL half_size = object_size / CU_REAL(2);
        for (int ix = 0; ix < *nnnr; ++ix) {
            const CU_REAL x = -half_size + *aretecube * (CU_REAL(ix) + CU_REAL(0.5));
            if (std::abs(x) < *rayon)
                xmin = std::min(xmin, x);
        }
        for (int ix = 0; ix < *nnnr; ++ix) {
            const CU_REAL x = -half_size + *aretecube * (CU_REAL(ix) + CU_REAL(0.5));
            if (std::abs(x) < *rayon) {
                const int field_index = static_cast<int>(std::nearbyint((x - xmin) / *aretecube)) + 1;
                if (field_index < 1 || field_index > nnx) {
                    object_error(infostr, infostr_length, nstop,
                              "PML sphere interior index outside material FFT domain");
                    return;
                }
            }
        }
        const CU_REAL contrast_scale = (*aretecube * *aretecube * *aretecube) /
            (CU_REAL(4) * CU_REAL(3.1415926535897932384626433832795));
        // Fortran objectinhomo.f90 supplies epsv=1 to abcchi outside the
        // sphere; the input epsilon is used only for the random interior.
        const COMPLEX epsilon_value(CU_REAL(1), CU_REAL(0));
        const COMPLEX epsilon_background(background_real, background_imag);
        if (!cuda_utils::check_cuda(pml_sphere_material_profile_wrapper(
                device_buffers.xs, device_buffers.ys, device_buffers.zs,
                device_buffers.material, device_buffers.epsilon_output,
                device_buffers.contrast_output, count, *nx, *ny, *nz, nnx,
                *rayon, xmin, *aretecube, base_epsilon_imag,
                cuda_utils::device_value(epsilon_value),
                cuda_utils::device_value(epsilon_background),
                contrast_scale, *ntypeabc, nrbc),
                "PML sphere material", "generate epsilon and contrast")) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML sphere material profile failed");
            return;
        }
        std::vector<COMPLEX> epsilon_values(static_cast<std::size_t>(count));
        std::vector<COMPLEX> contrast_values(static_cast<std::size_t>(count));
        if (!cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                epsilon_values.data(), device_buffers.epsilon_output,
                std::size_t(count),
                cuda_utils::TransferCategory::Material),
                "PML sphere material", "download epsilon values") ||
            !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                contrast_values.data(), device_buffers.contrast_output,
                std::size_t(count),
                cuda_utils::TransferCategory::Material),
                "PML sphere material", "download contrast values")) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA PML sphere result transfer failed");
            return;
        }
        for (int p = 0; p < count; ++p) {
            const CU_REAL x = xs[p];
            const CU_REAL y = ys[p];
            const CU_REAL z = zs[p];
            const COMPLEX local_epsilon = epsilon_values[std::size_t(p)];
            const COMPLEX local_contrast = contrast_values[std::size_t(p)];
            const bool inside = std::sqrt(x * x + y * y + z * z) < *rayon;
            if (inside) {
                *eps = local_epsilon;
                epsb[p] = local_epsilon;
            }
            epsilon[std::size_t(p)] = local_epsilon;
            polarisa[std::size_t(p)] = local_contrast;
        }
        std::vector<int> pml_cpu_map;
        if (!ifdda_object_all_cuda(geometry, size_t(count), cudaSuccess,
                pml_cpu_map, device_buffers.inside_map, true) ||
            !object_all_build_cpu_dipoles(geometry, pml_cpu_map, *nmax,
                false, xs, ys, zs, tabdip, nullptr, xs, ys, zs,
                nbsphere, ndipole, 1, true) ||
            !retain_inhomogeneous_materials(geometry,tabdip,size_t(count),*nmax,
                *nbsphere,*niso,device_buffers.inside_map,
                device_buffers.epsilon_output,device_buffers.contrast_output,
                epsilon,polarisa)) {
            object_error(infostr,infostr_length,nstop,
                         "Cannot retain compact PML sphere materials");
            return;
        }
        if (*nmat == 0 && !write_rectangular_mesh(*nnnr, *nnnr, *nnnr, count,
                xs, ys, zs, xs, ys, zs)) {
            object_error(infostr, infostr_length, nstop,
                      "Unable to write inhomogeneous PML sphere mesh files");
            return;
        }
        if (*nmat == 2) {
            const int dimensions[2] = {count, *nmax};
            hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dimensions, 17);
            hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dimensions, 17);
            hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dimensions, 17);
        }
        std::printf("OBJECT SPHERE INHOMOGENEOUS PML FINISHED\n");
        std::printf("SEED %d Average %.17g Standard deviation %.17g\n",
                    *ng, static_cast<double>(mean), static_cast<double>(deviation));
        (void)file_id;
        return;
    }
    if (*na != -1 && *na != 0) {
        object_error(infostr, infostr_length, nstop,
                  "na should be equal to -1 or 0 or 1");
        return;
    }
    *nstop = 0;
    std::memset(infostr, ' ', infostr_length);
    *nbsphere = 0;
    *ndipole = 0;
    if (*lc <= CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "coherence length equal to zero");
        return;
    }
    if (*hc <= CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "standard deviation equal to zero");
        return;
    }
    if (*rayon <= CU_REAL(1e-12)) {
        object_error(infostr, infostr_length, nstop,
                  "object sphere: radius=0!");
        return;
    }
    if (*nnnr > std::numeric_limits<int>::max() / *nnnr ||
        *nnnr * *nnnr > std::numeric_limits<int>::max() / *nnnr) {
        object_error(infostr, infostr_length, nstop,
                  "object sphere: dimensions exceed integer range");
        return;
    }

    constexpr CU_REAL nanometre = CU_REAL(1e-9);
    *rayon *= nanometre;
    *lc *= nanometre;
    *nx = *nnnr;
    *ny = *nnnr;
    *nz = *nnnr;
    const int count = *nnnr * *nnnr * *nnnr;
    if (count > *nmax) {
        object_error(infostr, infostr_length, nstop,
                  "nmax parameter too small: increase nxm nym nzm");
        return;
    }
    *aretecube = CU_REAL(2) * *rayon / CU_REAL(*nnnr);
    const std::size_t tensor_count = std::size_t(*nmax) *
                                     std::size_t(*niso) * std::size_t(*niso);
    std::fill_n(tabdip, *nmax, 0);
    std::fill_n(epsilon, tensor_count, COMPLEX{});
    std::fill_n(polarisa, tensor_count, COMPLEX{});

    std::vector<CU_REAL> mesh_x(static_cast<size_t>(count));
    std::vector<CU_REAL> mesh_y(static_cast<size_t>(count));
    std::vector<CU_REAL> mesh_z(static_cast<size_t>(count));
    CU_INT* device_map = cuda_data::get_dipole_map(size_t(count));
    std::vector<int> inside_map;
    if (!device_map || !ifdda_object_all_cuda(geometry, size_t(count),
            sphere_dipole_map_wrapper(device_map,*nnnr,*rayon,*aretecube),
            inside_map)) {
        object_error(infostr,infostr_length,nstop,"CUDA inhomogeneous sphere geometry generation failed");
        return;
    }
    for (int p=0;p<count;++p) {
        tabdip[p] = inside_map[size_t(p)] ? p+1 : 0;
        const int k=p%*nnnr,j=(p / *nnnr)%*nnnr,i=p/(*nnnr * *nnnr);
        mesh_x[size_t(p)] = -*rayon + *aretecube*(CU_REAL(k)+CU_REAL(0.5));
        mesh_y[size_t(p)] = -*rayon + *aretecube*(CU_REAL(j)+CU_REAL(0.5));
        mesh_z[size_t(p)] = -*rayon + *aretecube*(CU_REAL(i)+CU_REAL(0.5));
    }

    std::vector<CU_REAL> phases(static_cast<std::size_t>(count));
    std::vector<CU_REAL> material_real(static_cast<std::size_t>(count));
    random_bridge_init(*ng);
    for (CU_REAL &phase : phases)
        phase = random_bridge_next();
    CU_REAL mean = CU_REAL(0);
    CU_REAL standard_deviation = CU_REAL(0);
    if (ifdda_correlated_material_field_cuda(
            *nnnr, *nnnr, *nnnr, *aretecube, *lc, *hc, eps->real(),
            eps->imag(), phases.data(), material_real.data(), &mean,
            &standard_deviation) != 0) {
        object_error(infostr, infostr_length, nstop,
                  "CUDA correlated sphere material generation failed");
        return;
    }
    for (int p = 0; p < count; ++p)
        epsb[p] = COMPLEX(material_real[std::size_t(p)], eps->imag());
    std::printf(" average espilon   : %.17g\n standard deviation: %.17g\n",
                static_cast<double>(mean),
                static_cast<double>(standard_deviation));

    const int discretized = 1;
    const int method_code = object_polarizability_method_code(methode);
    const bool use_gpu_scatter = method_code != 0 && *niso == 1;
    std::vector<COMPLEX> batch_epsilon(static_cast<std::size_t>(count));
    std::vector<COMPLEX> batch_alpha(static_cast<std::size_t>(count));
    for (int p = 0; p < count; ++p)
        batch_epsilon[std::size_t(p)] = COMPLEX(
            material_real[std::size_t(p)], eps->imag());
    cuda_data::PolarizabilityBatchDeviceBuffers buffers{};
    if (method_code != 0) {
        if (!cuda_data::get_device_polarizability_batch_buffers(
                std::size_t(count), &buffers) ||
            !cuda_utils::check_cuda(cuda_data::copy_host_complex_to_device(
                buffers.epsilon, batch_epsilon.data(), std::size_t(count),
                cuda_utils::TransferCategory::Material),
                "inhomogeneous sphere polarizability", "upload epsilon batch") ||
            !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                buffers.inside, tabdip, std::size_t(count) * sizeof(int),
                cuda_utils::TransferCategory::Geometry),
                "inhomogeneous sphere polarizability", "upload inside map") ||
            !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                buffers.sldr, sldr, sizeof(CU_REAL),
                cuda_utils::TransferCategory::Material),
                "inhomogeneous sphere polarizability",
                "upload polarizability parameter") ||
            !cuda_utils::check_cuda(scalar_polarizability_batch_wrapper(
                buffers.epsilon, buffers.inside, buffers.values, buffers.sldr,
                count, *aretecube,
                cuda_utils::device_value(*eps0), *k0,
                discretized, method_code, *sldr),
                "inhomogeneous sphere polarizability",
                "compute alpha batch on GPU") ||
            (!use_gpu_scatter && !cuda_utils::check_cuda(
                cuda_data::copy_device_complex_to_host(
                    batch_alpha.data(), buffers.values, std::size_t(count),
                    cuda_utils::TransferCategory::Material),
                "inhomogeneous sphere polarizability", "download alpha batch")) ||
            (method_code == 6 && !cuda_utils::check_cuda(
                cuda_data::copy_device_to_host(
                    sldr, buffers.sldr, sizeof(CU_REAL),
                    cuda_utils::TransferCategory::Material),
                "inhomogeneous sphere polarizability",
                "download polarizability parameter"))) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA inhomogeneous sphere polarizability batch failed");
            return;
        }
    }
    if (!object_all_build_cpu_dipoles(geometry, inside_map, *nmax,
             *na == -1, mesh_x.data(), mesh_y.data(), mesh_z.data(),
             tabdip, nullptr, xs, ys, zs, nbsphere, ndipole)) {
        object_error(infostr, infostr_length, nstop,
                     "CUDA inhomogeneous sphere CPU map invalid");
        return;
    }
    for (int p = 0; p < count; ++p) {
        if (!inside_map[size_t(p)]) continue;
        *eps = batch_epsilon[std::size_t(p)];
        if (!use_gpu_scatter) {
            const size_t destination = size_t(tabdip[p] - 1);
            epsilon[destination] = *eps;
            polarisa[destination] = batch_alpha[std::size_t(p)];
        }
    }
    if (*na == 0) *nbsphere = count;
    if (!retain_inhomogeneous_materials(geometry,tabdip,size_t(count),*nmax,
            *nbsphere,*niso,
            use_gpu_scatter ? buffers.inside : nullptr,
            use_gpu_scatter ? buffers.epsilon : nullptr,
            use_gpu_scatter ? buffers.values : nullptr,
            epsilon,polarisa)) {
        object_error(infostr,infostr_length,nstop,
                     "Cannot retain compact inhomogeneous sphere materials");
        return;
    }
    if (*nmat == 0 &&
        !write_rectangular_mesh(*nnnr, *nnnr, *nnnr, *nbsphere, mesh_x.data(), mesh_y.data(),
                              mesh_z.data(), xs, ys, zs)) {
        object_error(infostr, infostr_length, nstop,
                  "Unable to write inhomogeneous sphere mesh files");
        return;
    }
    if (*nmat == 2) {
        const int dimensions[2] = {*nbsphere, *nmax};
        hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dimensions, 17);
    }
    std::printf("OBJECT SPHERE INHOMOGENEOUS FINISHED\n");
    std::printf("SEED %d Average %.17g Standard deviation %.17g\n", *ng,
                static_cast<double>(mean),
                static_cast<double>(standard_deviation));
    (void)lnrbc;
    (void)lcos;
    (void)ntypeabc;
    (void)tabmulti;
    (void)pmlwrk;
    (void)file_id;
}

extern "C" void ifdda_object_all_inhomocuboid2_cuda(
    int geometry, 
    COMPLEX *eps, COMPLEX *eps0, CU_REAL *xs, CU_REAL *ys,
    CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube, int *tabdip,
    const int *nnnr, const int *nmax, int *nbsphere, int *ndipole,
    int *nx, int *ny, int *nz, const int *nxm, const int *nym,
    const int *nzm, const int *nxmp, const int *nymp, const int *nzmp,
    const char *methode, COMPLEX *epsilon, COMPLEX *polarisa,
    const int *niso, CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez,
    CU_REAL *xg, CU_REAL *yg, CU_REAL *zg, CU_REAL *lc, const CU_REAL *hc,
    const int *ng, COMPLEX *epsb, const int *na, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip,
    char *infostr, int *nstop,
    std::size_t methode_length, std::size_t infostr_length) {
    if (geometry != int(GeometryId::InhomoCuboid2)) { object_error(infostr, infostr_length,nstop,"Unsupported geometry"); return; }
    if (!eps || !eps0 || !xs || !ys || !zs || !k0 || !aretecube || !tabdip ||
        !nnnr || !nmax || !nbsphere || !ndipole || !nx || !ny || !nz ||
        !nxm || !nym || !nzm || !nxmp || !nymp || !nzmp || !methode ||
        !epsilon || !polarisa || !niso || !sidex || !sidey || !sidez ||
        !xg || !yg || !zg || !lc || !hc || !ng || !epsb || !na || !sldr ||
        !nmat || !file_id || !group_iddip || !infostr || !nstop ||
        *nmax <= 0 || *nnnr < 0 || *niso <= 0 ||
        (*niso != 1 && *niso != 3)) {
        object_error(infostr, infostr_length, nstop,
                  "Invalid inhomogeneous cuboid arguments");
        return;
    }

    *nstop = 0;
    std::memset(infostr, ' ', infostr_length);
    *nbsphere = 0;
    *ndipole = 0;
    if (*lc <= CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "coherence length equal to zero");
        return;
    }
    if (*hc <= CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "standard deviation equal to zero");
        return;
    }

    const int nx_value = *nxm - 2 * *nxmp;
    const int ny_value = *nym - 2 * *nymp;
    const int nz_value = *nzm - 2 * *nzmp;
    if (nx_value <= 0 || ny_value <= 0 || nz_value <= 0 ||
        nx_value > std::numeric_limits<int>::max() / ny_value ||
        nx_value * ny_value > std::numeric_limits<int>::max() / nz_value) {
        object_error(infostr, infostr_length, nstop,
                  "Invalid inhomogeneous cuboid dimensions");
        return;
    }
    const int count = nx_value * ny_value * nz_value;
    if (count > *nmax) {
        object_error(infostr, infostr_length, nstop,
                  "nmax parameter too small: increase nxm nym nzm");
        return;
    }

    constexpr CU_REAL nanometre = CU_REAL(1e-9);
    *nx = nx_value;
    *ny = ny_value;
    *nz = nz_value;
    *aretecube *= nanometre;
    *sidex = *aretecube * CU_REAL(nx_value);
    *sidey = *aretecube * CU_REAL(ny_value);
    *sidez = *aretecube * CU_REAL(nz_value);
    *xg *= nanometre;
    *yg *= nanometre;
    *zg *= nanometre;
    *lc *= nanometre;

    if (*sidex == CU_REAL(0) || *sidey == CU_REAL(0) ||
        *sidez == CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "object cuboid: zero side length");
        return;
    }

    const std::size_t tensor_count = std::size_t(*nmax) *
                                     std::size_t(*niso) * std::size_t(*niso);
    std::fill_n(tabdip, *nmax, 0);
    std::fill_n(epsilon, tensor_count, COMPLEX{});
    std::fill_n(polarisa, tensor_count, COMPLEX{});

    if (*nmat == 0) {
        for (const char *path : {"x.mat", "y.mat", "z.mat", "xc.mat",
                                 "yc.mat", "zc.mat"}) {
            FILE *file = std::fopen(path, "w");
            if (!file) {
                object_error(infostr, infostr_length, nstop,
                          "Unable to open inhomogeneous cuboid output files");
                return;
            }
            std::fclose(file);
        }
    }

    CU_INT* d_map = cuda_data::get_dipole_map(size_t(count));
    std::vector<int> inside;
    if (!d_map || !ifdda_object_all_cuda(geometry, size_t(count),
            full_grid_dipole_map_wrapper(d_map, count), inside)) {
        object_error(infostr, infostr_length, nstop,
                     "CUDA inhomogeneous cuboid geometry generation failed");
        return;
    }
    for (int p = 0; p < count; ++p) {
        const int k = p % nx_value;
        const int j = (p / nx_value) % ny_value;
        const int i = p / (nx_value*ny_value);
        xs[p] = *aretecube*(CU_REAL(k)+CU_REAL(0.5)) - *sidex/CU_REAL(2);
        ys[p] = *aretecube*(CU_REAL(j)+CU_REAL(0.5)) - *sidey/CU_REAL(2);
        zs[p] = *aretecube*(CU_REAL(i)+CU_REAL(0.5)) - *sidez/CU_REAL(2);
    }
    if (!object_all_build_cpu_dipoles(geometry, inside, *nmax, false,
            xs, ys, zs, tabdip, nullptr, xs, ys, zs,
            nbsphere, ndipole, 1, true)) {
        object_error(infostr, infostr_length, nstop,
                     "CUDA inhomogeneous cuboid CPU map invalid");
        return;
    }

    std::vector<CU_REAL> phases(static_cast<std::size_t>(count), CU_REAL(0));
    std::vector<CU_REAL> material_real(static_cast<std::size_t>(count), CU_REAL(0));
    random_bridge_init(*ng);
    for (CU_REAL &phase : phases)
        phase = random_bridge_next();

    CU_REAL mean = CU_REAL(0);
    CU_REAL standard_deviation = CU_REAL(0);
    const int material_status = ifdda_correlated_material_field_cuda(
        nx_value, ny_value, nz_value, *aretecube, *lc, *hc, eps->real(),
        eps->imag(), phases.data(), material_real.data(), &mean,
        &standard_deviation);
    if (material_status != 0) {
        object_error(infostr, infostr_length, nstop,
                  "CUDA correlated cuboid material generation failed");
        return;
    }
    std::printf(" average espilon   : %.17g\n standard deviation: %.17g\n",
                static_cast<double>(mean),
                static_cast<double>(standard_deviation));

    const int discretized = 1;
    const int method_code = object_polarizability_method_code(methode);
    const bool use_gpu_scatter = method_code != 0 && *niso == 1;
    std::vector<COMPLEX> batch_epsilon(static_cast<std::size_t>(count));
    std::vector<COMPLEX> batch_alpha(static_cast<std::size_t>(count));
    for (int p = 0; p < count; ++p) {
        batch_epsilon[std::size_t(p)] = COMPLEX(
            material_real[std::size_t(p)], eps->imag());
        epsb[p] = batch_epsilon[std::size_t(p)];
    }
    cuda_data::PolarizabilityBatchDeviceBuffers buffers{};
    if (method_code != 0) {
        if (!cuda_data::get_device_polarizability_batch_buffers(
                std::size_t(count), &buffers) ||
            !cuda_utils::check_cuda(cuda_data::copy_host_complex_to_device(
                buffers.epsilon, batch_epsilon.data(), std::size_t(count),
                cuda_utils::TransferCategory::Material),
                "inhomogeneous cuboid polarizability",
                "upload epsilon batch") ||
            !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                buffers.inside, inside.data(), std::size_t(count) * sizeof(int),
                cuda_utils::TransferCategory::Geometry),
                "inhomogeneous cuboid polarizability",
                "upload inside mask") ||
            !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                buffers.sldr, sldr, sizeof(CU_REAL),
                cuda_utils::TransferCategory::Material),
                "inhomogeneous cuboid polarizability",
                "upload polarizability parameter") ||
            !cuda_utils::check_cuda(scalar_polarizability_batch_wrapper(
                buffers.epsilon, buffers.inside, buffers.values, buffers.sldr,
                count, *aretecube,
                cuda_utils::device_value(*eps0), *k0,
                discretized, method_code, *sldr),
                "inhomogeneous cuboid polarizability",
                "compute alpha batch on GPU") ||
            (!use_gpu_scatter && !cuda_utils::check_cuda(
                cuda_data::copy_device_complex_to_host(
                    batch_alpha.data(), buffers.values, std::size_t(count),
                    cuda_utils::TransferCategory::Material),
                "inhomogeneous cuboid polarizability",
                "download alpha batch")) ||
            (method_code == 6 && !cuda_utils::check_cuda(
                cuda_data::copy_device_to_host(
                    sldr, buffers.sldr, sizeof(CU_REAL),
                    cuda_utils::TransferCategory::Material),
                "inhomogeneous cuboid polarizability",
                "download polarizability parameter"))) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA inhomogeneous cuboid polarizability batch failed");
            return;
        }
    }
    std::vector<CU_REAL> field_x(static_cast<std::size_t>(count), CU_REAL(0));
    std::vector<CU_REAL> field_y(static_cast<std::size_t>(count), CU_REAL(0));
    std::vector<CU_REAL> field_z(static_cast<std::size_t>(count), CU_REAL(0));
    for (int i = 0; i < nz_value; ++i) {
        for (int j = 0; j < ny_value; ++j) {
            for (int k = 0; k < nx_value; ++k) {
                const int p = k + nx_value * (j + ny_value * i);
                const CU_REAL x = xs[p] + *xg;
                const CU_REAL y = ys[p] + *yg;
                const CU_REAL z = zs[p] + *zg;
                xs[p] = x;
                ys[p] = y;
                zs[p] = z;
                field_x[std::size_t(p)] = x;
                field_y[std::size_t(p)] = y;
                field_z[std::size_t(p)] = z;
                *eps = batch_epsilon[std::size_t(p)];
                if (!use_gpu_scatter) {
                    epsilon[std::size_t(p)] = *eps;
                    polarisa[std::size_t(p)] = batch_alpha[std::size_t(p)];
                }
            }
        }
    }

    // The full-grid batch is already compact and ordered like tabdip.
    // The common stage now publishes its GPU arrays and CPU output together,
    // avoiding a separate scratch allocation and a second scatter kernel.
    if (!publish_compact_materials(geometry, size_t(count), *nmax, count,
            *niso, polarisa, epsilon,
            use_gpu_scatter ? buffers.values : nullptr,
            use_gpu_scatter ? buffers.epsilon : nullptr)) {
        object_error(infostr,infostr_length,nstop,
                     "Cannot retain compact cuboid materials");
        return;
    }

    if (*nmat == 0 &&
        !write_rectangular_mesh(nx_value, ny_value, nz_value, count,
                         field_x.data(), field_y.data(), field_z.data(),
                         xs, ys, zs)) {
        object_error(infostr, infostr_length, nstop,
                  "Unable to write inhomogeneous cuboid mesh files");
        return;
    }
    if (*nmat == 2) {
        const int dimensions[2] = {count, *nmax};
        hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dimensions, 17);
    }
    std::printf(" Cuboid inhomogeneous::nbsphere= %d\n", *nbsphere);
}

extern "C" void ifdda_object_all_randommulti_cuda(
    int geometry, 
    const char *trope, COMPLEX *epss, COMPLEX *epsanis,
    const int *numbersphere, const int *numberspheremax, CU_REAL *xg,
    CU_REAL *yg, CU_REAL *zg, CU_REAL *rayons,
    const COMPLEX *eps0, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, const int *nnnr,
    const int *nmax, int *nbsphere, int *ndipole, int *nx, int *ny, int *nz,
    const int *nxm, const int *nym, const int *nzm, const char *methode,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso,
    CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez, CU_REAL *xsphe,
    CU_REAL *ysphe, CU_REAL *zsphe, CU_REAL *density, const int *nseed,
    int *na, CU_REAL *sldr, const int *nmat, const int64_t *file_id,
    const int64_t *group_iddip, char *infostr, int *nstop,
    std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length) {
    if (geometry != int(GeometryId::RandomMulti)) { object_error(infostr,infostr_length,nstop,"Unsupported geometry"); return; }
    if (!trope || !epss || !epsanis || !numbersphere ||
        !numberspheremax || !xg || !yg || !zg || !rayons || !eps0 ||
        !xs || !ys || !zs || !k0 || !aretecube || !tabdip || !nnnr ||
        !nmax || !nbsphere || !ndipole || !nx || !ny || !nz || !nxm ||
        !nym || !nzm || !methode || !epsilon || !polarisa || !niso ||
        !sidex || !sidey || !sidez || !xsphe || !ysphe || !zsphe ||
        !density || !nseed || !na || !sldr || !nmat || !file_id ||
        !group_iddip || !infostr || !nstop || *numbersphere <= 0 ||
        *numbersphere > *numberspheremax || *numbersphere > 20 ||
        *nnnr <= 0 || *nmax <= 0 || (*niso != 1 && *niso != 3)) {
        object_error(infostr, infostr_length, nstop,
                  "Invalid random multi-sphere arguments");
        return;
    }

    *nstop = 0;
    *nbsphere = 0;
    *ndipole = 0;
    std::memset(infostr, ' ', infostr_length);
    const std::size_t tensor_count = std::size_t(*nmax) *
                                    std::size_t(*niso) * std::size_t(*niso);
    std::fill_n(tabdip, *nmax, 0);
    std::fill_n(epsilon, tensor_count, COMPLEX{});
    std::fill_n(polarisa, tensor_count, COMPLEX{});

    random_bridge_init(*nseed);
    constexpr CU_REAL nanometre = CU_REAL(1e-9);
    constexpr CU_REAL pi = CU_REAL(3.141592653589793238462643383279502884L);
    *sidex *= nanometre;
    *sidey *= nanometre;
    *sidez *= nanometre;
    for (int sphere = 0; sphere < *numbersphere; ++sphere) {
        xg[sphere] *= nanometre;
        yg[sphere] *= nanometre;
        zg[sphere] *= nanometre;
        rayons[sphere] *= nanometre;
        if (rayons[sphere] >= *sidex || rayons[sphere] >= *sidey ||
            rayons[sphere] >= *sidez) {
            object_error(infostr, infostr_length, nstop,
                      "Radius larger than the side of the box");
            return;
        }
    }

    auto swap_value = [](auto *values, int first, int last) {
        std::swap(values[first], values[last]);
    };
    const int last_sphere = *numbersphere - 1;
    swap_value(rayons, 0, last_sphere);
    swap_value(xg, 0, last_sphere);
    swap_value(yg, 0, last_sphere);
    swap_value(zg, 0, last_sphere);
    swap_value(epss, 0, last_sphere);
    const CU_REAL random_radius = rayons[0];

    CU_REAL xmin = std::numeric_limits<CU_REAL>::max() / CU_REAL(100);
    CU_REAL ymin = xmin;
    CU_REAL zmin = xmin;
    CU_REAL xmax = -xmin;
    CU_REAL ymax = -xmin;
    CU_REAL zmax = -xmin;
    for (int sphere = 1; sphere < *numbersphere; ++sphere) {
        xmax = std::max(xmax, xg[sphere] + rayons[sphere]);
        xmin = std::min(xmin, xg[sphere] - rayons[sphere]);
        ymax = std::max(ymax, yg[sphere] + rayons[sphere]);
        ymin = std::min(ymin, yg[sphere] - rayons[sphere]);
        zmax = std::max(zmax, zg[sphere] + rayons[sphere]);
        zmin = std::min(zmin, zg[sphere] - rayons[sphere]);
        if (rayons[sphere] == CU_REAL(0)) {
            object_error(infostr, infostr_length, nstop,
                      "object nspheres: radius=0!");
            return;
        }
    }
    if (xmax > xg[0] + *sidex / CU_REAL(2)) {
        object_error(infostr, infostr_length, nstop, "Spheres not in the box x1");
        return;
    }
    if (xmin < xg[0] - *sidex / CU_REAL(2)) {
        object_error(infostr, infostr_length, nstop, "Spheres not in the box x2");
        return;
    }
    if (ymax > yg[0] + *sidey / CU_REAL(2)) {
        object_error(infostr, infostr_length, nstop, "Spheres not in the box y1");
        return;
    }
    if (ymin < yg[0] - *sidey / CU_REAL(2)) {
        object_error(infostr, infostr_length, nstop, "Spheres not in the box y2");
        return;
    }
    if (zmax > zg[0] + *sidez / CU_REAL(2)) {
        object_error(infostr, infostr_length, nstop, "Spheres not in the box z1");
        return;
    }
    if (zmin < zg[0] - *sidez / CU_REAL(2)) {
        object_error(infostr, infostr_length, nstop, "Spheres not in the box z2");
        return;
    }

    const CU_REAL side = std::max({*sidex, *sidey, *sidez});
    *aretecube = side / CU_REAL(*nnnr);
    *nx = int(*sidex / *aretecube);
    *ny = int(*sidey / *aretecube);
    *nz = int(*sidez / *aretecube);
    for (int sphere = 0; sphere < *numbersphere; ++sphere) {
        if (rayons[sphere] < *aretecube) {
            object_error(infostr, infostr_length, nstop,
                      "Radius smaller than the meshsize");
            return;
        }
    }
    if (*sidex == CU_REAL(0) || *sidey == CU_REAL(0) ||
        *sidez == CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "object cuboid: zero side length");
        return;
    }
    if (*nx <= 0 || *ny <= 0 || *nz <= 0 || *nx > *nxm || *ny > *nym ||
        *nz > *nzm || *nx > *nmax / *ny || *nx * *ny > *nmax / *nz) {
        object_error(infostr, infostr_length, nstop,
                  "Dimension Problem of the Box : Box too small!");
        return;
    }
    const int mesh_count = *nx * *ny * *nz;

    if (*density >= CU_REAL(0.5) && *density < CU_REAL(2)) {
        object_error(infostr, infostr_length, nstop, "density too large!");
        return;
    }
    if (*density <= CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop, "density negative!!");
        return;
    }
    const CU_REAL sphere_volume = CU_REAL(4) * pi / CU_REAL(3) *
        random_radius * random_radius * random_radius;
    CU_REAL fixed_volume = CU_REAL(0);
    for (int sphere = 1; sphere < *numbersphere; ++sphere)
        fixed_volume += CU_REAL(4) * pi / CU_REAL(3) *
                        rayons[sphere] * rayons[sphere] * rayons[sphere];
    int requested_random_spheres = int(
        *density * (*sidex * *sidey * *sidez - fixed_volume) / sphere_volume);
    std::printf("density %d %.17g %.17g %.17g %.17g %.17g %.17g\n",
                requested_random_spheres, double(*sidex), double(*sidey),
                double(*sidez), double(*sidex * *sidey * *sidez - fixed_volume),
                double(sphere_volume), double(fixed_volume));
    if (requested_random_spheres <= 1) {
        object_error(infostr, infostr_length, nstop, "density too small!");
        return;
    }
    if (*density > CU_REAL(2)) {
        requested_random_spheres = int(*density);
        *density = CU_REAL(requested_random_spheres) * sphere_volume /
                   *sidex / *sidey / *sidez;
    }
    if (requested_random_spheres > *nmax ||
        requested_random_spheres > std::numeric_limits<int>::max() / 10) {
        object_error(infostr, infostr_length, nstop,
                  "nmax parameter too small: increase nxm nym nzm");
        return;
    }

    std::vector<CU_REAL> random_x(static_cast<std::size_t>(requested_random_spheres));
    std::vector<CU_REAL> random_y(static_cast<std::size_t>(requested_random_spheres));
    std::vector<CU_REAL> random_z(static_cast<std::size_t>(requested_random_spheres));
    int random_count = 0;
    const int max_attempts = std::max(1000, requested_random_spheres * 10);
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        const CU_REAL x = -*sidex / CU_REAL(2) + random_radius +
            (*sidex - CU_REAL(2) * random_radius) * random_bridge_next();
        const CU_REAL y = -*sidey / CU_REAL(2) + random_radius +
            (*sidey - CU_REAL(2) * random_radius) * random_bridge_next();
        const CU_REAL z = -*sidez / CU_REAL(2) + random_radius +
            (*sidez - CU_REAL(2) * random_radius) * random_bridge_next();
        // Match the legacy Fortran routine: its EXIT statements leave the
        // validation loops but do not reject the candidate.
        random_x[std::size_t(random_count)] = x;
        random_y[std::size_t(random_count)] = y;
        random_z[std::size_t(random_count)] = z;
        if (++random_count == requested_random_spheres)
            break;
    }

    std::printf("Number of sphere %d %d %d %d %d\n", random_count,
                *na, *nx, *ny, *nz);
    const int fixed_sphere_count = *numbersphere - 1;
    const int sphere_count = fixed_sphere_count + random_count;
    std::vector<CU_REAL> sphere_parameters(std::size_t(sphere_count) * 4);
    for (int sphere = 0; sphere < fixed_sphere_count; ++sphere) {
        const std::size_t offset = std::size_t(sphere) * 4;
        sphere_parameters[offset] = xg[sphere + 1];
        sphere_parameters[offset + 1] = yg[sphere + 1];
        sphere_parameters[offset + 2] = zg[sphere + 1];
        sphere_parameters[offset + 3] = rayons[sphere + 1];
    }
    for (int sphere = 0; sphere < random_count; ++sphere) {
        const std::size_t offset = std::size_t(fixed_sphere_count + sphere) * 4;
        sphere_parameters[offset] = random_x[std::size_t(sphere)];
        sphere_parameters[offset + 1] = random_y[std::size_t(sphere)];
        sphere_parameters[offset + 2] = random_z[std::size_t(sphere)];
        sphere_parameters[offset + 3] = random_radius;
    }
    std::vector<CU_REAL> mesh_x(static_cast<size_t>(mesh_count));
    std::vector<CU_REAL> mesh_y(static_cast<size_t>(mesh_count));
    std::vector<CU_REAL> mesh_z(static_cast<size_t>(mesh_count));
    std::vector<int> sphere_map;
    CU_INT* device_map = cuda_data::get_dipole_map(size_t(mesh_count));
    CU_REAL* device_parameters = static_cast<CU_REAL*>(
        cuda_data::get_device_workspace_buffer_bytes(
            "object_all.randommulti.parameters", sphere_parameters.size()*sizeof(CU_REAL)));
    if (!device_map || !device_parameters ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(device_parameters, sphere_parameters.data(),
                sphere_parameters.size()*sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
            "object_all", "upload random multi-sphere parameters") ||
        !ifdda_object_all_cuda(geometry, size_t(mesh_count),
            multi_sphere_dipole_map_wrapper(device_map, *nx, *ny, *nz,
                *aretecube, -*sidex/CU_REAL(2), -*sidey/CU_REAL(2),
                -*sidez/CU_REAL(2), device_parameters, sphere_count,
                fixed_sphere_count), sphere_map)) {
        object_error(infostr, infostr_length, nstop, "CUDA random multi-sphere geometry failed");
        return;
    }
    for (int flat = 0; flat < mesh_count; ++flat) {
        const int k = flat % *nx;
        const int j = (flat / *nx) % *ny;
        const int i = flat / (*nx * *ny);
        mesh_x[size_t(flat)] = -*sidex/CU_REAL(2) + (CU_REAL(k)+CU_REAL(0.5))* *aretecube;
        mesh_y[size_t(flat)] = -*sidey/CU_REAL(2) + (CU_REAL(j)+CU_REAL(0.5))* *aretecube;
        mesh_z[size_t(flat)] = -*sidez/CU_REAL(2) + (CU_REAL(i)+CU_REAL(0.5))* *aretecube;
    }

    *na = 0;

    const bool isotropic = std::memcmp(trope, "iso", 3) == 0;
    const bool updates_sldr = methode[0] == 'L' && methode[1] == 'R';
    // The LR rule changes sldr at each occupied site; preserve that
    // stateful CPU path. Otherwise leave material assignment to CUDA.
    const bool use_gpu_materials = !isotropic || !updates_sldr;
    const int discretized = 1;
    const int invert_result = 1;
    const COMPLEX epsilon0_value = *eps0;
    std::vector<COMPLEX> scalar_alpha(
        static_cast<std::size_t>(*numbersphere));
    std::vector<COMPLEX> tensor_alpha(
        std::size_t(*numbersphere) * 9);
    if (isotropic && !updates_sldr) {
        for (int sphere = 0; sphere < *numbersphere; ++sphere) {
            const COMPLEX material = epss[sphere];
            poladiff_cuda(&*aretecube, &material, &epsilon0_value, k0,
                      &discretized, methode, sldr,
                      scalar_alpha.data() + sphere, methode_length);
        }
    } else if (!isotropic) {
        for (int sphere = 0; sphere < *numbersphere; ++sphere) {
            COMPLEX material_tensor[9];
            for (int component = 0; component < 9; ++component) {
                const COMPLEX value = epsanis[
                    std::size_t(sphere) * 9 + std::size_t(component)];
                material_tensor[component] = {value.real(), value.imag()};
            }
            polaepstens_cuda(&*aretecube, material_tensor, &epsilon0_value, k0,
                         &discretized, methode, &invert_result,
                         tensor_alpha.data() + std::size_t(sphere) * 9,
                         methode_length);
        }
    }

    std::vector<CU_REAL> output_mesh_x;
    std::vector<CU_REAL> output_mesh_y;
    std::vector<CU_REAL> output_mesh_z;
    if (*nmat == 0) {
        output_mesh_x.resize(std::size_t(mesh_count));
        output_mesh_y.resize(std::size_t(mesh_count));
        output_mesh_z.resize(std::size_t(mesh_count));
        for (int flat = 0; flat < mesh_count; ++flat) {
            output_mesh_x[std::size_t(flat)] = mesh_x[std::size_t(flat)] + xg[0];
            output_mesh_y[std::size_t(flat)] = mesh_y[std::size_t(flat)] + yg[0];
            output_mesh_z[std::size_t(flat)] = mesh_z[std::size_t(flat)] + zg[0];
        }
    }

    std::vector<int> material_labels;
    if (use_gpu_materials)
        material_labels.resize(std::size_t(*nmax), 0);

    if (!object_all_build_cpu_dipoles(geometry, sphere_map, *nmax, false,
             mesh_x.data(), mesh_y.data(), mesh_z.data(), tabdip, nullptr,
             xs, ys, zs, nbsphere, ndipole, sphere_count, false,
             xg[0], yg[0], zg[0])) {
        object_error(infostr, infostr_length, nstop,
                     "CUDA random multi-sphere CPU map invalid");
        return;
    }
    for (int flat = 0; flat < mesh_count; ++flat) {
        const int selected = sphere_map[std::size_t(flat)];
        if (selected == 0) continue;
        const int dipole = flat;
        const int material_index = selected <= fixed_sphere_count
            ? selected : 0;
        if (use_gpu_materials) {
            // Labels refer to the per-material lookup table, not to
            // individual random sphere instances.
            material_labels[std::size_t(dipole)] = material_index + 1;
            continue;
        }
        if (isotropic) {
            const COMPLEX material = epss[material_index];
            COMPLEX alpha =
                scalar_alpha[std::size_t(material_index)];
            if (updates_sldr) {
                const COMPLEX material_value = material;
                poladiff_cuda(&*aretecube, &material_value, &epsilon0_value, k0,
                          &discretized, methode, sldr, &alpha,
                          methode_length);
            }
            const std::size_t index = size_t(dipole);
            epsilon[index] = material;
            polarisa[index] = alpha;
        } else {
            for (int column = 0; column < 3; ++column) {
                for (int row = 0; row < 3; ++row) {
                    const int component = row + 3 * column;
                    const std::size_t source =
                        std::size_t(material_index) * 9 + component;
                    const std::size_t destination = size_t(dipole) + size_t(*nmax)*size_t(row + *niso*column);
                    epsilon[destination] = epsanis[source];
                    polarisa[destination] = tensor_alpha[source];
                }
            }
        }
    }

    if (use_gpu_materials) {
        const int components = isotropic ? 1 : 9;
        std::vector<COMPLEX> label_epsilon(
            std::size_t(*numbersphere) * std::size_t(components));
        std::vector<COMPLEX> label_alpha(label_epsilon.size());
        for (int sphere = 0; sphere < *numbersphere; ++sphere) {
            for (int component = 0; component < components; ++component) {
                const std::size_t lookup_index =
                    std::size_t(sphere) * std::size_t(components) +
                    std::size_t(component);
                const COMPLEX epsilon_value = isotropic
                    ? epss[sphere] : epsanis[lookup_index];
                const COMPLEX alpha_value = isotropic
                    ? scalar_alpha[std::size_t(sphere)]
                    : tensor_alpha[lookup_index];
                label_epsilon[lookup_index] = {
                    epsilon_value.real(), epsilon_value.imag()};
                label_alpha[lookup_index] = alpha_value;
            }
        }
        if (!fill_labeled_materials(geometry,
                tabdip, material_labels.data(), size_t(mesh_count), *nmax,
                *nbsphere, *niso, *numbersphere, false,
                label_epsilon.data(), label_alpha.data(),
                epsilon, polarisa)) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA random multi-sphere material initialization failed");
            return;
        }
    }

    if (!use_gpu_materials && *nbsphere > 0) {
        if (!publish_compact_materials(geometry, size_t(mesh_count), *nmax,
                *nbsphere, 1, polarisa, epsilon)) {
            object_error(infostr, infostr_length, nstop, "CUDA random multi LR publishing failed");
            return;
        }
    }

    if (*nmat == 0 &&
        !write_rectangular_mesh(*nx, *ny, *nz, *nbsphere,
                              output_mesh_x.data(), output_mesh_y.data(),
                              output_mesh_z.data(), xs, ys, zs)) {
        object_error(infostr, infostr_length, nstop,
                  "Unable to write random multi-sphere mesh files");
        return;
    }
    if (*nmat == 2) {
        const int dimensions[2] = {*nbsphere, *nmax};
        hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dimensions, 17);
    }
    std::printf("Random sphere::nbsphere    = %d %d\n", *nbsphere, *ndipole);
    std::printf("density of sphere obtained = %.17g\n",
                static_cast<double>(CU_REAL(random_count) * sphere_volume /
                                    *sidex / *sidey / *sidez));
    (void)trope_length;
    (void)file_id;
}

extern "C" void ifdda_object_all_randomsphere_cuda(
    int geometry, 
    const char *trope, const COMPLEX *eps,
    const COMPLEX *epsani, const COMPLEX *eps0, CU_REAL *xs,
    CU_REAL *ys, CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube,
    int *tabdip, const int *nnnr, const int *nmax, int *nbsphere,
    int *ndipole, int *nx, int *ny, int *nz, const int *nxm,
    const int *nym, const int *nzm, const char *methode,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso,
    CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez, CU_REAL *xg,
    CU_REAL *yg, CU_REAL *zg, CU_REAL *xsphe, CU_REAL *ysphe,
    CU_REAL *zsphe, CU_REAL *radius, CU_REAL *density, const int *nseed,
    const int *na, CU_REAL *sldr, const int *nmat, const int64_t *file_id,
    const int64_t *group_iddip, char *infostr, int *nstop,
    std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length) {
    if (geometry != int(GeometryId::RandomSphere1) && geometry != int(GeometryId::RandomSphere2)) { object_error(infostr, infostr_length, nstop, "Unsupported geometry"); return; }
    if (!trope || !eps || !epsani || !eps0 || !xs || !ys || !zs || !k0 ||
        !aretecube || !tabdip || !nnnr || !nmax || !nbsphere || !ndipole ||
        !nx || !ny || !nz || !nxm || !nym || !nzm || !methode || !epsilon ||
        !polarisa || !niso || !sidex || !sidey || !sidez || !xg || !yg ||
        !zg || !xsphe || !ysphe || !zsphe || !radius || !density || !nseed ||
        !na || !sldr || !nmat || !file_id || !group_iddip || !infostr ||
        !nstop || *nnnr <= 0 || *nmax <= 0 || *niso <= 0 ||
        (*niso != 1 && *niso != 3)) {
        object_error(infostr, infostr_length, nstop,
                  "Invalid random-sphere geometry arguments");
        return;
    }

    *nstop = 0;
    std::memset(infostr, ' ', infostr_length);
    *nbsphere = 0;
    *ndipole = 0;
    const std::size_t tensor_count = std::size_t(*nmax) *
                                     std::size_t(*niso) * std::size_t(*niso);
    std::fill_n(tabdip, *nmax, 0);

    constexpr CU_REAL nanometre = CU_REAL(1e-9);
    constexpr CU_REAL pi = CU_REAL(3.141592653589793238462643383279502884L);
    *sidex *= nanometre;
    *sidey *= nanometre;
    *sidez *= nanometre;
    *xg *= nanometre;
    *yg *= nanometre;
    *zg *= nanometre;
    *radius *= nanometre;

    if (*radius >= *sidex || *radius >= *sidey || *radius >= *sidez) {
        object_error(infostr, infostr_length, nstop,
                  "Radius larger than the side of the box");
        return;
    }
    if (*sidex == CU_REAL(0) || *sidey == CU_REAL(0) || *sidez == CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "object cuboid: zero side length");
        return;
    }

    const CU_REAL side = std::max({*sidex, *sidey, *sidez});
    *aretecube = side / CU_REAL(*nnnr);
    *nx = int(*sidex / *aretecube);
    *ny = int(*sidey / *aretecube);
    *nz = int(*sidez / *aretecube);
    if (*radius < *aretecube) {
        object_error(infostr, infostr_length, nstop,
                  "Radius smaller than the meshsize");
        return;
    }
    if (*nx <= 0 || *ny <= 0 || *nz <= 0 || *nx > *nxm || *ny > *nym ||
        *nz > *nzm || *nx > std::numeric_limits<int>::max() / *ny ||
        *nx * *ny > std::numeric_limits<int>::max() / *nz) {
        object_error(infostr, infostr_length, nstop,
                  "Dimension Problem of the Box : Box too small!");
        return;
    }
    const int mesh_count = *nx * *ny * *nz;
    if (mesh_count > *nmax) {
        object_error(infostr, infostr_length, nstop,
                  "nmax parameter too small: increase nxm nym nzm");
        return;
    }
    if (*density >= CU_REAL(0.5) && *density <= CU_REAL(2)) {
        object_error(infostr, infostr_length, nstop, "density too large!");
        return;
    }

    const CU_REAL sphere_volume = CU_REAL(4) * pi / CU_REAL(3) *
        *radius * *radius * *radius;
    int requested_spheres = int(*density * *sidex * *sidey * *sidez /
                                sphere_volume);
    if (requested_spheres <= 1) {
        object_error(infostr, infostr_length, nstop, "density too small!");
        return;
    }
    if (requested_spheres > *nmax || requested_spheres >
        std::numeric_limits<int>::max() / 10) {
        object_error(infostr, infostr_length, nstop,
                  "nmax parameter too small: increase nxm nym nzm");
        return;
    }
    if (*density > CU_REAL(2)) {
        requested_spheres = int(*density);
        *density = CU_REAL(requested_spheres) * sphere_volume /
                   *sidex / *sidey / *sidez;
    }
    if (*na != -1 && *na != 0) {
        object_error(infostr, infostr_length, nstop,
                  "na should be equal to -1 or 0");
        return;
    }

    random_bridge_init(*nseed);
    int sphere_count = 0;
    for (int attempt = 0; attempt < requested_spheres * 10; ++attempt) {
        const CU_REAL x = -*sidex / CU_REAL(2) + *radius +
            (*sidex - CU_REAL(2) * *radius) * random_bridge_next();
        const CU_REAL y = -*sidey / CU_REAL(2) + *radius +
            (*sidey - CU_REAL(2) * *radius) * random_bridge_next();
        const CU_REAL z = -*sidez / CU_REAL(2) + *radius +
            (*sidez - CU_REAL(2) * *radius) * random_bridge_next();
        bool intersects = false;
        for (int previous = 0; previous < sphere_count; ++previous) {
            const CU_REAL dx = x - xsphe[previous];
            const CU_REAL dy = y - ysphe[previous];
            const CU_REAL dz = z - zsphe[previous];
            if (std::sqrt(dx * dx + dy * dy + dz * dz) <
                CU_REAL(2) * *radius) {
                intersects = true;
                break;
            }
        }
        if (intersects)
            continue;
        xsphe[sphere_count] = x;
        ysphe[sphere_count] = y;
        zsphe[sphere_count] = z;
        if (++sphere_count == requested_spheres)
            break;
    }

    std::printf("Number of sphere %d %d %d %d %d\n", sphere_count,
                *na, *nx, *ny, *nz);
    std::vector<CU_REAL> sphere_parameters(std::size_t(sphere_count) * 4);
    for (int sphere = 0; sphere < sphere_count; ++sphere) {
        sphere_parameters[std::size_t(sphere) * 4] = xsphe[sphere];
        sphere_parameters[std::size_t(sphere) * 4 + 1] = ysphe[sphere];
        sphere_parameters[std::size_t(sphere) * 4 + 2] = zsphe[sphere];
        sphere_parameters[std::size_t(sphere) * 4 + 3] = *radius;
    }
    std::vector<CU_REAL> mesh_x(static_cast<size_t>(mesh_count));
    std::vector<CU_REAL> mesh_y(static_cast<size_t>(mesh_count));
    std::vector<CU_REAL> mesh_z(static_cast<size_t>(mesh_count));
    std::vector<int> sphere_map;
    CU_INT* device_map = cuda_data::get_dipole_map(size_t(mesh_count));
    CU_REAL* device_parameters = static_cast<CU_REAL*>(
        cuda_data::get_device_workspace_buffer_bytes(
            "object_all.randomsphere.parameters", sphere_parameters.size()*sizeof(CU_REAL)));
    if (!device_map || !device_parameters ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(device_parameters, sphere_parameters.data(),
                sphere_parameters.size()*sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
            "object_all", "upload random sphere parameters") ||
        !ifdda_object_all_cuda(geometry, size_t(mesh_count),
            multi_sphere_dipole_map_wrapper(device_map, *nx, *ny, *nz,
                *aretecube, -*sidex/CU_REAL(2), -*sidey/CU_REAL(2), -*sidez/CU_REAL(2),
                device_parameters, sphere_count, sphere_count), sphere_map)) {
        object_error(infostr, infostr_length, nstop, "CUDA random-sphere geometry failed");
        return;
    }
    for (int flat = 0; flat < mesh_count; ++flat) {
        const int k = flat % *nx;
        const int j = (flat / *nx) % *ny;
        const int i = flat / (*nx * *ny);
        mesh_x[size_t(flat)] = -*sidex/CU_REAL(2) + (CU_REAL(k)+CU_REAL(0.5))* *aretecube;
        mesh_y[size_t(flat)] = -*sidey/CU_REAL(2) + (CU_REAL(j)+CU_REAL(0.5))* *aretecube;
        mesh_z[size_t(flat)] = -*sidez/CU_REAL(2) + (CU_REAL(i)+CU_REAL(0.5))* *aretecube;
    }

    const bool isotropic = (trope_length >= 3 && std::memcmp(trope,"iso",3) == 0);
    const bool updates_sldr = methode[0] == 'L' && methode[1] == 'R';
    // The LR method changes sldr on every occupied-cell call; retain that
    // stateful CPU path. All other material assignments now run on the GPU.
    const bool use_gpu_materials = !isotropic || !updates_sldr;
    if (!use_gpu_materials) {
        std::fill_n(epsilon, tensor_count, COMPLEX{});
        std::fill_n(polarisa, tensor_count, COMPLEX{});
    }
    const int discretized = 1;
    const int invert_result = 1;
    const COMPLEX epsilon0_value = *eps0;
    COMPLEX scalar_alpha{};
    COMPLEX anisotropic_epsilon[9]{};
    COMPLEX anisotropic_alpha[9]{};
    if (!isotropic) {
        for (int component = 0; component < 9; ++component) {
            anisotropic_epsilon[component] = {
                epsani[component].real(), epsani[component].imag()};
        }
        polaepstens_cuda(aretecube, anisotropic_epsilon, &epsilon0_value, k0,
                     &discretized, methode, &invert_result, anisotropic_alpha,
                     methode_length);
    } else if (!updates_sldr) {
        const COMPLEX scalar_epsilon = *eps;
        poladiff_cuda(aretecube, &scalar_epsilon, &epsilon0_value, k0,
                  &discretized, methode, sldr, &scalar_alpha,
                  methode_length);
    }

    std::vector<CU_REAL> output_mesh_x;
    std::vector<CU_REAL> output_mesh_y;
    std::vector<CU_REAL> output_mesh_z;
    if (*nmat == 0) {
        output_mesh_x.resize(std::size_t(mesh_count));
        output_mesh_y.resize(std::size_t(mesh_count));
        output_mesh_z.resize(std::size_t(mesh_count));
        for (int flat = 0; flat < mesh_count; ++flat) {
            output_mesh_x[std::size_t(flat)] = mesh_x[std::size_t(flat)] + *xg;
            output_mesh_y[std::size_t(flat)] = mesh_y[std::size_t(flat)] + *yg;
            output_mesh_z[std::size_t(flat)] = mesh_z[std::size_t(flat)] + *zg;
        }
    }

    for (int& selected : sphere_map) {
        if (selected == std::numeric_limits<int>::min()) {
            object_error(infostr, infostr_length, nstop,
                         "CUDA random-sphere map is out of range");
            return;
        }
        if (selected < 0) selected = -selected;
    }
    std::vector<int> material_labels(std::size_t(*nmax), 0);
    if (!object_all_build_cpu_dipoles(geometry, sphere_map, *nmax,
             *na == -1, mesh_x.data(), mesh_y.data(), mesh_z.data(),
             tabdip, nullptr, xs, ys, zs, nbsphere, ndipole,
             std::max(1, sphere_count), false, *xg, *yg, *zg)) {
        object_error(infostr, infostr_length, nstop,
                     "CUDA random-sphere CPU map invalid");
        return;
    }
    for (int flat = 0; flat < mesh_count; ++flat) {
        if (sphere_map[std::size_t(flat)] == 0) continue;
        const std::size_t dipole = size_t(tabdip[flat] - 1);
        material_labels[dipole] = 1;
        if (!use_gpu_materials) {
            COMPLEX alpha = scalar_alpha;
            const COMPLEX scalar_epsilon = *eps;
            poladiff_cuda(aretecube, &scalar_epsilon, &epsilon0_value,
                          k0, &discretized, methode, sldr, &alpha,
                          methode_length);
            epsilon[dipole] = *eps;
            polarisa[dipole] = alpha;
        }
    }

    if (use_gpu_materials) {
        COMPLEX material_epsilon[9]{};
        COMPLEX material_alpha[9]{};
        if (isotropic) {
            material_epsilon[0] = {eps->real(), eps->imag()};
            material_alpha[0] = scalar_alpha;
        } else {
            for (int component = 0; component < 9; ++component) {
                material_epsilon[component] = {epsani[component].real(),
                                                epsani[component].imag()};
                material_alpha[component] = anisotropic_alpha[component];
            }
        }
        if (!fill_labeled_materials(geometry,
                tabdip, material_labels.data(), size_t(mesh_count), *nmax, *nbsphere,
                *niso, 1, false, material_epsilon, material_alpha,
                epsilon, polarisa)) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA random-sphere material initialization failed");
            return;
        }
    }

    if (!use_gpu_materials && *nbsphere > 0) {
        if (!publish_compact_materials(geometry, size_t(mesh_count), *nmax,
                *nbsphere, 1, polarisa, epsilon)) {
            object_error(infostr, infostr_length, nstop, "CUDA random LR material publishing failed");
            return;
        }
    }

    if (*nmat == 0 &&
        !write_rectangular_mesh(*nx, *ny, *nz, *nbsphere,
                              output_mesh_x.data(), output_mesh_y.data(),
                              output_mesh_z.data(), xs, ys, zs)) {
        object_error(infostr, infostr_length, nstop,
                  "Unable to write random-sphere mesh files");
        return;
    }
    if (*nmat == 2) {
        const int dimensions[2] = {*nbsphere, *nmax};
        hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dimensions, 17);
    }
    std::printf("Random sphere::nbsphere    = %d %d\n", *nbsphere, *ndipole);
    std::printf("density of sphere obtained = %.17g\n",
                static_cast<double>(CU_REAL(sphere_count) * sphere_volume /
                                    *sidex / *sidey / *sidez));
    (void)nxm;
    (void)nym;
    (void)nzm;
    (void)file_id;
}

extern "C" void ifdda_object_all_concentricsphere_cuda(
    int geometry, 
    const char *trope, const COMPLEX *epss, const COMPLEX *epsanis,
    const int *numbersphere, const int *numberspheremax, CU_REAL *xg, CU_REAL *yg,
    CU_REAL *zg, CU_REAL *rayons, const COMPLEX *eps0, CU_REAL *xs,
    CU_REAL *ys, CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube, int *tabdip,
    int *tabnbs, const int *nnnr, const int *nmax, int *nbsphere, int *ndipole,
    int *nx, int *ny, int *nz, const char *methode, int *na,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip, char *infostr,
    int *nstop, std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length) {
    if (geometry != int(GeometryId::ConcentricSphere)) { object_error(infostr,infostr_length,nstop,"Unsupported geometry"); return; }
    (void)trope_length;
    if (!trope || !epss || !epsanis || !numbersphere || !numberspheremax || !xg ||
        !yg || !zg || !rayons || !eps0 || !xs || !ys || !zs || !k0 ||
        !aretecube || !tabdip || !tabnbs || !nnnr || !nmax || !nbsphere ||
        !ndipole || !nx || !ny || !nz || !methode || !na || !epsilon ||
        !polarisa || !niso || !sldr || !nmat || !file_id || !group_iddip ||
        !infostr || !nstop || *numbersphere <= 0 || *numbersphere > *numberspheremax ||
        *numbersphere > 20 || *nnnr <= 0 || *nmax <= 0 ||
        (*niso != 1 && *niso != 3) || (*na != -1 && *na != 0)) {
        object_error(infostr, infostr_length, nstop,
                  "Invalid concentric-sphere geometry arguments");
        return;
    }

    *nstop = 0;
    *nbsphere = 0;
    *ndipole = 0;
    for (int dipole = 0; dipole < *nmax; ++dipole)
        tabnbs[dipole] = 0;

    xg[0] *= CU_REAL(1e-9);
    yg[0] *= CU_REAL(1e-9);
    zg[0] *= CU_REAL(1e-9);
    if (rayons[0] == CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop,
                  "spheres concentriques: 1st rayon=0");
        return;
    }
    for (int sphere = 0; sphere < *numbersphere; ++sphere) {
        rayons[sphere] *= CU_REAL(1e-9);
        if (sphere > 0 && rayons[sphere] - rayons[sphere - 1] <= CU_REAL(0)) {
            object_error(infostr, infostr_length, nstop,
                      "spheres concentriques: rayons non croissants");
            return;
        }
    }

    *aretecube = CU_REAL(2) * rayons[*numbersphere - 1] / CU_REAL(*nnnr);
    *nx = *nnnr;
    *ny = *nnnr;
    *nz = *nnnr;
    const CU_REAL center = *aretecube * CU_REAL(*nnnr) / CU_REAL(2);
    if (*nnnr > *nmax / *nnnr || *nnnr * *nnnr > *nmax / *nnnr) {
        object_error(infostr, infostr_length, nstop,
                  "nmax parameter too small: increase nxm nym nzm");
        return;
    }
    const int mesh_count = *nnnr * *nnnr * *nnnr;
    // Occupancy remains GPU-generated below; the one-based lattice map and
    // material labels are assembled on CPU in the original Fortran order.
    std::fill(tabdip, tabdip + *nmax, 0);

    std::vector<CU_REAL> mesh_x(static_cast<size_t>(mesh_count));
    std::vector<CU_REAL> mesh_y(static_cast<size_t>(mesh_count));
    std::vector<CU_REAL> mesh_z(static_cast<size_t>(mesh_count));
    std::vector<int> layer_map;
    CU_INT* device_map = cuda_data::get_dipole_map(size_t(mesh_count));
    CU_REAL* device_radii = static_cast<CU_REAL*>(
        cuda_data::get_device_workspace_buffer_bytes(
            "object_all.concentric.radii", size_t(*numbersphere)*sizeof(CU_REAL)));
    if (!device_map || !device_radii ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(device_radii, rayons,
                size_t(*numbersphere)*sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry),
            "object_all", "upload concentric radii") ||
        !ifdda_object_all_cuda(geometry, size_t(mesh_count),
            concentric_sphere_dipole_map_wrapper(device_map, *nnnr, *aretecube,
                center, device_radii, *numbersphere), layer_map)) {
        object_error(infostr, infostr_length, nstop, "CUDA concentric-sphere geometry failed");
        return;
    }
    for (int flat = 0; flat < mesh_count; ++flat) {
        const int k = flat % *nnnr;
        const int j = (flat / *nnnr) % *nnnr;
        const int i = flat / (*nnnr * *nnnr);
        mesh_x[size_t(flat)] = CU_REAL(k)* *aretecube + *aretecube/CU_REAL(2) - center;
        mesh_y[size_t(flat)] = CU_REAL(j)* *aretecube + *aretecube/CU_REAL(2) - center;
        mesh_z[size_t(flat)] = CU_REAL(i)* *aretecube + *aretecube/CU_REAL(2) - center;
    }

    const int discretized = 1;
    const int invert_result = 1;
    const bool anisotropic = std::memcmp(trope, "iso", 3) != 0;
    const bool updates_sldr = methode[0] == 'L' && methode[1] == 'R';
    // LR mutates sldr during each occupied-cell computation. Preserve that
    // stateful CPU path rather than broadcasting a potentially different alpha.
    const bool use_gpu_materials = anisotropic || !updates_sldr;
    if (!use_gpu_materials)
        polaepsto0_(polarisa, epsilon, tabdip, niso, nmax);
    std::vector<COMPLEX> scalar_polar(static_cast<size_t>(*numbersphere));
    std::vector<COMPLEX> tensor_polar(static_cast<size_t>(*numbersphere) * 9);
    if (!anisotropic && !updates_sldr) {
        for (int sphere = 0; sphere < *numbersphere; ++sphere)
            poladiff_cuda(&*aretecube, epss + sphere, eps0, k0, &discretized,
                      methode, sldr, scalar_polar.data() + sphere, methode_length);
    } else if (anisotropic) {
        for (int sphere = 0; sphere < *numbersphere; ++sphere)
            polaepstens_cuda(&*aretecube, epsanis + size_t(sphere) * 9, eps0, k0,
                         &discretized, methode, &invert_result,
                         tensor_polar.data() + size_t(sphere) * 9, methode_length);
    }

    const bool all_grid_sites = *na == 0;
    if (!object_all_build_cpu_dipoles(geometry, layer_map, *nmax,
             !all_grid_sites, mesh_x.data(), mesh_y.data(), mesh_z.data(),
             tabdip, tabnbs, xs, ys, zs, nbsphere, ndipole,
             *numbersphere, false, *xg, *yg, *zg)) {
        object_error(infostr, infostr_length, nstop,
                     "CUDA concentric-sphere CPU map invalid");
        return;
    }
    if (!use_gpu_materials) {
        for (int flat = 0; flat < mesh_count; ++flat) {
            const int layer = layer_map[size_t(flat)];
            if (layer == 0) continue;
            const size_t destination = size_t(tabdip[flat] - 1);
            const int sphere = layer - 1;
            COMPLEX alpha{};
            poladiff_cuda(&*aretecube, epss + sphere, eps0, k0, &discretized,
                      methode, sldr, &alpha, methode_length);
            polarisa[destination] = alpha;
            epsilon[destination] = epss[sphere];
        }
    }

    if (use_gpu_materials) {
        const size_t per_material = anisotropic ? 9 : 1;
        std::vector<COMPLEX> host_epsilon(size_t(*numbersphere) * per_material);
        std::vector<COMPLEX> host_alpha(size_t(*numbersphere) * per_material);
        for (int sphere = 0; sphere < *numbersphere; ++sphere) {
            for (size_t component = 0; component < per_material; ++component) {
                const size_t dest = size_t(sphere) * per_material + component;
                const COMPLEX eps_value = anisotropic ? epsanis[dest] : epss[sphere];
                const COMPLEX alpha_value = anisotropic ? tensor_polar[dest]
                                                               : scalar_polar[sphere];
                host_epsilon[dest] = eps_value;
                host_alpha[dest] = alpha_value;
            }
        }
        if (!fill_labeled_materials(geometry,
                tabdip, tabnbs, size_t(mesh_count), *nmax, *nbsphere,
                *niso, *numbersphere, true, host_epsilon.data(),
                host_alpha.data(), epsilon, polarisa)) {
            object_error(infostr, infostr_length, nstop,
                      "CUDA concentric-sphere material initialization failed");
            return;
        }
    }

    if (!use_gpu_materials && *nbsphere > 0) {
        if (!publish_compact_materials(geometry, size_t(mesh_count), *nmax,
                *nbsphere, 1, polarisa, epsilon)) {
            object_error(infostr, infostr_length, nstop, "CUDA LR material publishing failed");
            return;
        }
    }

    if (*nmat == 0 && !write_rectangular_mesh(*nnnr, *nnnr, *nnnr, *nbsphere, mesh_x.data(),
                                        mesh_y.data(), mesh_z.data(), xs, ys, zs)) {
        object_error(infostr, infostr_length, nstop,
                  "Unable to write concentric-sphere mesh files");
        return;
    }
    if (*nmat == 2) {
        const int dimensions[2] = {*nbsphere, *nmax};
        hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dimensions, 17);
    }
    (void)file_id;
}

// Cube: retain its original Fortran interface, while using exactly the same
// persistent map -> CPU tabdip -> compact GPU materials path as the sphere.
extern "C" void ifdda_object_all_cube_cuda(
    int geometry, const char* trope, const COMPLEX* eps,
    const COMPLEX* epsani, const COMPLEX* eps0,
    CU_REAL* xs, CU_REAL* ys, CU_REAL* zs, const CU_REAL* k0,
    CU_REAL* aretecube, int* tabdip, const int* nnnr, const int* nmax,
    int* nbsphere, int* ndipole, int* nx, int* ny, int* nz,
    const char* methode, COMPLEX* epsilon, COMPLEX* polarisa,
    const int* niso, CU_REAL* side, CU_REAL* xg, CU_REAL* yg, CU_REAL* zg,
    const CU_REAL* sldr, const int* nmat, const int64_t* file_id,
    const int64_t* group_iddip, char* infostr, int* nstop,
    size_t trope_length, size_t methode_length, size_t infostr_length) {
    (void)file_id;
    (void)group_iddip;
    if (geometry != static_cast<int>(GeometryId::Cube) ||
        !trope || !eps || !epsani || !eps0 || !xs || !ys || !zs || !k0 ||
        !aretecube || !tabdip || !nnnr || !nmax || !nbsphere || !ndipole ||
        !nx || !ny || !nz || !methode || !epsilon || !polarisa ||
        !niso || !side || !xg || !yg || !zg || !sldr || !nmat ||
        !infostr || !nstop) {
        object_error(infostr, infostr_length, nstop, "object_all: invalid cube arguments");
        return;
    }
    *nbsphere = *ndipole = 0;
    object_error(infostr, infostr_length, nstop, "");
    *nstop = 0;
    const bool isotropic = trope_length >= 3 && std::memcmp(trope, "iso", 3) == 0;
    const bool anisotropic = trope_length >= 3 && std::memcmp(trope, "ani", 3) == 0;
    if ((!isotropic && !anisotropic) || *nnnr <= 0 || *nmax <= 0 ||
        (*niso != 1 && *niso != 3) || (anisotropic && *niso != 3) ||
        *side == CU_REAL(0) || (*nmat != 0 && *nmat != 1)) {
        object_error(infostr, infostr_length, nstop, "object_all: invalid cube mode");
        return;
    }
    const size_t n = size_t(*nnnr);
    if (n > size_t(std::numeric_limits<int>::max()) / n ||
        n * n > size_t(std::numeric_limits<int>::max()) / n ||
        n * n * n > size_t(*nmax)) {
        object_error(infostr, infostr_length, nstop, "object_all: invalid cube grid");
        return;
    }
    const size_t grid = n * n * n;
    *side *= CU_REAL(1.0e-9);
    *xg *= CU_REAL(1.0e-9);
    *yg *= CU_REAL(1.0e-9);
    *zg *= CU_REAL(1.0e-9);
    *nx = *ny = *nz = *nnnr;
    *aretecube = *side / CU_REAL(*nnnr);

    CU_INT* device_map = cuda_data::get_dipole_map(grid);
    std::vector<int> host_map;
    if (!device_map || !ifdda_object_all_cuda(geometry, grid,
            cube_dipole_map_wrapper(device_map, *nnnr), host_map)) {
        object_error(infostr, infostr_length, nstop, "object_all: cube map failed");
        return;
    }
    const CU_REAL center = (*side + *aretecube) / CU_REAL(2);
    const ObjectAllRegularLattice lattice{
        *nnnr, *nnnr, *nnnr, *aretecube, -center, -center, -center,
        *xg, *yg, *zg, true};
    if (!object_all_build_regular_cpu_dipoles(
             geometry, host_map, *nmax, false, lattice, tabdip,
             xs, ys, zs, nbsphere, ndipole, true)) {
        object_error(infostr, infostr_length, nstop, "object_all: cube CPU map failed");
        return;
    }

    const int discretized = 1;
    const int invert_result = 1;
    CU_REAL local_sldr = *sldr;
    COMPLEX scalar_alpha{};
    COMPLEX tensor_alpha[9]{};
    if (anisotropic)
        polaepstens_cuda(aretecube, epsani, eps0, k0, &discretized,
                         methode, &invert_result, tensor_alpha, methode_length);
    else
        poladiff_cuda(aretecube, eps, eps0, k0, &discretized,
                      methode, &local_sldr, &scalar_alpha, methode_length);
    if (!fill_homogeneous_materials(geometry, tabdip, grid, *nmax, *nbsphere,
                                    *niso, false, anisotropic, eps, epsani,
                                    scalar_alpha, tensor_alpha, epsilon, polarisa)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: cube GPU materials failed");
        return;
    }
    if (*nmat == 0) {
        std::ofstream fx("x.mat"), fy("y.mat"), fz("z.mat");
        std::ofstream fcx("xc.mat"), fcy("yc.mat"), fcz("zc.mat");
        if (!fx || !fy || !fz || !fcx || !fcy || !fcz) {
            object_error(infostr, infostr_length, nstop,
                         "object_all: cube mesh output failed");
            return;
        }
        for (size_t p = 0; p < n; ++p) {
            fx << xs[p] << '\n';
            fy << ys[p * n] << '\n';
            fz << zs[p * n * n] << '\n';
        }
        for (size_t p = 0; p < grid; ++p) {
            fcx << xs[p] << '\n';
            fcy << ys[p] << '\n';
            fcz << zs[p] << '\n';
        }
    }
}

// Unrotated cuboid with dimensions derived from nxm/nym/nzm and their pads.
// Its occupied domain is the entire active rectangular grid.
extern "C" void ifdda_object_all_cuboid2_cuda(
    int geometry, const char* trope, COMPLEX* eps, COMPLEX* epsani,
    COMPLEX* eps0, CU_REAL* xs, CU_REAL* ys, CU_REAL* zs,
    const CU_REAL* k0, CU_REAL* aretecube, int* tabdip,
    const int* nnnr, const int* nmax, int* nbsphere, int* ndipole,
    int* nx, int* ny, int* nz, const int* nxm, const int* nym,
    const int* nzm, const int* nxmp, const int* nymp,
    const int* nzmp, const char* methode, COMPLEX* epsilon,
    COMPLEX* polarisa, const int* niso, CU_REAL* sidex,
    CU_REAL* sidey, CU_REAL* sidez, CU_REAL* xg, CU_REAL* yg,
    CU_REAL* zg, const int* na, CU_REAL* sldr, const int* nmat,
    const int64_t* file_id, const int64_t* group_iddip,
    char* infostr, int* nstop, size_t trope_length,
    size_t methode_length, size_t infostr_length) {
    (void)file_id;
    if (geometry != static_cast<int>(GeometryId::Cuboid2) ||
        !trope || !eps || !epsani || !eps0 || !xs || !ys || !zs ||
        !k0 || !aretecube || !tabdip || !nnnr || !nmax || !nbsphere ||
        !ndipole || !nx || !ny || !nz || !nxm || !nym || !nzm ||
        !nxmp || !nymp || !nzmp || !methode || !epsilon || !polarisa ||
        !niso || !sidex || !sidey || !sidez || !xg || !yg || !zg ||
        !na || !sldr || !nmat || !group_iddip || !infostr || !nstop ||
        *nmax <= 0 || *nnnr <= 0 || (*niso != 1 && *niso != 3)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: invalid cuboid2 arguments");
        return;
    }
    *nstop = 0;
    *nbsphere = *ndipole = 0;
    *nx = *nxm - 2 * *nxmp;
    *ny = *nym - 2 * *nymp;
    *nz = *nzm - 2 * *nzmp;
    *aretecube *= CU_REAL(1e-9);
    *sidex = *aretecube * CU_REAL(*nx);
    *sidey = *aretecube * CU_REAL(*ny);
    *sidez = *aretecube * CU_REAL(*nz);
    *xg *= CU_REAL(1e-9);
    *yg *= CU_REAL(1e-9);
    *zg *= CU_REAL(1e-9);
    if (*sidex == CU_REAL(0) || *sidey == CU_REAL(0) || *sidez == CU_REAL(0) ||
        *nx <= 0 || *ny <= 0 || *nz <= 0 ||
        *nx > *nxm || *ny > *nym || *nz > *nzm ||
        *nx > std::numeric_limits<int>::max() / *ny ||
        *nx * *ny > std::numeric_limits<int>::max() / *nz ||
        *na != -1 && *na != 0) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: invalid cuboid2 dimensions");
        return;
    }
    const size_t grid = size_t(*nx) * size_t(*ny) * size_t(*nz);
    if (grid > size_t(*nmax)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: insufficient cuboid2 nmax");
        return;
    }
    CU_INT* device_map = cuda_data::get_dipole_map(grid);
    std::vector<int> host_map;
    if (!device_map || !ifdda_object_all_cuda(geometry, grid,
            full_grid_dipole_map_wrapper(device_map, int(grid)), host_map)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: cuboid2 map failed");
        return;
    }
    const CU_REAL ox = -*sidex / CU_REAL(2);
    const CU_REAL oy = -*sidey / CU_REAL(2);
    const CU_REAL oz = -*sidez / CU_REAL(2);
    const size_t row = size_t(*nx);
    const size_t plane = row * size_t(*ny);
    const ObjectAllRegularLattice lattice{
        *nx, *ny, *nz, *aretecube, ox, oy, oz, *xg, *yg, *zg, false};
    if (!object_all_build_regular_cpu_dipoles(
             geometry, host_map, *nmax, false, lattice, tabdip,
             xs, ys, zs, nbsphere, ndipole, true)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: cuboid2 CPU map failed");
        return;
    }

    const int discretized = 1;
    const int invert_result = 1;
    const bool anisotropic = !(trope_length >= 3 && std::memcmp(trope, "iso", 3) == 0);
    COMPLEX scalar_alpha{};
    COMPLEX tensor_alpha[9]{};
    if (anisotropic)
        polaepstens_cuda(aretecube, epsani, eps0, k0, &discretized,
                         methode, &invert_result, tensor_alpha, methode_length);
    else
        poladiff_cuda(aretecube, eps, eps0, k0, &discretized,
                      methode, sldr, &scalar_alpha, methode_length);

    if (!fill_homogeneous_materials(geometry, tabdip, grid, *nmax, *nbsphere,
                                    *niso, false, anisotropic, eps, epsani,
                                    scalar_alpha, tensor_alpha, epsilon, polarisa)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: cuboid2 GPU materials failed");
        return;
    }
    if (*nmat == 0) {
        std::ofstream fx("x.mat"), fy("y.mat"), fz("z.mat");
        std::ofstream fcx("xc.mat"), fcy("yc.mat"), fcz("zc.mat");
        if (!fx || !fy || !fz || !fcx || !fcy || !fcz) {
            object_error(infostr, infostr_length, nstop,
                         "object_all: cuboid2 mesh output failed");
            return;
        }
        for (size_t p = 0; p < row; ++p) fx << xs[p] << '\n';
        for (size_t p = 0; p < size_t(*ny); ++p) fy << ys[p * row] << '\n';
        for (size_t p = 0; p < size_t(*nz); ++p) fz << zs[p * plane] << '\n';
        for (size_t p = 0; p < grid; ++p) {
            fcx << xs[p] << '\n';
            fcy << ys[p] << '\n';
            fcz << zs[p] << '\n';
        }
    } else if (*nmat == 2) {
        const int dims[2] = {*nbsphere, *nmax};
        hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dims, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dims, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dims, 17);
    }
}

// Rotated cuboid. Shape-specific setup remains in scalar CPU arithmetic;
// its grid-wide occupancy is computed in registers on the GPU.
extern "C" void ifdda_object_all_cuboid1_cuda(
    int geometry, const char* trope, const COMPLEX* eps, const COMPLEX* epsani,
    const COMPLEX* eps0, CU_REAL* xs, CU_REAL* ys, CU_REAL* zs,
    const CU_REAL* k0, CU_REAL* aretecube, int* tabdip,
    const int* nnnr, const int* nmax, int* nbsphere, int* ndipole,
    int* nx, int* ny, int* nz, const int* nxm, const int* nym,
    const int* nzm, const char* methode, COMPLEX* epsilon,
    COMPLEX* polarisa, const int* niso, CU_REAL* sidex,
    CU_REAL* sidey, CU_REAL* sidez, CU_REAL* xg, CU_REAL* yg,
    CU_REAL* zg, CU_REAL* phi, CU_REAL* theta, CU_REAL* psi,
    const int* na, CU_REAL* sldr, const int* nmat,
    const int64_t* file_id, const int64_t* group_iddip,
    char* infostr, int* nstop, size_t trope_length,
    size_t methode_length, size_t infostr_length) {
    (void)file_id;
    if (geometry != static_cast<int>(GeometryId::Cuboid1) ||
        !trope || !eps || !epsani || !eps0 || !xs || !ys || !zs || !k0 ||
        !aretecube || !tabdip || !nnnr || !nmax || !nbsphere || !ndipole ||
        !nx || !ny || !nz || !nxm || !nym || !nzm || !methode ||
        !epsilon || !polarisa || !niso || !sidex || !sidey || !sidez ||
        !xg || !yg || !zg || !phi || !theta || !psi || !na || !sldr ||
        !nmat || !group_iddip || !infostr || !nstop || *nnnr <= 0 ||
        *nmax <= 0 || (*niso != 1 && *niso != 3)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: invalid cuboid1 arguments");
        return;
    }
    *nstop = 0;
    *nbsphere = *ndipole = 0;
    constexpr CU_REAL pi = CU_REAL(3.141592653589793238462643383279502884L);
    const CU_REAL scale = CU_REAL(1e-9);
    *sidex *= scale;
    *sidey *= scale;
    *sidez *= scale;
    *xg *= scale; *yg *= scale; *zg *= scale;
    *theta *= pi / CU_REAL(180);
    *phi *= pi / CU_REAL(180);
    *psi *= pi / CU_REAL(180);
    if (*sidex == CU_REAL(0) || *sidey == CU_REAL(0) || *sidez == CU_REAL(0)) {
        object_error(infostr, infostr_length, nstop, "object_all: cuboid1 zero side");
        return;
    }
    const CU_REAL cp = std::cos(*phi), sp = std::sin(*phi);
    const CU_REAL ct = std::cos(*theta), st = std::sin(*theta);
    const CU_REAL cs = std::cos(*psi), ss = std::sin(*psi);
    const CU_REAL rotation[9] = {
        cs*cp-ss*ct*sp, -cs*sp-ss*ct*cp, ss*st,
        ss*cp+cs*ct*sp, -ss*sp+cs*ct*cp, -cs*st,
        st*sp, st*cp, ct
    };
    CU_REAL minimum[3] = {std::numeric_limits<CU_REAL>::max(),
                          std::numeric_limits<CU_REAL>::max(),
                          std::numeric_limits<CU_REAL>::max()};
    CU_REAL maximum[3] = {-minimum[0], -minimum[1], -minimum[2]};
    for (int corner = 0; corner < 8; ++corner) {
        const CU_REAL point[3] = {
            (corner & 1) ? *sidex/CU_REAL(2) : -*sidex/CU_REAL(2),
            (corner & 2) ? *sidey/CU_REAL(2) : -*sidey/CU_REAL(2),
            (corner & 4) ? *sidez/CU_REAL(2) : -*sidez/CU_REAL(2)
        };
        for (int axis = 0; axis < 3; ++axis) {
            const CU_REAL value = rotation[axis*3] * point[0] +
                                  rotation[axis*3+1] * point[1] +
                                  rotation[axis*3+2] * point[2];
            minimum[axis] = std::min(minimum[axis], value);
            maximum[axis] = std::max(maximum[axis], value);
        }
    }
    const CU_REAL extent_x = maximum[0] - minimum[0];
    const CU_REAL extent_y = maximum[1] - minimum[1];
    const CU_REAL extent_z = maximum[2] - minimum[2];
    const CU_REAL box_side = std::max(extent_x, std::max(extent_y, extent_z));
    *aretecube = box_side / CU_REAL(*nnnr);
    *nx = int(extent_x / *aretecube);
    *ny = int(extent_y / *aretecube);
    *nz = int(extent_z / *aretecube);
    if (*nx <= 0 || *ny <= 0 || *nz <= 0 || *nx > *nxm || *ny > *nym ||
        *nz > *nzm || *nx > std::numeric_limits<int>::max() / *ny ||
        *nx * *ny > std::numeric_limits<int>::max() / *nz) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: invalid cuboid1 grid");
        return;
    }
    const size_t grid = size_t(*nx) * size_t(*ny) * size_t(*nz);
    if (grid > size_t(*nmax) || (*na != 0 && *na != -1)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: unsupported cuboid1 capacity or mode");
        return;
    }
    CU_INT* device_map = cuda_data::get_dipole_map(grid);
    std::vector<int> inside;
    if (!device_map || !ifdda_object_all_cuda(geometry, grid,
            rotated_cuboid_dipole_map_wrapper(
                device_map, *nx, *ny, *nz, *aretecube,
                extent_x, extent_y, extent_z, *sidex, *sidey, *sidez,
                rotation[0],rotation[1],rotation[2],rotation[3],rotation[4],
                rotation[5],rotation[6],rotation[7],rotation[8]), inside)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: cuboid1 kernel failed");
        return;
    }
    const bool compact = *na == -1;
    const ObjectAllRegularLattice lattice{
        *nx, *ny, *nz, *aretecube,
        -extent_x / CU_REAL(2), -extent_y / CU_REAL(2), -extent_z / CU_REAL(2),
        *xg, *yg, *zg, false};
    if (!object_all_build_regular_cpu_dipoles(
             geometry, inside, *nmax, compact, lattice, tabdip,
             xs, ys, zs, nbsphere, ndipole)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: cuboid1 CPU map failed");
        return;
    }

    const bool anisotropic = !(trope_length >= 3 && std::memcmp(trope, "iso", 3) == 0);
    const int discretized = 1, invert_result = 1;
    COMPLEX scalar_alpha{};
    COMPLEX tensor_alpha[9]{};
    const bool stateful_lr = !anisotropic && methode_length >= 2 &&
                             methode[0] == 'L' && methode[1] == 'R';
    if (anisotropic)
        polaepstens_cuda(aretecube, epsani, eps0, k0, &discretized, methode,
                         &invert_result, tensor_alpha, methode_length);
    else if (!stateful_lr || std::any_of(inside.begin(), inside.end(),
                                       [](int v) { return v != 0; }))
        poladiff_cuda(aretecube, eps, eps0, k0, &discretized, methode,
                      sldr, &scalar_alpha, methode_length);
    if (!fill_homogeneous_materials(geometry, tabdip, grid, *nmax, *nbsphere, *niso,
                                    compact, anisotropic, eps, epsani,
                                    scalar_alpha, tensor_alpha, epsilon, polarisa)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: cuboid1 GPU materials failed");
        return;
    }
    if (*nmat == 0 && !write_rectangular_mesh_regular(
            lattice, *nbsphere, xs, ys, zs)) {
        object_error(infostr, infostr_length, nstop,
                     "object_all: cuboid1 mesh output failed");
    } else if (*nmat == 2) {
        const int dims[2] = {*nbsphere, *nmax};
        hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dims, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dims, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dims, 17);
    }
}

// Preserve the Fortran ABI while delegating the geometry calculations to
// the corresponding object_all entry, as for the other supported shapes.
extern "C" void objetsphere_(
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
    ifdda_object_all_sphere_cuda(static_cast<int>(GeometryId::Sphere),
                          trope, eps, epsani, eps0, lnrbc, lcos,
                          ntypeabc, xs, ys, zs, k0, aretecube, tabdip,
                          tabmulti, pmlwrk, nnnr, nmax, nbsphere, ndipole,
                          nx, ny, nz, methode, na, epsilon, polarisa, niso,
                          rayon, xg, yg, zg, sldr, nmat, file_id, group_iddip,
                          infostr, nstop, trope_length, methode_length,
                          infostr_length);
}
namespace {

//-----------------------------------------------------------
void set_error(char *info, std::size_t length, int *stop, const char *message) {
    if (stop) *stop = 1;
    if (!info || !length) return;
    std::memset(info, ' ', length);
    std::memcpy(info, message, std::min(length, std::strlen(message)));
}

//-----------------------------------------------------------
bool write_axis(const char *path, const CU_REAL *values, int count, int stride) {
    FILE *file = std::fopen(path, "w");
    if (!file) return false;
    for (int i = 0; i < count; ++i)
        std::fprintf(file, " %.17g\n", double(values[std::size_t(i) * stride]));
    return std::fclose(file) == 0;
}

//-----------------------------------------------------------
bool write_geometry_files(int nx, int ny, int nz, int dipoles,
                          const CU_REAL *mesh_x, const CU_REAL *mesh_y,
                          const CU_REAL *mesh_z, const CU_REAL *xs,
                          const CU_REAL *ys, const CU_REAL *zs) {
    return write_axis("x.mat", mesh_x, nx, 1) && write_axis("y.mat", mesh_y, ny, nx) &&
           write_axis("z.mat", mesh_z, nz, nx * ny) &&
           write_axis("xc.mat", xs, dipoles, 1) &&
           write_axis("yc.mat", ys, dipoles, 1) &&
           write_axis("zc.mat", zs, dipoles, 1);
}
} // namespace
extern "C" void ifdda_object_all_cylinder_cuda(
    int geometry,
    const char *trope, const COMPLEX *eps, const COMPLEX *epsani,
    const COMPLEX *eps0, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, const int *nnnr,
    const int *nmax, int *nbsphere, int *ndipole, int *nx, int *ny, int *nz,
    const int *nxm, const int *nym, const int *nzm, const char *methode,
    const int *na, COMPLEX *epsilon, COMPLEX *polarisa, const int *niso,
    CU_REAL *radius, CU_REAL *height, CU_REAL *xg, CU_REAL *yg, CU_REAL *zg,
    CU_REAL *phi, CU_REAL *theta, CU_REAL *psi, CU_REAL *sldr, const int *nmat,
    const int64_t *file_id, const int64_t *group_iddip, char *infostr, int *nstop,
    std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length) {
    if (geometry != static_cast<int>(GeometryId::Cylinder)) {
        set_error(infostr, infostr_length, nstop,
                  "object_all: invalid cylinder geometry selector");
        return;
    }
    (void)trope_length;
    if (!trope || !eps || !epsani || !eps0 || !xs || !ys || !zs || !k0 ||
        !aretecube || !tabdip || !nnnr || !nmax || !nbsphere || !ndipole ||
        !nx || !ny || !nz || !nxm || !nym || !nzm || !methode || !na ||
        !epsilon || !polarisa || !niso || !radius || !height || !xg || !yg ||
        !zg || !phi || !theta || !psi || !sldr || !nmat || !file_id ||
        !group_iddip || !infostr || !nstop || *nnnr <= 0 || *nmax <= 0 ||
        (*niso != 1 && *niso != 3)) {
        set_error(infostr, infostr_length, nstop, "Invalid cylinder arguments");
        return;
    }
    *nstop = 0;
    *nbsphere = 0;
    *ndipole = 0;
    const CU_REAL scale = CU_REAL(1e-9);
    const CU_REAL pi = CU_REAL(3.141592653589793238462643383279502884L);
    *radius *= scale;
    *height *= scale;
    *xg *= scale; *yg *= scale; *zg *= scale;
    *theta *= pi / CU_REAL(180);
    *phi *= pi / CU_REAL(180);
    *psi *= pi / CU_REAL(180);
    if (*radius == CU_REAL(0)) {
        set_error(infostr, infostr_length, nstop, "object cylinder: radius=0!");
        return;
    }
    if (*height == CU_REAL(0)) {
        set_error(infostr, infostr_length, nstop, "object cylinder: height=0!");
        return;
    }

    const CU_REAL cp = std::cos(*phi), sp = std::sin(*phi);
    const CU_REAL ct = std::cos(*theta), st = std::sin(*theta);
    const CU_REAL cs = std::cos(*psi), ss = std::sin(*psi);
    const CU_REAL rotation[9] = {
        cs*cp-ss*ct*sp, -cs*sp-ss*ct*cp, ss*st,
        ss*cp+cs*ct*sp, -ss*sp+cs*ct*cp, -cs*st,
        st*sp, st*cp, ct
    };
    CU_REAL inverse_rotation[9];
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            inverse_rotation[row*3+column] = rotation[column*3+row];

    CU_REAL minimum[3] = {std::numeric_limits<CU_REAL>::max(),
                          std::numeric_limits<CU_REAL>::max(),
                          std::numeric_limits<CU_REAL>::max()};
    CU_REAL maximum[3] = {-minimum[0], -minimum[1], -minimum[2]};
    for (int corner = 0; corner < 8; ++corner) {
        const CU_REAL point[3] = {
            (corner & 1) ? *radius : -*radius,
            (corner & 2) ? *radius : -*radius,
            (corner & 4) ? *height / CU_REAL(2) : -*height / CU_REAL(2)
        };
        for (int axis = 0; axis < 3; ++axis) {
            const CU_REAL value = rotation[axis*3] * point[0] +
                                  rotation[axis*3+1] * point[1] +
                                  rotation[axis*3+2] * point[2];
            minimum[axis] = std::min(minimum[axis], value);
            maximum[axis] = std::max(maximum[axis], value);
        }
    }
    const CU_REAL extent_x = maximum[0] - minimum[0];
    const CU_REAL extent_y = maximum[1] - minimum[1];
    const CU_REAL extent_z = maximum[2] - minimum[2];
    const CU_REAL box_side = std::max(extent_x, std::max(extent_y, extent_z));
    *aretecube = box_side / CU_REAL(*nnnr);
    *nx = int(extent_x / *aretecube);
    *ny = int(extent_y / *aretecube);
    *nz = int(extent_z / *aretecube);
    if (*nx <= 0 || *ny <= 0 || *nz <= 0 || *nx > *nxm || *ny > *nym || *nz > *nzm) {
        set_error(infostr, infostr_length, nstop, "Dimension Problem of the Box : Box too small!");
        return;
    }
    const std::size_t count = std::size_t(*nx) * std::size_t(*ny) * std::size_t(*nz);
    if (count > std::size_t(*nmax) || count > std::size_t(std::numeric_limits<int>::max())) {
        set_error(infostr, infostr_length, nstop, "nmax parameter too small: increase nxm nym nzm");
        return;
    }
    if (*na != -1 && *na != 0) {
        set_error(infostr, infostr_length, nstop, "na should be equal to -1 or 0");
        return;
    }
    std::vector<CU_REAL> mesh_x(count), mesh_y(count), mesh_z(count);
    std::vector<int> inside;
    CU_INT* device_map = cuda_data::get_dipole_map(count);
    if (!device_map || !ifdda_object_all_cuda(geometry, count,
            cylinder_dipole_map_wrapper(
                device_map, *nx, *ny, *nz, *aretecube,
                extent_x, extent_y, extent_z, *radius, *height,
                inverse_rotation[0], inverse_rotation[1], inverse_rotation[2],
                inverse_rotation[3], inverse_rotation[4], inverse_rotation[5],
                inverse_rotation[6], inverse_rotation[7], inverse_rotation[8]),
            inside)) {
        set_error(infostr, infostr_length, nstop,
                  "object_all: cylinder map kernel failed");
        return;
    }
    const size_t row = size_t(*nx), plane = row*size_t(*ny);
    for (size_t flat = 0; flat < count; ++flat) {
        const size_t k = flat%row, j = (flat/row)%size_t(*ny), i = flat/plane;
        mesh_x[flat] = std::fma(*aretecube, CU_REAL(k)+CU_REAL(0.5), -extent_x/CU_REAL(2));
        mesh_y[flat] = std::fma(*aretecube, CU_REAL(j)+CU_REAL(0.5), -extent_y/CU_REAL(2));
        mesh_z[flat] = std::fma(*aretecube, CU_REAL(i)+CU_REAL(0.5), -extent_z/CU_REAL(2));
    }
    const bool anisotropic = std::memcmp(trope, "iso", 3) != 0;
    const bool all_mesh_sites = *na == 0;
    const bool stateful_scalar = !anisotropic && methode[0] == 'L' && methode[1] == 'R';
    const int discretized = 1;
    const int invert = 1;
    COMPLEX scalar_alpha{};
    std::vector<COMPLEX> tensor_alpha(9);
    const COMPLEX scalar_eps = *eps;
    const COMPLEX background_eps = *eps0;
    COMPLEX anis_eps[9]{};
    if (anisotropic)
        for (int component = 0; component < 9; ++component)
            anis_eps[component] = {std::real(epsani[component]),
                                   std::imag(epsani[component])};
    if (!anisotropic && !stateful_scalar)
        poladiff_cuda(aretecube, &scalar_eps, &background_eps, k0, &discretized, methode,
                  sldr, &scalar_alpha, methode_length);
    if (anisotropic)
        polaepstens_cuda(aretecube, anis_eps, &background_eps, k0, &discretized, methode,
                     &invert, tensor_alpha.data(), methode_length);

    // The cylinder's occupancy grid was generated by CUDA above; keep it.
    // Only the Fortran-compatible one-based map and counts are built on CPU.
    if (!object_all_build_cpu_dipoles(geometry, inside, *nmax, !all_mesh_sites,
             mesh_x.data(), mesh_y.data(), mesh_z.data(), tabdip, nullptr,
             xs, ys, zs, nbsphere, ndipole, 1, false, *xg, *yg, *zg)) {
        set_error(infostr, infostr_length, nstop,
                  "object_all: cylinder CPU map failed");
        return;
    }
    if (stateful_scalar && std::any_of(inside.begin(), inside.end(),
                                      [](int value) { return value != 0; }))
        poladiff_cuda(aretecube, &scalar_eps, &background_eps, k0, &discretized,
                      methode, sldr, &scalar_alpha, methode_length);
    const COMPLEX scalar_epsilon = *eps;
    COMPLEX tensor_epsilon[9]{};
    if (anisotropic) {
        for (int component = 0; component < 9; ++component) {
            tensor_epsilon[component] = epsani[component];
        }
    }
    if (!fill_homogeneous_materials(geometry,
            tabdip, count, *nmax, *nbsphere, *niso, !all_mesh_sites, anisotropic,
            &scalar_epsilon, tensor_epsilon, scalar_alpha, tensor_alpha.data(),
            epsilon, polarisa)) {
        set_error(infostr, infostr_length, nstop,
                  "CUDA cylinder material initialization failed");
        return;
    }
    if (*nmat == 0 && !write_geometry_files(*nx, *ny, *nz, *nbsphere,
                                            mesh_x.data(), mesh_y.data(), mesh_z.data(),
                                            xs, ys, zs)) {
        set_error(infostr, infostr_length, nstop, "Unable to write cylinder geometry files");
        return;
    }
    if (*nmat == 2) {
        int dimensions[2] = {*nbsphere, *nmax};
        hdf5write1d_i8_(group_iddip, "Dipole position x", xs, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position y", ys, dimensions, 17);
        hdf5write1d_i8_(group_iddip, "Dipole position z", zs, dimensions, 17);
    }
    (void)file_id;
}

namespace {

//-----------------------------------------------------------
void set_ellipse_error(char* destination, size_t length, int* nstop,
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
bool mutable_sldr_method(const char* method, size_t length) {
    return method && length >= 2 && method[0] == 'L' && method[1] == 'R';
}

//-----------------------------------------------------------
bool write_ellipse_mesh_files(int nx, int ny, int nz, int occupied,
                              const CU_REAL* grid_x, const CU_REAL* grid_y,
                              const CU_REAL* grid_z, const CU_REAL* output_x,
                              const CU_REAL* output_y, const CU_REAL* output_z) {
    std::ofstream x_axis("x.mat");
    std::ofstream y_axis("y.mat");
    std::ofstream z_axis("z.mat");
    std::ofstream x_mesh("xc.mat");
    std::ofstream y_mesh("yc.mat");
    std::ofstream z_mesh("zc.mat");
    if (!x_axis || !y_axis || !z_axis || !x_mesh || !y_mesh || !z_mesh)
        return false;
    for (int k = 0; k < nx; ++k)
        x_axis << grid_x[size_t(k)] << '\n';
    for (int j = 0; j < ny; ++j)
        y_axis << grid_y[size_t(j) * size_t(nx)] << '\n';
    const size_t plane = size_t(nx) * size_t(ny);
    for (int i = 0; i < nz; ++i)
        z_axis << grid_z[size_t(i) * plane] << '\n';
    for (int i = 0; i < occupied; ++i) {
        x_mesh << output_x[size_t(i)] << '\n';
        y_mesh << output_y[size_t(i)] << '\n';
        z_mesh << output_z[size_t(i)] << '\n';
    }
    return bool(x_axis && y_axis && z_axis && x_mesh && y_mesh && z_mesh);
}

//-----------------------------------------------------------
void update_bound(CU_REAL x, CU_REAL y, CU_REAL z,
                  CU_REAL* x_min, CU_REAL* x_max,
                  CU_REAL* y_min, CU_REAL* y_max,
                  CU_REAL* z_min, CU_REAL* z_max) {
    *x_min = std::min(*x_min, x);
    *x_max = std::max(*x_max, x);
    *y_min = std::min(*y_min, y);
    *y_max = std::max(*y_max, y);
    *z_min = std::min(*z_min, z);
    *z_max = std::max(*z_max, z);
}

} // namespace
extern "C" void ifdda_object_all_ellipsoid_cuda(
    int geometry,
    const char* trope, const COMPLEX* eps, const COMPLEX* epsani,
    const COMPLEX* eps0, CU_REAL* xs, CU_REAL* ys, CU_REAL* zs,
    const CU_REAL* k0, CU_REAL* aretecube, int* tabdip,
    const int* nnnr, const int* nmax, int* nbsphere, int* ndipole,
    int* nx, int* ny, int* nz, const int* nxm, const int* nym,
    const int* nzm, const char* methode, const int* na,
    COMPLEX* epsilon, COMPLEX* polarisa, const int* niso,
    CU_REAL* aa, CU_REAL* bb, CU_REAL* cc, CU_REAL* xg,
    CU_REAL* yg, CU_REAL* zg, CU_REAL* phi, CU_REAL* theta,
    CU_REAL* psi, CU_REAL* sldr, const int* nmat, const int64_t* file_id,
    const int64_t* group_iddip, char* infostr, int* nstop,
    size_t trope_length, size_t methode_length, size_t infostr_length) {
    if (geometry != static_cast<int>(GeometryId::Ellipsoid)) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "object_all: invalid ellipsoid geometry selector");
        return;
    }
    (void)file_id;
    (void)group_iddip;
    (void)trope_length;

    if (!trope || !eps || !epsani || !eps0 || !xs || !ys || !zs || !k0 ||
        !aretecube || !tabdip || !nnnr || !nmax || !nbsphere || !ndipole ||
        !nx || !ny || !nz || !nxm || !nym || !nzm || !methode || !na ||
        !epsilon || !polarisa || !niso || !aa || !bb || !cc || !xg || !yg ||
        !zg || !phi || !theta || !psi || !sldr || !nmat || !infostr || !nstop) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "invalid ellipsoid arguments");
        return;
    }

    *nstop = 0;
    *nbsphere = 0;
    *ndipole = 0;
    std::memset(infostr, ' ', infostr_length);
    if (trope_length < 3 || methode_length < 2 || *nnnr <= 0 || *nmax <= 0 ||
        (*niso != 1 && *niso != 3) || (*na != -1 && *na != 0) ||
        (*nmat != 0 && *nmat != 1) || *aa == CU_REAL(0) ||
        *bb == CU_REAL(0) || *cc == CU_REAL(0) || *k0 <= CU_REAL(0)) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "unsupported ellipsoid mode or invalid dimensions");
        return;
    }
    const bool anisotropic = std::memcmp(trope, "ani", 3) == 0;
    if (!anisotropic && std::memcmp(trope, "iso", 3) != 0) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "invalid ellipsoid isotropy");
        return;
    }
    if (anisotropic && *niso != 3) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "anisotropic ellipsoid requires niso=3");
        return;
    }
    if (std::memcmp(methode, "PS", 2) == 0) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "PS ellipsoid polarizability is not ported to CUDA yet");
        return;
    }

    constexpr CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);
    const CU_REAL degrees_to_radians = pi / CU_REAL(180);
    *aa *= CU_REAL(1.0e-9);
    *bb *= CU_REAL(1.0e-9);
    *cc *= CU_REAL(1.0e-9);
    *xg *= CU_REAL(1.0e-9);
    *yg *= CU_REAL(1.0e-9);
    *zg *= CU_REAL(1.0e-9);
    *theta *= degrees_to_radians;
    *phi *= degrees_to_radians;
    *psi *= degrees_to_radians;

    const CU_REAL cp = std::cos(*phi);
    const CU_REAL sp = std::sin(*phi);
    const CU_REAL ct = std::cos(*theta);
    const CU_REAL st = std::sin(*theta);
    const CU_REAL cs = std::cos(*psi);
    const CU_REAL ss = std::sin(*psi);
    CU_REAL rotation[9] = {
        cs * cp - ss * ct * sp, -cs * sp - ss * ct * cp, ss * st,
        ss * cp + cs * ct * sp, -ss * sp + cs * ct * cp, -cs * st,
        st * sp, st * cp, ct};

    const CU_REAL half_huge = std::numeric_limits<CU_REAL>::max() / CU_REAL(100);
    CU_REAL x_min = half_huge;
    CU_REAL x_max = -half_huge;
    CU_REAL y_min = half_huge;
    CU_REAL y_max = -half_huge;
    CU_REAL z_min = half_huge;
    CU_REAL z_max = -half_huge;
    const CU_REAL corners[8][3] = {
        {-*aa, -*bb, -*cc}, {*aa, -*bb, -*cc}, {-*aa, *bb, -*cc},
        {-*aa, -*bb, *cc}, {*aa, *bb, -*cc}, {*aa, -*bb, *cc},
        {-*aa, *bb, *cc}, {*aa, *bb, *cc}};
    for (int corner = 0; corner < 8; ++corner) {
        const CU_REAL x = rotation[0] * corners[corner][0] +
                          rotation[1] * corners[corner][1] +
                          rotation[2] * corners[corner][2];
        const CU_REAL y = rotation[3] * corners[corner][0] +
                          rotation[4] * corners[corner][1] +
                          rotation[5] * corners[corner][2];
        const CU_REAL z = rotation[6] * corners[corner][0] +
                          rotation[7] * corners[corner][1] +
                          rotation[8] * corners[corner][2];
        update_bound(x, y, z, &x_min, &x_max, &y_min, &y_max, &z_min, &z_max);
    }

    const CU_REAL max_extent = std::max(x_max - x_min,
                                        std::max(y_max - y_min, z_max - z_min));
    *aretecube = max_extent / CU_REAL(*nnnr);
    *nx = int((x_max - x_min) / *aretecube);
    *ny = int((y_max - y_min) / *aretecube);
    *nz = int((z_max - z_min) / *aretecube);
    if (*nx <= 0 || *ny <= 0 || *nz <= 0 ||
        *nx > *nxm || *ny > *nym || *nz > *nzm) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "Dimension Problem of the Box : Box too small!");
        return;
    }
    if (*nx > std::numeric_limits<int>::max() / *ny ||
        *nx * *ny > std::numeric_limits<int>::max() / *nz) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "ellipsoid grid size overflow");
        return;
    }
    const size_t grid = size_t(*nx) * size_t(*ny) * size_t(*nz);
    if (grid > size_t(*nmax)) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "nmax parameter too small: increase nxm nym nzm");
        return;
    }

    const CU_REAL determinant =
        rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
        rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
        rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
    if (determinant == CU_REAL(0)) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "ellipsoid rotation matrix is singular");
        return;
    }
    CU_REAL inverse[9] = {
        (rotation[4] * rotation[8] - rotation[5] * rotation[7]) / determinant,
        (rotation[2] * rotation[7] - rotation[1] * rotation[8]) / determinant,
        (rotation[1] * rotation[5] - rotation[2] * rotation[4]) / determinant,
        (rotation[5] * rotation[6] - rotation[3] * rotation[8]) / determinant,
        (rotation[0] * rotation[8] - rotation[2] * rotation[6]) / determinant,
        (rotation[2] * rotation[3] - rotation[0] * rotation[5]) / determinant,
        (rotation[3] * rotation[7] - rotation[4] * rotation[6]) / determinant,
        (rotation[1] * rotation[6] - rotation[0] * rotation[7]) / determinant,
        (rotation[0] * rotation[4] - rotation[1] * rotation[3]) / determinant};

    std::vector<CU_REAL> grid_x(grid), grid_y(grid), grid_z(grid);
    std::vector<int> inside;
    CU_INT* device_map = cuda_data::get_dipole_map(grid);
    if (!device_map || !ifdda_object_all_cuda(geometry, grid,
            ellipsoid_dipole_map_wrapper(
                device_map, *nx, *ny, *nz, *aretecube,
                x_max-x_min, y_max-y_min, z_max-z_min, *aa, *bb, *cc,
                inverse[0], inverse[1], inverse[2], inverse[3], inverse[4],
                inverse[5], inverse[6], inverse[7], inverse[8]), inside)) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "object_all: ellipsoid map kernel failed");
        return;
    }
    const size_t row = size_t(*nx), plane = row*size_t(*ny);
    for (size_t flat = 0; flat < grid; ++flat) {
        const size_t k = flat%row, j = (flat/row)%size_t(*ny), i = flat/plane;
        grid_x[flat] = std::fma(*aretecube, CU_REAL(k)+CU_REAL(0.5),
                                -(x_max-x_min)/CU_REAL(2));
        grid_y[flat] = std::fma(*aretecube, CU_REAL(j)+CU_REAL(0.5),
                                -(y_max-y_min)/CU_REAL(2));
        grid_z[flat] = std::fma(*aretecube, CU_REAL(i)+CU_REAL(0.5),
                                -(z_max-z_min)/CU_REAL(2));
    }
    for (size_t cell = 0; cell < grid; ++cell) {
        grid_x[cell] += *xg;
        grid_y[cell] += *yg;
        grid_z[cell] += *zg;
    }

    // The GPU occupancy and the CPU compatibility arrays share the same
    // geometry-independent indexing stage.
    const bool compact_mesh = *na == -1;
    const int discretized = 1;
    const int invert_result = 1;
    CU_REAL local_sldr = *sldr;
    COMPLEX scalar_alpha{};
    COMPLEX tensor_alpha[9]{};
    if (!anisotropic && !mutable_sldr_method(methode, methode_length))
        poladiff_cuda(aretecube, eps, eps0, k0, &discretized, methode,
                  &local_sldr, &scalar_alpha, methode_length);
    if (anisotropic)
        polaepstens_cuda(aretecube, epsani, eps0, k0, &discretized, methode,
                     &invert_result, tensor_alpha, methode_length);

    if (!object_all_build_cpu_dipoles(geometry, inside, *nmax, compact_mesh,
             grid_x.data(), grid_y.data(), grid_z.data(), tabdip, nullptr,
             xs, ys, zs, nbsphere, ndipole)) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "object_all: ellipsoid CPU map failed");
        return;
    }
    if (!anisotropic && mutable_sldr_method(methode, methode_length) &&
        std::any_of(inside.begin(), inside.end(),
                    [](int value) { return value != 0; }))
        poladiff_cuda(aretecube, eps, eps0, k0, &discretized, methode,
                     &local_sldr, &scalar_alpha, methode_length);
    if (!fill_homogeneous_materials(geometry,
            tabdip, grid, *nmax, *nbsphere, *niso, compact_mesh, anisotropic,
            eps, epsani, scalar_alpha, tensor_alpha, epsilon, polarisa)) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "CUDA ellipsoid material initialization failed");
        return;
    }
    if (*nmat == 0 &&
        !write_ellipse_mesh_files(*nx, *ny, *nz, *nbsphere,
                                  grid_x.data(), grid_y.data(), grid_z.data(),
                                  compact_mesh ? xs : grid_x.data(),
                                  compact_mesh ? ys : grid_y.data(),
                                  compact_mesh ? zs : grid_z.data())) {
        set_ellipse_error(infostr, infostr_length, nstop,
                          "could not write ellipsoid mesh files");
        return;
    }
    *sldr = local_sldr;
}
