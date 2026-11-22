#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

//-----------------------------------------------------------



extern "C" void polaepsto0_(CUCOMPLEX* polarizability, CUCOMPLEX* epsilon,
                              int* dipole_map, const int* isotropy,
                              const int* dipole_count);

extern "C" void poladiff_(const CU_REAL* radius, const CUCOMPLEX* epsilon,
                           const CUCOMPLEX* epsilon0, const CU_REAL* k0,
                           const int* discretized, const char* method,
                           CU_REAL* sldr, CUCOMPLEX* polarizability,
                           size_t method_length);

extern "C" void polaepstens_(const CU_REAL* radius, const CUCOMPLEX* epsilon,
                              const CUCOMPLEX* epsilon0, const CU_REAL* k0,
                              const int* discretized, const char* method,
                              const int* invert_result, CUCOMPLEX* polarizability,
                              size_t method_length);

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

} // namespace

//-----------------------------------------------------------
extern "C" void objetsphere_(
    const char* trope, const CUCOMPLEX* eps, const CUCOMPLEX* epsani,
    const CUCOMPLEX* eps0, CU_REAL* lnrbc, CU_REAL* lcos, int* ntypeabc,
    CU_REAL* xs, CU_REAL* ys, CU_REAL* zs, const CU_REAL* k0,
    CU_REAL* aretecube, int* tabdip, int* tabmulti, int* pmlwrk,
    const int* nnnr, const int* nmax, int* nbsphere, int* ndipole,
    int* nx, int* ny, int* nz, const char* methode, const int* na,
    CUCOMPLEX* epsilon, CUCOMPLEX* polarisa, const int* niso,
    CU_REAL* rayon, CU_REAL* xg, CU_REAL* yg, CU_REAL* zg,
    CU_REAL* sldr, const int* nmat, const void* file_id,
    const void* group_iddip, char* infostr, int* nstop,
    size_t trope_length, size_t methode_length, size_t infostr_length) {

    (void)lnrbc;
    (void)lcos;
    (void)ntypeabc;
    (void)tabmulti;
    (void)pmlwrk;
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
        (*niso != 1 && *niso != 3) || (*na != -1 && *na != 0) ||
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

    std::vector<CU_REAL> mesh_x(grid);
    std::vector<CU_REAL> mesh_y(grid);
    std::vector<CU_REAL> mesh_z(grid);
    std::vector<int> inside(grid);
    const bool compact_mesh = *na == -1;
    const bool compact_coordinates_on_device = compact_mesh;
    int packed_geometry_count = 0;



    // calcul de inside_data sur le gpu
    int geometry_status;
    if (compact_mesh) {

        geometry_status = ifdda_sphere_geometry_compact_cuda(
                *nnnr, *rayon, *aretecube, *xg, *yg, *zg,
                mesh_x.data(), mesh_y.data(), mesh_z.data(), inside.data(),
                xs, ys, zs, &packed_geometry_count);

    }else {

        geometry_status = ifdda_sphere_geometry_cuda(
                *nnnr, *rayon, *aretecube, *xg, *yg, *zg,
                mesh_x.data(), mesh_y.data(), mesh_z.data(), inside.data());
    }
    if (geometry_status != 0) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "CUDA sphere geometry kernel failed");
        return;
    }

    polaepsto0_(polarisa, epsilon, tabdip, niso, nmax);
    size_t occupied = 0;
    size_t packed = 0;
    const int discretized = 1;
    const int invert_result = 1;
    const bool anisotropic = std::memcmp(trope, "ani", 3) == 0;
    CU_REAL local_sldr = *sldr;
    CUCOMPLEX scalar_alpha{};
    CUCOMPLEX tensor_alpha[9]{};

    if (!anisotropic && !method_uses_mutable_sldr(methode, methode_length))
        poladiff_(aretecube, eps, eps0, k0, &discretized, methode,
                  &local_sldr, &scalar_alpha, methode_length);
    if (anisotropic)
        polaepstens_(aretecube, epsani, eps0, k0, &discretized, methode,
                     &invert_result, tensor_alpha, methode_length);

    // Build tabdip on the CPU from the occupancy map returned by CUDA.
    // This preserves the Fortran grid-to-compact-index convention.
    for (size_t cell = 0; cell < grid; ++cell) {
        if (compact_mesh) {
            tabdip[cell] = 0;
            if (!inside[cell])
                continue;
            const size_t destination = packed++;
            tabdip[cell] = int(destination + 1);
            if (!compact_coordinates_on_device) {
                xs[destination] = mesh_x[cell];
                ys[destination] = mesh_y[cell];
                zs[destination] = mesh_z[cell];
            }
            if (!anisotropic) {
                if (method_uses_mutable_sldr(methode, methode_length))
                    poladiff_(aretecube, eps, eps0, k0, &discretized, methode,
                              &local_sldr, &scalar_alpha, methode_length);
                polarisa[destination] = scalar_alpha;
                epsilon[destination] = *eps;
            } else {
                for (int column = 0; column < 3; ++column) {
                    for (int row = 0; row < 3; ++row) {
                        const size_t tensor_index = destination + size_t(*nmax) *
                                                    size_t(row + 3 * column);
                        const size_t source_index = size_t(row + 3 * column);
                        polarisa[tensor_index] = tensor_alpha[source_index];
                        epsilon[tensor_index] = epsani[source_index];
                    }
                }
            }
            ++*ndipole;
        } else {
            tabdip[cell] = int(cell + 1);
            xs[cell] = mesh_x[cell];
            ys[cell] = mesh_y[cell];
            zs[cell] = mesh_z[cell];
            if (inside[cell]) {
                if (!anisotropic) {
                    if (method_uses_mutable_sldr(methode, methode_length))
                        poladiff_(aretecube, eps, eps0, k0, &discretized, methode,
                                  &local_sldr, &scalar_alpha, methode_length);
                    polarisa[cell] = scalar_alpha;
                    epsilon[cell] = *eps;
                } else {
                    for (int column = 0; column < 3; ++column) {
                        for (int row = 0; row < 3; ++row) {
                            const size_t tensor_index = cell + size_t(*nmax) *
                                                        size_t(row + 3 * column);
                            const size_t source_index = size_t(row + 3 * column);
                            polarisa[tensor_index] = tensor_alpha[source_index];
                            epsilon[tensor_index] = epsani[source_index];
                        }
                    }
                }
            }
            ++*ndipole;
            ++occupied;
        }
    }

    if (compact_mesh)
        *ndipole = int(grid);
    *nbsphere = int(compact_mesh ? packed : occupied);
    if (compact_mesh && packed != size_t(packed_geometry_count)) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "GPU sphere compaction count does not match the material map");
        return;
    }
    const bool mesh_written = *nmat != 0 ||
        (compact_mesh
             ? write_compact_sphere_mesh_files(*nnnr, *nbsphere, *rayon,
                                               *aretecube, *xg, *yg, *zg,
                                               xs, ys, zs)
             : write_sphere_mesh_files(*nnnr, *nbsphere, mesh_x.data(),
                                       mesh_y.data(), mesh_z.data(),
                                       mesh_x.data(), mesh_y.data(), mesh_z.data()));
    if (!mesh_written) {
        set_sphere_error(infostr, infostr_length, nstop,
                         "could not write sphere mesh files");
        return;
    }
    *sldr = local_sldr;
}
