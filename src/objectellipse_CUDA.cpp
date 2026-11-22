#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

//-----------------------------------------------------------


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

//-----------------------------------------------------------
extern "C" void objetellipse_(
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
    ifdda_object_all_ellipsoid_cuda(8, trope, eps, epsani, eps0,
                                   xs, ys, zs, k0, aretecube, tabdip,
                                   nnnr, nmax, nbsphere, ndipole, nx, ny, nz,
                                   nxm, nym, nzm, methode, na, epsilon, polarisa,
                                   niso, aa, bb, cc, xg, yg, zg, phi, theta, psi,
                                   sldr, nmat, file_id, group_iddip, infostr,
                                   nstop, trope_length, methode_length, infostr_length);
}
