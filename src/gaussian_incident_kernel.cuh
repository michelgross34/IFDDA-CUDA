#ifndef IFDDA_GAUSSIAN_INCIDENT_KERNEL_CUH
#define IFDDA_GAUSSIAN_INCIDENT_KERNEL_CUH

#include "gaussian_gkp_tables.h"

namespace {
__device__ __forceinline__ bool ifdda_incident_grid_fetch(
    const IfddaIncidentGrid& grid, int flat, int* compact,
    REAL* x, REAL* y, REAL* z) {
    if (flat >= grid.grid_count) return false;
    const int slot = grid.tabdip[flat].dat - 1;
    if (slot < 0 || slot >= grid.dipole_count) return false;
    const int ix = flat % grid.nx;
    const int iy = (flat / grid.nx) % grid.ny;
    const int iz = flat / (grid.nx * grid.ny);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    *x = fmaf(REAL(ix - grid.anchor_ix), grid.cell_size, grid.anchor_x);
    *y = fmaf(REAL(iy - grid.anchor_iy), grid.cell_size, grid.anchor_y);
    *z = fmaf(REAL(iz - grid.anchor_iz), grid.cell_size, grid.anchor_z);
#else
    *x = fma(REAL(ix - grid.anchor_ix), grid.cell_size, grid.anchor_x);
    *y = fma(REAL(iy - grid.anchor_iy), grid.cell_size, grid.anchor_y);
    *z = fma(REAL(iz - grid.anchor_iz), grid.cell_size, grid.anchor_z);
#endif
    *compact = slot;
    return true;
}

template <class R> struct GaussianComplex {
    R re, im;
};

template <class R>
__device__ GaussianComplex<R>
gaussian_integrand(R kz, R radius, R axial, R k02, R w02, int evanescent, int component) {
    const R kp = sqrt(evanescent ? k02 + kz * kz : fmax(R(0), k02 - kz * kz));
    const R envelope = w02 * exp(-kp * kp * w02 / R(2));
    const R bessel =
        component == 0 ? kz * R(j0(double(radius * kp))) : kp * R(j1(double(radius * kp)));
    if (evanescent)
        return {envelope * exp(-kz * axial) * bessel, R(0)};
    return {envelope * cos(kz * axial) * bessel, envelope * sin(kz * axial) * bessel};
}

template <class R>
__device__ bool gaussian_integrate(const double* nodes,
                                   const double* weights,
                                   R lo,
                                   R hi,
                                   R radius,
                                   R axial,
                                   R k02,
                                   R w02,
                                   R tolerance,
                                   int evanescent,
                                   GaussianComplex<R> result[2]) {
    const R midpoint = (lo + hi) / R(2), half_width = (hi - lo) / R(2);
    GaussianComplex<R> previous[2] = {{R(0), R(0)}, {R(0), R(0)}};
    for (int stage = 1; stage <= 9; ++stage) {
        GaussianComplex<R> current[2] = {{R(0), R(0)}, {R(0), R(0)}};
        for (int component = 0; component < 2; ++component) {
            if (stage == 1) {
                const GaussianComplex<R> f =
                    gaussian_integrand<R>(midpoint + half_width * R(nodes[255]),
                                          radius,
                                          axial,
                                          k02,
                                          w02,
                                          evanescent,
                                          component);
                current[component] = {f.re * R(weights[0]) * half_width,
                                      f.im * R(weights[0]) * half_width};
            } else {
                const int old_order = (1 << (stage - 1)) - 1;
                const int old_stride = 1 << (stage - 1);
                const int stride = 1 << stage;
                const int new_order = stride - 1;
                for (int j = 1; j <= old_order; ++j) {
                    const int node = (512 / old_stride) * j;
                    const int weight =
                        2 * old_order - (stage - 1) + (new_order + 1) / old_stride * j;
                    const GaussianComplex<R> f =
                        gaussian_integrand<R>(midpoint + half_width * R(nodes[node - 1]),
                                              radius,
                                              axial,
                                              k02,
                                              w02,
                                              evanescent,
                                              component);
                    current[component].re += f.re * R(weights[weight - 1]);
                    current[component].im += f.im * R(weights[weight - 1]);
                }
                for (int j = 1; j <= new_order; j += 2) {
                    const int node = (512 / stride) * j;
                    const int weight = 2 * old_order - (stage - 1) + (new_order + 1) / stride * j;
                    const GaussianComplex<R> f =
                        gaussian_integrand<R>(midpoint + half_width * R(nodes[node - 1]),
                                              radius,
                                              axial,
                                              k02,
                                              w02,
                                              evanescent,
                                              component);
                    current[component].re += f.re * R(weights[weight - 1]);
                    current[component].im += f.im * R(weights[weight - 1]);
                }
                current[component].re *= half_width;
                current[component].im *= half_width;
            }
        }
        R previous_max =
            fmax(hypot(previous[0].re, previous[0].im), hypot(previous[1].re, previous[1].im));
        bool converged = stage >= 4;
        for (int j = 0; j < 2; ++j) {
            const R error = hypot(current[j].re - previous[j].re, current[j].im - previous[j].im);
            if (stage >= 4 && !(error / previous_max <= tolerance || error <= R(0)))
                converged = false;
            previous[j] = current[j];
        }
        if (converged) {
            result[0] = current[0];
            result[1] = current[1];
            return true;
        }
    }
    result[0] = previous[0];
    result[1] = previous[1];
    return false;
}

template <class R>
__device__ GaussianComplex<R> gaussian_mul(GaussianComplex<R> a, GaussianComplex<R> b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

template <class R>
__global__ void rigorous_gaussian_incident_kernel(const CU_REAL* xs,
                                                  const CU_REAL* ys,
                                                  const CU_REAL* zs,
                                                  int count,
                                                  double x0,
                                                  double y0,
                                                  double z0,
                                                  double theta_deg,
                                                  double phi_deg,
                                                  double waist,
                                                  double k0,
                                                  double ss,
                                                  double pp,
                                                  CUCOMPLEX e0,
                                                  double tolerance,
                                                  int circular,
                                                  int nloin,
                                                  const double* nodes,
                                                  const double* weights,
                                                  CUCOMPLEX* field,
                                                  CU_INT* stop,
                                                  IfddaIncidentGrid grid) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    int id = flat;
    REAL pos_x = 0, pos_y = 0, pos_z = 0;
    if (grid.tabdip) {
        if (!ifdda_incident_grid_fetch(grid, flat, &id, &pos_x, &pos_y, &pos_z))
            return;
    } else if (id < count) {
        pos_x = xs[id]; pos_y = ys[id]; pos_z = zs[id];
    } else
        return;
    const R theta = R(theta_deg * 3.14159265358979323846 / 180.0);
    const R phi = R(phi_deg * 3.14159265358979323846 / 180.0);
    const R st = sin(theta), ct = cos(theta), sp = sin(phi), cp = cos(phi);
    const R vu[3] = {st * cp, st * sp, ct};
    R mat[3][3];
    if (vu[0] * vu[0] + vu[1] * vu[1] <= (sizeof(R) == 4 ? R(1e-6) : R(1e-12))) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                mat[i][j] = i == j ? R(1) : R(0);
    } else {
        mat[0][0] = 1 - vu[0] * vu[0] / (1 + vu[2]);
        mat[0][1] = -vu[0] * vu[1] / (1 + vu[2]);
        mat[0][2] = -vu[0];
        mat[1][0] = mat[0][1];
        mat[1][1] = 1 - vu[1] * vu[1] / (1 + vu[2]);
        mat[1][2] = -vu[1];
        mat[2][0] = vu[0];
        mat[2][1] = vu[1];
        mat[2][2] = vu[2];
    }
    GaussianComplex<R> e = {R(e0.x), R(e0.y)}, es, ep;
    const R s = circular ? R(1) / sqrt(R(2)) : sqrt(R(1) - R(pp));
    const R p = circular ? copysign(R(1) / sqrt(R(2)), R(ss)) : sqrt(R(pp));
    es = {e.re * s, e.im * s};
    ep =
        circular ? GaussianComplex<R>{-e.im * p, e.re * p} : GaussianComplex<R>{e.re * p, e.im * p};
    GaussianComplex<R> incident[3] = {
        {-es.re * sp + ep.re * ct * cp, -es.im * sp + ep.im * ct * cp},
        {es.re * cp + ep.re * ct * sp, es.im * cp + ep.im * ct * sp},
        {-ep.re * st, -ep.im * st}};
    GaussianComplex<R> rotated[3] = {{0, 0}, {0, 0}, {0, 0}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            rotated[i].re += mat[i][j] * incident[j].re;
            rotated[i].im += mat[i][j] * incident[j].im;
        }
    const R pos[3] = {R(pos_x) - R(x0), R(pos_y) - R(y0), R(pos_z) - R(z0)};
    R r[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i] += mat[i][j] * pos[j];
    const R radius = hypot(r[0], r[1]), axial = fabs(r[2]);
    const R k02 = R(k0) * R(k0), w02 = R(waist) * R(waist);
    GaussianComplex<R> ix = {0, 0}, iz = {0, 0};
    int failed_modes = 0;
    for (int mode = 0; mode < 2; ++mode) {
        if ((nloin == 1 && mode == 1) || (nloin == 2 && mode == 0))
            continue;
        const R lo = mode ? R(0) : sqrt(fmax(R(0), k02 - R(100) / w02));
        const R hi = mode ? fmin(R(15) / (axial + R(1e-10)), R(10) / R(waist)) : R(k0);
        GaussianComplex<R> q[2];
        const bool success = gaussian_integrate<R>(
            nodes, weights, lo, hi, radius, axial, k02, w02, R(tolerance), mode, q);
        if (!success)
            failed_modes |= 1 << mode;
        ix.re += q[0].re;
        ix.im += q[0].im;
        if (mode) {
            iz.re += q[1].im;
            iz.im -= q[1].re;
        } else {
            iz.re += q[1].re;
            iz.im += q[1].im;
        }
    }
    if (r[2] < 0) {
        ix.im = -ix.im;
        iz.im = -iz.im;
    }
    const R sr = radius == 0 ? R(0) : r[0] / radius, cr = radius == 0 ? R(0) : r[1] / radius;
    GaussianComplex<R> out[3] = {
        gaussian_mul(ix, rotated[0]), gaussian_mul(ix, rotated[1]), {0, 0}};
    const GaussianComplex<R> radial = {rotated[0].re * sr + rotated[1].re * cr,
                                       rotated[0].im * sr + rotated[1].im * cr};
    const GaussianComplex<R> ez = gaussian_mul(iz, radial);
    out[2] = {ez.im, -ez.re};
    for (int i = 0; i < 3; ++i) {
        R re = 0, im = 0;
        for (int j = 0; j < 3; ++j) {
            re += mat[j][i] * out[j].re;
            im += mat[j][i] * out[j].im;
        }
        field[3 * id + i].x = CU_REAL(re);
        field[3 * id + i].y = CU_REAL(im);
    }
    stop[id].dat = failed_modes;
}
} // namespace

#endif
