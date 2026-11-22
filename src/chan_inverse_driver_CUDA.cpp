#include "kernel.h"
#include "cuda_interfaces.h"
#include "cuda_solver_interfaces.h"
#include "cuda_solver_host_adapter.h"

#include <algorithm>
#include <cstring>





namespace {

//-----------------------------------------------------------
void set_error(char *info, size_t info_length, int *stop, const char *message) {
    if (stop)
        *stop = 1;
    if (info && info_length) {
        std::memset(info, ' ', info_length);
        if (message)
            std::memcpy(info, message, std::min(info_length, std::strlen(message)));
    }
}

//-----------------------------------------------------------
void run_left_chan(
    const COMPLEX *green_xx, const COMPLEX *green_xy,
    const COMPLEX *green_xz, const COMPLEX *green_yy,
    const COMPLEX *green_yz, const COMPLEX *green_zz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz,
    const int *tabdip, const int *ntotalm, const int *ntotal,
    const int *lda, const int *nlar, const int *nmax, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm, const int *nx,
    const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nxy2, const int *nz2, const int *ndipoles, const int *ndim,
    COMPLEX *xi, CUCOMPLEX *xr, CUCOMPLEX *work, CUCOMPLEX *field,
    const CUCOMPLEX *incident, CUCOMPLEX *local_field,
    const CUCOMPLEX *polarizability, const int *niso, const char *method,
    const CU_REAL *tolerance, CU_REAL *final_residual, int *products,
    int *product_calls, const int *max_iterations, const CU_REAL *cell_size,
    const CU_REAL *k0, const long long *planf, const long long *planb,
    int optimized, int *stop, char *info, size_t method_length,
    size_t info_length) {
    if (!stop)
        return;
    *stop = 1;
    if (info && info_length)
        std::memset(info, ' ', info_length);

    // Host FFT scratch vectx/vecty/vectz is optional for CUDA/cuFFT.
    if (!green_xx || !green_xy || !green_xz || !green_yy || !green_yz ||
        !green_zz || !tabdip || !ntotalm ||
        !ntotal || !lda || !nlar || !nmax || !ndipole || !nxm || !nym ||
        !nzm || !nx || !ny || !nz || !nx2 || !ny2 || !nxy2 || !nz2 ||
        !ndipoles || !ndim || !xi || !xr || !work || !field || !incident ||
        !local_field || !polarizability || !niso || !method || !tolerance ||
        !final_residual || !products || !product_calls || !max_iterations ||
        !cell_size || !k0 || !planf || !planb || !info || *ndipole <= 0 ||
        *nx <= 0 || *ny <= 0 || *nz <= 0 || *ndipoles <= 0 || *ndim <= 0) {
        set_error(info, info_length, stop, "invalid left Chan solver arguments");
        return;
    }

    const int build_status = chan_precon_pol_build_lu_cuda_(
        polarizability, ndipole, cell_size, k0, nx, ny, nz, nxm, nym, nzm);
    if (build_status != 0) {
        set_error(info, info_length, stop,
                  "CUDA Chan matrix generation/factorization failed");
        return;
    }

    const int enable_mode = 1;
    const int enable_status = ifdda_set_resident_chan_preconditioner_all_(
        &enable_mode, nx, ny, nz, nxm, nym, nzm);
    if (enable_status != 0) {
        set_error(info, info_length, stop,
                  "Unable to activate resident Chan preconditioner");
        return;
    }

    const int initial_guess_mode = -1;
    cuda_solver_host_adapter::InverseDriverArguments host;
    if (!host.initialize_solver_buffers(xr, work, field, incident, local_field,
                                        polarizability, *nmax, *niso,
                                        *lda, *nlar, *ndim)) {
        set_error(info, info_length, stop, "Unable to adapt left Chan host buffers");
    } else {
        *stop = 0;
        inverserig_all_cuda_driver_(
            green_xx, green_xy, green_xz, green_yy, green_yz, green_zz,
            vectx, vecty, vectz, optimized ? nullptr : tabdip, ntotalm, ntotal, lda, nlar, nmax,
            ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,
            ndipoles, ndim, xi, host.xr.data(), host.work.data(), host.field.data(),
            host.incident.data(), host.local_field.data(),
            host.polarizability.data(), niso, method, tolerance, final_residual,
            products, product_calls, max_iterations, planf, planb, planf, planb,
            planf, planb, planf, planb, &optimized, &initial_guess_mode,
            &enable_mode, stop,
            info, method_length, info_length);
        host.publish();
    }

    const int disable_mode = 0;
    const int disable_status = ifdda_set_resident_chan_preconditioner_all_(
        &disable_mode, nx, ny, nz, nxm, nym, nzm);
    if (disable_status != 0 && *stop == 0)
        set_error(info, info_length, stop,
                  "Unable to deactivate resident Chan preconditioner");
}

} // namespace

//-----------------------------------------------------------
extern "C" void inverserigpg_(
    const COMPLEX *gxx, const COMPLEX *gxy, const COMPLEX *gxz,
    const COMPLEX *gyy, const COMPLEX *gyz, const COMPLEX *gzz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz,
    const int *tabdip, const int *ntotalm, const int *ntotal,
    const int *lda, const int *nlar, const int *nmax, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm, const int *nx,
    const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nxy2, const int *nz2, const int *ndipoles, const int *ndim,
    COMPLEX *xi, CUCOMPLEX *xr, CUCOMPLEX *work, CUCOMPLEX *field,
    const CUCOMPLEX *incident, CUCOMPLEX *local_field,
    CUCOMPLEX *preconditioned_field, const CUCOMPLEX *polarizability,
    const int *niso, const char *method, const CU_REAL *tolerance,
    CU_REAL *final_residual, int *products, int *product_calls,
    const int *max_iterations, const CU_REAL *k0, const CU_REAL *cell_size,
    const CUCOMPLEX *chan_xx, const CUCOMPLEX *chan_xy,
    const CUCOMPLEX *chan_xz, const CUCOMPLEX *chan_yy,
    const CUCOMPLEX *chan_yz, const CUCOMPLEX *chan_zz,
    CUCOMPLEX *sdetnn, const long long *planf, const long long *planb,
    int *stop, char *info, size_t method_length, size_t info_length) {
    (void)preconditioned_field;
    (void)chan_xx;
    (void)chan_xy;
    (void)chan_xz;
    (void)chan_yy;
    (void)chan_yz;
    (void)chan_zz;
    (void)sdetnn;
    run_left_chan(gxx, gxy, gxz, gyy, gyz, gzz, vectx, vecty, vectz,
                  tabdip, ntotalm, ntotal, lda, nlar, nmax, ndipole, nxm,
                  nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, ndipoles,
                  ndim, xi, xr, work, field, incident, local_field,
                  polarizability, niso, method, tolerance, final_residual,
                  products, product_calls, max_iterations, cell_size, k0,
                  planf, planb, 0, stop, info, method_length, info_length);
}

//-----------------------------------------------------------
extern "C" void inverserigoptpg_(
    const COMPLEX *gxx, const COMPLEX *gxy, const COMPLEX *gxz,
    const COMPLEX *gyy, const COMPLEX *gyz, const COMPLEX *gzz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz,
    const int *tabdip, const int *ndipole, const int *ntotalm,
    const int *ntotal, const int *lda, const int *nlar, const int *nmax,
    const int *nxm, const int *nym, const int *nzm, const int *nx,
    const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nxy2, const int *nz2, const int *ndipoles, const int *ndim,
    COMPLEX *xi, CUCOMPLEX *xr, CUCOMPLEX *work, CUCOMPLEX *field,
    const CUCOMPLEX *incident, CUCOMPLEX *local_field,
    CUCOMPLEX *preconditioned_field, const CUCOMPLEX *polarizability,
    const int *niso, const char *method, const CU_REAL *tolerance,
    CU_REAL *final_residual, int *products, int *product_calls,
    const int *max_iterations, const CU_REAL *k0, const CU_REAL *cell_size,
    const CUCOMPLEX *chan_xx, const CUCOMPLEX *chan_xy,
    const CUCOMPLEX *chan_xz, const CUCOMPLEX *chan_yy,
    const CUCOMPLEX *chan_yz, const CUCOMPLEX *chan_zz,
    CUCOMPLEX *sdetnn, const long long *planf, const long long *planb,
    int *stop, char *info, size_t method_length, size_t info_length) {
    (void)preconditioned_field;
    (void)chan_xx;
    (void)chan_xy;
    (void)chan_xz;
    (void)chan_yy;
    (void)chan_yz;
    (void)chan_zz;
    (void)sdetnn;
    run_left_chan(gxx, gxy, gxz, gyy, gyz, gzz, vectx, vecty, vectz,
                  tabdip, ntotalm, ntotal, lda, nlar, nmax, ndipole, nxm,
                  nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, ndipoles,
                  ndim, xi, xr, work, field, incident, local_field,
                  polarizability, niso, method, tolerance, final_residual,
                  products, product_calls, max_iterations, cell_size, k0,
                  planf, planb, 1, stop, info, method_length, info_length);
}
