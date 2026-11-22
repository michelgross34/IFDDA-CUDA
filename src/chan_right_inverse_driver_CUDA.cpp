#include "kernel.h"
#include "cuda_interfaces.h"
#include "cuda_solver_interfaces.h"
#include "cuda_solver_host_adapter.h"

#include <algorithm>
#include <cstring>













namespace {
//-----------------------------------------------------------
void set_error(char *info, size_t length, int *stop, const char *message) {
    if (stop) *stop = 1;
    if (!info || length == 0) return;
    std::memset(info, ' ', length);
    std::memcpy(info, message, std::min(length, std::strlen(message)));
}

//-----------------------------------------------------------
int run_right_chan(
    const COMPLEX *gxx, const COMPLEX *gxy, const COMPLEX *gxz,
    const COMPLEX *gyy, const COMPLEX *gyz, const COMPLEX *gzz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz,
    const int *tabdip, const int *ntotalm, const int *ntotal,
    const int *ldabi, const int *nlar, const int *nmax,
    const int *ndipole, const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz, const int *nx2,
    const int *ny2, const int *nxy2, const int *nz2,
    const int *nbsphere, const int *nbsphere3, COMPLEX *xi,
    CUCOMPLEX *xr, CUCOMPLEX *work, CUCOMPLEX *field,
    CUCOMPLEX *incident, CUCOMPLEX *local_field,
    CUCOMPLEX *preconditioned_field, const CUCOMPLEX *polarizability,
    const int *niso, const char *method, const CU_REAL *tolerance,
    CU_REAL *final_residual, int *products, int *product_calls,
    const int *max_iterations, const CU_REAL *k0, const CU_REAL *cell_size,
    const int *nxm_unused, const int *nym_unused, const int *nzm_unused,
    int optimized, int *stop, char *info, size_t method_length,
    size_t info_length) {
    (void)nxm_unused;
    (void)nym_unused;
    (void)nzm_unused;
    if (!stop) return 1;
    *stop = 1;
    if (info && info_length) std::memset(info, ' ', info_length);
    // The CUDA product allocates its FFT scratch on device; host vectors may be null.
    if (!gxx || !gxy || !gxz || !gyy || !gyz || !gzz ||
        !tabdip || !ntotalm || !ntotal || !ldabi ||
        !nlar || !nmax || !ndipole || !nxm || !nym || !nzm || !nx ||
        !ny || !nz || !nx2 || !ny2 || !nxy2 || !nz2 || !nbsphere ||
        !nbsphere3 || !xi || !xr || !work || !field || !incident ||
        !local_field || !preconditioned_field || !polarizability || !niso ||
        !method || !tolerance || !final_residual || !products ||
        !product_calls || !max_iterations || !k0 || !cell_size || !info ||
        *ndipole <= 0 || *nbsphere <= 0 || *nbsphere3 <= 0 || *nx <= 0 ||
        *ny <= 0 || *nz <= 0) {
        set_error(info, info_length, stop, "invalid right Chan solver arguments");
        return 1;
    }

    long long no_plan = 0;
    const int mode = ifdda_get_initial_guess_mode_all_cuda_();
    int status = 0;
    cuda_solver_host_adapter::InverseDriverArguments host;
    if (!host.initialize_solver_buffers(xr, work, field, incident, local_field,
                                        polarizability, *nmax, *niso,
                                        *ldabi, *nlar, *nbsphere3)) {
        set_error(info, info_length, stop, "Unable to adapt right Chan host buffers");
        return 1;
    }
    if (mode < 0) {
        int ntest = 0;
        if (optimized) {
            preconditionneurinit_(preconditioned_field, xi, xr,
                polarizability, niso, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nx, ny, nz, nbsphere, nxm, nym, nzm,
                cell_size, k0, &ntest, nullptr, nullptr, nullptr, nullptr,
                &no_plan, &no_plan, info, stop, info_length);
        } else {
            preconditionneurinit2_(preconditioned_field, xi, xr, tabdip,
                polarizability, niso, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nx, ny, nz, nbsphere, ndipole, nxm, nym,
                nzm, cell_size, k0, &ntest, nullptr, nullptr, nullptr,
                nullptr, &no_plan, &no_plan, info, stop, info_length);
        }
        if (*stop != 0) return 2;
        ntest = 1;
        if (optimized) {
            preconditionneurinit_(preconditioned_field, xi, xr,
                polarizability, niso, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nx, ny, nz, nbsphere, nxm, nym, nzm,
                cell_size, k0, &ntest, nullptr, nullptr, nullptr, nullptr,
                &no_plan, &no_plan, info, stop, info_length);
        } else {
            preconditionneurinit2_(preconditioned_field, xi, xr, tabdip,
                polarizability, niso, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nx, ny, nz, nbsphere, ndipole, nxm, nym,
                nzm, cell_size, k0, &ntest, nullptr, nullptr, nullptr,
                nullptr, &no_plan, &no_plan, info, stop, info_length);
        }
        if (*stop != 0) return 3;
    } else {
        status = chan_precon_pol_build_cuda_(polarizability, ndipole,
            cell_size, k0, nx, ny, nz, nxm, nym, nzm);
        if (status != 0) {
            set_error(info, info_length, stop,
                      "CUDA right Chan matrix generation failed");
            return 4;
        }
        if (mode == 1) {
            const int dense = optimized;
            status = chan_preconditioner_initialize_born_resident_cuda_(
                incident, xi, nbsphere, tabdip, nx, ny, nz, nxm, nym, nzm,
                &dense);
            if (status != 0) {
                set_error(info, info_length, stop,
                          "CUDA resident right Chan Born initialization failed");
                return 5;
            }
            ifdda_set_initial_guess_mode_all_cuda_(3);
        } else if (mode == 3) {
            const int optimized_flag = optimized;
            status = ifdda_initialize_right_rytov_guess_all_cuda_(
                gxx, gxy, gxz, gyy, gyz, gzz,
                vectx, vecty, vectz,
                tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm,
                nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere3, ldabi, nlar,
                xi, host.xr.data(), host.work.data(), host.incident.data(),
                host.polarizability.data(), niso,
                &no_plan, &no_plan, &no_plan, &no_plan, &no_plan, &no_plan,
                &optimized_flag);
            if (status != 0) {
                set_error(info, info_length, stop,
                          "CUDA right Rytov initialization failed");
                return 6;
            }
            status = optimized
                ? chan_preconditioner_apply_initial_dense_resident_cuda_(
                    preconditioned_field, xi, nbsphere, nx, ny, nz,
                    nxm, nym, nzm)
                : chan_preconditioner_apply_initial_resident_cuda_(
                    preconditioned_field, xi, nbsphere, tabdip, nx, ny, nz,
                    nxm, nym, nzm);
            if (status != 0) {
                set_error(info, info_length, stop,
                          "CUDA right Chan Rytov preconditioning failed");
                return 7;
            }
        } else if (mode == 4) {
            status = optimized
                ? chan_preconditioner_apply_initial_dense_device_cuda_(
                    preconditioned_field, xi, nbsphere, nx, ny, nz,
                    nxm, nym, nzm)
                : chan_preconditioner_apply_initial_device_cuda_(
                    preconditioned_field, xi, nbsphere, tabdip, nx, ny, nz,
                    nxm, nym, nzm);
            if (status != 0) {
                set_error(info, info_length, stop,
                          "CUDA right Chan BPM preconditioning failed");
                return 8;
            }
        }
    }

    status = chan_preconditioner_factorize_resident_cuda_(
        nx, ny, nz, nxm, nym, nzm);
    if (status != 0) {
        set_error(info, info_length, stop,
                  "CUDA right Chan block factorization failed");
        return 9;
    }
    const int activate = 2;
    status = ifdda_set_resident_chan_preconditioner_all_(
        &activate, nx, ny, nz, nxm, nym, nzm);
    if (status != 0) {
        set_error(info, info_length, stop,
                  "Unable to activate resident right Chan preconditioner");
        return 10;
    }
    *stop = 0;
    const int driver_initial_guess = -1;
    inverserig_all_cuda_driver_(gxx, gxy, gxz, gyy, gyz, gzz, vectx, vecty,
        vectz, optimized ? nullptr : tabdip, ntotalm, ntotal, ldabi, nlar, nmax, ndipole,
        nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere,
        nbsphere3, xi, host.xr.data(), host.work.data(), host.field.data(),
        host.incident.data(), host.local_field.data(), host.polarizability.data(),
        niso, method, tolerance, final_residual, products,
        product_calls, max_iterations, &no_plan, &no_plan, &no_plan,
        &no_plan, &no_plan, &no_plan, &no_plan, &no_plan, &optimized,
        &driver_initial_guess, &activate, stop, info, method_length, info_length);
    host.publish();
    const int disable = 0;
    const int disable_status = ifdda_set_resident_chan_preconditioner_all_(
        &disable, nx, ny, nz, nxm, nym, nzm);
    if (disable_status != 0 && *stop == 0)
        set_error(info, info_length, stop,
                  "Unable to deactivate resident right Chan preconditioner");
    return *stop;
}
} // namespace

//-----------------------------------------------------------
extern "C" void inverserigpd_(
    const COMPLEX *gxx, const COMPLEX *gxy, const COMPLEX *gxz,
    const COMPLEX *gyy, const COMPLEX *gyz, const COMPLEX *gzz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz,
    const int *tabdip, const int *ntotalm, const int *ntotal,
    const int *ldabi, const int *nlar, const int *nmax,
    const int *ndipole, const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz, const int *nx2,
    const int *ny2, const int *nxy2, const int *nz2,
    const int *nbsphere, const int *nbsphere3, COMPLEX *xi,
    CUCOMPLEX *xr, CUCOMPLEX *work, CUCOMPLEX *field,
    CUCOMPLEX *incident, CUCOMPLEX *local_field,
    CUCOMPLEX *preconditioned_field, const CUCOMPLEX *polarizability,
    const int *niso, const char *method, const CU_REAL *tolerance,
    CU_REAL *final_residual, int *products, int *product_calls,
    const int *max_iterations, const CU_REAL *k0, const CU_REAL *cell_size,
    const CUCOMPLEX *, const CUCOMPLEX *, const CUCOMPLEX *,
    const CUCOMPLEX *, const CUCOMPLEX *, const CUCOMPLEX *,
    CUCOMPLEX *, const long long *planf, const long long *planb,
    int *stop, char *info, size_t method_length, size_t info_length) {
    run_right_chan(gxx, gxy, gxz, gyy, gyz, gzz, vectx, vecty, vectz,
        tabdip, ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym,
        nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, nbsphere3,
        xi, xr, work, field, incident, local_field, preconditioned_field,
        polarizability, niso, method, tolerance, final_residual, products,
        product_calls, max_iterations, k0, cell_size, nxm, nym, nzm, 0,
        stop, info, method_length, info_length);
    (void)planf;
    (void)planb;
}

//-----------------------------------------------------------
extern "C" void inverserigoptpd_(
    const COMPLEX *gxx, const COMPLEX *gxy, const COMPLEX *gxz,
    const COMPLEX *gyy, const COMPLEX *gyz, const COMPLEX *gzz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz,
    const int *tabdip, const int *ndipole, const int *ntotalm,
    const int *ntotal, const int *ldabi, const int *nlar,
    const int *nmax, const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz, const int *nx2,
    const int *ny2, const int *nxy2, const int *nz2,
    const int *nbsphere, const int *nbsphere3, COMPLEX *xi,
    CUCOMPLEX *xr, CUCOMPLEX *work, CUCOMPLEX *field,
    CUCOMPLEX *incident, CUCOMPLEX *local_field,
    CUCOMPLEX *preconditioned_field, const CUCOMPLEX *polarizability,
    const int *niso, const char *method, const CU_REAL *tolerance,
    CU_REAL *final_residual, int *products, int *product_calls,
    const int *max_iterations, const CU_REAL *k0, const CU_REAL *cell_size,
    const CUCOMPLEX *, const CUCOMPLEX *, const CUCOMPLEX *,
    const CUCOMPLEX *, const CUCOMPLEX *, const CUCOMPLEX *,
    CUCOMPLEX *, const long long *planf, const long long *planb,
    int *stop, char *info, size_t method_length, size_t info_length) {
    run_right_chan(gxx, gxy, gxz, gyy, gyz, gzz, vectx, vecty, vectz,
        tabdip, ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym,
        nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, nbsphere3,
        xi, xr, work, field, incident, local_field, preconditioned_field,
        polarizability, niso, method, tolerance, final_residual, products,
        product_calls, max_iterations, k0, cell_size, nxm, nym, nzm, 1,
        stop, info, method_length, info_length);
    (void)planf;
    (void)planb;
}
