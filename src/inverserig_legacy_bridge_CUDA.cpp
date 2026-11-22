#include "cdmlib_cpp_abi.hpp"
#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"
#include "cuda_solver_interfaces.h"
#include "cuda_solver_host_adapter.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

bool checked_product(size_t a, size_t b, size_t* result)
{
    if (result == nullptr || (b != 0 && a > std::numeric_limits<size_t>::max() / b))
        return false;
    *result = a * b;
    return true;
}

bool convert_to_cuda(const COMPLEX* source, size_t count, std::vector<CUCOMPLEX>* destination)
{
    if (source == nullptr || destination == nullptr)
        return false;
    try {
        destination->resize(count);
    } catch (...) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        (*destination)[i].x = static_cast<CU_REAL>(source[i].real());
        (*destination)[i].y = static_cast<CU_REAL>(source[i].imag());
    }
    return true;
}

void convert_from_cuda(const std::vector<CUCOMPLEX>& source, COMPLEX* destination)
{
    for (size_t i = 0; i < source.size(); ++i)
        destination[i] = COMPLEX(static_cast<REAL>(source[i].x),
                                 static_cast<REAL>(source[i].y));
}

void set_failure(int* nstop, char* info, size_t info_length, const char* message)
{
    if (nstop != nullptr)
        *nstop = 1;
    if (info != nullptr && info_length != 0) {
        std::memset(info, ' ', info_length);
        std::memcpy(info, message, std::min(info_length, std::strlen(message)));
    }
}

void run_legacy_bridge(
    int optimized, int initial_guess_mode, COMPLEX* const tensors[6],
    COMPLEX* const vectors[3],
    int* tabdip, int* ntotalm, int* ntotal, int* lda, int* nlar, int* nmax,
    int* ndipole, int* nxm, int* nym, int* nzm, int* nx, int* ny, int* nz,
    int* nx2, int* ny2, int* nxy2, int* nz2, int* nbsphere, int* nbsphere3,
    COMPLEX* xi, COMPLEX* xr, COMPLEX* work, COMPLEX* field,
    COMPLEX* incident, COMPLEX* local, COMPLEX* polar, int* niso,
    char* method, REAL* tolerance, REAL* final_residual, int* products,
    int* product_calls, int* max_iterations, int64_t* plan_xf,
    int64_t* plan_xb, int64_t* plan_yf, int64_t* plan_yb,
    int64_t* plan_zf, int64_t* plan_zb, int64_t* planf, int64_t* planb,
    int* stop, char* info, size_t method_length, size_t info_length)
{
    if (ntotalm == nullptr || ntotal == nullptr || lda == nullptr || nlar == nullptr ||
        nmax == nullptr || ndipole == nullptr || nxm == nullptr || nym == nullptr ||
        nzm == nullptr || nx == nullptr || ny == nullptr || nz == nullptr ||
        nx2 == nullptr || ny2 == nullptr || nxy2 == nullptr || nz2 == nullptr ||
        nbsphere == nullptr || nbsphere3 == nullptr || niso == nullptr ||
        *ntotalm <= 0 || *lda <= 0 || *nlar <= 0 || *nxm <= 0 || *nym <= 0 ||
        *nzm <= 0 || *nx2 <= 0 || *ny2 <= 0 || *nz2 <= 0 || *niso <= 0 ||
        *nbsphere3 <= 0 || method == nullptr || tolerance == nullptr ||
        final_residual == nullptr || products == nullptr || product_calls == nullptr ||
        max_iterations == nullptr || stop == nullptr || info == nullptr ||
        plan_xf == nullptr || plan_xb == nullptr || plan_yf == nullptr ||
        plan_yb == nullptr || plan_zf == nullptr || plan_zb == nullptr ||
        planf == nullptr || planb == nullptr) {
        set_failure(stop, info, info_length, "Invalid legacy CUDA solver arguments");
        return;
    }

    size_t tensor_count = 0, vector_xy = 0, vector_count = 0;
    size_t field_count = 0, work_count = 0, polar_count = 0;
    if (!checked_product(size_t(*nxm + 1), size_t(*nym + 1), &tensor_count) ||
        !checked_product(tensor_count, size_t(*nzm + 1), &tensor_count) ||
        !checked_product(size_t(*nx2), size_t(*ny2), &vector_xy) ||
        !checked_product(vector_xy, size_t(*nz2), &vector_count) ||
        !checked_product(size_t(*nxm), size_t(*nym), &field_count) ||
        !checked_product(field_count, size_t(*nzm), &field_count) ||
        !checked_product(field_count, size_t(3), &field_count) ||
        !checked_product(size_t(*lda), size_t(*nlar), &work_count) ||
        !checked_product(size_t(*nxm), size_t(*nym), &polar_count) ||
        !checked_product(polar_count, size_t(*nzm), &polar_count) ||
        !checked_product(polar_count, size_t(*niso), &polar_count) ||
        !checked_product(polar_count, size_t(*niso), &polar_count)) {
        set_failure(stop, info, info_length, "Legacy CUDA solver dimensions overflow");
        return;
    }
    std::vector<CUCOMPLEX> vector_cuda[3];
    std::vector<CUCOMPLEX> xi_cuda, xr_cuda, work_cuda, field_cuda;
    std::vector<CUCOMPLEX> incident_cuda, local_cuda, polar_cuda;
    CUCOMPLEX* incident_alias = nullptr;
    bool incident_alias_registered = false;
    for (int i = 0; i < 6; ++i) {
        if (!tensors[i]) {
            set_failure(stop, info, info_length, "Missing legacy Green component key");
            return;
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (!convert_to_cuda(vectors[i], vector_count, &vector_cuda[i])) {
            set_failure(stop, info, info_length, "Legacy CUDA vector conversion failed");
            return;
        }
    }
    if (!convert_to_cuda(xi, field_count, &xi_cuda)) {
        set_failure(stop, info, info_length, "Legacy CUDA solver buffer conversion failed");
        return;
    }
    if (!convert_to_cuda(xr, field_count, &xr_cuda)) {
        set_failure(stop, info, info_length, "Legacy CUDA solver buffer conversion failed");
        return;
    }
    if (!convert_to_cuda(work, work_count, &work_cuda)) {
        set_failure(stop, info, info_length, "Legacy CUDA solver buffer conversion failed");
        return;
    }
    if (!convert_to_cuda(field, field_count, &field_cuda)) {
        set_failure(stop, info, info_length, "Legacy CUDA solver buffer conversion failed");
        return;
    }
    const bool has_incident_alias = nbsphere3 != nullptr && *nbsphere3 > 0 &&
        get_gpbicg_device_b_alias(incident, size_t(*nbsphere3), &incident_alias);
    if (has_incident_alias)
        release_gpbicg_device_b_alias(incident);
    if (!has_incident_alias && !convert_to_cuda(incident, field_count, &incident_cuda)) {
        set_failure(stop, info, info_length, "Legacy CUDA solver buffer conversion failed");
        return;
    }
#ifdef USE_SIMP
    // inverserigopt.f90 normalizes FF0 and Xi using a double-precision norm
    // before any float solver reductions, then restores FF and FFloc. In
    // particular the greentensor RHS has norm ~4.7e25: its unscaled dot
    // products overflow float32. Keep the physical caller's FF0 unchanged.
    double rhs_scale = 1.0;
    if (optimized && !has_incident_alias) {
        double squared_norm = 0.0;
        for (int i = 0; i < *nbsphere3; ++i) {
            const double re = static_cast<double>(incident_cuda[size_t(i)].x);
            const double im = static_cast<double>(incident_cuda[size_t(i)].y);
            squared_norm += re * re + im * im;
        }
        rhs_scale = std::sqrt(squared_norm);
        if (!(rhs_scale > 0.0) || !std::isfinite(rhs_scale)) {
            set_failure(stop, info, info_length, "Non-finite simple precision RHS norm");
            return;
        }
        // The historical CUDA GPBICGSTABL(l) small dense solves are sensitive
        // to a global rescaling of an already safe RHS. Preserve their
        // conditioning for ordinary fields while scaling inputs whose float
        // dot products could overflow (or underflow).
        if (rhs_scale >= 1.0e10 || rhs_scale <= 1.0e-10) {
            for (int i = 0; i < *nbsphere3; ++i) {
                const size_t j = size_t(i);
                incident_cuda[j].x = static_cast<CU_REAL>(double(incident_cuda[j].x) / rhs_scale);
                incident_cuda[j].y = static_cast<CU_REAL>(double(incident_cuda[j].y) / rhs_scale);
                xi_cuda[j].x = static_cast<CU_REAL>(double(xi_cuda[j].x) / rhs_scale);
                xi_cuda[j].y = static_cast<CU_REAL>(double(xi_cuda[j].y) / rhs_scale);
            }
        } else {
            rhs_scale = 1.0;
        }
    }
#endif
    if (!convert_to_cuda(local, field_count, &local_cuda)) {
        set_failure(stop, info, info_length, "Legacy CUDA solver buffer conversion failed");
        return;
    }
    if (!convert_to_cuda(polar, polar_count, &polar_cuda)) {
        set_failure(stop, info, info_length, "Legacy CUDA solver buffer conversion failed");
        return;
    }
    if (has_incident_alias) {
        try {
            incident_cuda.resize(size_t(*nbsphere3));
        } catch (...) {
            set_failure(stop, info, info_length, "Legacy CUDA RHS alias setup failed");
            return;
        }
    }
    const int optimized_product = optimized;
    // Microscopy nrig=0 already owns the incident field on the GPU.  When
    // that alias is available, initialize Xi from the device RHS instead of
    // uploading the temporary host copy created by this legacy bridge.
    int driver_initial_guess_mode = initial_guess_mode;
    if (has_incident_alias && driver_initial_guess_mode < 0)
        driver_initial_guess_mode = 1;
    const bool device_only_field_result = take_device_only_result_vector_request(field);
    static_assert(std::is_same<REAL, CU_REAL>::value,
                  "C++ and CUDA solver precision must match");
    std::vector<COMPLEX> xi_driver(field_count);
    for (size_t i = 0; i < field_count; ++i)
        xi_driver[i] = COMPLEX(static_cast<REAL>(xi_cuda[i].x),
                               static_cast<REAL>(xi_cuda[i].y));
    cuda_solver_host_adapter::InverseDriverArguments host;
    if (!host.initialize_solver_buffers(xr_cuda.data(), work_cuda.data(),
                                        field_cuda.data(), incident_cuda.data(),
                                        local_cuda.data(), polar_cuda.data(),
                                        *nmax, *niso, *lda, *nlar, *nbsphere3)) {
        set_failure(stop, info, info_length, "Legacy CUDA host buffer adaptation failed");
        return;
    }
    if (has_incident_alias) {
        incident_alias_registered = register_gpbicg_device_b_alias(
            host.incident.data(), incident_alias, size_t(*nbsphere3));
        if (!incident_alias_registered) {
            set_failure(stop, info, info_length, "Legacy CUDA RHS alias setup failed");
            return;
        }
    }
    if (device_only_field_result &&
        !request_device_only_result_vector(host.field.data())) {
        if (incident_alias_registered)
            release_gpbicg_device_b_alias(host.incident.data());
        set_failure(stop, info, info_length, "Legacy CUDA result residency request failed");
        return;
    }
    // Both legacy entry points are selected only with nprecon=0.
    const int nprecon = 0;
    // The Fortran ABI exposes 64-bit plan handles as int64_t. CUDA uses
    // long long for its legacy (unused) FFT plan arguments on both platforms.
    // Convert their values rather than aliasing distinct pointer types on LP64.
    const long long cuda_plans[8] = {
        static_cast<long long>(*plan_xf), static_cast<long long>(*plan_xb),
        static_cast<long long>(*plan_yf), static_cast<long long>(*plan_yb),
        static_cast<long long>(*plan_zf), static_cast<long long>(*plan_zb),
        static_cast<long long>(*planf), static_cast<long long>(*planb)
    };
    // Fortran's inverserig/inverserigopt compatibility entry points and the
    // C99 caller now reach exactly the same CUDA inverse driver. The original
    // six CPU keys resolve directly to the six separately allocated GPU
    // Green components; no GPU->CPU->GPU round trip or host tensor repacking.
    inverserig_all_cuda_driver_(
        tensors[0], tensors[1], tensors[2],
        tensors[3], tensors[4], tensors[5],
        vector_cuda[0].data(), vector_cuda[1].data(), vector_cuda[2].data(),
        optimized ? nullptr : tabdip, ntotalm, ntotal, lda, nlar, nmax, ndipole, nxm, nym, nzm,
        nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, nbsphere3,
        xi_driver.data(), host.xr.data(), host.work.data(), host.field.data(),
        host.incident.data(), host.local_field.data(), host.polarizability.data(), niso,
        method, tolerance, final_residual, products, product_calls,
        max_iterations, &cuda_plans[0], &cuda_plans[1], &cuda_plans[2],
        &cuda_plans[3], &cuda_plans[4], &cuda_plans[5],
        &cuda_plans[6], &cuda_plans[7], &optimized_product, &driver_initial_guess_mode,
        &nprecon, stop, info,
        method_length, info_length);

    if (incident_alias_registered)
        release_gpbicg_device_b_alias(host.incident.data());

    host.publish();

    for (size_t i = 0; i < field_count; ++i) {
        xi_cuda[i].x = static_cast<CU_REAL>(xi_driver[i].real());
        xi_cuda[i].y = static_cast<CU_REAL>(xi_driver[i].imag());
    }

    const size_t resident_field_count = nbsphere3 != nullptr && *nbsphere3 > 0
        ? size_t(*nbsphere3) : field_count;
    const CUCOMPLEX* resident_field = device_only_field_result
        ? get_device_result_vector_alias(host.field.data(), resident_field_count) : nullptr;
#ifdef USE_SIMP
    if (resident_field != nullptr && *stop == 0 && rhs_scale != 1.0) {
        if (rhs_scale > double(std::numeric_limits<CU_REAL>::max()) ||
            resident_field_count > size_t(std::numeric_limits<int>::max()) ||
            complex_vector_scale_wrapper(
                resident_field, const_cast<CUCOMPLEX*>(resident_field),
                CUCOMPLEX{static_cast<CU_REAL>(rhs_scale), CU_REAL(0)},
                static_cast<int>(resident_field_count)) != cudaSuccess) {
            set_failure(stop, info, info_length, "CUDA normalized resident field restore failed");
        }
    }
#endif
    if (resident_field != nullptr &&
        !register_device_result_vector_alias(field, resident_field, resident_field_count)) {
        cuda_data::copy_device_complex_to_host(
            field, resident_field, resident_field_count,
            cuda_utils::TransferCategory::Result);
        resident_field = nullptr;
    }
    clear_device_result_vector_alias(host.field.data());

#ifdef USE_SIMP
    if (*stop == 0 && rhs_scale != 1.0) {
        // Restore the physical fields after solving the normalized equation.
        // Xi and work remain solver scratch, matching inverserigopt.f90.
        for (int i = 0; i < *nbsphere3; ++i) {
            const size_t j = size_t(i);
            field_cuda[j].x = static_cast<CU_REAL>(double(field_cuda[j].x) * rhs_scale);
            field_cuda[j].y = static_cast<CU_REAL>(double(field_cuda[j].y) * rhs_scale);
            local_cuda[j].x = static_cast<CU_REAL>(double(local_cuda[j].x) * rhs_scale);
            local_cuda[j].y = static_cast<CU_REAL>(double(local_cuda[j].y) * rhs_scale);
        }
    }
#endif

    convert_from_cuda(xi_cuda, xi);
    convert_from_cuda(xr_cuda, xr);
    convert_from_cuda(work_cuda, work);
    if (resident_field == nullptr)
        convert_from_cuda(field_cuda, field);
    convert_from_cuda(local_cuda, local);
    for (int i = 0; i < 3; ++i)
        convert_from_cuda(vector_cuda[i], vectors[i]);
}

} // namespace

//-----------------------------------------------------------
extern "C" void inverserig_(
    COMPLEX *txx, COMPLEX *txy, COMPLEX *txz, COMPLEX *tyy, COMPLEX *tyz,
    COMPLEX *tzz, COMPLEX *vx, COMPLEX *vy, COMPLEX *vz, int *tabdip,
    int *ntotalm, int *ntotal, int *lda, int *nlar, int *nmax, int *ndipole,
    int *nxm, int *nym, int *nzm, int *nx, int *ny, int *nz, int *nx2,
    int *ny2, int *nxy2, int *nz2, int *nbsphere, int *nbsphere3,
    COMPLEX *xi, COMPLEX *xr, COMPLEX *work, COMPLEX *field,
    COMPLEX *incident, COMPLEX *local, COMPLEX *polar, int *niso,
    char *method, REAL *tolerance, REAL *final_residual, int *products,
    int *product_calls, int *max_iterations, int64_t *plan_xf,
    int64_t *plan_xb, int64_t *plan_yf, int64_t *plan_yb,
    int64_t *plan_zf, int64_t *plan_zb, int64_t *planf, int64_t *planb,
    int *stop, char *info, size_t method_length, size_t info_length)
{
    COMPLEX* tensors[6] = {txx, txy, txz, tyy, tyz, tzz};
    COMPLEX* vectors[3] = {vx, vy, vz};
    run_legacy_bridge(0, -1, tensors, vectors, tabdip, ntotalm, ntotal, lda, nlar,
        nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,
        nbsphere, nbsphere3, xi, xr, work, field, incident, local, polar,
        niso, method, tolerance, final_residual, products, product_calls,
        max_iterations, plan_xf, plan_xb, plan_yf, plan_yb, plan_zf, plan_zb,
        planf, planb, stop, info, method_length, info_length);
}

//-----------------------------------------------------------
// Fortran calls inverserigopt(...), which the Fortran ABI resolves to
// inverserigopt_. Keep that exact public signature: the optimized dense
// path has neither a tabdip parameter nor an ndipole parameter.
extern "C" void inverserigopt_(
    COMPLEX *txx, COMPLEX *txy, COMPLEX *txz, COMPLEX *tyy, COMPLEX *tyz,
    COMPLEX *tzz, COMPLEX *vx, COMPLEX *vy, COMPLEX *vz,
    int *ntotalm, int *ntotal, int *lda, int *nlar, int *nmax,
    int *nxm, int *nym, int *nzm, int *nx, int *ny, int *nz,
    int *nx2, int *ny2, int *nxy2, int *nz2, int *nbsphere, int *nbsphere3,
    COMPLEX *xi, COMPLEX *xr, COMPLEX *work, COMPLEX *field,
    COMPLEX *incident, COMPLEX *local, COMPLEX *polar, int *niso,
    char *method, REAL *tolerance, REAL *final_residual, int *products,
    int *product_calls, int *max_iterations, int64_t *plan_xf,
    int64_t *plan_xb, int64_t *plan_yf, int64_t *plan_yb,
    int64_t *plan_zf, int64_t *plan_zb, int64_t *planf, int64_t *planb,
    int *stop, char *info, size_t method_length, size_t info_length)
{
    COMPLEX* tensors[6] = {txx, txy, txz, tyy, tyz, tzz};
    COMPLEX* vectors[3] = {vx, vy, vz};
    // For dense geometry, the number of occupied dipoles is nbsphere.
    // The common driver reconstructs its padded map when tabdip is null.
    int dense_ndipole = nbsphere ? *nbsphere : 0;
    run_legacy_bridge(1, -1, tensors, vectors, nullptr,
        ntotalm, ntotal, lda, nlar, nmax, &dense_ndipole,
        nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,
        nbsphere, nbsphere3, xi, xr, work, field, incident, local, polar,
        niso, method, tolerance, final_residual, products, product_calls,
        max_iterations, plan_xf, plan_xb, plan_yf, plan_yb, plan_zf, plan_zb,
        planf, planb, stop, info, method_length, info_length);
}

//-----------------------------------------------------------
extern "C" void inverserigopt_with_tabdip_cuda_(
    COMPLEX *txx, COMPLEX *txy, COMPLEX *txz, COMPLEX *tyy, COMPLEX *tyz,
    COMPLEX *tzz, COMPLEX *vx, COMPLEX *vy, COMPLEX *vz, int *ntotalm,
    int *ntotal, int *lda, int *nlar, int *nmax, int *nxm, int *nym,
    int *nzm, int *nx, int *ny, int *nz, int *nx2, int *ny2, int *nxy2,
    int *nz2, int *nbsphere, int *nbsphere3, int *ndipole, int *tabdip,
    COMPLEX *xi, COMPLEX *xr, COMPLEX *work, COMPLEX *field,
    COMPLEX *incident, COMPLEX *local, COMPLEX *polar, int *niso,
    char *method, REAL *tolerance, REAL *final_residual, int *products,
    int *product_calls, int *max_iterations, int64_t *plan_xf,
    int64_t *plan_xb, int64_t *plan_yf, int64_t *plan_yb,
    int64_t *plan_zf, int64_t *plan_zb, int64_t *planf, int64_t *planb,
    int *stop, char *info, size_t method_length, size_t info_length)
{
    COMPLEX* tensors[6] = {txx, txy, txz, tyy, tyz, tzz};
    COMPLEX* vectors[3] = {vx, vy, vz};
    run_legacy_bridge(1, 1, tensors, vectors, tabdip, ntotalm, ntotal, lda, nlar,
        nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,
        nbsphere, nbsphere3, xi, xr, work, field, incident, local, polar,
        niso, method, tolerance, final_residual, products, product_calls,
        max_iterations, plan_xf, plan_xb, plan_yf, plan_yb, plan_zf, plan_zb,
        planf, planb, stop, info, method_length, info_length);
}
