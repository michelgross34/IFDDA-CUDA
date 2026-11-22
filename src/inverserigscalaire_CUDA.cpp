#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>














namespace {

using StandardSolver = void (*)(COMPLEX *, COMPLEX *, COMPLEX *, const int *, const int *,
                                const int *, int *, COMPLEX *, int *, const int *,
                                const CU_REAL *, CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                                CUCOMPLEX *, CUCOMPLEX *, int *, int *);

void run_ssbicgsafe2(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                     const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                     const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                     CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn,
                     int *nstat, int *steperr) {
    ssbicgsafe2_(xi, xr, b, lda, ndim, nlar, nou, work, itno, maxit, tol, norm,
                 alpha, beta, eta, dzeta, r0rn, nstat, steperr);
}

//-----------------------------------------------------------
StandardSolver select_standard_solver(const char *method) {
    if (std::strncmp(method, "GPBICG1", 7) == 0)
        return gpbicg_;
    if (std::strncmp(method, "GPBICG2", 7) == 0)
        return gpbicg2_;
    if (std::strncmp(method, "GPBICGplus", 10) == 0)
        return gpbicgplus_;
    if (std::strncmp(method, "GPBICGsafe", 10) == 0)
        return gpbicgsafe_;
    if (std::strncmp(method, "GPBICGAR2", 9) == 0)
        return gpbicgar2_;
    if (std::strncmp(method, "GPBICGAR", 8) == 0)
        return gpbicgar_;
    if (std::strncmp(method, "BICGSTARPLUS", 12) == 0)
        return gpbicgstarplus_;
    if (std::strncmp(method, "SSBICGSAFE2", 11) == 0)
        return run_ssbicgsafe2;
    return nullptr;
}

struct ScalarDriverBuffers {
    size_t capacity = 0;
    size_t fft_capacity = 0;
    cuda_data::ScalarDriverDeviceBuffers device{};
};

ScalarDriverBuffers buffers;

//-----------------------------------------------------------
bool ensure_buffers(size_t count, size_t fft_count) {
    if (buffers.capacity >= count && buffers.fft_capacity >= fft_count)
        return true;
    // Named workspace buffers live in cuda_data and are reused across solves;
    // growing a problem no longer frees the previous pointers in this driver.
    if (!cuda_data::get_device_scalar_driver_buffers(count, fft_count, count, &buffers.device) ||
        !buffers.device.incident || !buffers.device.polarizability || !buffers.device.fft_work ||
        !buffers.device.scalar_solution || !buffers.device.local_solution ||
        !buffers.device.vector_field || !buffers.device.vector_local_field)
        return false;
    buffers.capacity = count;
    buffers.fft_capacity = fft_count;
    return true;
}

//-----------------------------------------------------------
void set_error(int *stop, char *info, size_t info_length, const char *message) {
    if (stop)
        *stop = 1;
    if (!info || info_length == 0)
        return;
    std::memset(info, ' ', info_length);
    const size_t length = std::min(info_length, std::strlen(message));
    std::memcpy(info, message, length);
}

//-----------------------------------------------------------
bool scalar_product_device(const CUCOMPLEX *green, const CU_INT *tabdip, const CUCOMPLEX *input,
                           CUCOMPLEX *output, CUCOMPLEX *fft_work, int ntotalm, int ntotal,
                           int nmax, int nx, int ny, int nz, int nx2, int ny2, int nz2, int nxy2,
                           bool optimized_product) {
    const size_t physical_count = size_t(nx) * size_t(ny) * size_t(nz);
    if (!cuda_utils::check_cuda(cudaMemset(fft_work, 0, size_t(ntotal) * sizeof(CUCOMPLEX)),
                                "scalar inverse driver", "clear scalar FFT workspace"))
        return false;
    const cufftHandle plan = get_fft_plan(nx2, ny2, nz2);
    // The optimized (dense) Green product has no dipole gather/scatter map.
    // Both variants operate directly on resident GPU vectors; only the FFT
    // layout differs. No solver iteration transfers a field through the CPU.
    const cudaError_t gather_status = optimized_product
        ? gather_scalar_product_dense_wrapper(input, fft_work, nx, ny, nz, nx2, nxy2)
        : gather_scalar_product_wrapper(input, tabdip, fft_work, nx, ny, nz, nx2, nxy2, nmax);
    return plan &&
           cuda_utils::check_cuda(gather_status, "scalar inverse driver", "gather scalar field") &&
           cuda_utils::check_cufft(CUFFT_EXECUTE(plan, fft_work, fft_work, CUFFT_INVERSE),
                                   "scalar inverse driver", "inverse cuFFT") &&
           cuda_utils::check_cuda(multiply_scalar_green_wrapper(green, fft_work, ntotal),
                                  "scalar inverse driver", "multiply scalar Green") &&
           cuda_utils::check_cufft(CUFFT_EXECUTE(plan, fft_work, fft_work, CUFFT_FORWARD),
                                   "scalar inverse driver", "forward cuFFT") &&
           cuda_utils::check_cuda(optimized_product
                                      ? scatter_scalar_product_dense_wrapper(
                                            fft_work, output, nx, ny, nz, nx2, nxy2, ntotal)
                                      : scatter_scalar_product_wrapper(
                                            fft_work, tabdip, output, nx, ny, nz, nx2, nxy2,
                                            nmax, ntotal),
                                  "scalar inverse driver", "scatter scalar product") &&
           physical_count <= size_t(ntotalm);
}

struct ResidentBuffersGuard {
    const COMPLEX *xi;
    const COMPLEX *work;
    const COMPLEX *rhs;
//-------------------------------------------

    ~ResidentBuffersGuard() {
        set_gpbicg_device_resident(xi, false);
        set_gpbicg_device_resident(work, false);
        set_gpbicg_device_b_resident(rhs, false);
    }
};

} // namespace

//-----------------------------------------------------------
extern "C" void inverserigscalaire_cuda_driver_(
    COMPLEX *vectx, COMPLEX *vecty, const int *tabdip, COMPLEX *ffscal, COMPLEX *fflocscal,
    const COMPLEX *uincx, const COMPLEX *uincy, const COMPLEX *uincz, const int *ntotalm,
    const int *ntotal, const int *ldabi, const int *nlar, const int *nmax, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm, const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2, const int *nbsphere,
    COMPLEX *xi, COMPLEX *xr, COMPLEX *work, COMPLEX *ff, const COMPLEX *ff0,
    COMPLEX *ffloc, const COMPLEX *polarizability, const int *niso, const char *method,
    const CU_REAL *tolerance, CU_REAL *final_residual, int *iterations, int *product_calls,
    const int *max_iterations, const int *preconditioner, const int *nrig, const int *initial_guess,
    const int *optimized_product,
    const CU_REAL *k0, const CU_REAL *aretecube, const long long *planf, const long long *planb,
    int *stop, char *info, size_t method_length, size_t info_length) {
    (void)vecty;
    (void)ndipole;
    (void)nxm;
    (void)nym;
    (void)nzm;
    (void)planf;
    (void)planb;

    const bool is_cors = method && std::strncmp(method, "CORS", 4) == 0;
    const bool is_zcg =
        method && (std::strncmp(method, "CG", 2) == 0 || std::strncmp(method, "ZCG", 3) == 0);
    const bool is_tfqmr = method && std::strncmp(method, "TFQMR", 5) == 0;
    const bool is_pimzbicgstab = method && std::strncmp(method, "BICGSTAB", 8) == 0;
    const bool is_qmrbicgstab = method && std::strncmp(method, "QMRBICGSTAB", 11) == 0;
    const bool is_gpbicor = method && std::strncmp(method, "GPBICOR", 7) == 0;
    const bool is_idrs = method && std::strncmp(method, "IDRS", 4) == 0;
    const bool is_bicglstab = method && std::strncmp(method, "BICGLSTAB", 9) == 0;
    const bool is_gpbicgstabl = method && std::strncmp(method, "GPBICGSTABL", 11) == 0;
    const bool is_pimzqmr = method && std::strncmp(method, "QMRCLA", 6) == 0;
    if (!vectx || !tabdip || !ffscal || !fflocscal || !uincx || !uincy || !uincz || !ntotalm ||
        !ntotal || !ldabi || !nlar || !nmax || !nx || !ny || !nz || !nx2 || !ny2 || !nxy2 || !nz2 ||
        !nbsphere || !xi || !xr || !work || !ff || !ff0 || !ffloc || !polarizability || !niso ||
        !method || method_length < 4 || !tolerance || !final_residual || !product_calls ||
        !iterations || !max_iterations || !preconditioner || !nrig || !initial_guess ||
        !optimized_product || (*optimized_product != 0 && *optimized_product != 1) || !k0 ||
        !aretecube || !stop || *nbsphere <= 0 || *nlar < 12 || *ldabi < *nbsphere || *niso <= 0 ||
        (*preconditioner < 0 || *preconditioner > 2) ||
        (*preconditioner != 0 && *nrig != 8 && *nrig != 9) || *ntotal != *nx2 * *ny2 * *nz2 ||
        *nxy2 != *nx2 * *ny2 ||
        (!is_cors && !is_zcg && !is_tfqmr && !is_pimzbicgstab && !is_qmrbicgstab && !is_gpbicor &&
         !is_idrs && !is_bicglstab && !is_gpbicgstabl && !is_pimzqmr &&
         !select_standard_solver(method))) {
        set_error(stop, info, info_length, "GPU scalar driver does not support this solver yet");
        return;
    }

    const size_t count = size_t(*nbsphere);
    const size_t work_count = size_t(*ldabi) * size_t(*nlar);
    if (count > size_t(std::numeric_limits<int>::max()) ||
        work_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        !initialize_cuda_device() || !ensure_buffers(count, size_t(*ntotal))) {
        set_error(stop, info, info_length, "Unable to initialize GPU scalar driver buffers");
        return;
    }

    // Dense gather/scatter processes every cell of the physical mesh. For a
    // sparse object the Krylov vectors contain only occupied dipoles, so the
    // mapped GPU kernels are required to avoid out-of-bounds accesses.
    const size_t physical_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
    const bool dense_product = *optimized_product != 0 && count >= physical_count;

    // Acquire Green and the optional dipole map once, outside the Krylov loop.
    const CUCOMPLEX *device_green = get_device_scalar_green(vectx, size_t(*ntotal));
    const CU_INT *device_tabdip = dense_product ? nullptr : get_device_tabdip(tabdip, physical_count);
    if (!device_green || (!dense_product && !device_tabdip)) {
        set_error(stop, info, info_length, "Unable to acquire scalar Green/map on GPU");
        return;
    }

    std::fill(ff, ff + count, COMPLEX(CU_REAL(0), CU_REAL(0)));

    CUCOMPLEX *gpu_xi = nullptr, *gpu_xr = nullptr;
    CUCOMPLEX *gpu_rhs = nullptr, *gpu_work = nullptr;
    CUCOMPLEX *gpu_polarizability = copy_polarizability_to_device(polarizability, count);
    const bool build_initial_guess_on_gpu = *initial_guess == 0 || *initial_guess == 1 ||
                                            *initial_guess == 2 || *initial_guess == 3 ||
                                            *initial_guess == 4 || *initial_guess == 5;
    if (!gpu_polarizability ||
        !get_gpbicg_device_arrays_resident(xi, xr, ff, work, count, work_count,
                                           !build_initial_guess_on_gpu, &gpu_xi, &gpu_xr, &gpu_rhs,
                                           &gpu_work) ||
        cuda_data::copy_host_to_device(
            buffers.device.incident, ff0, size_t(3) * count * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::IncidentField) != cudaSuccess ||
        cuda_data::copy_host_to_device(buffers.device.polarizability, polarizability,
                                       //-----------------------------------------------------------
                                       count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Material) != cudaSuccess) {
        set_error(stop, info, info_length, "Failed to upload scalar driver inputs");
        return;
    }

    if ((*initial_guess == 0 || *initial_guess == 4 || *initial_guess == 5) &&
        cudaMemset(gpu_xi, 0, count * sizeof(CUCOMPLEX)) != cudaSuccess) {
        set_error(stop, info, info_length, "Failed to initialize the scalar GPU guess to zero");
        return;
    }
    if (*initial_guess == 0 || *initial_guess == 4 || *initial_guess == 5)
        mark_gpbicg_device_xi_current(xi, true);
    if (*initial_guess == 1 &&
        scalar_driver_project_incident_wrapper(
            buffers.device.incident, gpu_xi, cuda_utils::device_value(*uincx),
            cuda_utils::device_value(*uincy), cuda_utils::device_value(*uincz), int(count)) !=
            cudaSuccess) {
        set_error(stop, info, info_length, "Failed to build the scalar Born guess on GPU");
        return;
    }
    if (*initial_guess == 1)
        mark_gpbicg_device_xi_current(xi, true);
    if (*initial_guess == 3) {
        if (std::getenv("IFDDA_TRACE_BPM_RESIDENT"))
            std::fprintf(stderr, "[scalar BPM resident] GPU field=%p count=%zu\n",
                         static_cast<void *>(buffers.device.vector_field), count);
        const bool trace_requires_host_handoff = std::getenv("IFDDA_BPM_TRACE") != nullptr;
        if ((trace_requires_host_handoff &&
             cuda_data::copy_host_to_device(
                 buffers.device.vector_field, xr, size_t(3) * count * sizeof(CUCOMPLEX),
                 cuda_utils::TransferCategory::IncidentField) != cudaSuccess) ||
            scalar_driver_project_incident_wrapper(
                buffers.device.vector_field, gpu_xi, cuda_utils::device_value(*uincx),
                cuda_utils::device_value(*uincy), cuda_utils::device_value(*uincz), int(count)) !=
            cudaSuccess) {
            set_error(stop, info, info_length, "Failed to project the scalar BPM guess on GPU");
            return;
        }
        mark_gpbicg_device_xi_current(xi, true);
    }
    if (*initial_guess == 2) {
        if (scalar_driver_project_incident_wrapper(
                buffers.device.incident, gpu_rhs, cuda_utils::device_value(*uincx),
                cuda_utils::device_value(*uincy), cuda_utils::device_value(*uincz), int(count)) !=
                cudaSuccess ||
            scalar_driver_apply_polarizability_wrapper(gpu_rhs, gpu_polarizability,
                                                       gpu_xi, int(count)) != cudaSuccess ||
            complex_vector_scale_wrapper(gpu_xi, gpu_xi, CUFFT_REAL(CU_REAL(-1)), int(count)) !=
                cudaSuccess ||
            !scalar_product_device(device_green, device_tabdip, gpu_xi, gpu_xr,
                                   buffers.device.fft_work, *ntotalm, *ntotal, *nmax, *nx, *ny, *nz,
                                   *nx2, *ny2, *nz2, *nxy2, dense_product) ||
            scalar_driver_exponential_guess_wrapper(gpu_rhs, gpu_xr, gpu_xi, int(count)) !=
                cudaSuccess) {
            set_error(stop, info, info_length, "Failed to build scalar exponential guess on GPU");
            return;
        }
        mark_gpbicg_device_xi_current(xi, true);
    }

    set_gpbicg_device_resident(xi, true);
    set_gpbicg_device_resident(work, true);
    set_gpbicg_device_b_resident(ff, true);
    ResidentBuffersGuard resident_guard{xi, work, ff};
    if (cudaMemset(gpu_work, 0, work_count * sizeof(CUCOMPLEX)) != cudaSuccess ||
        scalar_driver_project_incident_wrapper(
            buffers.device.incident, gpu_rhs, cuda_utils::device_value(*uincx),
            cuda_utils::device_value(*uincy), cuda_utils::device_value(*uincz), int(count)) !=
            cudaSuccess) {
        set_error(stop, info, info_length, "Failed to initialize GPU scalar right-hand side");
        return;
    }
    const CUCOMPLEX incident_x = cuda_utils::device_value(*uincx);
    const CUCOMPLEX incident_y = cuda_utils::device_value(*uincy);
    const CUCOMPLEX incident_z = cuda_utils::device_value(*uincz);
    bool preconditioner_ready = true;
    if (*preconditioner == 1) {
        preconditioner_ready = ifdda_scalar_chan_prepare_resident(
                                   gpu_polarizability, int(count), &incident_x, &incident_y,
                                   &incident_z, *aretecube, *k0,
                                   *nrig, *nx, *ny, *nz, *nxm, *nym, *nzm, 1) &&
                               ifdda_scalar_chan_apply_resident(gpu_rhs, int(count), tabdip, 1);
    } else if (*preconditioner == 2) {
        preconditioner_ready = ifdda_scalar_chan_prepare_resident(
                                   gpu_polarizability, int(count), &incident_x, &incident_y,
                                   &incident_z, *aretecube, *k0,
                                   *nrig, *nx, *ny, *nz, *nxm, *nym, *nzm, 0) &&
                               ifdda_scalar_chan_apply_resident(gpu_xi, int(count), tabdip, 0);
        if (preconditioner_ready) {
            // Chan has updated the resident right-preconditioned initial
            // guess. Preserve it across the solver's begin-solve acquisition
            // instead of round-tripping it through the host Xi buffer.
            mark_gpbicg_device_xi_current(xi, true);
            preconditioner_ready = ifdda_scalar_chan_prepare_resident(
                gpu_polarizability, int(count), &incident_x, &incident_y, &incident_z, *aretecube,
                *k0, *nrig, *nx, *ny,
                *nz, *nxm, *nym, *nzm, 1);
        }
    }
    if (!preconditioner_ready) {
        set_error(stop, info, info_length,
                  "GPU scalar left Chan preconditioner setup/application failed");
        return;
    }

    CUCOMPLEX alpha = CUFFT_ZERO, beta = CUFFT_ZERO, eta = CUFFT_ZERO;
    CUCOMPLEX dzeta = CUFFT_ZERO, r0rn = CUFFT_ZERO;
    CUCOMPLEX rho = CUFFT_ZERO, rhoa = CUFFT_ZERO;
    CUCOMPLEX alphat = CUFFT_ZERO, rhot = CUFFT_ZERO;
    CUCOMPLEX zcg_rho0 = CUFFT_ZERO, zcg_xxi = CUFFT_ZERO;
    CUCOMPLEX tfqmr_rho = CUFFT_ZERO, tfqmr_tau = CUFFT_ZERO;
    CUCOMPLEX tfqmr_theta = CUFFT_ZERO, tfqmr_eta = CUFFT_ZERO;
    COMPLEX pimz_alpha = COMPLEX(CU_REAL(0), CU_REAL(0));
    COMPLEX pimz_omega = COMPLEX(CU_REAL(0), CU_REAL(0));
    COMPLEX pimz_rho = COMPLEX(CU_REAL(0), CU_REAL(0));
    CUCOMPLEX qmr_rho = CUFFT_ZERO, qmr_omega = CUFFT_ZERO;
    CUCOMPLEX qmr_tau = CUFFT_ZERO, qmr_taut = CUFFT_ZERO;
    CUCOMPLEX qmr_theta = CUFFT_ZERO, qmr_thetat = CUFFT_ZERO;
    CUCOMPLEX qmr_eta = CUFFT_ZERO, qmr_etat = CUFFT_ZERO;
    CU_REAL qmr_residual = CU_REAL(0);
    CUCOMPLEX idrs_omega = CUFFT_ZERO;
    int idrs_s = method && std::strncmp(method, "IDRS8", 5) == 0
                     ? 8
                     : (method && std::strncmp(method, "IDRS4", 5) == 0 ? 4 : 2);
    int idrs_is = 0;
    const int bicglstab_l = method && std::strncmp(method, "BICGLSTAB8", 10) == 0
                                ? 8
                                : (method && std::strncmp(method, "BICGLSTAB4", 10) == 0 ? 4 : 2);
    int bicglstab_nr = 0, bicglstab_nu = 0, bicglstab_j = 0;
    CUCOMPLEX bicglstab_omega = CUFFT_ZERO;
    const int gpbicgstabl_l =
        method && std::strncmp(method, "GPBICGSTABL8", 12) == 0
            ? 8
            : (method && std::strncmp(method, "GPBICGSTABL4", 12) == 0 ? 4 : 2);
    CUCOMPLEX gpbicgstabl_sigma = CUFFT_ZERO;
    CUCOMPLEX gpbicgstabl_rho = CUFFT_ZERO;
    CU_REAL gpbicgstabl_tole = CU_REAL(0);
    int gpbicgstabl_nr = 0, gpbicgstabl_np = 0, gpbicgstabl_ns = 0;
    int gpbicgstabl_nq = 0, gpbicgstabl_j = 0;
    CUCOMPLEX qmr_lambda = CUFFT_ZERO, qmr_kappa = CUFFT_ZERO;
    CUCOMPLEX qmrcla_theta = CUFFT_ZERO, qmrcla_gamma = CUFFT_ZERO;
    CUCOMPLEX qmr_ksi = CUFFT_ZERO, qmrcla_rho = CUFFT_ZERO;
    CUCOMPLEX qmr_epsilon = CUFFT_ZERO, qmr_mu = CUFFT_ZERO;
    CUCOMPLEX qmrcla_tau = CUFFT_ZERO, qmr_dots1 = CUFFT_ZERO;
    CUCOMPLEX qmr_dots2 = CUFFT_ZERO, qmr_dots3 = CUFFT_ZERO;
    CUCOMPLEX qmr_dots4 = CUFFT_ZERO;
    CU_REAL qmr_tolet = CU_REAL(0);
    int qmr_nt = 1;
    const int qmr_method = method && std::strncmp(method, "QMRBICGSTAB2", 12) == 0 ? 2 : 1;
    CU_REAL rhs_norm = CU_REAL(0);
    CU_REAL zcg_tole = CU_REAL(0);
    int zcg_iterations = 0;
    const StandardSolver solver = select_standard_solver(method);
    int nou = 0, itno = 0, nstat = 0, steperr = -1;
    while (true) {
        if (std::strncmp(method, "CORS", 4) == 0) {
            cors_(xi, xr, ff, ldabi, nbsphere, nlar, &nou, work, &itno, max_iterations, tolerance,
                  &rhs_norm, &alpha, &rho, &rhoa, &beta, &nstat, &steperr);
        } else if (is_zcg) {
            zcg_(xi, xr, ff, &rhs_norm, work, &alpha, &zcg_rho0, &zcg_xxi, ldabi, nbsphere, nlar,
                 &nou, &nstat, &zcg_iterations, max_iterations, &zcg_tole, tolerance);
        } else if (is_tfqmr) {
            tfqmr_(ffloc, xi, xr, ff, ldabi, nbsphere, nlar, &nou, work, &itno, max_iterations,
                   tolerance, &rhs_norm, &alpha, &beta, &tfqmr_rho, &tfqmr_tau, &tfqmr_theta,
                   &tfqmr_eta, &nstat, &steperr);
        } else if (is_pimzbicgstab) {
            pimzbicgstab_(ffloc, xi, xr, ff, ldabi, nlar, nbsphere, &nou, work, &pimz_alpha,
                          &pimz_omega,
                          &pimz_rho, &rhs_norm, tolerance, &itno, max_iterations, &nstat, &steperr);
        } else if (is_qmrbicgstab) {
            qmrbicgstab_(ffloc, xi, xr, ff, ldabi, nbsphere, nlar, &nou, work, &itno,
                         max_iterations, tolerance, &qmr_residual, &rhs_norm, &alpha, &qmr_rho,
                         &qmr_omega, &qmr_tau, &qmr_taut, &qmr_theta, &qmr_thetat, &qmr_eta,
                         &qmr_etat, &qmr_method, &nstat, &steperr);
        } else if (is_gpbicor) {
            gpbicor_(xi, xr, ff, ffloc, ldabi, nbsphere, nlar, &nou, work, &itno, max_iterations,
                     tolerance, &rhs_norm, &alpha, &beta, &eta, &dzeta, &rho, &rhoa, &nstat,
                     &steperr);
        } else if (is_idrs) {
            idrs_(ffloc, xi, xr, ff, ldabi, nbsphere, nlar, &nou, &idrs_s, work, &itno,
                  max_iterations, tolerance, &qmr_residual, &rhs_norm, &idrs_omega, &idrs_is,
                  &nstat, &steperr);
        } else if (is_bicglstab) {
            zbicgstabl_(xi, xr, ff, ffloc, &rhs_norm, work, &alpha, &rho, &bicglstab_omega,
                        &bicglstab_l, &bicglstab_nr, &bicglstab_nu, &bicglstab_j, ldabi, nbsphere,
                        nlar, &nou, &nstat, &steperr, &itno, max_iterations, tolerance, tolerance);
        } else if (is_gpbicgstabl) {
            gpbicgstabl_(xi, xr, ff, ffloc, &rhs_norm, work, &alpha, &beta, &gpbicgstabl_sigma,
                         &gpbicgstabl_rho, &gpbicgstabl_l, ldabi, nbsphere, nlar, &nou,
                         &gpbicgstabl_nr, &gpbicgstabl_np, &gpbicgstabl_ns, &gpbicgstabl_nq,
                         &gpbicgstabl_j, &nstat, &steperr, &itno, max_iterations, &gpbicgstabl_tole,
                         tolerance);
        } else if (is_pimzqmr) {
            pimzqmr_(ffloc, xi, xr, ff, work, &rhs_norm, ldabi, nbsphere, nlar, &qmr_lambda,
                     &qmr_kappa, &qmrcla_theta, &qmrcla_gamma, &qmr_ksi, &qmrcla_rho, &qmr_epsilon,
                     &qmr_mu, &qmrcla_tau, &qmr_dots1, &qmr_dots2, &qmr_dots3, &qmr_dots4, &nou,
                     &qmr_nt, iterations, max_iterations, &qmr_tolet, tolerance, &nstat, &steperr);
        } else {
            solver(xi, xr, ff, ldabi, nbsphere, nlar, &nou, work, &itno, max_iterations, tolerance,
                   &rhs_norm, &alpha, &beta, &eta, &dzeta, &r0rn, &nstat, &steperr);
        }
        if (nstat < 0) {
            char solver_error[128];
            std::snprintf(solver_error, sizeof(solver_error),
                          "CUDA scalar solver failed (nstat=%d, steperr=%d, method=%.*s)", nstat,
                          steperr, int(std::min(method_length, size_t(48))), method);
            set_error(stop, info, info_length, solver_error);
            return;
        }
        ++(*product_calls);
        if (is_pimzqmr && nstat == 1)
            qmr_nt = 1;

        if (!get_gpbicg_device_arrays_resident(xi, xr, ff, work, count, work_count, false, &gpu_xi,
                                               &gpu_xr, &gpu_rhs, &gpu_work)) {
            set_error(stop, info, info_length, "Failed to access resident GPU scalar vectors");
            return;
        }
        if (nstat == 1 && (is_tfqmr || is_pimzbicgstab || is_qmrbicgstab || is_idrs ||
                           is_bicglstab || is_gpbicgstabl || is_pimzqmr)) {
            CUCOMPLEX *gpu_tfqmr_solution = get_tfqmr_device_x(ffloc, xi, count, false);
            if (!gpu_tfqmr_solution ||
                cuda_utils::tracked_memcpy(gpu_xi, gpu_tfqmr_solution, count * sizeof(CUCOMPLEX),
                                           cudaMemcpyDeviceToDevice) != cudaSuccess ||
                scalar_driver_copy_solution_wrapper(
                    gpu_tfqmr_solution, buffers.device.local_solution, int(count)) != cudaSuccess) {
                set_error(stop, info, info_length,
                          "Failed to publish external scalar solver solution");
                return;
            }
        } else if (nstat == 1 &&
//-------------------------------------------
                   scalar_driver_copy_solution_wrapper(gpu_xi, buffers.device.local_solution,
                                                       int(count)) != cudaSuccess) {
            set_error(stop, info, info_length, "Failed to save converged scalar field");
            return;
        }
        if (is_pimzqmr && qmr_nt == 2) {
            const CUCOMPLEX one = CUFFT_REAL(CU_REAL(1));
            cublasHandle_t handle = cuda_utils::get_cublas_handle();
            const CUCOMPLEX *transpose_input = gpu_xi;
            CUCOMPLEX *gpu_transpose_input = nullptr;
            if (*preconditioner == 1) {
                // For B=P^-1*A, QMRCLA requests B^T*x=A^T*P^-T*x.
                // Chan blocks are complex symmetric, so P^-T=P^-1. Stage
                // the transformed vector in an unused solver-work column;
                // keep this transpose input resident on the GPU.
                gpu_transpose_input = gpu_work + size_t(10) * size_t(*ldabi);
                if (cuda_utils::tracked_memcpy(
                        gpu_transpose_input, gpu_xi,
                        //-----------------------------------------------------------
                        count * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice) != cudaSuccess ||
                    !ifdda_scalar_chan_apply_resident(gpu_transpose_input, int(count), tabdip, 1)) {
                    set_error(stop, info, info_length,
                              "GPU scalar left Chan transpose input failed");
                    return;
                }
                transpose_input = gpu_transpose_input;
            } else if (*preconditioner == 2 &&
                       cudaMemset(gpu_xr, 0, count * sizeof(CUCOMPLEX)) != cudaSuccess) {
                set_error(stop, info, info_length,
                          "GPU scalar right Chan transpose output initialization failed");
                return;
            }
            if (!scalar_product_device(
                    device_green, device_tabdip, transpose_input, gpu_xr,
                    buffers.device.fft_work, *ntotalm, *ntotal, *nmax, *nx,
                    *ny, *nz, *nx2, *ny2, *nz2, *nxy2, dense_product) ||
                scalar_driver_apply_polarizability_wrapper(gpu_xr, buffers.device.polarizability,
                                                           gpu_xr, int(count)) != cudaSuccess ||
                !handle ||
                !cuda_utils::axpy(handle, int(count), one,
                                  (*preconditioner == 1) ? transpose_input : gpu_xi, gpu_xr,
                                  "scalar QMRCLA transpose product")) {
                set_error(stop, info, info_length, "CUDA QMRCLA transpose scalar product failed");
                return;
            }
            if (*preconditioner == 2 &&
                !ifdda_scalar_chan_apply_resident(gpu_xr, int(count), tabdip, 1)) {
                set_error(stop, info, info_length, "GPU scalar right Chan transpose output failed");
                return;
            }
        } else {
            if (scalar_driver_copy_solution_wrapper(gpu_xi, gpu_xr, int(count)) != cudaSuccess) {
                set_error(stop, info, info_length, "GPU scalar solution copy failed");
                return;
            }
            CUCOMPLEX *polarizability_input = gpu_xr;
            if (is_pimzqmr && *preconditioner == 2) {
                // QMRCLA solves the right-preconditioned system A*P^-1*y=b.
                // Form P^-1*y once and use it both as the identity contribution
                // and as the input to alpha and the Green product.
                CUCOMPLEX *gpu_right_preconditioned = gpu_work + size_t(10) * size_t(*ldabi);
                if (cuda_utils::tracked_memcpy(
                        gpu_right_preconditioned, gpu_xr,
                        //-----------------------------------------------------------
                        count * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice) != cudaSuccess ||
                    !ifdda_scalar_chan_apply_resident(gpu_right_preconditioned, int(count), tabdip,
                                                      1) ||
                    scalar_driver_copy_solution_wrapper(gpu_right_preconditioned, gpu_xr,
                                                        int(count)) != cudaSuccess) {
                    set_error(stop, info, info_length, "GPU scalar QMRCLA right Chan input failed");
                    return;
                }
                polarizability_input = gpu_right_preconditioned;
            }
            // The classic GPBICG family applies right Chan before polarization;
            // the other scalar solver routines apply it after polarization.
            if (*preconditioner == 2 && solver && !is_pimzqmr &&
                !ifdda_scalar_chan_apply_resident(gpu_xr, int(count), tabdip, 1)) {
                set_error(stop, info, info_length,
                          "GPU scalar right Chan preconditioner application failed");
                return;
            }
            if (scalar_driver_apply_polarizability_wrapper(polarizability_input,
                                                           buffers.device.polarizability, gpu_xi,
                                                           int(count)) != cudaSuccess) {
                set_error(stop, info, info_length, "GPU scalar polarizability application failed");
                return;
            }
            if ((*preconditioner == 2 && !solver && !is_pimzqmr &&
                 !ifdda_scalar_chan_apply_resident(gpu_xr, int(count), tabdip, 1)) ||
                !scalar_product_device(
                    device_green, device_tabdip, gpu_xi,
                    gpu_xr, buffers.device.fft_work, *ntotalm, *ntotal, *nmax, *nx, *ny, *nz, *nx2,
                    *ny2, *nz2, *nxy2, dense_product) ||
                (*preconditioner == 1 &&
                 !ifdda_scalar_chan_apply_resident(gpu_xr, int(count), tabdip, 1))) {
                set_error(stop, info, info_length, "GPU scalar matrix-vector product failed");
                return;
            }
            mark_gpbicg_device_xr_current(xr, true);
        }

        if (nstat != 1)
            continue;
        if (scalar_driver_residual_wrapper(gpu_xr, gpu_rhs, gpu_xi, buffers.device.scalar_solution,
                                           int(count)) != cudaSuccess) {
            set_error(stop, info, info_length, "GPU scalar residual calculation failed");
            return;
        }
        cublasHandle_t handle = cuda_utils::get_cublas_handle();
        CU_REAL residual_norm = CU_REAL(0);
        if (!handle || !cuda_utils::norm2(handle, int(count), gpu_xr, &residual_norm,
                                          "scalar inverse driver")) {
            set_error(stop, info, info_length, "cuBLAS scalar residual reduction failed");
            return;
        }
        *final_residual = residual_norm / rhs_norm;
        break;
    }

    if (*preconditioner == 2 &&
        !ifdda_scalar_chan_apply_resident(buffers.device.local_solution, int(count), tabdip, 1)) {
        set_error(stop, info, info_length, "GPU scalar right Chan solution recovery failed");
        return;
    }

    if (*preconditioner == 2 && scalar_driver_apply_polarizability_wrapper(
                                    buffers.device.local_solution, buffers.device.polarizability,
                                    buffers.device.scalar_solution, int(count)) != cudaSuccess) {
        set_error(stop, info, info_length,
                  "Failed to reconstruct right-preconditioned scalar solution");
        return;
    }

    if (scalar_driver_expand_fields_wrapper(
            buffers.device.scalar_solution, buffers.device.local_solution,
            buffers.device.vector_field, buffers.device.vector_local_field, incident_x, incident_y,
            incident_z,
            int(count)) != cudaSuccess ||
        cuda_data::copy_device_to_host(ff, buffers.device.vector_field,
                                       size_t(3) * count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(ffloc, buffers.device.vector_local_field,
                                       size_t(3) * count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(ffscal, buffers.device.scalar_solution,
                                       //-----------------------------------------------------------
                                       count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(fflocscal, buffers.device.local_solution,
                                       //-----------------------------------------------------------
                                       count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess) {
        set_error(stop, info, info_length, "Failed to copy final scalar fields to host");
        return;
    }
    // PIMZQMR updates the caller's iteration counter directly (the Fortran
    // routine receives NLOOP as ITNO). Do not add the driver's callback
    // counter again here; that counter is intentionally separate and unused
    // by this solver.
    if (!is_pimzqmr)
        *iterations += is_zcg
                           ? zcg_iterations
                           : ((is_pimzbicgstab || is_qmrbicgstab || is_gpbicor) ? itno : itno + 1);
    // The scalar solution has been expanded and published to host FF/FFloc.
    // A following vector solve may reuse the host xi address for 3*count
    // values: its scalar GPU cache entry must not masquerade as that vector.
    mark_gpbicg_device_xi_current(xi, false);
    *stop = 0;
    if (info && info_length)
        std::memset(info, ' ', info_length);
}

