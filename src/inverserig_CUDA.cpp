#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_solver_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {
std::atomic<int> pending_initial_guess_mode{-1};
std::atomic<int> last_initial_guess_mode{-1};
}

//-----------------------------------------------------------
extern "C" void ifdda_set_initial_guess_mode_cuda_(int mode) {
    pending_initial_guess_mode.store(mode, std::memory_order_relaxed);
}

//-----------------------------------------------------------
extern "C" int ifdda_get_initial_guess_mode_cuda_() {
    return pending_initial_guess_mode.load(std::memory_order_relaxed);
}

//-----------------------------------------------------------
extern "C" int ifdda_get_last_initial_guess_mode_cuda_() {
    return last_initial_guess_mode.load(std::memory_order_relaxed);
}

extern "C" void produitpolavect_(const CUCOMPLEX *polarizability, const int *niso, const int *nmax,
                                 const int *ndipoles, const CUCOMPLEX *input, CUCOMPLEX *output);

extern "C" void produitfftmatvectoptdevice_(
    const CUCOMPLEX *, const CUCOMPLEX *, const CUCOMPLEX *, const CUCOMPLEX *,
    const CUCOMPLEX *, const CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *,
    const int *, const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const CUCOMPLEX *,
    CUCOMPLEX *, const long long *, const long long *);
extern "C" int chan_preconditioner_apply_device_cuda_(CUCOMPLEX *device_field,
                                                      const int *cpu_dipole_map, const int *nx,
                                                      const int *ny, const int *nz, const int *nxm,
                                                      const int *nym, const int *nzm);
extern "C" bool register_gpbicg_device_b_alias(const void *cpu_b_key, CUCOMPLEX *gpu_b,
                                               size_t count);
extern "C" void gpbicg_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                        const int *, int *, CUCOMPLEX *, int *, const int *, const CU_REAL *,
                        CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                        int *, int *);
extern "C" void gpbicg2_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                         const int *, int *, CUCOMPLEX *, int *, const int *, const CU_REAL *,
                         CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                         int *, int *);
extern "C" void gpbicgplus_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                            const int *, int *, CUCOMPLEX *, int *, const int *, const CU_REAL *,
                            CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                            CUCOMPLEX *, int *, int *);
extern "C" void gpbicgsafe_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                            const int *, int *, CUCOMPLEX *, int *, const int *, const CU_REAL *,
                            CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                            CUCOMPLEX *, int *, int *);
extern "C" void gpbicgar_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                          const int *, int *, CUCOMPLEX *, int *, const int *, const CU_REAL *,
                          CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                          CUCOMPLEX *, int *, int *);
extern "C" void gpbicgar2_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                           const int *, int *, CUCOMPLEX *, int *, const int *, const CU_REAL *,
                           CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                           CUCOMPLEX *, int *, int *);
extern "C" void gpbicor_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *,
                         const int *, const int *, int *, CUCOMPLEX *, int *, const int *,
                         const CU_REAL *, CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                         CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, int *, int *);
extern "C" void cors_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *, const int *,
                      int *, CUCOMPLEX *, int *, const int *, const CU_REAL *, CU_REAL *,
                      CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, int *, int *);
extern "C" void tfqmr_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                       const int *, int *, CUCOMPLEX *, int *, const int *, const CU_REAL *,
                       CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                       CUCOMPLEX *, int *, int *);
extern "C" void zcg_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CU_REAL *, CUCOMPLEX *, CUCOMPLEX *,
                     CUCOMPLEX *, CUCOMPLEX *, const int *, const int *, const int *, int *, int *,
                     int *, const int *, CU_REAL *, const CU_REAL *);
extern "C" void pimzbicgstab_(COMPLEX *, COMPLEX *, COMPLEX *, const COMPLEX *, const int *,
                              const int *, const int *, int *, COMPLEX *, COMPLEX *,
                              COMPLEX *, COMPLEX *, CU_REAL *, const CU_REAL *, int *,
                              const int *, int *, int *);
extern "C" void qmrbicgstab_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *,
                             const int *, const int *, int *, CUCOMPLEX *, int *, const int *,
                             const CU_REAL *, CU_REAL *, CU_REAL *, CUCOMPLEX *, CUCOMPLEX *,
                             CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                             CUCOMPLEX *, CUCOMPLEX *, const int *, int *, int *);
extern "C" void gpbicgstarplus_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                                const int *, int *, CUCOMPLEX *, int *, const int *,
                                const CU_REAL *, CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                                CUCOMPLEX *, CUCOMPLEX *, int *, int *);
extern "C" void ssbicgsafe2_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                             const int *, int *, CUCOMPLEX *, int *, const int *, const CU_REAL *,
                             CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                             CUCOMPLEX *, int *, int *);
extern "C" void zbicgstabl_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CU_REAL *,
                            CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, int *,
                            int *, int *, const int *, const int *, const int *, int *, int *,
                            int *, int *, const int *, const CU_REAL *, const CU_REAL *);
extern "C" void gpbicgstabl_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CU_REAL *,
                             CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                             const int *, const int *, const int *, const int *, int *, int *,
                             int *, int *, int *, int *, int *, int *, int *, const int *,
                             CU_REAL *, const CU_REAL *);
extern "C" void pimzqmr_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CU_REAL *,
                         const int *, const int *, const int *, CUCOMPLEX *, CUCOMPLEX *,
                         CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                         CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                         CUCOMPLEX *, int *, int *, int *, const int *, CU_REAL *, const CU_REAL *,
                         int *, int *);
extern "C" void idrs_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                      const int *, int *, int *, CUCOMPLEX *, int *, const int *, const CU_REAL *,
                      CU_REAL *, CU_REAL *, CUCOMPLEX *, int *, int *, int *);
extern "C" void gcors2_(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *,
                        const int *, const int *, int *, CUCOMPLEX *, int *, const int *,
                        const CU_REAL *, CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                        CUCOMPLEX *, int *, int *);

namespace {

struct ResidentChanPreconditioner {
    int mode = 0; // 0 disabled, 1 left preconditioner, 2 right preconditioner
    int nx = 0, ny = 0, nz = 0;
    int nxm = 0, nym = 0, nzm = 0;
};

ResidentChanPreconditioner resident_chan_preconditioner;

using cuda_utils::axpy;
using cuda_utils::check_cuda;
using cuda_utils::checked_multiply;
using cuda_utils::copy;
using cuda_utils::device_value;
using cuda_utils::dotc;
using cuda_utils::get_cublas_handle;
using cuda_utils::host_value;
using cuda_utils::HostComplex;
using cuda_utils::norm2;
using cuda_utils::scale;

struct SolverResidentGuard {
    const COMPLEX *host_b = nullptr;
    const COMPLEX *host_xi = nullptr;
    const COMPLEX *host_output = nullptr;
    const COMPLEX *host_work = nullptr;
    bool active = false;
    bool rhs_alias_registered = false;
//-------------------------------------------

    ~SolverResidentGuard() {
        if (rhs_alias_registered && host_b)
            release_gpbicg_device_b_alias(host_b);
        if (active) {
            set_gpbicg_device_resident(host_xi, false);
            if (host_output)
                set_gpbicg_device_resident(host_output, false);
            if (host_work)
                set_gpbicg_device_resident(host_work, false);
        }
    }
};

using Solver = void (*)(CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, const int *, const int *,
                        const int *, int *, CUCOMPLEX *, int *, const int *, const CU_REAL *,
                        CU_REAL *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
                        int *, int *);

//-----------------------------------------------------------
bool select_solver(const char *method, size_t length, Solver *solver, const char **name,
                   bool *gpbicor, bool *cors, bool *tfqmr, bool *zcg, bool *pimzbicgstab,
                   int *qmrbicgstab_method, int *bicglstab_l, int *gpbicgstabl_l, bool *pimzqmr,
                   bool *idrs, int *idrs_s, bool *gcors2) {
    *bicglstab_l = 0;
    *gpbicgstabl_l = 0;
    *pimzqmr = false;
    *idrs = false;
    *idrs_s = 0;
    *gcors2 = false;

    struct Entry {
        const char *spelling;
        size_t length;
        Solver function;
    };

    static const Entry entries[] = {{"GPBICG1", 7, gpbicg_},
                                    {"GPBICG2", 7, gpbicg2_},
                                    {"GPBICGplus", 10, gpbicgplus_},
                                    {"GPBICGsafe", 10, gpbicgsafe_},
                                    {"GPBICGAR2", 9, gpbicgar2_},
                                    {"GPBICGAR", 8, gpbicgar_},
                                    {"BICGSTARPLUS", 12, gpbicgstarplus_},
                                    {"SSBICGSAFE2", 11, ssbicgsafe2_}};
    for (const Entry &entry : entries) {
        if (length >= entry.length && std::memcmp(method, entry.spelling, entry.length) == 0) {
            *solver = entry.function;
            *name = entry.spelling;
            *gpbicor = false;
            *cors = false;
            *tfqmr = false;
            *zcg = false;
            *pimzbicgstab = false;
            *qmrbicgstab_method = 0;
            return true;
        }
    }
    if (length >= 7 && std::memcmp(method, "GPBICOR", 7) == 0) {
        *solver = nullptr;
        *name = "GPBICOR";
        *gpbicor = true;
        *cors = false;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        return true;
    }
    if (length >= 4 && std::memcmp(method, "CORS", 4) == 0) {
        *solver = nullptr;
        *name = "CORS";
        *gpbicor = false;
        *cors = true;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        return true;
    }
    if (length >= 5 && std::memcmp(method, "TFQMR", 5) == 0) {
        *solver = nullptr;
        *name = "TFQMR";
        *gpbicor = false;
        *cors = false;
        *tfqmr = true;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        return true;
    }
    if (length >= 2 && method[0] == 'C' && method[1] == 'G') {
        *solver = nullptr;
        *name = "CG";
        *gpbicor = false;
        *cors = false;
        *tfqmr = false;
        *zcg = true;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        return true;
    }
    if (length >= 8 && std::memcmp(method, "BICGSTAB", 8) == 0) {
        *solver = nullptr;
        *name = "PIMZBICGSTAB";
        *gpbicor = false;
        *cors = false;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = true;
        *qmrbicgstab_method = 0;
        return true;
    }
    if (length >= 12 && std::memcmp(method, "QMRBICGSTAB1", 12) == 0) {
        *solver = nullptr;
        *name = "QMRBICGSTAB1";
        *gpbicor = false;
        *cors = false;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 1;
        return true;
    }
    if (length >= 12 && std::memcmp(method, "QMRBICGSTAB2", 12) == 0) {
        *solver = nullptr;
        *name = "QMRBICGSTAB2";
        *gpbicor = false;
        *cors = false;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 2;
        return true;
    }
    if (length >= 10 && std::memcmp(method, "BICGLSTAB2", 10) == 0) {
        *solver = nullptr;
        *name = "BICGLSTAB2";
        *gpbicor = false;
        *cors = false;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        *bicglstab_l = 2;
        return true;
    }
    if (length >= 10 && std::memcmp(method, "BICGLSTAB4", 10) == 0) {
        *solver = nullptr;
        *name = "BICGLSTAB4";
        *gpbicor = false;
        *cors = false;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        *bicglstab_l = 4;
        return true;
    }
    if (length >= 10 && std::memcmp(method, "BICGLSTAB8", 10) == 0) {
        *solver = nullptr;
        *name = "BICGLSTAB8";
        *gpbicor = false;
        *cors = false;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        *bicglstab_l = 8;
        return true;
    }
    if (length >= 12 && std::memcmp(method, "GPBICGSTABL2", 12) == 0) {
        *solver = nullptr;
        *name = "GPBICGSTABL2";
        *gpbicor = false;
        *cors = false;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        *gpbicgstabl_l = 2;
        return true;
    }
    if (length >= 12 && std::memcmp(method, "GPBICGSTABL4", 12) == 0) {
        *solver = nullptr;
        *name = "GPBICGSTABL4";
        *gpbicor = false;
        *cors = false;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        *gpbicgstabl_l = 4;
        return true;
    }
    if (length >= 12 && std::memcmp(method, "GPBICGSTABL8", 12) == 0) {
        *solver = nullptr;
        *name = "GPBICGSTABL8";
        *gpbicor = false;
        *cors = false;
        *tfqmr = false;
        *zcg = false;
        *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        *gpbicgstabl_l = 8;
        return true;
    }
    if (length >= 6 && std::memcmp(method, "QMRCLA", 6) == 0) {
        *solver = nullptr;
        *name = "QMRCLA";
        *gpbicor = *cors = *tfqmr = *zcg = *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        *pimzqmr = true;
        return true;
    }
    if (length >= 4 && std::memcmp(method, "IDRS", 4) == 0) {
        int shadow = 4;
        if (length >= 5 && method[4] >= '0' && method[4] <= '9')
            shadow = method[4] - '0';
        if (shadow != 2 && shadow != 4 && shadow != 8)
            return false;
        *solver = nullptr;
        *name = "IDRS";
        *gpbicor = *cors = *tfqmr = *zcg = *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        *idrs = true;
        *idrs_s = shadow;
        return true;
    }
    if (length >= 6 && std::memcmp(method, "GCORS2", 6) == 0) {
        *solver = nullptr;
        *name = "GCORS2";
        *gpbicor = *cors = *tfqmr = *zcg = *pimzbicgstab = false;
        *qmrbicgstab_method = 0;
        *pimzqmr = false;
        *idrs = false;
        *gcors2 = true;
        return true;
    }
    return false;
}

//-----------------------------------------------------------
void set_info(char *target, size_t length, const char *message) {
    if (!target)
        return;
    std::fill(target, target + length, ' ');
    const size_t count = std::min(length, std::strlen(message));
    std::memcpy(target, message, count);
}

//-----------------------------------------------------------
bool apply_matrix_product(const CUCOMPLEX *green_xx, const CUCOMPLEX *green_xy,
                          const CUCOMPLEX *green_xz, const CUCOMPLEX *green_yy,
                          const CUCOMPLEX *green_yz, const CUCOMPLEX *green_zz, CUCOMPLEX *vectx,
                          CUCOMPLEX *vecty, CUCOMPLEX *vectz, const int *tabdip, const int *ntotalm,
                          const int *ntotal, const int *nmax, const int *ndipole, const int *nxm,
                          const int *nym, const int *nzm, const int *nx, const int *ny,
                          const int *nz, const int *nx2, const int *ny2, const int *nxy2,
                          const int *nz2, const CUCOMPLEX *input, CUCOMPLEX *output,
                          const long long *plan_xf, const long long *plan_xb,
                          const long long *plan_yf, const long long *plan_yb,
                          const long long *plan_zf, const long long *plan_zb,
                          bool optimized_product, size_t tensor_count) {
    if (optimized_product) {
        return produitfftmatvectopt_cuda(GreenTensorDeviceView{{green_xx, green_xy, green_xz, green_yy, green_yz, green_zz}}, vectx, input, output, *nx, *ny, *nz, *nx2,
                                         *ny2, *nz2, *nxy2, *ntotal, *nmax, tensor_count);
    }
    produitfftmatvect3host_(green_xx, green_xy, green_xz, green_yy, green_yz, green_zz, vectx,
                              vecty, vectz, tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm,
                              nx, ny, nz, nx2, ny2, nxy2, nz2, input, output, plan_xf, plan_xb);
    return true;
}

//-----------------------------------------------------------
bool initialize_born_gamma_cuda(
    const COMPLEX *green_xx, const COMPLEX *green_xy, const COMPLEX *green_xz,
    const COMPLEX *green_yy, const COMPLEX *green_yz, const COMPLEX *green_zz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz, const int *tabdip, const int *ntotalm,
    const int *ntotal, const int *nmax, const int *ndipole, const int *nxm, const int *nym,
    const int *nzm, const int *nx, const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nxy2, const int *nz2, const COMPLEX *xi, COMPLEX *xi_scaled,
    const COMPLEX *incident, const COMPLEX *polarizability, const int *niso,
    const int *ndipoles, size_t n, bool optimized_product, const long long *plan_xf,
    const long long *plan_xb, const long long *plan_yf, const long long *plan_yb,
    const long long *plan_zf, const long long *plan_zb, CU_REAL *gamma) {
    if (!green_xx || !green_xy || !green_xz || !green_yy || !green_yz || !green_zz || !vectx ||
        !vecty || !vectz || !tabdip || !ntotalm || !ntotal || !nmax || !ndipole || !nxm || !nym ||
        !nzm || !nx || !ny || !nz || !nx2 || !nxy2 || !nz2 || !xi || !xi_scaled || !incident ||
        !polarizability || !niso || !ndipoles || !gamma || n == 0 || n > size_t(INT_MAX) ||
        *ntotal <= 0 || *ndipole <= 0)
        return false;

    if (!initialize_cuda_device())
        return false;
    const size_t tensor_count = size_t(*nxm + 1) * size_t(*nym + 1) * size_t(*nzm + 1);
    const size_t physical_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
    const size_t fft_vector_count = size_t(3) * size_t(*ntotal);
    const size_t polarizability_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
    if (tensor_count == 0 || physical_count > size_t(INT_MAX) || fft_vector_count > size_t(INT_MAX))
        return false;

    const COMPLEX *host_tensors[6] = {green_xx, green_xy, green_xz, green_yy, green_yz, green_zz};
    GreenTensorDevicePointers device_green = get_device_green(host_tensors, tensor_count);
    CUCOMPLEX *device_vectors = get_device_vectors(fft_vector_count);
    CU_INT *device_tabdip = get_device_tabdip(tabdip, physical_count);
    CUCOMPLEX *device_polarizability =
        copy_polarizability_to_device(polarizability, polarizability_count);
    cublasHandle_t handle = get_cublas_handle();
    if (!device_green || !device_vectors || !device_tabdip || !device_polarizability || !handle)
        return false;

    cuda_data::InverseDriverGammaBuffers gamma_buffers{};
    if (!cuda_data::get_device_inverse_driver_gamma_buffers(n, &gamma_buffers))
        return false;
    CUCOMPLEX *device_input = gamma_buffers.input;
    CUCOMPLEX *device_output = gamma_buffers.output;
    CUCOMPLEX *device_incident = gamma_buffers.incident;
    CUCOMPLEX *device_solution = gamma_buffers.solution;
    bool success = false;
    do {
        const size_t bytes = n * sizeof(CUCOMPLEX);
        if (!device_input || !device_output || !device_incident || !device_solution ||
            !cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(device_solution, xi, bytes,
                                               cuda_utils::TransferCategory::Other),
                "inverserig CUDA gamma", "upload initial solution") ||
            !cuda_utils::check_cuda(cuda_utils::tracked_memcpy(device_input, device_solution, bytes,
                                                               cudaMemcpyDeviceToDevice),
                                    "inverserig CUDA gamma", "stage polarization input") ||
            !cuda_utils::check_cuda(cuda_utils::tracked_memcpy(device_output, device_solution,
                                                               bytes, cudaMemcpyDeviceToDevice),
                                    "inverserig CUDA gamma", "initialize output field") ||
            !cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(device_incident, incident, bytes,
                                               cuda_utils::TransferCategory::Other),
                "inverserig CUDA gamma", "upload incident field") ||
            !check_cuda(polarization_inplace_wrapper(device_polarizability, device_input, *niso,
                                                     *nmax, *ndipoles),
                        "inverserig CUDA gamma", "apply polarizability"))
            break;

        if (optimized_product) {
            produitfftmatvectoptdevice_(
                device_green[0], device_green[1], device_green[2],
                device_green[3], device_green[4],
                device_green[5], vectx, vecty, vectz, ntotalm, ntotal, nmax, nxm,
                nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, device_input, device_output, plan_xf,
                plan_xb);
        } else {
            produitfftmatvect3device_(
                device_green[0], device_green[1], device_green[2],
                device_green[3], device_green[4],
                device_green[5], vectx, vecty, vectz, device_tabdip, ntotalm,
                ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, device_input,
                device_output, plan_xb, plan_xf);
        }
        if (!check_cuda(cudaDeviceSynchronize(), "inverserig CUDA gamma",
                        "finish incident Green product"))
            break;

        CUCOMPLEX dot = CUFFT_ZERO, squared_norm = CUFFT_ZERO;
        if (!dotc(handle, int(n), device_output, device_incident, &dot, "inverserig CUDA gamma") ||
            !dotc(handle, int(n), device_output, device_output, &squared_norm,
                  "inverserig CUDA gamma"))
            break;
        *gamma = (dot.x == CU_REAL(0) || squared_norm.x == CU_REAL(0)) ? CU_REAL(1)
                                                                       : dot.x / squared_norm.x;
        if (std::getenv("IFDDA_TRACE_GAMMA"))
            std::fprintf(
                stderr,
                "[inverserig CUDA gamma] gamma=%.17g dot=(%.17g,%.17g) norm2=(%.17g,%.17g)\n",
                double(*gamma), double(dot.x), double(dot.y), double(squared_norm.x),
                double(squared_norm.y));
        if (!std::isfinite(double(*gamma)) ||
            !scale(handle, int(n), device_value(HostComplex(*gamma, 0)), device_solution,
                   "inverserig CUDA gamma") ||
            !cuda_utils::check_cuda(
                cuda_data::copy_device_to_host(xi_scaled, device_solution, bytes,
                                               cuda_utils::TransferCategory::Other),
                "inverserig CUDA gamma", "download scaled initial solution"))
            break;
        success = true;
    } while (false);

    return success;
}

//-----------------------------------------------------------
bool initialize_born_gamma_resident_cuda(
    const COMPLEX *green_xx, const COMPLEX *green_xy, const COMPLEX *green_xz,
    const COMPLEX *green_yy, const COMPLEX *green_yz, const COMPLEX *green_zz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz, const int *tabdip, const int *ntotalm,
    const int *ntotal, const int *nmax, const int *ndipole, const int *nxm, const int *nym,
    const int *nzm, const int *nx, const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nxy2, const int *nz2, CUCOMPLEX *device_solution, const CUCOMPLEX *device_incident,
    const COMPLEX *polarizability, const int *niso, const int *ndipoles, size_t n,
    bool optimized_product, const long long *plan_xf, const long long *plan_xb,
    const long long *plan_yf, const long long *plan_yb, const long long *plan_zf,
    const long long *plan_zb, CU_REAL *gamma) {
    if (!green_xx || !green_xy || !green_xz || !green_yy || !green_yz || !green_zz || !vectx ||
        !vecty || !vectz || !tabdip || !ntotalm || !ntotal || !nmax || !ndipole || !nxm || !nym ||
        !nzm || !nx || !ny || !nz || !nx2 || !ny2 || !nxy2 || !nz2 || !device_solution ||
        !device_incident || !polarizability || !niso || !ndipoles || !gamma || n == 0 ||
        n > size_t(INT_MAX) || *ntotal <= 0 || *ndipole <= 0)
        return false;
    if (!initialize_cuda_device())
        return false;

    const size_t tensor_count = size_t(*nxm + 1) * size_t(*nym + 1) * size_t(*nzm + 1);
    const size_t physical_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
    const size_t fft_vector_count = size_t(3) * size_t(*ntotal);
    const size_t polarizability_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
    if (tensor_count == 0 || physical_count > size_t(INT_MAX) || fft_vector_count > size_t(INT_MAX))
        return false;

    const COMPLEX *host_tensors[6] = {green_xx, green_xy, green_xz, green_yy, green_yz, green_zz};
    GreenTensorDevicePointers device_green = get_device_green(host_tensors, tensor_count);
    CUCOMPLEX *device_vectors = get_device_vectors(fft_vector_count);
    CU_INT *device_tabdip = get_device_tabdip(tabdip, physical_count);
    CUCOMPLEX *device_polarizability =
        copy_polarizability_to_device(polarizability, polarizability_count);
    cublasHandle_t handle = get_cublas_handle();
    if (!device_green || !device_vectors || !device_tabdip || !device_polarizability || !handle)
        return false;

    cuda_data::InverseDriverBornBuffers born_buffers{};
    if (!cuda_data::get_device_inverse_driver_born_buffers(n, &born_buffers))
        return false;
    CUCOMPLEX *device_input = born_buffers.input;
    CUCOMPLEX *device_output = born_buffers.output;
    bool success = false;
    do {
        const size_t bytes = n * sizeof(CUCOMPLEX);
        if (!device_input || !device_output ||
            !cuda_utils::check_cuda(cuda_utils::tracked_memcpy(device_input, device_solution, bytes,
                                                               cudaMemcpyDeviceToDevice),
                                    "resident Born initialization", "stage initial solution") ||
            !cuda_utils::check_cuda(cuda_utils::tracked_memcpy(device_output, device_solution,
                                                               bytes, cudaMemcpyDeviceToDevice),
                                    "resident Born initialization",
                                    "seed Green product accumulator") ||
            !check_cuda(polarization_inplace_wrapper(device_polarizability, device_input, *niso,
                                                     *nmax, *ndipoles),
                        "resident Born initialization", "apply polarizability"))
            break;

        if (optimized_product) {
            produitfftmatvectoptdevice_(
                device_green[0], device_green[1], device_green[2],
                device_green[3], device_green[4],
                device_green[5], vectx, vecty, vectz, ntotalm, ntotal, nmax, nxm,
                nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, device_input, device_output, plan_xf,
                plan_xb);
        } else {
            produitfftmatvect3device_(
                device_green[0], device_green[1], device_green[2],
                device_green[3], device_green[4],
                device_green[5], vectx, vecty, vectz, device_tabdip, ntotalm,
                ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, device_input,
                device_output, plan_xb, plan_xf);
        }
        if (!check_cuda(cudaDeviceSynchronize(), "resident Born initialization",
                        "finish incident Green product"))
            break;

        CUCOMPLEX dot = CUFFT_ZERO, squared_norm = CUFFT_ZERO;
        if (!dotc(handle, int(n), device_output, device_incident, &dot,
                  "resident Born initialization") ||
            !dotc(handle, int(n), device_output, device_output, &squared_norm,
                  "resident Born initialization"))
            break;
        if (std::getenv("IFDDA_TRACE_GAMMA"))
            std::fprintf(
                stderr, "[resident Born CUDA gamma] dot=(%.17g,%.17g) norm2=(%.17g,%.17g) n=%zu\\n",
                double(dot.x), double(dot.y), double(squared_norm.x), double(squared_norm.y), n);
        *gamma = (dot.x == CU_REAL(0) || squared_norm.x == CU_REAL(0)) ? CU_REAL(1)
                                                                       : dot.x / squared_norm.x;
        if (!std::isfinite(double(*gamma)) ||
            !scale(handle, int(n), device_value(HostComplex(*gamma, 0)), device_solution,
                   "resident Born initialization"))
            break;
        success = true;
    } while (false);

    return success;
}

//-----------------------------------------------------------
bool initialize_rytov_guess_resident_cuda(
    const COMPLEX *green_xx, const COMPLEX *green_xy, const COMPLEX *green_xz,
    const COMPLEX *green_yy, const COMPLEX *green_yz, const COMPLEX *green_zz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz, const int *tabdip, const int *ntotalm,
    const int *ntotal, const int *nmax, const int *ndipole, const int *nxm, const int *nym,
    const int *nzm, const int *nx, const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nxy2, const int *nz2, CUCOMPLEX *device_xi, CUCOMPLEX *device_xr,
    CUCOMPLEX *device_green_output, const CUCOMPLEX *device_incident,
    CUCOMPLEX *device_polarized_input, const COMPLEX *polarizability, const int *niso,
    const int *ndipoles, size_t n, bool optimized_product, const long long *plan_xf,
    const long long *plan_xb, const long long *plan_yf, const long long *plan_yb,
    const long long *plan_zf, const long long *plan_zb) {
    if (!green_xx || !green_xy || !green_xz || !green_yy || !green_yz || !green_zz || !vectx ||
        !vecty || !vectz || !tabdip || !ntotalm || !ntotal || !nmax || !ndipole || !nxm || !nym ||
        !nzm || !nx || !ny || !nz || !nx2 || !ny2 || !nxy2 || !nz2 || !device_xi || !device_xr ||
        !device_green_output || !device_incident || !device_polarized_input || !polarizability ||
        !niso || !ndipoles || n == 0 || n > size_t(INT_MAX) || *ndipoles <= 0 || *ntotal <= 0)
        return false;

    const size_t tensor_count = size_t(*nxm + 1) * size_t(*nym + 1) * size_t(*nzm + 1);
    const size_t physical_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
    const size_t fft_vector_count = size_t(3) * size_t(*ntotal);
    cuda_data::InverseDriverRytovBuffers rytov_buffers{};
    if (!cuda_data::get_device_inverse_driver_rytov_buffers(&rytov_buffers))
        return false;
    const COMPLEX *host_tensors[6] = {green_xx, green_xy, green_xz, green_yy, green_yz, green_zz};
    GreenTensorDevicePointers device_green = get_device_green(host_tensors, tensor_count);
    CUCOMPLEX *device_fft_vectors = get_device_vectors(fft_vector_count);
    CU_INT *device_tabdip = get_device_tabdip(tabdip, physical_count);
    CUCOMPLEX *device_polarizability = copy_polarizability_to_device(
        polarizability, size_t(*nmax) * size_t(*niso) * size_t(*niso));
    CU_INT *device_error = rytov_buffers.error;
    bool success = false;
    do {
        const size_t bytes = n * sizeof(CUCOMPLEX);
        if (!device_green || !device_fft_vectors || !device_tabdip || !device_polarizability ||
            !device_error ||
            !cuda_utils::check_cuda(
                cuda_utils::tracked_memcpy(device_polarized_input, device_incident, bytes,
                                           cudaMemcpyDeviceToDevice),
                "resident Rytov initialization", "stage incident polarization") ||
            !check_cuda(polarization_inplace_wrapper(device_polarizability, device_polarized_input,
                                                     *niso, *nmax, *ndipoles),
                        "resident Rytov initialization", "apply polarizability") ||
            !check_cuda(complex_vector_scale_wrapper(device_polarized_input, device_polarized_input,
                                                     device_value(HostComplex(-1, 0)), int(n)),
                        "resident Rytov initialization", "negate polarization") ||
            !check_cuda(zero_vectors_wrapper(device_green_output, 3 * *nmax),
                        "resident Rytov initialization", "clear Green output"))
            break;

        if (optimized_product) {
            produitfftmatvectoptdevice_(
                device_green[0], device_green[1], device_green[2],
                device_green[3], device_green[4],
                device_green[5], vectx, vecty, vectz, ntotalm, ntotal, nmax, nxm,
                nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, device_polarized_input,
                device_green_output, plan_xf, plan_xb);
        } else {
            produitfftmatvect3device_(
                device_green[0], device_green[1], device_green[2],
                device_green[3], device_green[4],
                device_green[5], vectx, vecty, vectz, device_tabdip, ntotalm,
                ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,
                device_polarized_input, device_green_output, plan_xb, plan_xf);
        }
        if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), "resident Rytov initialization",
                                    "finish Green product") ||
            !cuda_utils::check_cuda(cudaMemset(device_error, 0, sizeof(int)),
                                    "resident Rytov initialization", "clear Rytov error flag") ||
            !check_cuda(rytov_vector_field_wrapper(device_incident, device_green_output, device_xi,
                                                   device_error, *ndipoles),
                        "resident Rytov initialization", "apply vector Rytov transform"))
            break;

        int error = 1;
        if (!cuda_utils::check_cuda(
                cuda_data::copy_device_to_host(&error, device_error, sizeof(error),
                                               cuda_utils::TransferCategory::Diagnostic),
                "resident Rytov initialization", "read error flag") ||
            error != 0)
            break;
        if (!cuda_utils::check_cuda(
                cuda_utils::tracked_memcpy(device_xr, device_xi, bytes, cudaMemcpyDeviceToDevice),
                "resident Rytov initialization", "seed XR from Rytov field"))
            break;
        success = true;
    } while (false);
    return success;
}

} // namespace

//-----------------------------------------------------------
extern "C" int ifdda_initialize_right_rytov_guess_cuda_(
    const COMPLEX *green_xx, const COMPLEX *green_xy, const COMPLEX *green_xz,
    const COMPLEX *green_yy, const COMPLEX *green_yz, const COMPLEX *green_zz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz, const int *tabdip, const int *ntotalm,
    const int *ntotal, const int *nmax, const int *ndipole, const int *nxm, const int *nym,
    const int *nzm, const int *nx, const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nxy2, const int *nz2, const int *ndim, const int *lda, const int *nlar,
    COMPLEX *xi, COMPLEX *xr, COMPLEX *work, const COMPLEX *incident,
    const COMPLEX *polarizability, const int *niso, const long long *plan_xf,
    const long long *plan_xb, const long long *plan_yf, const long long *plan_yb,
    const long long *plan_zf, const long long *plan_zb, const int *optimized_product) {
    if (!ndim || !lda || !nlar || !xi || !xr || !work || !incident || !optimized_product ||
        *ndim <= 0 || *lda <= 0 || *nlar <= 0 ||
        size_t(*lda) > std::numeric_limits<size_t>::max() / size_t(*nlar))
        return 1;

    CUCOMPLEX *gpu_xi = nullptr, *gpu_xr = nullptr, *gpu_b = nullptr;
    CUCOMPLEX *gpu_work = nullptr;
    const size_t count = size_t(*ndim);
    const size_t work_count = size_t(*lda) * size_t(*nlar);
    if (!get_gpbicg_device_arrays_resident_initialized(xi, xr, incident, work, count, work_count,
                                                       true, 2, &gpu_xi, &gpu_xr, &gpu_b,
                                                       &gpu_work))
        return 2;
    CUCOMPLEX *gpu_polarized_input = get_device_auxiliary_vector(count);
    if (!gpu_polarized_input ||
        !initialize_rytov_guess_resident_cuda(
            green_xx, green_xy, green_xz, green_yy, green_yz, green_zz, vectx, vecty, vectz, tabdip,
            ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, gpu_xi,
            gpu_xr, gpu_work, gpu_b, gpu_polarized_input, polarizability, niso, ndipole, count,
            *optimized_product != 0, plan_xf, plan_xb, plan_yf, plan_yb, plan_zf, plan_zb))
        return 3;
    mark_gpbicg_device_xi_current(xi, true);
    mark_gpbicg_device_xr_current(xr, true);
    std::fprintf(stderr, "[inverserig CUDA] right Rytov starting vector initialized on GPU\n");
    return 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_renormalized_born_cuda_(const COMPLEX *incident,
                                              const COMPLEX *polarizability,
                                              COMPLEX *local_field, COMPLEX *polarized_field,
                                              const int *nmax, const int *niso, const int *ndipoles,
                                              int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!incident || !polarizability || !local_field || !polarized_field || !nmax || !niso ||
        !ndipoles || *nmax <= 0 || *ndipoles <= 0 || *ndipoles > *nmax ||
        (*niso != 1 && *niso != 3) || size_t(*ndipoles) > std::numeric_limits<size_t>::max() / 3)
        return;
    if (!initialize_cuda_device()) {
        std::fprintf(stderr, "Renormalized Born CUDA: device initialization failed\n");
        return;
    }

    const size_t field_count = size_t(3) * size_t(*ndipoles);
    const size_t polarizability_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
    CUCOMPLEX *device_local = get_device_vectors(field_count);
    CUCOMPLEX *device_polarized = get_device_auxiliary_vector(field_count);
    CUCOMPLEX *device_polarizability =
        copy_polarizability_to_device(polarizability, polarizability_count);
    if (!device_local || !device_polarized || !device_polarizability)
        return;

    cudaError_t status =
        cuda_data::copy_host_to_device(device_local, incident, field_count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status =
            cuda_utils::tracked_memcpy(device_polarized, device_local,
                                       field_count * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice);
    if (status == cudaSuccess)
        status = polarization_inplace_wrapper(device_polarizability, device_polarized, *niso, *nmax,
                                              *ndipoles);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(local_field, device_local,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(polarized_field, device_polarized,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (!cuda_utils::check_cuda(status, "renormalized Born CUDA", "copy/polarize field"))
        return;
    *nstop = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_born_cuda_(const COMPLEX *incident, const COMPLEX *epsilon,
                                 const COMPLEX *polarizability, COMPLEX *local_field,
                                 COMPLEX *polarized_field, const int *nmax, const int *niso,
                                 const int *ndipoles, const CU_REAL *aretecube, const CU_REAL *k0,
                                 int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!incident || !epsilon || !polarizability || !local_field || !polarized_field || !nmax ||
        !niso || !ndipoles || !aretecube || !k0 || *nmax <= 0 || *ndipoles <= 0 ||
        *ndipoles > *nmax || (*niso != 1 && *niso != 3) || *aretecube <= CU_REAL(0) ||
        *k0 <= CU_REAL(0))
        return;
    if (!initialize_cuda_device())
        return;

    const size_t field_count = size_t(3) * size_t(*ndipoles);
    const size_t tensor_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
    CUCOMPLEX *device_field = get_device_vectors(field_count);
    CUCOMPLEX *device_polarized = get_device_auxiliary_vector(field_count);
    CUCOMPLEX *device_tensor = copy_polarizability_to_device(epsilon, tensor_count);
    if (!device_field || !device_polarized || !device_tensor)
        return;

    cudaError_t status =
        cuda_data::copy_host_to_device(device_field, incident, field_count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = local_macro_to_local_wrapper(device_tensor, device_field, *niso, *nmax, *ndipoles,
                                              *aretecube, *k0);
    // The persistent coefficient buffer is reused for polarizability. Ensure
    // the local-field kernel has consumed epsilon before refreshing it.
    if (status == cudaSuccess)
        status = cudaDeviceSynchronize();
    if (status == cudaSuccess)
        device_tensor = copy_polarizability_to_device(polarizability, tensor_count);
    if (status == cudaSuccess && device_tensor)
        status =
            cuda_utils::tracked_memcpy(device_polarized, device_field,
                                       field_count * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice);
    if (status == cudaSuccess && device_tensor)
        status =
            polarization_inplace_wrapper(device_tensor, device_polarized, *niso, *nmax, *ndipoles);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(local_field, device_field,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(polarized_field, device_polarized,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (device_tensor && status == cudaSuccess)
        *nstop = 0;
    else if (status != cudaSuccess)
        std::fprintf(stderr, "Born CUDA: %s\n", cudaGetErrorString(status));
}

//-----------------------------------------------------------
extern "C" void ifdda_vector_copy_cuda_(const COMPLEX *input, COMPLEX *output, const int *count,
                                        const int *negate, int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!input || !output || !count || !negate || *count <= 0 || (*negate != 0 && *negate != 1))
        return;
    CUCOMPLEX *device_input = get_device_vectors(size_t(*count));
    CUCOMPLEX *device_output = get_device_auxiliary_vector(size_t(*count));
    if (!device_input || !device_output)
        return;
    cudaError_t status =
        cuda_data::copy_host_to_device(device_input, input, size_t(*count) * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Other);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX scale = make_cuFloatComplex(*negate ? -1.0f : 1.0f, 0.0f);
#else
    const CUCOMPLEX scale = make_cuDoubleComplex(*negate ? -1.0 : 1.0, 0.0);
#endif
    if (status == cudaSuccess)
        status = complex_vector_scale_wrapper(device_input, device_output, scale, *count);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(output, device_output,
                                                size_t(*count) * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        *nstop = 0;
    else
        std::fprintf(stderr, "CUDA vector copy/scale: %s\n", cudaGetErrorString(status));
}

//-----------------------------------------------------------
extern "C" void ifdda_vector_zero_cuda_(COMPLEX *output, const int *count, int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!output || !count || *count <= 0)
        return;
    CUCOMPLEX *device_output = get_device_vectors(size_t(*count));
    if (!device_output)
        return;
    cudaError_t status = zero_vectors_wrapper(device_output, *count);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(output, device_output,
                                                size_t(*count) * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        *nstop = 0;
    else
        std::fprintf(stderr, "CUDA vector zero: %s\n", cudaGetErrorString(status));
}

//-----------------------------------------------------------
extern "C" void ifdda_vector_negate_cuda_(COMPLEX *values, const int *count, int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!values || !count || *count <= 0)
        return;
    CUCOMPLEX *device_values = get_device_vectors(size_t(*count));
    if (!device_values)
        return;
    cudaError_t status =
        cuda_data::copy_host_to_device(device_values, values, size_t(*count) * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Other);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX minus_one = make_cuFloatComplex(-1.0f, 0.0f);
#else
    const CUCOMPLEX minus_one = make_cuDoubleComplex(-1.0, 0.0);
#endif
    if (status == cudaSuccess)
        status = complex_vector_scale_wrapper(device_values, device_values, minus_one, *count);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(values, device_values,
                                                size_t(*count) * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        *nstop = 0;
    else
        std::fprintf(stderr, "CUDA vector negate: %s\n", cudaGetErrorString(status));
}

//-----------------------------------------------------------
extern "C" void ifdda_rytov_macro_prepare_cuda_(
    const COMPLEX *incident, const COMPLEX *epsilon, const COMPLEX *polarizability,
    const COMPLEX *background_epsilon, COMPLEX *macroscopic_local_field, COMPLEX *delta_field,
    COMPLEX *negative_polarized_field, const int *nmax, const int *niso, const int *ndipoles,
    const CU_REAL *aretecube, const CU_REAL *k0, int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!incident || !epsilon || !polarizability || !background_epsilon ||
        !macroscopic_local_field || !delta_field || !negative_polarized_field || !nmax || !niso ||
        !ndipoles || !aretecube || !k0 || *nmax <= 0 || *ndipoles <= 0 || *ndipoles > *nmax ||
        (*niso != 1 && *niso != 3) || *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0))
        return;
    if (!initialize_cuda_device())
        return;

    const size_t field_count = size_t(3) * size_t(*ndipoles);
    const size_t tensor_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
    CUCOMPLEX *device_field = get_device_vectors(field_count);
    CUCOMPLEX *device_auxiliary = get_device_auxiliary_vector(field_count);
    CUCOMPLEX *device_tensor = copy_polarizability_to_device(epsilon, tensor_count);
    if (!device_field || !device_auxiliary || !device_tensor)
        return;

    cudaError_t status =
        cuda_data::copy_host_to_device(device_field, incident, field_count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = local_macro_to_local_wrapper(device_tensor, device_field, *niso, *nmax, *ndipoles,
                                              *aretecube, *k0);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(macroscopic_local_field, device_field,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_auxiliary, incident,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = rytov_macro_delta_wrapper(
            device_auxiliary, device_tensor, device_field,
            device_value(HostComplex(background_epsilon->real(), background_epsilon->imag())),
            *niso, *nmax, *ndipoles);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(delta_field, device_field,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_field, macroscopic_local_field,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess) {
        device_tensor = copy_polarizability_to_device(polarizability, tensor_count);
        if (!device_tensor)
            status = cudaErrorMemoryAllocation;
    }
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX minus_one = make_cuFloatComplex(-1.0f, 0.0f);
#else
    const CUCOMPLEX minus_one = make_cuDoubleComplex(-1.0, 0.0);
#endif
    if (status == cudaSuccess)
        status = polarization_inplace_wrapper(device_tensor, device_field, *niso, *nmax, *ndipoles);
    if (status == cudaSuccess)
        status =
            complex_vector_scale_wrapper(device_field, device_field, minus_one, int(field_count));
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(negative_polarized_field, device_field,
                                                field_count * sizeof(CUCOMPLEX),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        *nstop = 0;
    else
        std::fprintf(stderr, "CUDA Rytov macro preparation: %s\n", cudaGetErrorString(status));
}

// The Fortran inverserigpg setup calls this immediately before invoking the
// shared CUDA driver. The ntest=0 setup has already uploaded the Chan LU data.
//-----------------------------------------------------------
extern "C" int ifdda_set_resident_chan_preconditioner_(const int *enabled, const int *nx,
                                                       const int *ny, const int *nz, const int *nxm,
                                                       const int *nym, const int *nzm) {
    if (!enabled)
        return 1;
    if (*enabled == 0) {
        resident_chan_preconditioner = {};
        return 0;
    }
    if ((*enabled != 1 && *enabled != 2) || !nx || !ny || !nz || !nxm || !nym || !nzm || *nx <= 0 ||
        *ny <= 0 || *nz <= 0 || *nxm < *nx || *nym < *ny || *nzm < *nz)
        return 2;
    resident_chan_preconditioner = {*enabled, *nx, *ny, *nz, *nxm, *nym, *nzm};
    return 0;
}

// Standard unpreconditioned inverse-driver slice. It preserves
// inverserig.f90's initialization and reverse-communication ordering while
// dispatching to CUDA implementations of the supported solver family.
//-----------------------------------------------------------
extern "C" void inverserig_cuda_driver_(
    const COMPLEX *green_xx, const COMPLEX *green_xy, const COMPLEX *green_xz,
    const COMPLEX *green_yy, const COMPLEX *green_yz, const COMPLEX *green_zz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz, const int *tabdip, const int *ntotalm,
    const int *ntotal, const int *lda, const int *nlar, const int *nmax, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm, const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2, const int *ndipoles,
    const int *ndim, COMPLEX *xi, COMPLEX *xr, COMPLEX *work, COMPLEX *field,
    const COMPLEX *incident, COMPLEX *local_field, const COMPLEX *polarizability,
    const int *niso, const char *method, const CU_REAL *tolerance, CU_REAL *final_residual,
    int *products, int *product_calls, const int *max_iterations, const long long *plan_xf,
    const long long *plan_xb, const long long *plan_yf, const long long *plan_yb,
    const long long *plan_zf, const long long *plan_zb, const long long *planf,
    const long long *planb, const int *optimized_product, const int *initial_guess_mode, int *stop,
    char *info, size_t method_length, size_t info_length) {
    SolverResidentGuard resident_guard{};
    const bool device_only_field_result =
        take_device_only_result_vector_request(field);
    resident_guard.host_b = incident;
    // Incident builders such as arbitrary-field interpolation may register
    // an already-resident RHS before entering this solver. Always release the
    // association when this solve exits, including on error paths.
    resident_guard.rhs_alias_registered = true;
    clear_device_result_vector_alias(incident);
    clear_device_result_vector_alias(field);
    clear_device_result_vector_alias(local_field);
    (void)planf;
    (void)planb;
    Solver solver = nullptr;
    const char *solver_name = nullptr;
    bool is_gpbicor = false;
    bool is_cors = false;
    bool is_tfqmr = false;
    bool is_zcg = false;
    bool is_pimzbicgstab = false;
    int qmrbicgstab_method = 0;
    int bicglstab_l = 0;
    int gpbicgstabl_l = 0;
    bool is_pimzqmr = false, is_idrs = false;
    bool is_gcors2 = false;
    int idrs_s = 0;
    const bool is_bornserie =
        method && method_length >= 9 && std::memcmp(method, "BORNSERIE", 9) == 0;
    const bool supported_solver =
        method &&
        select_solver(method, method_length, &solver, &solver_name, &is_gpbicor, &is_cors,
                      &is_tfqmr, &is_zcg, &is_pimzbicgstab, &qmrbicgstab_method, &bicglstab_l,
                      &gpbicgstabl_l, &is_pimzqmr, &is_idrs, &idrs_s, &is_gcors2);
    const bool supported_method = supported_solver || is_bornserie;
    if (is_bornserie)
        solver_name = "BORNSERIE";
    if (!green_xx || !green_xy || !green_xz || !green_yy || !green_yz || !green_zz || !vectx ||
        !vecty || !vectz || !tabdip || !ntotalm || !ntotal || !lda || !nlar || !nmax || !ndipole ||
        !nxm || !nym || !nzm || !nx || !ny || !nz || !nx2 || !nxy2 || !nz2 || !ndipoles || !ndim ||
        !xi || !xr || !work || !field || !incident || !local_field || !polarizability || !niso ||
        !method || !tolerance || !final_residual || !products || !product_calls ||
        !max_iterations || !optimized_product || !initial_guess_mode || !stop || !info ||
        *initial_guess_mode < -2 || *initial_guess_mode > 4 || *ndim <= 0 || *lda < *ndim ||
        *nlar < 12 || (is_gcors2 && *nlar < 14) || *ndipoles <= 0 || !supported_method) {
        if (stop)
            *stop = 1;
        set_info(info, info_length, "Invalid CUDA inverserig arguments");
        return;
    }

    const int pending_guess_mode =
        pending_initial_guess_mode.exchange(-1, std::memory_order_relaxed);
    const int guess_mode = *initial_guess_mode >= 0 ? *initial_guess_mode : pending_guess_mode;
    if (guess_mode < -2 || guess_mode > 4) {
        *stop = 1;
        set_info(info, info_length, "Invalid pending CUDA initial guess mode");
        return;
    }
    last_initial_guess_mode.store(guess_mode, std::memory_order_relaxed);

    static bool path_reported = false;
    if (!path_reported) {
        std::fprintf(stderr, "[inverserig CUDA] %s driver active\n", solver_name);
        path_reported = true;
    }

    const int n = *ndim;
    const int three_n = 3 * *ndipoles;
    const bool use_optimized_product = (*optimized_product != 0);
    const bool use_resident_chan = resident_chan_preconditioner.mode != 0;
    const bool right_resident_chan = resident_chan_preconditioner.mode == 2;
    if (use_resident_chan)
        std::fprintf(stderr, "[inverserig CUDA] resident Chan %s path active\n",
                     right_resident_chan ? "A P^-1 x" : "P^-1 A x");
    CU_REAL norm = CU_REAL(0);
    CUCOMPLEX alpha = CUFFT_ZERO, beta = CUFFT_ZERO, eta = CUFFT_ZERO;
    CUCOMPLEX dzeta = CUFFT_ZERO, r0rn = CUFFT_ZERO;
    CUCOMPLEX rho = CUFFT_ZERO, rhoa = CUFFT_ZERO;
    CU_REAL qmr_solver_residual = CU_REAL(0);
    CUCOMPLEX qmr_omega = CUFFT_ZERO, qmr_tau = CUFFT_ZERO;
    CUCOMPLEX qmr_tau_tilde = CUFFT_ZERO, qmr_theta = CUFFT_ZERO;
    CUCOMPLEX qmr_theta_tilde = CUFFT_ZERO, qmr_eta_tilde = CUFFT_ZERO;
    CU_REAL zcg_tolet = CU_REAL(0);
    CU_REAL bicglstab_tolet = CU_REAL(0);
    int bicglstab_nr = 0, bicglstab_nu = 0, bicglstab_j = 0;
    CUCOMPLEX gpbicgstabl_sigma = CUFFT_ZERO;
    COMPLEX pimz_alpha(REAL(0), REAL(0));
    COMPLEX pimz_omega(REAL(0), REAL(0));
    COMPLEX pimz_rho(REAL(0), REAL(0));
    int gpbicgstabl_nr = 0, gpbicgstabl_np = 0, gpbicgstabl_ns = 0;
    int gpbicgstabl_nq = 0, gpbicgstabl_j = 0;
    CU_REAL gpbicgstabl_tolet = CU_REAL(0);
    CUCOMPLEX qmr_lambda = CUFFT_ZERO, qmr_kappa = CUFFT_ZERO;
    CUCOMPLEX gcors_alpha_t = CUFFT_ZERO, gcors_rho_t = CUFFT_ZERO;
    CUCOMPLEX qmrcla_theta = CUFFT_ZERO, qmrcla_gamma = CUFFT_ZERO;
    CUCOMPLEX qmr_ksi = CUFFT_ZERO, qmr_rho = CUFFT_ZERO;
    CUCOMPLEX qmr_epsilon = CUFFT_ZERO, qmr_mu = CUFFT_ZERO;
    CUCOMPLEX qmrcla_tau = CUFFT_ZERO, qmr_dots1 = CUFFT_ZERO;
    CUCOMPLEX qmr_dots2 = CUFFT_ZERO, qmr_dots3 = CUFFT_ZERO;
    CUCOMPLEX qmr_dots4 = CUFFT_ZERO, idrs_omega = CUFFT_ZERO;
    CU_REAL qmr_tolet = CU_REAL(0), idrs_residual = CU_REAL(0);
    int qmr_nt = 1, idrs_is = 0;
    int nou = 0, nstat = 0, steperr = -1;

    CU_REAL gamma = CU_REAL(0);
    if (guess_mode == -1) {
        const HostComplex first0(xi[0].real(), xi[0].imag());
        const HostComplex first1(xi[1].real(), xi[1].imag());
        const HostComplex first2(xi[2].real(), xi[2].imag());
        gamma = CU_REAL(std::abs(first0) + std::abs(first1) + std::abs(first2));
    }
    if (!use_resident_chan && guess_mode == -1 && gamma != CU_REAL(0) &&
        !initialize_born_gamma_cuda(
            green_xx, green_xy, green_xz, green_yy, green_yz, green_zz, vectx, vecty, vectz, tabdip,
            ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, xi, xi,
            incident, polarizability, niso, ndipoles, size_t(n), use_optimized_product, plan_xf,
            plan_xb, plan_yf, plan_yb, plan_zf, plan_zb, &gamma)) {
        *stop = 1;
        set_info(info, info_length, "CUDA Born gamma initialization failed");
        return;
    }

    // The resident reverse-communication solver expects XR to contain the
    // current iterate before its first GPU matrix product (FFB is updated in
    // place by subtracting the Green contribution).
    if (!use_resident_chan && (guess_mode == -1 || guess_mode == -2))
        std::copy(xi, xi + n, xr);

    // Common-ABI solvers keep their solution, matrix product, and polarization
    // vectors on the device for the full reverse-communication iteration.
    // Initial Born normalization and its dot products are prepared on CUDA;
    // mode 4 instead preserves a caller-prepared iterate such as BPM and
    // uploads it once through the ordinary resident-buffer path.
    const bool external_solution_buffer = is_tfqmr || is_pimzbicgstab || qmrbicgstab_method != 0 ||
                                          bicglstab_l != 0 || gpbicgstabl_l != 0 || is_pimzqmr ||
                                          is_idrs;
    const bool device_resident_solver =
        solver == gpbicg_ || solver == gpbicg2_ || solver == gpbicgplus_ || solver == gpbicgsafe_ ||
        solver == gpbicgar_ || solver == gpbicgar2_ || solver == gpbicgstarplus_ ||
        solver == ssbicgsafe2_ || is_gpbicor || is_gcors2 || is_cors || is_zcg ||
        external_solution_buffer || is_bornserie;
    if (use_resident_chan && !device_resident_solver) {
        *stop = 1;
        set_info(info, info_length, "Resident Chan requires a resident CUDA solver entry point");
        return;
    }
    CUCOMPLEX *gpu_xi = nullptr, *gpu_xr = nullptr;
    CUCOMPLEX *gpu_b = nullptr, *gpu_work = nullptr;
    CUCOMPLEX *gpu_ffloc = nullptr;
    CUCOMPLEX *gpu_polarizability = nullptr;
    CUCOMPLEX *gpu_polarized_input = nullptr;
    CUCOMPLEX *gpu_original_incident = nullptr;
    cublasHandle_t resident_blas = nullptr;
    const size_t tensor_count = size_t(*nxm + 1) * size_t(*nym + 1) * size_t(*nzm + 1);
    const COMPLEX *const host_green_components[6] = {
        green_xx, green_xy, green_xz, green_yy, green_yz, green_zz};
    GreenTensorDevicePointers gpu_green = get_device_green(host_green_components, tensor_count);
    const CUCOMPLEX *components[6]{};
    if (gpu_green) {
        for (size_t component = 0; component < 6; ++component)
            components[component] = gpu_green[component];
    }
    if (device_resident_solver) {
        size_t work_count = 0, polarizability_count = 0, temp_count = 0;
        bool buffers_ready = checked_multiply(size_t(*lda), size_t(*nlar), &work_count) &&
                             checked_multiply(size_t(*nmax), size_t(*niso), &temp_count) &&
                             checked_multiply(temp_count, size_t(*niso), &polarizability_count);
        if (buffers_ready && (is_gpbicor || is_gcors2)) {
            buffers_ready = get_gpbicor_device_arrays(xi, xr, incident, local_field, work,
                                                      size_t(n), work_count, true, &gpu_xi, &gpu_xr,
                                                      &gpu_b, &gpu_ffloc, &gpu_work);
            if (buffers_ready)
                mark_gpbicg_device_xr_current(xr, true);
        } else if (buffers_ready) {
            // A prior solve may leave XR's device cache marked current for
            // the same host address. Start this solve from the caller's XR;
            // mode-specific device initialization below can then replace it.
            mark_gpbicg_device_xr_current(xr, false);
            if (guess_mode >= 0 && guess_mode <= 2)
                buffers_ready = get_gpbicg_device_arrays_resident_initialized(
                    xi, xr, incident, work, size_t(n), work_count, true, guess_mode, &gpu_xi,
                    &gpu_xr, &gpu_b, &gpu_work);
            else
                buffers_ready =
                    get_gpbicg_device_arrays_resident(xi, xr, incident, work, size_t(n), work_count,
                                                      true, &gpu_xi, &gpu_xr, &gpu_b, &gpu_work);
        }
        if (!buffers_ready || !gpu_green ||
            !(gpu_polarizability =
                  copy_polarizability_to_device(polarizability, polarizability_count)) ||
            !(gpu_polarized_input = get_device_auxiliary_vector(size_t(n))) ||
            !(resident_blas = get_cublas_handle())) {
            *stop = 1;
            set_info(info, info_length, "Unable to initialize resident CUDA solver buffers");
            return;
        }
        if (guess_mode == 4 && std::getenv("IFDDA_TRACE_BPM_RESIDENT"))
            std::fprintf(
                stderr,
                "[inverserig BPM resident] host XI key=%p GPU XI=%p GPU XR=%p rightChan=%d\n",
                static_cast<void *>(xi), static_cast<void *>(gpu_xi), static_cast<void *>(gpu_xr),
                right_resident_chan ? 1 : 0);
        gpu_original_incident = gpu_b;
        if (guess_mode == 1 && (!use_resident_chan || !right_resident_chan)) {
            // For direct-guess methods, cdmlib.f90 deliberately leaves Xi
            // untouched on the host. Seed the resident iterate from the
            // unpreconditioned incident field before computing the Born
            // normalization. The left Chan path has not transformed gpu_b
            // yet; the right Chan path initializes its transformed iterate
            // while constructing the preconditioner and must keep that value.
            // Fortran inverserigopt checks XI(1:3) before the Hegedus/Born
            // normalization and skips it when all three entries are zero.
            // In MDDA/PML the first masked dipole can be exactly zero even
            // though the incident field is nonzero elsewhere. Respect that
            // branch instead of always normalizing the CUDA initial guess.
            const bool skip_born_gamma = n >= 3 &&
                std::abs(incident[0]) == CU_REAL(0) &&
                std::abs(incident[1]) == CU_REAL(0) &&
                std::abs(incident[2]) == CU_REAL(0);
            if (!copy(resident_blas, n, gpu_b, gpu_xi, "inverserig CUDA Born initial guess") ||
                (!skip_born_gamma && !initialize_born_gamma_resident_cuda(
                    green_xx, green_xy, green_xz, green_yy, green_yz, green_zz, vectx, vecty, vectz,
                    tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2,
                    nxy2, nz2, gpu_xi, gpu_b, polarizability, niso, ndipoles, size_t(n),
                    use_optimized_product, plan_xf, plan_xb, plan_yf, plan_yb, plan_zf, plan_zb,
                    &gamma)) ||
                !check_cuda(cuda_utils::tracked_memcpy(gpu_xr, gpu_xi,
                                                       size_t(n) * sizeof(CUCOMPLEX),
                                                       cudaMemcpyDeviceToDevice),
                            "inverserig CUDA", "seed XR from normalized Born guess")) {
                *stop = 1;
                set_info(info, info_length, "CUDA resident Born initial guess failed");
                return;
            }
            mark_gpbicg_device_xr_current(xr, true);
            std::fprintf(
                stderr,
                "[inverserig CUDA] Born initial guess on device (gamma=%.9g, skipped=%d)\n",
                double(gamma), skip_born_gamma ? 1 : 0);
        }
        if (guess_mode == 2) {
            if (!initialize_rytov_guess_resident_cuda(
                    green_xx, green_xy, green_xz, green_yy, green_yz, green_zz, vectx, vecty, vectz,
                    tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2,
                    nxy2, nz2, gpu_xi, gpu_xr, gpu_work, gpu_b, gpu_polarized_input, polarizability,
                    niso, ndipoles, size_t(n), false, plan_xf, plan_xb, plan_yf, plan_yb, plan_zf,
                    plan_zb) ||
                (right_resident_chan &&
                 (chan_preconditioner_apply_device_cuda_(
                      gpu_xi, tabdip, &resident_chan_preconditioner.nx,
                      &resident_chan_preconditioner.ny, &resident_chan_preconditioner.nz,
                      &resident_chan_preconditioner.nxm, &resident_chan_preconditioner.nym,
                      &resident_chan_preconditioner.nzm) != 0 ||
                  !check_cuda(cuda_utils::tracked_memcpy(gpu_xr, gpu_xi,
                                                         size_t(n) * sizeof(CUCOMPLEX),
                                                         cudaMemcpyDeviceToDevice),
                              "resident Rytov initialization", "seed right-preconditioned XR")))) {
                *stop = 1;
                set_info(info, info_length, "CUDA resident Rytov initial guess failed");
                return;
            }
            // The unpreconditioned Fortran driver applies the Hegedus
            // normalization to a Rytov start before entering the iterative
            // solver. Keep that operation resident as well: alpha*Xi, G*(alpha*Xi),
            // the two dot products, and the final scaling all stay on-device.
            if (!use_resident_chan) {
                const size_t tensor_count = size_t(*nxm + 1) * size_t(*nym + 1) * size_t(*nzm + 1);
                const size_t physical_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
                const size_t fft_vector_count = size_t(3) * size_t(*ntotal);
                const size_t green_output_count = size_t(3) * size_t(*nmax);
                const COMPLEX *host_tensors[6] = {green_xx, green_xy, green_xz,
                                                    green_yy, green_yz, green_zz};
                GreenTensorDevicePointers device_green = get_device_green(host_tensors, tensor_count);
                CUCOMPLEX *device_fft_vectors = get_device_vectors(fft_vector_count);
                CU_INT *device_tabdip = get_device_tabdip(tabdip, physical_count);
                CUCOMPLEX numerator = CUFFT_ZERO;
                CUCOMPLEX denominator = CUFFT_ZERO;
                const bool prepared =
                    green_output_count <= work_count && device_green && device_fft_vectors &&
                    device_tabdip &&
                    cuda_utils::copy(resident_blas, n, gpu_xi, gpu_polarized_input,
                                     "Rytov Hegedus stage vector") &&
                    check_cuda(polarization_inplace_wrapper(gpu_polarizability, gpu_polarized_input,
                                                            *niso, *nmax, *ndipoles),
                               "Rytov Hegedus normalization", "apply polarizability") &&
                    check_cuda(zero_vectors_wrapper(gpu_work, int(green_output_count)),
                               "Rytov Hegedus normalization", "clear Green output") &&
                    cuda_utils::check_cuda(cuda_utils::tracked_memcpy(gpu_work, gpu_xi,
                                                                      size_t(n) * sizeof(CUCOMPLEX),
                                                                      cudaMemcpyDeviceToDevice),
                                           "Rytov Hegedus normalization", "seed Green output");
                if (!prepared) {
                    *stop = 1;
                    set_info(info, info_length, "CUDA Rytov Hegedus preparation failed");
                    return;
                }
                if (use_optimized_product) {
                    produitfftmatvectoptdevice_(
                        device_green[0], device_green[1], device_green[2],
                        device_green[3], device_green[4],
                        device_green[5], vectx, vecty, vectz, ntotalm, ntotal, nmax,
                        nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, gpu_polarized_input,
                        gpu_work, plan_xf, plan_xb);
                } else {
                    produitfftmatvect3device_(
                        device_green[0], device_green[1], device_green[2],
                        device_green[3], device_green[4],
                        device_green[5], vectx, vecty, vectz, device_tabdip,
                        ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2,
                        nz2, gpu_polarized_input, gpu_work, plan_xb, plan_xf);
                }
                if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), "Rytov Hegedus normalization",
                                            "finish Green product") ||
                    !cuda_utils::dotc(resident_blas, n, gpu_work, gpu_b, &numerator,
                                      "Rytov Hegedus numerator") ||
                    !cuda_utils::dotc(resident_blas, n, gpu_work, gpu_work, &denominator,
                                      "Rytov Hegedus denominator")) {
                    *stop = 1;
                    set_info(info, info_length, "CUDA Rytov Hegedus products failed");
                    return;
                }
                const CU_REAL gamma =
                    denominator.x == CU_REAL(0) ? CU_REAL(1) : numerator.x / denominator.x;
                if (!std::isfinite(double(gamma)) ||
                    !cuda_utils::scale(resident_blas, n,
                                       device_value(HostComplex(gamma, CU_REAL(0))), gpu_xi,
                                       "Rytov Hegedus scale") ||
                    !cuda_utils::copy(resident_blas, n, gpu_xi, gpu_xr,
                                      "Rytov Hegedus publish start")) {
                    *stop = 1;
                    set_info(info, info_length, "CUDA Rytov Hegedus scaling failed");
                    return;
                }
                std::fprintf(stderr, "[inverserig CUDA] Rytov Hegedus normalization on GPU\n");
            }
            mark_gpbicg_device_xr_current(xr, true);
            std::fprintf(stderr, "[inverserig CUDA] Rytov initial guess constructed on device\n");
        }
        if (guess_mode == 4) {
            // beampropagation writes the requested initial iterate into Xi.
            // The nonresident Fortran driver then applies the same Green
            // projection normalization used for a nonzero supplied guess.
            // Perform that normalization on-device before seeding XR so the
            // resident path preserves both the Fortran state transition and
            // the no-host-handoff property.
            if (!use_resident_chan &&
                !initialize_born_gamma_resident_cuda(
                    green_xx, green_xy, green_xz, green_yy, green_yz, green_zz, vectx, vecty, vectz,
                    tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2,
                    nxy2, nz2, gpu_xi, gpu_b, polarizability, niso, ndipoles, size_t(n),
                    use_optimized_product, plan_xf, plan_xb, plan_yf, plan_yb, plan_zf, plan_zb,
                    &gamma)) {
                *stop = 1;
                set_info(info, info_length, "CUDA resident BPM normalization failed");
                return;
            }
            if (!check_cuda(cuda_utils::tracked_memcpy(gpu_xr, gpu_xi,
                                                       size_t(n) * sizeof(CUCOMPLEX),
                                                       cudaMemcpyDeviceToDevice),
                            "inverserig CUDA BPM initial guess", "seed XR from Xi")) {
                *stop = 1;
                set_info(info, info_length, "Unable to seed CUDA XR from the BPM initial guess");
                return;
            }
            mark_gpbicg_device_xr_current(xr, true);
            if (!use_resident_chan)
                std::fprintf(stderr,
                             "[inverserig CUDA] normalized BPM guess on device (gamma=%.9g)\n",
                             double(gamma));
        }
        if (right_resident_chan && std::getenv("IFDDA_TRACE_CHAN_PRODUCT")) {
            std::vector<CUCOMPLEX> values(static_cast<size_t>(n));
            if (cuda_data::copy_device_to_host(values.data(), gpu_xi, size_t(n) * sizeof(CUCOMPLEX),
                                               cuda_utils::TransferCategory::Diagnostic) ==
                cudaSuccess) {
                double norm2 = 0.0;
                for (const CUCOMPLEX value : values)
                    norm2 += double(value.x) * double(value.x) + double(value.y) * double(value.y);
                std::fprintf(stderr, "[CHAN CUDA acquired Xi] (%.17g,%.17g) (%.17g,%.17g) %.17g\n",
                             double(values[0].x), double(values[0].y), double(values[1].x),
                             double(values[1].y), std::sqrt(norm2));
            }
        }
        set_gpbicg_device_resident(xi, true);
        // The driver has just uploaded Xi into the resident buffer. Keep that
        // copy authoritative when the reverse-communication solver makes its
        // first call; otherwise it can upload a host Xi already reused by the
        // surrounding driver for the incident field.
        mark_gpbicg_device_xi_current(xi, true);
        if (external_solution_buffer || is_gpbicor || is_gcors2)
            set_gpbicg_device_resident(local_field, true);
        resident_guard.host_xi = xi;
        resident_guard.host_output =
            (external_solution_buffer || is_gpbicor || is_gcors2) ? local_field : nullptr;
        resident_guard.host_work = work;
        set_gpbicg_device_resident(work, true);
        resident_guard.active = true;

        if (use_resident_chan && !right_resident_chan) {
            size_t incident_snapshot_count = 0;
            CUCOMPLEX *incident_snapshot = nullptr;
            const bool snapshot_ready =
                checked_multiply(size_t(n), size_t(3), &incident_snapshot_count) &&
                (incident_snapshot = get_device_result_vectors(incident_snapshot_count)) !=
                    nullptr &&
                check_cuda(cuda_utils::tracked_memcpy(incident_snapshot, gpu_b,
                                                      size_t(n) * sizeof(CUCOMPLEX),
                                                      cudaMemcpyDeviceToDevice),
                           "inverserig CUDA", "preserve unpreconditioned incident field");
            if (!snapshot_ready) {
                *stop = 1;
                set_info(info, info_length, "Unable to preserve incident field for CUDA results");
                return;
            }
            gpu_original_incident = incident_snapshot;
            if (chan_preconditioner_apply_device_cuda_(
                    gpu_b, tabdip, &resident_chan_preconditioner.nx,
                    &resident_chan_preconditioner.ny, &resident_chan_preconditioner.nz,
                    &resident_chan_preconditioner.nxm, &resident_chan_preconditioner.nym,
                    &resident_chan_preconditioner.nzm) != 0 ||
                !register_gpbicg_device_b_alias(incident, gpu_b, static_cast<size_t>(n))) {
                *stop = 1;
                set_info(info, info_length, "Unable to apply left Chan to the resident CUDA RHS");
                return;
            }
            resident_guard.rhs_alias_registered = true;
        }
    }

    // Born-series iteration is not a reverse-communication Krylov solver.
    // Keep its vector updates in this C++ driver, use the CUDA matrix-product
    // entry points for Green convolutions, and cuBLAS for norms/AXPY.
    if (is_bornserie) {
        const size_t bytes = size_t(n) * sizeof(CUCOMPLEX);
        size_t tensor_count = 0, physical_count = 0, vector_count = 0;
        bool sizes_ok = checked_multiply(size_t(*nxm + 1), size_t(*nym + 1), &tensor_count) &&
                        checked_multiply(tensor_count, size_t(*nzm + 1), &tensor_count) &&
                        checked_multiply(size_t(*nx), size_t(*ny), &physical_count) &&
                        checked_multiply(physical_count, size_t(*nz), &physical_count) &&
                        checked_multiply(size_t(3), size_t(*ntotal), &vector_count);
        const COMPLEX *host_tensors[6] = {green_xx, green_xy, green_xz,
                                            green_yy, green_yz, green_zz};
        GreenTensorDevicePointers device_green = sizes_ok ? get_device_green(host_tensors, tensor_count) : GreenTensorDevicePointers{};
        CU_INT *device_tabdip = sizes_ok ? get_device_tabdip(tabdip, physical_count) : nullptr;
        CUCOMPLEX *fft_vectors = sizes_ok ? get_device_vectors(vector_count) : nullptr;
        const bool initialized =
            sizes_ok && device_green && device_tabdip && fft_vectors &&
            norm2(resident_blas, n, gpu_b, &norm, "BORNSERIE CUDA") &&
            std::isfinite(double(norm)) && norm > CU_REAL(0) &&
            check_cuda(cuda_utils::tracked_memcpy(gpu_xr, gpu_b, bytes, cudaMemcpyDeviceToDevice),
                       "BORNSERIE CUDA", "initialize Born iterate") &&
            check_cuda(cuda_utils::tracked_memcpy(gpu_work, gpu_b, bytes, cudaMemcpyDeviceToDevice),
                       "BORNSERIE CUDA", "initialize local field");
        if (!initialized) {
            *stop = 1;
            set_info(info, info_length, "Unable to initialize Born-series CUDA buffers");
            return;
        }

        const CUCOMPLEX *const tensors[6] = {device_green[0],
                                             device_green[1],
                                             device_green[2],
                                             device_green[3],
                                             device_green[4],
                                             device_green[5]};
        const CUCOMPLEX one = device_value(HostComplex(1, 0));
        const CUCOMPLEX minus_one = device_value(HostComplex(-1, 0));
        const int limit = *max_iterations;
        *final_residual = std::numeric_limits<CU_REAL>::infinity();
        *products = 0;
        *product_calls = 0;
        bool success = true;
        for (int iteration = 1; iteration <= limit; ++iteration) {
            // xi = polarizability * xr; then xr = G * xi.
            success =
                check_cuda(cuda_utils::tracked_memcpy(gpu_polarized_input, gpu_xr, bytes,
                                                      cudaMemcpyDeviceToDevice),
                           "BORNSERIE CUDA", "stage Born polarization") &&
                check_cuda(polarization_inplace_wrapper(gpu_polarizability, gpu_polarized_input,
                                                        *niso, *nmax, *ndipoles),
                           "BORNSERIE CUDA", "polarize Born iterate") &&
                check_cuda(cudaMemset(gpu_xr, 0, bytes), "BORNSERIE CUDA", "clear scattered field");
            if (!success)
                break;
            if (use_optimized_product) {
                produitfftmatvectoptdevice_(tensors[0], tensors[1], tensors[2], tensors[3],
                                            tensors[4], tensors[5], vectx, vecty, vectz, ntotalm,
                                            ntotal, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2,
                                            nz2, gpu_polarized_input, gpu_xr, plan_xf, plan_xb);
            } else {
                produitfftmatvect3device_(
                    tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], vectx,
                    vecty, vectz, device_tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx,
                    ny, nz, nx2, ny2, nxy2, nz2, gpu_polarized_input, gpu_xr, plan_xb, plan_xf);
            }
            success = check_cuda(cudaDeviceSynchronize(), "BORNSERIE CUDA",
                                 "finish Born Green product") &&
                      scale(resident_blas, n, minus_one, gpu_xr, "BORNSERIE CUDA") &&
                      axpy(resident_blas, n, one, gpu_xr, gpu_work, "BORNSERIE CUDA");
            if (!success)
                break;

            // Residual = (I - G alpha) FFloc - FF0.
            success =
                check_cuda(cuda_utils::tracked_memcpy(gpu_polarized_input, gpu_work, bytes,
                                                      cudaMemcpyDeviceToDevice),
                           "BORNSERIE CUDA", "stage local-field polarization") &&
                check_cuda(polarization_inplace_wrapper(gpu_polarizability, gpu_polarized_input,
                                                        *niso, *nmax, *ndipoles),
                           "BORNSERIE CUDA", "polarize local field") &&
                check_cuda(
                    cuda_utils::tracked_memcpy(gpu_xi, gpu_work, bytes, cudaMemcpyDeviceToDevice),
                    "BORNSERIE CUDA", "initialize residual");
            if (!success)
                break;
            if (use_optimized_product) {
                produitfftmatvectoptdevice_(tensors[0], tensors[1], tensors[2], tensors[3],
                                            tensors[4], tensors[5], vectx, vecty, vectz, ntotalm,
                                            ntotal, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2,
                                            nz2, gpu_polarized_input, gpu_xi, plan_xf, plan_xb);
            } else {
                produitfftmatvect3device_(
                    tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], vectx,
                    vecty, vectz, device_tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx,
                    ny, nz, nx2, ny2, nxy2, nz2, gpu_polarized_input, gpu_xi, plan_xb, plan_xf);
            }
            success = check_cuda(cudaDeviceSynchronize(), "BORNSERIE CUDA",
                                 "finish residual Green product") &&
                      axpy(resident_blas, n, minus_one, gpu_b, gpu_xi, "BORNSERIE CUDA") &&
                      norm2(resident_blas, n, gpu_xi, final_residual, "BORNSERIE CUDA");
            if (!success)
                break;
            *final_residual /= norm;
            *products = iteration;
            *product_calls = 2 * iteration;
            *products = iteration;
            std::fprintf(stderr, "[BORNSERIE CUDA] iteration %d residual %.9g\n", iteration,
                         double(*final_residual));
            if (*final_residual <= *tolerance)
                break;
        }
        if (!success || !std::isfinite(double(*final_residual))) {
            *stop = 1;
            set_info(info, info_length, "Born-series CUDA iteration failed");
            return;
        }

        // Publish the same arrays the Fortran implementation leaves behind.
        success = check_cuda(cuda_utils::tracked_memcpy(gpu_polarized_input, gpu_work, bytes,
                                                        cudaMemcpyDeviceToDevice),
                             "BORNSERIE CUDA", "stage final polarization") &&
                  check_cuda(polarization_inplace_wrapper(gpu_polarizability, gpu_polarized_input,
                                                          *niso, *nmax, *ndipoles),
                             "BORNSERIE CUDA", "compute final polarization") &&
                  copy_device_complex_to_host(local_field, gpu_work, size_t(n)) &&
                  copy_device_complex_to_host(field, gpu_polarized_input, size_t(n)) &&
                  copy_device_complex_to_host(xi, gpu_xi, size_t(n)) &&
                  copy_device_complex_to_host(xr, gpu_xr, size_t(n));
        if (!success) {
            *stop = 1;
            set_info(info, info_length, "Unable to publish Born-series result");
            return;
        }
        return;
    }

    for (;;) {
        if (is_gpbicor) {
            gpbicor_(gpu_xi, gpu_xr, gpu_b, gpu_ffloc, lda, ndim, nlar, &nou,
                     gpu_work, products, max_iterations, tolerance, &norm, &alpha, &beta, &eta, &dzeta,
                     &rho, &rhoa, &nstat, &steperr);
        } else if (is_cors) {
            cors_(gpu_xi, gpu_xr, gpu_b, lda, ndim, nlar, &nou, gpu_work, products,
                  max_iterations, tolerance, &norm, &alpha, &rho, &rhoa, &beta, &nstat, &steperr);
        } else if (is_tfqmr) {
            tfqmr_(gpu_ffloc, gpu_xi, gpu_xr, gpu_b, lda, ndim, nlar, &nou,
                   gpu_work, products, max_iterations, tolerance, &norm, &alpha, &beta, &rho, &rhoa,
                   &dzeta, &eta, &nstat, &steperr);
        } else if (is_zcg) {
            zcg_(gpu_xi, gpu_xr, gpu_b, &norm, gpu_work, &alpha, &rho, &eta, lda,
                 ndim, nlar, &nou, &nstat, products, max_iterations, &zcg_tolet, tolerance);
        } else if (is_pimzbicgstab) {
            pimzbicgstab_(local_field, xi, xr, incident, lda, nlar, ndim,
                          &nou, work, &pimz_alpha, &pimz_omega, &pimz_rho, &norm, tolerance, products,
                          max_iterations, &nstat, &steperr);
        } else if (qmrbicgstab_method != 0) {
            qmrbicgstab_(gpu_ffloc, gpu_xi, gpu_xr, gpu_b, lda, ndim, nlar,
                         &nou, gpu_work, products, max_iterations, tolerance, &qmr_solver_residual,
                         &norm, &alpha, &rho, &qmr_omega, &qmr_tau, &qmr_tau_tilde, &qmr_theta,
                         &qmr_theta_tilde, &eta, &qmr_eta_tilde, &qmrbicgstab_method, &nstat,
                         &steperr);
        } else if (bicglstab_l != 0) {
            zbicgstabl_(gpu_xi, gpu_xr, gpu_b, gpu_ffloc, &norm, gpu_work, &alpha,
                        &rho, &rhoa, &bicglstab_l, &bicglstab_nr, &bicglstab_nu, &bicglstab_j, lda,
                        ndim, nlar, &nou, &nstat, &steperr, products, max_iterations,
                        &bicglstab_tolet, tolerance);
        } else if (gpbicgstabl_l != 0) {
            gpbicgstabl_(gpu_xi, gpu_xr, gpu_b, gpu_ffloc, &norm, gpu_work,
                         &alpha, &beta, &gpbicgstabl_sigma, &dzeta, &gpbicgstabl_l, lda, ndim, nlar,
                         &nou, &gpbicgstabl_nr, &gpbicgstabl_np, &gpbicgstabl_ns, &gpbicgstabl_nq,
                         &gpbicgstabl_j, &nstat, &steperr, products, max_iterations,
                         &gpbicgstabl_tolet, tolerance);
        } else if (is_pimzqmr) {
            pimzqmr_(gpu_ffloc, gpu_xi, gpu_xr, gpu_b, gpu_work, &norm, lda, ndim,
                     nlar, &qmr_lambda, &qmr_kappa, &qmrcla_theta, &qmrcla_gamma, &qmr_ksi,
                     &qmr_rho, &qmr_epsilon, &qmr_mu, &qmrcla_tau, &qmr_dots1, &qmr_dots2,
                     &qmr_dots3, &qmr_dots4, &nou, &qmr_nt, products, max_iterations, &qmr_tolet,
                     tolerance, &nstat, &steperr);
        } else if (is_idrs) {
            idrs_(gpu_ffloc, gpu_xi, gpu_xr, gpu_b, lda, ndim, nlar, &nou,
                  &idrs_s, gpu_work, products, max_iterations, tolerance, &idrs_residual, &norm,
                  &idrs_omega, &idrs_is, &nstat, &steperr);
        } else if (is_gcors2) {
            gcors2_(gpu_xi, gpu_xr, gpu_ffloc, gpu_b, lda, ndim, nlar, &nou,
                    gpu_work, products, max_iterations, tolerance, &norm, &alpha, &gcors_alpha_t, &rho,
                    &gcors_rho_t, &nstat, &steperr);
        } else {
            solver(gpu_xi, gpu_xr, gpu_b, lda, ndim, nlar, &nou, gpu_work, products,
                   max_iterations, tolerance, &norm, &alpha, &beta, &eta, &dzeta, &r0rn, &nstat,
                   &steperr);
        }
        if (nstat < 0) {
            *stop = 1;
            set_info(info, info_length, "Problem to solve Ax=b!");
            std::fprintf(stderr, "[inverserig CUDA] %s error %d\n", solver_name, steperr);
            return;
        }

        ++(*product_calls);
        if (is_pimzqmr && nstat == 1)
            qmr_nt = 1;
        if (external_solution_buffer && nstat == 1 && !device_resident_solver)
            std::copy(local_field, local_field + n, xi);
        if (device_resident_solver && external_solution_buffer && nstat == 1) {
            CUCOMPLEX *gpu_solution = get_tfqmr_device_x(local_field, xi, size_t(n), false);
            if (!gpu_solution ||
                !check_cuda(cuda_utils::tracked_memcpy(gpu_xi, gpu_solution,
                                                       size_t(n) * sizeof(CUCOMPLEX),
                                                       cudaMemcpyDeviceToDevice),
                            "inverserig CUDA solver", "publish device solution")) {
                *stop = 1;
                set_info(info, info_length, "Failed to publish resident CUDA solver solution");
                return;
            }
        }
        if (device_resident_solver) {
            const bool qmr_extra_product = is_pimzqmr && qmr_nt == 2;
            if ((!qmr_extra_product &&
                 (!check_cuda(cuda_utils::tracked_memcpy(gpu_xr, gpu_xi,
                                                         size_t(n) * sizeof(CUCOMPLEX),
                                                         cudaMemcpyDeviceToDevice),
                              "inverserig CUDA solver", "copy solution on device") ||
                  (!right_resident_chan &&
                   !check_cuda(cuda_utils::tracked_memcpy(gpu_polarized_input, gpu_xi,
                                                          size_t(n) * sizeof(CUCOMPLEX),
                                                          cudaMemcpyDeviceToDevice),
                               "inverserig CUDA solver", "prepare polarization input")) ||
                  (right_resident_chan && *products == 0 &&
                   std::getenv("IFDDA_TRACE_CHAN_PRODUCT") &&
                   [&]() {
                       std::vector<CUCOMPLEX> values(static_cast<size_t>(n));
                       std::vector<CUCOMPLEX> solution(static_cast<size_t>(n));
                       if (cuda_data::copy_device_to_host(
                               values.data(), gpu_xr, size_t(n) * sizeof(CUCOMPLEX),
                               cuda_utils::TransferCategory::Diagnostic) == cudaSuccess &&
                           cuda_data::copy_device_to_host(
                               solution.data(), gpu_xi, size_t(n) * sizeof(CUCOMPLEX),
                               cuda_utils::TransferCategory::Diagnostic) == cudaSuccess) {
                           double norm2 = 0.0, solution_norm2 = 0.0;
                           for (size_t i = 0; i < values.size(); ++i) {
                               const CUCOMPLEX value = values[i];
                               const CUCOMPLEX solution_value = solution[i];
                               norm2 += double(value.x) * double(value.x) +
                                        double(value.y) * double(value.y);
                               solution_norm2 +=
                                   double(solution_value.x) * double(solution_value.x) +
                                   double(solution_value.y) * double(solution_value.y);
                           }
                           std::fprintf(stderr,
                                        "[CHAN CUDA before apply] XR=(%.17g,%.17g) (%.17g,%.17g) "
                                        "%.17g Xi=(%.17g,%.17g) (%.17g,%.17g) %.17g\\n",
                                        double(values[0].x), double(values[0].y),
                                        double(values[1].x), double(values[1].y), std::sqrt(norm2),
                                        double(solution[0].x), double(solution[0].y),
                                        double(solution[1].x), double(solution[1].y),
                                        std::sqrt(solution_norm2));
                       }
                       return false;
                   }()) ||
                  (right_resident_chan &&
//-------------------------------------------
                   chan_preconditioner_apply_device_cuda_(
                       gpu_xr, tabdip, &resident_chan_preconditioner.nx,
                       &resident_chan_preconditioner.ny, &resident_chan_preconditioner.nz,
                       &resident_chan_preconditioner.nxm, &resident_chan_preconditioner.nym,
                       &resident_chan_preconditioner.nzm) != 0) ||
                  (right_resident_chan &&
                   !check_cuda(cuda_utils::tracked_memcpy(gpu_polarized_input, gpu_xr,
                                                          size_t(n) * sizeof(CUCOMPLEX),
                                                          cudaMemcpyDeviceToDevice),
                               "inverserig CUDA solver", "prepare right-preconditioned input")) ||
                  !check_cuda(polarization_inplace_wrapper(gpu_polarizability, gpu_polarized_input,
                                                           *niso, *nmax, *ndipoles),
                              "inverserig CUDA solver", "apply polarization"))) ||
                cudaDeviceSynchronize() != cudaSuccess) {
                *stop = 1;
                set_info(info, info_length, "Failed to prepare resident CUDA solver product");
                return;
            }
            if (right_resident_chan && *products == 0 && std::getenv("IFDDA_TRACE_CHAN_PRODUCT")) {
                std::vector<CUCOMPLEX> values(static_cast<size_t>(n));
                if (cuda_data::copy_device_to_host(
                        values.data(), gpu_xr, size_t(n) * sizeof(CUCOMPLEX),
                        cuda_utils::TransferCategory::Diagnostic) == cudaSuccess) {
                    double norm2 = 0.0;
                    for (const CUCOMPLEX value : values)
                        norm2 +=
                            double(value.x) * double(value.x) + double(value.y) * double(value.y);
                    std::fprintf(stderr,
                                 "[CHAN CUDA after solve] (%.17g,%.17g) (%.17g,%.17g) %.17g\n",
                                 double(values[0].x), double(values[0].y), double(values[1].x),
                                 double(values[1].y), std::sqrt(norm2));
                }
            }
            if (qmr_extra_product) {
                const CUCOMPLEX *extra_product_input = gpu_xi;
                // QMRCLA's nt=2 request applies the transpose operator. For
                // B=A*P^-1, B^T=P^-T*A^T, so the right Chan inverse follows
                // the transpose product. For B=P^-1*A, B^T=A^T*P^-T, so
                // the left Chan inverse is applied to the transpose input.
                // Chan blocks are complex symmetric, hence P^-T=P^-1.
                if (use_resident_chan && !right_resident_chan) {
                    if (!check_cuda(cuda_utils::tracked_memcpy(gpu_polarized_input, gpu_xi,
                                                               size_t(n) * sizeof(CUCOMPLEX),
                                                               cudaMemcpyDeviceToDevice),
                                    "inverserig QMRCLA",
                                    "stage left-preconditioned transpose input") ||
                        chan_preconditioner_apply_device_cuda_(
                            gpu_polarized_input, tabdip, &resident_chan_preconditioner.nx,
                            &resident_chan_preconditioner.ny, &resident_chan_preconditioner.nz,
                            &resident_chan_preconditioner.nxm, &resident_chan_preconditioner.nym,
                            &resident_chan_preconditioner.nzm) != 0) {
                        *stop = 1;
                        set_info(info, info_length,
                                 "QMRCLA left Chan preconditioning of transpose input failed");
                        return;
                    }
                    extra_product_input = gpu_polarized_input;
                }
                // For Chan-right, form A^T*x in a fresh output before applying
                // P^-T. The historical Chan-left path keeps its existing
                // reverse-communication staging, which is separately covered
                // by the established left-preconditioned reference case.
                if (right_resident_chan &&
                    !check_cuda(cudaMemset(gpu_xr, 0, size_t(n) * sizeof(CUCOMPLEX)),
                                "inverserig QMRCLA", "clear transpose product output")) {
                    *stop = 1;
                    set_info(info, info_length,
                             "QMRCLA transpose product output initialization failed");
                    return;
                }
                if (use_optimized_product) {
                    produitfftmatvectoptdevice_(components[0], components[1], components[2],
                                                components[3], components[4], components[5], vectx,
                                                vecty, vectz, ntotalm, ntotal, nmax, nxm, nym, nzm,
                                                nx, ny, nz, nx2, ny2, nxy2, nz2,
                                                extra_product_input, gpu_xr, plan_xf, plan_xb);
                } else {
                    produitfftmatvect3host_(components[0], components[1], components[2],
                                              components[3], components[4], components[5], vectx,
                                              vecty, vectz, tabdip, ntotalm, ntotal, nmax, ndipole,
                                              nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,
                                              extra_product_input, gpu_xr, plan_xb, plan_xf);
                }
                const CUCOMPLEX one = device_value(HostComplex(1, 0));
                if (!check_cuda(polarization_inplace_wrapper(gpu_polarizability, gpu_xr, *niso,
                                                             *nmax, *ndipoles),
                                "inverserig QMRCLA", "polarize extra product") ||
                    !axpy(resident_blas, n, one, right_resident_chan ? extra_product_input : gpu_xi,
                          gpu_xr, "inverserig QMRCLA")) {
                    *stop = 1;
                    set_info(info, info_length, "QMRCLA extra resident product failed");
                    return;
                }
                // B = A P^-1 for Chan-right, so B^T = P^-T A^T.
                // Apply P^-1 after the complete transpose operator, using
                // the complex-symmetric Chan blocks (P^-T = P^-1).
                if (right_resident_chan &&
                    chan_preconditioner_apply_device_cuda_(
                        gpu_xr, tabdip, &resident_chan_preconditioner.nx,
                        &resident_chan_preconditioner.ny, &resident_chan_preconditioner.nz,
                        &resident_chan_preconditioner.nxm, &resident_chan_preconditioner.nym,
                        &resident_chan_preconditioner.nzm) != 0) {
                    *stop = 1;
                    set_info(info, info_length, "QMRCLA right Chan transpose application failed");
                    return;
                }
            } else if (use_optimized_product) {
                produitfftmatvectoptdevice_(
                    components[0], components[1], components[2], components[3], components[4],
                    components[5], vectx, vecty, vectz, ntotalm, ntotal, nmax, nxm, nym, nzm, nx,
                    ny, nz, nx2, ny2, nxy2, nz2, gpu_polarized_input, gpu_xr, plan_xf, plan_xb);
            } else {
                produitfftmatvect3host_(components[0], components[1], components[2],
                                          components[3], components[4], components[5], vectx, vecty,
                                          vectz, tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym,
                                          nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, gpu_polarized_input,
                                          gpu_xr, plan_xb, plan_xf);
            }
            if (!check_cuda(cudaDeviceSynchronize(), "inverserig CUDA solver",
                            "finish resident matrix product")) {
                *stop = 1;
                set_info(info, info_length, "Resident CUDA solver matrix product failed");
                return;
            }
            if (right_resident_chan && *products == 0 && std::getenv("IFDDA_TRACE_CHAN_PRODUCT")) {
                std::vector<CUCOMPLEX> values(static_cast<size_t>(n));
                if (cuda_data::copy_device_to_host(
                        values.data(), gpu_xr, size_t(n) * sizeof(CUCOMPLEX),
                        cuda_utils::TransferCategory::Diagnostic) == cudaSuccess) {
                    double norm2 = 0.0;
                    for (const CUCOMPLEX value : values)
                        norm2 +=
                            double(value.x) * double(value.x) + double(value.y) * double(value.y);
                    std::fprintf(stderr,
                                 "[CHAN CUDA after Green] (%.17g,%.17g) (%.17g,%.17g) %.17g\n",
                                 double(values[0].x), double(values[0].y), double(values[1].x),
                                 double(values[1].y), std::sqrt(norm2));
                }
            }
            if (use_resident_chan && !right_resident_chan &&
                chan_preconditioner_apply_device_cuda_(
                    gpu_xr, tabdip, &resident_chan_preconditioner.nx,
                    &resident_chan_preconditioner.ny, &resident_chan_preconditioner.nz,
                    &resident_chan_preconditioner.nxm, &resident_chan_preconditioner.nym,
                    &resident_chan_preconditioner.nzm) != 0) {
                *stop = 1;
                set_info(info, info_length, "Resident Chan preconditioner application failed");
                return;
            }
            mark_gpbicg_device_xr_current(xr, true);
        } else {
            std::copy(xi, xi + n, xr);
            if (is_pimzqmr && qmr_nt == 2) {
                apply_matrix_product(components[0], components[1], components[2], components[3],
                                     components[4], components[5], vectx, vecty, vectz, tabdip,
                                     ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz,
                                     nx2, ny2, nxy2, nz2, gpu_xi, gpu_xr,
                                     plan_xf, plan_xb, plan_yf, plan_yb, plan_zf, plan_zb,
                                     use_optimized_product, tensor_count);
                produitpolavect_(gpu_polarizability, niso, nmax, ndipoles, gpu_xr,
                                 gpu_polarized_input);
                for (int i = 0; i < three_n; ++i)
                    gpu_xr[i] = device_value(host_value(gpu_polarized_input[i]) +
                                             host_value(gpu_xi[i]));
            } else {
                if (!is_tfqmr && !is_pimzbicgstab && qmrbicgstab_method == 0 && bicglstab_l == 0 &&
                    gpbicgstabl_l == 0 && !is_pimzqmr && !is_idrs && nstat == 1)
                    std::copy(xi, xi + n, local_field);

                produitpolavect_(gpu_polarizability, niso, nmax, ndipoles, gpu_xr,
                                 gpu_polarized_input);
                apply_matrix_product(components[0], components[1], components[2], components[3],
                                     components[4], components[5], vectx, vecty, vectz, tabdip,
                                     ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz,
                                     nx2, ny2, nxy2, nz2, gpu_polarized_input, gpu_xr,
                                     plan_xf, plan_xb, plan_yf, plan_yb, plan_zf, plan_zb,
                                     use_optimized_product, tensor_count);
            }
        }
        if (*stop == -1) {
            set_info(info, info_length, "Calculation cancelled during iterative method");
            return;
        }
        if (nstat != 1)
            continue;

        if (device_resident_solver) {
            const CUCOMPLEX minus_one = device_value(HostComplex(-1, 0));
            CU_REAL residual_norm = CU_REAL(0);
            if (std::getenv("IFDDA_TRACE_CHAN_PRODUCT")) {
                CUCOMPLEX values[2]{};
                CUCOMPLEX rhs[2]{};
                if (cuda_data::copy_device_to_host(values, gpu_xr, 2 * sizeof(CUCOMPLEX),
                                                   cuda_utils::TransferCategory::Diagnostic) ==
                        cudaSuccess &&
                    cuda_data::copy_device_to_host(rhs, gpu_b, 2 * sizeof(CUCOMPLEX),
                                                   cuda_utils::TransferCategory::Diagnostic) ==
                        cudaSuccess)
                    std::fprintf(stderr,
                                 "[RESIDENT final operator] XR=(%.17g,%.17g)(%.17g,%.17g) "
                                 "B=(%.17g,%.17g)(%.17g,%.17g)\n",
                                 double(values[0].x), double(values[0].y), double(values[1].x),
                                 double(values[1].y), double(rhs[0].x), double(rhs[0].y),
                                 double(rhs[1].x), double(rhs[1].y));
            }
            if (!axpy(resident_blas, n, minus_one, gpu_b, gpu_xr, "inverserig CUDA solver") ||
                !norm2(resident_blas, n, gpu_xr, &residual_norm, "inverserig CUDA solver") ||
                !std::isfinite(double(residual_norm)) || norm == CU_REAL(0) ||
                !copy_device_complex_to_host(xr, gpu_xr, size_t(n))) {
                *stop = 1;
                set_info(info, info_length, "Failed to finalize resident CUDA solver solution");
                return;
            }
            if (right_resident_chan &&
                (!check_cuda(
                     cuda_utils::tracked_memcpy(gpu_xr, gpu_xi, size_t(n) * sizeof(CUCOMPLEX),
                                                cudaMemcpyDeviceToDevice),
                     "inverserig CUDA solver", "prepare final right-preconditioned solution") ||
                 chan_preconditioner_apply_device_cuda_(
                     gpu_xr, tabdip, &resident_chan_preconditioner.nx,
                     &resident_chan_preconditioner.ny, &resident_chan_preconditioner.nz,
                     &resident_chan_preconditioner.nxm, &resident_chan_preconditioner.nym,
                     &resident_chan_preconditioner.nzm) != 0)) {
                *stop = 1;
                set_info(info, info_length, "Failed to finalize right-preconditioned solution");
                return;
            }
            if (!copy_device_complex_to_host(local_field, right_resident_chan ? gpu_xr : gpu_xi,
                                             size_t(n)) ||
                (!device_only_field_result &&
                 !copy_device_complex_to_host(field, gpu_polarized_input, size_t(n))) ||
                !copy_device_complex_to_host(xi, gpu_polarized_input, size_t(n)) ||
                !register_device_result_vector_alias(incident, gpu_original_incident, size_t(n)) ||
                !register_device_result_vector_alias(field, gpu_polarized_input, size_t(n)) ||
                !register_device_result_vector_alias(
                    local_field, right_resident_chan ? gpu_xr : gpu_xi, size_t(n))) {
                *stop = 1;
                set_info(info, info_length,
                         "Failed to publish resident solution and result aliases");
                return;
            }
            if (std::getenv("IFDDA_LOCAL_FIELD_TRACE")) {
                double local_max = 0.0;
                size_t local_max_index = 0;
                for (size_t i = 0; i < size_t(n); ++i) {
                    const double magnitude =
                        std::hypot(double(local_field[i].real()), double(local_field[i].imag()));
                    if (magnitude > local_max) {
                        local_max = magnitude;
                        local_max_index = i;
                    }
                }
                double dipole_max = 0.0;
                int dipole_max_index = -1;
                for (int dipole = 0; dipole < *ndipole; ++dipole) {
                    const int grid_index = tabdip[dipole] - 1;
                    if (grid_index < 0 || size_t(3) * size_t(grid_index + 1) > size_t(n))
                        continue;
                    for (int component = 0; component < 3; ++component) {
                        const size_t index = size_t(3) * size_t(grid_index) + size_t(component);
                        const double magnitude =
                            std::hypot(double(local_field[index].real()),
                                       double(local_field[index].imag()));
                        if (magnitude > dipole_max) {
                            dipole_max = magnitude;
                            dipole_max_index = dipole;
                        }
                    }
                }
                std::fprintf(stderr,
                             "[inverserig CUDA trace] local_field max=%.17g index=%zu "
                             "dipole_max=%.17g dipole=%d n=%d first=(%.17g,%.17g)\n",
                             local_max, local_max_index, dipole_max, dipole_max_index, n,
                             double(local_field[0].real()), double(local_field[0].imag()));
            }
            *final_residual = residual_norm / norm;
            std::copy(xi, xi + n, field);
            if (!is_gpbicor && !is_zcg && !is_pimzbicgstab && qmrbicgstab_method == 0 &&
                !is_pimzqmr && !is_gcors2)
                ++(*products);
            return;
        }

        double residual_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            const HostComplex residual(xr[i].real() - incident[i].real(),
                                       xr[i].imag() - incident[i].imag());
            xr[i] = COMPLEX(static_cast<REAL>(residual.real()),
                            static_cast<REAL>(residual.imag()));
            field[i] = xi[i];
            residual_sum += std::norm(residual);
        }
        *final_residual = CU_REAL(std::sqrt(residual_sum) / double(norm));
        // GPBICOR's CUDA reverse-communication entry already reports the
        // final iteration index, whereas the other GPBICG-family entries
        // require the same final increment performed by inverserig.f90.
        if (!is_gpbicor && !is_zcg && !is_pimzbicgstab && qmrbicgstab_method == 0 && !is_pimzqmr &&
            !is_gcors2)
            ++(*products);
        return;
    }
}
