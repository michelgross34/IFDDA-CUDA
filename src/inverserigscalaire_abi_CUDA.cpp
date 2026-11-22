#include "cdmlib_cpp_abi.hpp"
#include "kernel.h"
#include "cuda_solver_interfaces.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

static_assert(sizeof(COMPLEX) == sizeof(CUCOMPLEX),
              "Fortran and CUDA complex values must have equal storage size");

//-----------------------------------------------------------
namespace {

struct ScalarLegacyBuffers {
    std::vector<COMPLEX> vectors_x;
    std::vector<COMPLEX> vectors_y;
    std::vector<COMPLEX> ff_scalar;
    std::vector<COMPLEX> ff_local_scalar;
    std::vector<COMPLEX> xi;
    std::vector<COMPLEX> xr;
    std::vector<COMPLEX> work;
    std::vector<COMPLEX> ff;
    std::vector<COMPLEX> ff0;
    std::vector<COMPLEX> ff_local;
    std::vector<COMPLEX> ff_preconditioned;
    std::vector<COMPLEX> polarizability;
};

//-----------------------------------------------------------
void set_error(int* stop, char* info, size_t info_length, const char* message) {
    if (stop)
        *stop = 1;
    if (info && info_length) {
        std::memset(info, ' ', info_length);
        const size_t length = std::min(info_length, std::strlen(message));
        std::memcpy(info, message, length);
    }
}

//-----------------------------------------------------------
bool make_buffers(const std::complex<CU_REAL>* vectx,
                  const std::complex<CU_REAL>* vecty,
                  std::complex<CU_REAL>* ff_scalar,
                  std::complex<CU_REAL>* ff_local_scalar,
                  std::complex<CU_REAL>* xi, std::complex<CU_REAL>* xr,
                  std::complex<CU_REAL>* work, std::complex<CU_REAL>* ff,
                  const std::complex<CU_REAL>* ff0,
                  std::complex<CU_REAL>* ff_local,
                  std::complex<CU_REAL>* ff_preconditioned,
                  const std::complex<CU_REAL>* polarizability,
                  size_t grid_count, size_t field_count, size_t scalar_count,
                  size_t work_count, size_t polarization_count,
                  bool has_preconditioner_buffers, ScalarLegacyBuffers* buffers) {
    if (!vectx || !vecty || !ff_scalar || !ff_local_scalar || !xi || !xr ||
        !work || !ff || !ff0 || !ff_local || !polarizability || !buffers ||
        (has_preconditioner_buffers && !ff_preconditioned))
        return false;

    buffers->vectors_x.resize(grid_count);
    buffers->vectors_y.resize(grid_count);
    buffers->ff_scalar.resize(scalar_count);
    buffers->ff_local_scalar.resize(scalar_count);
    buffers->xi.resize(field_count);
    buffers->xr.resize(field_count);
    buffers->work.resize(work_count);
    buffers->ff.resize(field_count);
    buffers->ff0.resize(field_count);
    buffers->ff_local.resize(field_count);
    buffers->polarizability.resize(polarization_count);
    if (has_preconditioner_buffers)
        buffers->ff_preconditioned.resize(field_count);

    const size_t complex_bytes = sizeof(CUCOMPLEX);
    std::memcpy(buffers->vectors_x.data(), vectx, grid_count * complex_bytes);
    std::memcpy(buffers->vectors_y.data(), vecty, grid_count * complex_bytes);
    std::memcpy(buffers->ff_scalar.data(), ff_scalar, scalar_count * complex_bytes);
    std::memcpy(buffers->ff_local_scalar.data(), ff_local_scalar,
                scalar_count * complex_bytes);
    std::memcpy(buffers->xi.data(), xi, field_count * complex_bytes);
    std::memcpy(buffers->xr.data(), xr, field_count * complex_bytes);
    std::memcpy(buffers->work.data(), work, work_count * complex_bytes);
    std::memcpy(buffers->ff.data(), ff, field_count * complex_bytes);
    std::memcpy(buffers->ff0.data(), ff0, field_count * complex_bytes);
    std::memcpy(buffers->ff_local.data(), ff_local, field_count * complex_bytes);
    std::memcpy(buffers->polarizability.data(), polarizability,
                polarization_count * complex_bytes);
    if (has_preconditioner_buffers)
        std::memcpy(buffers->ff_preconditioned.data(), ff_preconditioned,
                    field_count * complex_bytes);
    return true;
}

//-----------------------------------------------------------
void publish_mutable_buffers(ScalarLegacyBuffers& buffers,
                             std::complex<CU_REAL>* ff_scalar,
                             std::complex<CU_REAL>* ff_local_scalar,
                             std::complex<CU_REAL>* xi, std::complex<CU_REAL>* xr,
                             std::complex<CU_REAL>* work, std::complex<CU_REAL>* ff,
                             std::complex<CU_REAL>* ff_local,
                             std::complex<CU_REAL>* ff_preconditioned,
                             size_t field_count, size_t scalar_count,
                             bool has_preconditioner_buffers) {
    const size_t complex_bytes = sizeof(CUCOMPLEX);
    std::memcpy(ff_scalar, buffers.ff_scalar.data(), scalar_count * complex_bytes);
    std::memcpy(ff_local_scalar, buffers.ff_local_scalar.data(), scalar_count * complex_bytes);
    std::memcpy(xi, buffers.xi.data(), field_count * complex_bytes);
    std::memcpy(xr, buffers.xr.data(), field_count * complex_bytes);
    std::memcpy(work, buffers.work.data(), buffers.work.size() * complex_bytes);
    std::memcpy(ff, buffers.ff.data(), field_count * complex_bytes);
    std::memcpy(ff_local, buffers.ff_local.data(), field_count * complex_bytes);
    if (has_preconditioner_buffers)
        std::memcpy(ff_preconditioned, buffers.ff_preconditioned.data(),
                    field_count * complex_bytes);
}

//-----------------------------------------------------------
void run_scalar_driver(
    const std::complex<CU_REAL>* vectx, std::complex<CU_REAL>* vecty,
    const int* tabdip, std::complex<CU_REAL>* ff_scalar,
    std::complex<CU_REAL>* ff_local_scalar,
    const std::complex<CU_REAL>* uincx, const std::complex<CU_REAL>* uincy,
    const std::complex<CU_REAL>* uincz, const int* ntotalm,
    const int* ntotal, const int* ldabi, const int* nlar, const int* nmax,
    const int* ndipole, const int* nxm, const int* nym, const int* nzm,
    const int* nx, const int* ny, const int* nz, const int* nx2,
    const int* ny2, const int* nxy2, const int* nz2, const int* nbsphere,
    std::complex<CU_REAL>* xi, std::complex<CU_REAL>* xr,
    std::complex<CU_REAL>* work, std::complex<CU_REAL>* ff,
    const std::complex<CU_REAL>* ff0, std::complex<CU_REAL>* ff_local,
    std::complex<CU_REAL>* ff_preconditioned,
    const std::complex<CU_REAL>* polarizability, const int* niso,
    char* methodeit, const CU_REAL* tol, CU_REAL* tol1, int* nloop,
    int* ncompte, const int* nlim, int preconditioner, const int* nrig,
    const CU_REAL* k0, const CU_REAL* aretecube,
    const long long* planf, const long long* planb, int* nstop,
    char* infostr, size_t methodeit_length, size_t infostr_length,
    bool has_preconditioner_buffers) {
    if (!uincx || !uincy || !uincz || !ntotalm || !ntotal || !ldabi || !nlar || !nmax || !ndipole ||
        !nxm || !nym || !nzm || !nx || !ny || !nz || !nx2 || !ny2 ||
        !nxy2 || !nz2 || !nbsphere || !niso || !nrig || !tol || !tol1 ||
        !nloop || !ncompte || !nlim || !methodeit || !nstop || !infostr ||
        !tabdip || *ntotal <= 0 || *nbsphere <= 0 || *nxm <= 0 ||
        *nym <= 0 || *nzm <= 0 || *nlar <= 0 || *niso <= 0) {
        set_error(nstop, infostr, infostr_length, "Invalid scalar inverse ABI arguments");
        return;
    }

    if (size_t(*nxm) > std::numeric_limits<size_t>::max() / size_t(*nym) ||
        size_t(*nxm) * size_t(*nym) > std::numeric_limits<size_t>::max() / size_t(*nzm)) {
        set_error(nstop, infostr, infostr_length, "Scalar inverse ABI mesh size overflow");
        return;
    }
    const size_t mesh_count = size_t(*nxm) * size_t(*nym) * size_t(*nzm);
    const size_t scalar_count = size_t(*nbsphere);
    if (mesh_count > std::numeric_limits<size_t>::max() / size_t(3) ||
        mesh_count > std::numeric_limits<size_t>::max() / size_t(*niso) ||
        mesh_count * size_t(*niso) > std::numeric_limits<size_t>::max() / size_t(*niso)) {
        set_error(nstop, infostr, infostr_length, "Scalar inverse ABI buffer size overflow");
        return;
    }
    const size_t field_count = size_t(3) * mesh_count;
    if (field_count > std::numeric_limits<size_t>::max() / size_t(*nlar)) {
        set_error(nstop, infostr, infostr_length, "Scalar inverse ABI workspace size overflow");
        return;
    }
    const size_t work_count = field_count * size_t(*nlar);
    const size_t polarization_count = mesh_count * size_t(*niso) * size_t(*niso);
    const size_t grid_count = size_t(*ntotal);
    if (field_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        polarization_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        grid_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX)) {
        set_error(nstop, infostr, infostr_length, "Scalar inverse ABI size overflow");
        return;
    }

    try {
        ScalarLegacyBuffers buffers;
        if (!make_buffers(vectx, vecty, ff_scalar, ff_local_scalar, xi, xr,
                          work, ff, ff0, ff_local, ff_preconditioned,
                          polarizability, grid_count, field_count, scalar_count,
                          work_count, polarization_count,
                          has_preconditioner_buffers, &buffers)) {
            set_error(nstop, infostr, infostr_length, "Missing scalar inverse ABI buffer");
            return;
        }

        const COMPLEX incident_x = *uincx;
        const COMPLEX incident_y = *uincy;
        const COMPLEX incident_z = *uincz;
        const int initial_guess = -1;
        const int optimized_product = 0;
        const CU_REAL zero = CU_REAL(0);
        inverserigscalaire_cuda_driver_(
            buffers.vectors_x.data(), buffers.vectors_y.data(), tabdip,
            buffers.ff_scalar.data(), buffers.ff_local_scalar.data(),
            &incident_x, &incident_y, &incident_z, ntotalm, ntotal, ldabi,
            nlar, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2,
            nxy2, nz2, nbsphere, buffers.xi.data(), buffers.xr.data(),
            buffers.work.data(), buffers.ff.data(), buffers.ff0.data(),
            buffers.ff_local.data(), buffers.polarizability.data(), niso,
            methodeit, tol, tol1, nloop, ncompte, nlim, &preconditioner,
            nrig, &initial_guess, &optimized_product, k0 ? k0 : &zero,
            aretecube ? aretecube : &zero, planf, planb, nstop, infostr,
            methodeit_length, infostr_length);
        publish_mutable_buffers(buffers, ff_scalar, ff_local_scalar, xi, xr,
                                work, ff, ff_local, ff_preconditioned,
                                field_count, scalar_count,
                                has_preconditioner_buffers);
    } catch (const std::bad_alloc&) {
        set_error(nstop, infostr, infostr_length, "Scalar inverse ABI staging allocation failed");
    }
}

} // namespace

//-----------------------------------------------------------
extern "C" void inverserigscalaire_(
    const std::complex<CU_REAL>* vectx, std::complex<CU_REAL>* vecty,
    const int* tabdip, std::complex<CU_REAL>* ff_scalar,
    std::complex<CU_REAL>* ff_local_scalar,
    const std::complex<CU_REAL>* uincx, const std::complex<CU_REAL>* uincy,
    const std::complex<CU_REAL>* uincz, const int* ntotalm,
    const int* ntotal, const int* ldabi, const int* nlar, const int* nmax,
    const int* ndipole, const int* nxm, const int* nym, const int* nzm,
    const int* nx, const int* ny, const int* nz, const int* nx2,
    const int* ny2, const int* nxy2, const int* nz2, const int* nbsphere,
    std::complex<CU_REAL>* xi, std::complex<CU_REAL>* xr,
    std::complex<CU_REAL>* work, std::complex<CU_REAL>* ff,
    const std::complex<CU_REAL>* ff0, std::complex<CU_REAL>* ff_local,
    const std::complex<CU_REAL>* polarizability, const int* niso,
    char* methodeit, const CU_REAL* tol, CU_REAL* tol1, int* nloop,
    int* ncompte, const int* nlim, const long long* planf,
    const long long* planb, int* nstop, char* infostr,
    size_t methodeit_length, size_t infostr_length) {
    const int preconditioner = 0;
    const int nrig = 0;
    run_scalar_driver(vectx, vecty, tabdip, ff_scalar, ff_local_scalar,
                      uincx, uincy, uincz, ntotalm, ntotal, ldabi, nlar,
                      nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2,
                      nxy2, nz2, nbsphere, xi, xr, work, ff, ff0, ff_local,
                      nullptr, polarizability, niso, methodeit, tol, tol1,
                      nloop, ncompte, nlim, preconditioner, &nrig, nullptr,
                      nullptr, planf, planb, nstop, infostr, methodeit_length,
                      infostr_length, false);
}

//-----------------------------------------------------------
extern "C" void inverserigscalairepg_(
    std::complex<CU_REAL>* vectx, std::complex<CU_REAL>* vecty,
    const int* tabdip, std::complex<CU_REAL>* ff_scalar,
    std::complex<CU_REAL>* ff_local_scalar,
    const std::complex<CU_REAL>* uincx, const std::complex<CU_REAL>* uincy,
    const std::complex<CU_REAL>* uincz, const int* ntotalm,
    const int* ntotal, const int* ldabi, const int* nlar, const int* nmax,
    const int* ndipole, const int* nxm, const int* nym, const int* nzm,
    const int* nx, const int* ny, const int* nz, const int* nx2,
    const int* ny2, const int* nxy2, const int* nz2, const int* nbsphere,
    std::complex<CU_REAL>* xi, std::complex<CU_REAL>* xr,
    std::complex<CU_REAL>* work, std::complex<CU_REAL>* ff,
    const std::complex<CU_REAL>* ff0, std::complex<CU_REAL>* ff_local,
    std::complex<CU_REAL>* ff_preconditioned,
    const std::complex<CU_REAL>* polarizability, const int* niso,
    char* methodeit, const CU_REAL* tol, CU_REAL* tol1, int* nloop,
    int* ncompte, const int* nlim, const CU_REAL* k0,
    const CU_REAL* aretecube, std::complex<CU_REAL>* chanxxvect,
    std::complex<CU_REAL>* sdetnn, const long long* planf,
    const long long* planb, const int* nrig, int* nstop, char* infostr,
    size_t methodeit_length, size_t infostr_length) {
    (void)chanxxvect;
    (void)sdetnn;
    const int preconditioner = 1;
    run_scalar_driver(vectx, vecty, tabdip, ff_scalar, ff_local_scalar,
                      uincx, uincy, uincz, ntotalm, ntotal, ldabi, nlar,
                      nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2,
                      nxy2, nz2, nbsphere, xi, xr, work, ff, ff0, ff_local,
                      ff_preconditioned, polarizability, niso, methodeit,
                      tol, tol1, nloop, ncompte, nlim, preconditioner, nrig,
                      k0, aretecube, planf, planb, nstop, infostr,
                      methodeit_length, infostr_length, true);
}

//-----------------------------------------------------------
extern "C" void inverserigscalairepd_(
    std::complex<CU_REAL>* vectx, std::complex<CU_REAL>* vecty,
    const int* tabdip, std::complex<CU_REAL>* ff_scalar,
    std::complex<CU_REAL>* ff_local_scalar,
    const std::complex<CU_REAL>* uincx, const std::complex<CU_REAL>* uincy,
    const std::complex<CU_REAL>* uincz, const int* ntotalm,
    const int* ntotal, const int* ldabi, const int* nlar, const int* nmax,
    const int* ndipole, const int* nxm, const int* nym, const int* nzm,
    const int* nx, const int* ny, const int* nz, const int* nx2,
    const int* ny2, const int* nxy2, const int* nz2, const int* nbsphere,
    std::complex<CU_REAL>* xi, std::complex<CU_REAL>* xr,
    std::complex<CU_REAL>* work, std::complex<CU_REAL>* ff,
    const std::complex<CU_REAL>* ff0, std::complex<CU_REAL>* ff_local,
    std::complex<CU_REAL>* ff_preconditioned,
    const std::complex<CU_REAL>* polarizability, const int* niso,
    char* methodeit, const CU_REAL* tol, CU_REAL* tol1, int* nloop,
    int* ncompte, const int* nlim, const CU_REAL* k0,
    const CU_REAL* aretecube, std::complex<CU_REAL>* chanxxvect,
    std::complex<CU_REAL>* sdetnn, const long long* planf,
    const long long* planb, const int* nrig, int* nstop, char* infostr,
    size_t methodeit_length, size_t infostr_length) {
    (void)chanxxvect;
    (void)sdetnn;
    const int preconditioner = 2;
    run_scalar_driver(vectx, vecty, tabdip, ff_scalar, ff_local_scalar,
                      uincx, uincy, uincz, ntotalm, ntotal, ldabi, nlar,
                      nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2,
                      nxy2, nz2, nbsphere, xi, xr, work, ff, ff0, ff_local,
                      ff_preconditioned, polarizability, niso, methodeit,
                      tol, tol1, nloop, ncompte, nlim, preconditioner, nrig,
                      k0, aretecube, planf, planb, nstop, infostr,
                      methodeit_length, infostr_length, true);
}
