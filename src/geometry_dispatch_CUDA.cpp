// Common geometry dispatcher used by BOTH CUDA orchestrators:
 // CUDA/cdmlib.cpp and CUDA/cdmlib.f90. All geometry branches call the
 // ifdda_object_all_* GPU engines directly, without legacy objet*.cpp adapters.
#include "cdmlib_cpp_precision.hpp"
#include "cuda_interfaces.h"
#include "cuda_data.h"
#define IFDDA_CDMLIB_PRECISION_C99_H
extern "C" {
#include "cdmlib_config.h"
}
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstddef>

static bool geometry_starts(const char* text, const char* prefix, size_t count) {
    return text && std::memcmp(text, prefix, count) == 0;
}
static void geometry_message(char* info, const char* message) {
    if (!info) return;
    std::memset(info, ' ', 64);
    if (message) std::memcpy(info, message, std::min(size_t(64), std::strlen(message)));
}
static int geometry_fail(int code, const char* message, int* stop, char* info) {
    if (stop) *stop = code;
    geometry_message(info, message);
    return 1;
}
static void geometry_error(char* info, int* stop, const char* message) {
    geometry_fail(1, message, stop, info);
}
cdmlib_object_kind_c99 cdmlib_classify_object_c99(const char* object) {
    // Preserve the historical first-match order exactly.
    if (geometry_starts(object, "sphere", 6)) return CDMLIB_OBJECT_SPHERE;
    if (geometry_starts(object, "inhomosphere", 12)) return CDMLIB_OBJECT_INHOMO_SPHERE;
    if (geometry_starts(object, "inhomocuboid1", 13)) return CDMLIB_OBJECT_INHOMO_CUBOID1;
    if (geometry_starts(object, "inhomocuboid2", 13)) return CDMLIB_OBJECT_INHOMO_CUBOID2;
    if (geometry_starts(object, "randomsphere1", 13)) return CDMLIB_OBJECT_RANDOM_SPHERE1;
    if (geometry_starts(object, "randomsphere2", 13)) return CDMLIB_OBJECT_RANDOM_SPHERE2;
    if (geometry_starts(object, "randomspheremulti1", 18)) return CDMLIB_OBJECT_RANDOM_SPHERE_MULTI1;
    if (geometry_starts(object, "cube", 4)) return CDMLIB_OBJECT_CUBE;
    if (geometry_starts(object, "cuboid1", 7)) return CDMLIB_OBJECT_CUBOID1;
    if (geometry_starts(object, "cuboid2", 7)) return CDMLIB_OBJECT_CUBOID2;
    if (geometry_starts(object, "ellipsoid", 9)) return CDMLIB_OBJECT_ELLIPSOID;
    if (geometry_starts(object, "nspheres", 8)) return CDMLIB_OBJECT_NSPHERES;
    if (geometry_starts(object, "cylinder", 8)) return CDMLIB_OBJECT_CYLINDER;
    if (geometry_starts(object, "concentricsphere", 16)) return CDMLIB_OBJECT_CONCENTRIC_SPHERE;
    if (geometry_starts(object, "arbitrary", 9)) return CDMLIB_OBJECT_ARBITRARY;
    return CDMLIB_OBJECT_UNKNOWN;
}

int cdmlib_call_object_geometry_c99(const cdmlib_object_phase_c99 *a)
{
    // Invalidate the previous host-grid descriptor before dispatching any
    // shape. Irregular/PML geometries must never reuse a stale lattice.
    cuda_data::clear_host_regular_geometry();
    cdmlib_object_kind_c99 kind=cdmlib_classify_object_c99(a->object64);
    if(kind!=CDMLIB_OBJECT_UNKNOWN && kind!=CDMLIB_OBJECT_NSPHERES &&
       kind!=CDMLIB_OBJECT_CONCENTRIC_SPHERE &&
       kind!=CDMLIB_OBJECT_RANDOM_SPHERE_MULTI1)
        *a->numberobjet=1;
    
    switch(kind) {
    case CDMLIB_OBJECT_CONCENTRIC_SPHERE:
        ifdda_object_all_concentricsphere_cuda(0, a->trope3,a->epsmulti,a->epsanimulti,
            a->numberobjet,a->numberobjet,a->xg,a->yg,a->zg,
            a->rayonmulti,a->eps0,a->xs,a->ys,a->zs,a->k0,a->aretecube,
            a->tabdip,a->tabmulti,a->nnnr,a->nmax,a->nbsphere,a->ndipole,
            a->nx,a->ny,a->nz,a->polarizability2,a->nproche,a->epsilon,
            a->polarisa,a->niso,a->sldr,a->nmat,a->file_id,a->group_iddip,
            a->infostr,a->nstop,3,2,64);
        return 1;
    case CDMLIB_OBJECT_INHOMO_SPHERE:
        if(!geometry_starts(a->trope3,"iso",3))
            return geometry_fail(1,"Permittivity not scalar for inhomogenous sphere",
                        a->nstop,a->infostr);
        ifdda_object_all_inhomosphere_cuda(5, (COMPLEX *)a->eps,a->eps0,a->lnrbc,a->lcos,
            a->ntypeabc,a->xs,a->ys,a->zs,a->k0,a->aretecube,a->tabdip,
            a->tabmulti,a->pmlwrk,a->nnnr,a->nmax,a->nbsphere,a->ndipole,
            a->nx,a->ny,a->nz,a->polarizability2,a->nproche,a->epsilon,
            a->polarisa,a->niso,a->radius,a->lc,a->hc,a->ng,a->macroscopicfieldx,
            a->sldr,a->nmat,a->file_id,a->group_iddip,a->infostr,a->nstop,2,64);
        if(a->macroscopicfieldx)
            std::fill_n(a->macroscopicfieldx,(size_t)*a->nmax,COMPLEX{});
        return 1;
    case CDMLIB_OBJECT_CUBE:
        ifdda_object_all_cube_cuda(2, a->trope3,a->eps,a->epsani,a->eps0,a->xs,a->ys,a->zs,
            a->k0,a->aretecube,a->tabdip,a->nnnr,a->nmax,a->nbsphere,
            a->ndipole,a->nx,a->ny,a->nz,a->polarizability2,a->epsilon,
            a->polarisa,a->niso,a->side,a->xg,a->yg,a->zg,a->sldr,a->nmat,
            a->file_id,a->group_iddip,a->infostr,a->nstop,3,2,64);
        return 1;
    case CDMLIB_OBJECT_SPHERE:
        ifdda_object_all_sphere_cuda(1, a->trope3,a->eps,a->epsani,a->eps0,a->lnrbc,a->lcos,
            a->ntypeabc,a->xs,a->ys,a->zs,a->k0,a->aretecube,a->tabdip,
            a->tabmulti,a->pmlwrk,a->nnnr,a->nmax,a->nbsphere,a->ndipole,
            a->nx,a->ny,a->nz,a->polarizability2,a->nproche,a->epsilon,
            a->polarisa,a->niso,a->radius,a->xg,a->yg,a->zg,a->sldr,
            a->nmat,a->file_id,a->group_iddip,a->infostr,a->nstop,3,2,64);
        return 1;
    case CDMLIB_OBJECT_CUBOID1:
        ifdda_object_all_cuboid1_cuda(3, a->trope3,a->eps,a->epsani,a->eps0,a->xs,a->ys,a->zs,
            a->k0,a->aretecube,a->tabdip,a->nnnr,a->nmax,a->nbsphere,
            a->ndipole,a->nx,a->ny,a->nz,a->nxm,a->nym,a->nzm,
            a->polarizability2,a->epsilon,a->polarisa,a->niso,a->sidex,
            a->sidey,a->sidez,a->xg,a->yg,a->zg,a->phiobj,a->thetaobj,
            a->psiobj,a->nproche,a->sldr,a->nmat,a->file_id,a->group_iddip,
            a->infostr,a->nstop,3,2,64);
        return 1;
    case CDMLIB_OBJECT_CUBOID2:
        ifdda_object_all_cuboid2_cuda(4, a->trope3,(COMPLEX *)a->eps,
            (COMPLEX *)a->epsani,(COMPLEX *)a->eps0,a->xs,a->ys,
            a->zs,a->k0,a->aretecube,a->tabdip,a->nnnr,a->nmax,a->nbsphere,
            a->ndipole,a->nx,a->ny,a->nz,a->nxm,a->nym,a->nzm,a->nxmp,
            a->nymp,a->nzmp,a->polarizability2,a->epsilon,a->polarisa,a->niso,
            a->sidex,a->sidey,a->sidez,a->xg,a->yg,a->zg,a->nproche,a->sldr,
            a->nmat,a->file_id,a->group_iddip,a->infostr,a->nstop,3,2,64);
        return 1;
    case CDMLIB_OBJECT_ELLIPSOID:
        ifdda_object_all_ellipsoid_cuda(8, a->trope3,a->eps,a->epsani,a->eps0,a->xs,a->ys,a->zs,
            a->k0,a->aretecube,a->tabdip,a->nnnr,a->nmax,a->nbsphere,a->ndipole,
            a->nx,a->ny,a->nz,a->nxm,a->nym,a->nzm,a->polarizability2,
            a->nproche,a->epsilon,a->polarisa,a->niso,a->axis_a,a->axis_b,
            a->axis_c,a->xg,a->yg,a->zg,a->phiobj,a->thetaobj,a->psiobj,
            a->sldr,a->nmat,a->file_id,a->group_iddip,a->infostr,a->nstop,3,2,64);
        return 1;
    case CDMLIB_OBJECT_CYLINDER:
        ifdda_object_all_cylinder_cuda(10, a->trope3,a->eps,a->epsani,a->eps0,a->xs,a->ys,a->zs,
            a->k0,a->aretecube,a->tabdip,a->nnnr,a->nmax,a->nbsphere,a->ndipole,
            a->nx,a->ny,a->nz,a->nxm,a->nym,a->nzm,a->polarizability2,
            a->nproche,a->epsilon,a->polarisa,a->niso,a->radius,a->height,
            a->xg,a->yg,a->zg,a->phiobj,a->thetaobj,a->psiobj,a->sldr,a->nmat,
            a->file_id,a->group_iddip,a->infostr,a->nstop,3,2,64);
        return 1;
    case CDMLIB_OBJECT_NSPHERES:
        ifdda_object_all_nspheres_cuda(9, a->trope3,a->epsmulti,a->epsanimulti,a->numberobjet,
            a->numberobjet,a->xgmulti,a->ygmulti,a->zgmulti,a->rayonmulti,
            a->eps0,a->xs,a->ys,a->zs,a->k0,a->aretecube,a->tabdip,a->tabmulti,
            a->nnnr,a->nmax,a->nbsphere,a->ndipole,a->nx,a->ny,a->nz,
            a->polarizability2,a->nproche,a->epsilon,a->polarisa,a->niso,a->sldr,
            a->nmat,a->file_id,a->group_iddip,a->infostr,a->nstop,3,2,64);
        return 1;
    case CDMLIB_OBJECT_RANDOM_SPHERE1:
        ifdda_object_all_randomsphere_cuda(11, a->trope3,a->eps,a->epsani,a->eps0,a->xs,a->ys,a->zs,
            a->k0,a->aretecube,a->tabdip,a->nnnr,a->nmax,a->nbsphere,a->ndipole,
            a->nx,a->ny,a->nz,a->nxm,a->nym,a->nzm,a->polarizability2,
            a->epsilon,a->polarisa,a->niso,a->sidex,a->sidey,a->sidez,a->xg,
            a->yg,a->zg,a->xswf,a->yswf,a->zswf,a->radius,a->density,a->ng,
            a->nproche,a->sldr,a->nmat,a->file_id,a->group_iddip,a->infostr,
            a->nstop,3,2,64);
        return 1;
    case CDMLIB_OBJECT_RANDOM_SPHERE2: {
        if (!a->nxm || !a->nym || !a->nzm ||
            !a->nxmp || !a->nymp || !a->nzmp ||
            !a->sidex || !a->sidey || !a->sidez || !a->aretecube) {
            geometry_error(a->infostr, a->nstop, "Invalid randomsphere2 geometry arguments");
            return 1;
        }
        const int nx_interior = *a->nxm - 2 * *a->nxmp;
        const int ny_interior = *a->nym - 2 * *a->nymp;
        const int nz_interior = *a->nzm - 2 * *a->nzmp;
        if (nx_interior <= 0 || ny_interior <= 0 || nz_interior <= 0) {
            geometry_error(a->infostr, a->nstop,
                           "Dimension Problem of the Box : Box too small!");
            return 1;
        }
        const int actual_subdivisions = std::max({nx_interior, ny_interior, nz_interior});
        const CU_REAL cell_nm = *a->aretecube;
        *a->sidex = cell_nm * CU_REAL(nx_interior);
        *a->sidey = cell_nm * CU_REAL(ny_interior);
        *a->sidez = cell_nm * CU_REAL(nz_interior);
        *a->aretecube = cell_nm;
        ifdda_object_all_randomsphere_cuda(12, a->trope3,a->eps,a->epsani,a->eps0,a->xs,a->ys,
            a->zs,a->k0,a->aretecube,a->tabdip,&actual_subdivisions,a->nmax,a->nbsphere,
            a->ndipole,a->nx,a->ny,a->nz,a->nxm,a->nym,a->nzm,
            a->polarizability2,a->epsilon,a->polarisa,a->niso,a->sidex,
            a->sidey,a->sidez,a->xg,a->yg,a->zg,a->xswf,a->yswf,a->zswf,
            a->radius,a->density,a->ng,a->nproche,a->sldr,a->nmat,a->file_id,
            a->group_iddip,a->infostr,a->nstop,3,2,64);
        return 1;
    }
    case CDMLIB_OBJECT_RANDOM_SPHERE_MULTI1:
        ifdda_object_all_randommulti_cuda(14, a->trope3,(COMPLEX *)a->epsmulti,
            (COMPLEX *)a->epsanimulti,a->numberobjet,a->numberobjet,
            a->xgmulti,a->ygmulti,a->zgmulti,a->rayonmulti,(COMPLEX *)a->eps0,
            a->xs,a->ys,a->zs,a->k0,a->aretecube,a->tabdip,a->nnnr,a->nmax,
            a->nbsphere,a->ndipole,a->nx,a->ny,a->nz,a->nxm,a->nym,a->nzm,
            a->polarizability2,a->epsilon,a->polarisa,a->niso,a->sidex,a->sidey,
            a->sidez,a->xswf,a->yswf,a->zswf,a->density,a->ng,a->nproche,a->sldr,
            a->nmat,a->file_id,a->group_iddip,a->infostr,a->nstop,3,2,64);
        return 1;
    case CDMLIB_OBJECT_INHOMO_CUBOID2:
        if(!geometry_starts(a->trope3,"iso",3))
            return geometry_fail(1,"Permittivity not scalar for inhomogenous cuboid",
                        a->nstop,a->infostr);
        ifdda_object_all_inhomocuboid2_cuda(7, (COMPLEX *)a->eps,
            (COMPLEX *)a->eps0,a->xs,a->ys,a->zs,a->k0,a->aretecube,
            a->tabdip,a->nnnr,a->nmax,a->nbsphere,a->ndipole,a->nx,a->ny,a->nz,
            a->nxm,a->nym,a->nzm,a->nxmp,a->nymp,a->nzmp,a->polarizability2,
            a->epsilon,a->polarisa,a->niso,a->sidex,a->sidey,a->sidez,a->xg,
            a->yg,a->zg,a->lc,a->hc,a->ng,a->macroscopicfieldx,
            a->nproche,a->sldr,a->nmat,a->file_id,a->group_iddip,a->infostr,
            a->nstop,2,64);
        if(a->macroscopicfieldx)
            std::fill_n(a->macroscopicfieldx,(size_t)*a->nmax,COMPLEX{});
        return 1;
    case CDMLIB_OBJECT_INHOMO_CUBOID1:
        if(!geometry_starts(a->trope3,"iso",3))
            return geometry_fail(1,"Permittivity not scalar for inhomogenous cuboid",
                        a->nstop,a->infostr);
        ifdda_object_all_inhomocuboid1_cuda(6, (COMPLEX *)a->eps,(COMPLEX *)a->eps0,
            a->lnrbc,a->lcos,a->ntypeabc,a->xs,a->ys,a->zs,a->k0,a->aretecube,
            a->tabdip,a->tabmulti,a->pmlwrk,a->nnnr,a->nmax,a->nbsphere,
            a->ndipole,a->nx,a->ny,a->nz,a->nxm,a->nym,a->nzm,
            a->polarizability2,a->epsilon,a->polarisa,a->niso,a->sidex,a->sidey,
            a->sidez,a->xg,a->yg,a->zg,a->lc,a->hc,a->ng,a->macroscopicfieldx,
            a->nproche,a->sldr,a->nmat,a->file_id,a->group_iddip,a->infostr,
            a->nstop,2,64);
        if(a->macroscopicfieldx)
            std::fill_n(a->macroscopicfieldx,(size_t)*a->nmax,COMPLEX{});
        return 1;
    case CDMLIB_OBJECT_ARBITRARY:
        ifdda_object_all_arbitrary_cuda(13, a->trope3,a->eps,
            a->epsani,a->eps0,a->xs,a->ys,a->zs,a->k0,
            a->aretecube,a->tabdip,a->nnnr,a->nmax,a->nbsphere,a->ndipole,
            a->nx,a->ny,a->nz,a->polarizability2,a->namefileobj,a->nproche,
            a->epsilon,a->polarisa,a->niso,a->sldr,a->nmat,a->file_id,
            a->group_iddip,a->infostr,a->nstop,3,2,64,64);
        if(a->lambda && *a->aretecube / *a->lambda <= 0.01) {
            geometry_message(a->infostr,"Meshsize too small");
            *a->nstop=1;
        }
        return 1;
    default:
        return 0;
    }
}

// Legacy GNU Fortran by-reference ABI. All CHARACTER hidden lengths follow
// the explicit arguments. This callback passes the original caller buffers:
// no intermediate CPU/GPU arrays are allocated by the dispatcher.
extern "C" void ifdda_geometry_dispatch_cuda_(
    const char* object64, const char* trope3, const char* polarizability2,
    const char* namefileobj, char* infostr,
    COMPLEX* eps, COMPLEX* epsani, const COMPLEX* epsmulti,
    const COMPLEX* epsanimulti, int* numberobjet,
    REAL* xgmulti, REAL* ygmulti, REAL* zgmulti, REAL* rayonmulti,
    REAL* xswf, REAL* yswf, REAL* zswf, REAL* density,
    REAL* xg, REAL* yg, REAL* zg,
    REAL* side, REAL* sidex, REAL* sidey, REAL* sidez, REAL* height,
    REAL* radius, REAL* axis_a, REAL* axis_b, REAL* axis_c,
    REAL* lnrbc, REAL* lcos, REAL* lc, REAL* hc,
    REAL* phiobj, REAL* thetaobj, REAL* psiobj,
    int* ntypeabc, int* pmlwrk, int* ng,
    const COMPLEX* eps0, REAL* xs, REAL* ys, REAL* zs,
    const REAL* k0, const REAL* lambda, REAL* aretecube,
    int* tabdip, int* tabmulti, const int* nnnr, const int* nmax,
    const int* nxm, const int* nym, const int* nzm,
    const int* nxmp, const int* nymp, const int* nzmp,
    int* nbsphere, int* ndipole, int* nx, int* ny, int* nz,
    int* nproche, const int* niso, COMPLEX* epsilon, COMPLEX* polarisa,
    COMPLEX* macroscopicfieldx, REAL* sldr, const int* nmat,
    const int64_t* file_id, const int64_t* group_iddip, int* nstop,
    size_t object_length, size_t trope_length, size_t polarizability_length,
    size_t namefileobj_length, size_t infostr_length) {
    if (!object64 || !trope3 || !polarizability2 ||
        !namefileobj || !infostr || !nstop ||
        object_length < 64 || trope_length < 3 || polarizability_length < 2 ||
        namefileobj_length < 64 || infostr_length < 64) {
        geometry_error(infostr, nstop, "Invalid Fortran geometry ABI strings");
        return;
    }
    cdmlib_object_phase_c99 a{};
    a.object64 = object64; a.trope3 = trope3;
    a.polarizability2 = polarizability2;
    a.namefileobj = namefileobj; a.infostr = infostr;
    a.eps = eps; a.epsani = epsani;
    a.epsmulti = epsmulti; a.epsanimulti = epsanimulti;
    a.numberobjet = numberobjet;
    a.xgmulti = xgmulti; a.ygmulti = ygmulti;
    a.zgmulti = zgmulti; a.rayonmulti = rayonmulti;
    a.xswf = xswf; a.yswf = yswf; a.zswf = zswf; a.density = density;
    a.xg = xg; a.yg = yg; a.zg = zg;
    a.side = side; a.sidex = sidex; a.sidey = sidey;
    a.sidez = sidez; a.height = height; a.radius = radius;
    a.axis_a = axis_a; a.axis_b = axis_b; a.axis_c = axis_c;
    a.lnrbc = lnrbc; a.lcos = lcos; a.lc = lc; a.hc = hc;
    a.phiobj = phiobj; a.thetaobj = thetaobj; a.psiobj = psiobj;
    a.ntypeabc = ntypeabc; a.pmlwrk = pmlwrk; a.ng = ng;
    a.eps0 = eps0; a.xs = xs; a.ys = ys; a.zs = zs;
    a.k0 = k0; a.lambda = lambda; a.aretecube = aretecube;
    a.tabdip = tabdip; a.tabmulti = tabmulti;
    a.nnnr = nnnr; a.nmax = nmax;
    a.nxm = nxm; a.nym = nym; a.nzm = nzm;
    a.nxmp = nxmp; a.nymp = nymp; a.nzmp = nzmp;
    a.nbsphere = nbsphere; a.ndipole = ndipole;
    a.nx = nx; a.ny = ny; a.nz = nz; a.nproche = nproche;
    a.niso = niso; a.epsilon = epsilon; a.polarisa = polarisa;
    a.macroscopicfieldx = macroscopicfieldx; a.sldr = sldr;
    a.nmat = nmat; a.file_id = file_id; a.group_iddip = group_iddip;
    a.nstop = nstop;
    if (!cdmlib_call_object_geometry_c99(&a)) {
        geometry_error(infostr, nstop, "Object unknown!");
    } else if (*nstop == 0 && nx && ny && nz && nmax && nbsphere &&
               tabdip && aretecube && *nmax > 0 && *nbsphere > 0 &&
               *nx > 0 && *ny > 0 && *nz > 0 &&
               *nx <= *nmax / *ny && *nx * *ny <= *nmax / *nz) {
        // Other regular geometries may also use the same tabdip/grid
        // illumination kernels. Never assume their coordinates are affine:
        // incident_field_CUDA.cpp checks EVERY active coordinate and the
        // one-based compact ordering, and falls back on a failed check.
        cuda_data::HostRegularGeometry existing{};
        if (!cuda_data::get_host_regular_geometry(
                xs, ys, zs, *nbsphere, &existing)) {
            // Guarantee that the grid kernel sees the current CPU map, even
            // for geometry-specific builders that did not upload it yet.
            // The central getter skips H2D if the CPU values are unchanged.
            if (get_tapdip(tabdip, size_t(*nmax)))
                (void)cuda_data::register_host_regular_geometry({
                    xs, ys, zs, tabdip, *nx, *ny, *nz,
                    *nbsphere, *aretecube});
        }
    }
}
