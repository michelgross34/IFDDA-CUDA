#include "cdmlib_cpp_precision.hpp"
#include <vector>
#define IFDDA_CDMLIB_PRECISION_C99_H
extern "C" {
#include "cdmlib_param.h"
}

extern "C" int cdmlib_prepare_geometry_params_c99(
    const cdmlib_abi_args_c99 *a,
    cdmlib_initphase_c99 *s,
    cdmlib_entry_runtime_c99 *r,
    REAL *k0,
    cdmlib_geometry_params_c99 *geometry)
{
    if (!a || !s || !r || !k0 || !geometry) return 1;
    // All fields are borrowed addresses. The views remain valid until the
    // geometry call returns; k0 is a local owned by that call.
    *geometry = cdmlib_geometry_params_c99{};
    cdmlib_object_phase_c99 *g = &geometry->object;
    g->object64=s->object; g->trope3=s->trope;
    g->polarizability2=s->polarizability;
    g->eps=&s->eps; g->epsani=s->epsani;
    g->epsmulti=(const COMPLEX *)a->epsmulti;
    g->epsanimulti=(const COMPLEX *)a->epsanimulti;
    g->numberobjet=(int *)a->numberobjet;
    g->xgmulti=(REAL *)a->xgmulti; g->ygmulti=(REAL *)a->ygmulti;
    g->zgmulti=(REAL *)a->zgmulti; g->rayonmulti=(REAL *)a->rayonmulti;
    g->xswf=a->xswf; g->yswf=a->yswf; g->zswf=a->zswf; g->density=a->density;
    g->xg=&s->xg; g->yg=&s->yg; g->zg=&s->zg;
    g->side=a->side; g->sidex=a->sidex; g->sidey=a->sidey; g->sidez=a->sidez;
    g->height=a->hauteur; g->radius=&s->rayon;
    g->axis_a=a->demiaxea; g->axis_b=a->demiaxeb; g->axis_c=a->demiaxec;
    g->lnrbc=a->lnrbc; g->lcos=a->lcos; g->lc=a->lc; g->hc=a->hc;
    g->phiobj=a->phiobj; g->thetaobj=a->thetaobj; g->psiobj=a->psiobj;
    g->ntypeabc=(int *)a->ntypeabc; g->pmlwrk=r->pmlwrk; g->ng=(int *)a->ng;
    g->eps0=&s->eps0; g->xs=a->xs; g->ys=a->ys; g->zs=a->zs;
    g->k0=k0; g->lambda=&s->lambda; g->aretecube=a->aretecube;
    g->tabdip=(int *)a->tabdip; g->tabmulti=(int *)a->tabmulti;
    g->nnnr=(const int *)a->nnnr; g->nmax=&s->nmax;
    g->nxm=(const int *)a->nxm; g->nym=(const int *)a->nym; g->nzm=(const int *)a->nzm;
    g->nxmp=(const int *)a->nxmp; g->nymp=(const int *)a->nymp; g->nzmp=(const int *)a->nzmp;
    g->nbsphere=(int *)a->nbsphere; g->ndipole=(int *)a->ndipole;
    g->nx=&r->nx; g->ny=&r->ny; g->nz=&r->nz; g->nproche=&s->nproche;
    g->niso=(const int *)a->niso; g->epsilon=(COMPLEX *)a->epsilon;
    g->polarisa=(COMPLEX *)a->polarisa;
    g->macroscopicfieldx=(COMPLEX *)a->macroscopicfieldx;
    g->epsb=r->epsb; g->sldr=&r->sldr;
    g->nmat=(const int *)a->nmat;
    g->file_id=&r->file_id; g->group_iddip=&r->group_iddip;
    g->infostr=s->infostr; g->nstop=&s->nstop;
    g->namefileobj=a->namefileobj;
    return 0;
}

extern "C" int cdmlib_prepare_illumination_params_c99(
    const cdmlib_abi_args_c99 *a,
    cdmlib_initphase_c99 *init,
    cdmlib_config_c99 *config,
    cdmlib_entry_runtime_c99 *runtime,
    std::vector<REAL> *pml_mask,
    cdmlib_illumination_params_c99 *illumination)
{
    if (!a || !init || !config || !runtime || !pml_mask || !illumination ||
        !a->xs || !a->ys || !a->zs || !a->k0 || !a->aretecube ||
        !a->ff0 || !a->tabmulti || !a->nbsphere || !a->ndipole ||
        !a->nlecture || !a->nstop || !a->infostr || !a->beam)
        return 1;

    *illumination = cdmlib_illumination_params_c99{};
    illumination->abi = a;
    illumination->init = init;
    illumination->config = config;
    illumination->runtime = runtime;
    illumination->xs = a->xs;
    illumination->ys = a->ys;
    illumination->zs = a->zs;
    illumination->k0 = a->k0;
    illumination->aretecube = a->aretecube;
    illumination->ff0 = (COMPLEX *)a->ff0;
    illumination->tabmulti = (const int *)a->tabmulti;
    illumination->nbsphere = a->nbsphere;
    illumination->ndipole = a->ndipole;
    illumination->nlecture = a->nlecture;
    illumination->nstop = a->nstop;
    illumination->infostr = a->infostr;
    illumination->infostr_len = a->infostr_len;
    illumination->beam = a->beam;
    illumination->pml_mask = pml_mask;
    return 0;
}

extern "C" int cdmlib_prepare_green_params_c99(
    const cdmlib_abi_args_c99 *a,
    const cdmlib_initphase_c99 *init,
    const cdmlib_config_c99 *config,
    cdmlib_entry_runtime_c99 *runtime,
    cdmlib_green_params_c99 *green)
{
    if (!a || !init || !config || !runtime || !green ||
        !a->ffttensorxx || !a->ffttensorxy || !a->ffttensorxz ||
        !a->ffttensoryy || !a->ffttensoryz || !a->ffttensorzz ||
        !a->k0 || !a->aretecube || !a->theta || !a->phi ||
        !a->nquad || !a->ntypeabc || !a->nstop || !a->infostr)
        return 1;

    *green = cdmlib_green_params_c99{};
    green->abi = a;
    green->init = init;
    green->config = config;
    green->runtime = runtime;
    green->xx = (COMPLEX *)a->ffttensorxx;
    green->xy = (COMPLEX *)a->ffttensorxy;
    green->xz = (COMPLEX *)a->ffttensorxz;
    green->yy = (COMPLEX *)a->ffttensoryy;
    green->yz = (COMPLEX *)a->ffttensoryz;
    green->zz = (COMPLEX *)a->ffttensorzz;
    green->k0 = (const REAL *)a->k0;
    green->aretecube = (const REAL *)a->aretecube;
    green->theta = (const REAL *)a->theta;
    green->phi = (const REAL *)a->phi;
    green->nquad = (const int *)a->nquad;
    green->ntypeabc = (const int *)a->ntypeabc;
    green->eps0 = &init->eps0;
    green->polarizability = init->polarizability;
    green->nstop = a->nstop;
    green->infostr = a->infostr;
    green->infostr_len = a->infostr_len;
    return 0;
}

extern "C" int cdmlib_prepare_green_scalar_params_c99(
    const cdmlib_abi_args_c99 *a,
    const cdmlib_initphase_c99 *init,
    cdmlib_entry_runtime_c99 *runtime,
    int mode,
    int polarization_prepared,
    COMPLEX *ux, COMPLEX *uy, COMPLEX *uz,
    int *ready,
    cdmlib_green_scalar_params_c99 *green)
{
    if (!a || !init || !runtime || !green || !ux || !uy || !uz || !ready ||
        (mode != 8 && mode != 9) ||
        (polarization_prepared < 0 || polarization_prepared > 2) ||
        !a->aretecube || !a->k0 || !a->nstop || !a->infostr ||
        !a->beam || !a->trope || !a->nprecon || !a->ninitest ||
        !a->ss || !a->pp || !a->theta || !a->phi || !a->w0 ||
        !a->xgaus || !a->ygaus || !a->zgaus || !a->e0 ||
        (mode == 8 && !runtime->vectx) ||
        (mode == 9 && !runtime->vecty))
        return 1;

    *green = cdmlib_green_scalar_params_c99{};
    green->mode = mode;
    green->polarization_prepared = polarization_prepared;
    green->beam = a->beam;
    green->trope = a->trope;
    green->nprecon = a->nprecon;
    green->ninitest = a->ninitest;
    green->ss = (const REAL *)a->ss;
    green->pp = (const REAL *)a->pp;
    green->theta = (const REAL *)a->theta;
    green->phi = (const REAL *)a->phi;
    green->w0 = (const REAL *)a->w0;
    green->xgaus = (const REAL *)a->xgaus;
    green->ygaus = (const REAL *)a->ygaus;
    green->zgaus = (const REAL *)a->zgaus;
    green->eps0 = &init->eps0;
    green->e0 = (const COMPLEX *)a->e0;
    green->nx = &runtime->nx;
    green->ny = &runtime->ny;
    green->nz = &runtime->nz;
    green->nx2 = &runtime->nx2;
    green->ny2 = &runtime->ny2;
    green->nz2 = &runtime->nz2;
    green->nxy2 = &runtime->nxy2;
    green->nxm = &init->nxm;
    green->nym = &init->nym;
    green->nzm = &init->nzm;
    green->ntotalm = &init->ntotalm;
    green->aretecube = (const REAL *)a->aretecube;
    green->k0 = (const REAL *)a->k0;
    green->ux = ux;
    green->uy = uy;
    green->uz = uz;
    green->ready = ready;
    green->green = mode == 9 ? runtime->vecty : runtime->vectx;
    green->planb = &runtime->planb;
    green->nstop = a->nstop;
    green->infostr = a->infostr;
    green->infostr_len = a->infostr_len;
    return 0;
}

extern "C" int cdmlib_prepare_scalar_initialization_params_c99(
    const cdmlib_abi_args_c99 *a,
    const cdmlib_initphase_c99 *init,
    cdmlib_entry_runtime_c99 *runtime,
    COMPLEX *ux, COMPLEX *uy, COMPLEX *uz,
    cdmlib_scalar_initialization_params_c99 *s)
{
    if (!a || !init || !runtime || !s || !ux || !uy || !uz ||
        !runtime->vectx || !runtime->vecty ||
        !a->tabdip || !a->macroscopicfieldx || !a->macroscopicfieldy ||
        !a->xi || !a->xr || !a->wrk || !a->ff || !a->ff0 ||
        !a->ffloc || !a->polarisa || !a->nlar || !a->nbsphere ||
        !a->ndipole || !a->niso || !a->nlim || !a->k0 ||
        !a->aretecube || !a->nstop || !a->nloop || !a->ncompte ||
        !a->tol1 || !a->methodeit || !a->infostr)
        return 1;

    *s = cdmlib_scalar_initialization_params_c99{};
    s->green = runtime->vectx;
    s->scratch = runtime->vecty;
    s->macroscopicfieldx = (COMPLEX *)a->macroscopicfieldx;
    s->macroscopicfieldy = (COMPLEX *)a->macroscopicfieldy;
    s->ux = ux; s->uy = uy; s->uz = uz;
    s->xi = (COMPLEX *)a->xi;
    s->xr = (COMPLEX *)a->xr;
    s->wrk = (COMPLEX *)a->wrk;
    s->ff = (COMPLEX *)a->ff;
    s->ff0 = (COMPLEX *)a->ff0;
    s->ffloc = (COMPLEX *)a->ffloc;
    s->polarisa = (COMPLEX *)a->polarisa;
    s->tabdip = (int *)a->tabdip;
    s->k0 = (const REAL *)a->k0;
    s->aretecube = (const REAL *)a->aretecube;
    s->ntotalm = init->ntotalm;
    s->ntotal = runtime->ntotal;
    s->ldabi = init->ldabi;
    s->nlar = (int)*a->nlar;
    s->nmax = init->nmax;
    s->ndipole = (int)*a->ndipole;
    s->nxm = init->nxm; s->nym = init->nym; s->nzm = init->nzm;
    s->nx = runtime->nx; s->ny = runtime->ny; s->nz = runtime->nz;
    s->nx2 = runtime->nx2; s->ny2 = runtime->ny2;
    s->nxy2 = runtime->nxy2; s->nz2 = runtime->nz2;
    s->nbsphere = (int)*a->nbsphere;
    s->niso = (int)*a->niso;
    s->nlim = (int)*a->nlim;
    s->nproche = init->nproche;
    s->planf = &runtime->planf;
    s->planb = &runtime->planb;
    s->nstop = a->nstop;
    s->nloop = a->nloop;
    s->ncompte = a->ncompte;
    s->tol1 = a->tol1;
    s->methodeit = a->methodeit;
    s->infostr = a->infostr;
    s->methodeit_len = a->methodeit_len;
    s->infostr_len = a->infostr_len;
    return 0;
}
