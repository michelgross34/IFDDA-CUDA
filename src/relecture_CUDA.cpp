#include "cdmlib_cpp_abi.hpp"
/* C99 implementation of the restart metadata routine relecture.
 * The .bin format intentionally uses GNU Fortran sequential-unformatted
 * records (4-byte byte-count markers) so existing restart files remain
 * readable by both implementations.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef enum { REC_INT, REC_REAL, REC_COMPLEX, REC_CHAR } rec_kind;
typedef struct {
    rec_kind kind;
    size_t size;
    union { int32_t i; REAL d; REAL z[2]; unsigned char s[64]; } v;
} record;
typedef struct { record *v; size_t n, cap; } records;

static int push(records *r, rec_kind kind, const void *p, size_t n)
{
    if (r->n == r->cap) {
        size_t cap = r->cap ? r->cap * 2 : 64;
        record *v = (record *)realloc(r->v, cap * sizeof(*v));
        if (!v) return 0;
        r->v = v; r->cap = cap;
    }
    record *q = &r->v[r->n++];
    q->kind = kind; q->size = n;
    memset(&q->v, 0, sizeof(q->v));
    if (n > sizeof(q->v)) return 0;
    memcpy(&q->v, p, n);
    return 1;
}
static int pi(records *r, int x) { int32_t a=(int32_t)x; return push(r,REC_INT,&a,4); }
static int pr(records *r, REAL x) { return push(r,REC_REAL,&x,sizeof(x)); }
static int pc(records *r, const REAL *x) { return push(r,REC_COMPLEX,x,2*sizeof(REAL)); }
static int ps(records *r, const char *x, size_t n)
{
    unsigned char b[64]; if (n>sizeof(b)) return 0;
    memset(b,' ',sizeof(b)); if (x && n) memcpy(b,x,n);
    return push(r,REC_CHAR,b,n);
}
static int prefix(const char *s, const char *p, size_t n) { return strncmp(s,p,n)==0; }
static size_t trimlen(const char *s, size_t n) { while(n && s[n-1]==' ') --n; return n; }
static void fixed(char *out, size_t n, const char *in, size_t inlen)
{ memset(out,' ',n); if(in) memcpy(out,in,inlen<n?inlen:n); }
static REAL relerr(REAL a,REAL b)
{ if (b==0.0) return a==0.0?0.0:INFINITY; return fabs((a-b)/b); }
static int equal_complex(const REAL *a,const REAL *b)
{ REAL er=relerr(a[0],b[0]), ei=relerr(a[1],b[1]); return er<=1e-12 && ei<=1e-12; }

static int add_material(records *r, const char *trope, const REAL *eps,
                        const REAL *epsani, int multi, int k)
{
    if (prefix(trope,"iso",3)) return pc(r, multi ? eps + 2*k : eps);
    for (int ii=0;ii<3;ii++) for(int jj=0;jj<3;jj++) {
        size_t index = multi ? (size_t)(ii + 3*jj + 9*k) : (size_t)(ii+3*jj);
        if (!pc(r,epsani + 2*index)) return 0;
    }
    return 1;
}
static int add_beam(records *r, const char *beam, REAL pp, REAL ss,
                    REAL theta, REAL phi, const REAL *thetam,
                    const REAL *phim, const REAL *ppm, const REAL *ssm,
                    const REAL *E0m, int nbinc, REAL xgaus,REAL ygaus,
                    REAL zgaus,int speckseed,REAL na,int nfft2)
{
    if (prefix(beam,"pwavelinear",11))
        return pr(r,pp)&&pr(r,ss)&&pr(r,theta)&&pr(r,phi);
    if (prefix(beam,"pwavecircular",13))
        return pr(r,ss)&&pr(r,theta)&&pr(r,phi);
    if (prefix(beam,"wavelinearmulti",15)) {
        if(!pi(r,nbinc)) return 0;
        for(int k=0;k<nbinc;k++) if(!pr(r,ppm[k])||!pr(r,ssm[k])||
           !pr(r,thetam[k])||!pr(r,phim[k])||!pc(r,E0m+2*k)) return 0;
        return 1;
    }
    if (prefix(beam,"gwavelinear",11)||prefix(beam,"gparawavelinear",15)||
        prefix(beam,"gfftwavelinear",14)) {
        if(!pr(r,pp)||!pr(r,ss)||!pr(r,theta)||!pr(r,phi)||!pr(r,xgaus)||
           !pr(r,ygaus)||!pr(r,zgaus)) return 0;
        return !prefix(beam,"gfftwavelinear",14)||pi(r,nfft2);
    }
    if (prefix(beam,"gwavecircular",13)||prefix(beam,"gparawavecircular",17)||
        prefix(beam,"gfftwavecircular",16)) {
        if(!pr(r,ss)||!pr(r,theta)||!pr(r,phi)||!pr(r,xgaus)||!pr(r,ygaus)||
           !pr(r,zgaus)) return 0;
        return !prefix(beam,"gfftwavecircular",16)||pi(r,nfft2);
    }
    if (prefix(beam,"antenna",7))
        return pr(r,xgaus)&&pr(r,ygaus)&&pr(r,zgaus)&&pr(r,theta)&&pr(r,phi);
    if (prefix(beam,"speckle",7)||prefix(beam,"demispeckle",11))
        return pr(r,pp)&&pr(r,xgaus)&&pr(r,ygaus)&&pr(r,zgaus)&&pr(r,na)&&pi(r,speckseed);
    if (prefix(beam,"confocal",8)||prefix(beam,"demiconfocal",12))
        return pr(r,pp)&&pr(r,xgaus)&&pr(r,ygaus)&&pr(r,zgaus)&&pr(r,na);
    return 1;
}

static int add_object(records *r,const char *object,const char *trope,
 int nnnr,REAL aretecube,REAL density,REAL side,REAL sidex,REAL sidey,
 REAL sidez,REAL hauteur,REAL rayon,REAL xg,REAL yg,REAL zg,
 REAL lc,REAL hc,int ng,int nx,int ny,int nz,int numberobjet,
 const REAL *rayonmulti,const REAL *xgmulti,const REAL *ygmulti,
 const REAL *zgmulti,const REAL *epsmulti,const REAL *epsanimulti,
 const REAL *eps,const REAL *epsani,REAL demiaxea,REAL demiaxeb,REAL demiaxec,
 REAL thetaobj,REAL phiobj,REAL psiobj,const char *namefileobj,size_t nameobjlen)
{
    if(!pi(r,nnnr)||!ps(r,object,64)) return 0;
    if(prefix(object,"sphere",6)) {
        if(!pr(r,rayon)||!pr(r,xg)||!pr(r,yg)||!pr(r,zg)||!ps(r,trope,3)) return 0;
        return add_material(r,trope,eps,epsani,0,0);
    }
    if(prefix(object,"randomsphere1",13))
        return pr(r,rayon)&&pi(r,ng)&&pr(r,xg)&&pr(r,yg)&&pr(r,zg)&&
          pr(r,sidex)&&pr(r,sidey)&&pr(r,sidez)&&pr(r,density);
    if(prefix(object,"randomsphere2",13))
        return pr(r,rayon)&&pi(r,ng)&&pr(r,xg)&&pr(r,yg)&&pr(r,zg)&&
          pi(r,nx)&&pi(r,ny)&&pi(r,nz)&&pr(r,aretecube)&&pr(r,density);
    if(prefix(object,"randomspheremulti1",18)||prefix(object,"nspheres",8)||
       prefix(object,"concentricsphere",16)) {
        if(prefix(object,"randomspheremulti1",18) &&
           (!pi(r,ng)||!pr(r,sidex)||!pr(r,sidey)||!pr(r,sidez)||!pr(r,aretecube)||!pr(r,density))) return 0;
        if(!pi(r,numberobjet)||!ps(r,trope,3)) return 0;
        for(int k=0;k<numberobjet;k++) {
            if(!pr(r,xgmulti[k])||!pr(r,ygmulti[k])||!pr(r,zgmulti[k])||!pr(r,rayonmulti[k])) return 0;
            if(prefix(trope,"iso",3)) { if(!pc(r,epsmulti+2*k)) return 0; }
            else for(int ii=0;ii<3;ii++) for(int jj=0;jj<3;jj++)
                if(!pc(r,epsanimulti+2*(ii+3*jj+9*k))) return 0;
        }
        return 1;
    }
    if(prefix(object,"inhomosphere",12)) return pr(r,rayon)&&pr(r,lc)&&pr(r,hc)&&pi(r,ng);
    if(prefix(object,"inhomocuboid1",13)) return pr(r,sidex)&&pr(r,sidey)&&pr(r,sidez)&&
        pr(r,xg)&&pr(r,yg)&&pr(r,zg)&&pr(r,lc)&&pr(r,hc)&&pi(r,ng);
    if(prefix(object,"inhomocuboid2",13)) return pi(r,nx)&&pi(r,ny)&&pi(r,nz)&&
        pr(r,aretecube)&&pr(r,xg)&&pr(r,yg)&&pr(r,zg)&&pr(r,lc)&&pr(r,hc)&&pi(r,ng);
    if(prefix(object,"cube",4)) {
        if(!pr(r,side)||!pr(r,xg)||!pr(r,yg)||!pr(r,zg)||!ps(r,trope,3)) return 0;
        return add_material(r,trope,eps,epsani,0,0);
    }
    if(prefix(object,"cuboid1",7)) {
        if(!pr(r,sidex)||!pr(r,sidey)||!pr(r,sidez)||!pr(r,psiobj)||!pr(r,thetaobj)||
           !pr(r,phiobj)||!pr(r,xg)||!pr(r,yg)||!pr(r,zg)||!ps(r,trope,3)) return 0;
        return add_material(r,trope,eps,epsani,0,0);
    }
    if(prefix(object,"cuboid2",7)) {
        if(!pi(r,nx)||!pi(r,ny)||!pi(r,nz)||!pr(r,aretecube)||!pr(r,xg)||!pr(r,yg)||
           !pr(r,zg)||!ps(r,trope,3)) return 0;
        return add_material(r,trope,eps,epsani,0,0);
    }
    if(prefix(object,"ellipsoid",9)) {
        if(!pr(r,demiaxea)||!pr(r,demiaxeb)||!pr(r,demiaxec)||!pr(r,psiobj)||!pr(r,thetaobj)||
           !pr(r,phiobj)||!pr(r,xg)||!pr(r,yg)||!pr(r,zg)||!ps(r,trope,3)) return 0;
        return add_material(r,trope,eps,epsani,0,0);
    }
    if(prefix(object,"cylinder",8)) {
        if(!pr(r,rayon)||!pr(r,hauteur)||!pr(r,psiobj)||!pr(r,thetaobj)||!pr(r,phiobj)||
           !pr(r,xg)||!pr(r,yg)||!pr(r,zg)||!ps(r,trope,3)) return 0;
        return add_material(r,trope,eps,epsani,0,0);
    }
    if(prefix(object,"arbitrary",9))
        return pi(r,(int)nameobjlen)&&ps(r,namefileobj,nameobjlen);
    return 1;
}

static int write_record(FILE *f,const record *r)
{
    uint32_t n=(uint32_t)r->size;
    if(fwrite(&n,4,1,f)!=1 || fwrite(&r->v,1,r->size,f)!=r->size ||
       fwrite(&n,4,1,f)!=1) return 0;
    return 1;
}
static int read_record(FILE *f,const record *want,size_t idx)
{
    uint32_t n1=0,n2=0; unsigned char b[64];
    int ok=1;
    if(want->size>sizeof(b)||fread(&n1,4,1,f)!=1||n1!=want->size||
       fread(b,1,n1,f)!=n1||fread(&n2,4,1,f)!=1||n2!=n1) ok=0;
    else if(want->kind==REC_INT) { int32_t a; memcpy(&a,b,4); ok=a==want->v.i; }
    else if(want->kind==REC_REAL) { REAL a; memcpy(&a,b,sizeof(a)); ok=relerr(a,want->v.d)<=1e-12; }
    else if(want->kind==REC_COMPLEX) { REAL a[2]; memcpy(a,b,sizeof(a)); ok=equal_complex(a,want->v.z); }
    else ok=memcmp(b,want->v.s,want->size)==0;
    (void)idx;
    return ok;
}
static void write_text_record(FILE *f,const record *r,size_t idx)
{
    fprintf(f,"record %04lu = ",(unsigned long)idx);
    if(r->kind==REC_INT) fprintf(f,"%d",(int)r->v.i);
    else if(r->kind==REC_REAL) fprintf(f,"%.17E",r->v.d);
    else if(r->kind==REC_COMPLEX) fprintf(f,"(%.17E,%.17E)",r->v.z[0],r->v.z[1]);
    else { for(size_t i=0;i<r->size;i++) fputc(r->v.s[i],f); }
    fputc('\n',f);
}
static int make_records(records *r, REAL aretecube,REAL lambda,int nrig,int nfft2,
 const char *beam,const char *object,const char *trope,int nnnr,REAL density,
 REAL side,REAL sidex,REAL sidey,REAL sidez,REAL hauteur,int numberobjet,
 const REAL *rayonmulti,const REAL *xgmulti,const REAL *ygmulti,const REAL *zgmulti,
 const REAL *epsmulti,const REAL *epsanimulti,REAL lc,REAL hc,int ng,
 REAL demiaxea,REAL demiaxeb,REAL demiaxec,REAL thetaobj,REAL phiobj,REAL psiobj,
 const char *namefileobj,size_t nameobjlen,REAL theta,REAL phi,REAL pp,REAL ss,
 const REAL *thetam,const REAL *phim,const REAL *ppm,const REAL *ssm,
 const REAL *E0m,int nbinc,REAL P0,REAL w0,REAL xgaus,REAL ygaus,REAL zgaus,
 int speckseed,REAL numaperil,int nx,int ny,int nz,const REAL *eps,const REAL *epsani,
 const char *polarizability,int nquad)
{
    char bf[64],of[64],tf[3]; fixed(bf,64,beam,64); fixed(of,64,object,64); fixed(tf,3,trope,3);
    if(!pr(r,lambda)||!pr(r,P0)||!pr(r,w0)||!ps(r,bf,64)||!pi(r,nrig)) return 0;
    if(!add_beam(r,bf,pp,ss,theta,phi,thetam,phim,ppm,ssm,E0m,nbinc,
        xgaus,ygaus,zgaus,speckseed,numaperil,nfft2)) return 0;
    if(!add_object(r,of,tf,nnnr,aretecube,density,side,sidex,sidey,sidez,hauteur,
        rayonmulti[0],xgmulti[0],ygmulti[0],zgmulti[0],lc,hc,ng,nx,ny,nz,numberobjet,
        rayonmulti,xgmulti,ygmulti,zgmulti,epsmulti,epsanimulti,eps,epsani,
        demiaxea,demiaxeb,demiaxec,thetaobj,phiobj,psiobj,namefileobj,nameobjlen)) return 0;
    return ps(r,polarizability,2)&&pi(r,nquad);
}

static void set_message(char *out,size_t n,const char *msg)
{ fixed(out,n,msg,strlen(msg)); }

extern "C" void relecture_(REAL *aretecube,REAL *lambda,int *nrig,int *nfft2,
 char *beam,char *object,char *trope,int *nnnr,REAL *tolinit,REAL *density,
 REAL *side,REAL *sidex,REAL *sidey,REAL *sidez,REAL *hauteur,
 int *numberobjet,REAL *rayonmulti,REAL *xgmulti,REAL *ygmulti,REAL *zgmulti,
 COMPLEX *epsmulti,COMPLEX *epsanimulti,REAL *lc,REAL *hc,int *ng,
 REAL *demiaxea,REAL *demiaxeb,REAL *demiaxec,REAL *thetaobj,REAL *phiobj,
 REAL *psiobj,char *namefileobj,REAL *theta,REAL *phi,REAL *pp,REAL *ss,
 REAL *thetam,REAL *phim,REAL *ppm,REAL *ssm,COMPLEX *E0m,int *nbinc,
 REAL *P0,REAL *w0,REAL *xgaus,REAL *ygaus,REAL *zgaus,int *speckseed,
 REAL *numaperil,char *namefileinc,int *numberobjetmax,char *polarizability,int *nquad,
 char *filereread,int *nlecture,int *nx,int *ny,int *nz,int *nstop,char *infostr,
 size_t beam_len,size_t object_len,size_t trope_len,size_t namefileobj_len,
 size_t namefileinc_len,size_t polarizability_len,size_t filereread_len,size_t infostr_len)
{
    (void)tolinit; (void)namefileinc; (void)namefileinc_len; (void)numberobjetmax;
    (void)beam_len; (void)object_len; (void)trope_len; (void)polarizability_len;
    char base[65],datafile[65],binfile[65]; size_t base_len=trimlen(filereread,filereread_len);
    if(base_len==0) {
        set_message(infostr,infostr_len,"No file given for read local field from file");
        *nstop=1; return;
    }
    memcpy(base,filereread,base_len); base[base_len]='\0';
    /* Fortran assigns the concatenated name to CHARACTER(64), truncating
       any suffix beyond column 64; reproduce that fixed-length behavior. */
    snprintf(datafile,sizeof(datafile),"%.*s.data",(int)base_len,base);
    snprintf(binfile,sizeof(binfile),"%.*s.bin",(int)base_len,base);
    REAL eps[2]={std::real(epsmulti[0]),std::imag(epsmulti[0])};
    REAL epsani[18];
    for(int i=0;i<18;i++) epsani[i]=((REAL *)epsanimulti)[i];
    REAL e0[20]; for(int i=0;i<20;i++) e0[i]=((REAL *)E0m)[i];
    records r={0};
    if(!make_records(&r,*aretecube,*lambda,*nrig,*nfft2,beam,object,trope,*nnnr,
       *density,*side,*sidex,*sidey,*sidez,*hauteur,*numberobjet,rayonmulti,xgmulti,
       ygmulti,zgmulti,(REAL *)epsmulti,(REAL *)epsanimulti,*lc,*hc,*ng,
       *demiaxea,*demiaxeb,*demiaxec,*thetaobj,*phiobj,*psiobj,namefileobj,
       trimlen(namefileobj,namefileobj_len),*theta,*phi,*pp,*ss,thetam,phim,ppm,ssm,e0,
       *nbinc,*P0,*w0,*xgaus,*ygaus,*zgaus,*speckseed,*numaperil,*nx,*ny,*nz,
       eps,epsani,polarizability,*nquad)) {
        free(r.v); set_message(infostr,infostr_len,"Unable to build reread metadata"); *nstop=1; return;
    }
    FILE *fd=fopen(datafile,"rb");
    if(!fd) {
        fd=fopen(datafile,"wb"); FILE *fb=fopen(binfile,"wb");
        if(!fd||!fb) {
            if(fd) fclose(fd);
            if(fb) fclose(fb);
            free(r.v);
            set_message(infostr,infostr_len,"Unable to create reread metadata files"); *nstop=1; return;
        }
        for(size_t i=0;i<r.n;i++) {
            if(!write_record(fb,&r.v[i])) { fclose(fd); fclose(fb); free(r.v); *nstop=1; return; }
            write_text_record(fd,&r.v[i],i+1);
        }
        fclose(fd); fclose(fb); *nlecture=0; free(r.v); return;
    }
    fclose(fd);
    /* Preserve the original reader's scalar-epsilon check for these two
       isotropic multi-sphere geometries: the Fortran branch compares every
       stored epsilon against eps (epsmulti(1)), although its writer stores
       epsmulti(k). This is observable for heterogeneous spheres. */
    if(prefix(object,"concentricsphere",16)||prefix(object,"nspheres",8)) {
        if(prefix(trope,"iso",3)) {
            size_t object_record=r.n;
            for(size_t i=0;i<r.n;i++)
                if(r.v[i].kind==REC_CHAR && r.v[i].size==64 &&
                   memcmp(r.v[i].v.s,object,16)==0) { object_record=i; break; }
            if(object_record<r.n) for(int k=0;k<*numberobjet;k++) {
                size_t e=object_record+7u+5u*(size_t)k;
                if(e<r.n && r.v[e].kind==REC_COMPLEX) {
                    r.v[e].v.z[0]=std::real(epsmulti[0]);
                    r.v[e].v.z[1]=std::imag(epsmulti[0]);
                }
            }
        }
    }
    FILE *fb=fopen(binfile,"rb");
    if(!fb) { free(r.v); *nstop=1; set_message(infostr,infostr_len,"Datafile do not match the data or file already exists"); return; }
    int ok=1;
    for(size_t i=0;i<r.n&&ok;i++) ok=read_record(fb,&r.v[i],i+1);
    fclose(fb); free(r.v);
    if(ok) *nlecture=1;
    else { *nstop=1; set_message(infostr,infostr_len,"Datafile do not match the data or file already exists"); }
}



