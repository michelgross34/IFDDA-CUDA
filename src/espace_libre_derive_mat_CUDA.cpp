#include "cdmlib_cpp_precision.hpp"
#include <limits>
#include <complex>
/* C99 port of cdmlib/espace_libre_derive_mat.f90. */
#include <float.h>
#include <cmath>

// Derivatives contain intermediate 1/r^5 and 1/r^7 terms. At the 5 nm
// mesh spacing these overflow float32 before the terms cancel. Match the
// Fortran reference's widened intermediates and narrow only the result.
using DerivReal = double;
using DerivComplex = std::complex<DerivReal>;

static void set_axis(DerivComplex out[27], int axis,
                     DerivReal s, DerivReal c, DerivReal aa,
                     DerivComplex txy,
                     DerivComplex txz,
                     DerivComplex txxd_a, DerivComplex txyd_a,
                     DerivComplex txzd_a, DerivComplex tzzd_a,
                     DerivComplex txxd_z, DerivComplex txyd_z,
                     DerivComplex txzd_z, DerivComplex tzzd_z)
{
    const DerivReal s2=DerivReal(2.0)*s*c, c2=c*c-s*s;
    const DerivComplex tzx=txz, tzxd_a=txzd_a, tzxd_z=txzd_z;
    DerivComplex *m=out+9*axis;
#define M(r,col) m[(r)+3*(col)]
    if (axis==0) {
        M(0,0)=s*txxd_a-DerivReal(2.0)*c*s2*txy/aa+c2*s*txyd_a;
        M(0,1)=-DerivReal(2.0)*c*c2*txy/aa-s2*s*txyd_a;
        M(0,2)=c*c*txz/aa+s*s*txzd_a;
        M(1,0)=M(0,1);
        M(1,1)=s*txxd_a+DerivReal(2.0)*c*s2*txy/aa-c2*s*txyd_a;
        M(1,2)=-s*c*txz/aa+c*s*txzd_a;
        M(2,0)=c*c*tzx/aa+s*s*tzxd_a;
        M(2,1)=-s*c*tzx/aa+c*s*tzxd_a;
        M(2,2)=s*tzzd_a;
    } else if (axis==1) {
        M(0,0)=c*txxd_a+DerivReal(2.0)*s*s2*txy/aa+c2*c*txyd_a;
        M(0,1)=DerivReal(2.0)*s*c2*txy/aa-s2*c*txyd_a;
        M(0,2)=-s*c*txz/aa+s*c*txzd_a;
        M(1,0)=M(0,1);
        M(1,1)=c*txxd_a-DerivReal(2.0)*s*s2*txy/aa-c2*c*txyd_a;
        M(1,2)=s*s*txz/aa+c*c*txzd_a;
        M(2,0)=-s*c*tzx/aa+s*c*tzxd_a;
        M(2,1)=s*s*tzx/aa+c*c*tzxd_a;
        M(2,2)=c*tzzd_a;
    } else {
        M(0,0)=txxd_z+c2*txyd_z;
        M(0,1)=-s2*txyd_z;
        M(0,2)=s*txzd_z;
        M(1,0)=-s2*txyd_z;
        M(1,1)=txxd_z-c2*txyd_z;
        M(1,2)=c*txzd_z;
        M(2,0)=s*tzxd_z;
        M(2,1)=c*tzxd_z;
        M(2,2)=tzzd_z;
    }
#undef M
}

extern "C" void propesplibdermat_(const REAL *x, const REAL *y, const REAL *z,
                       const REAL *x0, const REAL *y0, const REAL *z0,
                       const REAL *k0, const int *test, COMPLEX out[27])
{
    const DerivReal xxp=DerivReal(*x)-DerivReal(*x0);
    const DerivReal yyp=DerivReal(*y)-DerivReal(*y0);
    const DerivReal dz=DerivReal(*z)-DerivReal(*z0);
    const DerivReal wave=DerivReal(*k0);
    const DerivReal a=std::sqrt(xxp*xxp+yyp*yyp), a2=a*a;
    const DerivReal rab=std::sqrt(xxp*xxp+yyp*yyp+dz*dz), rab2=rab*rab;
    DerivReal aa=a, s, c;
    DerivComplex txx=0.0,txy=0.0,tzz=0.0,txz=0.0;
    DerivComplex txxdz=0.0,txydz=0.0,txzdz=0.0,tzzdz=0.0;
    DerivComplex txxda=0.0,txyda=0.0,txzda=0.0,tzzda=0.0;

    if (rab!=0.0) {
        const DerivReal rab3=rab2*rab, rab4=rab3*rab, rab5=rab4*rab;
        const DerivComplex i_k0(0.0,wave), imag(0.0,1.0);
        const DerivComplex const1=DerivReal(1.0)/rab3-i_k0/rab2;
        const DerivComplex const2=wave*wave/rab;
        const DerivComplex c13c2=DerivReal(3.0)*const1-const2;
        const DerivComplex dconst1=-DerivReal(3.0)/rab5+DerivReal(2.0)*i_k0/rab4;
        const DerivComplex dconst2=-wave*wave/rab3;
        const DerivComplex dc13c2=DerivReal(3.0)*dconst1-dconst2;
        const DerivComplex ddd=(dc13c2*rab2-c13c2*DerivReal(2.0))/rab4;
        const DerivComplex phase=std::exp(i_k0*rab);
        txx=((const2-const1)+a2*c13c2/rab2/DerivReal(2.0))*phase;
        tzz=((const2-const1)+dz*dz*c13c2/rab2)*phase;
        txy=-a2*c13c2/rab2/DerivReal(2.0)*phase;
        txz=a*dz*c13c2/rab2*phase;
        txxdz=imag*txx*dz*wave/rab+phase*((dconst2-dconst1)*dz+a2/DerivReal(2.0)*ddd*dz);
        tzzdz=imag*tzz*dz*wave/rab+phase*((dconst2-dconst1)*dz+DerivReal(2.0)*dz*c13c2/rab2+dz*dz*ddd*dz);
        txydz=imag*txy*dz*wave/rab-a2*phase/DerivReal(2.0)*ddd*dz;
        txzdz=imag*txz*dz*wave/rab+a*phase*(c13c2/rab2+dz*ddd*dz);
        txxda=imag*txx*a*wave/rab+phase*((dconst2-dconst1)*a+a*c13c2/rab2+a2/DerivReal(2.0)*ddd*a);
        tzzda=imag*tzz*a*wave/rab+phase*((dconst2-dconst1)*a+dz*dz*ddd*a);
        txyda=imag*txy*a*wave/rab+phase*(-a*c13c2/rab2-a2/DerivReal(2.0)*ddd*a);
        txzda=imag*txz*a*wave/rab+phase*(dz*c13c2/rab2+a*dz*ddd*a);
    }

    DerivComplex wide[27]{};

    if (*test==1 || *test==4) {
        if (a<=1.0e-30) { s=1.0; c=0.0; aa=std::numeric_limits<REAL>::max()/100.0; }
        else { s=xxp/a; c=yyp/a; }
        set_axis(wide,0,s,c,aa,txy,txz,txxda,txyda,txzda,tzzda,
                 txxdz,txydz,txzdz,tzzdz);
    }
    if (*test==2 || *test==4) {
        aa=a;
        if (a<=1.0e-30) { s=0.0; c=1.0; aa=std::numeric_limits<REAL>::max()/100.0; }
        else { s=xxp/a; c=yyp/a; }
        set_axis(wide,1,s,c,aa,txy,txz,txxda,txyda,txzda,tzzda,
                 txxdz,txydz,txzdz,tzzdz);
    }
    if (*test==3 || *test==4) {
        aa=a;
        if (a<=1.0e-30) { s=0.0; c=0.0; aa=std::numeric_limits<REAL>::max()/100.0; }
        else { s=xxp/a; c=yyp/a; }
        set_axis(wide,2,s,c,aa,txy,txz,txxda,txyda,txzda,tzzda,
                 txxdz,txydz,txzdz,tzzdz);
    }
    for (int i=0; i<27; ++i)
        out[i] = COMPLEX(static_cast<REAL>(wide[i].real()),
                         static_cast<REAL>(wide[i].imag()));
}

