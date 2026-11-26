// FP64 amplitude precontraction adapted from TMatrix-Holo (Michel Gross, 2026).
// Identity particle frame; arbitrary complex incident E_theta/E_phi.
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#define HD __host__ __device__
namespace {
constexpr double pi=3.141592653589793238462643383279502884;
constexpr double rad=pi/180.;
struct Z{
 double re,im;
 HD Z(double a=0,double b=0):re(a),im(b){}
 HD Z operator+(Z b)const{return {re+b.re,im+b.im};}
 HD Z operator-(Z b)const{return {re-b.re,im-b.im};}
 HD Z operator*(double b)const{return {re*b,im*b};}
 HD Z operator*(Z b)const{return {re*b.re-im*b.im,re*b.im+im*b.re};}
 HD Z& operator+=(Z b){re+=b.re;im+=b.im;return *this;}
 HD double norm()const{return re*re+im*im;}
};
struct V{double x,y,z;};
HD double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
struct Weight{Z a,b,c,d;};
struct Frame{double ct,phi,r00,r01,r10,r11;};
struct State{int n;double lam,alpha,ca,sa,cb,sb,inc_phi; Z ix,iy,px,py;};
HD V rotate(const State&s,V v){
 return {s.ca*s.cb*v.x+s.sa*s.cb*v.y-s.sb*v.z,
         -s.sa*v.x+s.ca*v.y,s.ca*s.sb*v.x+s.sa*s.sb*v.y+s.cb*v.z};
}
// Match the small angular displacements and rotated bases of NASA AMPL.
HD Frame frame(const State&s,double theta,double phi){
 double t=theta*rad,p=phi*rad;
 constexpr double eps=1e-7;
 if(t<pi/2)t+=eps;else if(t>pi/2)t-=eps;
 if(p<pi)p+=eps;else if(p>pi)p-=eps;
 double ct=cos(t),st=sin(t),cp=cos(p),sp=sin(p);
 double ca=cos(p-s.alpha),sa=sin(p-s.alpha);
 double c=fmax(-1.,fmin(1.,ct*s.cb+st*s.sb*ca));
 double ph=atan2(st*sa,s.cb*st*ca-s.sb*ct);
 if(ph<0)ph+=2*pi;
 double cp1=cos(ph),sp1=sin(ph),st1=sin(acos(c));
 V bt={c*cp1,c*sp1,-st1},bp={-sp1,cp1,0};
 V lt=rotate(s,{ct*cp,ct*sp,-st}),lp=rotate(s,{-sp,cp,0});
 return {c,ph,dot(bt,lt),dot(bt,lp),dot(bp,lt),dot(bp,lp)};
}
// Streaming equivalent of VIGAMPL: same pole limit and normalized recurrence.
struct Vig{
 double x,qs,inv,prev,cur;
 int m;bool pole;
 HD Vig(double v,int mm):x(v),m(mm){
  pole=fabs(1-fabs(x))<=1e-10;
  qs=pole?0:sqrt(fmax(0.,1-x*x));inv=pole?0:1/qs;
  prev=m?0:1;cur=m?1:x;
  if(m&&!pole)for(int j=1;j<=m;++j)cur*=sqrt(double(2*j-1)/(2*j))*qs;
 }
 HD void next(int n,double&v1,double&v2){
  if(pole){
   v1=v2=0;
   if(m==1){double d=.5*sqrt(double(n*(n+1)));
    if(x<0&&!(n&1))d=-d;
    v1=d;v2=x<0?-d:d;}
   return;
  }
  double q=n,q1=n+1,q2=2*n+1,d3,der;
  if(m==0){d3=(q2*x*cur-q*prev)/q1;der=inv*(q1*q/q2)*(-prev+d3);}
  else{double a=sqrt(q*q-m*m),b=sqrt(q1*q1-m*m);
   d3=(q2*x*cur-a*prev)/b;der=inv*(-q1*a*prev+q*b*d3)/q2;}
  v1=cur*inv;v2=der;prev=cur;cur=d3;
 }
};
HD void amplitude(State s,const Weight* ww,double th,double ph,Z&out1,Z&out2){
 Frame f=frame(s,th,ph);
 Z v,h;
 for(int m=0;m<=s.n;++m){
  Vig g(f.ct,m);
  double fc=2*cos(m*(f.phi-s.inc_phi)),fs=2*sin(m*(f.phi-s.inc_phi));
  for(int n=m>0?m:1;n<=s.n;++n){
   double d1,d2;g.next(n,d1,d2);d1*=m;
   const Weight w=ww[m*s.n+n-1];
   if(m==0){v+=w.b*s.ix*d2;h+=w.c*s.iy*d2;}
   else{
    v+=(w.a*d1+w.b*d2)*(s.ix*fc)+(w.c*d1+w.d*d2)*(s.iy*fs);
    h+=(w.a*d2+w.b*d1)*(s.ix*(-fs))+(w.c*d2+w.d*d1)*(s.iy*fc);
   }
  }
 }
 double factor=s.lam/(2*pi)/(f.r00*f.r11-f.r01*f.r10);
 out1=(v*f.r11-h*f.r01)*factor;
 out2=(h*f.r00-v*f.r10)*factor;
}
void check(cudaError_t e,const char*where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
template<class T>struct Buffer{
 T* p=nullptr;
 Buffer()=default;
 explicit Buffer(size_t n){alloc(n);}
 ~Buffer(){if(p)cudaFree(p);}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
 void alloc(size_t n){if(p){check(cudaFree(p),"cudaFree");p=nullptr;}if(n)check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"cudaMalloc");}
 void put(const T*q,size_t n){if(n)check(cudaMemcpy(p,q,n*sizeof(T),cudaMemcpyHostToDevice),"upload");}
 void get(T*q,size_t n){if(n)check(cudaMemcpy(q,p,n*sizeof(T),cudaMemcpyDeviceToHost),"download");}
};

__global__ void section_kernel(State s,const Weight* ww,const double*mu,const double*w,
                               int nt,int np,double theta,double*res) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;
 if(j>nt*np)return;
 Z a,b;
 if(j==nt*np) {
   amplitude(s,ww,theta,0,a,b);
   res[j]=2*s.lam*(s.px.re*a.im-s.px.im*a.re+s.py.re*b.im-s.py.im*b.re);
 } else {
   int k=j/np,l=j%np;
   amplitude(s,ww,acos(fmax(-1.,fmin(1.,mu[k])))/rad,(l+.5)*360/np,a,b);
   res[j]=w[k]*(2*pi/np)*(a.norm()+b.norm());
 }
}
}
#ifdef _WIN32
#define MISH_EXPORT __declspec(dllexport)
#else
#define MISH_EXPORT
#endif
extern "C" MISH_EXPORT int mish_cuda_sections(int n,double theta,
    double pr,double pi_,double sr,double si,
    int nt,int np,const double* mu,const double* qw,const double* t,double* out) {
 try {
  if(n<1||n>100||nt<1||np<1||!mu||!qw||!t||!out) return 1;
  // NASA AMPL is called with alpha=beta=0 in the reference wrapper.
  State state={n,1.,0.,1.,0.,1.,0.,0.,Z(),Z(),Z(pr,pi_),Z(sr,si)};
  Frame f=frame(state,theta,0);
  state.inc_phi=f.phi;
  state.ix=state.px*f.r00+state.py*f.r01;
  state.iy=state.px*f.r10+state.py*f.r11;
  std::vector<Weight> w(size_t(n+1)*n);

 for(int m=0;m<=n;++m){
  Vig g(f.ct,m);
  for(int nn=m?m:1;nn<=n;++nn){
   double d1,d2;g.next(nn,d1,d2);d1*=m;
   for(int k=m?m:1;k<=n;++k){
    const double*p=t+8*(size_t(m)*n*n+size_t(nn-1)*n+k-1);
    Z t11(p[0],p[1]),t12(p[2],p[3]),t21(p[4],p[5]),t22(p[6],p[7]);
    double r=sqrt(double((2*k+1)*(2*nn+1))/double(k*nn*(k+1)*(nn+1)));
    int phase=(nn-k-1)%4;if(phase<0)phase+=4;
    Z cal=phase==0?Z(r,0):phase==1?Z(0,r):phase==2?Z(-r,0):Z(0,-r);
    auto&v=w[m*n+k-1];
    if(m==0){v.b+=cal*t22*d2;v.c+=cal*t11*d2;}
    else{
     v.a+=cal*(t11*d1+t12*d2);v.b+=cal*(t21*d1+t22*d2);
     v.c+=cal*(t11*d2+t12*d1);v.d+=cal*(t21*d2+t22*d1);
    }
   }
  }
 }

  Buffer<Weight> weights(w.size()); weights.put(w.data(),w.size());
  int count=nt*np+1;
  Buffer<double> dmu(nt),dw(nt),dr(count);dmu.put(mu,nt);dw.put(qw,nt);
  section_kernel<<<(count+127)/128,128>>>(state,weights.p,dmu.p,dw.p,nt,np,theta,dr.p);
  check(cudaGetLastError(),"Mishchenko sections kernel");
  std::vector<double> r(count);dr.get(r.data(),count);
  // Same deterministic quadrature traversal as the serial Fortran reference.
  double sca=0;for(int j=0;j<count-1;++j)sca+=r[j];
  out[0]=r.back();out[1]=sca;out[2]=out[0]-out[1];return 0;
 } catch(const std::exception& e) {
  std::fprintf(stderr,"Mishchenko CUDA: %s\n",e.what());return 1;
 } catch(...) {return 1;}
}
