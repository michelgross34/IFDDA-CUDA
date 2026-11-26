"""Adapt the local NASA fixed-orientation DP program for a shared library."""
from argparse import ArgumentParser
from pathlib import Path
import re
from tmatrix_openmp import parallelize

root = Path(__file__).resolve().parent
source = root / "NASA"
parser = ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=root / "build")
build = parser.parse_args().output_dir
build.mkdir(parents=True, exist_ok=True)
code = (source / "ampld.lp.f").read_text(encoding="ascii")
comments, code = code.split("C   New release including", 1)
notice, code = ("C   New release including" + code).split("      IMPLICIT REAL*8 (A-H,O-Z)", 1)
declarations = code.split("C  OPEN FILES", 1)[0].replace("      REAL*4", "      REAL*8")
blocks = code.split("  155 CONTINUE", 1)[1].split("      WALB=", 1)[0]
routines = "      SUBROUTINE AMPL " + code.split("      SUBROUTINE AMPL ", 1)[1]
# Keep the common block at the same precision in the builder and AMPL.
routines = routines.replace("      REAL*4", "      REAL*8")
# Shape -4 shares the cylinder's split surface quadrature, with the
# breakpoint at the tangent junction between the flank and a cap.
routines = routines.replace("      IF (NP.EQ.-2) GO  TO 11",
    "      IF (NP.EQ.-2.OR.(NP.EQ.-4.AND.EPS.GT.0D0)) GO TO 11", 1)
routines = routines.replace("      CALL GAUSS(NG1,0,0,X1,W1)",
    "      IF (NP.EQ.-4) XX=-EPS/DSQRT(1D0+EPS*EPS)\n"
    "      CALL GAUSS(NG1,0,0,X1,W1)", 1)
capsule_hook = "      IF (NP.EQ.-2) CALL RSP3(X,NG,NGAUSS,A,EPS,R,DR)"
routines = routines.replace(capsule_hook, capsule_hook +
    "\n      IF (NP.EQ.-4) CALL RSPCAP(X,NG,NGAUSS,A,EPS,R,DR)", 1)
routines += """
C     REV is the volume-equivalent radius. EPS = straight length / diameter.
      SUBROUTINE RSPCAP(X,NG,NGAUSS,REV,EPS,R,DR)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 X(NG),R(NG),DR(NG)
      RAD=REV/(1D0+1.5D0*EPS)**(1D0/3D0)
      H=RAD*EPS
      XJOIN=EPS/DSQRT(1D0+EPS*EPS)
      DO I=1,NGAUSS
         CO=-X(I)
         SI=DSQRT((1D0-CO)*(1D0+CO))
         IF (CO.GT.XJOIN) THEN
            ROOT=DSQRT(RAD*RAD-(H*SI)**2)
            RI=H*CO+ROOT
            DLOG=-H*SI/ROOT
         ELSE
            RI=RAD/SI
            DLOG=-CO/SI
         ENDIF
         R(I)=RI**2
         R(NG-I+1)=R(I)
         DR(I)=-DLOG
         DR(NG-I+1)=DLOG
      ENDDO
      END
"""
routines = re.sub(r"(?mi)^      (WRITE\s*\(6,100[56]\).*|PRINT\s+110[1-4].*)$", r"C     \1", routines)
routines = routines.replace("          STOP", "          RETURN")
routines = routines.replace("      IF (NMAX.GT.NPN1) STOP", "      IF (NMAX.GT.NPN1) RETURN")
tt = routines.index("      SUBROUTINE TT(")
routines = routines[:tt] + routines[tt:].replace(
    "      COMMON /CT/ TR1,TI1",
    "      COMMON /CT/ TR1,TI1\n      COMMON /MISH_STATUS/ IFAIL", 1
).replace(
    "      IF (INFO.NE.0) WRITE (6,1100) INFO",
    "      IF (INFO.NE.0) THEN\n         IFAIL=1\n         RETURN\n      ENDIF",
)
blocks = blocks.replace("         NM=NMAX-M+1", "         IF (IFAIL.NE.0) RETURN\n         NM=NMAX-M+1")
start = """
      SUBROUTINE BUILD_MISH(NP,A,EPS,LAM,MRR,MRI,NMAX,NGAUSS,
     &                      ISTAT)
      IMPLICIT REAL*8 (A-H,O-Z)
"""
setup = """
      COMMON /MISH_STATUS/ IFAIL
      IFAIL=0
      ISTAT=1
      IF (NMAX.LT.1.OR.NMAX.GT.NPN1.OR.NMAX.GT.NPN4) RETURN
      IF (NGAUSS.LT.2.OR.NGAUSS.GT.NPNG1) RETURN
      P=DACOS(-1D0)
      NCHECK=1
      MMAX=NMAX
      CALL CONST(NGAUSS,NMAX,MMAX,P,X,W,AN,ANN,S,SS,NP,EPS)
      CALL VARY(LAM,MRR,MRI,A,EPS,NP,NGAUSS,X,P,PPI,PIR,PII,R,
     &          DR,DDR,DRR,DRI,NMAX)
      CALL TMATR0(NGAUSS,X,W,AN,ANN,S,SS,PPI,PIR,PII,R,DR,
     &            DDR,DRR,DRI,NMAX,NCHECK)
      IF (IFAIL.NE.0) RETURN
"""
pack = """
      SUBROUTINE MISH_PACK_TMAT(NMAX,W)
      IMPLICIT NONE
      INTEGER NPN1,NPNG1,NPNG2,NPN2,NPL,NPN3,NPN4,NPN5,NPN6
      INCLUDE 'ampld.par.f'
      INTEGER NMAX,M,N,NN
      REAL*8 W(8,NMAX,NMAX,0:NMAX)
      REAL*8 TR11(NPN6,NPN4,NPN4),TR12(NPN6,NPN4,NPN4),
     & TR21(NPN6,NPN4,NPN4),TR22(NPN6,NPN4,NPN4),
     & TI11(NPN6,NPN4,NPN4),TI12(NPN6,NPN4,NPN4),
     & TI21(NPN6,NPN4,NPN4),TI22(NPN6,NPN4,NPN4)
      COMMON /TMAT/ TR11,TR12,TR21,TR22,TI11,TI12,TI21,TI22
      W=0D0
      DO M=0,NMAX
        DO NN=MAX(M,1),NMAX
          DO N=MAX(M,1),NMAX
            W(1,N,NN,M)=TR11(M+1,N,NN)
            W(2,N,NN,M)=TI11(M+1,N,NN)
            W(3,N,NN,M)=TR12(M+1,N,NN)
            W(4,N,NN,M)=TI12(M+1,N,NN)
            W(5,N,NN,M)=TR21(M+1,N,NN)
            W(6,N,NN,M)=TI21(M+1,N,NN)
            W(7,N,NN,M)=TR22(M+1,N,NN)
            W(8,N,NN,M)=TI22(M+1,N,NN)
          ENDDO
        ENDDO
      ENDDO
      END
"""
(build / "mish_core.f").write_text(notice + start + declarations + setup + blocks +
                                   "\n      ISTAT=0\n      END\n\n" + parallelize(routines) + pack, encoding="ascii")
# NASA's LAPACK file also starts with a short non-Fortran header in some releases.
lapack = (source / "lpd.f").read_text(encoding="ascii")
lapack = "\n".join(line for line in lapack.splitlines()
                   if not line or line[0] in "Cc*! " or line[0].isdigit()) + "\n"
(build / "lpd.f").write_text(lapack, encoding="ascii")
