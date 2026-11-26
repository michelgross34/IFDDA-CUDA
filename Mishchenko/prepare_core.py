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

# Extended precision NASA solver. Only the final TMAT block is shared with DP.
from quad_intrinsics import convert_quad_intrinsics
qcode=convert_quad_intrinsics((source/"amplq.lp.f").read_text(encoding="ascii"))
qcode=qcode[qcode.index("C   New release including"):]
noticeq,qcode=qcode.split("      IMPLICIT REAL*8 (A-H,O-Z)",1)
declq=qcode.split("C  OPEN FILES",1)[0].replace("      REAL*4","      REAL*8")
blocksq=qcode.split("  155 CONTINUE",1)[1].split("      WALB=",1)[0]
rq=qcode[qcode.index("      SUBROUTINE CONST "):]
rq=rq.replace("      IF (NP.EQ.-2) GO  TO 11",
    "      IF (NP.EQ.-2.OR.(NP.EQ.-4.AND.EPS.GT.0Q0)) GO TO 11",1)
rq=rq.replace("      CALL QGAUSS(NG1,0,0,X1,W1)",
    "      IF (NP.EQ.-4) XX=-EPS/SQRT(1Q0+EPS*EPS)\n      CALL QGAUSS(NG1,0,0,X1,W1)",1)
rq=rq.replace(capsule_hook,capsule_hook+"\n      IF (NP.EQ.-4) CALL RSPCAP(X,NG,NGAUSS,A,EPS,R,DR)",1)
profile=routines[routines.index("      SUBROUTINE RSPCAP"):].replace("REAL*8","REAL*16").replace("DSQRT","SQRT")
profile=re.sub(r"(?<=\d)D0","Q0",profile)
rq+=profile
rq=re.sub(r"(?mi)^      (WRITE\s*\(6,100[56]\).*|PRINT\s+110[1-4].*)$",r"C     \1",rq)
rq=rq.replace("      IF (NMAX.GT.NPN1) STOP","      IF (NMAX.GT.NPN1) RETURN")
ttq=rq.index("      SUBROUTINE TT(")
rq=rq[:ttq]+rq[ttq:].replace("      COMMON /CT/ TR1,TI1","      COMMON /CT/ TR1,TI1\n      COMMON /MISH_STATUS/ IFAIL",1).replace("      IF (INFO.NE.0) WRITE (6,1100) INFO","      IF (INFO.NE.0) THEN\n         IFAIL=1\n         RETURN\n      ENDIF")
# Avoid rounding the matrix product to FP64 before the final TMAT export.
rq=rq[:ttq]+rq[ttq:].replace("      REAL*8 TR1(NPN2,NPN2),TI1(NPN2,NPN2)","      REAL*16 TR,TI,ARR,ARI,AR,AI\n      REAL*8 TR1(NPN2,NPN2),TI1(NPN2,NPN2)",1)
blocksq=blocksq.replace("         NM=NMAX-M+1","         IF (IFAIL.NE.0) RETURN\n         NM=NMAX-M+1")
startq="      SUBROUTINE BUILD_MISH_QP(NP,AD,EPSD,LAMD,MRRD,MRID,\n     &                         NMAX,NGAUSS,ISTAT)\n      IMPLICIT REAL*8 (A-H,O-Z)\n      REAL*8 AD,EPSD,LAMD,MRRD,MRID\n"
setupq=setup.replace("      IFAIL=0","      A=REAL(AD,16)\n      EPS=REAL(EPSD,16)\n      LAM=REAL(LAMD,16)\n      MRR=REAL(MRRD,16)\n      MRI=REAL(MRID,16)\n      IFAIL=0").replace("DACOS(-1D0)","ACOS(-1Q0)")
lq=convert_quad_intrinsics((source/"lpq.f").read_text(encoding="ascii"))
lq="\n".join(line for line in lq.splitlines() if not line or line[0] in "Cc*! " or line[0].isdigit())+"\n"
q=noticeq+startq+declq+setupq+blocksq+"\n      ISTAT=0\n      END\n\n"+rq+"\n"+lq
names={x.upper() for x in re.findall(r"(?mi)^ {6}[^!\n]*?\b(?:SUBROUTINE|FUNCTION)\s+([A-Z][A-Z0-9_]*)",q)}-{"BUILD_MISH_QP"}
names|={x.upper() for x in re.findall(r"(?mi)^ {6}COMMON\s*/([A-Z0-9_]+)/",q)}-{"TMAT"}
pat=re.compile(r"\b("+"|".join(sorted(names))+r")\b",re.I)
def rename_q(line):
    if not line.startswith(" "):return line
    return "".join(part if i%2 else pat.sub(lambda mat:"MQ_"+mat[0].upper(),part) for i,part in enumerate(re.split(r"('[^']*')",line)))
(build/"mish_qp.f").write_text(parallelize("\n".join(map(rename_q,q.splitlines()))+"\n"),encoding="ascii")
