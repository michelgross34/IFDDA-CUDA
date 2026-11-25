"""Adapt the local NASA fixed-orientation DP program for a shared library."""
from argparse import ArgumentParser
from pathlib import Path
import re

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
      IF (NGAUSS.LT.1.OR.NGAUSS.GT.NPNG1) RETURN
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
(build / "mish_core.f").write_text(notice + start + declarations + setup + blocks +
                                   "\n      ISTAT=0\n      END\n\n" + routines, encoding="ascii")
# NASA's LAPACK file also starts with a short non-Fortran header in some releases.
lapack = (source / "lpd.f").read_text(encoding="ascii")
lapack = "\n".join(line for line in lapack.splitlines()
                   if not line or line[0] in "Cc*! " or line[0].isdigit()) + "\n"
(build / "lpd.f").write_text(lapack, encoding="ascii")
