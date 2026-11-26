"""Compare the three standalone backends against the serial NASA amplitude code."""
import os
import re
import sys
import time
import math
import subprocess
from pathlib import Path

root = Path(sys.argv[1]).resolve()
suffix = ".exe" if os.name == "nt" else ""
env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS","8"))
cases = [
    ("spheroid", ["spheroid","630","140","280","1.5",".02","27"]),
    ("cylinder", ["cylinder","630","180","360","1.5",".02","27","0","0","1","0"]),
    ("capsule sphere", ["capsule","630","336","0","3",".1","27"]),
    ("complex polarization", ["spheroid","630","140","280","1.5",".02","49","0.3","0.6","0.2","-0.7"]),
    ("pole 0", ["spheroid","630","140","280","1.5",".02","0","1","0","0","1"]),
    ("pole 180", ["spheroid","630","140","280","1.5",".02","180","1","0","0","1"]),
    ("equator", ["spheroid","630","140","280","1.5",".02","90","1","0","0","1"]),
    ("transparent", ["spheroid","630","140","280","1","0","27"]),
]
if "--large" in sys.argv:
    cases.append(("large capsule", ["capsule","630","336","672","3",".1","27",
                                     "0.7431448254773942","0","0.6691306063588582","0"]))
mode="qp" if "--qp" in sys.argv else "dp"
for label,args in cases:
    results=[]
    for expected,backend in enumerate(("cpu","openmp","cuda")):
        exe=root/("ifdda_mishchenko_sections_"+backend+suffix)
        start=time.perf_counter()
        output=subprocess.check_output([str(exe),*args,"--precision",mode],env=env,text=True,stderr=subprocess.STDOUT)
        elapsed=time.perf_counter()-start
        vals=[float(v) for v in re.findall(r"C(?:ext|sca|abs) \(T-matrix\) = ([^\s]+)",output)]
        assert len(vals)==3 and all(math.isfinite(v) for v in vals),output
        assert f"Backend = {expected}" in output,output
        assert f"T-matrix precision = {mode}" in output,output
        order=re.search(r"Order = (\d+), surface quadrature = (\d+)",output)
        assert order,output
        results.append((vals,order.groups(),elapsed))
    ref=results[0]
    for vals,order,_ in results[1:]:
        assert order==ref[1],(label,order,ref[1])
        assert all(abs(a-b)<=1e-10*max(abs(a),abs(b),1e-30) for a,b in zip(vals,ref[0])),(label,vals,ref[0])
    if label=="capsule sphere":
        # Independent Mie sections for radius .336 um, lambda .63 um, m=3+0.1i.
        mie=[.8837754602141814e-12,.509378630252e-12,.374396829962e-12]
        assert all(abs(a/b-1)<1e-5 for a,b in zip(ref[0],mie)),(ref,mie)
    err=max(abs(a-b) for vals,_,_ in results[1:] for a,b in zip(vals,ref[0]))
    print(f"{label}: N/Q={ref[1]}, sections={ref[0]}, max_abs_diff={err:.3g} m2; times CPU/OMP/GPU="+"/".join(f"{r[2]:.4f}" for r in results)+" s",flush=True)
for backend in ("cpu","openmp","cuda"):
    exe=root/("ifdda_mishchenko_sections_"+backend+suffix)
    invalid=subprocess.run([str(exe),"ellipsoid","630","140","150","280","1.5",".02","27"],env=env,capture_output=True)
    assert invalid.returncode==1
print("PASS: serial/OpenMP/CUDA agreement, independent Mie and invalid input",flush=True)
