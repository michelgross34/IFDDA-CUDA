"""Check every IF-DDA executable against the accelerated reference."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("build_dir", type=Path)
parser.add_argument("log_dir", type=Path)
parser.add_argument("--count", type=int, required=True)
args = parser.parse_args()
build = args.build_dir.resolve()
logs = args.log_dir.resolve()
logs.mkdir(parents=True, exist_ok=True)
suffix = ".exe" if os.name == "nt" else ""
executables = sorted(p for p in build.glob("test_main*cuda_*" + suffix)
                     if p.is_file() and p.name.endswith(("double"+suffix, "float"+suffix)))
assert len(executables) == args.count, executables
env = dict(os.environ, OMP_NUM_THREADS="8")
cases = [
    ("cylinder", "180,360,1", "23,39,17",
     (2.659051266e-13, 1.776699783e-13, 8.823514830e-14)),
    ("ellipsoid", "140,140,280", "0,0,0",
     (1.640699264e-13, 1.000376720e-13, 6.403225445e-14)),
]
for exe in executables:
    for geometry, dimensions, angles, expected in cases:
        cwd = logs / exe.stem / geometry
        cwd.mkdir(parents=True, exist_ok=True)
        cmd = [str(exe), "--geometry", geometry, "--geometry-dimensions", dimensions,
               "--geometry-angles", angles, "--illumination", "pwavelinear",
               "--theta", "27", "--phi", "49", "--pp", "1",
               "--epsilon", "2.24,0.3", "--wavelength", "630", "--grid-size", "16",
               "--cross-section", "1", "--solver", "gpbicg"]
        result = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        text = result.stdout.decode("utf-8", errors="replace")
        (cwd / "output.txt").write_text(text, encoding="utf-8")
        assert result.returncode == 0, (exe.name, geometry, result.returncode, cwd)
        assert "T-matrix (Mishchenko, OpenMP + CUDA FP64)" in text, (exe.name, cwd)
        actual = []
        for label in ("Cext", "Csca", "Cabs"):
            match = re.search(label + r"\s*\(T-matrix\)\s*=\s*([0-9.eE+\-]+)", text)
            assert match, (label, cwd)
            actual.append(float(match.group(1)))
        error = max(abs(a-b)/abs(b) for a,b in zip(actual,expected))
        assert error < 2e-6, (exe.name, geometry, actual, error, cwd)
        # C++ and Fortran mains both emit the stable result record.
        match = re.search(r"CDMLIB_RESULT\s+(\d+)", text)
        assert match and int(match.group(1)) == 0, (exe.name, geometry, cwd)
        print(f"{exe.name} {geometry}: max relative difference {error:.3g}", flush=True)
print(f"Validated {len(executables)} executables, {2*len(executables)} IF-DDA runs.", flush=True)
