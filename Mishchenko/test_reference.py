import ctypes as C
import math
import sys
import os
from pathlib import Path
from time import perf_counter

library = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().with_name("ifdda_mishchenko.dll")
if os.name == "nt":
    dll_directory = os.add_dll_directory(str(library.parent))
dll = C.CDLL(str(library))
calculate = dll.ifdda_mishchenko_v1
calculate.argtypes = (
    [C.c_int] + [C.c_double] * 10
    + [C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_double)]
)
calculate.restype = C.c_int


def compute(shape, radius, ratio, longest, index, theta, ep, es, wavelength):
    sections = (C.c_double * 3)()
    order, quadrature, variation = C.c_int(), C.c_int(), C.c_double()
    start = perf_counter()
    status = calculate(
        shape, radius / wavelength, ratio, longest / wavelength, index.real, index.imag,
        theta, ep.real, ep.imag, es.real, es.imag,
        sections, C.byref(order), C.byref(quadrature), C.byref(variation)
    )
    seconds = perf_counter() - start
    return status, [value * wavelength**2 for value in sections], order.value, quadrature.value, variation.value, seconds


def report(name, result):
    status, sections, order, quadrature, variation, seconds = result
    print(f"{name}: status={status}, Cext/Csca/Cabs={sections}, N={order}, "
          f"quadrature={quadrature}, delta={variation:.3g}, {seconds:.2f} s", flush=True)


def verify_cylinder():
    result = compute(-2, (0.75*0.45**2*0.9)**(1/3), 1, math.hypot(0.45,0.45), 2.5+0.1j, 27, 0, 1, 1)
    report("circular cylinder", result)
    # Separate NASA extended-precision run at N=38 and 456 surface nodes.
    expected = (2.823814657383036, 1.7143897609652614, 1.1094248964177698)
    assert result[0] == 0
    assert all(abs(value / reference - 1) < 1e-3 for value, reference in zip(result[1], expected))


if __name__ == "__main__":
    wavelength = 2 * math.pi
    sphere = compute(-1, 5, 1, 5, 5+0j, 27, 1, 0, wavelength)
    report("sphere vs Mie", sphere)
    assert sphere[0] == 0 and abs(sphere[1][1] / 134.7134366055274 - 1) < 1e-3
    assert abs(sphere[1][0] / 134.7134366055274 - 1) < 1e-3
    for m, expected in [(2.5, (171.929366, 91.631025)), (4, (177.286520, 106.334910)),
                        (5, (183.024400, 119.674942)), (6, (172.778973, 113.562568))]:
        result = compute(-1, 5, 0.5, 5*2**(2/3), m+0.1j, 27, 1, 0, wavelength)
        report(f"prolate spheroid m={m}+0.1i", result)
        if result[0] == 0:
            assert all(abs(value / reference - 1) < 1e-3 for value, reference in zip(result[1], expected))
        elif m <= 4:
            raise AssertionError("Expected spheroid reference unavailable")
    verify_cylinder()
