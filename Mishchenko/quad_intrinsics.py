import re

def convert_quad_intrinsics(code):
    """Modernize obsolete F77 quad intrinsics for gfortran, preserving kind."""
    # Replace the two constructors with modern generic intrinsics and
    # explicitly request IEEE binary128 to avoid silent down-conversion.
    for old,new in (("QFLOAT","REAL"),("QCMPLX","CMPLX")):
        found=list(re.finditer(r"\b"+old+r"\s*\(",code,flags=re.I))
        for match in reversed(found):
            opening=code.find("(",match.start())
            depth=0
            end=None
            for pos in range(opening,len(code)):
                if code[pos]=="(":depth+=1
                elif code[pos]==")":
                    depth-=1
                    if depth==0:
                        end=pos
                        break
            if end is None:raise RuntimeError("unbalanced quad intrinsic "+old)
            code=(code[:match.start()]+new+code[opening:end]+",KIND=16"+
                  code[end:])
    mapping={"QABS":"ABS","QACOS":"ACOS","QARCOS":"ACOS",
             "QATAN":"ATAN","QCONJG":"CONJG","QCOS":"COS",
             "QCOSH":"COSH","QIMAG":"AIMAG","QSIN":"SIN",
             "QSINH":"SINH","QSQRT":"SQRT","QMAX1":"MAX"}
    for old,new in mapping.items():
        code=re.sub(r"\b"+old+r"\b",new,code,flags=re.I)
    return code.replace("Q10.3","E10.3")
