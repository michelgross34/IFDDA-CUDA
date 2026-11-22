// Device kernels ported from cdmlib_CUDA/derivativefield2_array_cuda.cu.
// Workspace allocation, FFT planning and copies are owned by CUDA/cuda_data.
__device__ static CUCOMPLEX derivative_make(CU_REAL real_part, CU_REAL imaginary_part) {
#ifdef USE_SIMP
    return make_cuComplex((float)real_part, (float)imaginary_part);
#else
    return make_cuDoubleComplex((double)real_part, (double)imaginary_part);
#endif
}

// The Green tensor derivatives contain inverse powers through r^-5. Keep
// their per-thread intermediates in double even in the single precision build;
// only the completed tensor is narrowed to the selected CUDA complex type.
struct DerivativeWideComplex {
    double x;
    double y;
};

__device__ static DerivativeWideComplex derivative_wide_make(double real_part,
                                                             double imaginary_part) {
    return {real_part, imaginary_part};
}

__device__ static CUCOMPLEX derivative_narrow(DerivativeWideComplex value) {
#ifdef USE_SIMP
    return make_cuComplex((float)value.x, (float)value.y);
#else
    return make_cuDoubleComplex(value.x, value.y);
#endif
}

__device__ static DerivativeWideComplex derivative_add(DerivativeWideComplex left,
                                                       DerivativeWideComplex right) {
    return derivative_wide_make(left.x + right.x, left.y + right.y);
}

__device__ static DerivativeWideComplex derivative_multiply(DerivativeWideComplex left,
                                                            DerivativeWideComplex right) {
    return derivative_wide_make(left.x * right.x - left.y * right.y,
                                left.x * right.y + left.y * right.x);
}

__device__ static DerivativeWideComplex derivative_scale(DerivativeWideComplex value,
                                                         double factor) {
    return derivative_wide_make(value.x * factor, value.y * factor);
}

//-----------------------------------------------------

__device__ static CUCOMPLEX derivative_add(CUCOMPLEX left, CUCOMPLEX right) {
    return derivative_make(left.x + right.x, left.y + right.y);
}

//-----------------------------------------------------

__device__ static CUCOMPLEX derivative_multiply(CUCOMPLEX left, CUCOMPLEX right) {
    return derivative_make(left.x * right.x - left.y * right.y,
                           left.x * right.y + left.y * right.x);
}

//-----------------------------------------------------

__device__ static CUCOMPLEX derivative_scale(CUCOMPLEX value, CU_REAL factor) {
    return derivative_make(value.x * factor, value.y * factor);
}

//-----------------------------------------------------

__global__ static void build_green_derivative_tensor_kernel(CUCOMPLEX* tensor_xx,
                                                            CUCOMPLEX* tensor_xy,
                                                            CUCOMPLEX* tensor_xz,
                                                            CUCOMPLEX* tensor_yy,
                                                            CUCOMPLEX* tensor_yz,
                                                            CUCOMPLEX* tensor_zz,
                                                            int nx,
                                                            int ny,
                                                            int nz,
                                                            int nx2,
                                                            int ny2,
                                                            int nz2,
                                                            int nxy2,
                                                            int padded_elements,
                                                            int test,
                                                            CU_REAL edge,
                                                            CU_REAL k0) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= padded_elements) {
        return;
    }

    const int plane = nx2 * ny2;
    const int kk = index / plane;
    const int remainder = index - kk * plane;
    const int jj = remainder / nx2;
    const int ii = remainder - jj * nx2;
    const int i = ii > nx ? ii - nx2 : ii;
    const int j = jj > ny ? jj - ny2 : jj;
    const int k = kk > nz ? kk - nz2 : kk;

    const CUCOMPLEX zero = derivative_make(0.0, 0.0);

    if (ii == nx || jj == ny || kk == nz) {
        tensor_xx[index] = zero;
        tensor_xy[index] = zero;
        tensor_xz[index] = zero;
        tensor_yy[index] = zero;
        tensor_yz[index] = zero;
        tensor_zz[index] = zero;
        return;
    }

    const double wide_edge = (double)edge;
    const double wide_k0 = (double)k0;
    const double x = (double)i * wide_edge;
    const double y = (double)j * wide_edge;
    const double z = (double)k * wide_edge;
    const double a = sqrt(x * x + y * y);
    const double a2 = a * a;
    const double rab = sqrt(x * x + y * y + z * z);

    if (rab == 0.0) {
        tensor_xx[index] = zero;
        tensor_xy[index] = zero;
        tensor_xz[index] = zero;
        tensor_yy[index] = zero;
        tensor_yz[index] = zero;
        tensor_zz[index] = zero;
        return;
    }

    const double rab2 = rab * rab;
    const double rab3 = rab2 * rab;
    const double rab4 = rab3 * rab;
    const double rab5 = rab4 * rab;
    const DerivativeWideComplex const1 = derivative_wide_make(1.0 / rab3, -wide_k0 / rab2);
    const DerivativeWideComplex const2 = derivative_wide_make(wide_k0 * wide_k0 / rab, 0.0);
    const DerivativeWideComplex c13c2 =
        derivative_add(derivative_scale(const1, 3.0), derivative_scale(const2, -1.0));
    const DerivativeWideComplex dconst1 = derivative_wide_make(-3.0 / rab5, 2.0 * wide_k0 / rab4);
    const DerivativeWideComplex dconst2 = derivative_wide_make(-wide_k0 * wide_k0 / rab3, 0.0);
    const DerivativeWideComplex dc13c2 =
        derivative_add(derivative_scale(dconst1, 3.0), derivative_scale(dconst2, -1.0));
    const DerivativeWideComplex ddd = derivative_scale(
        derivative_add(derivative_scale(dc13c2, rab2), derivative_scale(c13c2, -2.0)), 1.0 / rab4);
    const DerivativeWideComplex phase =
        derivative_wide_make(cos(wide_k0 * rab), sin(wide_k0 * rab));
    const DerivativeWideComplex base = derivative_add(const2, derivative_scale(const1, -1.0));
    const DerivativeWideComplex txx =
        derivative_multiply(derivative_add(base, derivative_scale(c13c2, a2 / rab2 / 2.0)), phase);
    const DerivativeWideComplex tzz =
        derivative_multiply(derivative_add(base, derivative_scale(c13c2, z * z / rab2)), phase);
    const DerivativeWideComplex txy =
        derivative_scale(derivative_multiply(c13c2, phase), -a2 / rab2 / 2.0);
    const DerivativeWideComplex txz =
        derivative_scale(derivative_multiply(c13c2, phase), a * z / rab2);
    const DerivativeWideComplex dbase = derivative_add(dconst2, derivative_scale(dconst1, -1.0));
    const DerivativeWideComplex imag = derivative_wide_make(0.0, 1.0);
    const DerivativeWideComplex txxdz =
        derivative_add(derivative_scale(derivative_multiply(imag, txx), z * wide_k0 / rab),
                       derivative_multiply(phase,
                                           derivative_add(derivative_scale(dbase, z),
                                                          derivative_scale(ddd, a2 / 2.0 * z))));
    const DerivativeWideComplex tzzdz = derivative_add(
        derivative_scale(derivative_multiply(imag, tzz), z * wide_k0 / rab),
        derivative_multiply(phase,
                            derivative_add(derivative_add(derivative_scale(dbase, z),
                                                          derivative_scale(c13c2, 2.0 * z / rab2)),
                                           derivative_scale(ddd, z * z * z))));
    const DerivativeWideComplex txydz =
        derivative_add(derivative_scale(derivative_multiply(imag, txy), z * wide_k0 / rab),
                       derivative_scale(derivative_multiply(phase, ddd), -a2 / 2.0 * z));
    const DerivativeWideComplex txzdz = derivative_add(
        derivative_scale(derivative_multiply(imag, txz), z * wide_k0 / rab),
        derivative_scale(derivative_multiply(phase,
                                             derivative_add(derivative_scale(c13c2, 1.0 / rab2),
                                                            derivative_scale(ddd, z * z))),
                         a));
    const DerivativeWideComplex txxda = derivative_add(
        derivative_scale(derivative_multiply(imag, txx), a * wide_k0 / rab),
        derivative_multiply(phase,
                            derivative_add(derivative_add(derivative_scale(dbase, a),
                                                          derivative_scale(c13c2, a / rab2)),
                                           derivative_scale(ddd, a2 / 2.0 * a))));
    const DerivativeWideComplex tzzda = derivative_add(
        derivative_scale(derivative_multiply(imag, tzz), a * wide_k0 / rab),
        derivative_multiply(
            phase, derivative_add(derivative_scale(dbase, a), derivative_scale(ddd, z * z * a))));
    const DerivativeWideComplex txyda =
        derivative_add(derivative_scale(derivative_multiply(imag, txy), a * wide_k0 / rab),
                       derivative_multiply(phase,
                                           derivative_add(derivative_scale(c13c2, -a / rab2),
                                                          derivative_scale(ddd, -a2 / 2.0 * a))));
    const DerivativeWideComplex txzda =
        derivative_add(derivative_scale(derivative_multiply(imag, txz), a * wide_k0 / rab),
                       derivative_multiply(phase,
                                           derivative_add(derivative_scale(c13c2, z / rab2),
                                                          derivative_scale(ddd, a * z * a))));

    double s = 0.0;
    double c = 0.0;
    double aa = a;

    if (a <= 1.0e-30) {
        if (test == 1) {
            s = 1.0;
            c = 0.0;
        } else if (test == 2) {
            s = 0.0;
            c = 1.0;
        }
#ifdef USE_SIMP
        aa = (double)FLT_MAX / 100.0;
#else
        aa = DBL_MAX / 100.0;
#endif
    } else {
        s = x / a;
        c = y / a;
    }

    const double s2 = 2.0 * s * c;
    const double c2 = c * c - s * s;
    DerivativeWideComplex values[6];

    if (test == 1) {
        values[0] = derivative_add(
            derivative_add(derivative_scale(txxda, s), derivative_scale(txy, -2.0 * c * s2 / aa)),
            derivative_scale(txyda, c2 * s));
        values[1] = derivative_add(derivative_scale(txy, -2.0 * c * c2 / aa),
                                   derivative_scale(txyda, -s2 * s));
        values[2] =
            derivative_add(derivative_scale(txz, c * c / aa), derivative_scale(txzda, s * s));
        values[3] = derivative_add(
            derivative_add(derivative_scale(txxda, s), derivative_scale(txy, 2.0 * c * s2 / aa)),
            derivative_scale(txyda, -c2 * s));
        values[4] =
            derivative_add(derivative_scale(txz, -s * c / aa), derivative_scale(txzda, c * s));
        values[5] = derivative_scale(tzzda, s);
    } else if (test == 2) {
        values[0] = derivative_add(
            derivative_add(derivative_scale(txxda, c), derivative_scale(txy, 2.0 * s * s2 / aa)),
            derivative_scale(txyda, c2 * c));
        values[1] = derivative_add(derivative_scale(txy, 2.0 * s * c2 / aa),
                                   derivative_scale(txyda, -s2 * c));
        values[2] =
            derivative_add(derivative_scale(txz, -s * c / aa), derivative_scale(txzda, s * c));
        values[3] = derivative_add(
            derivative_add(derivative_scale(txxda, c), derivative_scale(txy, -2.0 * s * s2 / aa)),
            derivative_scale(txyda, -c2 * c));
        values[4] =
            derivative_add(derivative_scale(txz, s * s / aa), derivative_scale(txzda, c * c));
        values[5] = derivative_scale(tzzda, c);
    } else {
        values[0] = derivative_add(txxdz, derivative_scale(txydz, c2));
        values[1] = derivative_scale(txydz, -s2);
        values[2] = derivative_scale(txzdz, s);
        values[3] = derivative_add(txxdz, derivative_scale(txydz, -c2));
        values[4] = derivative_scale(txzdz, c);
        values[5] = tzzdz;
    }

    tensor_xx[index] = derivative_narrow(values[0]);
    tensor_xy[index] = derivative_narrow(values[1]);
    tensor_xz[index] = derivative_narrow(values[2]);
    tensor_yy[index] = derivative_narrow(values[3]);
    tensor_yz[index] = derivative_narrow(values[4]);
    tensor_zz[index] = derivative_narrow(values[5]);
    (void)nz;
    (void)nxy2;
}

//-----------------------------------------------------

//-----------------------------------------------------

__global__ static void derivative_spectral_product_kernel(const CUCOMPLEX* vector_x,
                                                          const CUCOMPLEX* vector_y,
                                                          const CUCOMPLEX* vector_z,
                                                          CUCOMPLEX* tensor_xx,
                                                          const CUCOMPLEX* tensor_xy,
                                                          const CUCOMPLEX* tensor_xz,
                                                          CUCOMPLEX* tensor_yy,
                                                          const CUCOMPLEX* tensor_yz,
                                                          CUCOMPLEX* tensor_zz,
                                                          int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    const CUCOMPLEX x = vector_x[index];
    const CUCOMPLEX y = vector_y[index];
    const CUCOMPLEX z = vector_z[index];

    tensor_xx[index] = derivative_add(derivative_add(derivative_multiply(tensor_xx[index], x),
                                                     derivative_multiply(tensor_xy[index], y)),
                                      derivative_multiply(tensor_xz[index], z));
    tensor_yy[index] = derivative_add(derivative_add(derivative_multiply(tensor_xy[index], x),
                                                     derivative_multiply(tensor_yy[index], y)),
                                      derivative_multiply(tensor_yz[index], z));
    tensor_zz[index] = derivative_add(derivative_add(derivative_multiply(tensor_xz[index], x),
                                                     derivative_multiply(tensor_yz[index], y)),
                                      derivative_multiply(tensor_zz[index], z));
}

//-----------------------------------------------------

__global__ static void extract_derivative_kernel(const CUCOMPLEX* derivative_x,
                                                 const CUCOMPLEX* derivative_y,
                                                 const CUCOMPLEX* derivative_z,
                                                 CUCOMPLEX* output,
                                                 int nx,
                                                 int ny,
                                                 int nz,
                                                 int nx2,
                                                 int nxy2,
                                                 int count,
                                                 CU_REAL scale) {
    const int physical = blockIdx.x * blockDim.x + threadIdx.x;

    if (physical >= count) {
        return;
    }

    const int plane = nx * ny;
    const int k = physical / plane;
    const int remainder = physical - k * plane;
    const int j = remainder / nx;
    const int i = remainder - j * nx;
    const int source = i + nx2 * j + nxy2 * k;
    const int destination = 3 * physical;

#ifdef USE_SIMP
    const float inverse_scale = 1.0f / (float)scale;
#else
    const double inverse_scale = 1.0 / (double)scale;
#endif

    output[destination] = derivative_x[source];
    output[destination + 1] = derivative_y[source];
    output[destination + 2] = derivative_z[source];
    output[destination].x *= inverse_scale;
    output[destination].y *= inverse_scale;
    output[destination + 1].x *= inverse_scale;
    output[destination + 1].y *= inverse_scale;
    output[destination + 2].x *= inverse_scale;
    output[destination + 2].y *= inverse_scale;
    (void)ny;
    (void)nz;
}

//-----------------------------------------------------

//-----------------------------------------------------
