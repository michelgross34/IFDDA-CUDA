#include <atomic>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cub/device/device_scan.cuh>

#include "kernel.h"
#include "force_green_derivative_kernels.cuh"
#include "gaussian_incident_kernel.cuh"

namespace {

constexpr int THREADS_PER_BLOCK = 256;
ifdda_host_download_callback host_download_callback = nullptr;

// CUB scans the CU_INT values directly; no int* alias crosses the wrapper.
struct CuIntSum {
    __host__ __device__ CU_INT operator()(const CU_INT& lhs, const CU_INT& rhs) const {
        return CU_INT{lhs.dat + rhs.dat};
    }
};

//-----------------------------------------------------------
void trace_beampropagation_layer(const CUCOMPLEX *device_images, size_t plane, int layer,
                                 const char *stage) {
    if (!std::getenv("IFDDA_BPM_TRACE"))
        return;
    std::vector<CUCOMPLEX> host_images(size_t(3) * plane);
    const cudaError_t status = host_download_callback
                                   ? host_download_callback(host_images.data(), device_images,
                                                            host_images.size() * sizeof(CUCOMPLEX))
                                   : cudaErrorNotReady;
    if (status != cudaSuccess) {
        std::fprintf(stderr, "[BPM trace CUDA] layer %d %s copy failed: %s\n", layer, stage,
                     cudaGetErrorString(status));
        return;
    }
    double sums[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < plane; ++i) {
        for (int component = 0; component < 3; ++component) {
            const CUCOMPLEX value = host_images[size_t(component) * plane + i];
            sums[component] += std::hypot(double(value.x), double(value.y));
        }
    }
    std::fprintf(stderr, "BPM_TRACE %d %s %.17g %.17g %.17g\n", layer, stage, sums[0], sums[1],
                 sums[2]);
}

//-----------------------------------------------------------
__global__ void zero_vectors_kernel(CUCOMPLEX *vectors, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        vectors[i] = CUFFT_ZERO;
}

//-----------------------------------------------------------
__device__ CU_REAL mie_complex_norm_squared(CUCOMPLEX value) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CU_REAL real_part = cuCrealf(value);
    const CU_REAL imag_part = cuCimagf(value);
#else
    const CU_REAL real_part = cuCreal(value);
    const CU_REAL imag_part = cuCimag(value);
#endif
    return real_part * real_part + imag_part * imag_part;
}

//-----------------------------------------------------------
__global__ void mie_efficiencies_kernel(CUCOMPLEX *recurrence,
    CU_REAL relative_index_real, CU_REAL relative_index_imag,
    CU_REAL x, CU_REAL radius, CU_REAL *outputs, int max_order) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX relative = make_cuFloatComplex(relative_index_real,
                                                    relative_index_imag);
#else
    const CUCOMPLEX relative = make_cuDoubleComplex(relative_index_real,
                                                     relative_index_imag);
#endif
    const CUCOMPLEX y = CUFFT_MUL(relative, CUFFT_REAL(x));
    const CU_REAL y_magnitude = sqrt(mie_complex_norm_squared(y));
    const CU_REAL x_stop = x + CU_REAL(4) * pow(x, CU_REAL(0.3333)) + CU_REAL(2);
    const int nmx = int(fmax(x_stop, y_magnitude)) + 15;
    const int nstop = int(x_stop);
    if (nmx < 2 || nmx > max_order || nstop < 1) {
        outputs[0] = outputs[1] = outputs[2] = outputs[3] = CU_REAL(0);
        outputs[4] = CU_REAL(1);
        return;
    }
    outputs[4] = CU_REAL(0);
    const CUCOMPLEX inverse_relative = CUFFT_DIV(CUFFT_REAL(CU_REAL(1)), relative);
    recurrence[nmx - 1] = CUFFT_ZERO;
    for (int n = 1; n < nmx; ++n) {
        const CU_REAL rn = CU_REAL(nmx - n + 1);
        const CUCOMPLEX ratio = CUFFT_DIV(CUFFT_REAL(rn), y);
        recurrence[nmx - n - 1] = CUFFT_SUB(
            ratio, CUFFT_DIV(CUFFT_REAL(CU_REAL(1)), CUFFT_ADD(recurrence[nmx - n], ratio)));
    }
    CU_REAL psi0 = cos(x);
    CU_REAL psi1 = sin(x);
    CU_REAL chi0 = -sin(x);
    CU_REAL chi1 = cos(x);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    CUCOMPLEX xi1 = make_cuFloatComplex(psi1, -chi1);
#else
    CUCOMPLEX xi1 = make_cuDoubleComplex(psi1, -chi1);
#endif
    CUCOMPLEX an = CUFFT_ZERO;
    CUCOMPLEX bn = CUFFT_ZERO;
    CUCOMPLEX an_previous = CUFFT_ZERO;
    CUCOMPLEX bn_previous = CUFFT_ZERO;
    CUCOMPLEX s1_forward = CUFFT_ZERO;
    CU_REAL qsca_sum = CU_REAL(0);
    CU_REAL gsca_sum = CU_REAL(0);
    CU_REAL pi_previous = CU_REAL(0);
    CU_REAL pi_current = CU_REAL(1);
    for (int n = 1; n <= nstop; ++n) {
        const CU_REAL order = CU_REAL(n);
        const CU_REAL factor = (CU_REAL(2) * order + CU_REAL(1)) /
            (order * (order + CU_REAL(1)));
        const CU_REAL psi = (CU_REAL(2) * order - CU_REAL(1)) * psi1 / x - psi0;
        const CU_REAL chi = (CU_REAL(2) * order - CU_REAL(1)) * chi1 / x - chi0;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CUCOMPLEX xi = make_cuFloatComplex(psi, -chi);
#else
        const CUCOMPLEX xi = make_cuDoubleComplex(psi, -chi);
#endif
        if (n > 1) {
            an_previous = an;
            bn_previous = bn;
        }
        const CUCOMPLEX d = recurrence[n - 1];
        const CUCOMPLEX a_term = CUFFT_ADD(CUFFT_MUL(d, inverse_relative),
                                           CUFFT_REAL(order / x));
        const CUCOMPLEX b_term = CUFFT_ADD(CUFFT_MUL(relative, d),
                                           CUFFT_REAL(order / x));
        an = CUFFT_DIV(CUFFT_SUB(CUFFT_MUL(a_term, CUFFT_REAL(psi)),
                                 CUFFT_REAL(psi1)),
                       CUFFT_SUB(CUFFT_MUL(a_term, xi), xi1));
        bn = CUFFT_DIV(CUFFT_SUB(CUFFT_MUL(b_term, CUFFT_REAL(psi)),
                                 CUFFT_REAL(psi1)),
                       CUFFT_SUB(CUFFT_MUL(b_term, xi), xi1));
        qsca_sum += (CU_REAL(2) * order + CU_REAL(1)) *
            (mie_complex_norm_squared(an) + mie_complex_norm_squared(bn));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CU_REAL an_real = cuCrealf(an), an_imag = cuCimagf(an);
        const CU_REAL bn_real = cuCrealf(bn), bn_imag = cuCimagf(bn);
        const CU_REAL an_prev_real = cuCrealf(an_previous), an_prev_imag = cuCimagf(an_previous);
        const CU_REAL bn_prev_real = cuCrealf(bn_previous), bn_prev_imag = cuCimagf(bn_previous);
#else
        const CU_REAL an_real = cuCreal(an), an_imag = cuCimag(an);
        const CU_REAL bn_real = cuCreal(bn), bn_imag = cuCimag(bn);
        const CU_REAL an_prev_real = cuCreal(an_previous), an_prev_imag = cuCimag(an_previous);
        const CU_REAL bn_prev_real = cuCreal(bn_previous), bn_prev_imag = cuCimag(bn_previous);
#endif
        gsca_sum += factor * (an_real * bn_real + an_imag * bn_imag);
        if (n > 1)
            gsca_sum += ((order - CU_REAL(1)) * (order + CU_REAL(1)) / order) *
                (an_prev_real * an_real + an_prev_imag * an_imag +
                 bn_prev_real * bn_real + bn_prev_imag * bn_imag);
        s1_forward = CUFFT_ADD(s1_forward,
            CUFFT_MUL(CUFFT_REAL(factor * pi_current), an));
        const CU_REAL tau = order * pi_current - (order + CU_REAL(1)) * pi_previous;
        s1_forward = CUFFT_ADD(s1_forward,
            CUFFT_MUL(CUFFT_REAL(factor * tau), bn));
        const CU_REAL pi_next = ((CU_REAL(2) * order + CU_REAL(1)) * pi_current -
                                 (order + CU_REAL(1)) * pi_previous) / order;
        pi_previous = pi_current;
        pi_current = pi_next;
        psi0 = psi1;
        psi1 = psi;
        chi0 = chi1;
        chi1 = chi;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        xi1 = make_cuFloatComplex(psi1, -chi1);
#else
        xi1 = make_cuDoubleComplex(psi1, -chi1);
#endif
    }
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CU_REAL qext = (CU_REAL(4) / (x * x)) * cuCrealf(s1_forward);
#else
    const CU_REAL qext = (CU_REAL(4) / (x * x)) * cuCreal(s1_forward);
#endif
    const CU_REAL qsca = (CU_REAL(2) / (x * x)) * qsca_sum;
    const CU_REAL area = pi * radius * radius;
    outputs[0] = area * qext;
    outputs[1] = area * (qext - qsca);
    outputs[2] = area * qsca;
    outputs[3] = qsca_sum != CU_REAL(0) ? CU_REAL(2) * gsca_sum / qsca_sum : CU_REAL(0);
}

//-----------------------------------------------------------
__global__ void cube_geometry_kernel(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, CU_INT *dipole_map,
                                     int subdivisions, CU_REAL side, CU_REAL cell_size,
                                     CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
                                     int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int k = flat % subdivisions + 1;
    const int j = (flat / subdivisions) % subdivisions + 1;
    const int i = flat / (subdivisions * subdivisions) + 1;
    const CU_REAL center = (side + cell_size) / CU_REAL(2);
    xs[flat] = CU_REAL(k) * cell_size - center + center_x;
    ys[flat] = CU_REAL(j) * cell_size - center + center_y;
    zs[flat] = CU_REAL(i) * cell_size - center + center_z;
    dipole_map[flat].dat = flat + 1;
}

//-----------------------------------------------------------
__global__ void cube_dipole_map_kernel(CU_INT* dipole_map, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat < count)
        dipole_map[flat].dat = 1;
}

//-----------------------------------------------------------
__global__ void widefield_build_grid_kernel(
    CU_REAL* grid_x, CU_REAL* grid_y, CU_REAL* grid_z,
    int nx, int ny, int nz, int nxmp, int nymp, int nzmp,
    CU_REAL xmin, CU_REAL ymin, CU_REAL zmin, CU_REAL cell_size,
    int grid_count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= grid_count)
        return;
    const int i = flat % nx;
    const int j = (flat / nx) % ny;
    const int k = flat / (nx * ny);
    grid_x[flat] = xmin + CU_REAL(i - nxmp) * cell_size;
    grid_y[flat] = ymin + CU_REAL(j - nymp) * cell_size;
    grid_z[flat] = zmin + CU_REAL(k - nzmp) * cell_size;
}

//-----------------------------------------------------------
__global__ void widefield_map_dipoles_kernel(
    const CU_REAL* dipole_x, const CU_REAL* dipole_y, const CU_REAL* dipole_z,
    const CU_REAL* grid_x, const CU_REAL* grid_y, const CU_REAL* grid_z,
    CU_INT* dipole_map, int dipole_count, int nx, int ny, int nz,
    CU_REAL cell_size, CU_REAL tolerance) {
    const int dipole = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (dipole >= dipole_count)
        return;
    const int i = int(floor((dipole_x[dipole] - grid_x[0]) / cell_size + CU_REAL(0.5)));
    const int j = int(floor((dipole_y[dipole] - grid_y[0]) / cell_size + CU_REAL(0.5)));
    const int k = int(floor((dipole_z[dipole] - grid_z[0]) / cell_size + CU_REAL(0.5)));
    if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz)
        return;
    const int grid = i + nx * (j + ny * k);
    if (fabs(grid_x[grid] - dipole_x[dipole]) <= tolerance &&
        fabs(grid_y[grid] - dipole_y[dipole]) <= tolerance &&
        fabs(grid_z[grid] - dipole_z[dipole]) <= tolerance)
        atomicCAS(&dipole_map[grid].dat, -1, dipole);
}

//-----------------------------------------------------------
__global__ void widefield_scatter_dipoles_kernel(
    const CU_INT* grid_to_dipole, int grid_count, int dipole_count, int isotropy,
    int preserve_incident_field, const CUCOMPLEX* dipole_scattered,
    const CUCOMPLEX* dipole_incident, const CUCOMPLEX* dipole_epsilon,
    CUCOMPLEX* xi, CUCOMPLEX* xr, CUCOMPLEX* polarization) {
    const int grid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (grid >= grid_count)
        return;
    const int target_field = 3 * grid;
    xi[target_field] = xi[target_field + 1] = xi[target_field + 2] = CUFFT_ZERO;

    const int components = isotropy * isotropy;
    for (int tensor = 0; tensor < components; ++tensor) {
        const int row = tensor % isotropy;
        const int column = tensor / isotropy;
        polarization[grid + tensor * grid_count] =
            row == column ? CUFFT_REAL(CU_REAL(1)) : CUFFT_ZERO;
    }

    const int dipole = grid_to_dipole[grid].dat;
    if (dipole < 0 || dipole >= dipole_count)
        return;
    const int source_field = 3 * dipole;
    xi[target_field] = CUFFT_MUL(CUFFT_REAL(CU_REAL(-1)), dipole_scattered[source_field]);
    xi[target_field + 1] = CUFFT_MUL(CUFFT_REAL(CU_REAL(-1)), dipole_scattered[source_field + 1]);
    xi[target_field + 2] = CUFFT_MUL(CUFFT_REAL(CU_REAL(-1)), dipole_scattered[source_field + 2]);
    if (!preserve_incident_field) {
        xr[target_field] = dipole_incident[source_field];
        xr[target_field + 1] = dipole_incident[source_field + 1];
        xr[target_field + 2] = dipole_incident[source_field + 2];
    }

    for (int tensor = 0; tensor < components; ++tensor)
        polarization[grid + tensor * grid_count] =
            dipole_epsilon[dipole + tensor * dipole_count];
}

//-----------------------------------------------------------
__global__ void widefield_pack_dipole_epsilon_kernel(
    const CUCOMPLEX* fortran_epsilon, CUCOMPLEX* packed_epsilon,
    int dipole_count, int nmax, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int dipole = flat % dipole_count;
    const int tensor = flat / dipole_count;
    packed_epsilon[flat] = fortran_epsilon[dipole + nmax * tensor];
}

//-----------------------------------------------------------
__global__ void widefield_restore_material_local_field_kernel(
    const CU_INT* grid_to_dipole, int grid_count, int dipole_count, int isotropy,
    const CUCOMPLEX* dipole_epsilon, const CUCOMPLEX* dipole_local_field,
    CUCOMPLEX* grid_field) {
    const int grid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (grid >= grid_count)
        return;
    const int dipole = grid_to_dipole[grid].dat;
    if (dipole < 0 || dipole >= dipole_count)
        return;

    CU_REAL contrast = CU_REAL(0);
    if (isotropy == 1) {
        const CUCOMPLEX epsilon = dipole_epsilon[dipole];
        contrast = hypot(epsilon.x - CU_REAL(1), epsilon.y);
    } else {
        for (int column = 0; column < 3; ++column)
            for (int row = 0; row < 3; ++row) {
                const CUCOMPLEX epsilon = dipole_epsilon[
                    dipole + (row + column * 3) * dipole_count];
                const CU_REAL real_part = epsilon.x - (row == column ? CU_REAL(1) : CU_REAL(0));
                contrast += hypot(real_part, epsilon.y);
            }
    }
    if (contrast < CU_REAL(1.0e-8))
        return;

    for (int component = 0; component < 3; ++component)
        grid_field[3 * grid + component] = dipole_local_field[3 * dipole + component];
}

//-----------------------------------------------------------
__global__ void widefield_incident_magnitude_kernel(
    const CUCOMPLEX* grid_field, CU_REAL* magnitude, int grid_count) {
    const int grid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (grid >= grid_count)
        return;
    CU_REAL squared = CU_REAL(0);
    for (int component = 0; component < 3; ++component) {
        const CUCOMPLEX value = grid_field[3 * grid + component];
        squared += value.x * value.x + value.y * value.y;
    }
    magnitude[grid] = sqrt(squared);
}

//-----------------------------------------------------------
__global__ void initialize_polarizability_arrays_kernel(
    CU_INT*dipole_map, CUCOMPLEX *polarizability, CUCOMPLEX *epsilon,
    int dipole_count, int isotropy) {
    const int dipole = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (dipole >= dipole_count)
        return;
    // A null map is intentional when the geometry map is already resident.
    // In that case initialize only the compact material tensors.
    if (dipole_map)
        dipole_map[dipole].dat = 0;
    const int tensor_components = isotropy == 1 ? 1 : 9;
    for (int component = 0; component < tensor_components; ++component) {
        const int index = dipole + dipole_count * component;
        polarizability[index] = CUFFT_ZERO;
        epsilon[index] = isotropy == 1
            ? CUFFT_REAL(CU_REAL(1))
            : CUFFT_REAL((component % 4) == 0 ? CU_REAL(1) : CU_REAL(0));
    }
}

//-----------------------------------------------------------
// CPU creates tabdip in stable Fortran order (1-based, 0 for empty compact
// cells). All material assignment is performed on the GPU.
__global__ void sphere_material_from_cpu_tabdip_kernel(
    const CU_INT* tabdip, const CU_INT* inside_map,
    CUCOMPLEX* polarizability, CUCOMPLEX* epsilon,
    const CUCOMPLEX* tensor_alpha, const CUCOMPLEX* tensor_epsilon,
    CUCOMPLEX scalar_alpha, CUCOMPLEX scalar_epsilon,
    int grid_count, int nmax, int compact_mesh, int anisotropic) {
    const int cell = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= grid_count || !inside_map[cell].dat)
        return;

    const int destination = compact_mesh ? tabdip[cell].dat - 1 : cell;
    if (destination < 0 || destination >= nmax)
        return;

    if (!anisotropic) {
        polarizability[destination] = scalar_alpha;
        epsilon[destination] = scalar_epsilon;
    } else {
        for (int component = 0; component < 9; ++component) {
            const size_t index = size_t(destination) + size_t(nmax) * size_t(component);
            polarizability[index] = tensor_alpha[component];
            epsilon[index] = tensor_epsilon[component];
        }
    }
}

//-----------------------------------------------------------
// Labeled geometry: CPU supplies a stable 1-based tabdip plus a material ID
// for each destination dipole (0 means vacuum). Device performs every
// per-dipole scalar/tensor assignment from small per-material lookup tables.
__global__ void labeled_material_from_cpu_tabdip_kernel(
    const CU_INT* tabdip, const CU_INT* material_labels,
    CUCOMPLEX* polarizability, CUCOMPLEX* epsilon,
    const CUCOMPLEX* label_alpha, const CUCOMPLEX* label_epsilon,
    int grid_count, int nmax, int label_count, int anisotropic) {
    const int cell = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= grid_count)
        return;
    const int destination = tabdip[cell].dat - 1;
    if (destination < 0 || destination >= nmax)
        return;
    const int material_id = material_labels[destination].dat - 1;
    if (material_id < 0 || material_id >= label_count)
        return;
    if (!anisotropic) {
        polarizability[destination] = label_alpha[material_id];
        epsilon[destination] = label_epsilon[material_id];
    } else {
        for (int component = 0; component < 9; ++component) {
            const size_t destination_index = size_t(destination) + size_t(nmax) * size_t(component);
            const size_t lookup_index = size_t(material_id) * 9 + size_t(component);
            polarizability[destination_index] = label_alpha[lookup_index];
            epsilon[destination_index] = label_epsilon[lookup_index];
        }
    }
}

//-----------------------------------------------------------
// Inhomogeneous materials already have per-cell epsilon and alpha on device.
// The CPU constructs the final Fortran compact/full tabdip; CUDA then
// scatters these existing device values to the final dipole order.
__global__ void scatter_cell_material_from_cpu_tabdip_kernel(
    const CU_INT* tabdip, const CU_INT* inside_map,
    const CUCOMPLEX* cell_epsilon, const CUCOMPLEX* cell_alpha,
    CUCOMPLEX* output_epsilon, CUCOMPLEX* output_alpha,
    int grid_count, int nmax) {
    const int cell = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= grid_count || !inside_map[cell].dat)
        return;
    const int destination = tabdip[cell].dat - 1;
    if (destination < 0 || destination >= nmax)
        return;
    output_epsilon[destination] = cell_epsilon[cell];
    output_alpha[destination] = cell_alpha[cell];
}

//-----------------------------------------------------------
__device__ CUCOMPLEX scalar_polarizability_sqrt(CUCOMPLEX value) {
    const CU_REAL magnitude = hypot(value.x, value.y);
    const CU_REAL real_part = sqrt((magnitude + value.x) / CU_REAL(2));
    CU_REAL imag_part = sqrt(fmax(CU_REAL(0), (magnitude - value.x) / CU_REAL(2)));
    if (value.y < CU_REAL(0))
        imag_part = -imag_part;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real_part, imag_part);
#else
    return make_cuDoubleComplex(real_part, imag_part);
#endif
}

//-----------------------------------------------------------
__device__ CUCOMPLEX scalar_polarizability_exp(CUCOMPLEX value) {
    const CU_REAL scale = exp(value.x);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(scale * cos(value.y), scale * sin(value.y));
#else
    return make_cuDoubleComplex(scale * cos(value.y), scale * sin(value.y));
#endif
}

//-----------------------------------------------------------
__device__ CUCOMPLEX scalar_polarizability_log(CUCOMPLEX value) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(log(hypot(value.x, value.y)), atan2(value.y, value.x));
#else
    return make_cuDoubleComplex(log(hypot(value.x, value.y)), atan2(value.y, value.x));
#endif
}

//-----------------------------------------------------------
__device__ CUCOMPLEX scalar_polarizability_compute(
    CU_REAL radius_input, CUCOMPLEX epsilon, CUCOMPLEX epsilon0, CU_REAL k0,
    int discretized, int method, CU_REAL* sldr) {
    const CU_REAL pi_value = CU_REAL(3.1415926535897932384626433832795L);
    const CU_REAL one_third = CU_REAL(1) / CU_REAL(3);
    CU_REAL radius = radius_input;
    CU_REAL maille = radius_input;
    if (discretized == 1)
        radius *= pow(CU_REAL(0.75) / pi_value, one_third);
    const CU_REAL volume = radius * radius * radius;
    const CUCOMPLEX index = scalar_polarizability_sqrt(epsilon0);
    const CUCOMPLEX ratio = CUFFT_DIV(
        CUFFT_SUB(epsilon, epsilon0),
        CUFFT_ADD(epsilon, CUFFT_MUL(CUFFT_REAL(CU_REAL(2)), epsilon0)));
    const CUCOMPLEX pola = CUFFT_MUL(ratio, CUFFT_REAL(volume));
    const CUCOMPLEX index2 = CUFFT_MUL(index, index);
    const CUCOMPLEX index3 = CUFFT_MUL(index2, index);
    const CUCOMPLEX k03 = CUFFT_MUL(CUFFT_REAL(k0 * k0 * k0), index3);
    const CUCOMPLEX imaginary =
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        make_cuFloatComplex(0.0f, 1.0f);
#else
        make_cuDoubleComplex(0.0, 1.0);
#endif
    CUCOMPLEX output = CUFFT_ZERO;
    if (method == 1) {
        output = CUFFT_MUL(epsilon0, pola);
    } else if (method == 2 || method == 3) {
        const CUCOMPLEX correction = CUFFT_MUL(
            CUFFT_REAL(CU_REAL(2) / CU_REAL(3)),
            CUFFT_MUL(imaginary, CUFFT_MUL(k03, pola)));
        output = CUFFT_DIV(CUFFT_MUL(epsilon0, pola),
                           CUFFT_SUB(CUFFT_REAL(CU_REAL(1)), correction));
    } else if (method == 4) {
        CUCOMPLEX denominator = CUFFT_SUB(
            CUFFT_SUB(CUFFT_REAL(CU_REAL(1)), CUFFT_MUL(
                CUFFT_REAL(CU_REAL(2) / CU_REAL(3)), CUFFT_MUL(imaginary,
                CUFFT_MUL(k03, pola)))),
            CUFFT_DIV(CUFFT_MUL(CUFFT_REAL(k0 * k0),
                CUFFT_MUL(pola, index2)), CUFFT_REAL(radius)));
        output = CUFFT_DIV(pola, denominator);
    } else if (method == 5) {
        const CUCOMPLEX phase = CUFFT_MUL(imaginary, CUFFT_MUL(
            CUFFT_REAL(k0), CUFFT_MUL(index, CUFFT_REAL(radius))));
        const CUCOMPLEX term = CUFFT_SUB(
            CUFFT_MUL(CUFFT_SUB(CUFFT_REAL(CU_REAL(1)), phase),
                      scalar_polarizability_exp(phase)),
            CUFFT_REAL(CU_REAL(1)));
        const CUCOMPLEX denominator = CUFFT_SUB(
            CUFFT_REAL(CU_REAL(1)), CUFFT_MUL(
                CUFFT_REAL(CU_REAL(2)), CUFFT_MUL(ratio, term)));
        output = CUFFT_DIV(pola, denominator);
    } else if (method == 6 || method == 7) {
        const CU_REAL b1 = CU_REAL(-1.8915316529870796511106114030718259L);
        const CU_REAL b2 = CU_REAL(0.16484691508771947306079362778185226L);
        const CU_REAL b3 = CU_REAL(-1.7700004019321371908592738404451742L);
        if (method == 6)
            *sldr = CU_REAL(0.2);
        const CUCOMPLEX eps_ratio = CUFFT_DIV(epsilon, epsilon0);
        const CUCOMPLEX inner = CUFFT_ADD(
            CUFFT_ADD(CUFFT_REAL(b1), CUFFT_MUL(CUFFT_REAL(b2), eps_ratio)),
            CUFFT_MUL(CUFFT_REAL(b3 * *sldr), eps_ratio));
        const CUCOMPLEX correction = CUFFT_SUB(
            CUFFT_MUL(CUFFT_MUL(inner, CUFFT_DIV(
                CUFFT_REAL(k0 * k0), CUFFT_REAL(maille))), index2),
            CUFFT_MUL(CUFFT_REAL(CU_REAL(2) / CU_REAL(3)),
                      CUFFT_MUL(imaginary, CUFFT_MUL(k03, index3))));
        output = CUFFT_DIV(CUFFT_MUL(epsilon0, pola),
                           CUFFT_ADD(CUFFT_REAL(CU_REAL(1)),
                                     CUFFT_MUL(pola, correction)));
    } else if (method == 8) {
        const CU_REAL xx = k0 * maille;
        const CUCOMPLEX log_term = scalar_polarizability_log(CUFFT_DIV(
            CUFFT_REAL(pi_value - xx), CUFFT_REAL(pi_value + xx)));
        const CUCOMPLEX mint = CUFFT_ADD(
            CUFFT_REAL(CU_REAL(4) / CU_REAL(3) * xx * xx),
            CUFFT_MUL(CUFFT_REAL(CU_REAL(2) / CU_REAL(3)),
                CUFFT_MUL(CUFFT_ADD(imaginary,
                    CUFFT_MUL(CUFFT_REAL(CU_REAL(1) / pi_value), log_term)),
                    CUFFT_REAL(xx * xx * xx))));
        const CUCOMPLEX numerator = CUFFT_MUL(
            CUFFT_REAL(maille * maille * maille),
            CUFFT_MUL(epsilon0, CUFFT_SUB(epsilon, epsilon0)));
        const CUCOMPLEX four_pi = CUFFT_REAL(CU_REAL(4) * pi_value);
        const CUCOMPLEX denominator = CUFFT_ADD(epsilon0,
            CUFFT_MUL(CUFFT_SUB(CUFFT_REAL(CU_REAL(4) * pi_value / CU_REAL(3)), mint),
                CUFFT_DIV(CUFFT_SUB(epsilon, epsilon0), four_pi)));
        output = CUFFT_DIV(CUFFT_DIV(numerator, four_pi), denominator);
    }
    return output;
}

//-----------------------------------------------------------
__global__ void scalar_polarizability_kernel(
    CUCOMPLEX* result, CU_REAL* sldr, CU_REAL radius_input, CUCOMPLEX epsilon,
    CUCOMPLEX epsilon0, CU_REAL k0, int discretized, int method) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    *result = scalar_polarizability_compute(
        radius_input, epsilon, epsilon0, k0, discretized, method, sldr);
}

//-----------------------------------------------------------
__global__ void scalar_polarizability_batch_kernel(
    const CUCOMPLEX* epsilon, const CU_INT* inside, CUCOMPLEX* values,
    CU_REAL* sldr, int count, CU_REAL radius, CUCOMPLEX epsilon0,
    CU_REAL k0, int discretized, int method, CU_REAL initial_sldr) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count)
        return;
    if (inside[i].dat) {
        CU_REAL site_sldr = initial_sldr;
        values[i] = scalar_polarizability_compute(
            radius, epsilon[i], epsilon0, k0, discretized, method, &site_sldr);
    } else {
        values[i] = CUFFT_ZERO;
    }
    if (i == 0) {
        CU_REAL result_sldr = initial_sldr;
        if (method == 6) {
            for (int site = 0; site < count; ++site) {
                if (inside[site].dat) {
                    result_sldr = CU_REAL(0.2);
                    break;
                }
            }
        }
        *sldr = result_sldr;
    }
}

//-----------------------------------------------------------
__global__ void material_interpolation_kernel(CU_REAL *result, CU_REAL n, CU_REAL k) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    result[0] = n * n - k * k;
    result[1] = CU_REAL(2) * n * k;
}

//-----------------------------------------------------------
__global__ void rerange_isotropic_polarizability_kernel(
    const CUCOMPLEX *epsilon, CUCOMPLEX *polarizability, int count,
    CUCOMPLEX epsilon0, CU_REAL scale) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        const CUCOMPLEX contrast = CUFFT_SUB(epsilon[i], epsilon0);
        polarizability[i] = CUFFT_MUL(contrast, CUFFT_REAL(scale));
    }
}

//-----------------------------------------------------------
__device__ bool inverse_complex_matrix3(const CUCOMPLEX* input, CUCOMPLEX* output) {
    CUCOMPLEX augmented[3][6];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            augmented[row][column] = input[row + column * 3];
            augmented[row][column + 3] = row == column
                ? CUFFT_REAL(CU_REAL(1)) : CUFFT_ZERO;
        }
    }
    for (int pivot_column = 0; pivot_column < 3; ++pivot_column) {
        int pivot_row = pivot_column;
        CU_REAL pivot_norm = augmented[pivot_row][pivot_column].x *
                                 augmented[pivot_row][pivot_column].x +
                             augmented[pivot_row][pivot_column].y *
                                 augmented[pivot_row][pivot_column].y;
        for (int row = pivot_column + 1; row < 3; ++row) {
            const CU_REAL norm = augmented[row][pivot_column].x *
                                     augmented[row][pivot_column].x +
                                 augmented[row][pivot_column].y *
                                     augmented[row][pivot_column].y;
            if (norm > pivot_norm) {
                pivot_norm = norm;
                pivot_row = row;
            }
        }
        if (pivot_norm <= CU_REAL(1e-30))
            return false;
        if (pivot_row != pivot_column) {
            for (int column = 0; column < 6; ++column) {
                const CUCOMPLEX temporary = augmented[pivot_column][column];
                augmented[pivot_column][column] = augmented[pivot_row][column];
                augmented[pivot_row][column] = temporary;
            }
        }
        const CUCOMPLEX pivot = augmented[pivot_column][pivot_column];
        for (int column = 0; column < 6; ++column)
            augmented[pivot_column][column] =
                CUFFT_DIV(augmented[pivot_column][column], pivot);
        for (int row = 0; row < 3; ++row) {
            if (row == pivot_column)
                continue;
            const CUCOMPLEX factor = augmented[row][pivot_column];
            for (int column = 0; column < 6; ++column)
                augmented[row][column] = CUFFT_SUB(
                    augmented[row][column],
                    CUFFT_MUL(factor, augmented[pivot_column][column]));
        }
    }
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            output[row + column * 3] = augmented[row][column + 3];
    return true;
}

//-----------------------------------------------------------
__device__ void multiply_complex_matrix3(const CUCOMPLEX* left,
                                         const CUCOMPLEX* right,
                                         CUCOMPLEX* output) {
    for (int column = 0; column < 3; ++column) {
        for (int row = 0; row < 3; ++row) {
            CUCOMPLEX sum = CUFFT_ZERO;
            for (int k = 0; k < 3; ++k)
                sum = CUFFT_ADD(sum, CUFFT_MUL(left[row + k * 3],
                                               right[k + column * 3]));
            output[row + column * 3] = sum;
        }
    }
}

//-----------------------------------------------------------
__global__ void tensor_polarizability_kernel(
    const CUCOMPLEX* epsilon, CUCOMPLEX* result, CU_INT* status,
    CUCOMPLEX epsilon0, CU_REAL radius_input, CU_REAL k0,
    int discretized, int method, int invert_result) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    status->dat = 0;
    const CU_REAL pi_value = CU_REAL(3.1415926535897932384626433832795L);
    CU_REAL radius = radius_input;
    if (discretized == 1)
        radius *= pow(CU_REAL(0.75) / pi_value, CU_REAL(1) / CU_REAL(3));
    const CU_REAL volume = radius * radius * radius;
    CUCOMPLEX denominator[9];
    CUCOMPLEX numerator[9];
    CUCOMPLEX inverse[9];
    CUCOMPLEX alpha0[9];
    CUCOMPLEX temporary[9];
    for (int column = 0; column < 3; ++column) {
        for (int row = 0; row < 3; ++row) {
            const int index = row + column * 3;
            denominator[index] = epsilon[index];
            numerator[index] = epsilon[index];
            if (row == column) {
                denominator[index] = CUFFT_ADD(
                    denominator[index], CUFFT_MUL(CUFFT_REAL(CU_REAL(2)), epsilon0));
                numerator[index] = CUFFT_SUB(numerator[index], epsilon0);
            }
        }
    }
    if (!inverse_complex_matrix3(denominator, inverse)) {
        status->dat = 1;
        for (int i = 0; i < 9; ++i) result[i] = CUFFT_ZERO;
        return;
    }
    multiply_complex_matrix3(numerator, inverse, temporary);
    for (int i = 0; i < 9; ++i)
        alpha0[i] = CUFFT_MUL(temporary[i], CUFFT_REAL(volume));

    if (method == 1) {
        for (int i = 0; i < 9; ++i)
            result[i] = CUFFT_MUL(alpha0[i], epsilon0);
    } else {
        const CUCOMPLEX index = scalar_polarizability_sqrt(epsilon0);
        const CUCOMPLEX index3 = CUFFT_MUL(CUFFT_MUL(index, index), index);
        const CUCOMPLEX k03 = CUFFT_MUL(CUFFT_REAL(k0 * k0 * k0), index3);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CUCOMPLEX imaginary = make_cuFloatComplex(0.0f, 1.0f);
#else
        const CUCOMPLEX imaginary = make_cuDoubleComplex(0.0, 1.0);
#endif
        for (int column = 0; column < 3; ++column) {
            for (int row = 0; row < 3; ++row) {
                const int i = row + column * 3;
                const CUCOMPLEX correction = CUFFT_MUL(
                    CUFFT_REAL(CU_REAL(2) / CU_REAL(3)),
                    CUFFT_MUL(imaginary, CUFFT_MUL(k03, alpha0[i])));
                temporary[i] = CUFFT_SUB(
                    row == column ? CUFFT_REAL(CU_REAL(1)) : CUFFT_ZERO,
                    correction);
            }
        }
        if (!inverse_complex_matrix3(temporary, inverse)) {
            status->dat = 2;
            for (int i = 0; i < 9; ++i) result[i] = CUFFT_ZERO;
            return;
        }
        multiply_complex_matrix3(inverse, alpha0, temporary);
        for (int i = 0; i < 9; ++i)
            result[i] = CUFFT_MUL(temporary[i], epsilon0);
    }
    if (invert_result == -1) {
        if (!inverse_complex_matrix3(result, inverse)) {
            status->dat = 3;
            for (int i = 0; i < 9; ++i) result[i] = CUFFT_ZERO;
            return;
        }
        for (int i = 0; i < 9; ++i) result[i] = inverse[i];
    }
}

//-----------------------------------------------------------
__global__ void cuboid_geometry_kernel(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, CU_INT *dipole_map,
                                       int nx, int ny, int nz, CU_REAL cell_size, CU_REAL side_x,
                                       CU_REAL side_y, CU_REAL side_z, CU_REAL center_x,
                                       CU_REAL center_y, CU_REAL center_z, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int k = flat % nx + 1;
    const int j = (flat / nx) % ny + 1;
    const int i = flat / (nx * ny) + 1;
    const CU_REAL x = cell_size * (CU_REAL(k) - CU_REAL(0.5)) - side_x / CU_REAL(2);
    const CU_REAL y = cell_size * (CU_REAL(j) - CU_REAL(0.5)) - side_y / CU_REAL(2);
    const CU_REAL z = cell_size * (CU_REAL(i) - CU_REAL(0.5)) - side_z / CU_REAL(2);
    if (xs) xs[flat] = x + center_x;
    if (ys) ys[flat] = y + center_y;
    if (zs) zs[flat] = z + center_z;
    dipole_map[flat].dat = (fabs(x) < side_x / CU_REAL(2) && fabs(y) < side_y / CU_REAL(2) &&
                        fabs(z) < side_z / CU_REAL(2))
                           ? flat + 1
                           : 0;
}

//-----------------------------------------------------------
__global__ void rotated_cuboid_geometry_kernel(
    CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, CU_INT*inside_map, int nx, int ny, int nz,
    CU_REAL cell_size, CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z, CU_REAL side_x,
    CU_REAL side_y, CU_REAL side_z, CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
    CU_REAL r0, CU_REAL r1, CU_REAL r2, CU_REAL r3, CU_REAL r4, CU_REAL r5, CU_REAL r6, CU_REAL r7,
    CU_REAL r8, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int k = flat % nx + 1;
    const int j = (flat / nx) % ny + 1;
    const int i = flat / (nx * ny) + 1;
    const CU_REAL x = -extent_x / CU_REAL(2) + cell_size * (CU_REAL(k) - CU_REAL(0.5));
    const CU_REAL y = -extent_y / CU_REAL(2) + cell_size * (CU_REAL(j) - CU_REAL(0.5));
    const CU_REAL z = -extent_z / CU_REAL(2) + cell_size * (CU_REAL(i) - CU_REAL(0.5));
    const CU_REAL xr = r0 * x + r3 * y + r6 * z;
    const CU_REAL yr = r1 * x + r4 * y + r7 * z;
    const CU_REAL zr = r2 * x + r5 * y + r8 * z;
    xs[flat] = x + center_x;
    ys[flat] = y + center_y;
    zs[flat] = z + center_z;
    inside_map[flat].dat = (fabs(xr) <= side_x / CU_REAL(2) && fabs(yr) <= side_y / CU_REAL(2) &&
                        fabs(zr) <= side_z / CU_REAL(2))
                           ? 1
                           : 0;
}

//-----------------------------------------------------------
__global__ void inhomogeneous_material_spectrum_kernel(
    CUCOMPLEX *spectrum, const CU_REAL *phases, int nx, int ny, int nz,
    CU_REAL dkx, CU_REAL dky, CU_REAL dkz, CU_REAL coefficient1,
    CU_REAL coefficient2, CU_REAL coefficient3, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int k = flat % nx + 1;
    const int j = (flat / nx) % ny + 1;
    const int i = flat / (nx * ny) + 1;
    const CU_REAL kx = CU_REAL(k <= nx / 2 + 1 ? k - 1 : k - 1 - nx) * dkx;
    const CU_REAL ky = CU_REAL(j <= ny / 2 + 1 ? j - 1 : j - 1 - ny) * dky;
    const CU_REAL kz = CU_REAL(i <= nz / 2 + 1 ? i - 1 : i - 1 - nz) * dkz;
    const CU_REAL spec = coefficient1 * exp(-CU_REAL(0.25) * coefficient2 *
                                            (kx * kx + ky * ky + kz * kz));
    const CU_REAL amplitude = sqrt(spec * coefficient3);
    const CU_REAL angle = CU_REAL(2) * CU_REAL(3.141592653589793238462643383279502884L) *
                          phases[flat];
    CUCOMPLEX value;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    value = make_cuFloatComplex(amplitude * cos(angle), amplitude * sin(angle));
#else
    value = make_cuDoubleComplex(amplitude * cos(angle), amplitude * sin(angle));
#endif
    const int one_based = flat + 1;
    const int plane = nx * ny;
    if (one_based == 1 ||
        (one_based >= nx / 2 + 2 && one_based <= nx) ||
        (one_based >= (ny / 2 + 1) * nx + 1 && one_based <= nx * ny) ||
        (one_based >= (nz / 2 + 1) * plane + 1 && one_based <= count))
        value = CUFFT_ZERO;
    spectrum[flat] = value;
}

//-----------------------------------------------------------
__global__ void inhomogeneous_material_finalize_kernel(
    const CUCOMPLEX *spectrum, CU_REAL *real_values, CU_REAL *magnitude_squares,
    int count, CU_REAL epsilon_real, CU_REAL epsilon_imag) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const CU_REAL real_value = spectrum[index].x + epsilon_real;
    const CU_REAL imag_value = epsilon_imag;
    real_values[index] = real_value;
    magnitude_squares[index] = real_value * real_value + imag_value * imag_value;
}

//-----------------------------------------------------------
__global__ void fill_real_ones_kernel(CU_REAL *values, int count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
        values[index] = CU_REAL(1);
}

//-----------------------------------------------------------
__global__ void sphere_geometry_kernel(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, CU_INT*inside_map,
                                       int subdivisions, CU_REAL radius, CU_REAL cell_size,
                                       CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
                                       int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int k = flat % subdivisions + 1;
    const int j = (flat / subdivisions) % subdivisions + 1;
    const int i = flat / (subdivisions * subdivisions) + 1;
    const CU_REAL x = -radius + cell_size * (CU_REAL(k) - CU_REAL(0.5));
    const CU_REAL y = -radius + cell_size * (CU_REAL(j) - CU_REAL(0.5));
    const CU_REAL z = -radius + cell_size * (CU_REAL(i) - CU_REAL(0.5));
    xs[flat] = x + center_x;
    ys[flat] = y + center_y;
    zs[flat] = z + center_z;
    inside_map[flat].dat = (sqrt(x * x + y * y + z * z) < radius) ? 1 : 0;
}

//-----------------------------------------------------------
__global__ void rotated_cuboid_dipole_map_kernel(
    CU_INT* inside_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
    CU_REAL side_x, CU_REAL side_y, CU_REAL side_z,
    CU_REAL r0, CU_REAL r1, CU_REAL r2, CU_REAL r3,
    CU_REAL r4, CU_REAL r5, CU_REAL r6, CU_REAL r7, CU_REAL r8, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count) return;
    const int k = flat % nx + 1;
    const int j = (flat / nx) % ny + 1;
    const int i = flat / (nx * ny) + 1;
    const CU_REAL x = -extent_x / CU_REAL(2) + cell_size * (CU_REAL(k) - CU_REAL(0.5));
    const CU_REAL y = -extent_y / CU_REAL(2) + cell_size * (CU_REAL(j) - CU_REAL(0.5));
    const CU_REAL z = -extent_z / CU_REAL(2) + cell_size * (CU_REAL(i) - CU_REAL(0.5));
    const CU_REAL xr = r0*x + r3*y + r6*z;
    const CU_REAL yr = r1*x + r4*y + r7*z;
    const CU_REAL zr = r2*x + r5*y + r8*z;
    inside_map[flat].dat =
        fabs(xr) <= side_x / CU_REAL(2) &&
        fabs(yr) <= side_y / CU_REAL(2) &&
        fabs(zr) <= side_z / CU_REAL(2) ? 1 : 0;
}

//-----------------------------------------------------------
__global__ void sphere_dipole_map_kernel(CU_INT* inside_map, int subdivisions,
                                        CU_REAL radius, CU_REAL cell_size, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int k = flat % subdivisions;
    const int j = (flat / subdivisions) % subdivisions;
    const int i = flat / (subdivisions * subdivisions);
    const CU_REAL x = -radius + cell_size * (CU_REAL(k) + CU_REAL(0.5));
    const CU_REAL y = -radius + cell_size * (CU_REAL(j) + CU_REAL(0.5));
    const CU_REAL z = -radius + cell_size * (CU_REAL(i) + CU_REAL(0.5));
    // Preserve the Fortran and existing CUDA boundary predicate exactly.
    inside_map[flat].dat = sqrt(x*x + y*y + z*z) < radius ? 1 : 0;
}

//-----------------------------------------------------------
__global__ void sphere_geometry_compact_kernel(
    const CU_REAL* mesh_x, const CU_REAL* mesh_y, const CU_REAL* mesh_z,
    const CU_INT* inside_map, const CU_INT* prefix, int count,
    CU_REAL* packed_x, CU_REAL* packed_y, CU_REAL* packed_z, CU_INT* packed_count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    if (inside_map[index].dat != 0) {
        const int destination = prefix[index].dat;
        packed_x[destination] = mesh_x[index];
        packed_y[destination] = mesh_y[index];
        packed_z[destination] = mesh_z[index];
    }
    if (index == count - 1)
        packed_count->dat = prefix[index].dat + (inside_map[index].dat != 0 ? 1 : 0);
}

//-----------------------------------------------------------
__global__ void ellipsoid_geometry_kernel(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, CU_INT*inside_map,
                                          int nx, int ny, int nz, CU_REAL cell_size,
                                          CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
                                          CU_REAL radius_a, CU_REAL radius_b, CU_REAL radius_c,
                                          CU_REAL m0, CU_REAL m1, CU_REAL m2, CU_REAL m3,
                                          CU_REAL m4, CU_REAL m5, CU_REAL m6, CU_REAL m7,
                                          CU_REAL m8, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int k = flat % nx + 1;
    const int j = (flat / nx) % ny + 1;
    const int i = flat / (nx * ny) + 1;
    const CU_REAL x = -extent_x / CU_REAL(2) + cell_size * (CU_REAL(k) - CU_REAL(0.5));
    const CU_REAL y = -extent_y / CU_REAL(2) + cell_size * (CU_REAL(j) - CU_REAL(0.5));
    const CU_REAL z = -extent_z / CU_REAL(2) + cell_size * (CU_REAL(i) - CU_REAL(0.5));
    const CU_REAL xr = m0 * x + m1 * y + m2 * z;
    const CU_REAL yr = m3 * x + m4 * y + m5 * z;
    const CU_REAL zr = m6 * x + m7 * y + m8 * z;
    const CU_REAL normalized = xr * xr / radius_a / radius_a + yr * yr / radius_b / radius_b +
                               zr * zr / radius_c / radius_c;
    xs[flat] = x;
    ys[flat] = y;
    zs[flat] = z;
    inside_map[flat].dat = normalized <= CU_REAL(1) ? 1 : 0;
}

//-----------------------------------------------------------
__global__ void ellipsoid_dipole_map_kernel(
    CU_INT* inside_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
    CU_REAL radius_a, CU_REAL radius_b, CU_REAL radius_c,
    CU_REAL m0, CU_REAL m1, CU_REAL m2, CU_REAL m3, CU_REAL m4,
    CU_REAL m5, CU_REAL m6, CU_REAL m7, CU_REAL m8, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count) return;
    const int k = flat % nx + 1;
    const int j = (flat / nx) % ny + 1;
    const int i = flat / (nx * ny) + 1;
    const CU_REAL x = -extent_x/CU_REAL(2) + cell_size*(CU_REAL(k)-CU_REAL(0.5));
    const CU_REAL y = -extent_y/CU_REAL(2) + cell_size*(CU_REAL(j)-CU_REAL(0.5));
    const CU_REAL z = -extent_z/CU_REAL(2) + cell_size*(CU_REAL(i)-CU_REAL(0.5));
    const CU_REAL xr = m0*x + m1*y + m2*z;
    const CU_REAL yr = m3*x + m4*y + m5*z;
    const CU_REAL zr = m6*x + m7*y + m8*z;
    const CU_REAL normalized = xr*xr/radius_a/radius_a + yr*yr/radius_b/radius_b +
                               zr*zr/radius_c/radius_c;
    inside_map[flat].dat = normalized <= CU_REAL(1) ? 1 : 0;
}

//-----------------------------------------------------------
__global__ void cylinder_geometry_kernel(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, CU_INT*inside_map,
                                         int nx, int ny, int nz, CU_REAL cell_size,
                                         CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
                                         CU_REAL radius, CU_REAL height, CU_REAL m0, CU_REAL m1,
                                         CU_REAL m2, CU_REAL m3, CU_REAL m4, CU_REAL m5, CU_REAL m6,
                                         CU_REAL m7, CU_REAL m8, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int k = flat % nx + 1;
    const int j = (flat / nx) % ny + 1;
    const int i = flat / (nx * ny) + 1;
    const CU_REAL x = -extent_x / CU_REAL(2) + cell_size * (CU_REAL(k) - CU_REAL(0.5));
    const CU_REAL y = -extent_y / CU_REAL(2) + cell_size * (CU_REAL(j) - CU_REAL(0.5));
    const CU_REAL z = -extent_z / CU_REAL(2) + cell_size * (CU_REAL(i) - CU_REAL(0.5));
    const CU_REAL xr = m0 * x + m1 * y + m2 * z;
    const CU_REAL yr = m3 * x + m4 * y + m5 * z;
    const CU_REAL zr = m6 * x + m7 * y + m8 * z;
    xs[flat] = x;
    ys[flat] = y;
    zs[flat] = z;
    inside_map[flat].dat =
        (xr * xr + yr * yr <= radius * radius && fabs(zr) <= height / CU_REAL(2)) ? 1 : 0;
}

//-----------------------------------------------------------
__global__ void cylinder_dipole_map_kernel(
    CU_INT* inside_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
    CU_REAL radius, CU_REAL height,
    CU_REAL m0, CU_REAL m1, CU_REAL m2, CU_REAL m3, CU_REAL m4,
    CU_REAL m5, CU_REAL m6, CU_REAL m7, CU_REAL m8, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count) return;
    const int k = flat % nx + 1;
    const int j = (flat / nx) % ny + 1;
    const int i = flat / (nx * ny) + 1;
    const CU_REAL x = -extent_x/CU_REAL(2) + cell_size*(CU_REAL(k)-CU_REAL(0.5));
    const CU_REAL y = -extent_y/CU_REAL(2) + cell_size*(CU_REAL(j)-CU_REAL(0.5));
    const CU_REAL z = -extent_z/CU_REAL(2) + cell_size*(CU_REAL(i)-CU_REAL(0.5));
    const CU_REAL xr = m0*x + m1*y + m2*z;
    const CU_REAL yr = m3*x + m4*y + m5*z;
    const CU_REAL zr = m6*x + m7*y + m8*z;
    inside_map[flat].dat =
        xr*xr + yr*yr <= radius*radius && fabs(zr) <= height/CU_REAL(2) ? 1 : 0;
}

//-----------------------------------------------------------
__global__ void multi_sphere_geometry_kernel(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, CU_INT*sphere_map,
                                             int nx, int ny, int nz, CU_REAL cell_size,
                                             CU_REAL x_min, CU_REAL y_min, CU_REAL z_min,
                                             const CU_REAL *sphere_parameters, int sphere_count,
                                             int fixed_sphere_count, int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int k = flat % nx;
    const int j = (flat / nx) % ny;
    const int i = flat / (nx * ny);
    const CU_REAL x = x_min + (CU_REAL(k) + CU_REAL(0.5)) * cell_size;
    const CU_REAL y = y_min + (CU_REAL(j) + CU_REAL(0.5)) * cell_size;
    const CU_REAL z = z_min + (CU_REAL(i) + CU_REAL(0.5)) * cell_size;
    if (xs) xs[flat] = x;
    if (ys) ys[flat] = y;
    if (zs) zs[flat] = z;
    int selected_fixed = 0;
    int fixed_hits = 0;
    int selected_random = 0;
    for (int s = 0; s < sphere_count; ++s) {
        const CU_REAL *p = sphere_parameters + 4 * s;
        const CU_REAL dx = x - p[0];
        const CU_REAL dy = y - p[1];
        const CU_REAL dz = z - p[2];
        if (dx * dx + dy * dy + dz * dz <= p[3] * p[3]) {
            if (s < fixed_sphere_count) {
                ++fixed_hits;
                selected_fixed = s + 1;
            } else {
                // Fortran gives fixed spheres priority and lets later
                // random spheres overwrite earlier random spheres.
                selected_random = s + 1;
            }
        }
    }
    sphere_map[flat].dat =
        fixed_hits > 1 ? -selected_fixed : (fixed_hits == 1 ? selected_fixed : selected_random);
}

//-----------------------------------------------------------
__global__ void concentric_sphere_geometry_kernel(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
                                                  CU_INT*layer_map, int subdivisions,
                                                  CU_REAL cell_size, CU_REAL center,
                                                  const CU_REAL *radii, int sphere_count,
                                                  int count) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= count)
        return;
    const int k = flat % subdivisions + 1;
    const int j = (flat / subdivisions) % subdivisions + 1;
    const int i = flat / (subdivisions * subdivisions) + 1;
    const CU_REAL x = CU_REAL(k - 1) * cell_size + cell_size / CU_REAL(2) - center;
    const CU_REAL y = CU_REAL(j - 1) * cell_size + cell_size / CU_REAL(2) - center;
    const CU_REAL z = CU_REAL(i - 1) * cell_size + cell_size / CU_REAL(2) - center;
    const CU_REAL distance = sqrt(x * x + y * y + z * z);
    int layer = 0;
    if (distance <= radii[0])
        layer = 1;
    for (int s = 1; s < sphere_count; ++s)
        if (distance <= radii[s] && distance >= radii[s - 1])
            layer = s + 1;
    if (xs) xs[flat] = x;
    if (ys) ys[flat] = y;
    if (zs) zs[flat] = z;
    layer_map[flat].dat = layer;
}

//-----------------------------------------------------------
__global__ void polarization_inplace_kernel(const CUCOMPLEX *polarizability, CUCOMPLEX *field,
                                            int niso, int nmax, int ndipoles) {
    const int dipole = blockIdx.x * blockDim.x + threadIdx.x;
    if (dipole >= ndipoles)
        return;
    const int base = 3 * dipole;
    const CUCOMPLEX x = field[base];
    const CUCOMPLEX y = field[base + 1];
    const CUCOMPLEX z = field[base + 2];
    if (niso == 1) {
        const CUCOMPLEX alpha = polarizability[dipole];
        field[base] = CUFFT_MUL(alpha, x);
        field[base + 1] = CUFFT_MUL(alpha, y);
        field[base + 2] = CUFFT_MUL(alpha, z);
    } else {
        const size_t stride = size_t(nmax);
        const CUCOMPLEX a11 = polarizability[dipole];
        const CUCOMPLEX a21 = polarizability[dipole + stride];
        const CUCOMPLEX a31 = polarizability[dipole + 2 * stride];
        const CUCOMPLEX a12 = polarizability[dipole + 3 * stride];
        const CUCOMPLEX a22 = polarizability[dipole + 4 * stride];
        const CUCOMPLEX a32 = polarizability[dipole + 5 * stride];
        const CUCOMPLEX a13 = polarizability[dipole + 6 * stride];
        const CUCOMPLEX a23 = polarizability[dipole + 7 * stride];
        const CUCOMPLEX a33 = polarizability[dipole + 8 * stride];
        field[base] = CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(a11, x), CUFFT_MUL(a12, y)), CUFFT_MUL(a13, z));
        field[base + 1] =
            CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(a21, x), CUFFT_MUL(a22, y)), CUFFT_MUL(a23, z));
        field[base + 2] =
            CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(a31, x), CUFFT_MUL(a32, y)), CUFFT_MUL(a33, z));
    }
}

//-----------------------------------------------------------
__global__ void local_macro_to_local_kernel(const CUCOMPLEX *epsilon, CUCOMPLEX *field, int niso,
                                            int nmax, int ndipoles, CU_REAL aretecube, CU_REAL k0) {
    const int dipole = blockIdx.x * blockDim.x + threadIdx.x;
    if (dipole >= ndipoles)
        return;

    const CU_REAL constant = k0 * k0 * k0 * aretecube * aretecube * aretecube /
                             (CU_REAL(2) * CU_REAL(3.141592653589793238462643383279502884));
    CUCOMPLEX matrix[9];
    for (int column = 0; column < 3; ++column) {
        for (int row = 0; row < 3; ++row) {
            CUCOMPLEX value;
            if (niso == 1)
                value = row == column ? epsilon[dipole] : CUFFT_ZERO;
            else
                value = epsilon[size_t(dipole) + size_t(nmax) * size_t(row + 3 * column)];

            const CUCOMPLEX difference =
                row == column ? CUFFT_SUB(value, CUFFT_REAL(CU_REAL(1))) : value;
            CUCOMPLEX local_matrix =
                row == column ? CUFFT_ADD(value, CUFFT_REAL(CU_REAL(2))) : value;
            // Fortran local_macro forms i*constant*(epsilon-I), then subtracts it.
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            const CUCOMPLEX correction =
                make_cuFloatComplex(-difference.y * constant, difference.x * constant);
#else
            const CUCOMPLEX correction =
                make_cuDoubleComplex(-difference.y * constant, difference.x * constant);
#endif
            local_matrix = CUFFT_SUB(local_matrix, correction);
            matrix[row + 3 * column] = local_matrix;
        }
    }

    const int base = 3 * dipole;
    const CUCOMPLEX incident[3] = {field[base], field[base + 1], field[base + 2]};
    for (int row = 0; row < 3; ++row) {
        CUCOMPLEX value = CUFFT_ZERO;
        for (int column = 0; column < 3; ++column)
            value = CUFFT_ADD(value, CUFFT_MUL(matrix[row + 3 * column], incident[column]));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        field[base + row] = make_cuFloatComplex(value.x / 3.0f, value.y / 3.0f);
#else
        field[base + row] = make_cuDoubleComplex(value.x / 3.0, value.y / 3.0);
#endif
    }
}

//-----------------------------------------------------------
__global__ void local_macro_batch_kernel(const CUCOMPLEX *epsilon,
                                         const CUCOMPLEX *input_field,
                                         CUCOMPLEX *output_field,
                                         CU_REAL *field_magnitude, CU_INT*status,
                                         int niso, int nmax, int ndipoles,
                                         CU_REAL aretecube, CU_REAL k0, int nsens) {
    const int dipole = blockIdx.x * blockDim.x + threadIdx.x;
    if (dipole >= ndipoles)
        return;
    const CU_REAL constant = k0 * k0 * k0 * aretecube * aretecube * aretecube /
                             (CU_REAL(2) * CU_REAL(3.141592653589793238462643383279502884));
    CUCOMPLEX matrix[9];
    for (int column = 0; column < 3; ++column) {
        for (int row = 0; row < 3; ++row) {
            const int tensor_component = row + 3 * column;
            const int tensor_index = niso == 1
                ? dipole
                : dipole + nmax * tensor_component;
            const CUCOMPLEX value = niso == 1
                ? (row == column ? epsilon[dipole] : CUFFT_ZERO)
                : epsilon[tensor_index];
            const CUCOMPLEX difference = row == column
                ? CUFFT_SUB(value, CUFFT_REAL(CU_REAL(1))) : value;
            CUCOMPLEX local_matrix = row == column
                ? CUFFT_ADD(value, CUFFT_REAL(CU_REAL(2))) : value;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            const CUCOMPLEX correction =
                make_cuFloatComplex(-difference.y * constant, difference.x * constant);
#else
            const CUCOMPLEX correction =
                make_cuDoubleComplex(-difference.y * constant, difference.x * constant);
#endif
            matrix[tensor_component] = CUFFT_SUB(local_matrix, correction);
        }
    }

    CUCOMPLEX operation_matrix[9];
    if (nsens == 1) {
        if (!inverse_complex_matrix3(matrix, operation_matrix)) {
            const int base = 3 * dipole;
            output_field[base] = CUFFT_ZERO;
            output_field[base + 1] = CUFFT_ZERO;
            output_field[base + 2] = CUFFT_ZERO;
            field_magnitude[dipole] = CU_REAL(0);
            atomicExch(&status->dat, 1);
            return;
        }
    } else {
        for (int i = 0; i < 9; ++i)
            operation_matrix[i] = matrix[i];
    }

    const int base = 3 * dipole;
    CU_REAL intensity = CU_REAL(0);
    for (int row = 0; row < 3; ++row) {
        CUCOMPLEX value = CUFFT_ZERO;
        for (int column = 0; column < 3; ++column)
            value = CUFFT_ADD(value, CUFFT_MUL(operation_matrix[row + 3 * column],
                                               input_field[base + column]));
        if (nsens == 1) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            output_field[base + row] = make_cuFloatComplex(value.x * 3.0f, value.y * 3.0f);
#else
            output_field[base + row] = make_cuDoubleComplex(value.x * 3.0, value.y * 3.0);
#endif
        } else {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            output_field[base + row] = make_cuFloatComplex(value.x / 3.0f, value.y / 3.0f);
#else
            output_field[base + row] = make_cuDoubleComplex(value.x / 3.0, value.y / 3.0);
#endif
        }
        const CUCOMPLEX component = output_field[base + row];
        intensity += component.x * component.x + component.y * component.y;
    }
    field_magnitude[dipole] = sqrt(intensity);
}

//-----------------------------------------------------------
__global__ void complex_vector_scale_kernel(const CUCOMPLEX *input, CUCOMPLEX *output,
                                            CUCOMPLEX scale, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = CUFFT_MUL(input[i], scale);
}

//-----------------------------------------------------------
__global__ void rytov_macro_delta_kernel(const CUCOMPLEX *incident, const CUCOMPLEX *epsilon,
                                         CUCOMPLEX *delta, CUCOMPLEX background_epsilon, int niso,
                                         int nmax, int ndipoles) {
    const int dipole = blockIdx.x * blockDim.x + threadIdx.x;
    if (dipole >= ndipoles)
        return;
    const int base = 3 * dipole;
    for (int component = 0; component < 3; ++component) {
        const CUCOMPLEX material =
            niso == 1 ? epsilon[dipole]
                      : epsilon[size_t(dipole) + size_t(nmax) * size_t(component + 3 * component)];
        const CUCOMPLEX contrast = CUFFT_SUB(material, background_epsilon);
        const CUCOMPLEX scaled = CUFFT_MUL(incident[base + component], contrast);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        delta[base + component] = make_cuFloatComplex(-scaled.x / 3.0f, -scaled.y / 3.0f);
#else
        delta[base + component] = make_cuDoubleComplex(-scaled.x / 3.0, -scaled.y / 3.0);
#endif
    }
}

struct GpbicgstablTerms {
    int source_columns[10];
    CUCOMPLEX coefficients[10];
    int count;
};

//-----------------------------------------------------------
__global__ void gpbicgstabl_combine_kernel(CUCOMPLEX *work, CUCOMPLEX *xsol, int lda, int n,
                                           int dst_col, GpbicgstablTerms terms,
                                           int add_to_destination) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CUCOMPLEX value = CUFFT_ZERO;
    for (int k = 0; k < terms.count; ++k)
        value = CUFFT_ADD(
            value, CUFFT_MUL(terms.coefficients[k],
                             work[size_t(terms.source_columns[k]) * size_t(lda) + size_t(i)]));
    CUCOMPLEX *destination = dst_col < 0 ? xsol : work + size_t(dst_col) * size_t(lda);
    if (add_to_destination)
        destination[i] = CUFFT_ADD(destination[i], value);
    else
        destination[i] = value;
}

//-----------------------------------------------------------
__global__ void solve_small_dense_system_kernel(CUCOMPLEX *matrix, CUCOMPLEX *rhs, int order,
                                                CU_INT*info) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    info->dat = 0;
    for (int k = 0; k < order; ++k) {
        int pivot = k;
        CU_REAL best =
            matrix[k * 9 + k].x * matrix[k * 9 + k].x + matrix[k * 9 + k].y * matrix[k * 9 + k].y;
        for (int i = k + 1; i < order; ++i) {
            const CU_REAL magnitude = matrix[i * 9 + k].x * matrix[i * 9 + k].x +
                                      matrix[i * 9 + k].y * matrix[i * 9 + k].y;
            if (magnitude > best) {
                best = magnitude;
                pivot = i;
            }
        }
        if (!(best > CU_REAL(0)) || !isfinite(best)) {
            info->dat = k + 1;
            return;
        }
        if (pivot != k) {
            for (int j = 0; j < order; ++j) {
                const CUCOMPLEX tmp = matrix[k * 9 + j];
                matrix[k * 9 + j] = matrix[pivot * 9 + j];
                matrix[pivot * 9 + j] = tmp;
            }
            const CUCOMPLEX tmp = rhs[k];
            rhs[k] = rhs[pivot];
            rhs[pivot] = tmp;
        }
        for (int i = k + 1; i < order; ++i) {
            const CUCOMPLEX factor = CUFFT_DIV(matrix[i * 9 + k], matrix[k * 9 + k]);
            matrix[i * 9 + k] = CUFFT_ZERO;
            for (int j = k + 1; j < order; ++j)
                matrix[i * 9 + j] =
                    CUFFT_SUB(matrix[i * 9 + j], CUFFT_MUL(factor, matrix[k * 9 + j]));
            rhs[i] = CUFFT_SUB(rhs[i], CUFFT_MUL(factor, rhs[k]));
        }
    }
    for (int i = order - 1; i >= 0; --i) {
        for (int j = i + 1; j < order; ++j)
            rhs[i] = CUFFT_SUB(rhs[i], CUFFT_MUL(matrix[i * 9 + j], rhs[j]));
        rhs[i] = CUFFT_DIV(rhs[i], matrix[i * 9 + i]);
        if (!isfinite(rhs[i].x) || !isfinite(rhs[i].y)) {
            info->dat = order + i + 1;
            return;
        }
    }
}

//------------------------------------------

__global__ void gather_incident_kernel(const CUCOMPLEX *incident, const CU_INT *dipole_map,
                                       CUCOMPLEX *vectors, int nx, int ny, int nz, int nx2,
                                       int nxy2, int ntotal, int nmax) {
    const int physical = blockIdx.x * blockDim.x + threadIdx.x;
    const int physical_count = nx * ny * nz;
    if (physical >= physical_count)
        return;

    const int dipole = dipole_map[physical].dat;
    if (dipole <= 0 || dipole > nmax)
        return;

    const int i = physical % nx;
    const int j = (physical / nx) % ny;
    const int k = physical / (nx * ny);
    const int padded = i + nx2 * j + nxy2 * k;
    const int source = 3 * (dipole - 1);
    vectors[padded] = incident[source];
    vectors[padded + ntotal] = incident[source + 1];
    vectors[padded + 2 * ntotal] = incident[source + 2];
}

//------------------------------------------

__global__ void multiply_green_tensor_kernel(GreenTensorDeviceView green, CUCOMPLEX *vectors, int nx,
                                             int ny, int nz, int nx2, int ny2, int nz2,
                                             size_t tensor_count) {
    const size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t plane = size_t(nx2) * size_t(ny2);
    const size_t count = plane * size_t(nz2);
    if (flat >= count)
        return;

    const int i = int(flat % size_t(nx2));
    const int j = int((flat / size_t(nx2)) % size_t(ny2));
    const int k = int(flat / plane);
    const int ti = i <= nx ? i : nx2 - i;
    const int tj = j <= ny ? j : ny2 - j;
    const int tk = k <= nz ? k : nz2 - k;
    if (ti > nx || tj > ny || tk > nz) {
        vectors[flat] = CUFFT_ZERO;
        vectors[flat + count] = CUFFT_ZERO;
        vectors[flat + 2 * count] = CUFFT_ZERO;
        return;
    }

    const double sx = i <= nx ? 1.0 : -1.0;
    const double sy = j <= ny ? 1.0 : -1.0;
    const double sz = k <= nz ? 1.0 : -1.0;
    const size_t tensor_index =
        size_t(ti) + size_t(nx + 1) * size_t(tj) + size_t(nx + 1) * size_t(ny + 1) * size_t(tk);
    if (tensor_index >= tensor_count) {
        vectors[flat] = CUFFT_ZERO;
        vectors[flat + count] = CUFFT_ZERO;
        vectors[flat + 2 * count] = CUFFT_ZERO;
        return;
    }

    const CUCOMPLEX x = vectors[flat];
    const CUCOMPLEX y = vectors[flat + count];
    const CUCOMPLEX z = vectors[flat + 2 * count];
    const CUCOMPLEX txx = green.data[0][tensor_index];
    const CUCOMPLEX txy = green.data[1][tensor_index];
    const CUCOMPLEX txz = green.data[2][tensor_index];
    const CUCOMPLEX tyy = green.data[3][tensor_index];
    const CUCOMPLEX tyz = green.data[4][tensor_index];
    const CUCOMPLEX tzz = green.data[5][tensor_index];

    vectors[flat] =
        CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(txx, x), CUFFT_MUL(CUFFT_MUL(txy, y), CUFFT_REAL(sx * sy))),
                  CUFFT_MUL(CUFFT_MUL(txz, z), CUFFT_REAL(sx * sz)));
    vectors[flat + count] =
        CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(CUFFT_MUL(txy, x), CUFFT_REAL(sx * sy)), CUFFT_MUL(tyy, y)),
                  CUFFT_MUL(CUFFT_MUL(tyz, z), CUFFT_REAL(sy * sz)));
    vectors[flat + 2 * count] =
        CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(CUFFT_MUL(txz, x), CUFFT_REAL(sx * sz)),
                            CUFFT_MUL(CUFFT_MUL(tyz, y), CUFFT_REAL(sy * sz))),
                  CUFFT_MUL(tzz, z));
}

//------------------------------------------

__global__ void update_output_kernel(const CUCOMPLEX *vectors, const CU_INT *dipole_map,
                                     CUCOMPLEX *output, int nx, int ny, int nz, int nx2, int nxy2,
                                     int nmax, int ntotal) {
    const int physical = blockIdx.x * blockDim.x + threadIdx.x;
    const int physical_count = nx * ny * nz;
    if (physical >= physical_count)
        return;

    const int dipole = dipole_map[physical].dat;
    if (dipole <= 0 || dipole > nmax)
        return;

    const int i = physical % nx;
    const int j = (physical / nx) % ny;
    const int k = physical / (nx * ny);
    const int padded = i + nx2 * j + nxy2 * k;
    const int target = 3 * (dipole - 1);
    const double scale = 1.0 / double(ntotal);
    output[target] = CUFFT_SUB(output[target], CUFFT_MUL(vectors[padded], CUFFT_REAL(scale)));
    output[target + 1] =
        CUFFT_SUB(output[target + 1], CUFFT_MUL(vectors[padded + ntotal], CUFFT_REAL(scale)));
    output[target + 2] =
        CUFFT_SUB(output[target + 2], CUFFT_MUL(vectors[padded + 2 * ntotal], CUFFT_REAL(scale)));
}

//------------------------------------------

__global__ void gather_incident_regular_kernel(const CUCOMPLEX *incident, CUCOMPLEX *vectors,
                                               int nx, int ny, int nz, int nx2, int nxy2,
                                               int ntotal, int nmax) {
    const int physical = blockIdx.x * blockDim.x + threadIdx.x;
    const int physical_count = nx * ny * nz;
    if (physical >= physical_count || physical >= nmax)
        return;

    const int i = physical % nx;
    const int j = (physical / nx) % ny;
    const int k = physical / (nx * ny);
    const int padded = i + nx2 * j + nxy2 * k;
    const int source = 3 * physical;
    vectors[padded] = incident[source];
    vectors[padded + ntotal] = incident[source + 1];
    vectors[padded + 2 * ntotal] = incident[source + 2];
}

//------------------------------------------

__global__ void multiply_green_tensor_opt_kernel(GreenTensorDeviceView green, CUCOMPLEX *vectors, int nx,
                                                 int ny, int nz, int nx2, int ny2, int nz2,
                                                 size_t tensor_count) {
    const size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t plane = size_t(nx2) * size_t(ny2);
    const size_t count = plane * size_t(nz2);
    if (flat >= count)
        return;

    const int i = int(flat % size_t(nx2));
    const int j = int((flat / size_t(nx2)) % size_t(ny2));
    const int k = int(flat / plane);
    const int ti = i <= nx ? i : nx2 - i;
    const int tj = j <= ny ? j : ny2 - j;
    const int tk = k <= nz ? k : nz2 - k;
    const double sx = i <= nx ? 1.0 : -1.0;
    const double sy = j <= ny ? 1.0 : -1.0;
    const double sz = k <= nz ? 1.0 : -1.0;

    const size_t nx1 = size_t(nx) + 1;
    const size_t ny1 = size_t(ny) + 1;
    const size_t tensor_index = size_t(ti) + nx1 * size_t(tj) + nx1 * ny1 * size_t(tk);
    if (tensor_index >= tensor_count) {
        vectors[flat] = CUFFT_ZERO;
        vectors[flat + count] = CUFFT_ZERO;
        vectors[flat + 2 * count] = CUFFT_ZERO;
        return;
    }

    const CUCOMPLEX x = vectors[flat];
    const CUCOMPLEX y = vectors[flat + count];
    const CUCOMPLEX z = vectors[flat + 2 * count];
    const CUCOMPLEX txx = green.data[0][tensor_index];
    const CUCOMPLEX txy = green.data[1][tensor_index];
    const CUCOMPLEX txz = green.data[2][tensor_index];
    const CUCOMPLEX tyy = green.data[3][tensor_index];
    const CUCOMPLEX tyz = green.data[4][tensor_index];
    const CUCOMPLEX tzz = green.data[5][tensor_index];

    vectors[flat] =
        CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(txx, x), CUFFT_MUL(CUFFT_MUL(txy, y), CUFFT_REAL(sx * sy))),
                  CUFFT_MUL(CUFFT_MUL(txz, z), CUFFT_REAL(sx * sz)));
    vectors[flat + count] =
        CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(CUFFT_MUL(txy, x), CUFFT_REAL(sx * sy)), CUFFT_MUL(tyy, y)),
                  CUFFT_MUL(CUFFT_MUL(tyz, z), CUFFT_REAL(sy * sz)));
    vectors[flat + 2 * count] =
        CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(CUFFT_MUL(txz, x), CUFFT_REAL(sx * sz)),
                            CUFFT_MUL(CUFFT_MUL(tyz, y), CUFFT_REAL(sy * sz))),
                  CUFFT_MUL(tzz, z));
}

//------------------------------------------

__global__ void update_output_regular_kernel(const CUCOMPLEX *vectors, CUCOMPLEX *output, int nx,
                                             int ny, int nz, int nx2, int nxy2, int nmax,
                                             int ntotal) {
    const int physical = blockIdx.x * blockDim.x + threadIdx.x;
    const int physical_count = nx * ny * nz;
    if (physical >= physical_count || physical >= nmax)
        return;

    const int i = physical % nx;
    const int j = (physical / nx) % ny;
    const int k = physical / (nx * ny);
    const int padded = i + nx2 * j + nxy2 * k;
    const int target = 3 * physical;
    const double scale = 1.0 / double(ntotal);
    output[target] = CUFFT_SUB(output[target], CUFFT_MUL(vectors[padded], CUFFT_REAL(scale)));
    output[target + 1] =
        CUFFT_SUB(output[target + 1], CUFFT_MUL(vectors[padded + ntotal], CUFFT_REAL(scale)));
    output[target + 2] =
        CUFFT_SUB(output[target + 2], CUFFT_MUL(vectors[padded + 2 * ntotal], CUFFT_REAL(scale)));
}

//-----------------------------------------------------------
__global__ void gather_scalar_product_kernel(const CUCOMPLEX *input, const CU_INT *dipole_map,
                                             CUCOMPLEX *work, int nx, int ny, int nz, int nx2,
                                             int nxy2, int nmax) {
    const int physical = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny * nz;
    if (physical >= count)
        return;
    const int dipole = dipole_map[physical].dat;
    if (dipole <= 0 || dipole > nmax)
        return;
    const int i = physical % nx;
    const int j = (physical / nx) % ny;
    const int k = physical / (nx * ny);
    const int padded = i + nx2 * j + nxy2 * k;
    work[padded] = input[dipole - 1];
}

//-----------------------------------------------------------
__global__ void gather_scalar_product_dense_kernel(const CUCOMPLEX *input, CUCOMPLEX *work,
                                                    int nx, int ny, int nz, int nx2,
                                                    int nxy2) {
    const int physical = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nx * ny * nz;
    if (physical >= count)
        return;
    const int i = physical % nx;
    const int j = (physical / nx) % ny;
    const int k = physical / (nx * ny);
    const int padded = i + nx2 * j + nxy2 * k;
    work[padded] = input[physical];
}

//-----------------------------------------------------------
__global__ void multiply_scalar_green_kernel(const CUCOMPLEX *green, CUCOMPLEX *work, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        work[index] = CUFFT_MUL(work[index], green[index]);
}

//-----------------------------------------------------------
__global__ void scatter_scalar_product_kernel(const CUCOMPLEX *work, const CU_INT *dipole_map,
                                              CUCOMPLEX *output, int nx, int ny, int nz, int nx2,
                                              int nxy2, int nmax, int ntotal) {
    const int physical = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny * nz;
    if (physical >= count)
        return;
    const int dipole = dipole_map[physical].dat;
    if (dipole <= 0 || dipole > nmax)
        return;
    const int i = physical % nx;
    const int j = (physical / nx) % ny;
    const int k = physical / (nx * ny);
    const int padded = i + nx2 * j + nxy2 * k;
    const CUCOMPLEX scale = CUFFT_REAL(CU_REAL(1) / CU_REAL(ntotal));
    output[dipole - 1] = CUFFT_SUB(output[dipole - 1], CUFFT_MUL(work[padded], scale));
}

//-----------------------------------------------------------
__global__ void scatter_scalar_product_dense_kernel(const CUCOMPLEX *work, CUCOMPLEX *output,
                                                    int nx, int ny, int nz, int nx2,
                                                    int nxy2, int ntotal) {
    const int physical = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nx * ny * nz;
    if (physical >= count)
        return;
    const int i = physical % nx;
    const int j = (physical / nx) % ny;
    const int k = physical / (nx * ny);
    const int padded = i + nx2 * j + nxy2 * k;
    const CUCOMPLEX scaled = CUFFT_MUL(work[padded], CUFFT_REAL(CU_REAL(1) / CU_REAL(ntotal)));
    output[physical] = CUFFT_SUB(output[physical], scaled);
}

//------------------------------------------

__global__ void gpbicg_save_initial_kernel(const CUCOMPLEX *xi, CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(10) * lda] = xi[i];
}

//-----------------------------------------------------------
__global__ void gpbicg_initialize_kernel(const CUCOMPLEX *xr, const CUCOMPLEX *b, CUCOMPLEX *work,
                                         int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX residual = CUFFT_SUB(b[i], xr[i]);
    work[i + size_t(1) * lda] = CUFFT_ZERO;
    work[i + size_t(2) * lda] = residual;
    work[i] = CUFFT_CONJ(residual);
    work[i + size_t(4) * lda] = CUFFT_ZERO;
    work[i + size_t(7) * lda] = CUFFT_ZERO;
    work[i + size_t(8) * lda] = CUFFT_ZERO;
    work[i + size_t(9) * lda] = CUFFT_ZERO;
}

//-----------------------------------------------------------
__global__ void gpbicg2_initialize_kernel(const CUCOMPLEX *xr, const CUCOMPLEX *b, CUCOMPLEX *work,
                                          int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX residual = CUFFT_SUB(b[i], xr[i]);
    work[i + size_t(1) * lda] = CUFFT_ZERO;
    work[i + size_t(2) * lda] = residual;
    work[i] = residual;
    work[i + size_t(4) * lda] = CUFFT_ZERO;
    work[i + size_t(7) * lda] = CUFFT_ZERO;
    work[i + size_t(8) * lda] = CUFFT_ZERO;
    work[i + size_t(9) * lda] = CUFFT_ZERO;
}

//-----------------------------------------------------------
__global__ void gpbicg_prepare_direction_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX beta,
                                                int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CUCOMPLEX *p = work + i + size_t(1) * lda;
    const CUCOMPLEX r = work[i + size_t(2) * lda];
    const CUCOMPLEX u = work[i + size_t(7) * lda];
    *p = CUFFT_ADD(r, CUFFT_MUL(beta, CUFFT_SUB(*p, u)));
    xi[i] = *p;
}

//-----------------------------------------------------------
__global__ void gpbicg_save_xr_column_kernel(const CUCOMPLEX *xr, CUCOMPLEX *work, int lda, int n,
                                             int column) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(column) * lda] = xr[i];
}

//-----------------------------------------------------------
__global__ void gpbicg_update_stage2_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX alpha,
                                            int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX r = work[i + size_t(2) * lda];
    const CUCOMPLEX ap = work[i + size_t(5) * lda];
    const CUCOMPLEX w = work[i + size_t(8) * lda];
    const CUCOMPLEX t = work[i + size_t(4) * lda];
    work[i + size_t(3) * lda] = CUFFT_ADD(CUFFT_SUB(t, r), CUFFT_MUL(alpha, CUFFT_SUB(ap, w)));
    work[i + size_t(11) * lda] = t;
    work[i + size_t(4) * lda] = CUFFT_SUB(r, CUFFT_MUL(alpha, ap));
    xi[i] = work[i + size_t(4) * lda];
}

//-----------------------------------------------------------
__global__ void gpbicg_update_stage3_kernel(CUCOMPLEX *work, CUCOMPLEX alpha, CUCOMPLEX beta,
                                            CUCOMPLEX eta, CUCOMPLEX dzeta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CUCOMPLEX *r = work + i + size_t(2) * lda;
    CUCOMPLEX *u = work + i + size_t(7) * lda;
    CUCOMPLEX *z = work + i + size_t(9) * lda;
    const CUCOMPLEX ap = work[i + size_t(5) * lda];
    const CUCOMPLEX y = work[i + size_t(3) * lda];
    const CUCOMPLEX old_t = work[i + size_t(11) * lda];
    const CUCOMPLEX p = work[i + size_t(1) * lda];
    const CUCOMPLEX x = work[i + size_t(10) * lda];
    const CUCOMPLEX w = work[i + size_t(4) * lda];
    *u = CUFFT_ADD(CUFFT_MUL(dzeta, ap),
                   CUFFT_MUL(eta, CUFFT_ADD(CUFFT_SUB(old_t, *r), CUFFT_MUL(beta, *u))));
    *z = CUFFT_SUB(CUFFT_ADD(CUFFT_MUL(dzeta, *r), CUFFT_MUL(eta, *z)), CUFFT_MUL(alpha, *u));
    work[i + size_t(10) * lda] = CUFFT_ADD(x, CUFFT_ADD(CUFFT_MUL(alpha, p), *z));
    *r = CUFFT_SUB(CUFFT_SUB(w, CUFFT_MUL(eta, y)), CUFFT_MUL(dzeta, work[i + size_t(6) * lda]));
}

//-----------------------------------------------------------
__global__ void gpbicg_update_stage4_kernel(CUCOMPLEX *work, CUCOMPLEX beta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(8) * lda] =
            CUFFT_ADD(work[i + size_t(6) * lda], CUFFT_MUL(beta, work[i + size_t(5) * lda]));
}

//-----------------------------------------------------------
__global__ void gpbicg_publish_solution_kernel(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda,
                                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = work[i + size_t(10) * lda];
}

//-----------------------------------------------------------
__global__ void gpbicgplus_initialize_kernel(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                             CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX residual = CUFFT_SUB(b[i], xr[i]);
    work[i] = residual;
    work[i + size_t(1) * lda] = CUFFT_ZERO;
    work[i + size_t(2) * lda] = residual;
    work[i + size_t(3) * lda] = CUFFT_ZERO;
    work[i + size_t(6) * lda] = CUFFT_ZERO;
    work[i + size_t(7) * lda] = CUFFT_ZERO;
    work[i + size_t(8) * lda] = CUFFT_ZERO;
    work[i + size_t(9) * lda] = CUFFT_ZERO;
}

//-----------------------------------------------------------
__global__ void gpbicgplus_prepare_direction_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX beta,
                                                    int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CUCOMPLEX *p = work + i + size_t(1) * lda;
    const CUCOMPLEX r = work[i + size_t(2) * lda];
    const CUCOMPLEX u = work[i + size_t(7) * lda];
    *p = CUFFT_ADD(r, CUFFT_MUL(beta, CUFFT_SUB(*p, u)));
    xi[i] = r;
}

//-----------------------------------------------------------
__global__ void gpbicgplus_update_ap_kernel(const CUCOMPLEX *xr, CUCOMPLEX *xi, CUCOMPLEX *work,
                                            CUCOMPLEX beta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX ar = xr[i];
    const CUCOMPLEX au = work[i + size_t(6) * lda];
    CUCOMPLEX *ap = work + i + size_t(3) * lda;
    *ap = CUFFT_ADD(ar, CUFFT_MUL(beta, CUFFT_SUB(*ap, au)));
    work[i + size_t(5) * lda] = ar;
    xi[i] = work[i + size_t(7) * lda];
}

//-----------------------------------------------------------
__global__ void gpbicgplus_update_stage2_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX alpha,
                                                CUCOMPLEX beta, CUCOMPLEX eta, CUCOMPLEX dzeta,
                                                int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CUCOMPLEX *u = work + i + size_t(7) * lda;
    const CUCOMPLEX ap = work[i + size_t(3) * lda];
    const CUCOMPLEX r = work[i + size_t(2) * lda];
    const CUCOMPLEX az = work[i + size_t(8) * lda];
    const CUCOMPLEX z = work[i + size_t(9) * lda];
    *u = CUFFT_ADD(CUFFT_MUL(dzeta, ap), CUFFT_MUL(eta, CUFFT_ADD(az, CUFFT_MUL(beta, *u))));
    work[i + size_t(4) * lda] = CUFFT_SUB(r, CUFFT_MUL(alpha, ap));
    work[i + size_t(9) * lda] =
        CUFFT_SUB(CUFFT_ADD(CUFFT_MUL(dzeta, r), CUFFT_MUL(eta, z)), CUFFT_MUL(alpha, *u));
    xi[i] = *u;
}

//-----------------------------------------------------------
__global__ void gpbicgplus_update_stage3_kernel(CUCOMPLEX *work, const CUCOMPLEX *xr,
                                                CUCOMPLEX alpha, CUCOMPLEX eta, CUCOMPLEX dzeta,
                                                int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX au = xr[i];
    work[i + size_t(6) * lda] = au;
    const CUCOMPLEX az = work[i + size_t(8) * lda];
    work[i + size_t(8) * lda] =
        CUFFT_SUB(CUFFT_ADD(CUFFT_MUL(dzeta, work[i + size_t(5) * lda]), CUFFT_MUL(eta, az)),
                  CUFFT_MUL(alpha, au));
    const CUCOMPLEX z = work[i + size_t(9) * lda];
    work[i + size_t(10) * lda] = CUFFT_ADD(
        work[i + size_t(10) * lda], CUFFT_ADD(CUFFT_MUL(alpha, work[i + size_t(1) * lda]), z));
    const CUCOMPLEX residual = CUFFT_SUB(work[i + size_t(4) * lda], work[i + size_t(8) * lda]);
    work[i + size_t(2) * lda] = residual;
}

//-----------------------------------------------------------
__global__ void gpbicgsafe_initialize_kernel(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                             CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX residual = CUFFT_SUB(b[i], xr[i]);
    work[i] = residual;
    work[i + size_t(1) * lda] = CUFFT_ZERO;
    work[i + size_t(2) * lda] = CUFFT_ZERO;
    work[i + size_t(3) * lda] = residual;
    work[i + size_t(4) * lda] = CUFFT_ZERO;
    work[i + size_t(6) * lda] = CUFFT_ZERO;
    work[i + size_t(7) * lda] = CUFFT_ZERO;
    work[i + size_t(8) * lda] = CUFFT_ZERO;
    work[i + size_t(9) * lda] = CUFFT_ZERO;
}

//-----------------------------------------------------------
__global__ void gpbicgsafe_prepare_direction_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX beta,
                                                    int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CUCOMPLEX *p = work + i + size_t(1) * lda;
    const CUCOMPLEX r = work[i + size_t(3) * lda];
    const CUCOMPLEX u = work[i + size_t(2) * lda];
    *p = CUFFT_ADD(r, CUFFT_MUL(beta, CUFFT_SUB(*p, u)));
    xi[i] = r;
}

//-----------------------------------------------------------
__global__ void gpbicgsafe_update_ap_kernel(const CUCOMPLEX *xr, CUCOMPLEX *xi, CUCOMPLEX *work,
                                            CUCOMPLEX beta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX ar = xr[i];
    work[i + size_t(5) * lda] = ar;
    CUCOMPLEX *ap = work + i + size_t(6) * lda;
    *ap = CUFFT_ADD(ar, CUFFT_MUL(beta, CUFFT_SUB(*ap, work[i + size_t(4) * lda])));
}

//-----------------------------------------------------------
__global__ void gpbicgsafe_update_stage2_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX beta,
                                                CUCOMPLEX eta, CUCOMPLEX dzeta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CUCOMPLEX *u = work + i + size_t(2) * lda;
    const CUCOMPLEX ap = work[i + size_t(6) * lda];
    const CUCOMPLEX az = work[i + size_t(9) * lda];
    *u = CUFFT_ADD(CUFFT_MUL(dzeta, ap), CUFFT_MUL(eta, CUFFT_ADD(az, CUFFT_MUL(beta, *u))));
    xi[i] = *u;
}

//-----------------------------------------------------------
__global__ void gpbicgsafe_update_stage3_kernel(CUCOMPLEX *work, const CUCOMPLEX *xr,
                                                CUCOMPLEX alpha, CUCOMPLEX eta, CUCOMPLEX dzeta,
                                                int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX au = xr[i];
    work[i + size_t(4) * lda] = au;
    const CUCOMPLEX r = work[i + size_t(3) * lda];
    const CUCOMPLEX ap = work[i + size_t(6) * lda];
    const CUCOMPLEX u = work[i + size_t(2) * lda];
    const CUCOMPLEX ar = work[i + size_t(5) * lda];
    const CUCOMPLEX az = work[i + size_t(9) * lda];
    const CUCOMPLEX z = work[i + size_t(8) * lda];
    const CUCOMPLEX p = work[i + size_t(1) * lda];
    work[i + size_t(7) * lda] = CUFFT_SUB(r, CUFFT_MUL(alpha, ap));
    work[i + size_t(8) * lda] =
        CUFFT_SUB(CUFFT_ADD(CUFFT_MUL(dzeta, r), CUFFT_MUL(eta, z)), CUFFT_MUL(alpha, u));
    work[i + size_t(9) * lda] =
        CUFFT_SUB(CUFFT_ADD(CUFFT_MUL(dzeta, ar), CUFFT_MUL(eta, az)), CUFFT_MUL(alpha, au));
    work[i + size_t(10) * lda] = CUFFT_ADD(
        work[i + size_t(10) * lda], CUFFT_ADD(CUFFT_MUL(alpha, p), work[i + size_t(8) * lda]));
    work[i + size_t(3) * lda] = CUFFT_SUB(work[i + size_t(7) * lda], work[i + size_t(9) * lda]);
}

//-----------------------------------------------------------
__global__ void gpbicgar_initialize_kernel(const CUCOMPLEX *xr, const CUCOMPLEX *b, CUCOMPLEX *xi,
                                           CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX r = CUFFT_SUB(b[i], xr[i]);
    work[i] = r;
    work[i + size_t(2) * lda] = r;
    xi[i] = r;
}

//-----------------------------------------------------------
__global__ void gpbicgar_update_ap_kernel(const CUCOMPLEX *xr, CUCOMPLEX *work, CUCOMPLEX beta,
                                          int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX ar = xr[i];
    work[i + size_t(6) * lda] = ar;
    CUCOMPLEX *p = work + i + size_t(1) * lda;
    CUCOMPLEX *ap = work + i + size_t(5) * lda;
    *p = CUFFT_ADD(work[i + size_t(2) * lda],
                   CUFFT_MUL(beta, CUFFT_SUB(*p, work[i + size_t(7) * lda])));
    *ap = CUFFT_ADD(ar, CUFFT_MUL(beta, CUFFT_SUB(*ap, work[i + size_t(11) * lda])));
}

//-----------------------------------------------------------
__global__ void gpbicgar_update_u_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX beta,
                                         CUCOMPLEX eta, CUCOMPLEX dzeta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CUCOMPLEX *u = work + i + size_t(7) * lda;
    *u = CUFFT_ADD(
        CUFFT_MUL(dzeta, work[i + size_t(5) * lda]),
        CUFFT_MUL(eta, CUFFT_ADD(CUFFT_SUB(work[i + size_t(3) * lda], work[i + size_t(2) * lda]),
                                 CUFFT_MUL(beta, *u))));
    xi[i] = *u;
}

//-----------------------------------------------------------
__global__ void gpbicgar_update_stage_kernel(CUCOMPLEX *work, const CUCOMPLEX *xr, CUCOMPLEX alpha,
                                             CUCOMPLEX eta, CUCOMPLEX dzeta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    work[i + size_t(11) * lda] = xr[i];
    const CUCOMPLEX r = work[i + size_t(2) * lda];
    const CUCOMPLEX ap = work[i + size_t(5) * lda];
    const CUCOMPLEX u = work[i + size_t(7) * lda];
    const CUCOMPLEX ar = work[i + size_t(6) * lda];
    const CUCOMPLEX old_z = work[i + size_t(4) * lda];
    const CUCOMPLEX az = work[i + size_t(8) * lda];
    const CUCOMPLEX p = work[i + size_t(1) * lda];
    const CUCOMPLEX t = CUFFT_SUB(r, CUFFT_MUL(alpha, ap));
    const CUCOMPLEX z =
        CUFFT_SUB(CUFFT_ADD(CUFFT_MUL(dzeta, r), CUFFT_MUL(eta, old_z)), CUFFT_MUL(alpha, u));
    const CUCOMPLEX new_az =
        CUFFT_SUB(CUFFT_ADD(CUFFT_MUL(dzeta, ar), CUFFT_MUL(eta, az)), CUFFT_MUL(alpha, xr[i]));
    work[i + size_t(3) * lda] = t;
    work[i + size_t(4) * lda] = z;
    work[i + size_t(8) * lda] = new_az;
    work[i + size_t(10) * lda] =
        CUFFT_ADD(work[i + size_t(10) * lda], CUFFT_ADD(CUFFT_MUL(alpha, p), z));
    work[i + size_t(2) * lda] = CUFFT_SUB(t, new_az);
}

//-----------------------------------------------------------
__global__ void gpbicgar_publish_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda,
                                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = work[i + size_t(2) * lda];
}

//-----------------------------------------------------------
__global__ void gpbicgar_copy_ar_kernel(const CUCOMPLEX *xr, CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(6) * lda] = xr[i];
}

//-----------------------------------------------------------
__global__ void gpbicor_initialize_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                          CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX r = CUFFT_SUB(b[i], xr[i]);
    work[i] = r;
    work[i + size_t(9) * lda] = r;
    xi[i] = r;
}

//-----------------------------------------------------------
__global__ void gpbicor_first_direction_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, const CUCOMPLEX *xr,
                                               int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    work[i + size_t(1) * lda] = work[i];
    work[i + size_t(2) * lda] = xr[i];
    work[i + size_t(3) * lda] = xr[i];
    work[i + size_t(11) * lda] = CUFFT_ZERO;
    work[i + size_t(4) * lda] = CUFFT_ZERO;
    work[i + size_t(7) * lda] = CUFFT_ZERO;
    work[i + size_t(8) * lda] = CUFFT_ZERO;
    work[i + size_t(5) * lda] = CUFFT_ZERO;
    xi[i] = xr[i];
}

//-----------------------------------------------------------
__global__ void gpbicor_update_directions_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                 CUCOMPLEX *work, CUCOMPLEX beta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CUCOMPLEX *p = work + i + size_t(1) * lda;
    CUCOMPLEX *q = work + i + size_t(3) * lda;
    work[i + size_t(2) * lda] = xr[i];
    *p = CUFFT_ADD(work[i + size_t(9) * lda],
                   CUFFT_MUL(beta, CUFFT_SUB(*p, work[i + size_t(7) * lda])));
    work[i + size_t(4) * lda] = CUFFT_ADD(work[i + size_t(5) * lda], CUFFT_MUL(beta, *q));
    *q = CUFFT_ADD(work[i + size_t(2) * lda],
                   CUFFT_MUL(beta, CUFFT_SUB(*q, work[i + size_t(8) * lda])));
    xi[i] = *q;
}

//-----------------------------------------------------------
__global__ void gpbicor_prepare_stage_kernel(CUCOMPLEX *xi, CUCOMPLEX *xr, CUCOMPLEX *work,
                                             CUCOMPLEX alpha, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    xi[i] = CUFFT_ADD(
        CUFFT_SUB(work[i + size_t(11) * lda], work[i + size_t(9) * lda]),
        CUFFT_MUL(alpha, CUFFT_SUB(work[i + size_t(3) * lda], work[i + size_t(4) * lda])));
    work[i + size_t(6) * lda] =
        CUFFT_SUB(work[i + size_t(9) * lda], CUFFT_MUL(alpha, work[i + size_t(3) * lda]));
    xr[i] = CUFFT_SUB(work[i + size_t(2) * lda], CUFFT_MUL(alpha, xr[i]));
}

//-----------------------------------------------------------
__global__ void gpbicor_finish_stage_kernel(CUCOMPLEX *ffloc, CUCOMPLEX *work, const CUCOMPLEX *xi,
                                            const CUCOMPLEX *xr, CUCOMPLEX alpha, CUCOMPLEX beta,
                                            CUCOMPLEX eta, CUCOMPLEX dzeta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX old_u = work[i + size_t(7) * lda];
    const CUCOMPLEX u = CUFFT_ADD(
        CUFFT_MUL(dzeta, work[i + size_t(3) * lda]),
        CUFFT_MUL(eta, CUFFT_ADD(CUFFT_SUB(work[i + size_t(11) * lda], work[i + size_t(9) * lda]),
                                 CUFFT_MUL(beta, old_u))));
    const CUCOMPLEX qtilde = CUFFT_ADD(
        CUFFT_MUL(dzeta, CUFFT_DIV(CUFFT_SUB(work[i + size_t(2) * lda], xr[i]), alpha)),
        CUFFT_MUL(eta, CUFFT_ADD(CUFFT_SUB(work[i + size_t(5) * lda], work[i + size_t(2) * lda]),
                                 CUFFT_MUL(beta, work[i + size_t(8) * lda]))));
    const CUCOMPLEX y = CUFFT_SUB(CUFFT_ADD(CUFFT_MUL(dzeta, work[i + size_t(9) * lda]),
                                            CUFFT_MUL(eta, work[i + size_t(10) * lda])),
                                  CUFFT_MUL(alpha, u));
    const CUCOMPLEX residual = CUFFT_SUB(
        CUFFT_SUB(work[i + size_t(6) * lda], CUFFT_MUL(eta, xi[i])), CUFFT_MUL(dzeta, xr[i]));
    ffloc[i] = CUFFT_ADD(ffloc[i], CUFFT_ADD(CUFFT_MUL(alpha, work[i + size_t(1) * lda]), y));
    work[i + size_t(7) * lda] = u;
    work[i + size_t(8) * lda] = qtilde;
    work[i + size_t(10) * lda] = y;
    work[i + size_t(9) * lda] = residual;
    work[i + size_t(5) * lda] = xr[i];
    work[i + size_t(11) * lda] = work[i + size_t(6) * lda];
}

//-----------------------------------------------------------
__global__ void gpbicor_publish_solution_kernel(CUCOMPLEX *xi, const CUCOMPLEX *ffloc, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = ffloc[i];
}

//-----------------------------------------------------------
__global__ void gpbicor_publish_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda,
                                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = work[i + size_t(9) * lda];
}

//-----------------------------------------------------------
__global__ void cors_save_initial_kernel(const CUCOMPLEX *xi, CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(9) * lda] = xi[i];
}

//-----------------------------------------------------------
__global__ void cors_initialize_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                const CUCOMPLEX *b, CUCOMPLEX *work, int lda,
                                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX r = CUFFT_SUB(b[i], xr[i]);
        work[i + size_t(1) * lda] = r;
        xi[i] = r;
    }
}

//-----------------------------------------------------------
__global__ void cors_initialize_direction_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                 CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        work[i] = CUFFT_CONJ(xr[i]);
        work[i + size_t(2) * lda] = xr[i];
        work[i + size_t(3) * lda] = work[i + size_t(1) * lda];
        work[i + size_t(4) * lda] = xr[i];
        work[i + size_t(5) * lda] = xr[i];
        xi[i] = xr[i];
    }
}

//-----------------------------------------------------------
__global__ void cors_update_direction_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr, CUCOMPLEX *work,
                                             CUCOMPLEX beta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        work[i + size_t(2) * lda] = xr[i];
        work[i + size_t(3) * lda] =
            CUFFT_ADD(work[i + size_t(1) * lda], CUFFT_MUL(beta, work[i + size_t(6) * lda]));
        work[i + size_t(4) * lda] =
            CUFFT_ADD(work[i + size_t(2) * lda], CUFFT_MUL(beta, work[i + size_t(7) * lda]));
        work[i + size_t(5) * lda] =
            CUFFT_ADD(work[i + size_t(4) * lda],
                      CUFFT_MUL(beta, CUFFT_ADD(work[i + size_t(7) * lda],
                                                CUFFT_MUL(beta, work[i + size_t(5) * lda]))));
        xi[i] = work[i + size_t(5) * lda];
    }
}

//-----------------------------------------------------------
__global__ void cors_update_stage_kernel(CUCOMPLEX *work, const CUCOMPLEX *xr, CUCOMPLEX alpha,
                                         int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX q = xr[i];
    const CUCOMPLEX e = work[i + size_t(3) * lda];
    const CUCOMPLEX d = work[i + size_t(4) * lda];
    const CUCOMPLEX v = work[i + size_t(5) * lda];
    const CUCOMPLEX twice = CUFFT_REAL(2.0);
    work[i + size_t(8) * lda] = q;
    work[i + size_t(6) * lda] = CUFFT_SUB(e, CUFFT_MUL(alpha, v));
    work[i + size_t(7) * lda] = CUFFT_SUB(d, CUFFT_MUL(alpha, q));
    work[i + size_t(9) * lda] =
        CUFFT_ADD(work[i + size_t(9) * lda],
                  CUFFT_MUL(alpha, CUFFT_SUB(CUFFT_MUL(twice, e), CUFFT_MUL(alpha, v))));
    work[i + size_t(1) * lda] =
        CUFFT_SUB(work[i + size_t(1) * lda],
                  CUFFT_MUL(alpha, CUFFT_SUB(CUFFT_MUL(twice, d), CUFFT_MUL(alpha, q))));
}

//-----------------------------------------------------------
__global__ void cors_publish_solution_kernel(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = work[i + size_t(9) * lda];
}

//-----------------------------------------------------------
__global__ void cors_publish_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = work[i + size_t(1) * lda];
}

//-----------------------------------------------------------
__global__ void tfqmr_initialize_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                 const CUCOMPLEX *b, CUCOMPLEX *work, int lda,
                                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX r = CUFFT_SUB(b[i], xr[i]);
        work[i + size_t(2) * lda] = r;
        work[i + size_t(1) * lda] = r;
        work[i] = r;
        xi[i] = r;
    }
}

//-----------------------------------------------------------
__global__ void tfqmr_initialize_vectors_kernel(const CUCOMPLEX *xr, CUCOMPLEX *work, int lda,
                                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        work[i + size_t(3) * lda] = xr[i];
        work[i + size_t(4) * lda] = CUFFT_ZERO;
        work[i + size_t(7) * lda] = xr[i];
        work[i + size_t(5) * lda] = work[i + size_t(2) * lda];
    }
}

//-----------------------------------------------------------
__global__ void tfqmr_prepare_uplus_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX alpha, int lda,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        work[i + size_t(6) * lda] = CUFFT_SUB(work[i], CUFFT_MUL(alpha, work[i + size_t(3) * lda]));
        xi[i] = work[i + size_t(6) * lda];
    }
}

//-----------------------------------------------------------
__global__ void tfqmr_update_w_kernel(CUCOMPLEX *work, CUCOMPLEX alpha, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(1) * lda] =
            CUFFT_SUB(work[i + size_t(1) * lda], CUFFT_MUL(alpha, work[i + size_t(7) * lda]));
}

//-----------------------------------------------------------
__global__ void tfqmr_update_halfstep_kernel(CUCOMPLEX *x, CUCOMPLEX *work, CUCOMPLEX dcoef,
                                             CUCOMPLEX eta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        work[i + size_t(4) * lda] = CUFFT_ADD(work[i], CUFFT_MUL(dcoef, work[i + size_t(4) * lda]));
        x[i] = CUFFT_ADD(x[i], CUFFT_MUL(eta, work[i + size_t(4) * lda]));
    }
}

//-----------------------------------------------------------
__global__ void tfqmr_continue_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr, CUCOMPLEX *work,
                                      CUCOMPLEX beta, int odd, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        work[i + size_t(7) * lda] = xr[i];
        if (odd) {
            work[i + size_t(3) * lda] = CUFFT_ADD(
                xr[i], CUFFT_MUL(beta, CUFFT_ADD(work[i + size_t(8) * lda],
                                                 CUFFT_MUL(beta, work[i + size_t(3) * lda]))));
        }
        xi[i] = work[i + size_t(6) * lda];
        work[i] = work[i + size_t(6) * lda];
    }
}

//-----------------------------------------------------------
__global__ void tfqmr_end_halfstep_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX beta, int odd,
                                          int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (odd) {
            work[i + size_t(6) * lda] =
                CUFFT_ADD(work[i + size_t(1) * lda], CUFFT_MUL(beta, work[i]));
            work[i + size_t(8) * lda] = work[i + size_t(7) * lda];
        }
        xi[i] = work[i + size_t(6) * lda];
        work[i] = work[i + size_t(6) * lda];
    }
}

//-----------------------------------------------------------
__global__ void bicgstarplus_save_solution_kernel(const CUCOMPLEX *xi, CUCOMPLEX *work, int lda,
                                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(10) * lda] = xi[i];
}

//-----------------------------------------------------------
__global__ void bicgstarplus_initialize_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                        const CUCOMPLEX *b, CUCOMPLEX *work,
                                                        int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX r = CUFFT_SUB(b[i], xr[i]);
        work[i + size_t(2) * lda] = r;
        work[i] = CUFFT_CONJ(r);
        work[i + size_t(3) * lda] = CUFFT_ZERO;
        work[i + size_t(4) * lda] = CUFFT_ZERO;
        work[i + size_t(7) * lda] = CUFFT_ZERO;
        work[i + size_t(8) * lda] = CUFFT_ZERO;
        xi[i] = r;
    }
}

//-----------------------------------------------------------
__global__ void bicgstarplus_update_direction_kernel(CUCOMPLEX *xi, CUCOMPLEX *work,
                                                     CUCOMPLEX alpha, CUCOMPLEX beta, CUCOMPLEX eta,
                                                     CUCOMPLEX dzeta, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX ap =
            CUFFT_SUB(work[i + size_t(1) * lda], CUFFT_MUL(beta, work[i + size_t(11) * lda]));
        const CUCOMPLEX v = CUFFT_ADD(CUFFT_MUL(dzeta, work[i + size_t(2) * lda]),
                                      CUFFT_MUL(eta, work[i + size_t(8) * lda]));
        const CUCOMPLEX c =
            CUFFT_ADD(CUFFT_MUL(dzeta, ap),
                      CUFFT_MUL(eta, CUFFT_SUB(work[i + size_t(4) * lda],
                                               CUFFT_MUL(beta, work[i + size_t(3) * lda]))));
        work[i + size_t(6) * lda] = ap;
        work[i + size_t(5) * lda] = v;
        work[i + size_t(3) * lda] = c;
        xi[i] = c;
    }
}

//-----------------------------------------------------------
__global__ void bicgstarplus_update_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                    CUCOMPLEX *work, CUCOMPLEX alpha,
                                                    CUCOMPLEX beta, CUCOMPLEX eta, CUCOMPLEX dzeta,
                                                    int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX ac = xr[i];
        const CUCOMPLEX w = CUFFT_SUB(
            CUFFT_SUB(work[i + size_t(2) * lda], CUFFT_MUL(beta, work[i + size_t(7) * lda])),
            work[i + size_t(3) * lda]);
        const CUCOMPLEX aw = CUFFT_SUB(work[i + size_t(6) * lda], ac);
        const CUCOMPLEX y = CUFFT_ADD(CUFFT_MUL(dzeta, work[i + size_t(1) * lda]),
                                      CUFFT_MUL(eta, work[i + size_t(4) * lda]));
        const CUCOMPLEX t = CUFFT_SUB(y, CUFFT_MUL(alpha, ac));
        const CUCOMPLEX r =
            CUFFT_SUB(CUFFT_SUB(work[i + size_t(2) * lda], y), CUFFT_MUL(alpha, aw));
        work[i + size_t(9) * lda] = ac;
        work[i + size_t(7) * lda] = w;
        work[i + size_t(11) * lda] = aw;
        work[i + size_t(8) * lda] =
            CUFFT_SUB(work[i + size_t(5) * lda], CUFFT_MUL(alpha, work[i + size_t(3) * lda]));
        xi[i] = y;
        work[i + size_t(4) * lda] = t;
        work[i + size_t(10) * lda] = CUFFT_ADD(
            work[i + size_t(10) * lda], CUFFT_ADD(work[i + size_t(5) * lda], CUFFT_MUL(alpha, w)));
        work[i + size_t(2) * lda] = r;
    }
}

//-----------------------------------------------------------
__global__ void bicgstarplus_publish_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda,
                                                     int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = work[i + size_t(2) * lda];
}

//-----------------------------------------------------------
__global__ void zcg_save_solution_kernel(const CUCOMPLEX *xi, CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i] = xi[i];
}

//-----------------------------------------------------------
__global__ void zcg_initialize_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                               const CUCOMPLEX *b, CUCOMPLEX *work, int lda,
                                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX r = CUFFT_SUB(b[i], xr[i]);
        work[i + size_t(1) * lda] = r;
        work[i + size_t(2) * lda] = r;
        xi[i] = r;
    }
}

//-----------------------------------------------------------
__global__ void zcg_save_initial_w_kernel(const CUCOMPLEX *xr, CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(4) * lda] = xr[i];
}

//-----------------------------------------------------------
__global__ void zcg_step_kernel(CUCOMPLEX *work, CUCOMPLEX alpha, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        work[i] = CUFFT_ADD(work[i], CUFFT_MUL(alpha, work[i + size_t(2) * lda]));
        work[i + size_t(1) * lda] =
            CUFFT_SUB(work[i + size_t(1) * lda], CUFFT_MUL(alpha, work[i + size_t(4) * lda]));
    }
}

//-----------------------------------------------------------
__global__ void zcg_update_directions_kernel(const CUCOMPLEX *xr, CUCOMPLEX *work, CUCOMPLEX beta,
                                             int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX oldw = work[i + size_t(4) * lda];
        work[i + size_t(3) * lda] = xr[i];
        work[i + size_t(2) * lda] =
            CUFFT_ADD(work[i + size_t(1) * lda], CUFFT_MUL(beta, work[i + size_t(2) * lda]));
        work[i + size_t(4) * lda] = CUFFT_ADD(xr[i], CUFFT_MUL(beta, oldw));
    }
}

//-----------------------------------------------------------
__global__ void zcg_publish_solution_kernel(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = work[i];
}

//-----------------------------------------------------------
__global__ void zcg_publish_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = work[i + size_t(1) * lda];
}

//-----------------------------------------------------------
__global__ void pimzbicgstab_save_rhs_kernel(const CUCOMPLEX *b, CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i] = b[i];
}

//-----------------------------------------------------------
__global__ void pimzbicgstab_initialize_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                        CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX r = CUFFT_SUB(work[i], xr[i]);
        work[i] = r;
        work[i + size_t(1) * lda] = r;
        work[i + size_t(2) * lda] = CUFFT_ZERO;
        work[i + size_t(3) * lda] = CUFFT_ZERO;
        xi[i] = r;
    }
}

//-----------------------------------------------------------
__global__ void pimzbicgstab_prepare_direction_kernel(CUCOMPLEX *xi, CUCOMPLEX *work,
                                                      CUCOMPLEX beta, CUCOMPLEX omega, int lda,
                                                      int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX p = CUFFT_ADD(
            work[i], CUFFT_MUL(beta, CUFFT_SUB(work[i + size_t(2) * lda],
                                               CUFFT_MUL(omega, work[i + size_t(3) * lda]))));
        work[i + size_t(2) * lda] = p;
        xi[i] = p;
    }
}

//-----------------------------------------------------------
__global__ void pimzbicgstab_prepare_s_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr, CUCOMPLEX *work,
                                              CUCOMPLEX alpha, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        work[i + size_t(3) * lda] = xr[i];
        const CUCOMPLEX s = CUFFT_SUB(work[i], CUFFT_MUL(alpha, xr[i]));
        work[i + size_t(4) * lda] = s;
        xi[i] = s;
    }
}

//-----------------------------------------------------------
__global__ void pimzbicgstab_update_xr_kernel(CUCOMPLEX *x, CUCOMPLEX *work, CUCOMPLEX alpha,
                                              CUCOMPLEX omega, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX t = work[i + size_t(5) * lda];
        const CUCOMPLEX s = work[i + size_t(4) * lda];
        x[i] = CUFFT_ADD(
            x[i], CUFFT_ADD(CUFFT_MUL(alpha, work[i + size_t(2) * lda]), CUFFT_MUL(omega, s)));
        work[i] = CUFFT_SUB(s, CUFFT_MUL(omega, t));
    }
}

//-----------------------------------------------------------
__global__ void pimzbicgstab_soft_breakdown_kernel(CUCOMPLEX *x, const CUCOMPLEX *work, int lda,
                                                   int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = work[i + size_t(2) * lda];
}

//-----------------------------------------------------------
__global__ void pimzbicgstab_publish_vector_kernel(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda,
                                                   int column, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = work[i + size_t(column) * lda];
}

//------------------------------------------

__global__ void qmrbicgstab_initialize_kernel(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                              CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX r = CUFFT_SUB(b[i], xr[i]);
    work[i] = r;
    work[i + size_t(1) * lda] = CUFFT_ZERO;
    work[i + size_t(2) * lda] = CUFFT_ZERO;
    work[i + size_t(3) * lda] = CUFFT_ZERO;
    work[i + size_t(4) * lda] = r;
}

//-----------------------------------------------------------
__global__ void qmrbicgstab_update_p_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX beta,
                                            CUCOMPLEX omega, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CUCOMPLEX *p = work + i + size_t(1) * lda;
    const CUCOMPLEX v = work[i + size_t(2) * lda];
    *p = CUFFT_ADD(work[i], CUFFT_MUL(beta, CUFFT_SUB(*p, CUFFT_MUL(omega, v))));
    xi[i] = *p;
}

//-----------------------------------------------------------
__global__ void qmrbicgstab_prepare_s_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr, CUCOMPLEX *work,
                                             CUCOMPLEX alpha, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX v = xr[i];
    work[i + size_t(2) * lda] = v;
    const CUCOMPLEX s = CUFFT_SUB(work[i], CUFFT_MUL(alpha, v));
    work[i + size_t(5) * lda] = s;
    xi[i] = s;
}

//-----------------------------------------------------------
__global__ void qmrbicgstab_prepare_x_kernel(CUCOMPLEX *x, CUCOMPLEX *work, CUCOMPLEX alpha,
                                             CUCOMPLEX theta, CUCOMPLEX eta, CUCOMPLEX etat,
                                             int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX factor = CUFFT_DIV(CUFFT_MUL(CUFFT_MUL(theta, theta), eta), alpha);
    const CUCOMPLEX dtilde =
        CUFFT_ADD(work[i + size_t(1) * lda], CUFFT_MUL(factor, work[i + size_t(3) * lda]));
    work[i + size_t(7) * lda] = dtilde;
    work[i + size_t(6) * lda] = CUFFT_ADD(x[i], CUFFT_MUL(etat, dtilde));
}

//-----------------------------------------------------------
__global__ void qmrbicgstab_save_t_kernel(const CUCOMPLEX *xr, CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(8) * lda] = xr[i];
}

//-----------------------------------------------------------
__global__ void qmrbicgstab_update_r_kernel(CUCOMPLEX *work, CUCOMPLEX omega, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i] = CUFFT_SUB(work[i + size_t(5) * lda], CUFFT_MUL(omega, work[i + size_t(8) * lda]));
}

//-----------------------------------------------------------
__global__ void qmrbicgstab_finalize_x_kernel(CUCOMPLEX *x, CUCOMPLEX *work, CUCOMPLEX omega,
                                              CUCOMPLEX eta, CUCOMPLEX thetat, CUCOMPLEX etat,
                                              int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX factor = CUFFT_DIV(CUFFT_MUL(CUFFT_MUL(thetat, thetat), etat), omega);
    const CUCOMPLEX d =
        CUFFT_ADD(work[i + size_t(5) * lda], CUFFT_MUL(factor, work[i + size_t(7) * lda]));
    work[i + size_t(3) * lda] = d;
    x[i] = CUFFT_ADD(work[i + size_t(6) * lda], CUFFT_MUL(eta, d));
}

//-----------------------------------------------------------
__global__ void qmrbicgstab_publish_x_kernel(CUCOMPLEX *xi, const CUCOMPLEX *x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = x[i];
}

//------------------------------------------

__global__ void idrs_initialize_kernel(const CUCOMPLEX *xr, const CUCOMPLEX *b, CUCOMPLEX *work,
                                       int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    work[i] = CUFFT_SUB(b[i], xr[i]);
    for (int col = 1; col < 5; ++col)
        work[i + size_t(col) * lda] = CUFFT_ZERO;
}

//-----------------------------------------------------------
__global__ void idrs_copy_column_kernel(CUCOMPLEX *work, int dst, int src, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(dst) * lda] = work[i + size_t(src) * lda];
}

//-----------------------------------------------------------
__global__ void idrs_copy_xr_kernel(CUCOMPLEX *work, const CUCOMPLEX *xr, int dst, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(dst) * lda] = xr[i];
}

//-----------------------------------------------------------
__global__ void idrs_scale_copy_kernel(CUCOMPLEX *work, int dst, int src, CUCOMPLEX scale, int lda,
                                       int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(dst) * lda] = CUFFT_MUL(scale, work[i + size_t(src) * lda]);
}

//-----------------------------------------------------------
__global__ void idrs_axpy_kernel(CUCOMPLEX *work, int dst, int src, CUCOMPLEX scale, int lda,
                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        CUCOMPLEX *d = work + i + size_t(dst) * lda;
        *d = CUFFT_ADD(*d, CUFFT_MUL(scale, work[i + size_t(src) * lda]));
    }
}

//-----------------------------------------------------------
__global__ void idrs_normalize_basis_kernel(CUCOMPLEX *work, int col, CU_REAL norm, int lda,
                                            int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        CUCOMPLEX *p = work + i + size_t(col) * lda;
        *p = CUFFT_CONJ(CUFFT_MUL(*p, CUFFT_REAL(CU_REAL(1) / norm)));
    }
}

//-----------------------------------------------------------
__global__ void idrs_initial_pair_kernel(CUCOMPLEX *work, CUCOMPLEX *x, CUCOMPLEX omega, int is,
                                         int ns, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX r = work[i], v = work[i + size_t(1) * lda];
    const CUCOMPLEX q = CUFFT_MUL(omega, r), e = CUFFT_MUL(CUFFT_REAL(-1), CUFFT_MUL(omega, v));
    work[i + size_t(3) * lda] = q;
    work[i + size_t(4) * lda] = e;
    work[i] = CUFFT_ADD(r, e);
    x[i] = CUFFT_ADD(x[i], q);
    work[i + size_t(4 + 2 * ns - is) * lda] = q;
    work[i + size_t(4 + 3 * ns - is) * lda] = e;
}

//-----------------------------------------------------------
__global__ void idrs_apply_pair_kernel(CUCOMPLEX *work, CUCOMPLEX *x, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        work[i] = CUFFT_ADD(work[i], work[i + size_t(4) * lda]);
        x[i] = CUFFT_ADD(x[i], work[i + size_t(3) * lda]);
    }
}

//-----------------------------------------------------------
__global__ void idrs_negate_xr_kernel(CUCOMPLEX *work, const CUCOMPLEX *xr, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(4) * lda] = CUFFT_MUL(CUFFT_REAL(-1), xr[i]);
}

//-----------------------------------------------------------
__global__ void idrs_scale_xr_kernel(CUCOMPLEX *work, const CUCOMPLEX *xr, int dst, CUCOMPLEX scale,
                                     int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        work[i + size_t(dst) * lda] = CUFFT_MUL(scale, xr[i]);
}

//-----------------------------------------------------------
__global__ void idrs_shift_qe_kernel(CUCOMPLEX *work, int ns, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    for (int s = ns; s >= 2; --s) {
        work[i + size_t(4 + ns + s) * lda] = work[i + size_t(3 + ns + s) * lda];
        work[i + size_t(4 + 2 * ns + s) * lda] = work[i + size_t(3 + 2 * ns + s) * lda];
    }
    work[i + size_t(5 + ns) * lda] = work[i + size_t(3) * lda];
    work[i + size_t(5 + 2 * ns) * lda] = work[i + size_t(4) * lda];
}

//-----------------------------------------------------------
__global__ void idrs_copy_work_to_xi_kernel(CUCOMPLEX *xi, const CUCOMPLEX *work, int col, int lda,
                                            int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = work[i + size_t(col) * lda];
}

//------------------------------------------

__global__ void pimzqmr_initialize_kernel(const CUCOMPLEX *xr, const CUCOMPLEX *b, CUCOMPLEX *work,
                                          int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX r = CUFFT_SUB(b[i], xr[i]);
    work[i] = r;
    work[i + size_t(1) * lda] = CUFFT_ZERO;
    work[i + size_t(2) * lda] = xr[i];
    work[i + size_t(3) * lda] = CUFFT_ZERO;
    work[i + size_t(4) * lda] = CUFFT_ZERO;
    work[i + size_t(5) * lda] = CUFFT_ZERO;
    work[i + size_t(6) * lda] = r;
    work[i + size_t(7) * lda] = r;
}

//-----------------------------------------------------------
__global__ void pimzqmr_shadow_kernel(CUCOMPLEX *xi, CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX value = CUFFT_CONJ(work[i]);
        work[i + size_t(7) * lda] = value;
        xi[i] = value;
    }
}

//-----------------------------------------------------------
__global__ void ssbicgsafe2_initialize_kernel(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                              CUCOMPLEX *work, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX r = CUFFT_SUB(b[i], xr[i]);
    work[i] = r;
    work[i + size_t(1) * lda] = CUFFT_ZERO;
    work[i + size_t(2) * lda] = r;
    work[i + size_t(3) * lda] = CUFFT_ZERO;
    work[i + size_t(4) * lda] = CUFFT_ZERO;
    work[i + size_t(7) * lda] = CUFFT_ZERO;
    work[i + size_t(9) * lda] = CUFFT_ZERO;
}

//-----------------------------------------------------------
__global__ void gcors2_initialize_residual_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                  const CUCOMPLEX *b, CUCOMPLEX *work, int lda,
                                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const CUCOMPLEX r = CUFFT_SUB(b[i], xr[i]);
        work[i] = r;
        xi[i] = r;
    }
}

//-----------------------------------------------------------
__global__ void gcors2_setup_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xr, CUCOMPLEX *work,
                                    const CUCOMPLEX *random_values, int lda, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const CUCOMPLEX r0 = work[i];
    const CUCOMPLEX art = xr[i];
    work[i + size_t(13) * lda] = r0;
    work[i + size_t(8) * lda] = r0;
    work[i + size_t(4) * lda] = r0;
    work[i + size_t(1) * lda] = art;
    work[i + size_t(2) * lda] = art;
    work[i + size_t(9) * lda] = art;
    work[i + size_t(5) * lda] = art;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    work[i + size_t(12) * lda] =
        make_cuFloatComplex(random_values[i].x * r0.x, random_values[i].y * r0.y);
#else
    work[i + size_t(12) * lda] =
        make_cuDoubleComplex(random_values[i].x * r0.x, random_values[i].y * r0.y);
#endif
    xi[i] = art;
}

//-----------------------------------------------------------
__global__ void gcors2_publish_solution_kernel(CUCOMPLEX *xi, const CUCOMPLEX *xsol, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        xi[i] = xsol[i];
}

//-----------------------------------------------------------
unsigned int block_count(size_t count) {
    return static_cast<unsigned int>(count / THREADS_PER_BLOCK + (count % THREADS_PER_BLOCK != 0));
}

//------------------------------------------

cudaError_t report_kernel_execution_once(unsigned int slot, const char *name) {
    // Synchronize the first launch of each wrapper. A successful return proves
    // that the launched GPU kernel completed, rather than merely that the
    // CUDA runtime initialized or that the host wrapper was entered.
    static std::atomic_flag reported[256] = {};
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
        return status;

    if (!reported[slot].test_and_set()) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const char *precision = "single";
#else
        const char *precision = "double";
#endif
        std::fprintf(stderr, "[kernel.cu] GPU kernel executed: %s (%s precision)\n", name,
                     precision);
        std::fflush(stderr);
    }
    return cudaSuccess;
}

//------------------------------------------

cudaError_t finish_kernel_launch(unsigned int slot, const char *name) {
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess)
        return launch_status;
    return report_kernel_execution_once(slot, name);
}

} // namespace

//------------------------------------------

extern "C" PRODUITMATVECTFFT_KERNEL_API void ifdda_set_host_download_callback(
    ifdda_host_download_callback callback) {
    host_download_callback = callback;
}

//-----------------------------------------------------------
extern "C" cudaError_t zero_vectors_wrapper(CUCOMPLEX *vectors, int count) {
    if (!vectors || count <= 0)
        return cudaErrorInvalidValue;
    zero_vectors_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(vectors, count);
    return finish_kernel_launch(0, "zero_vectors_kernel");
}

//-----------------------------------------------------------
__global__ void dipole_inside_find_kernel(const CU_REAL *xs, const CU_REAL *ys,
                                          const CU_REAL *zs, int count,
                                          CU_REAL xdip, CU_REAL ydip, CU_REAL zdip,
                                          CU_REAL cell_size, CU_INT*first_match) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const CU_REAL scale = CU_REAL(0.9999) / cell_size;
    const int ix = int(round((xdip - xs[index]) * scale));
    const int iy = int(round((ydip - ys[index]) * scale));
    const int iz = int(round((zdip - zs[index]) * scale));
    if (ix == 0 && iy == 0 && iz == 0)
        atomicMin(&first_match->dat, index);
}

//-----------------------------------------------------------
__global__ void microscope_mask_extrema_kernel(const CU_REAL *xs, const CU_REAL *ys, int count,
                                                CU_REAL *extrema) {
    __shared__ CU_REAL partial[4][THREADS_PER_BLOCK];
    const int thread = int(threadIdx.x);
    const CU_REAL high = sizeof(CU_REAL) == sizeof(float) ? CU_REAL(FLT_MAX) : CU_REAL(DBL_MAX);
    CU_REAL xmin = high;
    CU_REAL xmax = -high;
    CU_REAL ymin = high;
    CU_REAL ymax = -high;
    for (int i = thread; i < count; i += int(blockDim.x)) {
        xmin = min(xmin, xs[i]);
        xmax = max(xmax, xs[i]);
        ymin = min(ymin, ys[i]);
        ymax = max(ymax, ys[i]);
    }
    partial[0][thread] = xmin;
    partial[1][thread] = xmax;
    partial[2][thread] = ymin;
    partial[3][thread] = ymax;
    __syncthreads();
    for (int stride = int(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            partial[0][thread] = min(partial[0][thread], partial[0][thread + stride]);
            partial[1][thread] = max(partial[1][thread], partial[1][thread + stride]);
            partial[2][thread] = min(partial[2][thread], partial[2][thread + stride]);
            partial[3][thread] = max(partial[3][thread], partial[3][thread + stride]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        extrema[0] = partial[0][0];
        extrema[1] = partial[1][0];
        extrema[2] = partial[2][0];
        extrema[3] = partial[3][0];
    }
}

//-----------------------------------------------------------
__global__ void microscope_mask_fill_kernel(CUCOMPLEX *mask, CU_INT*selected_count, int imaxinc,
                                             int sizemask, int mask_type, CU_REAL deltak,
                                             CU_REAL numaperk, CU_REAL numaperk2, CU_REAL kcnax,
                                             CU_REAL kcnay) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = sizemask * sizemask;
    if (index >= count)
        return;
    const int idelta = index % sizemask - imaxinc;
    const int jdelta = index / sizemask - imaxinc;
    const CU_REAL kxinc = CU_REAL(idelta) * deltak;
    const CU_REAL kyinc = CU_REAL(jdelta) * deltak;
    const CU_REAL radius2 = kxinc * kxinc + kyinc * kyinc;
    bool selected = false;
    if (mask_type == 1 || mask_type == 6 || mask_type == 7) {
        selected = radius2 <= numaperk * numaperk;
    } else if (mask_type == 2 || mask_type == 5) {
        selected = radius2 <= numaperk * numaperk && radius2 >= numaperk2 * numaperk2;
    } else if (mask_type == 3) {
        const CU_REAL dx = kxinc - kcnax;
        const CU_REAL dy = kyinc - kcnay;
        selected = dx * dx + dy * dy <= numaperk * numaperk;
    }
    mask[index] = selected ? CUFFT_REAL(CU_REAL(1)) : CUFFT_REAL(CU_REAL(1.0e30));
    if (selected)
        atomicAdd(&selected_count->dat, 1);
}

//-----------------------------------------------------------
__global__ void microscope_copy_scale_kernel(const CUCOMPLEX *source,
                                               CUCOMPLEX *destination,
                                               size_t count, CU_REAL scale) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    destination[index].x = source[index].x * scale;
    destination[index].y = source[index].y * scale;
}

//-----------------------------------------------------------
__global__ void microscope_axes_kernel(CU_REAL *kxy, CU_REAL *xy, int count,
                                        int half, CU_REAL delta_k,
                                        CU_REAL gross, CU_REAL k0) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int coordinate = index - half;
    const CU_REAL pi = CU_REAL(3.141592653589793238462643383279502884);
    const CU_REAL delta_x = CU_REAL(2) * pi / CU_REAL(count) / delta_k;
    xy[index] = delta_x * CU_REAL(coordinate) * gross;
    kxy[index] = delta_k * CU_REAL(coordinate) / k0;
}

//-----------------------------------------------------------
__global__ void microscope_identity_dipole_map_kernel(CU_INT*dipole_map,
                                                       int count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
        dipole_map[index].dat = index + 1;
}

//-----------------------------------------------------------
__global__ void microscope_project_scalar_rhs_kernel(
    const CUCOMPLEX *incident, CUCOMPLEX *rhs, int count,
    CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int base = 3 * index;
    CUCOMPLEX value = CUFFT_MUL(incident[base], CUFFT_CONJ(ux));
    value = CUFFT_ADD(value,
        CUFFT_MUL(incident[base + 1], CUFFT_CONJ(uy)));
    value = CUFFT_ADD(value,
        CUFFT_MUL(incident[base + 2], CUFFT_CONJ(uz)));
    rhs[index] = value;
}

//-----------------------------------------------------------
__global__ void microscope_expand_scalar_fields_kernel(
    CUCOMPLEX *xi, const CUCOMPLEX *polarization,
    const CUCOMPLEX *scalar_local, CUCOMPLEX *scattered,
    CUCOMPLEX *local, int count, CUCOMPLEX ux,
    CUCOMPLEX uy, CUCOMPLEX uz) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const CUCOMPLEX local_value = scalar_local[index];
    const CUCOMPLEX dipole_value = CUFFT_MUL(polarization[index], local_value);
    const int base = 3 * index;
    xi[index] = dipole_value;
    scattered[base] = CUFFT_MUL(dipole_value, ux);
    scattered[base + 1] = CUFFT_MUL(dipole_value, uy);
    scattered[base + 2] = CUFFT_MUL(dipole_value, uz);
    local[base] = CUFFT_MUL(local_value, ux);
    local[base + 1] = CUFFT_MUL(local_value, uy);
    local[base + 2] = CUFFT_MUL(local_value, uz);
}

//-----------------------------------------------------------
extern "C" cudaError_t dipole_inside_find_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, int count,
    CU_REAL xdip, CU_REAL ydip, CU_REAL zdip, CU_REAL cell_size, CU_INT*first_match) {
    if (!xs || !ys || !zs || !first_match || count <= 0 || cell_size == CU_REAL(0))
        return cudaErrorInvalidValue;
    dipole_inside_find_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, count, xdip, ydip, zdip, cell_size, first_match);
    return finish_kernel_launch(220, "dipole_inside_find_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_mask_extrema_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, int count, CU_REAL *extrema) {
    if (!xs || !ys || !extrema || count <= 0)
        return cudaErrorInvalidValue;
    microscope_mask_extrema_kernel<<<1, THREADS_PER_BLOCK>>>(xs, ys, count, extrema);
    return finish_kernel_launch(224, "microscope_mask_extrema_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_mask_fill_wrapper(
    CUCOMPLEX *mask, CU_INT*selected_count, int imaxinc, int sizemask, int mask_type,
    CU_REAL deltak, CU_REAL numaperk, CU_REAL numaperk2, CU_REAL kcnax, CU_REAL kcnay) {
    if (!mask || !selected_count || imaxinc < 0 || sizemask != 2 * imaxinc + 1 ||
        mask_type < 1 || mask_type > 7 || deltak <= CU_REAL(0))
        return cudaErrorInvalidValue;
    if (cudaMemset(selected_count, 0, sizeof(int)) != cudaSuccess)
        return cudaGetLastError();
    microscope_mask_fill_kernel<<<block_count(size_t(sizemask) * size_t(sizemask)),
                                  THREADS_PER_BLOCK>>>(
        mask, selected_count, imaxinc, sizemask, mask_type, deltak, numaperk, numaperk2, kcnax,
        kcnay);
    return finish_kernel_launch(225, "microscope_mask_fill_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_copy_scale_wrapper(
    const CUCOMPLEX *source, CUCOMPLEX *destination, size_t count,
    CU_REAL scale) {
    if (!source || !destination || count == 0)
        return cudaErrorInvalidValue;
    microscope_copy_scale_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        source, destination, count, scale);
    return finish_kernel_launch(226, "microscope_copy_scale_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t microscope_axes_wrapper(
    CU_REAL *kxy, CU_REAL *xy, int count, int half, CU_REAL delta_k,
    CU_REAL gross, CU_REAL k0) {
    if (!kxy || !xy || count <= 0 || half < 0 || half >= count ||
        delta_k == CU_REAL(0) || k0 == CU_REAL(0))
        return cudaErrorInvalidValue;
    microscope_axes_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        kxy, xy, count, half, delta_k, gross, k0);
    return finish_kernel_launch(227, "microscope_axes_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
microscope_identity_dipole_map_wrapper(CU_INT*dipole_map, int count) {
    if (!dipole_map || count <= 0)
        return cudaErrorInvalidValue;
    microscope_identity_dipole_map_kernel<<<block_count(size_t(count)),
                                             THREADS_PER_BLOCK>>>(dipole_map, count);
    return finish_kernel_launch(228, "microscope_identity_dipole_map_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
microscope_project_scalar_rhs_wrapper(const CUCOMPLEX *incident,
    CUCOMPLEX *rhs, int count, CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz) {
    if (!incident || !rhs || count <= 0)
        return cudaErrorInvalidValue;
    microscope_project_scalar_rhs_kernel<<<block_count(size_t(count)),
                                            THREADS_PER_BLOCK>>>(
        incident, rhs, count, ux, uy, uz);
    return finish_kernel_launch(229, "microscope_project_scalar_rhs_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
microscope_expand_scalar_fields_wrapper(CUCOMPLEX *xi,
    const CUCOMPLEX *polarization, const CUCOMPLEX *scalar_local,
    CUCOMPLEX *scattered, CUCOMPLEX *local, int count,
    CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz) {
    if (!xi || !polarization || !scalar_local || !scattered || !local ||
        count <= 0)
        return cudaErrorInvalidValue;
    microscope_expand_scalar_fields_kernel<<<block_count(size_t(count)),
                                               THREADS_PER_BLOCK>>>(
        xi, polarization, scalar_local, scattered, local, count, ux, uy, uz);
    return finish_kernel_launch(230, "microscope_expand_scalar_fields_kernel");
}

//-----------------------------------------------------------
__device__ CUCOMPLEX abcchi_make_complex(CU_REAL real_part, CU_REAL imag_part) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real_part, imag_part);
#else
    return make_cuDoubleComplex(real_part, imag_part);
#endif
}

//-----------------------------------------------------------
__device__ CUCOMPLEX abcchi_complex_sqrt(CUCOMPLEX value) {
    const CU_REAL magnitude = hypot(value.x, value.y);
    const CU_REAL real_part = sqrt(max(CU_REAL(0), (magnitude + value.x) / CU_REAL(2)));
    const CU_REAL imag_magnitude = sqrt(max(CU_REAL(0), (magnitude - value.x) / CU_REAL(2)));
    const CU_REAL imag_part = value.y < CU_REAL(0) ? -imag_magnitude : imag_magnitude;
    return abcchi_make_complex(real_part, imag_part);
}

//-----------------------------------------------------------
__global__ void abcchi_profile_kernel(CUCOMPLEX* epsilon_output,
                                       CUCOMPLEX* contrast_output,
                                       CUCOMPLEX epsilon_value,
                                       CUCOMPLEX epsilon_background,
                                       CU_REAL contrast_scale,
                                       int profile_type, int i, int j, int k,
                                       int nx, int ny, int nz, int neps,
                                       int isotropy) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    CUCOMPLEX selected_epsilon = epsilon_value;
    const bool inside_profile = profile_type >= 1 && neps > 0 &&
        (i <= neps || i > nx - neps || j <= neps || j > ny - neps ||
         k <= neps || k > nz - neps);
    if (inside_profile && profile_type <= 6) {
        const int ii = i <= neps ? i : (i > nx - neps ? nx - i + 1 : neps + 1);
        const int jj = j <= neps ? j : (j > ny - neps ? ny - j + 1 : neps + 1);
        const int kk = k <= neps ? k : (k > nz - neps ? nz - k + 1 : neps + 1);
        const CU_REAL alpha = profile_type <= 2 ? CU_REAL(1) :
                              (profile_type <= 4 ? CU_REAL(2) : CU_REAL(3));
        const CU_REAL argument = alpha *
            (CU_REAL(2) * CU_REAL(ii) * CU_REAL(jj) * CU_REAL(kk) /
             (CU_REAL(neps + 1) * CU_REAL(neps + 1) * CU_REAL(neps + 1)) - CU_REAL(1));
        const CUCOMPLEX index_v = CUFFT_REAL(CU_REAL(1));
        const CUCOMPLEX index_0 = abcchi_complex_sqrt(epsilon_background);
        const CU_REAL transition = (profile_type & 1) ?
            erff(argument) / erff(alpha) : tanhf(argument) / tanhf(alpha);
        const CUCOMPLEX mean_index = abcchi_make_complex(CU_REAL(0.5), CU_REAL(0));
        const CUCOMPLEX delta_index = abcchi_make_complex(CU_REAL(0.5) * transition, CU_REAL(0));
        const CUCOMPLEX half_sum = CUFFT_MUL(mean_index, CUFFT_ADD(index_0, index_v));
        const CUCOMPLEX half_delta = CUFFT_MUL(delta_index, CUFFT_SUB(index_v, index_0));
        const CUCOMPLEX index_abc = CUFFT_ADD(half_sum, half_delta);
        selected_epsilon = CUFFT_MUL(index_abc, index_abc);
    }
    for (int index = 0; index < 9; ++index)
        epsilon_output[index] = CUFFT_ZERO;
    if (isotropy == 3) {
        epsilon_output[0] = selected_epsilon;
        epsilon_output[4] = selected_epsilon;
        epsilon_output[8] = selected_epsilon;
    } else {
        epsilon_output[0] = selected_epsilon;
    }
    *contrast_output = CUFFT_MUL(CUFFT_SUB(selected_epsilon, epsilon_background),
                                 CUFFT_REAL(contrast_scale));
}

//-----------------------------------------------------------
__global__ void pml_sphere_material_profile_kernel(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
    const CU_REAL *material, CUCOMPLEX *epsilon_output,
    CUCOMPLEX *contrast_output, int count, int nx, int ny, int nz,
    int material_extent, CU_REAL radius, CU_REAL xmin, CU_REAL spacing,
    CU_REAL epsilon_imag, CUCOMPLEX epsilon_value,
    CUCOMPLEX epsilon_background,
    CU_REAL contrast_scale, int profile_type, int profile_extent) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;

    CUCOMPLEX selected_epsilon;
    const CU_REAL x = xs[index];
    const CU_REAL y = ys[index];
    const CU_REAL z = zs[index];
    if (sqrt(x * x + y * y + z * z) < radius) {
        const int nint_x = int(floor((x - xmin) / spacing + CU_REAL(0.5)));
        const int nint_y = int(floor((y - xmin) / spacing + CU_REAL(0.5)));
        const int nint_z = int(floor((z - xmin) / spacing + CU_REAL(0.5)));
        const int material_count = material_extent * material_extent * material_extent;
        const int material_index = nint_x + nint_y * material_extent +
            nint_z * material_extent * material_extent;
        // Fortran uses epsb(nnn), backed by an nmax-sized array. In the
        // reference fixture its tail beyond the generated nnx^3 random
        // field is zero. Clamping to the last random cell changes 32 sites
        // on the 32^3 PML test grid, so reproduce that observed convention.
        selected_epsilon = (material_index >= 0 && material_index < material_count)
            ? abcchi_make_complex(material[material_index], epsilon_imag)
            : abcchi_make_complex(CU_REAL(0), CU_REAL(0));
    } else {
        const int k = index % nx + 1;
        const int j = (index / nx) % ny + 1;
        const int i = index / (nx * ny) + 1;
        const bool inside_profile = profile_type >= 1 && profile_extent > 0 &&
            (i <= profile_extent || i > nz - profile_extent ||
             j <= profile_extent || j > ny - profile_extent ||
             k <= profile_extent || k > nx - profile_extent);
        selected_epsilon = epsilon_value;
        if (inside_profile && profile_type <= 6) {
            const int ii = i <= profile_extent ? i :
                (i > nz - profile_extent ? nz - i + 1 : profile_extent + 1);
            const int jj = j <= profile_extent ? j :
                (j > ny - profile_extent ? ny - j + 1 : profile_extent + 1);
            const int kk = k <= profile_extent ? k :
                (k > nx - profile_extent ? nx - k + 1 : profile_extent + 1);
            const CU_REAL alpha = profile_type <= 2 ? CU_REAL(1) :
                (profile_type <= 4 ? CU_REAL(2) : CU_REAL(3));
            const CU_REAL denominator = CU_REAL(profile_extent + 1);
            const CU_REAL argument = alpha *
                (CU_REAL(2) * CU_REAL(ii) * CU_REAL(jj) * CU_REAL(kk) /
                 (denominator * denominator * denominator) - CU_REAL(1));
            const CUCOMPLEX index_0 = abcchi_complex_sqrt(epsilon_background);
            // Match Fortran's real(wp) ERF/TANH: using the float intrinsics
            // here would silently truncate the double-precision PML profile.
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            const CU_REAL transition = (profile_type & 1) ?
                erff(argument) / erff(alpha) : tanhf(argument) / tanhf(alpha);
#else
            const CU_REAL transition = (profile_type & 1) ?
                erf(argument) / erf(alpha) : tanh(argument) / tanh(alpha);
#endif
            const CUCOMPLEX mean_index = abcchi_make_complex(CU_REAL(0.5), CU_REAL(0));
            const CUCOMPLEX delta_index = abcchi_make_complex(
                CU_REAL(0.5) * transition, CU_REAL(0));
            const CUCOMPLEX half_sum = CUFFT_MUL(
                mean_index, CUFFT_ADD(index_0, CUFFT_REAL(CU_REAL(1))));
            const CUCOMPLEX half_delta = CUFFT_MUL(
                delta_index, CUFFT_SUB(CUFFT_REAL(CU_REAL(1)), index_0));
            const CUCOMPLEX index_abc = CUFFT_ADD(half_sum, half_delta);
            selected_epsilon = CUFFT_MUL(index_abc, index_abc);
        }
    }
    epsilon_output[index] = selected_epsilon;
    contrast_output[index] = CUFFT_MUL(
        CUFFT_SUB(selected_epsilon, epsilon_background),
        CUFFT_REAL(contrast_scale));
}

//-----------------------------------------------------------
__global__ void pml_cuboid_material_profile_kernel(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    const CU_REAL* material, CUCOMPLEX* epsilon_output,
    CUCOMPLEX* contrast_output, int count, int nx, int ny, int nz,
    int material_nx, int material_ny, int material_nz,
    CU_REAL side_x, CU_REAL side_y, CU_REAL side_z,
    CU_REAL xmin, CU_REAL ymin, CU_REAL zmin, CU_REAL spacing,
    CU_REAL epsilon_imag, CUCOMPLEX epsilon_value,
    CUCOMPLEX epsilon_background, CU_REAL contrast_scale,
    int profile_type, int profile_extent) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const CU_REAL x = xs[index];
    const CU_REAL y = ys[index];
    const CU_REAL z = zs[index];
    CUCOMPLEX selected_epsilon = epsilon_value;
    const bool inside_object = fabs(x) < side_x / CU_REAL(2) &&
        fabs(y) < side_y / CU_REAL(2) && fabs(z) < side_z / CU_REAL(2);
    if (inside_object) {
        const int ii = int(floor((x - xmin) / spacing + CU_REAL(0.5)));
        const int jj = int(floor((y - ymin) / spacing + CU_REAL(0.5)));
        const int kk = int(floor((z - zmin) / spacing + CU_REAL(0.5)));
        const int material_count = material_nx * material_ny * material_nz;
        const int material_index = max(0, min(material_count - 1,
            ii + jj * material_nx + kk * material_nx * material_ny));
        selected_epsilon = abcchi_make_complex(
            material[material_index], epsilon_imag);
    } else {
        const int k = index % nx + 1;
        const int j = (index / nx) % ny + 1;
        const int i = index / (nx * ny) + 1;
        const bool inside_profile = profile_type >= 1 && profile_extent > 0 &&
            (i <= profile_extent || i > nz - profile_extent ||
             j <= profile_extent || j > ny - profile_extent ||
             k <= profile_extent || k > nx - profile_extent);
        if (inside_profile && profile_type <= 6) {
            const int ii = i <= profile_extent ? i :
                (i > nz - profile_extent ? nz - i + 1 : profile_extent + 1);
            const int jj = j <= profile_extent ? j :
                (j > ny - profile_extent ? ny - j + 1 : profile_extent + 1);
            const int kk = k <= profile_extent ? k :
                (k > nx - profile_extent ? nx - k + 1 : profile_extent + 1);
            const CU_REAL alpha = profile_type <= 2 ? CU_REAL(1) :
                (profile_type <= 4 ? CU_REAL(2) : CU_REAL(3));
            const CU_REAL denominator = CU_REAL(profile_extent + 1);
            const CU_REAL argument = alpha *
                (CU_REAL(2) * CU_REAL(ii) * CU_REAL(jj) * CU_REAL(kk) /
                 (denominator * denominator * denominator) - CU_REAL(1));
            const CUCOMPLEX index_0 = abcchi_complex_sqrt(epsilon_background);
            // Use the active solver precision for the absorbing-layer profile.
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            const CU_REAL transition = (profile_type & 1) ?
                erff(argument) / erff(alpha) : tanhf(argument) / tanhf(alpha);
#else
            const CU_REAL transition = (profile_type & 1) ?
                erf(argument) / erf(alpha) : tanh(argument) / tanh(alpha);
#endif
            const CUCOMPLEX mean_index = abcchi_make_complex(CU_REAL(0.5), CU_REAL(0));
            const CUCOMPLEX delta_index = abcchi_make_complex(
                CU_REAL(0.5) * transition, CU_REAL(0));
            const CUCOMPLEX half_sum = CUFFT_MUL(
                mean_index, CUFFT_ADD(index_0, CUFFT_REAL(CU_REAL(1))));
            const CUCOMPLEX half_delta = CUFFT_MUL(
                delta_index, CUFFT_SUB(CUFFT_REAL(CU_REAL(1)), index_0));
            const CUCOMPLEX index_abc = CUFFT_ADD(half_sum, half_delta);
            selected_epsilon = CUFFT_MUL(index_abc, index_abc);
        }
    }
    epsilon_output[index] = selected_epsilon;
    contrast_output[index] = CUFFT_MUL(
        CUFFT_SUB(selected_epsilon, epsilon_background),
        CUFFT_REAL(contrast_scale));
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t abcchi_profile_wrapper(
    CUCOMPLEX* epsilon_output,
    CUCOMPLEX* contrast_output, CUCOMPLEX epsilon_value, CUCOMPLEX epsilon_background,
    CU_REAL contrast_scale, int profile_type, int i, int j, int k,
    int nx, int ny, int nz, int neps, int isotropy) {
    if (!epsilon_output || !contrast_output || nx <= 0 || ny <= 0 ||
        nz <= 0 || i <= 0 || j <= 0 || k <= 0 || i > nx || j > ny || k > nz ||
        isotropy != 1 && isotropy != 3)
        return cudaErrorInvalidValue;
    abcchi_profile_kernel<<<1, 1>>>(epsilon_output, contrast_output,
                                    epsilon_value, epsilon_background, contrast_scale,
                                    profile_type, i, j, k, nx, ny, nz, neps, isotropy);
    return finish_kernel_launch(226, "abcchi_profile_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
pml_sphere_material_profile_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
    const CU_REAL *material, CUCOMPLEX *epsilon_output,
    CUCOMPLEX *contrast_output, int count, int nx, int ny, int nz,
    int material_extent, CU_REAL radius, CU_REAL xmin, CU_REAL spacing,
    CU_REAL epsilon_imag, CUCOMPLEX epsilon_value,
    CUCOMPLEX epsilon_background,
    CU_REAL contrast_scale, int profile_type, int profile_extent) {
    if (!xs || !ys || !zs || !material || !epsilon_output ||
        !contrast_output || count <= 0 || nx <= 0 || ny <= 0 || nz <= 0 ||
        nx > INT_MAX / ny || nx * ny > INT_MAX / nz ||
        count != nx * ny * nz || material_extent <= 0 ||
        material_extent > INT_MAX / material_extent ||
        material_extent * material_extent > INT_MAX / material_extent ||
        radius <= CU_REAL(0) || spacing <= CU_REAL(0))
        return cudaErrorInvalidValue;
    pml_sphere_material_profile_kernel<<<block_count(size_t(count)),
                                          THREADS_PER_BLOCK>>>(
        xs, ys, zs, material, epsilon_output, contrast_output, count, nx, ny,
        nz, material_extent, radius, xmin, spacing, epsilon_imag,
        epsilon_value, epsilon_background, contrast_scale, profile_type,
        profile_extent);
    return finish_kernel_launch(227, "pml_sphere_material_profile_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
pml_cuboid_material_profile_wrapper(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    const CU_REAL* material, CUCOMPLEX* epsilon_output,
    CUCOMPLEX* contrast_output, int nx, int ny, int nz,
    int material_nx, int material_ny, int material_nz,
    CU_REAL side_x, CU_REAL side_y, CU_REAL side_z,
    CU_REAL xmin, CU_REAL ymin, CU_REAL zmin, CU_REAL spacing,
    CU_REAL epsilon_imag, CUCOMPLEX epsilon_value,
    CUCOMPLEX epsilon_background, CU_REAL contrast_scale,
    int profile_type, int profile_extent) {
    if (!xs || !ys || !zs || !material || !epsilon_output ||
        !contrast_output || nx <= 0 || ny <= 0 || nz <= 0 ||
        nx > INT_MAX / ny || nx * ny > INT_MAX / nz ||
        material_nx <= 0 || material_ny <= 0 || material_nz <= 0 ||
        material_nx > INT_MAX / material_ny ||
        material_nx * material_ny > INT_MAX / material_nz ||
        side_x <= CU_REAL(0) || side_y <= CU_REAL(0) ||
        side_z <= CU_REAL(0) || spacing <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    pml_cuboid_material_profile_kernel<<<block_count(size_t(count)),
                                           THREADS_PER_BLOCK>>>(
        xs, ys, zs, material, epsilon_output, contrast_output, count,
        nx, ny, nz, material_nx, material_ny, material_nz,
        side_x, side_y, side_z, xmin, ymin, zmin, spacing, epsilon_imag,
        epsilon_value, epsilon_background, contrast_scale, profile_type,
        profile_extent);
    return finish_kernel_launch(228, "pml_cuboid_material_profile_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cube_geometry_wrapper(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, CU_INT *dipole_map,
                                             int subdivisions, CU_REAL side, CU_REAL cell_size,
                                             CU_REAL center_x, CU_REAL center_y, CU_REAL center_z) {
    if (!xs || !ys || !zs || !dipole_map || subdivisions <= 0 ||
        subdivisions > INT_MAX / subdivisions ||
        subdivisions * subdivisions > INT_MAX / subdivisions)
        return cudaErrorInvalidValue;
    const int count = subdivisions * subdivisions * subdivisions;
    cube_geometry_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, dipole_map, subdivisions, side, cell_size, center_x, center_y, center_z, count);
    return finish_kernel_launch(120, "cube_geometry_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cube_dipole_map_wrapper(CU_INT* dipole_map, int subdivisions) {
    if (!dipole_map || subdivisions <= 0 || subdivisions > INT_MAX / subdivisions ||
        subdivisions * subdivisions > INT_MAX / subdivisions)
        return cudaErrorInvalidValue;
    const int count = subdivisions * subdivisions * subdivisions;
    cube_dipole_map_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(dipole_map, count);
    return finish_kernel_launch(264, "cube_dipole_map_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t full_grid_dipole_map_wrapper(CU_INT* dipole_map,
                                                     int grid_count) {
    if (!dipole_map || grid_count <= 0)
        return cudaErrorInvalidValue;
    cube_dipole_map_kernel<<<block_count(size_t(grid_count)), THREADS_PER_BLOCK>>>(
        dipole_map, grid_count);
    return finish_kernel_launch(265, "cube_dipole_map_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t widefield_build_grid_wrapper(
    CU_REAL* grid_x, CU_REAL* grid_y, CU_REAL* grid_z,
    int nx, int ny, int nz, int nxmp, int nymp, int nzmp,
    CU_REAL xmin, CU_REAL ymin, CU_REAL zmin, CU_REAL cell_size) {
    if (!grid_x || !grid_y || !grid_z || nx <= 0 || ny <= 0 || nz <= 0 ||
        nxmp < 0 || nymp < 0 || nzmp < 0 || cell_size <= CU_REAL(0) ||
        nx > INT_MAX / ny || nx * ny > INT_MAX / nz)
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    widefield_build_grid_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        grid_x, grid_y, grid_z, nx, ny, nz, nxmp, nymp, nzmp,
        xmin, ymin, zmin, cell_size, count);
    return finish_kernel_launch(250, "widefield_build_grid_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t widefield_map_dipoles_wrapper(
    const CU_REAL* dipole_x, const CU_REAL* dipole_y, const CU_REAL* dipole_z,
    const CU_REAL* grid_x, const CU_REAL* grid_y, const CU_REAL* grid_z,
    CU_INT* dipole_map, int dipole_count, int nx, int ny, int nz,
    CU_REAL cell_size, CU_REAL tolerance) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny ||
        nx * ny > INT_MAX / nz)
        return cudaErrorInvalidValue;
    const int grid_count = nx * ny * nz;
    if (!dipole_x || !dipole_y || !dipole_z || !grid_x || !grid_y || !grid_z ||
        !dipole_map || dipole_count <= 0 || grid_count <= 0 ||
        cell_size <= CU_REAL(0) || tolerance < CU_REAL(0))
        return cudaErrorInvalidValue;
    cudaError_t status = cudaMemset(dipole_map, 0xff,
                                    size_t(grid_count) * sizeof(int));
    if (status != cudaSuccess)
        return status;
    widefield_map_dipoles_kernel<<<block_count(size_t(dipole_count)), THREADS_PER_BLOCK>>>(
        dipole_x, dipole_y, dipole_z, grid_x, grid_y, grid_z, dipole_map,
        dipole_count, nx, ny, nz, cell_size, tolerance);
    return finish_kernel_launch(251, "widefield_map_dipoles_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t widefield_scatter_dipoles_wrapper(
    const CU_INT* grid_to_dipole, int grid_count, int dipole_count, int isotropy,
    int preserve_incident_field, const CUCOMPLEX* dipole_scattered,
    const CUCOMPLEX* dipole_incident, const CUCOMPLEX* dipole_epsilon,
    CUCOMPLEX* xi, CUCOMPLEX* xr, CUCOMPLEX* polarization) {
    if (!grid_to_dipole || !dipole_scattered ||
        (!preserve_incident_field && !dipole_incident) ||
        !dipole_epsilon || !xi || !xr || !polarization || grid_count <= 0 ||
        dipole_count <= 0 || (isotropy != 1 && isotropy != 3) ||
        (preserve_incident_field != 0 && preserve_incident_field != 1))
        return cudaErrorInvalidValue;
    widefield_scatter_dipoles_kernel<<<block_count(size_t(grid_count)), THREADS_PER_BLOCK>>>(
        grid_to_dipole, grid_count, dipole_count, isotropy,
        preserve_incident_field, dipole_scattered, dipole_incident,
        dipole_epsilon, xi, xr, polarization);
    return finish_kernel_launch(252, "widefield_scatter_dipoles_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t widefield_pack_dipole_epsilon_wrapper(
    const CUCOMPLEX* fortran_epsilon, CUCOMPLEX* packed_epsilon,
    int dipole_count, int nmax, int isotropy) {
    if (!fortran_epsilon || !packed_epsilon || dipole_count <= 0 ||
        nmax < dipole_count || (isotropy != 1 && isotropy != 3) ||
        dipole_count > INT_MAX / (isotropy * isotropy))
        return cudaErrorInvalidValue;
    const int components = isotropy * isotropy;
    const int count = dipole_count * components;
    widefield_pack_dipole_epsilon_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        fortran_epsilon, packed_epsilon, dipole_count, nmax, count);
    return finish_kernel_launch(259, "widefield_pack_dipole_epsilon_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t widefield_restore_material_local_field_wrapper(
    const CU_INT* grid_to_dipole, int grid_count, int dipole_count, int isotropy,
    const CUCOMPLEX* dipole_epsilon, const CUCOMPLEX* dipole_local_field,
    CUCOMPLEX* grid_field) {
    if (!grid_to_dipole || !dipole_epsilon || !dipole_local_field || !grid_field ||
        grid_count <= 0 || dipole_count <= 0 ||
        (isotropy != 1 && isotropy != 3))
        return cudaErrorInvalidValue;
    widefield_restore_material_local_field_kernel<<<block_count(size_t(grid_count)),
        THREADS_PER_BLOCK>>>(grid_to_dipole, grid_count, dipole_count, isotropy,
        dipole_epsilon, dipole_local_field, grid_field);
    return finish_kernel_launch(253, "widefield_restore_material_local_field_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t widefield_incident_magnitude_wrapper(
    const CUCOMPLEX* grid_field, CU_REAL* magnitude, int grid_count) {
    if (!grid_field || !magnitude || grid_count <= 0)
        return cudaErrorInvalidValue;
    widefield_incident_magnitude_kernel<<<block_count(size_t(grid_count)),
        THREADS_PER_BLOCK>>>(grid_field, magnitude, grid_count);
    return finish_kernel_launch(254, "widefield_incident_magnitude_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t initialize_polarizability_arrays_wrapper(
    CU_INT*dipole_map, CUCOMPLEX *polarizability, CUCOMPLEX *epsilon,
    int dipole_count, int isotropy) {
    if (!polarizability || !epsilon || dipole_count <= 0 ||
        (isotropy != 1 && isotropy != 3))
        return cudaErrorInvalidValue;
    initialize_polarizability_arrays_kernel<<<block_count(size_t(dipole_count)), THREADS_PER_BLOCK>>>(
        dipole_map, polarizability, epsilon, dipole_count, isotropy);
    return finish_kernel_launch(134, "initialize_polarizability_arrays_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t sphere_material_from_cpu_tabdip_wrapper(
    const CU_INT* tabdip, const CU_INT* inside_map,
    CUCOMPLEX* polarizability, CUCOMPLEX* epsilon,
    const CUCOMPLEX* tensor_alpha, const CUCOMPLEX* tensor_epsilon,
    CUCOMPLEX scalar_alpha, CUCOMPLEX scalar_epsilon,
    int grid_count, int nmax, int compact_mesh, int anisotropic) {
    if (!tabdip || !inside_map || !polarizability || !epsilon ||
        grid_count <= 0 || nmax <= 0 ||
        (!compact_mesh && nmax < grid_count) ||
        (compact_mesh != 0 && compact_mesh != 1) ||
        (anisotropic != 0 && anisotropic != 1) ||
        (anisotropic && (!tensor_alpha || !tensor_epsilon)))
        return cudaErrorInvalidValue;

    sphere_material_from_cpu_tabdip_kernel<<<block_count(size_t(grid_count)), THREADS_PER_BLOCK>>>(
        tabdip, inside_map, polarizability, epsilon,
        tensor_alpha, tensor_epsilon, scalar_alpha, scalar_epsilon,
        grid_count, nmax, compact_mesh, anisotropic);
    return finish_kernel_launch(263, "sphere_material_from_cpu_tabdip_kernel");
}

//-----------------------------------------------------------
// Arbitrary objects store one distinct 3x3 permittivity tensor per cell.
// Process all cells in one launch instead of launching a single-cell tensor
// solver (and downloading its result) for each lattice position.
__global__ void arbitrary_tensor_material_batch_kernel(
    const CUCOMPLEX* cell_epsilon, CUCOMPLEX* output_epsilon,
    CUCOMPLEX* output_alpha, int count, int nmax, CUCOMPLEX epsilon0,
    CU_REAL radius_input, CU_REAL k0, int method) {
    const int cell = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= count)
        return;

    CUCOMPLEX epsilon[9];
    for (int i = 0; i < 9; ++i) {
        epsilon[i] = cell_epsilon[size_t(cell) * 9 + size_t(i)];
        output_epsilon[size_t(cell) + size_t(nmax) * size_t(i)] = epsilon[i];
    }
    CUCOMPLEX result[9];
    for (int i = 0; i < 9; ++i)
        result[i] = CUFFT_ZERO;

    // polaepstens_ only supports CM and RR; an unknown method leaves alpha=0.
    if (method == 1 || method == 2) {
        const CU_REAL pi_value = CU_REAL(3.1415926535897932384626433832795L);
        CU_REAL radius = radius_input;
        radius *= pow(CU_REAL(0.75) / pi_value, CU_REAL(1) / CU_REAL(3));
        const CU_REAL volume = radius * radius * radius;
        CUCOMPLEX denominator[9];
        CUCOMPLEX numerator[9];
        CUCOMPLEX inverse[9];
        CUCOMPLEX alpha0[9];
        CUCOMPLEX temporary[9];
        for (int column = 0; column < 3; ++column) {
            for (int row = 0; row < 3; ++row) {
                const int i = row + column * 3;
                denominator[i] = epsilon[i];
                numerator[i] = epsilon[i];
                if (row == column) {
                    denominator[i] = CUFFT_ADD(denominator[i],
                        CUFFT_MUL(CUFFT_REAL(CU_REAL(2)), epsilon0));
                    numerator[i] = CUFFT_SUB(numerator[i], epsilon0);
                }
            }
        }
        if (inverse_complex_matrix3(denominator, inverse)) {
            multiply_complex_matrix3(numerator, inverse, temporary);
            for (int i = 0; i < 9; ++i)
                alpha0[i] = CUFFT_MUL(temporary[i], CUFFT_REAL(volume));
            if (method == 1) {
                for (int i = 0; i < 9; ++i)
                    result[i] = CUFFT_MUL(alpha0[i], epsilon0);
            } else {
                const CUCOMPLEX index = scalar_polarizability_sqrt(epsilon0);
                const CUCOMPLEX index3 = CUFFT_MUL(CUFFT_MUL(index, index), index);
                const CUCOMPLEX k03 = CUFFT_MUL(CUFFT_REAL(k0 * k0 * k0), index3);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
                const CUCOMPLEX imaginary = make_cuFloatComplex(0.0f, 1.0f);
#else
                const CUCOMPLEX imaginary = make_cuDoubleComplex(0.0, 1.0);
#endif
                for (int column = 0; column < 3; ++column) {
                    for (int row = 0; row < 3; ++row) {
                        const int i = row + column * 3;
                        const CUCOMPLEX correction = CUFFT_MUL(
                            CUFFT_REAL(CU_REAL(2) / CU_REAL(3)),
                            CUFFT_MUL(imaginary, CUFFT_MUL(k03, alpha0[i])));
                        temporary[i] = CUFFT_SUB(
                            row == column ? CUFFT_REAL(CU_REAL(1)) : CUFFT_ZERO,
                            correction);
                    }
                }
                if (inverse_complex_matrix3(temporary, inverse)) {
                    multiply_complex_matrix3(inverse, alpha0, temporary);
                    for (int i = 0; i < 9; ++i)
                        result[i] = CUFFT_MUL(temporary[i], epsilon0);
                }
            }
        }
    }
    for (int i = 0; i < 9; ++i)
        output_alpha[size_t(cell) + size_t(nmax) * size_t(i)] = result[i];
}

//-----------------------------------------------------------
extern "C" cudaError_t labeled_material_from_cpu_tabdip_wrapper(
    const CU_INT* tabdip, const CU_INT* material_labels,
    CUCOMPLEX* polarizability, CUCOMPLEX* epsilon,
    const CUCOMPLEX* label_alpha, const CUCOMPLEX* label_epsilon,
    int grid_count, int nmax, int label_count, int anisotropic) {
    if (!tabdip || !material_labels || !polarizability || !epsilon ||
        !label_alpha || !label_epsilon || grid_count <= 0 || nmax <= 0 ||
        label_count <= 0 || (anisotropic != 0 && anisotropic != 1))
        return cudaErrorInvalidValue;
    labeled_material_from_cpu_tabdip_kernel<<<block_count(size_t(grid_count)), THREADS_PER_BLOCK>>>(
        tabdip, material_labels, polarizability, epsilon,
        label_alpha, label_epsilon, grid_count, nmax, label_count, anisotropic);
    return finish_kernel_launch(264, "labeled_material_from_cpu_tabdip_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scatter_cell_material_from_cpu_tabdip_wrapper(
    const CU_INT* tabdip, const CU_INT* inside_map,
    const CUCOMPLEX* cell_epsilon, const CUCOMPLEX* cell_alpha,
    CUCOMPLEX* output_epsilon, CUCOMPLEX* output_alpha,
    int grid_count, int nmax) {
    if (!tabdip || !inside_map || !cell_epsilon || !cell_alpha ||
        !output_epsilon || !output_alpha || grid_count <= 0 ||
        nmax <= 0)
        return cudaErrorInvalidValue;
    scatter_cell_material_from_cpu_tabdip_kernel<<<
        block_count(size_t(grid_count)), THREADS_PER_BLOCK>>>(
            tabdip, inside_map, cell_epsilon, cell_alpha,
            output_epsilon, output_alpha, grid_count, nmax);
    return finish_kernel_launch(265, "scatter_cell_material_from_cpu_tabdip_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t arbitrary_tensor_material_batch_wrapper(
    const CUCOMPLEX* cell_epsilon, CUCOMPLEX* output_epsilon,
    CUCOMPLEX* output_alpha, int count, int nmax, CUCOMPLEX epsilon0,
    CU_REAL radius, CU_REAL k0, int method) {
    if (!cell_epsilon || !output_epsilon || !output_alpha ||
        count <= 0 || count > nmax || radius <= CU_REAL(0) || k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    arbitrary_tensor_material_batch_kernel<<<
        block_count(size_t(count)), THREADS_PER_BLOCK>>>(
            cell_epsilon, output_epsilon, output_alpha,
            count, nmax, epsilon0, radius, k0, method);
    return finish_kernel_launch(266, "arbitrary_tensor_material_batch_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_polarizability_wrapper(
    CUCOMPLEX* result, CU_REAL* sldr, CU_REAL radius, CUCOMPLEX epsilon,
    CUCOMPLEX epsilon0, CU_REAL k0, int discretized, int method) {
    if (!result || !sldr || radius <= CU_REAL(0) || k0 <= CU_REAL(0) ||
        (discretized != 0 && discretized != 1) || method < 1 || method > 8)
        return cudaErrorInvalidValue;
    scalar_polarizability_kernel<<<1, 1>>>(
        result, sldr, radius, epsilon, epsilon0, k0, discretized, method);
    return finish_kernel_launch(203, "scalar_polarizability_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_polarizability_batch_wrapper(
    const CUCOMPLEX* epsilon, const CU_INT* inside, CUCOMPLEX* values, CU_REAL* sldr,
    int count, CU_REAL radius, CUCOMPLEX epsilon0, CU_REAL k0,
    int discretized, int method, CU_REAL initial_sldr) {
    if (!epsilon || !inside || !values || !sldr || count <= 0 || radius <= CU_REAL(0) ||
        k0 <= CU_REAL(0) || (discretized != 0 && discretized != 1) ||
        method < 1 || method > 8)
        return cudaErrorInvalidValue;
    scalar_polarizability_batch_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        epsilon, inside, values, sldr, count, radius, epsilon0, k0,
        discretized, method, initial_sldr);
    return finish_kernel_launch(255, "scalar_polarizability_batch_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t rerange_isotropic_polarizability_wrapper(
    const CUCOMPLEX *epsilon, CUCOMPLEX *polarizability, int count,
    CUCOMPLEX epsilon0, CU_REAL scale) {
    if (!epsilon || !polarizability || count <= 0)
        return cudaErrorInvalidValue;
    rerange_isotropic_polarizability_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        epsilon, polarizability, count, epsilon0, scale);
    return finish_kernel_launch(246, "rerange_isotropic_polarizability_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t material_interpolation_wrapper(
    CU_REAL *result, CU_REAL refractive_index, CU_REAL extinction_coefficient) {
    if (!result)
        return cudaErrorInvalidValue;
    material_interpolation_kernel<<<1, 1>>>(result, refractive_index,
                                             extinction_coefficient);
    return finish_kernel_launch(205, "material_interpolation_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t tensor_polarizability_wrapper(
    const CUCOMPLEX* epsilon, CUCOMPLEX* result, CU_INT* status,
    CUCOMPLEX epsilon0, CU_REAL radius, CU_REAL k0,
    int discretized, int method, int invert_result) {
    if (!epsilon || !result || !status || radius <= CU_REAL(0) || k0 <= CU_REAL(0) ||
        (discretized != 0 && discretized != 1) || (method != 1 && method != 2) ||
        (invert_result != 1 && invert_result != -1))
        return cudaErrorInvalidValue;
    tensor_polarizability_kernel<<<1, 1>>>(
        epsilon, result, status, epsilon0, radius, k0, discretized,
        method, invert_result);
    return finish_kernel_launch(204, "tensor_polarizability_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cuboid_geometry_wrapper(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
                                               CU_INT *dipole_map, int nx, int ny, int nz,
                                               CU_REAL cell_size, CU_REAL side_x, CU_REAL side_y,
                                               CU_REAL side_z, CU_REAL center_x, CU_REAL center_y,
                                               CU_REAL center_z) {
    if (!xs || !ys || !zs || !dipole_map || nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny ||
        nx * ny > INT_MAX / nz)
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    cuboid_geometry_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, dipole_map, nx, ny, nz, cell_size, side_x, side_y, side_z, center_x, center_y,
        center_z, count);
    return finish_kernel_launch(121, "cuboid_geometry_kernel");
}

extern "C" cudaError_t cuboid_dipole_map_wrapper(
    CU_INT* dipole_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL side_x, CU_REAL side_y, CU_REAL side_z) {
    if (!dipole_map || nx <= 0 || ny <= 0 || nz <= 0 ||
        nx > INT_MAX / ny || nx*ny > INT_MAX / nz)
        return cudaErrorInvalidValue;
    const int count = nx*ny*nz;
    cuboid_geometry_kernel<<<block_count(size_t(count)),THREADS_PER_BLOCK>>>(
        nullptr,nullptr,nullptr,dipole_map,nx,ny,nz,cell_size,
        side_x,side_y,side_z,CU_REAL(0),CU_REAL(0),CU_REAL(0),count);
    return finish_kernel_launch(267,"cuboid_dipole_map_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t rotated_cuboid_geometry_wrapper(
    CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, CU_INT*inside_map, int nx, int ny, int nz,
    CU_REAL cell_size, CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z, CU_REAL side_x,
    CU_REAL side_y, CU_REAL side_z, CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
    CU_REAL r0, CU_REAL r1, CU_REAL r2, CU_REAL r3, CU_REAL r4, CU_REAL r5, CU_REAL r6, CU_REAL r7,
    CU_REAL r8) {
    if (!xs || !ys || !zs || !inside_map || nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny ||
        nx * ny > INT_MAX / nz)
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    rotated_cuboid_geometry_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, inside_map, nx, ny, nz, cell_size, extent_x, extent_y, extent_z, side_x, side_y,
        side_z, center_x, center_y, center_z, r0, r1, r2, r3, r4, r5, r6, r7, r8, count);
    return finish_kernel_launch(122, "rotated_cuboid_geometry_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t rotated_cuboid_dipole_map_wrapper(
    CU_INT* inside_map, int nx, int ny, int nz,
    CU_REAL cell_size, CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
    CU_REAL side_x, CU_REAL side_y, CU_REAL side_z,
    CU_REAL r0, CU_REAL r1, CU_REAL r2, CU_REAL r3,
    CU_REAL r4, CU_REAL r5, CU_REAL r6, CU_REAL r7, CU_REAL r8) {
    if (!inside_map || nx <= 0 || ny <= 0 || nz <= 0 ||
        nx > INT_MAX / ny || nx * ny > INT_MAX / nz)
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    rotated_cuboid_dipole_map_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        inside_map, nx, ny, nz, cell_size, extent_x, extent_y, extent_z,
        side_x, side_y, side_z, r0,r1,r2,r3,r4,r5,r6,r7,r8,count);
    return finish_kernel_launch(266, "rotated_cuboid_dipole_map_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t sphere_geometry_wrapper(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
                                               CU_INT*inside_map, int subdivisions, CU_REAL radius,
                                               CU_REAL cell_size, CU_REAL center_x,
                                               CU_REAL center_y, CU_REAL center_z) {
    if (!xs || !ys || !zs || !inside_map || subdivisions <= 0 ||
        subdivisions > INT_MAX / subdivisions ||
        subdivisions * subdivisions > INT_MAX / subdivisions)
        return cudaErrorInvalidValue;
    const int count = subdivisions * subdivisions * subdivisions;
    sphere_geometry_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, inside_map, subdivisions, radius, cell_size, center_x, center_y, center_z,
        count);
    return finish_kernel_launch(123, "sphere_geometry_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t sphere_dipole_map_wrapper(
    CU_INT* dipole_map, int subdivisions, CU_REAL radius, CU_REAL cell_size) {
    if (!dipole_map || subdivisions <= 0 || subdivisions > INT_MAX / subdivisions ||
        subdivisions * subdivisions > INT_MAX / subdivisions)
        return cudaErrorInvalidValue;
    const int count = subdivisions * subdivisions * subdivisions;
    sphere_dipole_map_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        dipole_map, subdivisions, radius, cell_size, count);
    return finish_kernel_launch(123, "sphere_dipole_map_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t sphere_geometry_compaction_workspace_wrapper(
    int count, size_t* temporary_bytes) {
    if (count <= 0 || !temporary_bytes)
        return cudaErrorInvalidValue;
    *temporary_bytes = 0;
    return cub::DeviceScan::ExclusiveScan(
        nullptr, *temporary_bytes, static_cast<const CU_INT*>(nullptr),
        static_cast<CU_INT*>(nullptr), CuIntSum{}, CU_INT{0}, count);
}

//-----------------------------------------------------------
extern "C" cudaError_t sphere_geometry_compact_wrapper(
    const CU_REAL* mesh_x, const CU_REAL* mesh_y, const CU_REAL* mesh_z,
    const CU_INT* inside_map, int count, CU_INT* prefix,
    CU_REAL* packed_x, CU_REAL* packed_y, CU_REAL* packed_z,
    CU_INT* packed_count, void* temporary, size_t temporary_bytes) {
    if (!mesh_x || !mesh_y || !mesh_z || !inside_map || count <= 0 || !prefix ||
        !packed_x || !packed_y || !packed_z || !packed_count || !temporary)
        return cudaErrorInvalidValue;
    size_t required_bytes = temporary_bytes;
    cudaError_t status = cub::DeviceScan::ExclusiveScan(
        temporary, required_bytes, inside_map, prefix, CuIntSum{}, CU_INT{0}, count);
    if (status != cudaSuccess)
        return status;
    sphere_geometry_compact_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        mesh_x, mesh_y, mesh_z, inside_map, prefix, count,
        packed_x, packed_y, packed_z, packed_count);
    return finish_kernel_launch(241, "sphere_geometry_compact_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t ellipsoid_geometry_wrapper(
    CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, CU_INT*inside_map, int nx, int ny, int nz,
    CU_REAL cell_size, CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z, CU_REAL radius_a,
    CU_REAL radius_b, CU_REAL radius_c, CU_REAL m0, CU_REAL m1, CU_REAL m2, CU_REAL m3, CU_REAL m4,
    CU_REAL m5, CU_REAL m6, CU_REAL m7, CU_REAL m8) {
    if (!xs || !ys || !zs || !inside_map || nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny ||
        nx * ny > INT_MAX / nz || radius_a == CU_REAL(0) || radius_b == CU_REAL(0) ||
        radius_c == CU_REAL(0))
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    ellipsoid_geometry_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, inside_map, nx, ny, nz, cell_size, extent_x, extent_y, extent_z, radius_a,
        radius_b, radius_c, m0, m1, m2, m3, m4, m5, m6, m7, m8, count);
    return finish_kernel_launch(124, "ellipsoid_geometry_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t ellipsoid_dipole_map_wrapper(
    CU_INT* inside_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
    CU_REAL radius_a, CU_REAL radius_b, CU_REAL radius_c,
    CU_REAL m0, CU_REAL m1, CU_REAL m2, CU_REAL m3, CU_REAL m4,
    CU_REAL m5, CU_REAL m6, CU_REAL m7, CU_REAL m8) {
    if (!inside_map || nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny ||
        nx * ny > INT_MAX / nz || radius_a == CU_REAL(0) ||
        radius_b == CU_REAL(0) || radius_c == CU_REAL(0))
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    ellipsoid_dipole_map_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        inside_map, nx, ny, nz, cell_size, extent_x, extent_y, extent_z,
        radius_a, radius_b, radius_c, m0,m1,m2,m3,m4,m5,m6,m7,m8,count);
    return finish_kernel_launch(267, "ellipsoid_dipole_map_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cylinder_geometry_wrapper(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
                                                 CU_INT*inside_map, int nx, int ny, int nz,
                                                 CU_REAL cell_size, CU_REAL extent_x,
                                                 CU_REAL extent_y, CU_REAL extent_z, CU_REAL radius,
                                                 CU_REAL height, CU_REAL m0, CU_REAL m1, CU_REAL m2,
                                                 CU_REAL m3, CU_REAL m4, CU_REAL m5, CU_REAL m6,
                                                 CU_REAL m7, CU_REAL m8) {
    if (!xs || !ys || !zs || !inside_map || nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny ||
        nx * ny > INT_MAX / nz || radius == CU_REAL(0) || height == CU_REAL(0))
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    cylinder_geometry_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, inside_map, nx, ny, nz, cell_size, extent_x, extent_y, extent_z, radius, height,
        m0, m1, m2, m3, m4, m5, m6, m7, m8, count);
    return finish_kernel_launch(125, "cylinder_geometry_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cylinder_dipole_map_wrapper(
    CU_INT* inside_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
    CU_REAL radius, CU_REAL height,
    CU_REAL m0, CU_REAL m1, CU_REAL m2, CU_REAL m3, CU_REAL m4,
    CU_REAL m5, CU_REAL m6, CU_REAL m7, CU_REAL m8) {
    if (!inside_map || nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny ||
        nx * ny > INT_MAX / nz || radius == CU_REAL(0) || height == CU_REAL(0))
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    cylinder_dipole_map_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        inside_map, nx, ny, nz, cell_size, extent_x, extent_y, extent_z,
        radius, height, m0,m1,m2,m3,m4,m5,m6,m7,m8,count);
    return finish_kernel_launch(268, "cylinder_dipole_map_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t inhomogeneous_material_spectrum_wrapper(
    CUCOMPLEX *spectrum, const CU_REAL *phases, int nx, int ny, int nz,
    CU_REAL dkx, CU_REAL dky, CU_REAL dkz, CU_REAL coefficient1,
    CU_REAL coefficient2, CU_REAL coefficient3) {
    if (!spectrum || !phases || nx <= 0 || ny <= 0 || nz <= 0 ||
        nx > INT_MAX / ny || nx * ny > INT_MAX / nz)
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    inhomogeneous_material_spectrum_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        spectrum, phases, nx, ny, nz, dkx, dky, dkz, coefficient1, coefficient2,
        coefficient3, count);
    return finish_kernel_launch(230, "inhomogeneous_material_spectrum_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t inhomogeneous_material_finalize_wrapper(
    const CUCOMPLEX *spectrum, CU_REAL *real_values, CU_REAL *magnitude_squares,
    int count, CU_REAL epsilon_real, CU_REAL epsilon_imag) {
    if (!spectrum || !real_values || !magnitude_squares || count <= 0)
        return cudaErrorInvalidValue;
    inhomogeneous_material_finalize_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        spectrum, real_values, magnitude_squares, count, epsilon_real, epsilon_imag);
    return finish_kernel_launch(231, "inhomogeneous_material_finalize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t fill_real_ones_wrapper(CU_REAL *values, int count) {
    if (!values || count <= 0)
        return cudaErrorInvalidValue;
    fill_real_ones_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(values, count);
    return finish_kernel_launch(232, "fill_real_ones_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multi_sphere_geometry_wrapper(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
                                                     CU_INT*sphere_map, int nx, int ny, int nz,
                                                     CU_REAL cell_size, CU_REAL x_min,
                                                     CU_REAL y_min, CU_REAL z_min,
                                                     const CU_REAL *sphere_parameters,
                                                     int sphere_count, int fixed_sphere_count) {
    if (!xs || !ys || !zs || !sphere_map || !sphere_parameters || nx <= 0 || ny <= 0 || nz <= 0 ||
        nx > INT_MAX / ny || nx * ny > INT_MAX / nz || sphere_count <= 0 ||
        fixed_sphere_count < 0 || fixed_sphere_count > sphere_count)
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    multi_sphere_geometry_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, sphere_map, nx, ny, nz, cell_size, x_min, y_min, z_min, sphere_parameters,
        sphere_count, fixed_sphere_count, count);
    return finish_kernel_launch(126, "multi_sphere_geometry_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t concentric_sphere_geometry_wrapper(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
                                                          CU_INT*layer_map, int subdivisions,
                                                          CU_REAL cell_size, CU_REAL center,
                                                          const CU_REAL *radii, int sphere_count) {
    if (!xs || !ys || !zs || !layer_map || !radii || subdivisions <= 0 ||
        subdivisions > INT_MAX / subdivisions ||
        subdivisions * subdivisions > INT_MAX / subdivisions || sphere_count <= 0 ||
        sphere_count > 20)
        return cudaErrorInvalidValue;
    const int count = subdivisions * subdivisions * subdivisions;
    concentric_sphere_geometry_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, layer_map, subdivisions, cell_size, center, radii, sphere_count, count);
    return finish_kernel_launch(127, "concentric_sphere_geometry_kernel");
}

//-----------------------------------------------------------
// Reuse the original occupancy calculations without materializing the XYZ arrays.
extern "C" cudaError_t multi_sphere_dipole_map_wrapper(
    CU_INT* sphere_map, int nx, int ny, int nz, CU_REAL cell_size,
    CU_REAL x_min, CU_REAL y_min, CU_REAL z_min,
    const CU_REAL* sphere_parameters, int sphere_count, int fixed_sphere_count) {
    if (!sphere_map || !sphere_parameters || nx <= 0 || ny <= 0 || nz <= 0 ||
        nx > INT_MAX / ny || nx * ny > INT_MAX / nz || sphere_count <= 0 ||
        fixed_sphere_count < 0 || fixed_sphere_count > sphere_count)
        return cudaErrorInvalidValue;
    const int count = nx * ny * nz;
    multi_sphere_geometry_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        nullptr, nullptr, nullptr, sphere_map, nx, ny, nz, cell_size,
        x_min, y_min, z_min, sphere_parameters, sphere_count, fixed_sphere_count, count);
    return finish_kernel_launch(265, "multi_sphere_dipole_map_kernel");
}

extern "C" cudaError_t concentric_sphere_dipole_map_wrapper(
    CU_INT* layer_map, int subdivisions, CU_REAL cell_size, CU_REAL center,
    const CU_REAL* radii, int sphere_count) {
    if (!layer_map || !radii || subdivisions <= 0 ||
        subdivisions > INT_MAX / subdivisions ||
        subdivisions * subdivisions > INT_MAX / subdivisions ||
        sphere_count <= 0 || sphere_count > 20)
        return cudaErrorInvalidValue;
    const int count = subdivisions * subdivisions * subdivisions;
    concentric_sphere_geometry_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        nullptr, nullptr, nullptr, layer_map, subdivisions, cell_size,
        center, radii, sphere_count, count);
    return finish_kernel_launch(266, "concentric_sphere_dipole_map_kernel");
}

//-----------------------------------------------------------
__global__ void incident_plane_wave_kernel(const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
                                           CUCOMPLEX *field, int count, CU_REAL kx, CU_REAL ky,
                                           CUCOMPLEX kz, CUCOMPLEX ex0, CUCOMPLEX ey0,
                                           CUCOMPLEX ez0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CU_REAL phase = xs[i] * kx + ys[i] * ky + zs[i] * kz.x;
    const CU_REAL amplitude = exp(-zs[i] * kz.y);
    const CUCOMPLEX wave = make_cuFloatComplex(amplitude * cosf(phase), amplitude * sinf(phase));
#else
    const CU_REAL phase = xs[i] * kx + ys[i] * ky + zs[i] * kz.x;
    const CU_REAL amplitude = exp(-zs[i] * kz.y);
    const CUCOMPLEX wave = make_cuDoubleComplex(amplitude * cos(phase), amplitude * sin(phase));
#endif
    field[3 * i] = CUFFT_MUL(ex0, wave);
    field[3 * i + 1] = CUFFT_MUL(ey0, wave);
    field[3 * i + 2] = CUFFT_MUL(ez0, wave);
}

//-----------------------------------------------------------
// Host-only conversion at the CUDA launch boundary. CPU scalars are COMPLEX;
// CUCOMPLEX is used for values consumed by the device kernels.
static CUCOMPLEX plane_wave_device_scalar(COMPLEX value) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(static_cast<CU_REAL>(value.real()),
                               static_cast<CU_REAL>(value.imag()));
#else
    return make_cuDoubleComplex(static_cast<CU_REAL>(value.real()),
                                static_cast<CU_REAL>(value.imag()));
#endif
}

//-----------------------------------------------------------
extern "C" cudaError_t incident_plane_wave_wrapper(const CU_REAL *xs, const CU_REAL *ys,
                                                   const CU_REAL *zs, CUCOMPLEX *field, int count,
                                                   REAL kx, REAL ky, COMPLEX kz,
                                                   COMPLEX ex0, COMPLEX ey0, COMPLEX ez0) {
    if (!xs || !ys || !zs || !field || count < 0)
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    incident_plane_wave_kernel<<<(count + 255) / 256, 256>>>(
        xs, ys, zs, field, count, kx, ky,
        plane_wave_device_scalar(kz), plane_wave_device_scalar(ex0),
        plane_wave_device_scalar(ey0), plane_wave_device_scalar(ez0));
    return finish_kernel_launch(200, "incident_plane_wave_kernel");
}

//-----------------------------------------------------------
__global__ void incident_plane_wave_grid_kernel(
    const CU_INT* tabdip, CUCOMPLEX* field, int nx, int ny, int grid_count,
    int dipole_count, int anchor_ix, int anchor_iy, int anchor_iz,
    CU_REAL anchor_x, CU_REAL anchor_y, CU_REAL anchor_z, CU_REAL cell_size,
    CU_REAL kx, CU_REAL ky, CUCOMPLEX kz,
    CUCOMPLEX ex0, CUCOMPLEX ey0, CUCOMPLEX ez0) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= grid_count)
        return;
    // This is a full-grid -> one-based compact index map, not its inverse.
    const int compact = tabdip[flat].dat - 1;
    if (compact < 0 || compact >= dipole_count)
        return;
    const int ix = flat % nx;
    const int iy = (flat / nx) % ny;
    const int iz = flat / (nx * ny);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CU_REAL x = fmaf(CU_REAL(ix - anchor_ix), cell_size, anchor_x);
    const CU_REAL y = fmaf(CU_REAL(iy - anchor_iy), cell_size, anchor_y);
    const CU_REAL z = fmaf(CU_REAL(iz - anchor_iz), cell_size, anchor_z);
    const CU_REAL phase = x * kx + y * ky + z * kz.x;
    const CU_REAL amplitude = expf(-z * kz.y);
    const CUCOMPLEX wave = make_cuFloatComplex(amplitude * cosf(phase),
                                               amplitude * sinf(phase));
#else
    const CU_REAL x = fma(CU_REAL(ix - anchor_ix), cell_size, anchor_x);
    const CU_REAL y = fma(CU_REAL(iy - anchor_iy), cell_size, anchor_y);
    const CU_REAL z = fma(CU_REAL(iz - anchor_iz), cell_size, anchor_z);
    const CU_REAL phase = x * kx + y * ky + z * kz.x;
    const CU_REAL amplitude = exp(-z * kz.y);
    const CUCOMPLEX wave = make_cuDoubleComplex(amplitude * cos(phase),
                                                amplitude * sin(phase));
#endif
    field[3 * compact] = CUFFT_MUL(ex0, wave);
    field[3 * compact + 1] = CUFFT_MUL(ey0, wave);
    field[3 * compact + 2] = CUFFT_MUL(ez0, wave);
}

//-----------------------------------------------------------
extern "C" cudaError_t incident_plane_wave_grid_wrapper(
    const CU_INT* tabdip, CUCOMPLEX* field, int nx, int ny, int nz,
    int dipole_count, int anchor_ix, int anchor_iy, int anchor_iz,
    REAL anchor_x, REAL anchor_y, REAL anchor_z, REAL cell_size,
    REAL kx, REAL ky, COMPLEX kz,
    COMPLEX ex0, COMPLEX ey0, COMPLEX ez0) {
    if (!tabdip || !field || nx <= 0 || ny <= 0 || nz <= 0 ||
        nx > INT_MAX / ny || nx * ny > INT_MAX / nz ||
        dipole_count <= 0 || cell_size <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const int grid_count = nx * ny * nz;
    incident_plane_wave_grid_kernel<<<(grid_count + 255) / 256, 256>>>(
        tabdip, field, nx, ny, grid_count, dipole_count,
        anchor_ix, anchor_iy, anchor_iz, anchor_x, anchor_y, anchor_z,
        cell_size, kx, ky,
        plane_wave_device_scalar(kz), plane_wave_device_scalar(ex0),
        plane_wave_device_scalar(ey0), plane_wave_device_scalar(ez0));
    return finish_kernel_launch(200, "incident_plane_wave_grid_kernel");
}

//-----------------------------------------------------------
__device__ void incident_plane_wave_derivative_point(
    CUCOMPLEX* output, CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL k0,
    CUCOMPLEX amplitude, CU_REAL ss, CU_REAL pp_input, CU_REAL theta_degrees,
    CU_REAL phi_degrees, int derivative_selector, int circular) {
    for (int i = 0; i < 9; ++i) {
        output[i] = CUFFT_ZERO;
    }

    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);
    const CU_REAL theta = theta_degrees * pi / CU_REAL(180);
    const CU_REAL phi = phi_degrees * pi / CU_REAL(180);
    const CU_REAL sin_theta = sin(theta);
    const CU_REAL cos_theta = cos(theta);
    const CU_REAL sin_phi = sin(phi);
    const CU_REAL cos_phi = cos(phi);
    const CU_REAL kx = k0 * sin_theta * cos_phi;
    const CU_REAL ky = k0 * sin_theta * sin_phi;
    const CU_REAL kz = k0 * cos_theta;
    CU_REAL p = CU_REAL(0);
    CU_REAL s = CU_REAL(0);

    if (!circular) {
        CU_REAL pp = pp_input;
        p = sqrt(pp);
        s = sqrt(CU_REAL(1) - pp);
        if (fabs(pp - CU_REAL(3)) <= CU_REAL(3e-12)) {
            CU_REAL angle = phi_degrees;
            if (angle < CU_REAL(0)) angle += CU_REAL(360);
            pp = (sin_phi * sin_phi) /
                 (cos_theta * cos_theta * cos_phi * cos_phi + sin_phi * sin_phi);
            p = sqrt(pp);
            s = sqrt(CU_REAL(1) - pp);
            if (angle >= CU_REAL(90) && angle < CU_REAL(180)) s = -s;
            else if (angle >= CU_REAL(180) && angle < CU_REAL(270)) { p = -p; s = -s; }
            else if (angle >= CU_REAL(270) && angle < CU_REAL(360)) p = -p;
        }
        if (fabs(pp - CU_REAL(2)) <= CU_REAL(2e-12)) {
            CU_REAL angle = phi_degrees;
            pp = (cos_phi * cos_phi) /
                 (cos_theta * cos_theta * sin_phi * sin_phi + cos_phi * cos_phi);
            if (angle < CU_REAL(0)) angle += CU_REAL(360);
            p = sqrt(pp);
            s = sqrt(CU_REAL(1) - pp);
            if (angle >= CU_REAL(0) && angle < CU_REAL(90)) s = -s;
            else if (angle >= CU_REAL(90) && angle < CU_REAL(180)) { p = -p; s = -s; }
            else if (angle >= CU_REAL(180) && angle < CU_REAL(270)) { p = -p; }
        }
    }

    const CU_REAL direction = theta_degrees < CU_REAL(0) ? CU_REAL(-1) : CU_REAL(1);
    CUCOMPLEX e0s;
    CUCOMPLEX e0p;
    if (circular) {
        const CU_REAL inv_sqrt_two = CU_REAL(1) / sqrt(CU_REAL(2));
        const CU_REAL handedness = ss < CU_REAL(0) ? CU_REAL(-1) : CU_REAL(1);
        e0s = CUFFT_MUL(amplitude, CUFFT_REAL(inv_sqrt_two));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        e0p = make_cuFloatComplex(-amplitude.y * inv_sqrt_two * handedness,
                                  amplitude.x * inv_sqrt_two * handedness);
#else
        e0p = make_cuDoubleComplex(-amplitude.y * inv_sqrt_two * handedness,
                                   amplitude.x * inv_sqrt_two * handedness);
#endif
    } else {
        e0s = CUFFT_MUL(amplitude, CUFFT_REAL(s * direction));
        e0p = CUFFT_MUL(amplitude, CUFFT_REAL(p * direction));
    }

    CUCOMPLEX ex = CUFFT_ADD(CUFFT_MUL(e0s, CUFFT_REAL(-sin_phi)),
                             CUFFT_MUL(e0p, CUFFT_REAL(cos_theta * cos_phi)));
    CUCOMPLEX ey = CUFFT_ADD(CUFFT_MUL(e0s, CUFFT_REAL(cos_phi)),
                             CUFFT_MUL(e0p, CUFFT_REAL(cos_theta * sin_phi)));
    CUCOMPLEX ez = CUFFT_MUL(e0p, CUFFT_REAL(-sin_theta));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cos(x * kx + y * ky + z * kz),
                                                 sin(x * kx + y * ky + z * kz));
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(x * kx + y * ky + z * kz),
                                                  sin(x * kx + y * ky + z * kz));
#endif
    const CUCOMPLEX fields[3] = {CUFFT_MUL(ex, phase), CUFFT_MUL(ey, phase),
                                 CUFFT_MUL(ez, phase)};
    const CU_REAL wave_numbers[3] = {kx, ky, kz};
    const int first_column = derivative_selector == 4 ? 0 : derivative_selector - 1;
    const int last_column = derivative_selector == 4 ? 2 : first_column;
    for (int column = first_column; column <= last_column; ++column) {
        for (int row = 0; row < 3; ++row) {
            const CUCOMPLEX field = fields[row];
            const CU_REAL wave_number = wave_numbers[column];
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            output[row + column * 3] = make_cuFloatComplex(-field.y * wave_number,
                                                            field.x * wave_number);
#else
            output[row + column * 3] = make_cuDoubleComplex(-field.y * wave_number,
                                                             field.x * wave_number);
#endif
        }
    }
}

//-----------------------------------------------------------
__global__ void incident_plane_wave_derivative_kernel(
    CUCOMPLEX* output, CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL k0,
    CUCOMPLEX amplitude, CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees,
    CU_REAL phi_degrees, int derivative_selector, int circular) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    incident_plane_wave_derivative_point(
        output, x, y, z, k0, amplitude, ss, pp, theta_degrees, phi_degrees,
        derivative_selector, circular);
}

//-----------------------------------------------------------
__global__ void incident_plane_wave_derivative_batch_kernel(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    CUCOMPLEX* output, int count, CU_REAL k0, CUCOMPLEX amplitude,
    CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees, CU_REAL phi_degrees,
    int circular) {
    const int point = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (point >= count)
        return;
    incident_plane_wave_derivative_point(
        output + size_t(point) * 9, xs[point], ys[point], zs[point], k0,
        amplitude, ss, pp, theta_degrees, phi_degrees, 4, circular);
}

//-----------------------------------------------------------
extern "C" cudaError_t incident_plane_wave_derivative_wrapper(
    CUCOMPLEX* output, CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL k0,
    CUCOMPLEX amplitude, CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees,
    CU_REAL phi_degrees, int derivative_selector, int circular) {
    if (!output || k0 <= CU_REAL(0) || derivative_selector < 1 ||
        derivative_selector > 4 || (circular != 0 && circular != 1)) {
        return cudaErrorInvalidValue;
    }
    incident_plane_wave_derivative_kernel<<<1, 1>>>(
        output, x, y, z, k0, amplitude, ss, pp, theta_degrees, phi_degrees,
        derivative_selector, circular);
    return finish_kernel_launch(202, "incident_plane_wave_derivative_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t incident_plane_wave_derivative_batch_wrapper(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, CUCOMPLEX* output,
    int count, CU_REAL k0, CUCOMPLEX amplitude, CU_REAL ss, CU_REAL pp,
    CU_REAL theta_degrees, CU_REAL phi_degrees, int circular) {
    if (!xs || !ys || !zs || !output || count < 0 || k0 <= CU_REAL(0) ||
        (circular != 0 && circular != 1))
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    incident_plane_wave_derivative_batch_kernel<<<(count + 255) / 256, 256>>>(
        xs, ys, zs, output, count, k0, amplitude, ss, pp, theta_degrees,
        phi_degrees, circular);
    return finish_kernel_launch(261, "incident_plane_wave_derivative_batch_kernel");
}

//-----------------------------------------------------------
__global__ void incident_multi_plane_wave_kernel(const CU_REAL *xs, const CU_REAL *ys,
                                                 const CU_REAL *zs, const CUCOMPLEX *parameters,
                                                 CUCOMPLEX *field, int count, int plane_count,
                                                 IfddaIncidentGrid grid) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    int i = flat;
    REAL x = 0, y = 0, z = 0;
    if (grid.tabdip) {
        if (!ifdda_incident_grid_fetch(grid, flat, &i, &x, &y, &z)) return;
    } else if (i < count) {
        x = xs[i]; y = ys[i]; z = zs[i];
    } else return;
    CUCOMPLEX ex = CUFFT_ZERO, ey = CUFFT_ZERO, ez = CUFFT_ZERO;
    for (int b = 0; b < plane_count; ++b) {
        const CUCOMPLEX *p = parameters + 6 * b;
        const REAL phase = x * p[0].x + y * p[1].x + z * p[2].x;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const REAL amplitude = exp(-z * p[2].y);
        const CUCOMPLEX wave =
            make_cuFloatComplex(amplitude * cosf(phase), amplitude * sinf(phase));
#else
        const REAL amplitude = exp(-z * p[2].y);
        const CUCOMPLEX wave = make_cuDoubleComplex(amplitude * cos(phase), amplitude * sin(phase));
#endif
        ex = CUFFT_ADD(ex, CUFFT_MUL(p[3], wave));
        ey = CUFFT_ADD(ey, CUFFT_MUL(p[4], wave));
        ez = CUFFT_ADD(ez, CUFFT_MUL(p[5], wave));
    }
    field[3 * i] = ex;
    field[3 * i + 1] = ey;
    field[3 * i + 2] = ez;
}

//-----------------------------------------------------------
__global__ void incident_multi_plane_wave_derivative_batch_kernel(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    const CUCOMPLEX* parameters, CUCOMPLEX* derivatives,
    int count, int plane_count) {
    const int point = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (point >= count)
        return;
    CUCOMPLEX result[9];
    for (int i = 0; i < 9; ++i)
        result[i] = CUFFT_ZERO;
    for (int beam = 0; beam < plane_count; ++beam) {
        const CUCOMPLEX* p = parameters + size_t(beam) * 6;
        const CU_REAL phase = xs[point] * p[0].x + ys[point] * p[1].x +
                              zs[point] * p[2].x;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CUCOMPLEX wave = make_cuFloatComplex(cosf(phase), sinf(phase));
#else
        const CUCOMPLEX wave = make_cuDoubleComplex(cos(phase), sin(phase));
#endif
        for (int axis = 0; axis < 3; ++axis) {
            const CU_REAL wave_number = p[axis].x;
            for (int row = 0; row < 3; ++row) {
                const CUCOMPLEX field = CUFFT_MUL(p[3 + row], wave);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
                const CUCOMPLEX term = make_cuFloatComplex(-field.y * wave_number,
                                                            field.x * wave_number);
#else
                const CUCOMPLEX term = make_cuDoubleComplex(-field.y * wave_number,
                                                             field.x * wave_number);
#endif
                result[row + axis * 3] = CUFFT_ADD(result[row + axis * 3], term);
            }
        }
    }
    for (int i = 0; i < 9; ++i)
        derivatives[size_t(point) * 9 + i] = result[i];
}

//-----------------------------------------------------------
extern "C" cudaError_t incident_multi_plane_wave_wrapper(const CU_REAL *xs, const CU_REAL *ys,
                                                         const CU_REAL *zs,
                                                         const CUCOMPLEX *parameters,
                                                         CUCOMPLEX *field, int count,
                                                         int plane_count) {
    if (!xs || !ys || !zs || !parameters || !field || count < 0 || plane_count <= 0 ||
        plane_count > 10)
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    incident_multi_plane_wave_kernel<<<(count + 255) / 256, 256>>>(xs, ys, zs, parameters, field,
                                                                   count, plane_count, {});
    return finish_kernel_launch(201, "incident_multi_plane_wave_kernel");
}

extern "C" cudaError_t incident_multi_plane_wave_grid_wrapper(
    IfddaIncidentGrid grid, const CUCOMPLEX* parameters, CUCOMPLEX* field,
    int plane_count) {
    if (!grid.tabdip || !parameters || !field || grid.grid_count <= 0 ||
        grid.dipole_count <= 0 || grid.nx <= 0 || grid.ny <= 0 ||
        plane_count <= 0 || plane_count > 10) return cudaErrorInvalidValue;
    incident_multi_plane_wave_kernel<<<(grid.grid_count + 255) / 256, 256>>>(
        nullptr, nullptr, nullptr, parameters, field, grid.dipole_count,
        plane_count, grid);
    return finish_kernel_launch(201, "incident_multi_plane_wave_grid_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t incident_multi_plane_wave_derivative_batch_wrapper(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    const CUCOMPLEX* plane_parameters, CUCOMPLEX* field_derivatives,
    int count, int plane_count) {
    if (!xs || !ys || !zs || !plane_parameters || !field_derivatives ||
        count < 0 || plane_count <= 0 || plane_count > 10)
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    incident_multi_plane_wave_derivative_batch_kernel<<<(count + 255) / 256, 256>>>(
        xs, ys, zs, plane_parameters, field_derivatives, count, plane_count);
    return finish_kernel_launch(262, "incident_multi_plane_wave_derivative_batch_kernel");
}

//-----------------------------------------------------------
__device__ CUCOMPLEX speckle_exp_i(CU_REAL angle) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(cosf(angle), sinf(angle));
#else
    return make_cuDoubleComplex(cos(angle), sin(angle));
#endif
}

//-----------------------------------------------------------
__global__ void speckle_uniforms_to_phase_kernel(
    const CU_REAL* uniforms, CUCOMPLEX* phases, size_t count, CU_REAL two_pi)
{
    const size_t index = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (index >= count)
        return;
    const CU_REAL uniform = uniforms[index];
    phases[index] = uniform < CU_REAL(0)
        ? CUFFT_ZERO
        : speckle_exp_i(two_pi * uniform);
}

//-----------------------------------------------------------
__global__ void speckle_spectrum_kernel(const CUCOMPLEX *random_phase, CUCOMPLEX *spectrum,
                                        int nfft, CU_REAL k0, CU_REAL deltak, CU_REAL kmax,
                                        CU_REAL psi, CUCOMPLEX e0, CU_REAL x0, CU_REAL y0,
                                        CU_REAL z0, CU_REAL z_plane, int direction) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane = nfft * nfft;
    if (index >= plane)
        return;
    const int half = nfft / 2;
    const int nx = index % nfft;
    const int ny = index / nfft;
    const int nkx = nx < half ? nx : nx - nfft;
    const int nky = ny < half ? ny : ny - nfft;
    const CU_REAL kx = deltak * CU_REAL(nkx);
    const CU_REAL ky = deltak * CU_REAL(nky);
    const CU_REAL kp2 = kx * kx + ky * ky;
    if (kp2 > kmax * kmax * CU_REAL(0.999)) {
        spectrum[index] = CUFFT_ZERO;
        spectrum[plane + index] = CUFFT_ZERO;
        spectrum[2 * plane + index] = CUFFT_ZERO;
        return;
    }
    const CU_REAL kz = CU_REAL(direction) * sqrt(k0 * k0 - kp2);
    const CU_REAL denom_x = sqrt(kz * kz + kx * kx);
    const CU_REAL denom_y = sqrt(kz * kz + ky * ky);
    if (denom_x == CU_REAL(0) || denom_y == CU_REAL(0)) {
        spectrum[index] = CUFFT_ZERO;
        spectrum[plane + index] = CUFFT_ZERO;
        spectrum[2 * plane + index] = CUFFT_ZERO;
        return;
    }
    const CU_REAL cpsi = cos(psi), spsi = sin(psi);
    const CU_REAL axp = kz / denom_x;
    const CU_REAL azp = -kx / denom_x;
    const CU_REAL ays = kz / denom_y;
    const CU_REAL azs = -ky / denom_y;
    const CUCOMPLEX ax = CUFFT_MUL(e0, CUFFT_REAL(cpsi * axp));
    const CUCOMPLEX ay = CUFFT_MUL(e0, CUFFT_REAL(spsi * ays));
    const CUCOMPLEX az = CUFFT_MUL(e0, CUFFT_REAL(cpsi * azp + spsi * azs));
    const CUCOMPLEX phase = speckle_exp_i(-(kx * x0 + ky * y0 + kz * z0) + kz * z_plane);
    CUCOMPLEX sign = CUFFT_REAL(CU_REAL(direction));
    const CUCOMPLEX random = random_phase ? random_phase[index] : CUFFT_REAL(CU_REAL(1));
    spectrum[index] = CUFFT_MUL(CUFFT_MUL(CUFFT_MUL(ax, phase), random), sign);
    spectrum[plane + index] = CUFFT_MUL(CUFFT_MUL(CUFFT_MUL(ay, phase), random), sign);
    spectrum[2 * plane + index] = CUFFT_MUL(CUFFT_MUL(CUFFT_MUL(az, phase), random), sign);
}

//-----------------------------------------------------------
__global__ void diffraction_incident_spectrum_kernel(
    CUCOMPLEX* spectrum, const CUCOMPLEX* plane_parameters,
    const CUCOMPLEX* random_phases, int nfft, int imax, int mode,
    int plane_count, CU_REAL k0, CU_REAL deltakx, CU_REAL deltaky,
    CU_REAL kmax_squared, CU_REAL psi, CU_REAL x0, CU_REAL y0,
    CU_REAL z0, CUCOMPLEX e0) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int width = 2 * imax + 1;
    const int count = width * width;
    if (index >= count)
        return;

    const int i = index % width - imax;
    const int j = index / width - imax;
    const int ii = imax + i;
    const int jj = imax + j;
    const int target = ii + nfft * jj;
    const int plane = nfft * nfft;
    CUCOMPLEX ex = CUFFT_ZERO;
    CUCOMPLEX ey = CUFFT_ZERO;
    CUCOMPLEX ez = CUFFT_ZERO;

    if (mode <= 1) {
        for (int wave = 0; wave < plane_count; ++wave) {
            const CUCOMPLEX* parameters = plane_parameters + 5 * wave;
            if (int(parameters[0].x) == i && int(parameters[0].y) == j) {
                ex = CUFFT_ADD(ex, parameters[1]);
                ey = CUFFT_ADD(ey, parameters[2]);
                ez = CUFFT_ADD(ez, parameters[3]);
            }
        }
    } else {
        const CU_REAL kx = CU_REAL(i) * deltakx;
        const CU_REAL ky = CU_REAL(j) * deltaky;
        const CU_REAL kp2 = kx * kx + ky * ky;
        const CU_REAL cutoff = kmax_squared * CU_REAL(0.999);
        const CU_REAL kz2 = k0 * k0 - kp2;
        if (kp2 <= cutoff && kz2 > CU_REAL(0)) {
            const CU_REAL kz = sqrt(kz2);
            const CU_REAL denom_x = sqrt(kz * kz + kx * kx);
            const CU_REAL denom_y = sqrt(kz * kz + ky * ky);
            if (denom_x > CU_REAL(0) && denom_y > CU_REAL(0)) {
                const CU_REAL axp = kz / denom_x;
                const CU_REAL azp = -kx / denom_x;
                const CU_REAL ays = kz / denom_y;
                const CU_REAL azs = -ky / denom_y;
                const CU_REAL cpsi = cos(psi);
                const CU_REAL spsi = sin(psi);
                CUCOMPLEX phase = speckle_exp_i(-(kx * x0 + ky * y0 + kz * z0));
                if (mode == 2 && random_phases)
                    phase = CUFFT_MUL(phase, random_phases[index]);
                ex = CUFFT_MUL(e0, CUFFT_REAL(cpsi * axp));
                ey = CUFFT_MUL(e0, CUFFT_REAL(spsi * ays));
                ez = CUFFT_MUL(e0, CUFFT_REAL(cpsi * azp + spsi * azs));
                ex = CUFFT_MUL(ex, phase);
                ey = CUFFT_MUL(ey, phase);
                ez = CUFFT_MUL(ez, phase);
            }
        }
    }

    spectrum[target] = ex;
    spectrum[plane + target] = ey;
    spectrum[2 * plane + target] = ez;
}

//-----------------------------------------------------------
extern "C" cudaError_t diffraction_incident_spectrum_wrapper(
    CUCOMPLEX* spectrum, const CUCOMPLEX* plane_parameters,
    const CUCOMPLEX* random_phases, int nfft, int imax, int mode,
    int plane_count, CU_REAL k0, CU_REAL deltakx, CU_REAL deltaky,
    CU_REAL kmax_squared, CU_REAL psi, CU_REAL x0, CU_REAL y0,
    CU_REAL z0, CUCOMPLEX e0) {
    if (!spectrum || !plane_parameters || nfft <= 0 || imax < 0 ||
        2 * imax + 1 > nfft || mode < 0 || mode > 3 || plane_count < 0 ||
        plane_count > 10 ||
        k0 <= CU_REAL(0) || deltakx <= CU_REAL(0) || deltaky <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t plane_size = size_t(nfft) * size_t(nfft);
    cudaError_t status = cudaMemset(spectrum, 0, 3 * plane_size * sizeof(CUCOMPLEX));
    if (status != cudaSuccess)
        return status;
    const int width = 2 * imax + 1;
    const size_t count = size_t(width) * size_t(width);
    diffraction_incident_spectrum_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        spectrum, plane_parameters, random_phases, nfft, imax, mode, plane_count,
        k0, deltakx, deltaky, kmax_squared, psi, x0, y0, z0, e0);
    return finish_kernel_launch(224, "diffraction_incident_spectrum_kernel");
}

//-----------------------------------------------------------
__global__ void diffraction_incident_fft_coordinates_kernel(
    CU_REAL* xs, CU_REAL* ys, CU_REAL* zs, int nfft, CU_REAL deltax) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nfft * nfft;
    if (index >= count)
        return;
    const int column = index % nfft;
    const int row = index / nfft;
    const int half = nfft / 2;
    const int i = column < half ? column : column - nfft;
    const int j = row < half ? row : row - nfft;
    xs[index] = CU_REAL(i) * deltax;
    ys[index] = CU_REAL(j) * deltax;
    zs[index] = CU_REAL(0);
}

//-----------------------------------------------------------
extern "C" cudaError_t diffraction_incident_fft_coordinates_wrapper(
    CU_REAL* xs, CU_REAL* ys, CU_REAL* zs, int nfft, CU_REAL deltax) {
    if (!xs || !ys || !zs || nfft <= 0 || nfft > INT_MAX / nfft ||
        deltax <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t count = size_t(nfft) * size_t(nfft);
    diffraction_incident_fft_coordinates_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        xs, ys, zs, nfft, deltax);
    return finish_kernel_launch(225, "diffraction_incident_fft_coordinates_kernel");
}

//-----------------------------------------------------------
__global__ void diffraction_incident_scale_kernel(CUCOMPLEX* field, int count, CU_REAL scale) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
        field[index] = CUFFT_MUL(field[index], CUFFT_REAL(scale));
}

//-----------------------------------------------------------
extern "C" cudaError_t diffraction_incident_scale_wrapper(
    CUCOMPLEX* field, int count, CU_REAL scale) {
    if (!field || count <= 0)
        return cudaErrorInvalidValue;
    diffraction_incident_scale_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        field, count, scale);
    return finish_kernel_launch(226, "diffraction_incident_scale_kernel");
}

//-----------------------------------------------------------
__global__ void diffraction_incident_crop_kernel(
    const CUCOMPLEX* fft_spectrum, CUCOMPLEX* spectrum, int nfft, int imax,
    CU_REAL deltakx, CU_REAL deltaky, CU_REAL numerical_aperture) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int width = 2 * imax + 1;
    const int count = width * width;
    if (index >= count)
        return;
    const int i = index % width - imax;
    const int j = index / width - imax;
    const CU_REAL kx = CU_REAL(i) * deltakx;
    const CU_REAL ky = CU_REAL(j) * deltaky;
    if (kx * kx + ky * ky >= numerical_aperture * numerical_aperture)
        return;
    const int ix = i >= 0 ? i : nfft + i;
    const int iy = j >= 0 ? j : nfft + j;
    const int source = ix + nfft * iy;
    const int target = (imax + i) + nfft * (imax + j);
    const int plane = nfft * nfft;
    for (int component = 0; component < 3; ++component)
        spectrum[component * plane + target] = fft_spectrum[component * plane + source];
}

//-----------------------------------------------------------
extern "C" cudaError_t diffraction_incident_crop_wrapper(
    const CUCOMPLEX* fft_spectrum, CUCOMPLEX* spectrum, int nfft,
    int imax, CU_REAL deltakx, CU_REAL deltaky, CU_REAL numerical_aperture) {
    if (!fft_spectrum || !spectrum || nfft <= 0 || nfft > INT_MAX / nfft ||
        imax < 0 || 2 * imax + 1 > nfft || deltakx <= CU_REAL(0) ||
        deltaky <= CU_REAL(0) || numerical_aperture <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft) * size_t(nfft);
    cudaError_t status = cudaMemset(spectrum, 0, 3 * plane * sizeof(CUCOMPLEX));
    if (status != cudaSuccess)
        return status;
    const int width = 2 * imax + 1;
    const size_t count = size_t(width) * size_t(width);
    diffraction_incident_crop_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        fft_spectrum, spectrum, nfft, imax, deltakx, deltaky, numerical_aperture);
    return finish_kernel_launch(227, "diffraction_incident_crop_kernel");
}

//-----------------------------------------------------------
__global__ void diffraction_incident_interleaved_to_planes_kernel(
    const CUCOMPLEX *interleaved, CUCOMPLEX *planes, int count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        planes[index] = interleaved[3 * index];
        planes[count + index] = interleaved[3 * index + 1];
        planes[2 * count + index] = interleaved[3 * index + 2];
    }
}

//-----------------------------------------------------------
extern "C" cudaError_t diffraction_incident_interleaved_to_planes_wrapper(
    const CUCOMPLEX *interleaved, CUCOMPLEX *planes, int count) {
    if (!interleaved || !planes || count <= 0)
        return cudaErrorInvalidValue;
    diffraction_incident_interleaved_to_planes_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        interleaved, planes, count);
    return finish_kernel_launch(228, "diffraction_incident_interleaved_to_planes_kernel");
}

//-----------------------------------------------------------
__global__ void speckle_flux_normalization_kernel(CU_REAL *output, int nfft, CU_REAL k0,
                                                  CU_REAL deltak, CU_REAL kmax, CU_REAL psi,
                                                  CUCOMPLEX e0, CU_REAL power,
                                                  CU_REAL flux_multiplier) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    CU_REAL flux = CU_REAL(0);
    const int half = nfft / 2;
    const CU_REAL cpsi = cos(psi), spsi = sin(psi);
    for (int nkx = -half; nkx < half; ++nkx) {
        const CU_REAL kx = deltak * CU_REAL(nkx);
        for (int nky = -half; nky < half; ++nky) {
            const CU_REAL ky = deltak * CU_REAL(nky);
            const CU_REAL kp2 = kx * kx + ky * ky;
            if (kp2 > kmax * kmax * CU_REAL(0.999))
                continue;
            const CU_REAL kz = sqrt(k0 * k0 - kp2);
            const CU_REAL axp = kz / sqrt(kz * kz + kx * kx);
            const CU_REAL azp = -kx / sqrt(kz * kz + kx * kx);
            const CU_REAL ays = kz / sqrt(kz * kz + ky * ky);
            const CU_REAL azs = -ky / sqrt(kz * kz + ky * ky);
            const CU_REAL ex = cpsi * axp;
            const CU_REAL ey = spsi * ays;
            const CU_REAL ez = cpsi * azp + spsi * azs;
            const CU_REAL e02 = e0.x * e0.x + e0.y * e0.y;
            flux += e02 * (ex * ex + ey * ey + ez * ez) * kz;
        }
    }
    const CU_REAL irra = deltak * deltak /
                         (CU_REAL(8) * CU_REAL(3.1415926535897932384626433832795) *
                          CU_REAL(1.0e-7) * CU_REAL(299792458.0)) /
                         k0 * CU_REAL(4) * CU_REAL(3.1415926535897932384626433832795) *
                         CU_REAL(3.1415926535897932384626433832795);
    flux *= flux_multiplier * irra;
    output[0] = sqrt(power / flux);
}

//-----------------------------------------------------------
__global__ void speckle_extract_plane_kernel(const CUCOMPLEX *spectrum, CUCOMPLEX *field, int nx,
                                             int ny, int nfft, int z_index, CU_REAL factor,
                                             int add_to_field) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny;
    if (index >= count)
        return;
    const int half = nfft / 2;
    const int ix = index % nx;
    const int iy = index / nx;
    const int fx = (ix - half + nfft) % nfft;
    const int fy = (iy - half + nfft) % nfft;
    const int source = fx + nfft * fy;
    const int target = 3 * (index + count * z_index);
    const int plane = nfft * nfft;
    for (int component = 0; component < 3; ++component) {
        CUCOMPLEX value = CUFFT_MUL(spectrum[component * plane + source], CUFFT_REAL(factor));
        if (add_to_field)
            value = CUFFT_ADD(field[target + component], value);
        field[target + component] = value;
    }
}

//-----------------------------------------------------------
__global__ void speckle_scale_field_kernel(CUCOMPLEX *field, int count,
                                           const CU_REAL *normalization) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        field[i] = CUFFT_MUL(field[i], CUFFT_REAL(normalization[0]));
}

//-----------------------------------------------------------
extern "C" cudaError_t speckle_spectrum_wrapper(const CUCOMPLEX *random_phase, CUCOMPLEX *spectrum,
                                                int nfft, CU_REAL k0, CU_REAL deltak, CU_REAL kmax,
                                                CU_REAL psi, COMPLEX e0, CU_REAL x0, CU_REAL y0,
                                                CU_REAL z0, CU_REAL z_plane, int direction) {
    if (!spectrum || nfft <= 0 || k0 <= CU_REAL(0) || deltak <= CU_REAL(0) || kmax <= CU_REAL(0) ||
        (direction != -1 && direction != 1))
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft) * size_t(nfft);
    speckle_spectrum_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        random_phase, spectrum, nfft, k0, deltak, kmax, psi,
        plane_wave_device_scalar(e0), x0, y0, z0, z_plane, direction);
    return finish_kernel_launch(208, "speckle_spectrum_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t speckle_uniforms_to_phase_wrapper(
    const CU_REAL* uniforms, CUCOMPLEX* phases, size_t count, CU_REAL two_pi)
{
    if (!uniforms || !phases || count == 0 || two_pi <= CU_REAL(0))
        return cudaErrorInvalidValue;
    speckle_uniforms_to_phase_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        uniforms, phases, count, two_pi);
    return finish_kernel_launch(241, "speckle_uniforms_to_phase_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t speckle_flux_normalization_wrapper(CU_REAL *normalization, int nfft,
                                                          CU_REAL k0, CU_REAL deltak, CU_REAL kmax,
                                                          CU_REAL psi, COMPLEX e0, CU_REAL power,
                                                          CU_REAL flux_multiplier) {
    if (!normalization || nfft <= 0 || k0 <= CU_REAL(0) || deltak <= CU_REAL(0) ||
        kmax <= CU_REAL(0) || power <= CU_REAL(0) || flux_multiplier <= CU_REAL(0))
        return cudaErrorInvalidValue;
    speckle_flux_normalization_kernel<<<1, 1>>>(normalization, nfft, k0, deltak, kmax, psi,
                                                plane_wave_device_scalar(e0),
                                                power, flux_multiplier);
    return finish_kernel_launch(209, "speckle_flux_normalization_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t speckle_extract_plane_wrapper(const CUCOMPLEX *spectrum, CUCOMPLEX *field,
                                                     int nx, int ny, int nfft, int z_index,
                                                     CU_REAL factor, int add_to_field) {
    if (!spectrum || !field || nx <= 0 || ny <= 0 || nfft <= 0 || z_index < 0 ||
        (add_to_field != 0 && add_to_field != 1))
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx) * size_t(ny);
    speckle_extract_plane_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        spectrum, field, nx, ny, nfft, z_index, factor, add_to_field);
    return finish_kernel_launch(210, "speckle_extract_plane_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t speckle_scale_field_wrapper(CUCOMPLEX *field, int count,
                                                   const CU_REAL *normalization) {
    if (!field || !normalization || count <= 0)
        return cudaErrorInvalidValue;
    speckle_scale_field_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(field, count,
                                                                                  normalization);
    return finish_kernel_launch(211, "speckle_scale_field_kernel");
}

//------------------------------------------

extern "C" cudaError_t polarization_inplace_wrapper(const CUCOMPLEX *polarizability,
                                                    CUCOMPLEX *field, int niso, int nmax,
                                                    int ndipoles) {
    if (!polarizability || !field || (niso != 1 && niso != 3) || nmax <= 0 || ndipoles <= 0 ||
        ndipoles > nmax)
        return cudaErrorInvalidValue;
    polarization_inplace_kernel<<<block_count(size_t(ndipoles)), THREADS_PER_BLOCK>>>(
        polarizability, field, niso, nmax, ndipoles);
    return finish_kernel_launch(50, "polarization_inplace_kernel");
}

//------------------------------------------

extern "C" cudaError_t local_macro_to_local_wrapper(const CUCOMPLEX *epsilon, CUCOMPLEX *field,
                                                    int niso, int nmax, int ndipoles,
                                                    CU_REAL aretecube, CU_REAL k0) {
    if (!epsilon || !field || (niso != 1 && niso != 3) || nmax <= 0 || ndipoles <= 0 ||
        ndipoles > nmax || aretecube <= CU_REAL(0) || k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    local_macro_to_local_kernel<<<block_count(size_t(ndipoles)), THREADS_PER_BLOCK>>>(
        epsilon, field, niso, nmax, ndipoles, aretecube, k0);
    return finish_kernel_launch(51, "local_macro_to_local_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t local_macro_batch_wrapper(const CUCOMPLEX *epsilon,
                                                 const CUCOMPLEX *input_field,
                                                 CUCOMPLEX *output_field,
                                                 CU_REAL *field_magnitude, CU_INT*status,
                                                 int niso, int nmax, int ndipoles,
                                                 CU_REAL aretecube, CU_REAL k0, int nsens) {
    if (!epsilon || !input_field || !output_field || !field_magnitude || !status ||
        (niso != 1 && niso != 3) || nmax <= 0 || ndipoles <= 0 || ndipoles > nmax ||
        aretecube <= CU_REAL(0) || k0 <= CU_REAL(0) || (nsens != 1 && nsens != -1))
        return cudaErrorInvalidValue;
    local_macro_batch_kernel<<<block_count(size_t(ndipoles)), THREADS_PER_BLOCK>>>(
        epsilon, input_field, output_field, field_magnitude, status, niso, nmax, ndipoles,
        aretecube, k0, nsens);
    return finish_kernel_launch(222, "local_macro_batch_kernel");
}

//------------------------------------------

extern "C" cudaError_t complex_vector_scale_wrapper(const CUCOMPLEX *input, CUCOMPLEX *output,
                                                    CUCOMPLEX scale, int count) {
    if (!input || !output || count <= 0)
        return cudaErrorInvalidValue;
    complex_vector_scale_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(input, output,
                                                                                   scale, count);
    return finish_kernel_launch(52, "complex_vector_scale_kernel");
}

//------------------------------------------

extern "C" cudaError_t rytov_macro_delta_wrapper(const CUCOMPLEX *incident,
                                                 const CUCOMPLEX *epsilon, CUCOMPLEX *delta,
                                                 CUCOMPLEX background_epsilon, int niso, int nmax,
                                                 int ndipoles) {
    if (!incident || !epsilon || !delta || (niso != 1 && niso != 3) || nmax <= 0 || ndipoles <= 0 ||
        ndipoles > nmax)
        return cudaErrorInvalidValue;
    rytov_macro_delta_kernel<<<block_count(size_t(ndipoles)), THREADS_PER_BLOCK>>>(
        incident, epsilon, delta, background_epsilon, niso, nmax, ndipoles);
    return finish_kernel_launch(53, "rytov_macro_delta_kernel");
}

//-----------------------------------------------------------
__global__ void rytov_reference_local_kernel(const CUCOMPLEX *reference_field,
                                             const CUCOMPLEX *delta_field,
                                             CUCOMPLEX *local_field, int ndipoles) {
    const int dipole = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (dipole >= ndipoles)
        return;
    const int base = 3 * dipole;
    const CUCOMPLEX ex = reference_field[base];
    const CUCOMPLEX ey = reference_field[base + 1];
    const CUCOMPLEX ez = reference_field[base + 2];
    const CU_REAL norm = sqrt((ex.x * ex.x + ex.y * ex.y) +
                              (ey.x * ey.x + ey.y * ey.y) +
                              (ez.x * ez.x + ez.y * ez.y));
    const CU_REAL threshold = norm * CU_REAL(1.0e-6);
    for (int component = 0; component < 3; ++component) {
        const CUCOMPLEX reference = reference_field[base + component];
        const CU_REAL magnitude = sqrt(reference.x * reference.x + reference.y * reference.y);
        if (magnitude <= threshold) {
            local_field[base + component] = CUFFT_ZERO;
            continue;
        }
        const CUCOMPLEX ratio = CUFFT_DIV(delta_field[base + component], reference);
        const CU_REAL exp_real = exp(ratio.x);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CUCOMPLEX exponential = make_cuFloatComplex(exp_real * cosf(ratio.y),
                                                          exp_real * sinf(ratio.y));
#else
        const CUCOMPLEX exponential = make_cuDoubleComplex(exp_real * cos(ratio.y),
                                                           exp_real * sin(ratio.y));
#endif
        local_field[base + component] = CUFFT_MUL(reference, exponential);
    }
}

//-------------------------------------------
extern "C" cudaError_t rytov_reference_local_wrapper(
    const CUCOMPLEX *reference_field, const CUCOMPLEX *delta_field,
    CUCOMPLEX *local_field, int ndipoles) {
    if (!reference_field || !delta_field || !local_field || ndipoles <= 0)
        return cudaErrorInvalidValue;
    rytov_reference_local_kernel<<<block_count(size_t(ndipoles)), THREADS_PER_BLOCK>>>(
        reference_field, delta_field, local_field, ndipoles);
    return finish_kernel_launch(253, "rytov_reference_local_kernel");
}

//------------------------------------------

extern "C" cudaError_t gather_incident_wrapper(const CUCOMPLEX *incident, const CU_INT *dipole_map,
                                               CUCOMPLEX *vectors, int nx, int ny, int nz, int nx2,
                                               int nxy2, int ntotal, int nmax) {
    if (!incident || !dipole_map || !vectors || nx <= 0 || ny <= 0 || nz <= 0 || nx2 <= 0 ||
        nxy2 <= 0 || ntotal <= 0 || nmax <= 0)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    if (count > size_t(INT_MAX))
        return cudaErrorInvalidValue;
    gather_incident_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        incident, dipole_map, vectors, nx, ny, nz, nx2, nxy2, ntotal, nmax);
    return finish_kernel_launch(1, "gather_incident_kernel");
}

//------------------------------------------

extern "C" cudaError_t multiply_green_tensor_wrapper(GreenTensorDeviceView green, CUCOMPLEX *vectors,
                                                     int nx, int ny, int nz, int nx2, int ny2,
                                                     int nz2, size_t tensor_count) {
    if (!green || !vectors || nx <= 0 || ny <= 0 || nz <= 0 || nx2 <= 0 || ny2 <= 0 || nz2 <= 0 ||
        tensor_count == 0)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx2) * size_t(ny2) * size_t(nz2);
    multiply_green_tensor_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        green, vectors, nx, ny, nz, nx2, ny2, nz2, tensor_count);
    return finish_kernel_launch(2, "multiply_green_tensor_kernel");
}

//------------------------------------------

extern "C" cudaError_t update_output_wrapper(const CUCOMPLEX *vectors, const CU_INT *dipole_map,
                                             CUCOMPLEX *output, int nx, int ny, int nz, int nx2,
                                             int nxy2, int nmax, int ntotal) {
    if (!vectors || !dipole_map || !output || nx <= 0 || ny <= 0 || nz <= 0 || nx2 <= 0 ||
        nxy2 <= 0 || nmax <= 0 || ntotal <= 0)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    if (count > size_t(INT_MAX))
        return cudaErrorInvalidValue;
    update_output_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        vectors, dipole_map, output, nx, ny, nz, nx2, nxy2, nmax, ntotal);
    return finish_kernel_launch(3, "update_output_kernel");
}

//------------------------------------------

extern "C" cudaError_t gather_incident_regular_wrapper(const CUCOMPLEX *incident,
                                                       CUCOMPLEX *vectors, int nx, int ny, int nz,
                                                       int nx2, int nxy2, int ntotal, int nmax) {
    if (!incident || !vectors || nx <= 0 || ny <= 0 || nz <= 0 || nx2 <= 0 || nxy2 <= 0 ||
        ntotal <= 0 || nmax <= 0)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    if (count > size_t(INT_MAX))
        return cudaErrorInvalidValue;
    gather_incident_regular_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        incident, vectors, nx, ny, nz, nx2, nxy2, ntotal, nmax);
    return finish_kernel_launch(4, "gather_incident_regular_kernel");
}

//------------------------------------------

extern "C" cudaError_t multiply_green_tensor_opt_wrapper(GreenTensorDeviceView green, CUCOMPLEX *vectors,
                                                         int nx, int ny, int nz, int nx2, int ny2,
                                                         int nz2, size_t tensor_count) {
    if (!green || !vectors || nx <= 0 || ny <= 0 || nz <= 0 || nx2 <= 0 || ny2 <= 0 || nz2 <= 0 ||
        tensor_count == 0)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx2) * size_t(ny2) * size_t(nz2);
    multiply_green_tensor_opt_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        green, vectors, nx, ny, nz, nx2, ny2, nz2, tensor_count);
    return finish_kernel_launch(5, "multiply_green_tensor_opt_kernel");
}

//------------------------------------------

extern "C" cudaError_t update_output_regular_wrapper(const CUCOMPLEX *vectors, CUCOMPLEX *output,
                                                     int nx, int ny, int nz, int nx2, int nxy2,
                                                     int nmax, int ntotal) {
    if (!vectors || !output || nx <= 0 || ny <= 0 || nz <= 0 || nx2 <= 0 || nxy2 <= 0 ||
        nmax <= 0 || ntotal <= 0)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    if (count > size_t(INT_MAX))
        return cudaErrorInvalidValue;
    update_output_regular_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        vectors, output, nx, ny, nz, nx2, nxy2, nmax, ntotal);
    return finish_kernel_launch(6, "update_output_regular_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gather_scalar_product_wrapper(const CUCOMPLEX *input, const CU_INT *dipole_map,
                                                     CUCOMPLEX *work, int nx, int ny, int nz,
                                                     int nx2, int nxy2, int nmax) {
    if (!input || !dipole_map || !work || nx <= 0 || ny <= 0 || nz <= 0 || nx2 < nx ||
        nxy2 < nx2 * ny || nmax <= 0)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    if (count > size_t(INT_MAX))
        return cudaErrorInvalidValue;
    gather_scalar_product_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        input, dipole_map, work, nx, ny, nz, nx2, nxy2, nmax);
    return finish_kernel_launch(7, "gather_scalar_product_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multiply_scalar_green_wrapper(const CUCOMPLEX *green, CUCOMPLEX *work,
                                                     int count) {
    if (!green || !work || count <= 0)
        return cudaErrorInvalidValue;
    multiply_scalar_green_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(green, work,
                                                                                    count);
    return finish_kernel_launch(8, "multiply_scalar_green_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scatter_scalar_product_wrapper(const CUCOMPLEX *work, const CU_INT *dipole_map,
                                                      CUCOMPLEX *output, int nx, int ny, int nz,
                                                      int nx2, int nxy2, int nmax, int ntotal) {
    if (!work || !dipole_map || !output || nx <= 0 || ny <= 0 || nz <= 0 || nx2 < nx ||
        nxy2 < nx2 * ny || nmax <= 0 || ntotal <= 0)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    if (count > size_t(INT_MAX))
        return cudaErrorInvalidValue;
    scatter_scalar_product_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        work, dipole_map, output, nx, ny, nz, nx2, nxy2, nmax, ntotal);
    return finish_kernel_launch(9, "scatter_scalar_product_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gather_scalar_product_dense_wrapper(
    const CUCOMPLEX *input, CUCOMPLEX *work, int nx, int ny, int nz, int nx2, int nxy2) {
    if (!input || !work || nx <= 0 || ny <= 0 || nz <= 0 || nx2 < nx ||
        size_t(nxy2) < size_t(nx2) * size_t(ny))
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    if (count > size_t(INT_MAX))
        return cudaErrorInvalidValue;
    gather_scalar_product_dense_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        input, work, nx, ny, nz, nx2, nxy2);
    return finish_kernel_launch(10, "gather_scalar_product_dense_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scatter_scalar_product_dense_wrapper(
    const CUCOMPLEX *work, CUCOMPLEX *output, int nx, int ny, int nz, int nx2, int nxy2,
    int ntotal) {
    if (!work || !output || nx <= 0 || ny <= 0 || nz <= 0 || nx2 < nx || ntotal <= 0 ||
        size_t(nxy2) < size_t(nx2) * size_t(ny))
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    if (count > size_t(INT_MAX))
        return cudaErrorInvalidValue;
    scatter_scalar_product_dense_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        work, output, nx, ny, nz, nx2, nxy2, ntotal);
    return finish_kernel_launch(11, "scatter_scalar_product_dense_kernel");
}

//------------------------------------------

namespace {
//-----------------------------------------------------------
cudaError_t finish_gpbicg_launch(unsigned int slot, const char *name) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
        return status;
    static std::atomic<unsigned long long> reported[4] = {};
    if (slot >= 256)
        return cudaErrorInvalidValue;
    const unsigned long long bit = 1ULL << (slot % 64);
    if ((reported[slot / 64].fetch_or(bit) & bit) == 0) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const char *precision = "single";
#else
        const char *precision = "double";
#endif
        std::fprintf(stderr, "[kernel.cu] GPU kernel executed: %s (%s precision)\n", name,
                     precision);
        std::fflush(stderr);
    }
    return cudaSuccess;
}
} // namespace

//-----------------------------------------------------------
extern "C" cudaError_t gpbicg_save_initial_wrapper(const CUCOMPLEX *xi, CUCOMPLEX *work, int lda,
                                                   int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicg_save_initial_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda, n);
    return finish_gpbicg_launch(0, "gpbicg_save_initial_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicg_initialize_wrapper(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                                 CUCOMPLEX *work, int lda, int n) {
    if (!xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicg_initialize_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, b, work, lda, n);
    return finish_gpbicg_launch(1, "gpbicg_initialize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicg2_initialize_wrapper(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                                  CUCOMPLEX *work, int lda, int n) {
    if (!xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicg2_initialize_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, b, work, lda, n);
    return finish_gpbicg_launch(2, "gpbicg2_initialize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicg_prepare_direction_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work,
                                                        CUCOMPLEX beta, int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicg_prepare_direction_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, beta,
                                                                                   lda, n);
    return finish_gpbicg_launch(3, "gpbicg_prepare_direction_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicg_save_xr_column_wrapper(const CUCOMPLEX *xr, CUCOMPLEX *work, int lda,
                                                     int n, int column) {
    if (!xr || !work || lda < n || n <= 0 || column < 0 || column >= 12)
        return cudaErrorInvalidValue;
    gpbicg_save_xr_column_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, work, lda, n,
                                                                                column);
    return finish_gpbicg_launch(4, "gpbicg_save_xr_column_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicg_update_stage2_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX alpha,
                                                    int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicg_update_stage2_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, alpha, lda,
                                                                               n);
    return finish_gpbicg_launch(5, "gpbicg_update_stage2_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicg_update_stage3_wrapper(CUCOMPLEX *work, CUCOMPLEX alpha,
                                                    CUCOMPLEX beta, CUCOMPLEX eta, CUCOMPLEX dzeta,
                                                    int lda, int n) {
    if (!work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicg_update_stage3_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, alpha, beta,
                                                                               eta, dzeta, lda, n);
    return finish_gpbicg_launch(6, "gpbicg_update_stage3_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicg_update_stage4_wrapper(CUCOMPLEX *work, CUCOMPLEX beta, int lda,
                                                    int n) {
    if (!work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicg_update_stage4_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, beta, lda, n);
    return finish_gpbicg_launch(7, "gpbicg_update_stage4_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicg_publish_solution_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *work,
                                                       int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicg_publish_solution_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda, n);
    return finish_gpbicg_launch(8, "gpbicg_publish_solution_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgplus_initialize_wrapper(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                                     CUCOMPLEX *work, int lda, int n) {
    if (!xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgplus_initialize_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, b, work, lda,
                                                                                n);
    return finish_gpbicg_launch(9, "gpbicgplus_initialize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgplus_prepare_direction_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work,
                                                            CUCOMPLEX beta, int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgplus_prepare_direction_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        xi, work, beta, lda, n);
    return finish_gpbicg_launch(10, "gpbicgplus_prepare_direction_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgplus_update_ap_wrapper(const CUCOMPLEX *xr, CUCOMPLEX *xi,
                                                    CUCOMPLEX *work, CUCOMPLEX beta, int lda,
                                                    int n) {
    if (!xr || !xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgplus_update_ap_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, xi, work, beta,
                                                                               lda, n);
    return finish_gpbicg_launch(11, "gpbicgplus_update_ap_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgplus_update_stage2_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work,
                                                        CUCOMPLEX alpha, CUCOMPLEX beta,
                                                        CUCOMPLEX eta, CUCOMPLEX dzeta, int lda,
                                                        int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgplus_update_stage2_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        xi, work, alpha, beta, eta, dzeta, lda, n);
    return finish_gpbicg_launch(12, "gpbicgplus_update_stage2_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgplus_update_stage3_wrapper(CUCOMPLEX *work, const CUCOMPLEX *xr,
                                                        CUCOMPLEX alpha, CUCOMPLEX eta,
                                                        CUCOMPLEX dzeta, int lda, int n) {
    if (!work || !xr || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgplus_update_stage3_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        work, xr, alpha, eta, dzeta, lda, n);
    return finish_gpbicg_launch(13, "gpbicgplus_update_stage3_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgsafe_initialize_wrapper(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                                     CUCOMPLEX *work, int lda, int n) {
    if (!xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgsafe_initialize_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, b, work, lda,
                                                                                n);
    return finish_gpbicg_launch(14, "gpbicgsafe_initialize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgsafe_prepare_direction_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work,
                                                            CUCOMPLEX beta, int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgsafe_prepare_direction_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        xi, work, beta, lda, n);
    return finish_gpbicg_launch(15, "gpbicgsafe_prepare_direction_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgsafe_update_ap_wrapper(const CUCOMPLEX *xr, CUCOMPLEX *xi,
                                                    CUCOMPLEX *work, CUCOMPLEX beta, int lda,
                                                    int n) {
    if (!xr || !xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgsafe_update_ap_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, xi, work, beta,
                                                                               lda, n);
    return finish_gpbicg_launch(16, "gpbicgsafe_update_ap_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgsafe_update_stage2_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work,
                                                        CUCOMPLEX beta, CUCOMPLEX eta,
                                                        CUCOMPLEX dzeta, int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgsafe_update_stage2_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        xi, work, beta, eta, dzeta, lda, n);
    return finish_gpbicg_launch(17, "gpbicgsafe_update_stage2_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgsafe_update_stage3_wrapper(CUCOMPLEX *work, const CUCOMPLEX *xr,
                                                        CUCOMPLEX alpha, CUCOMPLEX eta,
                                                        CUCOMPLEX dzeta, int lda, int n) {
    if (!work || !xr || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgsafe_update_stage3_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        work, xr, alpha, eta, dzeta, lda, n);
    return finish_gpbicg_launch(18, "gpbicgsafe_update_stage3_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgar_initialize_wrapper(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                                   CUCOMPLEX *xi, CUCOMPLEX *work, int lda, int n) {
    if (!xr || !b || !xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgar_initialize_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, b, xi, work, lda,
                                                                              n);
    return finish_gpbicg_launch(19, "gpbicgar_initialize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgar_update_ap_wrapper(const CUCOMPLEX *xr, CUCOMPLEX *work,
                                                  CUCOMPLEX beta, int lda, int n) {
    if (!xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgar_update_ap_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, work, beta, lda,
                                                                             n);
    return finish_gpbicg_launch(20, "gpbicgar_update_ap_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgar_update_u_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX beta,
                                                 CUCOMPLEX eta, CUCOMPLEX dzeta, int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgar_update_u_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, beta, eta,
                                                                            dzeta, lda, n);
    return finish_gpbicg_launch(21, "gpbicgar_update_u_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgar_update_stage_wrapper(CUCOMPLEX *work, const CUCOMPLEX *xr,
                                                     CUCOMPLEX alpha, CUCOMPLEX eta,
                                                     CUCOMPLEX dzeta, int lda, int n) {
    if (!work || !xr || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgar_update_stage_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, xr, alpha,
                                                                                eta, dzeta, lda, n);
    return finish_gpbicg_launch(22, "gpbicgar_update_stage_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgar_publish_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *work,
                                                         int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgar_publish_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda,
                                                                                    n);
    return finish_gpbicg_launch(23, "gpbicgar_publish_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgar_copy_ar_wrapper(const CUCOMPLEX *xr, CUCOMPLEX *work, int lda,
                                                int n) {
    if (!xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicgar_copy_ar_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, work, lda, n);
    return finish_gpbicg_launch(24, "gpbicgar_copy_ar_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicor_initialize_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                  const CUCOMPLEX *b, CUCOMPLEX *work, int lda,
                                                  int n) {
    if (!xi || !xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicor_initialize_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, b, work, lda,
                                                                             n);
    return finish_gpbicg_launch(25, "gpbicor_initialize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicor_first_direction_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work,
                                                       const CUCOMPLEX *xr, int lda, int n) {
    if (!xi || !work || !xr || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicor_first_direction_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, xr, lda,
                                                                                  n);
    return finish_gpbicg_launch(26, "gpbicor_first_direction_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicor_update_directions_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                         CUCOMPLEX *work, CUCOMPLEX beta, int lda,
                                                         int n) {
    if (!xi || !xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicor_update_directions_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, work,
                                                                                    beta, lda, n);
    return finish_gpbicg_launch(27, "gpbicor_update_directions_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicor_prepare_stage_wrapper(CUCOMPLEX *xi, CUCOMPLEX *xr, CUCOMPLEX *work,
                                                     CUCOMPLEX alpha, int lda, int n) {
    if (!xi || !xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicor_prepare_stage_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, work, alpha,
                                                                                lda, n);
    return finish_gpbicg_launch(28, "gpbicor_prepare_stage_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicor_finish_stage_wrapper(CUCOMPLEX *ffloc, CUCOMPLEX *work,
                                                    const CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                    CUCOMPLEX alpha, CUCOMPLEX beta, CUCOMPLEX eta,
                                                    CUCOMPLEX dzeta, int lda, int n) {
    if (!ffloc || !work || !xi || !xr || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicor_finish_stage_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        ffloc, work, xi, xr, alpha, beta, eta, dzeta, lda, n);
    return finish_gpbicg_launch(29, "gpbicor_finish_stage_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicor_publish_solution_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *ffloc,
                                                        int n) {
    if (!xi || !ffloc || n <= 0)
        return cudaErrorInvalidValue;
    gpbicor_publish_solution_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, ffloc, n);
    return finish_gpbicg_launch(30, "gpbicor_publish_solution_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicor_publish_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *work,
                                                        int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gpbicor_publish_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda,
                                                                                   n);
    return finish_gpbicg_launch(31, "gpbicor_publish_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cors_save_initial_wrapper(const CUCOMPLEX *xi, CUCOMPLEX *work, int lda,
                                                 int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    cors_save_initial_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda, n);
    return finish_gpbicg_launch(32, "cors_save_initial_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cors_initialize_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                        const CUCOMPLEX *b, CUCOMPLEX *work,
                                                        int lda, int n) {
    if (!xi || !xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    cors_initialize_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, b, work,
                                                                                   lda, n);
    return finish_gpbicg_launch(33, "cors_initialize_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cors_initialize_direction_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                         CUCOMPLEX *work, int lda, int n) {
    if (!xi || !xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    cors_initialize_direction_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, work,
                                                                                    lda, n);
    return finish_gpbicg_launch(34, "cors_initialize_direction_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cors_update_direction_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                     CUCOMPLEX *work, CUCOMPLEX beta, int lda,
                                                     int n) {
    if (!xi || !xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    cors_update_direction_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, work, beta,
                                                                                lda, n);
    return finish_gpbicg_launch(35, "cors_update_direction_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cors_update_stage_wrapper(CUCOMPLEX *work, const CUCOMPLEX *xr,
                                                 CUCOMPLEX alpha, int lda, int n) {
    if (!work || !xr || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    cors_update_stage_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, xr, alpha, lda,
                                                                            n);
    return finish_gpbicg_launch(36, "cors_update_stage_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cors_publish_solution_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda,
                                                     int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    cors_publish_solution_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda, n);
    return finish_gpbicg_launch(37, "cors_publish_solution_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t cors_publish_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda,
                                                     int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    cors_publish_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda, n);
    return finish_gpbicg_launch(38, "cors_publish_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t tfqmr_initialize_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                         const CUCOMPLEX *b, CUCOMPLEX *work,
                                                         int lda, int n) {
    if (!xi || !xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    tfqmr_initialize_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, b, work,
                                                                                    lda, n);
    return finish_gpbicg_launch(39, "tfqmr_initialize_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t tfqmr_initialize_vectors_wrapper(const CUCOMPLEX *xr, CUCOMPLEX *work,
                                                        int lda, int n) {
    if (!xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    tfqmr_initialize_vectors_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, work, lda,
                                                                                   n);
    return finish_gpbicg_launch(40, "tfqmr_initialize_vectors_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t tfqmr_prepare_uplus_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX alpha,
                                                   int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    tfqmr_prepare_uplus_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, alpha, lda,
                                                                              n);
    return finish_gpbicg_launch(41, "tfqmr_prepare_uplus_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t tfqmr_update_w_wrapper(CUCOMPLEX *work, CUCOMPLEX alpha, int lda, int n) {
    if (!work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    tfqmr_update_w_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, alpha, lda, n);
    return finish_gpbicg_launch(45, "tfqmr_update_w_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t tfqmr_update_halfstep_wrapper(CUCOMPLEX *x, CUCOMPLEX *work, CUCOMPLEX dcoef,
                                                     CUCOMPLEX eta, int lda, int n) {
    if (!x || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    tfqmr_update_halfstep_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(x, work, dcoef, eta,
                                                                                lda, n);
    return finish_gpbicg_launch(42, "tfqmr_update_halfstep_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t tfqmr_continue_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr, CUCOMPLEX *work,
                                              CUCOMPLEX beta, int odd, int lda, int n) {
    if (!xi || !xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    tfqmr_continue_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, work, beta, odd,
                                                                         lda, n);
    return finish_gpbicg_launch(43, "tfqmr_continue_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t tfqmr_end_halfstep_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX beta,
                                                  int odd, int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    tfqmr_end_halfstep_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, beta, odd,
                                                                             lda, n);
    return finish_gpbicg_launch(44, "tfqmr_end_halfstep_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t bicgstarplus_save_solution_wrapper(const CUCOMPLEX *xi, CUCOMPLEX *work,
                                                          int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    bicgstarplus_save_solution_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda,
                                                                                     n);
    return finish_gpbicg_launch(46, "bicgstarplus_save_solution_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t bicgstarplus_initialize_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                                const CUCOMPLEX *b, CUCOMPLEX *work,
                                                                int lda, int n) {
    if (!xi || !xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    bicgstarplus_initialize_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        xi, xr, b, work, lda, n);
    return finish_gpbicg_launch(47, "bicgstarplus_initialize_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t bicgstarplus_update_direction_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work,
                                                             CUCOMPLEX alpha, CUCOMPLEX beta,
                                                             CUCOMPLEX eta, CUCOMPLEX dzeta,
                                                             int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    bicgstarplus_update_direction_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        xi, work, alpha, beta, eta, dzeta, lda, n);
    return finish_gpbicg_launch(48, "bicgstarplus_update_direction_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t bicgstarplus_update_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                            CUCOMPLEX *work, CUCOMPLEX alpha,
                                                            CUCOMPLEX beta, CUCOMPLEX eta,
                                                            CUCOMPLEX dzeta, int lda, int n) {
    if (!xi || !xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    bicgstarplus_update_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        xi, xr, work, alpha, beta, eta, dzeta, lda, n);
    return finish_gpbicg_launch(49, "bicgstarplus_update_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t bicgstarplus_publish_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *work,
                                                             int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    bicgstarplus_publish_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work,
                                                                                        lda, n);
    return finish_gpbicg_launch(50, "bicgstarplus_publish_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t zcg_save_solution_wrapper(const CUCOMPLEX *xi, CUCOMPLEX *work, int lda,
                                                 int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    zcg_save_solution_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda, n);
    return finish_gpbicg_launch(51, "zcg_save_solution_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t zcg_initialize_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                       const CUCOMPLEX *b, CUCOMPLEX *work, int lda,
                                                       int n) {
    if (!xi || !xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    zcg_initialize_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, b, work,
                                                                                  lda, n);
    return finish_gpbicg_launch(52, "zcg_initialize_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t zcg_save_initial_w_wrapper(const CUCOMPLEX *xr, CUCOMPLEX *work, int lda,
                                                  int n) {
    if (!xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    zcg_save_initial_w_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, work, lda, n);
    return finish_gpbicg_launch(53, "zcg_save_initial_w_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t zcg_step_wrapper(CUCOMPLEX *work, CUCOMPLEX alpha, int lda, int n) {
    if (!work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    zcg_step_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, alpha, lda, n);
    return finish_gpbicg_launch(54, "zcg_step_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t zcg_update_directions_wrapper(const CUCOMPLEX *xr, CUCOMPLEX *work,
                                                     CUCOMPLEX beta, int lda, int n) {
    if (!xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    zcg_update_directions_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, work, beta, lda,
                                                                                n);
    return finish_gpbicg_launch(55, "zcg_update_directions_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t zcg_publish_solution_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda,
                                                    int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    zcg_publish_solution_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda, n);
    return finish_gpbicg_launch(56, "zcg_publish_solution_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t zcg_publish_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *work, int lda,
                                                    int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    zcg_publish_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda, n);
    return finish_gpbicg_launch(57, "zcg_publish_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pimzbicgstab_save_rhs_wrapper(const CUCOMPLEX *b, CUCOMPLEX *work, int lda,
                                                     int n) {
    if (!b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    pimzbicgstab_save_rhs_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(b, work, lda, n);
    return finish_gpbicg_launch(58, "pimzbicgstab_save_rhs_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pimzbicgstab_initialize_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                                CUCOMPLEX *work, int lda, int n) {
    if (!xi || !xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    pimzbicgstab_initialize_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        xi, xr, work, lda, n);
    return finish_gpbicg_launch(59, "pimzbicgstab_initialize_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pimzbicgstab_prepare_direction_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work,
                                                              CUCOMPLEX beta, CUCOMPLEX omega,
                                                              int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    pimzbicgstab_prepare_direction_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        xi, work, beta, omega, lda, n);
    return finish_gpbicg_launch(60, "pimzbicgstab_prepare_direction_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pimzbicgstab_prepare_s_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                      CUCOMPLEX *work, CUCOMPLEX alpha, int lda,
                                                      int n) {
    if (!xi || !xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    pimzbicgstab_prepare_s_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, work,
                                                                                 alpha, lda, n);
    return finish_gpbicg_launch(61, "pimzbicgstab_prepare_s_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pimzbicgstab_update_xr_wrapper(CUCOMPLEX *x, CUCOMPLEX *work,
                                                      CUCOMPLEX alpha, CUCOMPLEX omega, int lda,
                                                      int n) {
    if (!x || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    pimzbicgstab_update_xr_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(x, work, alpha,
                                                                                 omega, lda, n);
    return finish_gpbicg_launch(62, "pimzbicgstab_update_xr_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pimzbicgstab_soft_breakdown_wrapper(CUCOMPLEX *x, const CUCOMPLEX *work,
                                                           int lda, int n) {
    if (!x || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    pimzbicgstab_soft_breakdown_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(x, work, lda,
                                                                                      n);
    return finish_gpbicg_launch(63, "pimzbicgstab_soft_breakdown_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pimzbicgstab_publish_vector_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *work,
                                                           int lda, int column, int n) {
    if (!xi || !work || lda < n || n <= 0 || column < 0)
        return cudaErrorInvalidValue;
    pimzbicgstab_publish_vector_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda,
                                                                                      column, n);
    return finish_gpbicg_launch(64, "pimzbicgstab_publish_vector_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t qmrbicgstab_initialize_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                      const CUCOMPLEX *b, CUCOMPLEX *work, int lda,
                                                      int n) {
    if (!xi || !xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    qmrbicgstab_initialize_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, b, work, lda,
                                                                                 n);
    return finish_gpbicg_launch(65, "qmrbicgstab_initialize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t qmrbicgstab_update_p_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work, CUCOMPLEX beta,
                                                    CUCOMPLEX omega, int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    qmrbicgstab_update_p_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, beta,
                                                                               omega, lda, n);
    return finish_gpbicg_launch(66, "qmrbicgstab_update_p_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t qmrbicgstab_prepare_s_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                     CUCOMPLEX *work, CUCOMPLEX alpha, int lda,
                                                     int n) {
    if (!xi || !xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    qmrbicgstab_prepare_s_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, work, alpha,
                                                                                lda, n);
    return finish_gpbicg_launch(67, "qmrbicgstab_prepare_s_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t qmrbicgstab_prepare_x_wrapper(CUCOMPLEX *x, CUCOMPLEX *work, CUCOMPLEX alpha,
                                                     CUCOMPLEX theta, CUCOMPLEX eta, CUCOMPLEX etat,
                                                     int lda, int n) {
    if (!x || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    qmrbicgstab_prepare_x_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        x, work, alpha, theta, eta, etat, lda, n);
    return finish_gpbicg_launch(72, "qmrbicgstab_prepare_x_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t qmrbicgstab_save_t_wrapper(const CUCOMPLEX *xr, CUCOMPLEX *work, int lda,
                                                  int n) {
    if (!xr || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    qmrbicgstab_save_t_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, work, lda, n);
    return finish_gpbicg_launch(68, "qmrbicgstab_save_t_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t qmrbicgstab_update_xr_wrapper(CUCOMPLEX *work, CUCOMPLEX omega, int lda,
                                                     int n) {
    if (!work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    qmrbicgstab_update_r_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, omega, lda, n);
    return finish_gpbicg_launch(69, "qmrbicgstab_update_r_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t qmrbicgstab_finalize_x_wrapper(CUCOMPLEX *x, CUCOMPLEX *work,
                                                      CUCOMPLEX omega, CUCOMPLEX eta,
                                                      CUCOMPLEX thetat, CUCOMPLEX etat, int lda,
                                                      int n) {
    if (!x || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    qmrbicgstab_finalize_x_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        x, work, omega, eta, thetat, etat, lda, n);
    return finish_gpbicg_launch(70, "qmrbicgstab_finalize_x_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t qmrbicgstab_publish_x_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *x, int n) {
    if (!xi || !x || n <= 0)
        return cudaErrorInvalidValue;
    qmrbicgstab_publish_x_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, x, n);
    return finish_gpbicg_launch(71, "qmrbicgstab_publish_x_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_initialize_wrapper(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                               CUCOMPLEX *work, int lda, int n) {
    if (!xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    idrs_initialize_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, b, work, lda, n);
    return finish_gpbicg_launch(72, "idrs_initialize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_copy_column_wrapper(CUCOMPLEX *work, int dst, int src, int lda, int n) {
    if (!work || lda < n || n <= 0 || dst < 0 || src < 0)
        return cudaErrorInvalidValue;
    idrs_copy_column_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, dst, src, lda, n);
    return finish_gpbicg_launch(73, "idrs_copy_column_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_copy_xr_wrapper(CUCOMPLEX *work, const CUCOMPLEX *xr, int dst, int lda,
                                            int n) {
    if (!work || !xr || lda < n || n <= 0 || dst < 0)
        return cudaErrorInvalidValue;
    idrs_copy_xr_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, xr, dst, lda, n);
    return finish_gpbicg_launch(83, "idrs_copy_xr_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_scale_copy_wrapper(CUCOMPLEX *work, int dst, int src, CUCOMPLEX scale,
                                               int lda, int n) {
    if (!work || lda < n || n <= 0 || dst < 0 || src < 0)
        return cudaErrorInvalidValue;
    idrs_scale_copy_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, dst, src, scale,
                                                                          lda, n);
    return finish_gpbicg_launch(74, "idrs_scale_copy_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_axpy_wrapper(CUCOMPLEX *work, int dst, int src, CUCOMPLEX scale,
                                         int lda, int n) {
    if (!work || lda < n || n <= 0 || dst < 0 || src < 0)
        return cudaErrorInvalidValue;
    idrs_axpy_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, dst, src, scale, lda, n);
    return finish_gpbicg_launch(75, "idrs_axpy_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_normalize_basis_wrapper(CUCOMPLEX *work, int col, CU_REAL norm, int lda,
                                                    int n) {
    if (!work || lda < n || n <= 0 || col < 0 || norm <= CU_REAL(0))
        return cudaErrorInvalidValue;
    idrs_normalize_basis_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, col, norm, lda,
                                                                               n);
    return finish_gpbicg_launch(76, "idrs_normalize_basis_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_initial_pair_wrapper(CUCOMPLEX *work, CUCOMPLEX *x, CUCOMPLEX omega,
                                                 int is, int ns, int lda, int n) {
    if (!work || !x || lda < n || n <= 0 || ns <= 0 || is < 0 || is >= ns)
        return cudaErrorInvalidValue;
    idrs_initial_pair_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, x, omega, is, ns,
                                                                            lda, n);
    return finish_gpbicg_launch(77, "idrs_initial_pair_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_apply_pair_wrapper(CUCOMPLEX *work, CUCOMPLEX *x, int lda, int n) {
    if (!work || !x || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    idrs_apply_pair_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, x, lda, n);
    return finish_gpbicg_launch(78, "idrs_apply_pair_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_negate_xr_wrapper(CUCOMPLEX *work, const CUCOMPLEX *xr, int lda,
                                              int n) {
    if (!work || !xr || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    idrs_negate_xr_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, xr, lda, n);
    return finish_gpbicg_launch(79, "idrs_negate_xr_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_scale_xr_wrapper(CUCOMPLEX *work, const CUCOMPLEX *xr, int dst,
                                             CUCOMPLEX scale, int lda, int n) {
    if (!work || !xr || lda < n || n <= 0 || dst < 0)
        return cudaErrorInvalidValue;
    idrs_scale_xr_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, xr, dst, scale, lda,
                                                                        n);
    return finish_gpbicg_launch(82, "idrs_scale_xr_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_shift_qe_wrapper(CUCOMPLEX *work, int ns, int lda, int n) {
    if (!work || lda < n || n <= 0 || ns <= 0)
        return cudaErrorInvalidValue;
    idrs_shift_qe_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(work, ns, lda, n);
    return finish_gpbicg_launch(80, "idrs_shift_qe_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t idrs_copy_work_to_xi_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *work, int col,
                                                    int lda, int n) {
    if (!xi || !work || lda < n || n <= 0 || col < 0)
        return cudaErrorInvalidValue;
    idrs_copy_work_to_xi_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, col, lda,
                                                                               n);
    return finish_gpbicg_launch(81, "idrs_copy_work_to_xi_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gpbicgstabl_combine_wrapper(CUCOMPLEX *work, CUCOMPLEX *xsol, int lda, int n,
                                                   int dst_col, const int*source_cols,
                                                   const CUCOMPLEX *coefficients, int term_count,
                                                   int add_to_destination) {
    if (!work || !xsol || lda < n || n <= 0 || dst_col < -1 || !source_cols || !coefficients ||
        term_count < 1 || term_count > 10)
        return cudaErrorInvalidValue;
    GpbicgstablTerms terms{};
    terms.count = term_count;
    for (int k = 0; k < term_count; ++k) {
        if (source_cols[k] < 0)
            return cudaErrorInvalidValue;
        terms.source_columns[k] = source_cols[k];
        terms.coefficients[k] = coefficients[k];
    }
    gpbicgstabl_combine_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(
        work, xsol, lda, n, dst_col, terms, add_to_destination);
    return finish_gpbicg_launch(84, "gpbicgstabl_combine_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t solve_small_dense_system_device_wrapper(CUCOMPLEX *matrix_device,
                                                               CUCOMPLEX *rhs_device, int order,
                                                               CU_INT*info_device) {
    if (!matrix_device || !rhs_device || !info_device || order < 1 || order > 9)
        return cudaErrorInvalidValue;
    solve_small_dense_system_kernel<<<1, 1>>>(matrix_device, rhs_device, order, info_device);
    return finish_gpbicg_launch(87, "solve_small_dense_system_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pimzqmr_initialize_wrapper(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                                  CUCOMPLEX *work, int lda, int n) {
    if (!xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    pimzqmr_initialize_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, b, work, lda, n);
    return finish_gpbicg_launch(85, "pimzqmr_initialize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pimzqmr_shadow_wrapper(CUCOMPLEX *xi, CUCOMPLEX *work, int lda, int n) {
    if (!xi || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    pimzqmr_shadow_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, work, lda, n);
    return finish_gpbicg_launch(86, "pimzqmr_shadow_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t ssbicgsafe2_initialize_wrapper(const CUCOMPLEX *xr, const CUCOMPLEX *b,
                                                      CUCOMPLEX *work, int lda, int n) {
    if (!xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    ssbicgsafe2_initialize_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xr, b, work, lda,
                                                                                 n);
    return finish_gpbicg_launch(87, "ssbicgsafe2_initialize_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gcors2_initialize_residual_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr,
                                                          const CUCOMPLEX *b, CUCOMPLEX *work,
                                                          int lda, int n) {
    if (!xi || !xr || !b || !work || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gcors2_initialize_residual_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, b,
                                                                                     work, lda, n);
    return finish_gpbicg_launch(88, "gcors2_initialize_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gcors2_setup_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xr, CUCOMPLEX *work,
                                            const CUCOMPLEX *random_values, int lda, int n) {
    if (!xi || !xr || !work || !random_values || lda < n || n <= 0)
        return cudaErrorInvalidValue;
    gcors2_setup_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xr, work, random_values,
                                                                       lda, n);
    return finish_gpbicg_launch(89, "gcors2_setup_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gcors2_publish_solution_wrapper(CUCOMPLEX *xi, const CUCOMPLEX *xsol,
                                                       int n) {
    if (!xi || !xsol || n <= 0)
        return cudaErrorInvalidValue;
    gcors2_publish_solution_kernel<<<block_count(size_t(n)), THREADS_PER_BLOCK>>>(xi, xsol, n);
    return finish_gpbicg_launch(90, "gcors2_publish_solution_kernel");
}

namespace {

//-----------------------------------------------------------
__global__ void chan_preconditioner_pack_fft_kernel(const CUCOMPLEX *field, const CU_INT *dipole_map,
                                                    CUCOMPLEX *fft_planes, int nx, int ny, int nz,
                                                    int nxm, int nym, int nzm, int nmax) {
    const int plane = nx * ny;
    const int total = 3 * nz * plane;
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= total)
        return;
    const int xy = flat % plane;
    const int z = (flat / plane) % nz;
    const int component = flat / (plane * nz);
    const int x = xy % nx;
    const int y = xy / nx;
    // The Fortran routine indexes Tabdip with active FFT dimensions nx,ny.
    const int map_index = x + nx * y + nx * ny * z;
    CUCOMPLEX value = CUFFT_ZERO;
    if (map_index < nxm * nym * nzm) {
        const int dipole = dipole_map[map_index].dat;
        if (dipole > 0 && dipole <= nmax)
            value = field[3 * (dipole - 1) + component];
    }
    fft_planes[flat] = value;
}

//-----------------------------------------------------------
__global__ void chan_preconditioner_fft_to_rhs_kernel(const CUCOMPLEX *fft_planes, CUCOMPLEX *rhs,
                                                      int nx, int ny, int nz) {
    const int plane = nx * ny;
    const int total = 3 * nz * plane;
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= total)
        return;
    const int l = flat / (3 * nz);
    const int inner = flat % (3 * nz);
    // Fortran stores the three components interleaved at each z plane:
    // [x(z=0), y(z=0), z(z=0), x(z=1), ...].
    const int component = inner % 3;
    const int z = inner / 3;
    const int x = l / ny;
    const int y = l % ny;
    const int xy = x + nx * y;
    rhs[flat] = fft_planes[(component * nz + z) * plane + xy];
}

//-----------------------------------------------------------
__global__ void chan_preconditioner_rhs_to_fft_kernel(const CUCOMPLEX *rhs, CUCOMPLEX *fft_planes,
                                                      int nx, int ny, int nz) {
    const int plane = nx * ny;
    const int total = 3 * nz * plane;
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= total)
        return;
    const int l = flat / (3 * nz);
    const int inner = flat % (3 * nz);
    const int component = inner % 3;
    const int z = inner / 3;
    const int x = l / ny;
    const int y = l % ny;
    const int xy = x + nx * y;
    fft_planes[(component * nz + z) * plane + xy] = rhs[flat];
}

//-----------------------------------------------------------
__global__ void chan_preconditioner_unpack_fft_kernel(const CUCOMPLEX *fft_planes,
                                                      const CU_INT *dipole_map, CUCOMPLEX *field,
                                                      int nx, int ny, int nz, int nxm, int nym,
                                                      int nzm, int nmax, CU_REAL normalization) {
    const int plane = nx * ny;
    const int total = 3 * nz * plane;
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= total)
        return;
    const int xy = flat % plane;
    const int z = (flat / plane) % nz;
    const int component = flat / (plane * nz);
    const int x = xy % nx;
    const int y = xy / nx;
    const int map_index = x + nx * y + nx * ny * z;
    if (map_index >= nxm * nym * nzm)
        return;
    const int dipole = dipole_map[map_index].dat;
    if (dipole > 0 && dipole <= nmax)
        field[3 * (dipole - 1) + component] =
            CUFFT_DIV(fft_planes[flat], CUFFT_REAL(normalization));
}

//-----------------------------------------------------------
__global__ void chan_preconditioner_build_blocks_kernel(const CUCOMPLEX *chan_vectors,
                                                        CUCOMPLEX *matrices, int nx, int ny,
                                                        int nz) {
    const int plane = nx * ny;
    const int order = 3 * nz;
    const size_t block_size = size_t(order) * size_t(order);
    const size_t total = block_size * size_t(plane);
    const size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= total)
        return;

    const int batch = int(flat / block_size);
    const int matrix_index = int(flat % block_size);
    const int row = matrix_index % order;
    const int col = matrix_index / order;
    const int row_z = row / 3;
    const int row_component = row % 3;
    const int col_z = col / 3;
    const int col_component = col % 3;
    const int vector_index = batch + plane * abs(row_z - col_z);
    const CU_REAL sign = row_z >= col_z ? CU_REAL(1) : CU_REAL(-1);

    const CUCOMPLEX *chan_xx = chan_vectors;
    const CUCOMPLEX *chan_xy = chan_vectors + size_t(plane) * nz;
    const CUCOMPLEX *chan_xz = chan_vectors + size_t(2) * plane * nz;
    const CUCOMPLEX *chan_yy = chan_vectors + size_t(3) * plane * nz;
    const CUCOMPLEX *chan_yz = chan_vectors + size_t(4) * plane * nz;
    const CUCOMPLEX *chan_zz = chan_vectors + size_t(5) * plane * nz;

    CUCOMPLEX value;
    if (row_component == 0) {
        if (col_component == 0)
            value = chan_xx[vector_index];
        else if (col_component == 1)
            value = chan_xy[vector_index];
        else
            value = CUFFT_MUL(chan_xz[vector_index], CUFFT_REAL(sign));
    } else if (row_component == 1) {
        if (col_component == 0)
            value = chan_xy[vector_index];
        else if (col_component == 1)
            value = chan_yy[vector_index];
        else
            value = CUFFT_MUL(chan_yz[vector_index], CUFFT_REAL(sign));
    } else {
        if (col_component == 0)
            value = CUFFT_MUL(chan_xz[vector_index], CUFFT_REAL(sign));
        else if (col_component == 1)
            value = CUFFT_MUL(chan_yz[vector_index], CUFFT_REAL(sign));
        else
            value = chan_zz[vector_index];
    }
    matrices[flat] = value;
}

//-----------------------------------------------------------
__global__ void chan_preconditioner_green_tensor_kernel(CUCOMPLEX *green_components,
                                                        CU_REAL aretecube, CU_REAL k0, int nx,
                                                        int ny, int nz) {
    const int plane = nx * ny;
    const int count = plane * nz;
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int x_index = index % nx;
    const int y_index = (index / nx) % ny;
    const int z_index = index / plane;
    const CU_REAL x = aretecube * CU_REAL(x_index);
    const CU_REAL y = aretecube * CU_REAL(y_index);
    const CU_REAL z = aretecube * CU_REAL(z_index);
    const CU_REAL radius_squared = x * x + y * y + z * z;

    if (radius_squared == CU_REAL(0)) {
        for (int component = 0; component < 6; ++component)
            green_components[size_t(component) * count + index] = CUFFT_ZERO;
        return;
    }

    const CU_REAL radius = sqrt(radius_squared);
    const CU_REAL inverse_radius_squared = CU_REAL(1) / radius_squared;
    const CU_REAL phase_angle = k0 * radius;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase_complex = make_cuFloatComplex(cosf(phase_angle), sinf(phase_angle));
#else
    const CUCOMPLEX phase_complex = make_cuDoubleComplex(cos(phase_angle), sin(phase_angle));
#endif
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX constant1_complex =
        make_cuFloatComplex(CU_REAL(1) / (radius * radius_squared), -k0 * inverse_radius_squared);
#else
    const CUCOMPLEX constant1_complex =
        make_cuDoubleComplex(CU_REAL(1) / (radius * radius_squared), -k0 * inverse_radius_squared);
#endif
    const CU_REAL constant2 = k0 * k0 / radius;
    const CU_REAL ratios[6] = {x * x * inverse_radius_squared, x * y * inverse_radius_squared,
                               x * z * inverse_radius_squared, y * y * inverse_radius_squared,
                               y * z * inverse_radius_squared, z * z * inverse_radius_squared};
    const CU_REAL identity[6] = {CU_REAL(1), CU_REAL(0), CU_REAL(0),
                                 CU_REAL(1), CU_REAL(0), CU_REAL(1)};
    for (int component = 0; component < 6; ++component) {
        const CU_REAL coefficient1 = CU_REAL(3) * ratios[component] - identity[component];
        const CU_REAL coefficient2 = identity[component] - ratios[component];
        const CUCOMPLEX term1 = CUFFT_MUL(constant1_complex, CUFFT_REAL(coefficient1));
        const CUCOMPLEX term2 = CUFFT_REAL(constant2 * coefficient2);
        const CUCOMPLEX propagator = CUFFT_MUL(CUFFT_ADD(term1, term2), phase_complex);
        green_components[size_t(component) * count + index] =
            CUFFT_MUL(propagator, CUFFT_REAL(CU_REAL(-1)));
    }
}

//-----------------------------------------------------------
#if !PRODUITMATVECTFFT_SINGLE_PRECISION
__device__ double chan_atomic_add_double(double *address, double value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto *integer_address = reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old_value = *integer_address;
    unsigned long long int assumed;
    do {
        assumed = old_value;
        old_value = atomicCAS(integer_address, assumed,
                              __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old_value);
    return __longlong_as_double(old_value);
#endif
}
#endif

//-----------------------------------------------------------
__global__ void chan_preconditioner_mean_polarizability_kernel(const CUCOMPLEX *polarizabilities,
                                                               int count, CUCOMPLEX *sum,
                                                               CU_INT*nonzero_count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const CUCOMPLEX value = polarizabilities[index];
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    atomicAdd(&sum->x, value.x);
    atomicAdd(&sum->y, value.y);
#else
    chan_atomic_add_double(&sum->x, value.x);
    chan_atomic_add_double(&sum->y, value.y);
#endif
    if (value.x != CU_REAL(0) || value.y != CU_REAL(0))
        atomicAdd(&nonzero_count->dat, 1);
}

//-----------------------------------------------------------
__global__ void chan_preconditioner_finalize_mean_polarizability_kernel(CUCOMPLEX *sum,
                                                                        const CU_INT*nonzero_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const int count = nonzero_count->dat;
    if (count <= 0) {
        *sum = CUFFT_ZERO;
        return;
    }
    const CU_REAL inverse_count = CU_REAL(1) / CU_REAL(count);
    *sum = CUFFT_MUL(*sum, CUFFT_REAL(inverse_count));
}

//-----------------------------------------------------------
__global__ void chan_preconditioner_interpolate_kernel(const CUCOMPLEX *green_components,
                                                       CUCOMPLEX *fft_components,
                                                       const CUCOMPLEX *mean_polarizability, int nx,
                                                       int ny, int nz) {
    const int plane = nx * ny;
    const int total = plane * nz;
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total)
        return;
    const CUCOMPLEX mean = *mean_polarizability;

    const int x = index % nx;
    const int y = (index / nx) % ny;
    const int z = index / plane;
    int x1 = nx - x;
    int y1 = ny - y;
    if (x == 0)
        x1 = 0;
    if (y == 0)
        y1 = 0;

    const CU_REAL dx1 = CU_REAL(x);
    const CU_REAL dx2 = CU_REAL(nx - x);
    const CU_REAL dy1 = CU_REAL(y);
    const CU_REAL dy2 = CU_REAL(ny - y);
    const int a = x1 + nx * y1;
    const int b = x1 + nx * y;
    const int c = x + nx * y1;
    const int d = x + nx * y;
    const size_t component_stride = size_t(plane) * nz;
    const int output_index = z * plane + (x1 + nx * y1);

    const CUCOMPLEX *Txx = green_components;
    const CUCOMPLEX *Txy = green_components + component_stride;
    const CUCOMPLEX *Txz = green_components + size_t(2) * component_stride;
    const CUCOMPLEX *Tyy = green_components + size_t(3) * component_stride;
    const CUCOMPLEX *Tyz = green_components + size_t(4) * component_stride;
    const CUCOMPLEX *Tzz = green_components + size_t(5) * component_stride;
    const size_t slice = size_t(z) * plane;
    const CU_REAL inv_plane = CU_REAL(1) / CU_REAL(plane);

    const CU_REAL w_a = dx1 * dy1 * inv_plane;
    const CU_REAL w_b = dx1 * dy2 * inv_plane;
    const CU_REAL w_c = dx2 * dy1 * inv_plane;
    const CU_REAL w_d = dx2 * dy2 * inv_plane;

    CUCOMPLEX values[6];
    values[0] = CUFFT_ADD(CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(Txx[slice + a], CUFFT_REAL(w_a)),
                                              CUFFT_MUL(Txx[slice + b], CUFFT_REAL(w_b))),
                                    CUFFT_ADD(CUFFT_MUL(Txx[slice + c], CUFFT_REAL(w_c)),
                                              CUFFT_MUL(Txx[slice + d], CUFFT_REAL(w_d)))),
                          CUFFT_ZERO);
    values[1] = CUFFT_ADD(CUFFT_SUB(CUFFT_SUB(CUFFT_MUL(Txy[slice + a], CUFFT_REAL(w_a)),
                                              CUFFT_MUL(Txy[slice + b], CUFFT_REAL(w_b))),
                                    CUFFT_MUL(Txy[slice + c], CUFFT_REAL(w_c))),
                          CUFFT_MUL(Txy[slice + d], CUFFT_REAL(w_d)));
    values[2] = CUFFT_SUB(CUFFT_SUB(CUFFT_ADD(CUFFT_MUL(Txz[slice + a], CUFFT_REAL(w_a)),
                                              CUFFT_MUL(Txz[slice + b], CUFFT_REAL(w_b))),
                                    CUFFT_MUL(Txz[slice + c], CUFFT_REAL(w_c))),
                          CUFFT_MUL(Txz[slice + d], CUFFT_REAL(w_d)));
    values[3] = CUFFT_ADD(CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(Tyy[slice + a], CUFFT_REAL(w_a)),
                                              CUFFT_MUL(Tyy[slice + b], CUFFT_REAL(w_b))),
                                    CUFFT_ADD(CUFFT_MUL(Tyy[slice + c], CUFFT_REAL(w_c)),
                                              CUFFT_MUL(Tyy[slice + d], CUFFT_REAL(w_d)))),
                          CUFFT_ZERO);
    values[4] = CUFFT_SUB(CUFFT_ADD(CUFFT_SUB(CUFFT_MUL(Tyz[slice + a], CUFFT_REAL(w_a)),
                                              CUFFT_MUL(Tyz[slice + b], CUFFT_REAL(w_b))),
                                    CUFFT_MUL(Tyz[slice + c], CUFFT_REAL(w_c))),
                          CUFFT_MUL(Tyz[slice + d], CUFFT_REAL(w_d)));
    values[5] = CUFFT_ADD(CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(Tzz[slice + a], CUFFT_REAL(w_a)),
                                              CUFFT_MUL(Tzz[slice + b], CUFFT_REAL(w_b))),
                                    CUFFT_ADD(CUFFT_MUL(Tzz[slice + c], CUFFT_REAL(w_c)),
                                              CUFFT_MUL(Tzz[slice + d], CUFFT_REAL(w_d)))),
                          CUFFT_ZERO);

    for (int component = 0; component < 6; ++component)
        fft_components[size_t(component) * total + output_index] =
            CUFFT_MUL(values[component], mean);
    if (z == 0 && output_index == 0) {
        fft_components[0] = CUFFT_ADD(fft_components[0], CUFFT_REAL(CU_REAL(1)));
        fft_components[size_t(3) * total] =
            CUFFT_ADD(fft_components[size_t(3) * total], CUFFT_REAL(CU_REAL(1)));
        fft_components[size_t(5) * total] =
            CUFFT_ADD(fft_components[size_t(5) * total], CUFFT_REAL(CU_REAL(1)));
    }
}

//-----------------------------------------------------------
__global__ void chan_preconditioner_transpose_vectors_kernel(const CUCOMPLEX *fft_components,
                                                             CUCOMPLEX *chan_vectors, int nx,
                                                             int ny, int nz) {
    const int plane = nx * ny;
    const int total = plane * nz;
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= 6 * total)
        return;
    const int component = flat / total;
    const int index = flat % total;
    const int z = index / plane;
    const int xy = index % plane;
    const int x = xy % nx;
    const int y = xy / nx;
    const int transposed_xy = y + ny * x;
    chan_vectors[size_t(component) * total + size_t(z) * plane + transposed_xy] =
        fft_components[flat];
}

//-----------------------------------------------------------
__device__ CUCOMPLEX scalar_chan_projected_green(CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL k0,
                                                 CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz) {
    const CU_REAL radius_squared = x * x + y * y + z * z;
    if (radius_squared == CU_REAL(0))
        return CUFFT_ZERO;
    const CU_REAL radius = sqrt(radius_squared);
    const CU_REAL inverse_radius_squared = CU_REAL(1) / radius_squared;
    const CU_REAL phase_angle = k0 * radius;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cosf(phase_angle), sinf(phase_angle));
    const CUCOMPLEX constant1 =
        make_cuFloatComplex(CU_REAL(1) / (radius * radius_squared), -k0 * inverse_radius_squared);
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(phase_angle), sin(phase_angle));
    const CUCOMPLEX constant1 =
        make_cuDoubleComplex(CU_REAL(1) / (radius * radius_squared), -k0 * inverse_radius_squared);
#endif
    const CU_REAL coordinates[3] = {x, y, z};
    CUCOMPLEX tensor[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const CU_REAL identity = row == col ? CU_REAL(1) : CU_REAL(0);
            const CU_REAL ratio = coordinates[row] * coordinates[col] * inverse_radius_squared;
            const CU_REAL coefficient1 = CU_REAL(3) * ratio - identity;
            const CU_REAL coefficient2 = identity - ratio;
            const CUCOMPLEX radial = CUFFT_ADD(CUFFT_MUL(constant1, CUFFT_REAL(coefficient1)),
                                               CUFFT_REAL((k0 * k0 / radius) * coefficient2));
            tensor[row][col] = CUFFT_MUL(radial, phase);
        }
    }
    const CUCOMPLEX incoming[3] = {ux, uy, uz};
    CUCOMPLEX projected = CUFFT_ZERO;
    for (int row = 0; row < 3; ++row) {
        CUCOMPLEX product = CUFFT_ZERO;
        for (int col = 0; col < 3; ++col)
            product = CUFFT_ADD(product, CUFFT_MUL(tensor[row][col], incoming[col]));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CUCOMPLEX conjugate = cuConjf(incoming[row]);
#else
        const CUCOMPLEX conjugate = cuConj(incoming[row]);
#endif
        projected = CUFFT_ADD(projected, CUFFT_MUL(conjugate, product));
    }
    return projected;
}

//-----------------------------------------------------------
__global__ void scalar_driver_project_incident_kernel(const CUCOMPLEX *vector_incident,
                                                      CUCOMPLEX *scalar_incident, CUCOMPLEX uincx,
                                                      CUCOMPLEX uincy, CUCOMPLEX uincz, int count) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count)
        return;
    const CUCOMPLEX cx = CUFFT_CONJ(uincx);
    const CUCOMPLEX cy = CUFFT_CONJ(uincy);
    const CUCOMPLEX cz = CUFFT_CONJ(uincz);
    scalar_incident[i] = CUFFT_ADD(
        CUFFT_ADD(CUFFT_MUL(vector_incident[3 * i], cx), CUFFT_MUL(vector_incident[3 * i + 1], cy)),
        CUFFT_MUL(vector_incident[3 * i + 2], cz));
}

//-----------------------------------------------------------
__global__ void scalar_driver_apply_polarizability_kernel(const CUCOMPLEX *input,
                                                          const CUCOMPLEX *polarizability,
                                                          CUCOMPLEX *output, int count) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count)
        output[i] = CUFFT_MUL(polarizability[i], input[i]);
}

//-----------------------------------------------------------
__global__ void scalar_driver_exponential_guess_kernel(const CUCOMPLEX *incident,
                                                       const CUCOMPLEX *green_correction,
                                                       CUCOMPLEX *initial_guess, int count) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count)
        return;
    const CUCOMPLEX field = incident[i];
    if (field.x == CU_REAL(0) && field.y == CU_REAL(0)) {
        initial_guess[i] = CUFFT_ZERO;
        return;
    }
    const CUCOMPLEX exponent = CUFFT_DIV(green_correction[i], field);
    const CU_REAL magnitude = exp(exponent.x);
    const CU_REAL phase = exponent.y;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX exponential =
        make_cuFloatComplex(magnitude * cosf(phase), magnitude * sinf(phase));
#else
    const CUCOMPLEX exponential =
        make_cuDoubleComplex(magnitude * cos(phase), magnitude * sin(phase));
#endif
    initial_guess[i] = CUFFT_MUL(field, exponential);
}

//-----------------------------------------------------------
__global__ void scalar_driver_copy_solution_kernel(const CUCOMPLEX *solution,
                                                   CUCOMPLEX *local_solution, int count) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count)
        local_solution[i] = solution[i];
}

//-----------------------------------------------------------
__global__ void scalar_driver_residual_kernel(CUCOMPLEX *product, const CUCOMPLEX *rhs,
                                              const CUCOMPLEX *solution, CUCOMPLEX *scalar_solution,
                                              int count) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count)
        return;
    product[i] = CUFFT_SUB(product[i], rhs[i]);
    scalar_solution[i] = solution[i];
}

//-----------------------------------------------------------
__global__ void scalar_driver_expand_fields_kernel(const CUCOMPLEX *solution,
                                                   const CUCOMPLEX *local_solution,
                                                   CUCOMPLEX *vector_field,
                                                   CUCOMPLEX *vector_local_field, CUCOMPLEX uincx,
                                                   CUCOMPLEX uincy, CUCOMPLEX uincz, int count) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count)
        return;
    const CUCOMPLEX value = solution[i];
    const CUCOMPLEX local_value = local_solution[i];
    vector_field[3 * i] = CUFFT_MUL(value, uincx);
    vector_field[3 * i + 1] = CUFFT_MUL(value, uincy);
    vector_field[3 * i + 2] = CUFFT_MUL(value, uincz);
    vector_local_field[3 * i] = CUFFT_MUL(local_value, uincx);
    vector_local_field[3 * i + 1] = CUFFT_MUL(local_value, uincy);
    vector_local_field[3 * i + 2] = CUFFT_MUL(local_value, uincz);
}

//-----------------------------------------------------------
__device__ CUCOMPLEX scalar_chan_green_value(CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL k0, int nrig,
                                             CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz) {
    if (nrig == 8)
        return scalar_chan_projected_green(x, y, z, k0, ux, uy, uz);
    const CU_REAL radius = sqrt(x * x + y * y + z * z);
    if (radius == CU_REAL(0))
        return CUFFT_ZERO;
    const CU_REAL phase_angle = k0 * radius;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cosf(phase_angle), sinf(phase_angle));
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(phase_angle), sin(phase_angle));
#endif
    return CUFFT_MUL(CUFFT_REAL(k0 * k0 / radius), phase);
}

//-----------------------------------------------------------
__global__ void generate_scalar_green_kernel(CUCOMPLEX *green, int nx, int ny, int nz, int nx2,
                                             int ny2, int nz2, int fft_count, int tensor_count,
                                             CU_REAL aretecube, CU_REAL k0, CUCOMPLEX uincx,
                                             CUCOMPLEX uincy, CUCOMPLEX uincz, int nrig) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= tensor_count)
        return;
    if (index >= fft_count) {
        green[index] = CUFFT_ZERO;
        return;
    }

    const int i1 = index % nx2 + 1;
    const int j1 = (index / nx2) % ny2 + 1;
    const int k1 = index / (nx2 * ny2) + 1;
    if (i1 == nx + 1 || j1 == ny + 1 || k1 == nz + 1) {
        green[index] = CUFFT_ZERO;
        return;
    }

    const int i = i1 > nx ? i1 - 1 - nx2 : i1 - 1;
    const int j = j1 > ny ? j1 - 1 - ny2 : j1 - 1;
    const int k = k1 > nz ? k1 - 1 - nz2 : k1 - 1;
    const CU_REAL x = CU_REAL(i) * aretecube;
    const CU_REAL y = CU_REAL(j) * aretecube;
    const CU_REAL z = CU_REAL(k) * aretecube;
    green[index] = scalar_chan_green_value(x, y, z, k0, nrig, uincx, uincy, uincz);
}

//-----------------------------------------------------------
__global__ void generate_multilayer_scalar_green_kernel(CUCOMPLEX *green, int nfft2d, int count,
                                                        CU_REAL aretecube, CU_REAL k0) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int side = 2 * nfft2d;
    const int ii = index % side;
    const int jj = index / side;
    if (ii == nfft2d || jj == nfft2d) {
        green[index] = CUFFT_ZERO;
        return;
    }
    const int i = ii > nfft2d ? ii - side : ii;
    const int j = jj > nfft2d ? jj - side : jj;
    const CU_REAL x = CU_REAL(i) * aretecube;
    const CU_REAL y = CU_REAL(j) * aretecube;
    const CU_REAL z = aretecube;
    const CU_REAL radius = sqrt(x * x + y * y + z * z);
    const CU_REAL phase_angle = k0 * radius;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cosf(phase_angle), sinf(phase_angle));
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(phase_angle), sin(phase_angle));
#endif
    green[index] = CUFFT_MUL(CUFFT_REAL(k0 * k0 / radius), phase);
}

//-----------------------------------------------------------
__global__ void generate_multilayer_projected_green_kernel(CUCOMPLEX *green, int nfft2d, int count,
                                                           CU_REAL aretecube, CU_REAL k0,
                                                           CUCOMPLEX ux, CUCOMPLEX uy,
                                                           CUCOMPLEX uz) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int side = 2 * nfft2d;
    const int ii = index % side;
    const int jj = index / side;
    if (ii == nfft2d || jj == nfft2d) {
        green[index] = CUFFT_ZERO;
        return;
    }
    const CU_REAL x = CU_REAL(ii > nfft2d ? ii - side : ii) * aretecube;
    const CU_REAL y = CU_REAL(jj > nfft2d ? jj - side : jj) * aretecube;
    const CU_REAL z = aretecube;
    const CU_REAL r2 = x * x + y * y + z * z;
    const CU_REAL r = sqrt(r2);
    const CU_REAL phase_angle = k0 * r;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cosf(phase_angle), sinf(phase_angle));
    const CUCOMPLEX imaginary_unit = make_cuFloatComplex(0.0f, 1.0f);
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(phase_angle), sin(phase_angle));
    const CUCOMPLEX imaginary_unit = make_cuDoubleComplex(0.0, 1.0);
#endif
    const CU_REAL rv[3] = {x, y, z};
    const CUCOMPLEX u[3] = {ux, uy, uz};
    const CU_REAL c1r = CU_REAL(1) / (r * r2);
    const CUCOMPLEX c1 = CUFFT_SUB(CUFFT_REAL(c1r), CUFFT_MUL(imaginary_unit, CUFFT_REAL(k0 / r2)));
    const CUCOMPLEX c2 = CUFFT_REAL(k0 * k0 / r);
    CUCOMPLEX projected = CUFFT_ZERO;
    for (int row = 0; row < 3; ++row) {
        CUCOMPLEX row_value = CUFFT_ZERO;
        for (int col = 0; col < 3; ++col) {
            const CU_REAL delta = row == col ? CU_REAL(1) : CU_REAL(0);
            const CU_REAL rr = rv[row] * rv[col] / r2;
            const CUCOMPLEX a = CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * rr - delta), c1);
            const CUCOMPLEX b = CUFFT_MUL(CUFFT_REAL(delta - rr), c2);
            const CUCOMPLEX tensor = CUFFT_MUL(CUFFT_ADD(a, b), phase);
            row_value = CUFFT_ADD(row_value, CUFFT_MUL(tensor, u[col]));
        }
        projected = CUFFT_ADD(projected, CUFFT_MUL(CUFFT_CONJ(u[row]), row_value));
    }
    green[index] = projected;
}

//-----------------------------------------------------------
__device__ CUCOMPLEX multilayer_dyadic_component(CU_REAL ri, CU_REAL rj, CU_REAL delta,
                                                 CU_REAL radius2, CUCOMPLEX constant1,
                                                 CUCOMPLEX constant2, CUCOMPLEX phase) {
    const CU_REAL rr = ri * rj / radius2;
    const CUCOMPLEX a = CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * rr - delta), constant1);
    const CUCOMPLEX b = CUFFT_MUL(CUFFT_REAL(delta - rr), constant2);
    return CUFFT_MUL(CUFFT_ADD(a, b), phase);
}

//-----------------------------------------------------------
__global__ void generate_multilayer_dyadic_green_kernel(CUCOMPLEX *gxx, CUCOMPLEX *gxy,
                                                        CUCOMPLEX *gxz, CUCOMPLEX *gyy,
                                                        CUCOMPLEX *gyz, CUCOMPLEX *gzz, int nfft2d,
                                                        int count, CU_REAL aretecube, CU_REAL k0) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int side = 2 * nfft2d, ix = index % side, iy = index / side;
    if (ix == nfft2d || iy == nfft2d) {
        gxx[index] = gxy[index] = gxz[index] = gyy[index] = gyz[index] = gzz[index] = CUFFT_ZERO;
        return;
    }
    const CU_REAL x = CU_REAL(ix > nfft2d ? ix - side : ix) * aretecube;
    const CU_REAL y = CU_REAL(iy > nfft2d ? iy - side : iy) * aretecube;
    const CU_REAL z = aretecube, r2 = x * x + y * y + z * z, r = sqrt(r2);
    const CU_REAL angle = k0 * r;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cosf(angle), sinf(angle));
    const CUCOMPLEX imaginary = make_cuFloatComplex(0.0f, 1.0f);
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(angle), sin(angle));
    const CUCOMPLEX imaginary = make_cuDoubleComplex(0.0, 1.0);
#endif
    const CUCOMPLEX c1 =
        CUFFT_SUB(CUFFT_REAL(CU_REAL(1) / (r * r2)), CUFFT_MUL(imaginary, CUFFT_REAL(k0 / r2)));
    const CUCOMPLEX c2 = CUFFT_REAL(k0 * k0 / r);
    gxx[index] = multilayer_dyadic_component(x, x, CU_REAL(1), r2, c1, c2, phase);
    gxy[index] = multilayer_dyadic_component(x, y, CU_REAL(0), r2, c1, c2, phase);
    gxz[index] = multilayer_dyadic_component(x, z, CU_REAL(0), r2, c1, c2, phase);
    gyy[index] = multilayer_dyadic_component(y, y, CU_REAL(1), r2, c1, c2, phase);
    gyz[index] = multilayer_dyadic_component(y, z, CU_REAL(0), r2, c1, c2, phase);
    gzz[index] = multilayer_dyadic_component(z, z, CU_REAL(1), r2, c1, c2, phase);
}

//-----------------------------------------------------------
__device__ void green_tensor_cisi(CU_REAL x, CU_REAL *ci, CU_REAL *si) {
    const CU_REAL euler = CU_REAL(0.5772156649015329), eps = CU_REAL(1.0e-15);
    const CU_REAL x2 = x * x;
    if (x == CU_REAL(0)) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        *ci = CU_REAL(-1.0e30f);
#else
        *ci = -CU_REAL(1.0e300);
#endif
        *si = CU_REAL(0);
        return;
    }
    if (x <= CU_REAL(16)) {
        CU_REAL term = -CU_REAL(0.25) * x2;
        *ci = euler + log(x) + term;
        for (int k = 2; k <= 40; ++k) {
            term = -CU_REAL(0.5) * term * CU_REAL(k - 1) / CU_REAL(k * k * (2 * k - 1)) * x2;
            *ci += term;
            if (fabs(term) < fabs(*ci) * eps)
                break;
        }
        term = x;
        *si = x;
        for (int k = 1; k <= 40; ++k) {
            term = -CU_REAL(0.5) * term * CU_REAL(2 * k - 1) / CU_REAL(k) /
                   CU_REAL(4 * k * k + 4 * k + 1) * x2;
            *si += term;
            if (fabs(term) < fabs(*si) * eps)
                break;
        }
        return;
    }
    if (x <= CU_REAL(32)) {
        CU_REAL bj[101];
        const int m = int(CU_REAL(47.2) + CU_REAL(0.82) * x);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        CU_REAL xa1 = CU_REAL(0), xa0 = CU_REAL(1.0e-30f);
#else
        CU_REAL xa1 = CU_REAL(0), xa0 = CU_REAL(1.0e-300);
#endif
        for (int k = m; k >= 1; --k) {
            const CU_REAL xa = CU_REAL(4 * k) * xa0 / x - xa1;
            bj[k] = xa;
            xa1 = xa0;
            xa0 = xa;
        }
        CU_REAL sum = bj[1];
        for (int k = 3; k <= m; k += 2)
            sum += CU_REAL(2) * bj[k];
        for (int k = 1; k <= m; ++k)
            bj[k] /= sum;
        CU_REAL term = CU_REAL(1), g1 = bj[1];
        for (int k = 2; k <= m; ++k) {
            term = CU_REAL(0.25) * term * CU_REAL((2 * k - 3) * (2 * k - 3)) / CU_REAL(k - 1) /
                   CU_REAL((2 * k - 1) * (2 * k - 1)) * x;
            g1 += bj[k] * term;
        }
        term = CU_REAL(1);
        CU_REAL g2 = bj[1];
        for (int k = 2; k <= m; ++k) {
            term = CU_REAL(0.25) * term * CU_REAL((2 * k - 5) * (2 * k - 5)) / CU_REAL(k - 1) /
                   CU_REAL((2 * k - 3) * (2 * k - 3)) * x;
            g2 += bj[k] * term;
        }
        const CU_REAL c = cos(x * CU_REAL(0.5)), s = sin(x * CU_REAL(0.5));
        *ci = euler + log(x) - x * s * g1 + CU_REAL(2) * c * g2 - CU_REAL(2) * c * c;
        *si = x * c * g1 + CU_REAL(2) * s * g2 - sin(x);
        return;
    }
    CU_REAL f = CU_REAL(1), term = CU_REAL(1);
    for (int k = 1; k <= 9; ++k) {
        term = -CU_REAL(2) * term * CU_REAL(k) * CU_REAL(2 * k - 1) / x2;
        f += term;
    }
    CU_REAL g = CU_REAL(1) / x;
    term = g;
    for (int k = 1; k <= 8; ++k) {
        term = -CU_REAL(2) * term * CU_REAL(2 * k + 1) * CU_REAL(k) / x2;
        g += term;
    }
    *ci = f * sin(x) / x - g * cos(x) / x;
    *si = CU_REAL(1.570796326794897) - f * cos(x) / x - g * sin(x) / x;
}

//-----------------------------------------------------------
__device__ void green_tensor_free_space_value(CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL k0,
                                              CUCOMPLEX value[6]) {
    const CU_REAL r2 = x * x + y * y + z * z;
    if (r2 == CU_REAL(0)) {
        for (int c = 0; c < 6; ++c)
            value[c] = CUFFT_ZERO;
        return;
    }
    const CU_REAL r = sqrt(r2), invr2 = CU_REAL(1) / r2;
    const CU_REAL angle = k0 * r;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cosf(angle), sinf(angle));
    const CUCOMPLEX c1 = make_cuFloatComplex(CU_REAL(1) / (r * r2), -k0 * invr2);
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(angle), sin(angle));
    const CUCOMPLEX c1 = make_cuDoubleComplex(CU_REAL(1) / (r * r2), -k0 * invr2);
#endif
    const CU_REAL c2 = k0 * k0 / r;
    const CU_REAL qx = x * x * invr2, qy = y * y * invr2, qz = z * z * invr2;
    const CU_REAL qxy = x * y * invr2, qxz = x * z * invr2, qyz = y * z * invr2;
    value[0] = CUFFT_MUL(CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qx - CU_REAL(1)), c1),
                                   CUFFT_REAL((CU_REAL(1) - qx) * c2)),
                         phase);
    value[1] = CUFFT_MUL(
        CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qxy), c1), CUFFT_REAL(-qxy * c2)), phase);
    value[2] = CUFFT_MUL(
        CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qxz), c1), CUFFT_REAL(-qxz * c2)), phase);
    value[3] = CUFFT_MUL(CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qy - CU_REAL(1)), c1),
                                   CUFFT_REAL((CU_REAL(1) - qy) * c2)),
                         phase);
    value[4] = CUFFT_MUL(
        CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qyz), c1), CUFFT_REAL(-qyz * c2)), phase);
    value[5] = CUFFT_MUL(CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qz - CU_REAL(1)), c1),
                                   CUFFT_REAL((CU_REAL(1) - qz) * c2)),
                         phase);
}

//-----------------------------------------------------------
__device__ CU_REAL green_tensor_simpson_weight(int i) {
    if (i == 0 || i == 128)
        return CU_REAL(1);
    return (i & 1) ? CU_REAL(4) : CU_REAL(2);
}

//-----------------------------------------------------------
__device__ CUCOMPLEX green_tensor_complex_sqrt(CUCOMPLEX value) {
    const CU_REAL magnitude = hypot(value.x, value.y);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CU_REAL real_part = sqrtf((magnitude + value.x) / CU_REAL(2));
    CU_REAL imag_part = sqrtf(fmaxf(CU_REAL(0), (magnitude - value.x) / CU_REAL(2)));
#else
    const CU_REAL real_part = sqrt((magnitude + value.x) / CU_REAL(2));
    CU_REAL imag_part = sqrt(fmax(CU_REAL(0), (magnitude - value.x) / CU_REAL(2)));
#endif
    if (value.y < CU_REAL(0))
        imag_part = -imag_part;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real_part, imag_part);
#else
    return make_cuDoubleComplex(real_part, imag_part);
#endif
}

//-----------------------------------------------------------
__device__ CUCOMPLEX green_tensor_exp_i(CUCOMPLEX value) {
    const CU_REAL scale = exp(-value.y);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(scale * cosf(value.x), scale * sinf(value.x));
#else
    return make_cuDoubleComplex(scale * cos(value.x), scale * sin(value.x));
#endif
}

//-----------------------------------------------------------
__device__ void green_tensor_medium_value(CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL aretecube,
                                          CU_REAL k0, CUCOMPLEX eps0, CUCOMPLEX value[6]) {
    const CU_REAL r2 = x * x + y * y + z * z;
    const CUCOMPLEX iunit =
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        make_cuFloatComplex(CU_REAL(0), CU_REAL(1));
#else
        make_cuDoubleComplex(CU_REAL(0), CU_REAL(1));
#endif
    const CUCOMPLEX k0c = CUFFT_MUL(CUFFT_REAL(k0), green_tensor_complex_sqrt(eps0));
    if (r2 == CU_REAL(0)) {
        const CU_REAL radius =
            aretecube * pow(CU_REAL(3) / (CU_REAL(4) * CU_REAL(3.14159265358979323846)),
                            CU_REAL(1) / CU_REAL(3));
        const CU_REAL volume = aretecube * aretecube * aretecube;
        const CUCOMPLEX k0ca = CUFFT_MUL(k0c, CUFFT_REAL(radius));
        const CUCOMPLEX inside =
            CUFFT_SUB(CUFFT_MUL(CUFFT_SUB(CUFFT_REAL(CU_REAL(1)), CUFFT_MUL(iunit, k0ca)),
                                green_tensor_exp_i(k0ca)),
                      CUFFT_REAL(CU_REAL(1)));
        const CUCOMPLEX term1 =
            CUFFT_DIV(CUFFT_REAL(-CU_REAL(4) * CU_REAL(3.14159265358979323846) / CU_REAL(3)), eps0);
        const CUCOMPLEX term2 = CUFFT_DIV(
            CUFFT_MUL(CUFFT_REAL(CU_REAL(8) * CU_REAL(3.14159265358979323846) / CU_REAL(3)),
                      inside),
            eps0);
        const CUCOMPLEX diagonal =
            CUFFT_MUL(CUFFT_REAL(CU_REAL(1) / volume), CUFFT_ADD(term1, term2));
        value[0] = diagonal;
        value[1] = CUFFT_ZERO;
        value[2] = CUFFT_ZERO;
        value[3] = diagonal;
        value[4] = CUFFT_ZERO;
        value[5] = diagonal;
        return;
    }
    const CU_REAL r = sqrt(r2), invr2 = CU_REAL(1) / r2;
    const CUCOMPLEX kc2 = CUFFT_MUL(k0c, k0c);
    const CUCOMPLEX term_a = CUFFT_DIV(CUFFT_REAL(CU_REAL(1) / r), kc2);
    const CUCOMPLEX term_b = CUFFT_DIV(iunit, k0c);
    const CUCOMPLEX c1 = CUFFT_MUL(CUFFT_SUB(term_a, term_b), CUFFT_REAL(invr2));
    const CU_REAL c2 = CU_REAL(1) / r;
    const CUCOMPLEX phase = green_tensor_exp_i(CUFFT_MUL(k0c, CUFFT_REAL(r)));
    const CUCOMPLEX factor = CUFFT_MUL(CUFFT_REAL(k0 * k0), phase);
    const CU_REAL qx = x * x * invr2, qy = y * y * invr2, qz = z * z * invr2;
    const CU_REAL qxy = x * y * invr2, qxz = x * z * invr2, qyz = y * z * invr2;
#define GREEN_MEDIUM_COMPONENT(q, delta)                                                           \
    CUFFT_MUL(CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * (q) - (delta)), c1),                     \
                        CUFFT_REAL(((delta) - (q)) * c2)),                                         \
              factor)
    value[0] = GREEN_MEDIUM_COMPONENT(qx, CU_REAL(1));
    value[1] = GREEN_MEDIUM_COMPONENT(qxy, CU_REAL(0));
    value[2] = GREEN_MEDIUM_COMPONENT(qxz, CU_REAL(0));
    value[3] = GREEN_MEDIUM_COMPONENT(qy, CU_REAL(1));
    value[4] = GREEN_MEDIUM_COMPONENT(qyz, CU_REAL(0));
    value[5] = GREEN_MEDIUM_COMPONENT(qz, CU_REAL(1));
#undef GREEN_MEDIUM_COMPONENT
}

//-----------------------------------------------------------
__device__ void green_tensor_integrated_value(CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL aretecube,
                                              CU_REAL k0, CUCOMPLEX eps0, int medium,
                                              CUCOMPLEX value[6]) {
    const CU_REAL radius = sqrt(x * x + y * y + z * z);
    const CU_REAL small_radius = aretecube * CU_REAL(0.1);
    if (radius <= small_radius) {
        if (medium)
            green_tensor_medium_value(x, y, z, aretecube, k0, eps0, value);
        else
            for (int c = 0; c < 6; ++c)
                value[c] = CUFFT_ZERO;
        return;
    }
    if (radius >= CU_REAL(6.2831853071795864769) / k0) {
        if (medium)
            green_tensor_medium_value(x, y, z, aretecube, k0, eps0, value);
        else
            green_tensor_free_space_value(x, y, z, k0, value);
        return;
    }
    constexpr int steps = 128;
    const CU_REAL step = aretecube / CU_REAL(steps);
    const CU_REAL xstart = x - aretecube / CU_REAL(2);
    const CU_REAL ystart = y - aretecube / CU_REAL(2);
    const CU_REAL zstart = z - aretecube / CU_REAL(2);
    CUCOMPLEX sum[6];
    for (int c = 0; c < 6; ++c)
        sum[c] = CUFFT_ZERO;
    for (int i = 0; i <= steps; ++i) {
        const CU_REAL xa = xstart + step * CU_REAL(i);
        const CU_REAL wi = green_tensor_simpson_weight(i);
        for (int j = 0; j <= steps; ++j) {
            const CU_REAL ya = ystart + step * CU_REAL(j);
            const CU_REAL wij = wi * green_tensor_simpson_weight(j);
            for (int k = 0; k <= steps; ++k) {
                const CU_REAL za = zstart + step * CU_REAL(k);
                const CU_REAL weight = wij * green_tensor_simpson_weight(k);
                CUCOMPLEX sample[6];
                if (medium)
                    green_tensor_medium_value(xa, ya, za, aretecube, k0, eps0, sample);
                else
                    green_tensor_free_space_value(xa, ya, za, k0, sample);
                for (int c = 0; c < 6; ++c)
                    sum[c] = CUFFT_ADD(sum[c], CUFFT_MUL(sample[c], CUFFT_REAL(weight)));
            }
        }
    }
    const CU_REAL scale = (step / CU_REAL(3)) * (step / CU_REAL(3)) * (step / CU_REAL(3)) /
                          (aretecube * aretecube * aretecube);
    for (int c = 0; c < 6; ++c)
        value[c] = CUFFT_MUL(sum[c], CUFFT_REAL(scale));
    if (fabs(x) <= small_radius) {
        value[1] = CUFFT_ZERO;
        value[2] = CUFFT_ZERO;
    }
    if (fabs(y) <= small_radius) {
        value[1] = CUFFT_ZERO;
        value[4] = CUFFT_ZERO;
    }
    if (fabs(z) <= small_radius) {
        value[2] = CUFFT_ZERO;
        value[4] = CUFFT_ZERO;
    }
}

//-----------------------------------------------------------
__global__ void green_tensor_direct_space_kernel(GreenTensorDevicePointers spatial, int nx, int ny, int nz,
                                                 int nxm, int nym, int nzm, size_t tensor_count,
                                                 CU_REAL aretecube, CU_REAL k0, CUCOMPLEX eps0,
                                                 int medium, int nquad, int filtered) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tensor_count)
        return;
    // Fortran allocates using nxm/nym/nzm capacities, but packs the active
    // tensor with nx+1 and ny+1 strides. Keep that compact layout so padded
    // meshes (where capacity exceeds the active dimensions) match Fortran.
    const size_t sx = size_t(nx) + 1;
    const size_t sy = size_t(ny) + 1;
    const size_t active_count = sx * sy * (size_t(nz) + 1);
    if (index >= active_count) {
        CUCOMPLEX *gxx = spatial.data[0];
        CUCOMPLEX *gxy = spatial.data[1];
        CUCOMPLEX *gxz = spatial.data[2];
        CUCOMPLEX *gyy = spatial.data[3];
        CUCOMPLEX *gyz = spatial.data[4];
        CUCOMPLEX *gzz = spatial.data[5];
        gxx[index] = gxy[index] = gxz[index] = CUFFT_ZERO;
        gyy[index] = gyz[index] = gzz[index] = CUFFT_ZERO;
        return;
    }
    const size_t i = index % sx;
    const size_t j = (index / sx) % sy;
    const size_t k = index / (sx * sy);
    CUCOMPLEX *gxx = spatial.data[0];
    CUCOMPLEX *gxy = spatial.data[1];
    CUCOMPLEX *gxz = spatial.data[2];
    CUCOMPLEX *gyy = spatial.data[3];
    CUCOMPLEX *gyz = spatial.data[4];
    CUCOMPLEX *gzz = spatial.data[5];
    if (i == size_t(nx) || j == size_t(ny) || k == size_t(nz)) {
        gxx[index] = gxy[index] = gxz[index] = CUFFT_ZERO;
        gyy[index] = gyz[index] = gzz[index] = CUFFT_ZERO;
        return;
    }
    const CU_REAL x = CU_REAL(i) * aretecube;
    const CU_REAL y = CU_REAL(j) * aretecube;
    const CU_REAL z = CU_REAL(k) * aretecube;
    if (medium) {
        CUCOMPLEX value[6];
        if (nquad > 0 && i * i + j * j + k * k <= nquad * nquad)
            green_tensor_integrated_value(x, y, z, aretecube, k0, eps0, 1, value);
        else
            green_tensor_medium_value(x, y, z, aretecube, k0, eps0, value);
        gxx[index] = value[0];
        gxy[index] = value[1];
        gxz[index] = value[2];
        gyy[index] = value[3];
        gyz[index] = value[4];
        gzz[index] = value[5];
        return;
    }
    if (nquad > 0 && i * i + j * j + k * k <= nquad * nquad) {
        CUCOMPLEX value[6];
        green_tensor_integrated_value(x, y, z, aretecube, k0, eps0, 0, value);
        gxx[index] = value[0];
        gxy[index] = value[1];
        gxz[index] = value[2];
        gyy[index] = value[3];
        gyz[index] = value[4];
        gzz[index] = value[5];
        return;
    }
    const CU_REAL r2 = x * x + y * y + z * z;
    if (r2 == CU_REAL(0)) {
        gxx[index] = gxy[index] = gxz[index] = CUFFT_ZERO;
        gyy[index] = gyz[index] = gzz[index] = CUFFT_ZERO;
        return;
    }
    const CU_REAL r = sqrt(r2);
    if (filtered) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        // Match the original Fortran USE_SIMP normalization: evaluating the
        // filtered radial derivatives directly in SI units can overflow float
        // intermediates (1/r^3 times derivative terms) even when the final
        // tensor is finite. Work in k0-scaled coordinates and rescale the
        // six tensor components only after cancellation of those terms.
        const CU_REAL physical_k0 = k0;
        const CU_REAL normalized_cell = aretecube * physical_k0;
        const CU_REAL x = CU_REAL(i) * normalized_cell;
        const CU_REAL y = CU_REAL(j) * normalized_cell;
        const CU_REAL z = CU_REAL(k) * normalized_cell;
        const CU_REAL r2 = x*x + y*y + z*z;
        const CU_REAL r = sqrt(r2);
        const CU_REAL aretecube = normalized_cell;
        const CU_REAL k0 = CU_REAL(1);
#endif
        const CU_REAL pi = CU_REAL(3.14159265358979323846);
        const CU_REAL kf = pi / aretecube, kr = k0 * r, kp = (kf + k0) * r, km = (kf - k0) * r;
        CU_REAL cip, sip, cim, sim;
        green_tensor_cisi(kp, &cip, &sip);
        green_tensor_cisi(km, &cim, &sim);
        const CU_REAL alpha = sin(kr) * (cip - cim) + cos(kr) * (pi - sip - sim);
        const CU_REAL alphap = k0 * sin(kr) * (-pi + sim + sip) + k0 * cos(kr) * (cip - cim) -
                               CU_REAL(2) * sin(kf * r) / r;
        const CU_REAL alphapp = k0 * k0 * sin(kr) * (cim - cip) +
                                k0 * k0 * cos(kr) * (sim + sip - pi) +
                                CU_REAL(2) * sin(kf * r) / r2 - CU_REAL(2) * kf * cos(kf * r) / r;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CUCOMPLEX imaginary = make_cuFloatComplex(0.0f, 1.0f);
        const CUCOMPLEX const1 = make_cuFloatComplex(cos(kr) / r, sin(kr) / r);
#else
        const CUCOMPLEX imaginary = make_cuDoubleComplex(0.0, 1.0);
        const CUCOMPLEX const1 = make_cuDoubleComplex(cos(kr) / r, sin(kr) / r);
#endif
        const CU_REAL hh = (sin(kf * r) - kf * r * cos(kf * r)) / (r2 * r * CU_REAL(2) * pi * pi);
        const CUCOMPLEX gf =
            CUFFT_MUL(CUFFT_SUB(const1, CUFFT_REAL(alpha / (pi * r))), CUFFT_REAL(k0 * k0));
        const CUCOMPLEX gfp =
            CUFFT_ADD(CUFFT_SUB(CUFFT_MUL(CUFFT_MUL(imaginary, CUFFT_REAL(k0)), const1),
                                CUFFT_MUL(CUFFT_REAL(CU_REAL(1) / r), const1)),
                      CUFFT_REAL(alpha / (pi * r2) - alphap / (pi * r)));
        const CUCOMPLEX gfpp = CUFFT_ADD(
            CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(-k0 * k0), const1),
                      CUFFT_MUL(CUFFT_REAL(-CU_REAL(2) * k0 / r), CUFFT_MUL(imaginary, const1))),
            CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(2) / r2), const1),
                      CUFFT_REAL(-CU_REAL(2) * alpha / (r2 * r * pi) +
                                 CU_REAL(2) * alphap / (r2 * pi) - alphapp / (r * pi))));
        const CUCOMPLEX diagonal =
            CUFFT_ADD(CUFFT_ADD(gf, CUFFT_MUL(CUFFT_REAL(CU_REAL(1) / r), gfp)),
                      CUFFT_REAL(CU_REAL(4) * pi * hh / CU_REAL(3)));
        const CUCOMPLEX radial = CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(-CU_REAL(1) / (r2 * r)), gfp),
                                           CUFFT_MUL(CUFFT_REAL(CU_REAL(1) / r2), gfpp));
        gxx[index] = CUFFT_ADD(diagonal, CUFFT_MUL(CUFFT_REAL(x * x), radial));
        gxy[index] = CUFFT_MUL(CUFFT_REAL(x * y), radial);
        gxz[index] = CUFFT_MUL(CUFFT_REAL(x * z), radial);
        gyy[index] = CUFFT_ADD(diagonal, CUFFT_MUL(CUFFT_REAL(y * y), radial));
        gyz[index] = CUFFT_MUL(CUFFT_REAL(y * z), radial);
        gzz[index] = CUFFT_ADD(diagonal, CUFFT_MUL(CUFFT_REAL(z * z), radial));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CUCOMPLEX restore_units = CUFFT_REAL(physical_k0 * physical_k0 * physical_k0);
        gxx[index] = CUFFT_MUL(gxx[index], restore_units);
        gxy[index] = CUFFT_MUL(gxy[index], restore_units);
        gxz[index] = CUFFT_MUL(gxz[index], restore_units);
        gyy[index] = CUFFT_MUL(gyy[index], restore_units);
        gyz[index] = CUFFT_MUL(gyz[index], restore_units);
        gzz[index] = CUFFT_MUL(gzz[index], restore_units);
#endif
        return;
    }
    const CU_REAL inv_r2 = CU_REAL(1) / r2;
    const CU_REAL angle = k0 * r;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cosf(angle), sinf(angle));
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(angle), sin(angle));
#endif
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX c1full = make_cuFloatComplex(CU_REAL(1) / (r * r2), -k0 * inv_r2);
#else
    const CUCOMPLEX c1full = make_cuDoubleComplex(CU_REAL(1) / (r * r2), -k0 * inv_r2);
#endif
    const CUCOMPLEX c2 = CUFFT_REAL(k0 * k0 / r);
    const CU_REAL qx = x * x * inv_r2;
    const CU_REAL qy = y * y * inv_r2;
    const CU_REAL qz = z * z * inv_r2;
    const CU_REAL qxy = x * y * inv_r2;
    const CU_REAL qxz = x * z * inv_r2;
    const CU_REAL qyz = y * z * inv_r2;
    const CUCOMPLEX a_xx = CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qx - CU_REAL(1)), c1full);
    const CUCOMPLEX b_xx = CUFFT_MUL(CUFFT_REAL(CU_REAL(1) - qx), c2);
    const CUCOMPLEX a_xy = CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qxy), c1full);
    const CUCOMPLEX b_xy = CUFFT_MUL(CUFFT_REAL(-qxy), c2);
    const CUCOMPLEX a_xz = CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qxz), c1full);
    const CUCOMPLEX b_xz = CUFFT_MUL(CUFFT_REAL(-qxz), c2);
    const CUCOMPLEX a_yy = CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qy - CU_REAL(1)), c1full);
    const CUCOMPLEX b_yy = CUFFT_MUL(CUFFT_REAL(CU_REAL(1) - qy), c2);
    const CUCOMPLEX a_yz = CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qyz), c1full);
    const CUCOMPLEX b_yz = CUFFT_MUL(CUFFT_REAL(-qyz), c2);
    const CUCOMPLEX a_zz = CUFFT_MUL(CUFFT_REAL(CU_REAL(3) * qz - CU_REAL(1)), c1full);
    const CUCOMPLEX b_zz = CUFFT_MUL(CUFFT_REAL(CU_REAL(1) - qz), c2);
    gxx[index] = CUFFT_MUL(CUFFT_ADD(a_xx, b_xx), phase);
    gxy[index] = CUFFT_MUL(CUFFT_ADD(a_xy, b_xy), phase);
    gxz[index] = CUFFT_MUL(CUFFT_ADD(a_xz, b_xz), phase);
    gyy[index] = CUFFT_MUL(CUFFT_ADD(a_yy, b_yy), phase);
    gyz[index] = CUFFT_MUL(CUFFT_ADD(a_yz, b_yz), phase);
    gzz[index] = CUFFT_MUL(CUFFT_ADD(a_zz, b_zz), phase);
}

//-----------------------------------------------------------
__global__ void green_tensor_embed_diagonal_kernel(GreenTensorDevicePointers spatial, GreenTensorDevicePointers spectra,
                                                   int nx, int ny, int nz, int nxm, int nym,
                                                   int nzm, int nx2, int ny2, int nz2,
                                                   size_t tensor_count, size_t fft_count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= fft_count)
        return;
    const int i = int(index % size_t(nx2));
    const int j = int((index / size_t(nx2)) % size_t(ny2));
    const int k = int(index / (size_t(nx2) * size_t(ny2)));
    const int mi = i <= nx ? i : nx2 - i;
    const int mj = j <= ny ? j : ny2 - j;
    const int mk = k <= nz ? k : nz2 - k;
    const size_t source =
        size_t(mi) + (size_t(nx) + 1) * (size_t(mj) + (size_t(ny) + 1) * size_t(mk));
    spectra.data[0][index] = spatial.data[0][source];
    spectra.data[1][index] = spatial.data[3][source];
    spectra.data[2][index] = spatial.data[5][source];
}

//-----------------------------------------------------------
__global__ void green_tensor_embed_offdiagonal_kernel(GreenTensorDevicePointers spatial, GreenTensorDevicePointers spectra,
                                                      int nx, int ny, int nz, int nxm, int nym,
                                                      int nzm, int nx2, int ny2, int nz2,
                                                      size_t tensor_count, size_t fft_count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= fft_count)
        return;
    const int i = int(index % size_t(nx2));
    const int j = int((index / size_t(nx2)) % size_t(ny2));
    const int k = int(index / (size_t(nx2) * size_t(ny2)));
    const int mi = i <= nx ? i : nx2 - i;
    const int mj = j <= ny ? j : ny2 - j;
    const int mk = k <= nz ? k : nz2 - k;
    const CU_REAL signx = i <= nx ? CU_REAL(1) : CU_REAL(-1);
    const CU_REAL signy = j <= ny ? CU_REAL(1) : CU_REAL(-1);
    const CU_REAL signz = k <= nz ? CU_REAL(1) : CU_REAL(-1);
    const size_t source =
        size_t(mi) + (size_t(nx) + 1) * (size_t(mj) + (size_t(ny) + 1) * size_t(mk));
    spectra.data[0][index] = CUFFT_MUL(CUFFT_REAL(signx * signy), spatial.data[1][source]);
    spectra.data[1][index] =
        CUFFT_MUL(CUFFT_REAL(signx * signz), spatial.data[2][source]);
    spectra.data[2][index] =
        CUFFT_MUL(CUFFT_REAL(signy * signz), spatial.data[4][source]);
}

//-----------------------------------------------------------
__global__ void green_tensor_store_spectra_kernel(GreenTensorDevicePointers spectra, GreenTensorDevicePointers green,
                                                  int nx, int ny, int nz, int nxm, int nym, int nx2,
                                                  int ny2, int nz2, size_t tensor_count,
                                                  size_t fft_count, int component_a,
                                                  int component_b, int component_c) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tensor_count)
        return;
    const size_t sx = size_t(nx) + 1;
    const size_t sy = size_t(ny) + 1;
    const size_t active_count = sx * sy * (size_t(nz) + 1);
    if (index >= active_count)
        return;
    const size_t i = index % sx;
    const size_t j = (index / sx) % sy;
    const size_t k = index / (sx * sy);
    if (i > size_t(nx) || j > size_t(ny) || k > size_t(nz))
        return;
    const size_t source = i + size_t(nx2) * (j + size_t(ny2) * k);
    green.data[component_a][index] = spectra.data[0][source];
    green.data[component_b][index] = spectra.data[1][source];
    green.data[component_c][index] = spectra.data[2][source];
}

//-----------------------------------------------------------
__global__ void multilayer_vector_prepare_kernel(const CUCOMPLEX *ix, const CUCOMPLEX *iy,
                                                 const CUCOMPLEX *iz, const CU_INT*map,
                                                 const CUCOMPLEX *alpha, CUCOMPLEX *px,
                                                 CUCOMPLEX *py, CUCOMPLEX *pz, int nfft, int nx,
                                                 int ny, int layer, int count) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count)
        return;
    px[i] = py[i] = pz[i] = CUFFT_ZERO;
    const int side = 2 * nfft, x = i % side, y = i / side;
    if (x >= nfft || y >= nfft)
        return;
    const int p = x + y * nfft, local = map[p].dat;
    if (local == 0)
        return;
    const int dipole = nx * ny * layer + local - 1;
    const CUCOMPLEX a = alpha[dipole];
    px[i] = CUFFT_MUL(ix[p], a);
    py[i] = CUFFT_MUL(iy[p], a);
    pz[i] = CUFFT_MUL(iz[p], a);
}

//-----------------------------------------------------------
__global__ void multilayer_vector_prepare_reflected_kernel(
    const CUCOMPLEX *fx, const CUCOMPLEX *fy, const CUCOMPLEX *fz, const CUCOMPLEX *rx,
    const CUCOMPLEX *ry, const CUCOMPLEX *rz, const CU_INT*map, const CUCOMPLEX *alpha, CUCOMPLEX *px,
    CUCOMPLEX *py, CUCOMPLEX *pz, int nfft, int nx, int ny, int layer, int count) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count)
        return;
    px[i] = py[i] = pz[i] = CUFFT_ZERO;
    const int side = 2 * nfft, x = i % side, y = i / side;
    if (x >= nfft || y >= nfft)
        return;
    const int p = x + y * nfft, local = map[p].dat;
    if (!local)
        return;
    const CUCOMPLEX a = alpha[nx * ny * layer + local - 1];
    px[i] = CUFFT_MUL(CUFFT_ADD(fx[p], rx[p]), a);
    py[i] = CUFFT_MUL(CUFFT_ADD(fy[p], ry[p]), a);
    pz[i] = CUFFT_MUL(CUFFT_ADD(fz[p], rz[p]), a);
}

//-----------------------------------------------------------
__global__ void multilayer_vector_product_kernel(const CUCOMPLEX *px, const CUCOMPLEX *py,
                                                 const CUCOMPLEX *pz, const CUCOMPLEX *gxx,
                                                 const CUCOMPLEX *gxy, const CUCOMPLEX *gxz,
                                                 const CUCOMPLEX *gyy, const CUCOMPLEX *gyz,
                                                 const CUCOMPLEX *gzz, CUCOMPLEX *fx, CUCOMPLEX *fy,
                                                 CUCOMPLEX *fz, int count, int reflected) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count)
        return;
    CUCOMPLEX xz = CUFFT_MUL(pz[i], gxz[i]), yz = CUFFT_MUL(pz[i], gyz[i]);
    CUCOMPLEX xx = CUFFT_MUL(px[i], gxx[i]), xy = CUFFT_MUL(py[i], gxy[i]);
    CUCOMPLEX yy = CUFFT_MUL(py[i], gyy[i]), xz2 = CUFFT_MUL(px[i], gxz[i]);
    CUCOMPLEX yz2 = CUFFT_MUL(py[i], gyz[i]), zz = CUFFT_MUL(pz[i], gzz[i]);
    if (reflected) {
        xz = CUFFT_MUL(CUFFT_REAL(-1), xz);
        yz = CUFFT_MUL(CUFFT_REAL(-1), yz);
        xz2 = CUFFT_MUL(CUFFT_REAL(-1), xz2);
        yz2 = CUFFT_MUL(CUFFT_REAL(-1), yz2);
    }
    fx[i] = CUFFT_ADD(CUFFT_ADD(xx, xy), xz);
    fy[i] = CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(px[i], gxy[i]), yy), yz);
    fz[i] = CUFFT_ADD(CUFFT_ADD(xz2, yz2), zz);
}

//-----------------------------------------------------------
__global__ void multilayer_vector_accumulate_kernel(
    CUCOMPLEX *ix, CUCOMPLEX *iy, CUCOMPLEX *iz, const CUCOMPLEX *fx, const CUCOMPLEX *fy,
    const CUCOMPLEX *fz, CUCOMPLEX *local, const CU_INT*map, int nfft, int nx, int ny, int layer,
    int nmax, int count, CU_REAL image_scale, CU_REAL field_scale, int add_local) {
    const int p = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= count)
        return;
    const int x = p % nfft, y = p / nfft;
    if (y >= nfft)
        return;
    const int image = p, padded = x + y * (2 * nfft);
    ix[image] = CUFFT_ADD(CUFFT_MUL(ix[image], CUFFT_REAL(image_scale)),
                          CUFFT_MUL(fx[padded], CUFFT_REAL(field_scale)));
    iy[image] = CUFFT_ADD(CUFFT_MUL(iy[image], CUFFT_REAL(image_scale)),
                          CUFFT_MUL(fy[padded], CUFFT_REAL(field_scale)));
    iz[image] = CUFFT_ADD(CUFFT_MUL(iz[image], CUFFT_REAL(image_scale)),
                          CUFFT_MUL(fz[padded], CUFFT_REAL(field_scale)));
    const int local_dipole = map[image].dat;
    if (!local_dipole)
        return;
    const int dipole = nx * ny * layer + local_dipole - 1;
    const int base = 3 * dipole;
    if (add_local) {
        local[base] = CUFFT_ADD(local[base], ix[image]);
        local[base + 1] = CUFFT_ADD(local[base + 1], iy[image]);
        local[base + 2] = CUFFT_ADD(local[base + 2], iz[image]);
    } else {
        local[base] = ix[image];
        local[base + 1] = iy[image];
        local[base + 2] = iz[image];
    }
    (void)nmax;
}

//-----------------------------------------------------------
__global__ void multilayer_scalar_prepare_polarization_kernel(
    const CUCOMPLEX *image, const CUCOMPLEX *add_image, const CU_INT*plane_map,
    const CUCOMPLEX *polarizability, CUCOMPLEX *polarization, int nfft2d, int nx, int ny, int layer,
    int plane_count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= plane_count)
        return;
    const int side = 2 * nfft2d;
    polarization[index] = CUFFT_ZERO;
    const int x = index % side;
    const int y = index / side;
    if (x >= nfft2d || y >= nfft2d)
        return;
    const int image_index = x + y * nfft2d;
    const int local_dipole = plane_map[image_index].dat;
    if (local_dipole == 0)
        return;
    const int global_dipole = nx * ny * layer + local_dipole - 1;
    CUCOMPLEX field = image[image_index];
    if (add_image)
        field = CUFFT_ADD(field, add_image[image_index]);
    polarization[index] = CUFFT_MUL(field, polarizability[global_dipole]);
}

//-----------------------------------------------------------
__global__ void multilayer_scalar_map_first_plane_kernel(const CU_REAL *xs, const CU_REAL *ys,
                                                         const CU_REAL *zs, CU_INT*plane_map,
                                                         int ndipole, int nfft2d, int kkm, int jjm,
                                                         CU_REAL xmin, CU_REAL ymin,
                                                         CU_REAL first_z, CU_REAL aretecube) {
    const int dipole = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (dipole >= ndipole)
        return;
    const CU_REAL tolerance = aretecube / CU_REAL(1000);
    if (fabs(zs[dipole] - first_z) > tolerance)
        return;
    const long long ix_offset = llround((xs[dipole] - xmin) / aretecube);
    const long long iy_offset = llround((ys[dipole] - ymin) / aretecube);
    const long long ix = ix_offset + kkm;
    const long long iy = iy_offset + jjm;
    if (ix < 0 || ix >= nfft2d || iy < 0 || iy >= nfft2d)
        return;
    const CU_REAL expected_x = xmin + CU_REAL(ix_offset) * aretecube;
    const CU_REAL expected_y = ymin + CU_REAL(iy_offset) * aretecube;
    if (fabs(xs[dipole] - expected_x) > tolerance || fabs(ys[dipole] - expected_y) > tolerance)
        return;
    atomicExch(&plane_map[int(ix) + nfft2d * int(iy)].dat, dipole + 1);
}

struct BeamPropagationIncidentField {
    CUCOMPLEX x, y, z;
};

__device__ BeamPropagationIncidentField
beampropagation_plane_field(CUCOMPLEX E0, CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees,
                            CU_REAL phi_degrees, CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL k0,
                            CU_REAL xgaus, CU_REAL ygaus, CU_REAL zgaus, CU_REAL w0, int beam_type);

// Match gaussianparalinear/gaussianparacirc from gaussianpara.f90, including
// the complex refractive index of the embedding medium.
//-----------------------------------------------------------
__device__ BeamPropagationIncidentField multilayer_paraxial_gaussian_field(
    CUCOMPLEX E0, CUCOMPLEX eps0, CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees,
    CU_REAL phi_degrees, CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL x0, CU_REAL y0, CU_REAL z0,
    CU_REAL w0, CU_REAL k0, int circular) {
    const CU_REAL pi = CU_REAL(3.14159265358979323846);
    const CU_REAL theta = theta_degrees * pi / CU_REAL(180);
    const CU_REAL phi = phi_degrees * pi / CU_REAL(180);
    const CU_REAL st = sin(theta), ct = cos(theta);
    const CU_REAL sp = sin(phi), cp = cos(phi);
    const CU_REAL vx = st * cp, vy = st * sp, vz = ct;
    CU_REAL m00 = CU_REAL(1), m01 = CU_REAL(0), m02 = CU_REAL(0);
    CU_REAL m10 = CU_REAL(0), m11 = CU_REAL(1), m12 = CU_REAL(0);
    CU_REAL m20 = CU_REAL(0), m21 = CU_REAL(0), m22 = CU_REAL(1);
    if (vx * vx + vy * vy > CU_REAL(1.0e-12)) {
        m00 = CU_REAL(1) - vx * vx / (CU_REAL(1) + vz);
        m01 = -vx * vy / (CU_REAL(1) + vz);
        m02 = -vx;
        m10 = m01;
        m11 = CU_REAL(1) - vy * vy / (CU_REAL(1) + vz);
        m12 = -vy;
        m20 = vx;
        m21 = vy;
        m22 = vz;
    }
    CUCOMPLEX e0s, e0p;
    if (circular) {
        const CU_REAL us = CU_REAL(1) / sqrt(CU_REAL(2));
        e0s = CUFFT_MUL(E0, CUFFT_REAL(us));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        e0p = CUFFT_MUL(E0, make_cuFloatComplex(CU_REAL(0), ss < 0 ? -us : us));
#else
        e0p = CUFFT_MUL(E0, make_cuDoubleComplex(CU_REAL(0), ss < 0 ? -us : us));
#endif
    } else {
        e0s = CUFFT_MUL(E0, CUFFT_REAL(sqrt(CU_REAL(1) - pp)));
        e0p = CUFFT_MUL(E0, CUFFT_REAL(sqrt(pp)));
    }
    const CUCOMPLEX e0x =
        CUFFT_ADD(CUFFT_MUL(e0s, CUFFT_REAL(-sp)), CUFFT_MUL(e0p, CUFFT_REAL(ct * cp)));
    const CUCOMPLEX e0y =
        CUFFT_ADD(CUFFT_MUL(e0s, CUFFT_REAL(cp)), CUFFT_MUL(e0p, CUFFT_REAL(ct * sp)));
    const CUCOMPLEX e0z = CUFFT_MUL(e0p, CUFFT_REAL(-st));
    const CUCOMPLEX e00x =
        CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(e0x, CUFFT_REAL(m00)), CUFFT_MUL(e0y, CUFFT_REAL(m01))),
                  CUFFT_MUL(e0z, CUFFT_REAL(m02)));
    const CUCOMPLEX e00y =
        CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(e0x, CUFFT_REAL(m10)), CUFFT_MUL(e0y, CUFFT_REAL(m11))),
                  CUFFT_MUL(e0z, CUFFT_REAL(m12)));
    const CU_REAL rx = x - x0, ry = y - y0, rz = z - z0;
    const CU_REAL xx = m00 * rx + m01 * ry + m02 * rz;
    const CU_REAL yy = m10 * rx + m11 * ry + m12 * rz;
    const CU_REAL zz = m20 * rx + m21 * ry + m22 * rz;
    const CUCOMPLEX index = green_tensor_complex_sqrt(eps0);
    const CU_REAL waist2 = w0 * w0;
    const CU_REAL rayleigh = k0 * waist2 * index.x;
    const CU_REAL width2 = CU_REAL(2) * waist2 * (CU_REAL(1) + zz * zz / (rayleigh * rayleigh));
    const CU_REAL width = sqrt(width2);
    const CU_REAL eta = atan2(zz, rayleigh);
    const CU_REAL rinv = zz / (zz * zz + rayleigh * rayleigh);
    const CU_REAL r2 = xx * xx + yy * yy;
    const CU_REAL amplitude = sqrt(CU_REAL(2)) * w0 / width * exp(-r2 / width2);
    const CUCOMPLEX phase_argument = CUFFT_ADD(
        CUFFT_MUL(index, CUFFT_REAL(k0 * zz + k0 * r2 * rinv / CU_REAL(2))), CUFFT_REAL(eta));
    const CUCOMPLEX phase = CUFFT_MUL(CUFFT_REAL(amplitude), green_tensor_exp_i(phase_argument));
    const CUCOMPLEX epx = CUFFT_MUL(e00x, phase), epy = CUFFT_MUL(e00y, phase);
    return {CUFFT_ADD(CUFFT_MUL(epx, CUFFT_REAL(m00)), CUFFT_MUL(epy, CUFFT_REAL(m10))),
            CUFFT_ADD(CUFFT_MUL(epx, CUFFT_REAL(m01)), CUFFT_MUL(epy, CUFFT_REAL(m11))),
            CUFFT_ADD(CUFFT_MUL(epx, CUFFT_REAL(m02)), CUFFT_MUL(epy, CUFFT_REAL(m12)))};
}

//-----------------------------------------------------------
__global__ void incident_paraxial_gaussian_kernel(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, CUCOMPLEX *field, int count,
    CUCOMPLEX e0, CUCOMPLEX epsilon, CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees,
    CU_REAL phi_degrees, CU_REAL x0, CU_REAL y0, CU_REAL z0, CU_REAL w0, CU_REAL k0,
    int circular, IfddaIncidentGrid grid) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    int i = flat;
    REAL x = 0, y = 0, z = 0;
    if (grid.tabdip) {
        if (!ifdda_incident_grid_fetch(grid, flat, &i, &x, &y, &z)) return;
    } else if (i < count) {
        x = xs[i]; y = ys[i]; z = zs[i];
    } else return;
    const BeamPropagationIncidentField value =
        multilayer_paraxial_gaussian_field(e0, epsilon, ss, pp, theta_degrees, phi_degrees,
                                           x, y, z, x0, y0, z0, w0, k0, circular);
    field[3 * i] = value.x;
    field[3 * i + 1] = value.y;
    field[3 * i + 2] = value.z;
}

//-----------------------------------------------------------
__global__ void gaussian_power_kernel(CUCOMPLEX* field, CU_REAL* irradiance,
                                     CU_REAL waist, CU_REAL power) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const CU_REAL c = CU_REAL(299792458);
    const CU_REAL pi_value = CU_REAL(3.1415926535897932384626433832795L);
    const CU_REAL epsilon0 = CU_REAL(1) /
        (c * c * CU_REAL(4) * pi_value * CU_REAL(1e-7));
    const CU_REAL amplitude = sqrt(CU_REAL(2) * power /
        (pi_value * c * epsilon0 * waist * waist));
    field[0] = CUFFT_REAL(amplitude);
    irradiance[0] = epsilon0 * c * amplitude * amplitude / CU_REAL(2);
}

//-----------------------------------------------------------
__device__ void wofz_real_positive(CU_REAL x, CU_REAL* real_part,
                                   CU_REAL* imag_part) {
    const CU_REAL factor = CU_REAL(1.1283791670955125739L);
    const CU_REAL qrho0 = (x / CU_REAL(6.3)) * (x / CU_REAL(6.3));
    if (qrho0 < CU_REAL(0.085264)) {
        const CU_REAL qrho = x / CU_REAL(6.3);
        const int n = int(floor(CU_REAL(6) + CU_REAL(72) * qrho + CU_REAL(0.5)));
        int j = 2 * n + 1;
        CU_REAL xsum = CU_REAL(1) / CU_REAL(j);
        CU_REAL ysum = CU_REAL(0);
        const CU_REAL xquad = x * x;
        for (int i = n; i >= 1; --i) {
            j -= 2;
            const CU_REAL xaux = (xsum * xquad) / CU_REAL(i);
            ysum = (ysum * xquad) / CU_REAL(i);
            xsum = xaux + CU_REAL(1) / CU_REAL(j);
        }
        const CU_REAL v1 = factor * xsum * x;
        const CU_REAL exponential = exp(-xquad);
        *real_part = exponential;
        *imag_part = v1 * exponential;
        return;
    }

    CU_REAL qrho = sqrt(qrho0);
    CU_REAL h = CU_REAL(0);
    int kapn = 0;
    int nu = 0;
    if (qrho0 >= CU_REAL(1)) {
        nu = int(CU_REAL(3) + CU_REAL(1442) /
                 (CU_REAL(26) * qrho + CU_REAL(77)));
    } else {
        qrho = sqrt(CU_REAL(1) - qrho0);
        h = CU_REAL(1.88) * qrho;
        const CU_REAL h2 = CU_REAL(2) * h;
        kapn = int(floor(CU_REAL(7) + CU_REAL(34) * qrho + CU_REAL(0.5)));
        nu = int(floor(CU_REAL(16) + CU_REAL(26) * qrho + CU_REAL(0.5)));
        CU_REAL qlambda = pow(h2, CU_REAL(kapn));
        CU_REAL rx = CU_REAL(0), ry = CU_REAL(0);
        CU_REAL sx = CU_REAL(0), sy = CU_REAL(0);
        for (int n = nu; n >= 0; --n) {
            const int np1 = n + 1;
            const CU_REAL tx = h + CU_REAL(np1) * rx;
            const CU_REAL ty = x - CU_REAL(np1) * ry;
            const CU_REAL cc = CU_REAL(0.5) / (tx * tx + ty * ty);
            rx = cc * tx;
            ry = cc * ty;
            if (n <= kapn) {
                const CU_REAL tx_sum = qlambda + sx;
                sx = rx * tx_sum - ry * sy;
                sy = ry * tx_sum + rx * sy;
                qlambda /= h2;
            }
        }
        *real_part = exp(-x * x);
        *imag_part = factor * sy;
        return;
    }

    CU_REAL rx = CU_REAL(0), ry = CU_REAL(0);
    for (int n = nu; n >= 0; --n) {
        const int np1 = n + 1;
        const CU_REAL tx = CU_REAL(np1) * rx;
        const CU_REAL ty = x - CU_REAL(np1) * ry;
        const CU_REAL cc = CU_REAL(0.5) / (tx * tx + ty * ty);
        rx = cc * tx;
        ry = cc * ty;
    }
    *real_part = exp(-x * x);
    *imag_part = factor * ry;
}

//-----------------------------------------------------------
__global__ void rigorous_gaussian_power_kernel(
    CUCOMPLEX* field, CU_REAL* irradiance, CU_REAL waist, CU_REAL k0, CU_REAL power) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const CU_REAL c = CU_REAL(299792458);
    const CU_REAL pi_value = CU_REAL(3.1415926535897932384626433832795L);
    const CU_REAL epsilon0 = CU_REAL(1) /
        (c * c * CU_REAL(4) * pi_value * CU_REAL(1e-7));
    const CU_REAL argument = k0 * waist;
    CU_REAL w_real = CU_REAL(0), w_imag = CU_REAL(0);
    wofz_real_positive(argument, &w_real, &w_imag);
    const CU_REAL constant0 = pi_value / CU_REAL(4) * waist * waist * c * epsilon0;
    const CU_REAL constant = constant0 * (CU_REAL(1) +
        (argument * argument - CU_REAL(1)) / argument * sqrt(pi_value) /
            CU_REAL(2) * w_imag);
    const CU_REAL gaussian_intensity = power / constant;
    const CU_REAL amplitude = sqrt(gaussian_intensity);

    wofz_real_positive(argument / sqrt(CU_REAL(2)), &w_real, &w_imag);
    const CU_REAL factor = sqrt(CU_REAL(2) * pi_value) *
        (argument * argument - CU_REAL(1)) / argument / CU_REAL(4);
    *irradiance = c * epsilon0 / CU_REAL(2) *
        (CU_REAL(0.5) + factor * w_imag) * gaussian_intensity;
    field[0] = CUFFT_REAL(amplitude);
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t gaussian_power_wrapper(
    CUCOMPLEX* field, CU_REAL* irradiance, CU_REAL waist, CU_REAL power) {
    if (!field || !irradiance || waist <= CU_REAL(0) || power < CU_REAL(0))
        return cudaErrorInvalidValue;
    gaussian_power_kernel<<<1, 1>>>(field, irradiance, waist, power);
    return finish_kernel_launch(205, "gaussian_power_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t rigorous_gaussian_power_wrapper(
    CUCOMPLEX* field, CU_REAL* irradiance, CU_REAL waist, CU_REAL k0, CU_REAL power) {
    if (!field || !irradiance || waist <= CU_REAL(0) || k0 <= CU_REAL(0) ||
        power < CU_REAL(0))
        return cudaErrorInvalidValue;
    rigorous_gaussian_power_kernel<<<1, 1>>>(field, irradiance, waist, k0, power);
    return finish_kernel_launch(206, "rigorous_gaussian_power_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_paraxial_gaussian_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, CUCOMPLEX *field, int count,
    COMPLEX e0, COMPLEX epsilon, CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees,
    CU_REAL phi_degrees, CU_REAL x0, CU_REAL y0, CU_REAL z0, CU_REAL w0, CU_REAL k0, int circular) {
    if (!xs || !ys || !zs || !field || count < 0 || w0 <= CU_REAL(0) || k0 <= CU_REAL(0) ||
        (!circular && (pp < CU_REAL(0) || pp > CU_REAL(1))))
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    incident_paraxial_gaussian_kernel<<<(count + 255) / 256, 256>>>(
        xs, ys, zs, field, count, plane_wave_device_scalar(e0), plane_wave_device_scalar(epsilon),
        ss, pp, theta_degrees, phi_degrees, x0, y0, z0, w0,
        k0, circular, {});
    return finish_kernel_launch(202, "incident_paraxial_gaussian_kernel");
}

extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_paraxial_gaussian_grid_wrapper(
    IfddaIncidentGrid grid, CUCOMPLEX* field, COMPLEX e0, COMPLEX epsilon,
    REAL ss, REAL pp, REAL theta_degrees, REAL phi_degrees,
    REAL x0, REAL y0, REAL z0, REAL w0, REAL k0, int circular) {
    if (!grid.tabdip || !field || grid.grid_count <= 0 ||
        grid.dipole_count <= 0 || w0 <= REAL(0) || k0 <= REAL(0) ||
        (!circular && (pp < REAL(0) || pp > REAL(1)))) return cudaErrorInvalidValue;
    incident_paraxial_gaussian_kernel<<<(grid.grid_count + 255) / 256, 256>>>(
        nullptr, nullptr, nullptr, field, grid.dipole_count,
        plane_wave_device_scalar(e0), plane_wave_device_scalar(epsilon),
        ss, pp, theta_degrees, phi_degrees, x0, y0, z0, w0, k0, circular, grid);
    return finish_kernel_launch(202, "incident_paraxial_gaussian_grid_kernel");
}

//-----------------------------------------------------------
__device__ CUCOMPLEX gaussian_derivative_complex(CU_REAL real_part, CU_REAL imag_part) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real_part, imag_part);
#else
    return make_cuDoubleComplex(real_part, imag_part);
#endif
}

//-----------------------------------------------------------
__device__ void gaussianparad_derivative_point(
    CUCOMPLEX* output, CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL waist,
    CU_REAL k0, CUCOMPLEX e0x, CUCOMPLEX e0y, const CU_REAL* rotation) {
    const CU_REAL w02 = waist * waist;
    const CU_REAL z0 = k0 * w02;
    const CU_REAL z02 = z0 * z0;
    const CU_REAL r2 = x * x + y * y;
    const CU_REAL den = z * z + z02;
    const CU_REAL w2 = CU_REAL(2) * w02 * (CU_REAL(1) + z * z / z02);
    const CU_REAL w = sqrt(w2);
    const CU_REAL eta = atan2(z, z0);
    const CU_REAL rinv = z / den;
    const CU_REAL amp = sqrt(CU_REAL(2)) * waist / w * exp(-r2 / w2);
    const CU_REAL total_phase = k0 * r2 * rinv / CU_REAL(2) + k0 * z + eta;
    const CUCOMPLEX constant = gaussian_derivative_complex(
        amp * cos(total_phase), amp * sin(total_phase));
    const CUCOMPLEX transverse_factor =
        gaussian_derivative_complex(-CU_REAL(2) / w2, k0 * rinv);
    const CUCOMPLEX longitudinal_factor = CUFFT_ADD(
        CUFFT_ADD(gaussian_derivative_complex(CU_REAL(0), k0),
                  gaussian_derivative_complex(-z / den, z0 / den)),
        gaussian_derivative_complex(z * k0 * k0 * w02 * r2 / (den * den),
                                    k0 * (z02 - z * z) * r2 /
                                        (CU_REAL(2) * den * den)));
    const CUCOMPLEX fx = CUFFT_MUL(constant,
                                   CUFFT_MUL(transverse_factor, CUFFT_REAL(x)));
    const CUCOMPLEX fy = CUFFT_MUL(constant,
                                   CUFFT_MUL(transverse_factor, CUFFT_REAL(y)));
    const CUCOMPLEX fz = CUFFT_MUL(constant, longitudinal_factor);
    CUCOMPLEX local[9];
    local[0] = CUFFT_MUL(e0x, fx);
    local[1] = CUFFT_MUL(e0y, fx);
    local[2] = CUFFT_ZERO;
    local[3] = CUFFT_MUL(e0x, fy);
    local[4] = CUFFT_MUL(e0y, fy);
    local[5] = CUFFT_ZERO;
    local[6] = CUFFT_MUL(e0x, fz);
    local[7] = CUFFT_MUL(e0y, fz);
    local[8] = CUFFT_ZERO;
    CUCOMPLEX right_rotated[9];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            CUCOMPLEX sum = CUFFT_ZERO;
            for (int k = 0; k < 3; ++k)
                sum = CUFFT_ADD(sum, CUFFT_MUL(local[row + 3 * k],
                                               CUFFT_REAL(rotation[k + 3 * col])));
            right_rotated[row + 3 * col] = sum;
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            CUCOMPLEX sum = CUFFT_ZERO;
            for (int k = 0; k < 3; ++k)
                sum = CUFFT_ADD(sum, CUFFT_MUL(CUFFT_REAL(rotation[k + 3 * row]),
                                               right_rotated[k + 3 * col]));
            output[row + 3 * col] = sum;
        }
    }
}

//-----------------------------------------------------------
__global__ void gaussianparad_derivative_kernel(
    CUCOMPLEX* output, CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL waist,
    CU_REAL k0, CUCOMPLEX e0x, CUCOMPLEX e0y, const CU_REAL* rotation) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    gaussianparad_derivative_point(output, x, y, z, waist, k0, e0x, e0y, rotation);
}

//-----------------------------------------------------------
__global__ void gaussianparad_derivative_batch_kernel(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    CUCOMPLEX* output, int count, CU_REAL x0, CU_REAL y0, CU_REAL z0,
    CU_REAL waist, CU_REAL k0, CUCOMPLEX e0x, CUCOMPLEX e0y,
    const CU_REAL* rotation) {
    const int point = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (point >= count)
        return;
    const CU_REAL rx = xs[point] - x0;
    const CU_REAL ry = ys[point] - y0;
    const CU_REAL rz = zs[point] - z0;
    const CU_REAL local_x = rotation[0] * rx + rotation[3] * ry + rotation[6] * rz;
    const CU_REAL local_y = rotation[1] * rx + rotation[4] * ry + rotation[7] * rz;
    const CU_REAL local_z = rotation[2] * rx + rotation[5] * ry + rotation[8] * rz;
    gaussianparad_derivative_point(output + size_t(point) * 9, local_x, local_y,
                                   local_z, waist, k0, e0x, e0y, rotation);
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t gaussianparad_derivative_wrapper(
    CUCOMPLEX* derivative, CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL waist,
    CU_REAL k0, CUCOMPLEX e0x, CUCOMPLEX e0y, const CU_REAL* rotation) {
    if (!derivative || !rotation || waist <= CU_REAL(0) || k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    gaussianparad_derivative_kernel<<<1, 1>>>(derivative, x, y, z, waist, k0,
                                               e0x, e0y, rotation);
    return finish_kernel_launch(207, "gaussianparad_derivative_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t gaussianparad_derivative_batch_wrapper(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, CUCOMPLEX* output,
    int count, CU_REAL x0, CU_REAL y0, CU_REAL z0, CU_REAL waist, CU_REAL k0,
    CUCOMPLEX e0x, CUCOMPLEX e0y, const CU_REAL* rotation) {
    if (!xs || !ys || !zs || !output || !rotation || count < 0 ||
        waist <= CU_REAL(0) || k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    gaussianparad_derivative_batch_kernel<<<(count + 255) / 256, 256>>>(
        xs, ys, zs, output, count, x0, y0, z0, waist, k0, e0x, e0y, rotation);
    return finish_kernel_launch(263, "gaussianparad_derivative_batch_kernel");
}

//-----------------------------------------------------------
__device__ CUCOMPLEX antenna_complex(CU_REAL real_part, CU_REAL imag_part) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real_part, imag_part);
#else
    return make_cuDoubleComplex(real_part, imag_part);
#endif
}

//-----------------------------------------------------------
__device__ CUCOMPLEX antenna_field_at_point(CU_REAL x, CU_REAL y, CU_REAL z, CU_REAL xdip,
                                            CU_REAL ydip, CU_REAL zdip, CU_REAL aretecube,
                                            CU_REAL k0, CUCOMPLEX e0, CU_REAL theta_degrees,
                                            CU_REAL phi_degrees, int component) {
    const CU_REAL pi = CU_REAL(3.14159265358979323846);
    const CU_REAL theta = theta_degrees * pi / CU_REAL(180);
    const CU_REAL phi = phi_degrees * pi / CU_REAL(180);
    const CU_REAL p[3] = {sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)};
    const CU_REAL r[3] = {x - xdip, y - ydip, z - zdip};
    const CU_REAL radius2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    const CU_REAL radius = sqrt(radius2);
    if (radius <= aretecube / CU_REAL(100)) {
        const CU_REAL vol = CU_REAL(1) / (CU_REAL(4) * pi);
        const CUCOMPLEX constant =
            antenna_complex(-CU_REAL(1) / (CU_REAL(3) * aretecube * aretecube * aretecube),
                            CU_REAL(2) / CU_REAL(3) * vol * k0 * k0 * k0);
        return CUFFT_MUL(CUFFT_MUL(e0, CUFFT_REAL(p[component])), constant);
    }

    const CU_REAL inv_r2 = CU_REAL(1) / radius2;
    const CU_REAL phase = k0 * radius;
    const CUCOMPLEX wave = antenna_complex(cos(phase), sin(phase));
    const CUCOMPLEX c1 = antenna_complex(inv_r2 / radius, -k0 * inv_r2);
    const CUCOMPLEX c2 = CUFFT_REAL(k0 * k0 / radius);
    CUCOMPLEX value = CUFFT_ZERO;
    for (int j = 0; j < 3; ++j) {
        const CU_REAL delta = component == j ? CU_REAL(1) : CU_REAL(0);
        const CU_REAL rr = r[component] * r[j] * inv_r2;
        const CUCOMPLEX t1 = CUFFT_MUL(c1, CUFFT_REAL(CU_REAL(3) * rr - delta));
        const CUCOMPLEX t2 = CUFFT_MUL(c2, CUFFT_REAL(delta - rr));
        const CUCOMPLEX green = CUFFT_MUL(CUFFT_ADD(t1, t2), wave);
        const CUCOMPLEX moment = CUFFT_MUL(e0, CUFFT_REAL(p[j]));
        value = CUFFT_ADD(value, CUFFT_MUL(green, moment));
    }
    const CU_REAL vol = CU_REAL(1) / (CU_REAL(4) * pi);
    return CUFFT_MUL(value, CUFFT_REAL(vol));
}

//-----------------------------------------------------------
__global__ void incident_antenna_kernel(const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
                                        CUCOMPLEX *field, CUCOMPLEX *reference_field, int count,
                                        CU_REAL xdip, CU_REAL ydip, CU_REAL zdip, CU_REAL aretecube,
                                        CU_REAL k0, CUCOMPLEX e0, CU_REAL theta_degrees,
                                        CU_REAL phi_degrees, CU_REAL reference_x,
                                        CU_REAL reference_y, CU_REAL reference_z,
                                        IfddaIncidentGrid grid) {
    const int flat = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int work_count = grid.tabdip ? grid.grid_count : count;
    if (flat < work_count) {
        int i = flat;
        REAL x = 0, y = 0, z = 0;
        if (grid.tabdip) {
            if (!ifdda_incident_grid_fetch(grid, flat, &i, &x, &y, &z)) return;
        } else { x = xs[i]; y = ys[i]; z = zs[i]; }
        field[3 * i] = antenna_field_at_point(x, y, z, xdip, ydip, zdip, aretecube, k0,
                                              e0, theta_degrees, phi_degrees, 0);
        field[3 * i + 1] = antenna_field_at_point(x, y, z, xdip, ydip, zdip, aretecube,
                                                  k0, e0, theta_degrees, phi_degrees, 1);
        field[3 * i + 2] = antenna_field_at_point(x, y, z, xdip, ydip, zdip, aretecube,
                                                  k0, e0, theta_degrees, phi_degrees, 2);
    }
    if (flat == work_count) {
        reference_field[0] =
            antenna_field_at_point(reference_x, reference_y, reference_z, xdip, ydip, zdip,
                                   aretecube, k0, e0, theta_degrees, phi_degrees, 0);
        reference_field[1] =
            antenna_field_at_point(reference_x, reference_y, reference_z, xdip, ydip, zdip,
                                   aretecube, k0, e0, theta_degrees, phi_degrees, 1);
        reference_field[2] =
            antenna_field_at_point(reference_x, reference_y, reference_z, xdip, ydip, zdip,
                                   aretecube, k0, e0, theta_degrees, phi_degrees, 2);
    }
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_antenna_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, CUCOMPLEX *field,
    CUCOMPLEX *reference_field, int count, CU_REAL xdip, CU_REAL ydip, CU_REAL zdip,
    CU_REAL aretecube, CU_REAL k0, COMPLEX e0, CU_REAL theta_degrees, CU_REAL phi_degrees,
    CU_REAL reference_x, CU_REAL reference_y, CU_REAL reference_z) {
    if (!xs || !ys || !zs || !field || !reference_field || count < 0 || aretecube <= CU_REAL(0) ||
        k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    incident_antenna_kernel<<<(count + 1 + 255) / 256, 256>>>(
        xs, ys, zs, field, reference_field, count, xdip, ydip, zdip, aretecube, k0,
        plane_wave_device_scalar(e0),
        theta_degrees, phi_degrees, reference_x, reference_y, reference_z, {});
    return finish_kernel_launch(203, "incident_antenna_kernel");
}

extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t incident_antenna_grid_wrapper(
    IfddaIncidentGrid grid, CUCOMPLEX* field, CUCOMPLEX* reference_field,
    REAL xdip, REAL ydip, REAL zdip, REAL aretecube, REAL k0,
    COMPLEX e0, REAL theta_degrees, REAL phi_degrees,
    REAL reference_x, REAL reference_y, REAL reference_z) {
    if (!grid.tabdip || !field || !reference_field || grid.grid_count <= 0 ||
        grid.dipole_count <= 0 || aretecube <= REAL(0) || k0 <= REAL(0))
        return cudaErrorInvalidValue;
    incident_antenna_kernel<<<(grid.grid_count + 1 + 255) / 256, 256>>>(
        nullptr, nullptr, nullptr, field, reference_field, grid.dipole_count,
        xdip, ydip, zdip, aretecube, k0, plane_wave_device_scalar(e0),
        theta_degrees, phi_degrees, reference_x, reference_y, reference_z, grid);
    return finish_kernel_launch(203, "incident_antenna_grid_kernel");
}

//-----------------------------------------------------------
__global__ void dipole_moment_field_kernel(const CU_REAL *xs, const CU_REAL *ys,
                                           const CU_REAL *zs, CUCOMPLEX *field,
                                           CU_REAL xdip, CU_REAL ydip, CU_REAL zdip,
                                           CU_REAL aretecube, CU_REAL k0,
                                           CUCOMPLEX moment_x, CUCOMPLEX moment_y,
                                           CUCOMPLEX moment_z) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);
    const CU_REAL r[3] = {xs[0] - xdip, ys[0] - ydip, zs[0] - zdip};
    const CUCOMPLEX moment[3] = {moment_x, moment_y, moment_z};
    const CU_REAL radius2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    const CU_REAL radius = sqrt(radius2);
    if (radius <= aretecube / CU_REAL(100)) {
        const CUCOMPLEX self = antenna_complex(
            -CU_REAL(4) * pi / (CU_REAL(3) * aretecube * aretecube * aretecube),
            CU_REAL(2) * k0 * k0 * k0 / CU_REAL(3));
        field[0] = CUFFT_MUL(self, moment[0]);
        field[1] = CUFFT_MUL(self, moment[1]);
        field[2] = CUFFT_MUL(self, moment[2]);
        return;
    }

    const CU_REAL inverse_radius2 = CU_REAL(1) / radius2;
    const CUCOMPLEX constant1 = antenna_complex(
        inverse_radius2 / radius, -k0 * inverse_radius2);
    const CUCOMPLEX constant2 = CUFFT_REAL(k0 * k0 / radius);
    const CUCOMPLEX wave = antenna_complex(cos(k0 * radius), sin(k0 * radius));
    for (int row = 0; row < 3; ++row) {
        CUCOMPLEX value = CUFFT_ZERO;
        for (int column = 0; column < 3; ++column) {
            const CU_REAL delta = row == column ? CU_REAL(1) : CU_REAL(0);
            const CU_REAL rr = r[row] * r[column] * inverse_radius2;
            const CUCOMPLEX first = CUFFT_MUL(
                constant1, CUFFT_REAL(CU_REAL(3) * rr - delta));
            const CUCOMPLEX second = CUFFT_MUL(
                constant2, CUFFT_REAL(delta - rr));
            const CUCOMPLEX green = CUFFT_MUL(CUFFT_ADD(first, second), wave);
            value = CUFFT_ADD(value, CUFFT_MUL(green, moment[column]));
        }
        field[row] = value;
    }
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t dipole_moment_field_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, CUCOMPLEX *field,
    CU_REAL xdip, CU_REAL ydip, CU_REAL zdip, CU_REAL aretecube, CU_REAL k0,
    CUCOMPLEX moment_x, CUCOMPLEX moment_y, CUCOMPLEX moment_z) {
    if (!xs || !ys || !zs || !field || aretecube <= CU_REAL(0) || k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    dipole_moment_field_kernel<<<1, 1>>>(
        xs, ys, zs, field, xdip, ydip, zdip, aretecube, k0,
        moment_x, moment_y, moment_z);
    return finish_kernel_launch(221, "dipole_moment_field_kernel");
}

//-----------------------------------------------------------
__global__ void dipole_moment_field_batch_kernel(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, CUCOMPLEX *field, int count,
    CU_REAL xdip, CU_REAL ydip, CU_REAL zdip, CU_REAL aretecube, CU_REAL k0,
    CUCOMPLEX moment_x, CUCOMPLEX moment_y, CUCOMPLEX moment_z) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);
    const CU_REAL r[3] = {xs[index] - xdip, ys[index] - ydip, zs[index] - zdip};
    const CUCOMPLEX moment[3] = {moment_x, moment_y, moment_z};
    const CU_REAL radius2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    const CU_REAL radius = sqrt(radius2);
    if (radius <= aretecube / CU_REAL(100)) {
        const CUCOMPLEX self = antenna_complex(
            -CU_REAL(4) * pi / (CU_REAL(3) * aretecube * aretecube * aretecube),
            CU_REAL(2) * k0 * k0 * k0 / CU_REAL(3));
        for (int row = 0; row < 3; ++row)
            field[3 * index + row] = CUFFT_MUL(self, moment[row]);
        return;
    }

    const CU_REAL inverse_radius2 = CU_REAL(1) / radius2;
    const CUCOMPLEX constant1 = antenna_complex(
        inverse_radius2 / radius, -k0 * inverse_radius2);
    const CUCOMPLEX constant2 = CUFFT_REAL(k0 * k0 / radius);
    const CUCOMPLEX wave = antenna_complex(cos(k0 * radius), sin(k0 * radius));
    for (int row = 0; row < 3; ++row) {
        CUCOMPLEX value = CUFFT_ZERO;
        for (int column = 0; column < 3; ++column) {
            const CU_REAL delta = row == column ? CU_REAL(1) : CU_REAL(0);
            const CU_REAL rr = r[row] * r[column] * inverse_radius2;
            const CUCOMPLEX first = CUFFT_MUL(
                constant1, CUFFT_REAL(CU_REAL(3) * rr - delta));
            const CUCOMPLEX second = CUFFT_MUL(
                constant2, CUFFT_REAL(delta - rr));
            const CUCOMPLEX green = CUFFT_MUL(CUFFT_ADD(first, second), wave);
            value = CUFFT_ADD(value, CUFFT_MUL(green, moment[column]));
        }
        field[3 * index + row] = value;
    }
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t dipole_moment_field_batch_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, CUCOMPLEX *field, int count,
    CU_REAL xdip, CU_REAL ydip, CU_REAL zdip, CU_REAL aretecube, CU_REAL k0,
    CUCOMPLEX moment_x, CUCOMPLEX moment_y, CUCOMPLEX moment_z) {
    if (!xs || !ys || !zs || !field || count <= 0 ||
        aretecube <= CU_REAL(0) || k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    dipole_moment_field_batch_kernel<<<grid_size, block_size>>>(
        xs, ys, zs, field, count, xdip, ydip, zdip, aretecube, k0,
        moment_x, moment_y, moment_z);
    return finish_kernel_launch(222, "dipole_moment_field_batch_kernel");
}

//-----------------------------------------------------------
__global__ void dipole_incident_derivative_kernel(
    CUCOMPLEX *derivative, CU_REAL xdip, CU_REAL ydip, CU_REAL zdip,
    CU_REAL theta_degrees, CU_REAL phi_degrees, CU_REAL x, CU_REAL y, CU_REAL z,
    CU_REAL aretecube, CU_REAL k0, CUCOMPLEX e0, CU_REAL step, int derivative_axis) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const CU_REAL dx = xdip - x;
    const CU_REAL dy = ydip - y;
    const CU_REAL dz = zdip - z;
    const CU_REAL distance = sqrt(dx * dx + dy * dy + dz * dz);
    const CU_REAL cutoff = aretecube / CU_REAL(100);
    for (int i = 0; i < 3; ++i)
        derivative[i] = CUFFT_ZERO;
    if (distance <= cutoff)
        return;

    const CU_REAL coordinate[3] = {x, y, z};
    const int axis = derivative_axis - 1;
    {
        CU_REAL plus[3] = {coordinate[0], coordinate[1], coordinate[2]};
        CU_REAL minus[3] = {coordinate[0], coordinate[1], coordinate[2]};
        plus[axis] += step;
        minus[axis] -= step;
        for (int component = 0; component < 3; ++component) {
            const CUCOMPLEX upper = antenna_field_at_point(
                plus[0], plus[1], plus[2], xdip, ydip, zdip, aretecube, k0,
                e0, theta_degrees, phi_degrees, component);
            const CUCOMPLEX lower = antenna_field_at_point(
                minus[0], minus[1], minus[2], xdip, ydip, zdip, aretecube, k0,
                e0, theta_degrees, phi_degrees, component);
            const CU_REAL inverse_step = CU_REAL(1) / (CU_REAL(2) * step);
            derivative[component] = antenna_complex(
                (upper.x - lower.x) * inverse_step,
                (upper.y - lower.y) * inverse_step);
        }
    }
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t dipole_incident_derivative_wrapper(
    CUCOMPLEX *derivative, CU_REAL xdip, CU_REAL ydip, CU_REAL zdip,
    CU_REAL theta_degrees, CU_REAL phi_degrees, CU_REAL x, CU_REAL y, CU_REAL z,
    CU_REAL aretecube, CU_REAL k0, CUCOMPLEX e0, int derivative_axis) {
    if (!derivative || aretecube <= CU_REAL(0) || k0 <= CU_REAL(0) ||
        derivative_axis < 1 || derivative_axis > 3)
        return cudaErrorInvalidValue;
    const CU_REAL dx = xdip - x;
    const CU_REAL dy = ydip - y;
    const CU_REAL dz = zdip - z;
    const CU_REAL distance = sqrt(dx * dx + dy * dy + dz * dz);
    const CU_REAL scale = fmin(aretecube, CU_REAL(1) / k0);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    CU_REAL step = cbrtf(FLT_EPSILON) * scale;
#else
    CU_REAL step = cbrt(DBL_EPSILON) * scale;
#endif
    if (distance > aretecube / CU_REAL(100)) {
        const CU_REAL room = distance - aretecube / CU_REAL(100);
        step = fmin(step, room * CU_REAL(0.25));
    }
    if (step <= CU_REAL(0) || !isfinite(step))
        return cudaErrorInvalidValue;
    dipole_incident_derivative_kernel<<<1, 1>>>(
        derivative, xdip, ydip, zdip, theta_degrees, phi_degrees, x, y, z,
        aretecube, k0, e0, step, derivative_axis);
    return finish_kernel_launch(247, "dipole_incident_derivative_kernel");
}

//-----------------------------------------------------------
__global__ void dipole_incident_derivative_batch_kernel(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    CUCOMPLEX* derivatives, int count, CU_REAL xdip, CU_REAL ydip,
    CU_REAL zdip, CU_REAL theta_degrees, CU_REAL phi_degrees,
    CU_REAL aretecube, CU_REAL k0, CUCOMPLEX e0) {
    const int point = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (point >= count)
        return;
    const size_t output_base = size_t(point) * 9;
    const CU_REAL x = xs[point], y = ys[point], z = zs[point];
    const CU_REAL dx = xdip - x, dy = ydip - y, dz = zdip - z;
    const CU_REAL distance = sqrt(dx * dx + dy * dy + dz * dz);
    const CU_REAL cutoff = aretecube / CU_REAL(100);
    if (distance <= cutoff) {
        for (int i = 0; i < 9; ++i)
            derivatives[output_base + size_t(i)] = CUFFT_ZERO;
        return;
    }
    const CU_REAL scale = fmin(aretecube, CU_REAL(1) / k0);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    CU_REAL step = cbrtf(FLT_EPSILON) * scale;
#else
    CU_REAL step = cbrt(DBL_EPSILON) * scale;
#endif
    const CU_REAL room = distance - cutoff;
    step = fmin(step, room * CU_REAL(0.25));
    const CU_REAL inverse_step = CU_REAL(1) / (CU_REAL(2) * step);
    for (int axis = 0; axis < 3; ++axis) {
        CU_REAL plus[3] = {x, y, z};
        CU_REAL minus[3] = {x, y, z};
        plus[axis] += step;
        minus[axis] -= step;
        for (int component = 0; component < 3; ++component) {
            const CUCOMPLEX upper = antenna_field_at_point(
                plus[0], plus[1], plus[2], xdip, ydip, zdip, aretecube, k0,
                e0, theta_degrees, phi_degrees, component);
            const CUCOMPLEX lower = antenna_field_at_point(
                minus[0], minus[1], minus[2], xdip, ydip, zdip, aretecube, k0,
                e0, theta_degrees, phi_degrees, component);
            derivatives[output_base + size_t(component + 3 * axis)] = antenna_complex(
                (upper.x - lower.x) * inverse_step,
                (upper.y - lower.y) * inverse_step);
        }
    }
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t dipole_incident_derivative_batch_wrapper(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    CUCOMPLEX* derivatives, int count, CU_REAL xdip, CU_REAL ydip,
    CU_REAL zdip, CU_REAL theta_degrees, CU_REAL phi_degrees,
    CU_REAL aretecube, CU_REAL k0, CUCOMPLEX e0) {
    if (!xs || !ys || !zs || !derivatives || count < 0 ||
        aretecube <= CU_REAL(0) || k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    dipole_incident_derivative_batch_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, derivatives, count, xdip, ydip, zdip, theta_degrees,
        phi_degrees, aretecube, k0, e0);
    return finish_kernel_launch(265, "dipole_incident_derivative_batch_kernel");
}

//-----------------------------------------------------------
__global__ void multilayer_scalar_initialize_plane_kernel(
    const CU_INT*plane_map, const CUCOMPLEX *incident_dipoles, CUCOMPLEX *image,
    CUCOMPLEX *local_field, int nfft2d, int nmax, int kkm, int jjm, CU_REAL xmin, CU_REAL ymin,
    CU_REAL first_z, CU_REAL aretecube, CU_REAL k0, CUCOMPLEX E0, CU_REAL ss, CU_REAL pp,
    CUCOMPLEX eps0, CU_REAL w0, CU_REAL theta_degrees, CU_REAL phi_degrees, int beam_type,
    int conjugate_projection, CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nfft2d * nfft2d;
    if (index >= count)
        return;
    const int ix = index % nfft2d;
    const int iy = index / nfft2d;
    const CU_REAL x = xmin + CU_REAL(ix - kkm) * aretecube;
    const CU_REAL y = ymin + CU_REAL(iy - jjm) * aretecube;
    const int local_dipole = plane_map[index].dat;
    if (local_dipole > 0) {
        const int base = 3 * (local_dipole - 1);
        if (local_dipole > nmax) {
            image[index] = CUFFT_ZERO;
            return;
        }
        const CUCOMPLEX ex = incident_dipoles[base];
        const CUCOMPLEX ey = incident_dipoles[base + 1];
        const CUCOMPLEX ez = incident_dipoles[base + 2];
        CUCOMPLEX px = ux, py = uy, pz = uz;
        if (conjugate_projection) {
            px = CUFFT_CONJ(px);
            py = CUFFT_CONJ(py);
            pz = CUFFT_CONJ(pz);
        }
        image[index] =
            CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(px, ex), CUFFT_MUL(py, ey)), CUFFT_MUL(pz, ez));
        local_field[base] = ex;
        local_field[base + 1] = ey;
        local_field[base + 2] = ez;
        return;
    }

    const BeamPropagationIncidentField field =
        beam_type >= 2
            ? multilayer_paraxial_gaussian_field(E0, eps0, ss, pp, theta_degrees, phi_degrees, x, y,
                                                 first_z, CU_REAL(0), CU_REAL(0), CU_REAL(0), w0,
                                                 k0, beam_type == 3)
            : beampropagation_plane_field(E0, ss, pp, theta_degrees, phi_degrees, x, y, first_z, k0,
                                          CU_REAL(0), CU_REAL(0), CU_REAL(0), CU_REAL(1),
                                          beam_type);
    CUCOMPLEX px = ux, py = uy, pz = uz;
    if (conjugate_projection) {
        px = CUFFT_CONJ(px);
        py = CUFFT_CONJ(py);
        pz = CUFFT_CONJ(pz);
    }
    image[index] = CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(px, field.x), CUFFT_MUL(py, field.y)),
                             CUFFT_MUL(pz, field.z));
}

//-----------------------------------------------------------
__global__ void multilayer_vector_initialize_plane_kernel(
    const CU_INT*plane_map, const CUCOMPLEX *incident_dipoles, CUCOMPLEX *image_x, CUCOMPLEX *image_y,
    CUCOMPLEX *image_z, CUCOMPLEX *local_field, int nfft2d, int nmax, int kkm, int jjm,
    CU_REAL xmin, CU_REAL ymin, CU_REAL first_z, CU_REAL aretecube, CU_REAL k0, CUCOMPLEX E0,
    CU_REAL ss, CU_REAL pp, CUCOMPLEX eps0, CU_REAL w0, CU_REAL theta_degrees, CU_REAL phi_degrees,
    int beam_type) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nfft2d * nfft2d;
    if (index >= count)
        return;
    const int local_dipole = plane_map[index].dat;
    if (local_dipole > 0) {
        const int base = 3 * (local_dipole - 1);
        if (local_dipole > nmax) {
            image_x[index] = image_y[index] = image_z[index] = CUFFT_ZERO;
            return;
        }
        image_x[index] = incident_dipoles[base];
        image_y[index] = incident_dipoles[base + 1];
        image_z[index] = incident_dipoles[base + 2];
        local_field[base] = incident_dipoles[base];
        local_field[base + 1] = incident_dipoles[base + 1];
        local_field[base + 2] = incident_dipoles[base + 2];
        return;
    }
    const int ix = index % nfft2d;
    const int iy = index / nfft2d;
    const CU_REAL x = xmin + CU_REAL(ix - kkm) * aretecube;
    const CU_REAL y = ymin + CU_REAL(iy - jjm) * aretecube;
    const BeamPropagationIncidentField field =
        beam_type >= 2
            ? multilayer_paraxial_gaussian_field(E0, eps0, ss, pp, theta_degrees, phi_degrees, x, y,
                                                 first_z, CU_REAL(0), CU_REAL(0), CU_REAL(0), w0,
                                                 k0, beam_type == 3)
            : beampropagation_plane_field(E0, ss, pp, theta_degrees, phi_degrees, x, y, first_z, k0,
                                          CU_REAL(0), CU_REAL(0), CU_REAL(0), CU_REAL(1),
                                          beam_type);
    image_x[index] = field.x;
    image_y[index] = field.y;
    image_z[index] = field.z;
}

//-----------------------------------------------------------
__global__ void multilayer_scalar_multiply_green_kernel(CUCOMPLEX *polarization,
                                                        const CUCOMPLEX *green, int count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
        polarization[index] = CUFFT_MUL(polarization[index], green[index]);
}

//-----------------------------------------------------------
__global__ void multilayer_scalar_propagate_kernel(CUCOMPLEX *image, int nfft2d, CU_REAL deltakx,
                                                   CU_REAL deltaky, CU_REAL k0, CU_REAL aretecube) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nfft2d * nfft2d;
    if (index >= count)
        return;
    const int i = index % nfft2d;
    const int j = index / nfft2d;
    const int ii = i >= nfft2d / 2 ? i - nfft2d : i;
    const int jj = j >= nfft2d / 2 ? j - nfft2d : j;
    const CU_REAL kx = deltakx * CU_REAL(ii);
    const CU_REAL ky = deltaky * CU_REAL(jj);
    CU_REAL kz2 = k0 * k0 - kx * kx - ky * ky;
    if (kz2 < CU_REAL(0)) {
        image[index] = CUFFT_ZERO;
        return;
    }
    const CU_REAL angle = sqrt(kz2) * aretecube;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cosf(angle), sinf(angle));
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(angle), sin(angle));
#endif
    image[index] = CUFFT_MUL(image[index], phase);
}

//-----------------------------------------------------------
__global__ void multilayer_scalar_accumulate_kernel(CUCOMPLEX *image, const CUCOMPLEX *scattered,
                                                    CUCOMPLEX *local_field, const CU_INT*plane_map,
                                                    int nfft2d, int nx, int ny, int layer, int nmax,
                                                    int plane_count, CUCOMPLEX ux, CUCOMPLEX uy,
                                                    CUCOMPLEX uz, CU_REAL image_scale,
                                                    CU_REAL scattered_scale, int add_to_local) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int image_count = nfft2d * nfft2d;
    if (index >= image_count)
        return;
    const int side = 2 * nfft2d;
    const CUCOMPLEX value =
        CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(image_scale), image[index]),
                  CUFFT_MUL(CUFFT_REAL(scattered_scale),
                            scattered[index % nfft2d + (index / nfft2d) * side]));
    image[index] = value;
    if (index >= plane_count)
        return;
    const int local_dipole = plane_map[index].dat;
    if (local_dipole == 0)
        return;
    const int global_dipole = nx * ny * layer + local_dipole - 1;
    if (global_dipole < 0 || global_dipole >= nmax)
        return;
    const CUCOMPLEX field = value;
    const CUCOMPLEX local_x = CUFFT_MUL(field, ux);
    const CUCOMPLEX local_y = CUFFT_MUL(field, uy);
    const CUCOMPLEX local_z = CUFFT_MUL(field, uz);
    if (add_to_local) {
        local_field[3 * global_dipole] = CUFFT_ADD(local_field[3 * global_dipole], local_x);
        local_field[3 * global_dipole + 1] = CUFFT_ADD(local_field[3 * global_dipole + 1], local_y);
        local_field[3 * global_dipole + 2] = CUFFT_ADD(local_field[3 * global_dipole + 2], local_z);
    } else {
        local_field[3 * global_dipole] = local_x;
        local_field[3 * global_dipole + 1] = local_y;
        local_field[3 * global_dipole + 2] = local_z;
    }
}

//-----------------------------------------------------------
__global__ void multilayer_scalar_save_plane_kernel(const CUCOMPLEX *image, CUCOMPLEX *history,
                                                    int image_count, int layer, int layer_count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < image_count)
        history[size_t(layer) * size_t(image_count) + size_t(index)] = image[index];
}

//-----------------------------------------------------------
__global__ void scalar_chan_build_kernel(CUCOMPLEX *channel, CUCOMPLEX mean_polarizability,
                                         CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz,
                                         CU_REAL aretecube, CU_REAL k0, int nrig, int nx, int ny,
                                         int nz) {
    const int plane = nx * ny;
    const int count = plane * nz;
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int x = index % nx;
    const int y = (index / nx) % ny;
    const int z = index / plane;
    const int x1 = x == 0 ? 0 : nx - x;
    const int y1 = y == 0 ? 0 : ny - y;
    const CU_REAL dx1 = CU_REAL(x), dx2 = CU_REAL(nx - x);
    const CU_REAL dy1 = CU_REAL(y), dy2 = CU_REAL(ny - y);
    const CU_REAL inverse_plane = CU_REAL(1) / CU_REAL(plane);
    const CU_REAL zcoord = aretecube * CU_REAL(z);
    const CUCOMPLEX g11 = scalar_chan_green_value(aretecube * CU_REAL(x1), aretecube * CU_REAL(y1),
                                                  zcoord, k0, nrig, ux, uy, uz);
    const CUCOMPLEX g12 = scalar_chan_green_value(aretecube * CU_REAL(x1), aretecube * CU_REAL(y),
                                                  zcoord, k0, nrig, ux, uy, uz);
    const CUCOMPLEX g21 = scalar_chan_green_value(aretecube * CU_REAL(x), aretecube * CU_REAL(y1),
                                                  zcoord, k0, nrig, ux, uy, uz);
    const CUCOMPLEX g22 = scalar_chan_green_value(aretecube * CU_REAL(x), aretecube * CU_REAL(y),
                                                  zcoord, k0, nrig, ux, uy, uz);
    CUCOMPLEX weighted = CUFFT_ZERO;
    weighted = CUFFT_ADD(weighted, CUFFT_MUL(g11, CUFFT_REAL(dx1 * dy1 * inverse_plane)));
    weighted = CUFFT_ADD(weighted, CUFFT_MUL(g12, CUFFT_REAL(dx1 * dy2 * inverse_plane)));
    weighted = CUFFT_ADD(weighted, CUFFT_MUL(g21, CUFFT_REAL(dx2 * dy1 * inverse_plane)));
    weighted = CUFFT_ADD(weighted, CUFFT_MUL(g22, CUFFT_REAL(dx2 * dy2 * inverse_plane)));
    const int output_index = z * plane + x1 + nx * y1;
    channel[output_index] =
        CUFFT_MUL(CUFFT_MUL(weighted, CUFFT_REAL(CU_REAL(-1))), mean_polarizability);
    if (index == 0)
        channel[output_index] = CUFFT_ADD(channel[output_index], CUFFT_REAL(CU_REAL(1)));
}

//-----------------------------------------------------------
__global__ void scalar_chan_transpose_kernel(const CUCOMPLEX *fft_channel, CUCOMPLEX *chan_vectors,
                                             int nx, int ny, int nz) {
    const int plane = nx * ny;
    const int count = plane * nz;
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int x = index % nx;
    const int y = (index / nx) % ny;
    const int z = index / plane;
    chan_vectors[z * plane + y + ny * x] = fft_channel[index];
}

//-----------------------------------------------------------
__global__ void scalar_chan_build_blocks_kernel(const CUCOMPLEX *chan_vectors, CUCOMPLEX *matrices,
                                                int nx, int ny, int nz) {
    const int plane = nx * ny;
    const int matrix_size = nz * nz;
    const int count = plane * matrix_size;
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int batch = index / matrix_size;
    const int in_matrix = index % matrix_size;
    const int row = in_matrix % nz;
    const int col = in_matrix / nz;
    matrices[index] = chan_vectors[batch + plane * abs(row - col)];
}

//-----------------------------------------------------------
__global__ void scalar_preconditioner_gather_kernel(const CUCOMPLEX *field, const CU_INT *dipole_map,
                                                    CUCOMPLEX *fft_planes, int nx, int ny, int nz,
                                                    int nxm, int nym, int nzm) {
    const int total = nx * ny * nz;
    const int mesh_index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (mesh_index >= total)
        return;
    const int x = mesh_index % nx;
    const int y = (mesh_index / nx) % ny;
    const int z = mesh_index / (nx * ny);
    const int map_index = x + nx * y + nx * ny * z;
    const int dipole = (x < nxm && y < nym && z < nzm) ? dipole_map[map_index].dat : 0;
    fft_planes[mesh_index] = dipole > 0 ? field[dipole - 1] : CUFFT_ZERO;
}

//-----------------------------------------------------------
__global__ void scalar_preconditioner_fft_to_rhs_kernel(const CUCOMPLEX *fft_planes, CUCOMPLEX *rhs,
                                                        int nx, int ny, int nz) {
    const int total = nx * ny * nz;
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total)
        return;
    const int plane = nx * ny;
    const int x = index % nx;
    const int y = (index / nx) % ny;
    const int z = index / plane;
    const int batch = y + ny * x;
    rhs[batch * nz + z] = fft_planes[index];
}

//-----------------------------------------------------------
__global__ void scalar_preconditioner_rhs_to_fft_kernel(const CUCOMPLEX *rhs, CUCOMPLEX *fft_planes,
                                                        int nx, int ny, int nz) {
    const int total = nx * ny * nz;
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total)
        return;
    const int plane = nx * ny;
    const int x = index % nx;
    const int y = (index / nx) % ny;
    const int z = index / plane;
    const int batch = y + ny * x;
    fft_planes[index] = rhs[batch * nz + z];
}

//-----------------------------------------------------------
__global__ void scalar_preconditioner_matvec_kernel(const CUCOMPLEX *matrices, const CUCOMPLEX *rhs,
                                                    CUCOMPLEX *output, int nx, int ny, int nz) {
    const int count = nx * ny * nz;
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int batch = index / nz;
    const int row = index % nz;
    const size_t matrix_offset = size_t(batch) * size_t(nz) * size_t(nz);
    CUCOMPLEX sum = CUFFT_ZERO;
    for (int col = 0; col < nz; ++col)
        sum = CUFFT_ADD(sum, CUFFT_MUL(matrices[matrix_offset + size_t(col) * nz + row],
                                       rhs[size_t(batch) * nz + col]));
    output[index] = sum;
}

//-----------------------------------------------------------
__global__ void scalar_preconditioner_scatter_kernel(const CUCOMPLEX *fft_planes,
                                                     const CU_INT *dipole_map, CUCOMPLEX *field,
                                                     int nx, int ny, int nz, int nxm, int nym,
                                                     int nzm, CU_REAL normalization) {
    const int total = nx * ny * nz;
    const int mesh_index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (mesh_index >= total)
        return;
    const int x = mesh_index % nx;
    const int y = (mesh_index / nx) % ny;
    const int z = mesh_index / (nx * ny);
    const int map_index = x + nx * y + nx * ny * z;
    const int dipole = (x < nxm && y < nym && z < nzm) ? dipole_map[map_index].dat : 0;
    if (dipole > 0)
        field[dipole - 1] = CUFFT_MUL(fft_planes[mesh_index], CUFFT_REAL(normalization));
}

} // namespace

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_pack_fft_wrapper(const CUCOMPLEX *field,
                                                            const CU_INT *dipole_map,
                                                            CUCOMPLEX *fft_planes, int nx, int ny,
                                                            int nz, int nxm, int nym, int nzm,
                                                            int nmax) {
    if (!field || !dipole_map || !fft_planes || nx <= 0 || ny <= 0 || nz <= 0 || nxm < nx ||
        nym < ny || nzm < nz || nmax <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(3) * size_t(nx) * size_t(ny) * size_t(nz);
    chan_preconditioner_pack_fft_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        field, dipole_map, fft_planes, nx, ny, nz, nxm, nym, nzm, nmax);
    return finish_gpbicg_launch(91, "chan_preconditioner_pack_fft_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_fft_to_rhs_wrapper(const CUCOMPLEX *fft_planes,
                                                              CUCOMPLEX *rhs, int nx, int ny,
                                                              int nz) {
    if (!fft_planes || !rhs || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(3) * size_t(nx) * size_t(ny) * size_t(nz);
    chan_preconditioner_fft_to_rhs_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        fft_planes, rhs, nx, ny, nz);
    return finish_gpbicg_launch(92, "chan_preconditioner_fft_to_rhs_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_rhs_to_fft_wrapper(const CUCOMPLEX *rhs,
                                                              CUCOMPLEX *fft_planes, int nx, int ny,
                                                              int nz) {
    if (!rhs || !fft_planes || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(3) * size_t(nx) * size_t(ny) * size_t(nz);
    chan_preconditioner_rhs_to_fft_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        rhs, fft_planes, nx, ny, nz);
    return finish_gpbicg_launch(93, "chan_preconditioner_rhs_to_fft_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_unpack_fft_wrapper(const CUCOMPLEX *fft_planes,
                                                              const CU_INT *dipole_map,
                                                              CUCOMPLEX *field, int nx, int ny,
                                                              int nz, int nxm, int nym, int nzm,
                                                              int nmax, CU_REAL normalization) {
    if (!fft_planes || !dipole_map || !field || nx <= 0 || ny <= 0 || nz <= 0 || nxm < nx ||
        nym < ny || nzm < nz || nmax <= 0 || normalization <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t total = size_t(3) * size_t(nx) * size_t(ny) * size_t(nz);
    chan_preconditioner_unpack_fft_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        fft_planes, dipole_map, field, nx, ny, nz, nxm, nym, nzm, nmax, normalization);
    return finish_gpbicg_launch(94, "chan_preconditioner_unpack_fft_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_build_blocks_wrapper(const CUCOMPLEX *chan_vectors,
                                                                CUCOMPLEX *matrices, int nx, int ny,
                                                                int nz) {
    if (!chan_vectors || !matrices || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t order = size_t(3) * size_t(nz);
    const size_t total = order * order * size_t(nx) * size_t(ny);
    chan_preconditioner_build_blocks_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        chan_vectors, matrices, nx, ny, nz);
    return finish_gpbicg_launch(95, "chan_preconditioner_build_blocks_kernel");
}

namespace {
//-----------------------------------------------------------
__global__ void chan_preconditioner_pack_matrices_kernel(const CUCOMPLEX *source, CUCOMPLEX *packed,
                                                         int order, int source_order,
                                                         int batch_count) {
    const size_t packed_stride = size_t(order) * size_t(order);
    const size_t total = packed_stride * size_t(batch_count);
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total)
        return;
    const size_t batch = index / packed_stride;
    const size_t in_batch = index - batch * packed_stride;
    const int row = int(in_batch % size_t(order));
    const int column = int(in_batch / size_t(order));
    const size_t source_stride = size_t(source_order) * size_t(source_order);
    packed[index] =
        source[batch * source_stride + size_t(column) * size_t(source_order) + size_t(row)];
}
//-----------------------------------------------------------
__global__ void chan_preconditioner_init_batch_ptrs_kernel(
    CUCOMPLEX *matrices, CUCOMPLEX *rhs, CUCOMPLEX **matrix_ptrs,
    CUCOMPLEX **rhs_ptrs, size_t matrix_stride, size_t rhs_stride, int batch_count) {
    const int batch = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (batch < batch_count) {
        matrix_ptrs[batch] = matrices + size_t(batch) * matrix_stride;
        rhs_ptrs[batch] = rhs + size_t(batch) * rhs_stride;
    }
}
//-----------------------------------------------------------
__global__ void chan_preconditioner_identity_map_kernel(CU_INT *identity_map, int map_count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < map_count)
        identity_map[index].dat = index + 1;
}
} // namespace

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_pack_matrices_wrapper(const CUCOMPLEX *source,
                                                                 CUCOMPLEX *packed, int nz, int nzm,
                                                                 int batch_count) {
    if (!source || !packed || nz <= 0 || nzm < nz || batch_count <= 0)
        return cudaErrorInvalidValue;
    const size_t order = size_t(3) * size_t(nz);
    const size_t total = order * order * size_t(batch_count);
    chan_preconditioner_pack_matrices_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        source, packed, 3 * nz, 3 * nzm, batch_count);
    return finish_gpbicg_launch(131, "chan_preconditioner_pack_matrices_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_init_batch_ptrs_wrapper(
    CUCOMPLEX *matrices, CUCOMPLEX *rhs, CUCOMPLEX **matrix_ptrs,
    CUCOMPLEX **rhs_ptrs, size_t matrix_stride, size_t rhs_stride, int batch_count) {
    if (!matrices || !rhs || !matrix_ptrs || !rhs_ptrs || !matrix_stride || !rhs_stride ||
        batch_count <= 0)
        return cudaErrorInvalidValue;
    chan_preconditioner_init_batch_ptrs_kernel<<<block_count(size_t(batch_count)), THREADS_PER_BLOCK>>>(
        matrices, rhs, matrix_ptrs, rhs_ptrs, matrix_stride, rhs_stride, batch_count);
    return finish_gpbicg_launch(132, "chan_preconditioner_init_batch_ptrs_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_identity_map_wrapper(CU_INT *identity_map, int map_count) {
    if (!identity_map || map_count <= 0)
        return cudaErrorInvalidValue;
    chan_preconditioner_identity_map_kernel<<<block_count(size_t(map_count)), THREADS_PER_BLOCK>>>(
        identity_map, map_count);
    return finish_gpbicg_launch(133, "chan_preconditioner_identity_map_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_green_tensor_wrapper(CUCOMPLEX *green_components,
                                                                CU_REAL aretecube, CU_REAL k0,
                                                                int nx, int ny, int nz) {
    if (!green_components || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(nx) * size_t(ny) * size_t(nz);
    chan_preconditioner_green_tensor_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        green_components, aretecube, k0, nx, ny, nz);
    return finish_gpbicg_launch(96, "chan_preconditioner_green_tensor_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_mean_polarizability_wrapper(
    const CUCOMPLEX *polarizabilities, int count, CUCOMPLEX *sum, CU_INT*nonzero_count) {
    if (!polarizabilities || count <= 0 || !sum || !nonzero_count)
        return cudaErrorInvalidValue;
    if (cudaMemset(sum, 0, sizeof(CUCOMPLEX)) != cudaSuccess ||
        cudaMemset(nonzero_count, 0, sizeof(int)) != cudaSuccess)
        return cudaGetLastError();
    chan_preconditioner_mean_polarizability_kernel<<<block_count(size_t(count)),
                                                     THREADS_PER_BLOCK>>>(polarizabilities, count,
                                                                          sum, nonzero_count);
    return finish_gpbicg_launch(99, "chan_preconditioner_mean_polarizability_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_finalize_mean_polarizability_wrapper(
    CUCOMPLEX *sum, const CU_INT*nonzero_count) {
    if (!sum || !nonzero_count)
        return cudaErrorInvalidValue;
    chan_preconditioner_finalize_mean_polarizability_kernel<<<1, 1>>>(sum, nonzero_count);
    return finish_gpbicg_launch(100, "chan_preconditioner_finalize_mean_polarizability_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_interpolate_wrapper(const CUCOMPLEX *green_components,
                                                               CUCOMPLEX *fft_components,
                                                               const CUCOMPLEX *mean_polarizability,
                                                               int nx, int ny, int nz) {
    if (!green_components || !fft_components || !mean_polarizability || nx <= 0 || ny <= 0 ||
        nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(nx) * size_t(ny) * size_t(nz);
    chan_preconditioner_interpolate_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        green_components, fft_components, mean_polarizability, nx, ny, nz);
    return finish_gpbicg_launch(97, "chan_preconditioner_interpolate_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t chan_preconditioner_transpose_vectors_wrapper(
    const CUCOMPLEX *fft_components, CUCOMPLEX *chan_vectors, int nx, int ny, int nz) {
    if (!fft_components || !chan_vectors || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(6) * size_t(nx) * size_t(ny) * size_t(nz);
    chan_preconditioner_transpose_vectors_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        fft_components, chan_vectors, nx, ny, nz);
    return finish_gpbicg_launch(98, "chan_preconditioner_transpose_vectors_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_driver_project_incident_wrapper(const CUCOMPLEX *vector_incident,
                                                              CUCOMPLEX *scalar_incident,
                                                              CUCOMPLEX uincx, CUCOMPLEX uincy,
                                                              CUCOMPLEX uincz, int count) {
    if (!vector_incident || !scalar_incident || count <= 0)
        return cudaErrorInvalidValue;
    scalar_driver_project_incident_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        vector_incident, scalar_incident, uincx, uincy, uincz, count);
    return finish_gpbicg_launch(108, "scalar_driver_project_incident_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_driver_apply_polarizability_wrapper(const CUCOMPLEX *input,
                                                                  const CUCOMPLEX *polarizability,
                                                                  CUCOMPLEX *output, int count) {
    if (!input || !polarizability || !output || count <= 0)
        return cudaErrorInvalidValue;
    scalar_driver_apply_polarizability_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        input, polarizability, output, count);
    return finish_gpbicg_launch(109, "scalar_driver_apply_polarizability_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_driver_exponential_guess_wrapper(const CUCOMPLEX *incident,
                                                               const CUCOMPLEX *green_correction,
                                                               CUCOMPLEX *initial_guess,
                                                               int count) {
    if (!incident || !green_correction || !initial_guess || count <= 0)
        return cudaErrorInvalidValue;
    scalar_driver_exponential_guess_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        incident, green_correction, initial_guess, count);
    return finish_gpbicg_launch(113, "scalar_driver_exponential_guess_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_driver_copy_solution_wrapper(const CUCOMPLEX *solution,
                                                           CUCOMPLEX *local_solution, int count) {
    if (!solution || !local_solution || count <= 0)
        return cudaErrorInvalidValue;
    scalar_driver_copy_solution_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        solution, local_solution, count);
    return finish_gpbicg_launch(110, "scalar_driver_copy_solution_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_driver_residual_wrapper(CUCOMPLEX *product, const CUCOMPLEX *rhs,
                                                      const CUCOMPLEX *solution,
                                                      CUCOMPLEX *scalar_solution, int count) {
    if (!product || !rhs || !solution || !scalar_solution || count <= 0)
        return cudaErrorInvalidValue;
    scalar_driver_residual_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        product, rhs, solution, scalar_solution, count);
    return finish_gpbicg_launch(111, "scalar_driver_residual_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_driver_expand_fields_wrapper(
    const CUCOMPLEX *solution, const CUCOMPLEX *local_solution, CUCOMPLEX *vector_field,
    CUCOMPLEX *vector_local_field, CUCOMPLEX uincx, CUCOMPLEX uincy, CUCOMPLEX uincz, int count) {
    if (!solution || !local_solution || !vector_field || !vector_local_field || count <= 0)
        return cudaErrorInvalidValue;
    scalar_driver_expand_fields_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        solution, local_solution, vector_field, vector_local_field, uincx, uincy, uincz, count);
    return finish_gpbicg_launch(112, "scalar_driver_expand_fields_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_chan_build_wrapper(CUCOMPLEX *channel, CUCOMPLEX mean_polarizability,
                                                 CUCOMPLEX uincx, CUCOMPLEX uincy, CUCOMPLEX uincz,
                                                 CU_REAL aretecube, CU_REAL k0, int nrig, int nx,
                                                 int ny, int nz) {
    if (!channel || (nrig != 8 && nrig != 9) || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(nx) * size_t(ny) * size_t(nz);
    scalar_chan_build_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        channel, mean_polarizability, uincx, uincy, uincz, aretecube, k0, nrig, nx, ny, nz);
    return finish_gpbicg_launch(100, "scalar_chan_build_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t generate_scalar_green_wrapper(CUCOMPLEX *green, int nx, int ny, int nz,
                                                     int nx2, int ny2, int nz2, int tensor_count,
                                                     CU_REAL aretecube, CU_REAL k0, CUCOMPLEX uincx,
                                                     CUCOMPLEX uincy, CUCOMPLEX uincz, int nrig) {
    if (!green || nx <= 0 || ny <= 0 || nz <= 0 || nx2 <= 0 || ny2 <= 0 || nz2 <= 0 ||
        tensor_count <= 0 || (nrig != 8 && nrig != 9))
        return cudaErrorInvalidValue;
    const size_t fft_count = size_t(nx2) * size_t(ny2) * size_t(nz2);
    if (fft_count > size_t(INT_MAX) || size_t(tensor_count) < fft_count)
        return cudaErrorInvalidValue;
    generate_scalar_green_kernel<<<block_count(size_t(tensor_count)), THREADS_PER_BLOCK>>>(
        green, nx, ny, nz, nx2, ny2, nz2, int(fft_count), tensor_count, aretecube, k0, uincx, uincy,
        uincz, nrig);
    return finish_gpbicg_launch(113, "generate_scalar_green_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t generate_multilayer_scalar_green_wrapper(CUCOMPLEX *green, int nfft2d,
                                                                int count, CU_REAL aretecube,
                                                                CU_REAL k0) {
    if (!green || nfft2d <= 0 || count != 4 * nfft2d * nfft2d || aretecube <= CU_REAL(0) ||
        k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    generate_multilayer_scalar_green_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        green, nfft2d, count, aretecube, k0);
    return finish_gpbicg_launch(114, "generate_multilayer_scalar_green_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t generate_multilayer_projected_green_wrapper(CUCOMPLEX *green, int nfft2d,
                                                                   int count, CU_REAL aretecube,
                                                                   CU_REAL k0, CUCOMPLEX uincx,
                                                                   CUCOMPLEX uincy,
                                                                   CUCOMPLEX uincz) {
    if (!green || nfft2d <= 0 || count != 4 * nfft2d * nfft2d || aretecube <= CU_REAL(0) ||
        k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    generate_multilayer_projected_green_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        green, nfft2d, count, aretecube, k0, uincx, uincy, uincz);
    return finish_gpbicg_launch(124, "generate_multilayer_projected_green_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t generate_multilayer_dyadic_green_wrapper(CUCOMPLEX *gxx, CUCOMPLEX *gxy,
                                                                CUCOMPLEX *gxz, CUCOMPLEX *gyy,
                                                                CUCOMPLEX *gyz, CUCOMPLEX *gzz,
                                                                int nfft2d, int count,
                                                                CU_REAL aretecube, CU_REAL k0) {
    if (!gxx || !gxy || !gxz || !gyy || !gyz || !gzz || nfft2d <= 0 ||
        count != 4 * nfft2d * nfft2d || aretecube <= CU_REAL(0) || k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    generate_multilayer_dyadic_green_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        gxx, gxy, gxz, gyy, gyz, gzz, nfft2d, count, aretecube, k0);
    return finish_gpbicg_launch(125, "generate_multilayer_dyadic_green_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t generate_vector_green_tensor_wrapper(
    GreenTensorDevicePointers green, GreenTensorDevicePointers spatial,
    GreenTensorDevicePointers spectra, int nx, int ny, int nz,
    int nxm, int nym, int nzm, int nx2, int ny2, int nz2, cufftHandle plan, REAL aretecube, REAL k0,
    COMPLEX eps0, int medium, int nquad, int filtered) {
    const CUCOMPLEX device_eps0{CU_REAL(eps0.real()), CU_REAL(eps0.imag())};
    if (!green || !spatial || !spectra.data[0] || !spectra.data[1] || !spectra.data[2] ||
        nx <= 0 || ny <= 0 || nz <= 0 || nxm < nx ||
        nym < ny || nzm < nz || nx2 < 2 * nx || ny2 < 2 * ny || nz2 < 2 * nz ||
        aretecube <= CU_REAL(0) || k0 <= CU_REAL(0) || nquad < 0 || nquad > 5 ||
        (medium && device_eps0.x == CU_REAL(0) && device_eps0.y == CU_REAL(0)))
        return cudaErrorInvalidValue;

    const size_t sx = size_t(nxm) + 1, sy = size_t(nym) + 1, sz = size_t(nzm) + 1;
    if (sx > SIZE_MAX / sy || sx * sy > SIZE_MAX / sz)
        return cudaErrorInvalidValue;
    const size_t tensor_count = sx * sy * sz;
    const size_t fx = size_t(nx2), fy = size_t(ny2), fz = size_t(nz2);
    if (fx > SIZE_MAX / fy || fx * fy > SIZE_MAX / fz)
        return cudaErrorInvalidValue;
    const size_t fft_count = fx * fy * fz;
    if (tensor_count > size_t(2147483647) / sizeof(CUCOMPLEX) ||
        fft_count > size_t(2147483647) / sizeof(CUCOMPLEX) ||
        tensor_count > size_t(INT_MAX) * size_t(THREADS_PER_BLOCK))
        return cudaErrorInvalidValue;

    cudaError_t cuda_status = cudaSuccess;
    if (!plan)
        return cudaErrorInvalidValue;

    green_tensor_direct_space_kernel<<<block_count(tensor_count), THREADS_PER_BLOCK>>>(
        spatial, nx, ny, nz, nxm, nym, nzm, tensor_count, aretecube, k0, device_eps0, medium, nquad,
        filtered);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
        goto cleanup;
    for (int component = 0; component < 6; ++component) {
        cuda_status = cudaMemset(green.data[component], 0, tensor_count * sizeof(CUCOMPLEX));
        if (cuda_status != cudaSuccess)
            goto cleanup;
    }

    green_tensor_embed_diagonal_kernel<<<block_count(fft_count), THREADS_PER_BLOCK>>>(
        spatial, spectra, nx, ny, nz, nxm, nym, nzm, nx2, ny2, nz2, tensor_count, fft_count);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
        goto cleanup;
    for (int component = 0; component < 3; ++component) {
        const cufftResult fft_status =
            CUFFT_EXECUTE(plan, spectra.data[component],
                          spectra.data[component], CUFFT_INVERSE);
        if (fft_status != CUFFT_SUCCESS) {
            cuda_status = cudaErrorUnknown;
            goto cleanup;
        }
    }
    green_tensor_store_spectra_kernel<<<block_count(tensor_count), THREADS_PER_BLOCK>>>(
        spectra, green, nx, ny, nz, nxm, nym, nx2, ny2, nz2, tensor_count, fft_count, 0, 3,
        5);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
        goto cleanup;

    green_tensor_embed_offdiagonal_kernel<<<block_count(fft_count), THREADS_PER_BLOCK>>>(
        spatial, spectra, nx, ny, nz, nxm, nym, nzm, nx2, ny2, nz2, tensor_count, fft_count);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
        goto cleanup;
    for (int component = 0; component < 3; ++component) {
        const cufftResult fft_status =
            CUFFT_EXECUTE(plan, spectra.data[component],
                          spectra.data[component], CUFFT_INVERSE);
        if (fft_status != CUFFT_SUCCESS) {
            cuda_status = cudaErrorUnknown;
            goto cleanup;
        }
    }
    green_tensor_store_spectra_kernel<<<block_count(tensor_count), THREADS_PER_BLOCK>>>(
        spectra, green, nx, ny, nz, nxm, nym, nx2, ny2, nz2, tensor_count, fft_count, 1, 2,
        4);
    cuda_status = cudaGetLastError();
    if (cuda_status == cudaSuccess)
        cuda_status = cudaDeviceSynchronize();

cleanup:
    if (cuda_status == cudaSuccess)
        std::fprintf(stderr, "[kernel.cu] GPU kernel executed: %s (%s precision)\n",
                     filtered ? "filtered_vector_green_tensor_generation"
                              : "vector_green_tensor_generation",
#if PRODUITMATVECTFFT_SINGLE_PRECISION
                     "single"
#else
                     "double"
#endif
        );
    return cuda_status;
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_vector_prepare_wrapper(const CUCOMPLEX *ix, const CUCOMPLEX *iy,
                                                         const CUCOMPLEX *iz, const CU_INT*map,
                                                         const CUCOMPLEX *alpha, CUCOMPLEX *px,
                                                         CUCOMPLEX *py, CUCOMPLEX *pz, int nfft,
                                                         int nx, int ny, int layer, int count) {
    if (!ix || !iy || !iz || !map || !alpha || !px || !py || !pz || nfft <= 0 || nx <= 0 ||
        ny <= 0 || layer < 1 || count < 4 * nfft * nfft)
        return cudaErrorInvalidValue;
    multilayer_vector_prepare_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        ix, iy, iz, map, alpha, px, py, pz, nfft, nx, ny, layer, count);
    return finish_gpbicg_launch(126, "multilayer_vector_prepare_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_vector_prepare_reflected_wrapper(
    const CUCOMPLEX *fx, const CUCOMPLEX *fy, const CUCOMPLEX *fz, const CUCOMPLEX *rx,
    const CUCOMPLEX *ry, const CUCOMPLEX *rz, const CU_INT*map, const CUCOMPLEX *alpha, CUCOMPLEX *px,
    CUCOMPLEX *py, CUCOMPLEX *pz, int nfft, int nx, int ny, int layer, int count) {
    if (!fx || !fy || !fz || !rx || !ry || !rz || !map || !alpha || !px || !py || !pz ||
        nfft <= 0 || nx <= 0 || ny <= 0 || layer < 1 || count < 4 * nfft * nfft)
        return cudaErrorInvalidValue;
    multilayer_vector_prepare_reflected_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        fx, fy, fz, rx, ry, rz, map, alpha, px, py, pz, nfft, nx, ny, layer, count);
    return finish_gpbicg_launch(129, "multilayer_vector_prepare_reflected_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_vector_product_wrapper(
    const CUCOMPLEX *px, const CUCOMPLEX *py, const CUCOMPLEX *pz, const CUCOMPLEX *gxx,
    const CUCOMPLEX *gxy, const CUCOMPLEX *gxz, const CUCOMPLEX *gyy, const CUCOMPLEX *gyz,
    const CUCOMPLEX *gzz, CUCOMPLEX *fx, CUCOMPLEX *fy, CUCOMPLEX *fz, int count, int reflected) {
    if (!px || !py || !pz || !gxx || !gxy || !gxz || !gyy || !gyz || !gzz || !fx || !fy || !fz ||
        count <= 0)
        return cudaErrorInvalidValue;
    multilayer_vector_product_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        px, py, pz, gxx, gxy, gxz, gyy, gyz, gzz, fx, fy, fz, count, reflected);
    return finish_gpbicg_launch(127, "multilayer_vector_product_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_vector_accumulate_wrapper(
    CUCOMPLEX *ix, CUCOMPLEX *iy, CUCOMPLEX *iz, const CUCOMPLEX *fx, const CUCOMPLEX *fy,
    const CUCOMPLEX *fz, CUCOMPLEX *local, const CU_INT*map, int nfft, int nx, int ny, int layer,
    int nmax, int count, CU_REAL image_scale, CU_REAL field_scale, int add_local) {
    if (!ix || !iy || !iz || !fx || !fy || !fz || !local || !map || nfft <= 0 || nx <= 0 ||
        ny <= 0 || layer < 0 || nmax <= 0 || count < nfft * nfft)
        return cudaErrorInvalidValue;
    multilayer_vector_accumulate_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        ix, iy, iz, fx, fy, fz, local, map, nfft, nx, ny, layer, nmax, count, image_scale,
        field_scale, add_local);
    return finish_gpbicg_launch(128, "multilayer_vector_accumulate_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_scalar_prepare_polarization_wrapper(
    const CUCOMPLEX *image, const CUCOMPLEX *add_image, const CU_INT*plane_map,
    const CUCOMPLEX *polarizability, CUCOMPLEX *polarization, int nfft2d, int nx, int ny, int layer,
    int plane_count) {
    if (!image || !plane_map || !polarizability || !polarization || nfft2d <= 0 || nx <= 0 ||
        ny <= 0 || layer < 1 || plane_count <= 0 || plane_count < nfft2d * nfft2d)
        return cudaErrorInvalidValue;
    multilayer_scalar_prepare_polarization_kernel<<<block_count(size_t(plane_count)),
                                                    THREADS_PER_BLOCK>>>(
        image, add_image, plane_map, polarizability, polarization, nfft2d, nx, ny, layer,
        plane_count);
    return finish_gpbicg_launch(115, "multilayer_scalar_prepare_polarization_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_scalar_map_first_plane_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, CU_INT*plane_map, int ndipole,
    int nfft2d, int kkm, int jjm, CU_REAL xmin, CU_REAL ymin, CU_REAL first_z, CU_REAL aretecube) {
    if (!xs || !ys || !zs || !plane_map || ndipole <= 0 || nfft2d <= 0 || aretecube <= CU_REAL(0))
        return cudaErrorInvalidValue;
    multilayer_scalar_map_first_plane_kernel<<<block_count(size_t(ndipole)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, plane_map, ndipole, nfft2d, kkm, jjm, xmin, ymin, first_z, aretecube);
    return finish_gpbicg_launch(132, "multilayer_scalar_map_first_plane_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_scalar_initialize_plane_wrapper(
    const CU_INT*plane_map, const CUCOMPLEX *incident_dipoles, CUCOMPLEX *image,
    CUCOMPLEX *local_field, int nfft2d, int nmax, int kkm, int jjm, CU_REAL xmin, CU_REAL ymin,
    CU_REAL first_z, CU_REAL aretecube, CU_REAL k0, CUCOMPLEX E0, CUCOMPLEX eps0, CU_REAL w0,
    CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees, CU_REAL phi_degrees, int beam_type,
    int conjugate_projection, CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz) {
    if (!plane_map || !incident_dipoles || !image || !local_field || nfft2d <= 0 || nmax <= 0 ||
        aretecube <= CU_REAL(0) || k0 <= CU_REAL(0) || beam_type < 0 || beam_type > 3 ||
        ((beam_type == 2 || beam_type == 3) && w0 <= CU_REAL(0)) ||
        (conjugate_projection != 0 && conjugate_projection != 1))
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft2d) * size_t(nfft2d);
    multilayer_scalar_initialize_plane_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        plane_map, incident_dipoles, image, local_field, nfft2d, nmax, kkm, jjm, xmin, ymin,
        first_z, aretecube, k0, E0, ss, pp, eps0, w0, theta_degrees, phi_degrees, beam_type,
        conjugate_projection, ux, uy, uz);
    return finish_gpbicg_launch(133, "multilayer_scalar_initialize_plane_kernel");
}

//-----------------------------------------------------------
__global__ void multilayer_scalar_initialize_rigorous_plane_kernel(
    const CU_INT*plane_map, const CUCOMPLEX *incident_dipoles,
    const CUCOMPLEX *rigorous_grid, CUCOMPLEX *image, CUCOMPLEX *local_field,
    int nfft2d, int nmax, int conjugate_projection,
    CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nfft2d * nfft2d;
    if (index >= count)
        return;
    const int local_dipole = plane_map[index].dat;
    CUCOMPLEX ex, ey, ez;
    if (local_dipole > 0) {
        if (local_dipole > nmax) {
            image[index] = CUFFT_ZERO;
            return;
        }
        const int base = 3 * (local_dipole - 1);
        ex = incident_dipoles[base];
        ey = incident_dipoles[base + 1];
        ez = incident_dipoles[base + 2];
        local_field[base] = ex;
        local_field[base + 1] = ey;
        local_field[base + 2] = ez;
    } else {
        const int base = 3 * index;
        ex = rigorous_grid[base];
        ey = rigorous_grid[base + 1];
        ez = rigorous_grid[base + 2];
    }
    CUCOMPLEX px = ux, py = uy, pz = uz;
    if (conjugate_projection) {
        px = CUFFT_CONJ(px);
        py = CUFFT_CONJ(py);
        pz = CUFFT_CONJ(pz);
    }
    image[index] = CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(px, ex), CUFFT_MUL(py, ey)),
                             CUFFT_MUL(pz, ez));
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_scalar_initialize_rigorous_plane_wrapper(
    const CU_INT*plane_map, const CUCOMPLEX *incident_dipoles,
    const CUCOMPLEX *rigorous_grid, CUCOMPLEX *image, CUCOMPLEX *local_field,
    int nfft2d, int nmax, int conjugate_projection,
    CUCOMPLEX ux, CUCOMPLEX uy, CUCOMPLEX uz) {
    if (!plane_map || !incident_dipoles || !rigorous_grid || !image || !local_field ||
        nfft2d <= 0 || nmax <= 0 || (conjugate_projection != 0 && conjugate_projection != 1))
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft2d) * size_t(nfft2d);
    multilayer_scalar_initialize_rigorous_plane_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        plane_map, incident_dipoles, rigorous_grid, image, local_field, nfft2d, nmax,
        conjugate_projection, ux, uy, uz);
    return finish_gpbicg_launch(248, "multilayer_scalar_initialize_rigorous_plane_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_vector_initialize_plane_wrapper(
    const CU_INT*plane_map, const CUCOMPLEX *incident_dipoles, CUCOMPLEX *image_x, CUCOMPLEX *image_y,
    CUCOMPLEX *image_z, CUCOMPLEX *local_field, int nfft2d, int nmax, int kkm, int jjm,
    CU_REAL xmin, CU_REAL ymin, CU_REAL first_z, CU_REAL aretecube, CU_REAL k0, CUCOMPLEX E0,
    CUCOMPLEX eps0, CU_REAL w0, CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees, CU_REAL phi_degrees,
    int beam_type) {
    if (!plane_map || !incident_dipoles || !image_x || !image_y || !image_z || !local_field ||
        nfft2d <= 0 || nmax <= 0 || aretecube <= CU_REAL(0) || k0 <= CU_REAL(0) || beam_type < 0 ||
        beam_type > 3 || ((beam_type == 2 || beam_type == 3) && w0 <= CU_REAL(0)))
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft2d) * size_t(nfft2d);
    multilayer_vector_initialize_plane_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        plane_map, incident_dipoles, image_x, image_y, image_z, local_field, nfft2d, nmax, kkm, jjm,
        xmin, ymin, first_z, aretecube, k0, E0, ss, pp, eps0, w0, theta_degrees, phi_degrees,
        beam_type);
    return finish_gpbicg_launch(134, "multilayer_vector_initialize_plane_kernel");
}

//-----------------------------------------------------------
__global__ void multilayer_vector_initialize_rigorous_plane_kernel(
    const CU_INT*plane_map, const CUCOMPLEX *incident_dipoles,
    const CUCOMPLEX *rigorous_grid, CUCOMPLEX *image_x, CUCOMPLEX *image_y,
    CUCOMPLEX *image_z, CUCOMPLEX *local_field, int nfft2d, int nmax) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nfft2d * nfft2d;
    if (index >= count)
        return;
    const int local_dipole = plane_map[index].dat;
    if (local_dipole > 0) {
        if (local_dipole > nmax) {
            image_x[index] = image_y[index] = image_z[index] = CUFFT_ZERO;
            return;
        }
        const int base = 3 * (local_dipole - 1);
        image_x[index] = incident_dipoles[base];
        image_y[index] = incident_dipoles[base + 1];
        image_z[index] = incident_dipoles[base + 2];
        local_field[base] = incident_dipoles[base];
        local_field[base + 1] = incident_dipoles[base + 1];
        local_field[base + 2] = incident_dipoles[base + 2];
        return;
    }
    const int base = 3 * index;
    image_x[index] = rigorous_grid[base];
    image_y[index] = rigorous_grid[base + 1];
    image_z[index] = rigorous_grid[base + 2];
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
multilayer_vector_initialize_rigorous_plane_wrapper(
    const CU_INT*plane_map, const CUCOMPLEX *incident_dipoles,
    const CUCOMPLEX *rigorous_grid, CUCOMPLEX *image_x, CUCOMPLEX *image_y,
    CUCOMPLEX *image_z, CUCOMPLEX *local_field, int nfft2d, int nmax) {
    if (!plane_map || !incident_dipoles || !rigorous_grid || !image_x || !image_y ||
        !image_z || !local_field || nfft2d <= 0 || nmax <= 0)
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft2d) * size_t(nfft2d);
    multilayer_vector_initialize_rigorous_plane_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        plane_map, incident_dipoles, rigorous_grid, image_x, image_y, image_z,
        local_field, nfft2d, nmax);
    return finish_gpbicg_launch(249, "multilayer_vector_initialize_rigorous_plane_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_scalar_multiply_green_wrapper(CUCOMPLEX *polarization,
                                                                const CUCOMPLEX *green, int count) {
    if (!polarization || !green || count <= 0)
        return cudaErrorInvalidValue;
    multilayer_scalar_multiply_green_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        polarization, green, count);
    return finish_gpbicg_launch(116, "multilayer_scalar_multiply_green_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_scalar_propagate_wrapper(CUCOMPLEX *image, int nfft2d,
                                                           CU_REAL deltakx, CU_REAL deltaky,
                                                           CU_REAL k0, CU_REAL aretecube) {
    if (!image || nfft2d <= 0 || k0 <= CU_REAL(0) || aretecube <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t count = size_t(nfft2d) * size_t(nfft2d);
    multilayer_scalar_propagate_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        image, nfft2d, deltakx, deltaky, k0, aretecube);
    return finish_gpbicg_launch(117, "multilayer_scalar_propagate_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_scalar_accumulate_wrapper(
    CUCOMPLEX *image, const CUCOMPLEX *scattered, CUCOMPLEX *local_field, const CU_INT*plane_map,
    int nfft2d, int nx, int ny, int layer, int nmax, int plane_count, CUCOMPLEX ux, CUCOMPLEX uy,
    CUCOMPLEX uz, CU_REAL image_scale, CU_REAL scattered_scale, int add_to_local) {
    if (!image || !scattered || !local_field || !plane_map || nfft2d <= 0 || nx <= 0 || ny <= 0 ||
        layer < 1 || nmax <= 0 || plane_count <= 0 || plane_count < nfft2d * nfft2d)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nfft2d) * size_t(nfft2d);
    multilayer_scalar_accumulate_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        image, scattered, local_field, plane_map, nfft2d, nx, ny, layer, nmax, plane_count, ux, uy,
        uz, image_scale, scattered_scale, add_to_local);
    return finish_gpbicg_launch(118, "multilayer_scalar_accumulate_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t multilayer_scalar_save_plane_wrapper(const CUCOMPLEX *image,
                                                            CUCOMPLEX *history, int image_count,
                                                            int layer, int layer_count) {
    if (!image || !history || image_count <= 0 || layer < 0 || layer >= layer_count ||
        layer_count <= 0)
        return cudaErrorInvalidValue;
    multilayer_scalar_save_plane_kernel<<<block_count(size_t(image_count)), THREADS_PER_BLOCK>>>(
        image, history, image_count, layer, layer_count);
    return finish_gpbicg_launch(119, "multilayer_scalar_save_plane_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_chan_transpose_wrapper(const CUCOMPLEX *fft_channel,
                                                     CUCOMPLEX *chan_vectors, int nx, int ny,
                                                     int nz) {
    if (!fft_channel || !chan_vectors || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(nx) * size_t(ny) * size_t(nz);
    scalar_chan_transpose_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        fft_channel, chan_vectors, nx, ny, nz);
    return finish_gpbicg_launch(101, "scalar_chan_transpose_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_chan_build_blocks_wrapper(const CUCOMPLEX *chan_vectors,
                                                        CUCOMPLEX *matrices, int nx, int ny,
                                                        int nz) {
    if (!chan_vectors || !matrices || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(nx) * size_t(ny) * size_t(nz) * size_t(nz);
    scalar_chan_build_blocks_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        chan_vectors, matrices, nx, ny, nz);
    return finish_gpbicg_launch(102, "scalar_chan_build_blocks_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_preconditioner_gather_wrapper(const CUCOMPLEX *field,
                                                            const CU_INT *dipole_map,
                                                            CUCOMPLEX *fft_planes, int nx, int ny,
                                                            int nz, int nxm, int nym, int nzm) {
    if (!field || !dipole_map || !fft_planes || nx <= 0 || ny <= 0 || nz <= 0 || nxm < nx ||
        nym < ny || nzm < nz)
        return cudaErrorInvalidValue;
    const size_t total = size_t(nx) * size_t(ny) * size_t(nz);
    scalar_preconditioner_gather_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        field, dipole_map, fft_planes, nx, ny, nz, nxm, nym, nzm);
    return finish_gpbicg_launch(103, "scalar_preconditioner_gather_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_preconditioner_fft_to_rhs_wrapper(const CUCOMPLEX *fft_planes,
                                                                CUCOMPLEX *rhs, int nx, int ny,
                                                                int nz) {
    if (!fft_planes || !rhs || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(nx) * size_t(ny) * size_t(nz);
    scalar_preconditioner_fft_to_rhs_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        fft_planes, rhs, nx, ny, nz);
    return finish_gpbicg_launch(104, "scalar_preconditioner_fft_to_rhs_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_preconditioner_rhs_to_fft_wrapper(const CUCOMPLEX *rhs,
                                                                CUCOMPLEX *fft_planes, int nx,
                                                                int ny, int nz) {
    if (!rhs || !fft_planes || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(nx) * size_t(ny) * size_t(nz);
    scalar_preconditioner_rhs_to_fft_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        rhs, fft_planes, nx, ny, nz);
    return finish_gpbicg_launch(105, "scalar_preconditioner_rhs_to_fft_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_preconditioner_matvec_wrapper(const CUCOMPLEX *matrices,
                                                            const CUCOMPLEX *rhs, CUCOMPLEX *output,
                                                            int nx, int ny, int nz) {
    if (!matrices || !rhs || !output || nx <= 0 || ny <= 0 || nz <= 0)
        return cudaErrorInvalidValue;
    const size_t total = size_t(nx) * size_t(ny) * size_t(nz);
    scalar_preconditioner_matvec_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        matrices, rhs, output, nx, ny, nz);
    return finish_gpbicg_launch(107, "scalar_preconditioner_matvec_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t scalar_preconditioner_scatter_wrapper(const CUCOMPLEX *fft_planes,
                                                             const CU_INT *dipole_map,
                                                             CUCOMPLEX *field, int nx, int ny,
                                                             int nz, int nxm, int nym, int nzm,
                                                             CU_REAL normalization) {
    if (!fft_planes || !dipole_map || !field || nx <= 0 || ny <= 0 || nz <= 0 || nxm < nx ||
        nym < ny || nzm < nz)
        return cudaErrorInvalidValue;
    const size_t total = size_t(nx) * size_t(ny) * size_t(nz);
    scalar_preconditioner_scatter_kernel<<<block_count(total), THREADS_PER_BLOCK>>>(
        fft_planes, dipole_map, field, nx, ny, nz, nxm, nym, nzm, normalization);
    return finish_gpbicg_launch(106, "scalar_preconditioner_scatter_kernel");
}

namespace {

//-----------------------------------------------------------
__device__ inline bool rytov_finite(CUCOMPLEX value) {
    return isfinite(value.x) && isfinite(value.y);
}

//-----------------------------------------------------------
__global__ void rytov_vector_field_kernel(const CUCOMPLEX *eref, const CUCOMPLEX *delta,
                                          CUCOMPLEX *result, CU_INT*error_flag, int dipole_count) {
    const int dipole = blockIdx.x * blockDim.x + threadIdx.x;
    if (dipole >= dipole_count)
        return;
    const int base = 3 * dipole;
    const CUCOMPLEX ex = eref[base];
    const CUCOMPLEX ey = eref[base + 1];
    const CUCOMPLEX ez = eref[base + 2];
    const CUCOMPLEX dx = delta[base];
    const CUCOMPLEX dy = delta[base + 1];
    const CUCOMPLEX dz = delta[base + 2];
    const CU_REAL en2 =
        (ex.x * ex.x + ex.y * ex.y) + (ey.x * ey.x + ey.y * ey.y) + (ez.x * ez.x + ez.y * ez.y);
    if (!isfinite(en2)) {
        atomicExch(&error_flag->dat, 1);
        return;
    }

#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CU_REAL tiny_value = CU_REAL(FLT_MIN);
    const CU_REAL huge_value = CU_REAL(FLT_MAX);
    const CU_REAL one = CU_REAL(1.0f);
#else
    const CU_REAL tiny_value = CU_REAL(DBL_MIN);
    const CU_REAL huge_value = CU_REAL(DBL_MAX);
    const CU_REAL one = CU_REAL(1.0);
#endif
    CUCOMPLEX rx, ry, rz;
    if (en2 <= tiny_value) {
        rx = CUFFT_ADD(ex, dx);
        ry = CUFFT_ADD(ey, dy);
        rz = CUFFT_ADD(ez, dz);
    } else {
        CUCOMPLEX q = CUFFT_ZERO;
        q = CUFFT_ADD(q, CUFFT_MUL(CUFFT_CONJ(ex), dx));
        q = CUFFT_ADD(q, CUFFT_MUL(CUFFT_CONJ(ey), dy));
        q = CUFFT_ADD(q, CUFFT_MUL(CUFFT_CONJ(ez), dz));
        q = CUFFT_DIV(q, CUFFT_REAL(en2));
        if (!isfinite(q.x) || !isfinite(q.y)) {
            atomicExch(&error_flag->dat, 1);
            return;
        }
        const CU_REAL max_amp = sqrt(en2);
        const CU_REAL exponent_limit =
            log(sqrt(huge_value)) - log(fmax(one, max_amp)) - CU_REAL(2.0);
        CUCOMPLEX nonlinear;
        if (q.x > exponent_limit) {
            nonlinear = CUFFT_ZERO;
        } else {
            const CU_REAL exp_real = exp(q.x);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            const CUCOMPLEX exp_value =
                make_cuFloatComplex(exp_real * cosf(q.y), exp_real * sinf(q.y));
#else
            const CUCOMPLEX exp_value =
                make_cuDoubleComplex(exp_real * cos(q.y), exp_real * sin(q.y));
#endif
            nonlinear = CUFFT_SUB(CUFFT_SUB(exp_value, CUFFT_REAL(one)), q);
        }
        rx = CUFFT_ADD(CUFFT_ADD(ex, dx), CUFFT_MUL(ex, nonlinear));
        ry = CUFFT_ADD(CUFFT_ADD(ey, dy), CUFFT_MUL(ey, nonlinear));
        rz = CUFFT_ADD(CUFFT_ADD(ez, dz), CUFFT_MUL(ez, nonlinear));
    }
    if (!rytov_finite(rx) || !rytov_finite(ry) || !rytov_finite(rz)) {
        atomicExch(&error_flag->dat, 1);
        return;
    }
    result[base] = rx;
    result[base + 1] = ry;
    result[base + 2] = rz;
}

} // namespace

//-----------------------------------------------------------
extern "C" cudaError_t rytov_vector_field_wrapper(const CUCOMPLEX *eref, const CUCOMPLEX *delta,
                                                  CUCOMPLEX *result, CU_INT*error_flag,
                                                  int dipole_count) {
    if (!eref || !delta || !result || !error_flag || dipole_count <= 0)
        return cudaErrorInvalidValue;
    rytov_vector_field_kernel<<<block_count(size_t(dipole_count)), THREADS_PER_BLOCK>>>(
        eref, delta, result, error_flag, dipole_count);
    return finish_gpbicg_launch(114, "rytov_vector_field_kernel");
}

namespace {

//-----------------------------------------------------------
__global__ void beampropagation_spectral_kernel(CUCOMPLEX *image_planes, int nfft2d,
                                                CU_REAL aretecube, CU_REAL k0) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft2d * nfft2d;
    if (index >= plane)
        return;
    const int ix = index % nfft2d;
    const int iy = index / nfft2d;
    const int signed_ix = ix >= nfft2d / 2 ? ix - nfft2d : ix;
    const int signed_iy = iy >= nfft2d / 2 ? iy - nfft2d : iy;
    const CU_REAL delta_k =
        CU_REAL(2.0) * CU_REAL(3.14159265358979323846) / (CU_REAL(nfft2d) * aretecube);
    const CU_REAL kx = delta_k * CU_REAL(signed_ix);
    const CU_REAL ky = delta_k * CU_REAL(signed_iy);
    const CU_REAL kz2 = k0 * k0 - kx * kx - ky * ky;
    CUCOMPLEX phase = CUFFT_ZERO;
    if (kz2 >= CU_REAL(0)) {
        const CU_REAL kz = sqrt(kz2);
        const CU_REAL angle = -(k0 - kz) * aretecube;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        phase = make_cuFloatComplex(cosf(angle), sinf(angle));
#else
        phase = make_cuDoubleComplex(cos(angle), sin(angle));
#endif
    }
    for (int component = 0; component < 3; ++component) {
        CUCOMPLEX &value = image_planes[component * plane + index];
        value = CUFFT_MUL(value, phase);
    }
}

//-----------------------------------------------------------
__device__ CUCOMPLEX beampropagation_material_phase(CUCOMPLEX epsilon, CU_REAL scale) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CU_REAL magnitude = hypotf(epsilon.x, epsilon.y);
    const CU_REAL root_real = sqrtf(fmaxf(CU_REAL(0), (magnitude + epsilon.x) * CU_REAL(0.5)));
    const CU_REAL root_imag =
        copysignf(sqrtf(fmaxf(CU_REAL(0), (magnitude - epsilon.x) * CU_REAL(0.5))), epsilon.y);
    const CU_REAL attenuation = expf(-scale * root_imag);
    const CU_REAL angle = scale * root_real;
    return make_cuFloatComplex(attenuation * cosf(angle), attenuation * sinf(angle));
#else
    const CU_REAL magnitude = hypot(epsilon.x, epsilon.y);
    const CU_REAL root_real = sqrt(fmax(CU_REAL(0), (magnitude + epsilon.x) * CU_REAL(0.5)));
    const CU_REAL root_imag =
        copysign(sqrt(fmax(CU_REAL(0), (magnitude - epsilon.x) * CU_REAL(0.5))), epsilon.y);
    const CU_REAL attenuation = exp(-scale * root_imag);
    const CU_REAL angle = scale * root_real;
    return make_cuDoubleComplex(attenuation * cos(angle), attenuation * sin(angle));
#endif
}

//-----------------------------------------------------------
__device__ BeamPropagationIncidentField beampropagation_plane_field(
    CUCOMPLEX E0, CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees, CU_REAL phi_degrees, CU_REAL x,
    CU_REAL y, CU_REAL z, CU_REAL k0, CU_REAL xgaus, CU_REAL ygaus, CU_REAL zgaus, CU_REAL w0,
    int beam_type) {
    const CU_REAL theta = theta_degrees * (CU_REAL(3.14159265358979323846) / CU_REAL(180));
    const CU_REAL phi = phi_degrees * (CU_REAL(3.14159265358979323846) / CU_REAL(180));
    const CU_REAL sin_theta = sin(theta);
    const CU_REAL cos_theta = cos(theta);
    const CU_REAL sin_phi = sin(phi);
    const CU_REAL cos_phi = cos(phi);

    if (beam_type == 2 || beam_type == 3) {
        const CU_REAL vx = sin_theta * cos_phi;
        const CU_REAL vy = sin_theta * sin_phi;
        const CU_REAL vz = cos_theta;
        const CU_REAL eps = CU_REAL(1.0e-12);
        CU_REAL m00 = CU_REAL(1), m01 = CU_REAL(0), m02 = CU_REAL(0);
        CU_REAL m10 = CU_REAL(0), m11 = CU_REAL(1), m12 = CU_REAL(0);
        CU_REAL m20 = CU_REAL(0), m21 = CU_REAL(0), m22 = CU_REAL(1);
        if (vx * vx + vy * vy > eps) {
            m00 = CU_REAL(1) - vx * vx / (CU_REAL(1) + vz);
            m01 = -vx * vy / (CU_REAL(1) + vz);
            m02 = -vx;
            m10 = m01;
            m11 = CU_REAL(1) - vy * vy / (CU_REAL(1) + vz);
            m12 = -vy;
            m20 = vx;
            m21 = vy;
            m22 = vz;
        }

        CUCOMPLEX e0s, e0p;
        if (beam_type == 3) {
            const CU_REAL us = CU_REAL(1) / sqrt(CU_REAL(2));
            e0s = CUFFT_MUL(E0, CUFFT_REAL(us));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            e0p = CUFFT_MUL(E0, make_cuFloatComplex(CU_REAL(0), ss < CU_REAL(0) ? -us : us));
#else
            e0p = CUFFT_MUL(E0, make_cuDoubleComplex(CU_REAL(0), ss < CU_REAL(0) ? -us : us));
#endif
        } else {
            e0s = CUFFT_MUL(E0, CUFFT_REAL(sqrt(CU_REAL(1) - pp)));
            e0p = CUFFT_MUL(E0, CUFFT_REAL(sqrt(pp)));
        }
        const CUCOMPLEX e0x = CUFFT_ADD(CUFFT_MUL(e0s, CUFFT_REAL(-sin_phi)),
                                        CUFFT_MUL(e0p, CUFFT_REAL(cos_theta * cos_phi)));
        const CUCOMPLEX e0y = CUFFT_ADD(CUFFT_MUL(e0s, CUFFT_REAL(cos_phi)),
                                        CUFFT_MUL(e0p, CUFFT_REAL(cos_theta * sin_phi)));
        const CUCOMPLEX e0z = CUFFT_MUL(e0p, CUFFT_REAL(-sin_theta));
        const CUCOMPLEX e00x =
            CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(e0x, CUFFT_REAL(m00)), CUFFT_MUL(e0y, CUFFT_REAL(m01))),
                      CUFFT_MUL(e0z, CUFFT_REAL(m02)));
        const CUCOMPLEX e00y =
            CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(e0x, CUFFT_REAL(m10)), CUFFT_MUL(e0y, CUFFT_REAL(m11))),
                      CUFFT_MUL(e0z, CUFFT_REAL(m12)));
        const CU_REAL rx = x - xgaus;
        const CU_REAL ry = y - ygaus;
        const CU_REAL rz = z - zgaus;
        const CU_REAL xx = m00 * rx + m01 * ry + m02 * rz;
        const CU_REAL yy = m10 * rx + m11 * ry + m12 * rz;
        const CU_REAL zz = m20 * rx + m21 * ry + m22 * rz;
        const CU_REAL w02 = w0 * w0;
        const CU_REAL z0 = k0 * w02;
        const CU_REAL w2 = CU_REAL(2) * w02 * (CU_REAL(1) + zz * zz / (z0 * z0));
        const CU_REAL width = sqrt(w2);
        const CU_REAL eta = atan2(zz, z0);
        const CU_REAL rinv = zz / (zz * zz + z0 * z0);
        const CU_REAL r2 = xx * xx + yy * yy;
        const CU_REAL amplitude = sqrt(CU_REAL(2)) * w0 / width * exp(-r2 / w2);
        const CU_REAL angle = k0 * zz + eta + k0 * r2 * rinv / CU_REAL(2);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CUCOMPLEX phase =
            make_cuFloatComplex(amplitude * cosf(angle), amplitude * sinf(angle));
#else
        const CUCOMPLEX phase =
            make_cuDoubleComplex(amplitude * cos(angle), amplitude * sin(angle));
#endif
        const CUCOMPLEX epx = CUFFT_MUL(e00x, phase);
        const CUCOMPLEX epy = CUFFT_MUL(e00y, phase);
        return {CUFFT_ADD(CUFFT_MUL(epx, CUFFT_REAL(m00)), CUFFT_MUL(epy, CUFFT_REAL(m10))),
                CUFFT_ADD(CUFFT_MUL(epx, CUFFT_REAL(m01)), CUFFT_MUL(epy, CUFFT_REAL(m11))),
                CUFFT_ADD(CUFFT_MUL(epx, CUFFT_REAL(m02)), CUFFT_MUL(epy, CUFFT_REAL(m12)))};
    }

    CUCOMPLEX e0s, e0p;
    if (beam_type == 1) {
        const CU_REAL inverse_sqrt_two = CU_REAL(1) / sqrt(CU_REAL(2));
        e0s = CUFFT_MUL(E0, CUFFT_REAL(inverse_sqrt_two));
        const CU_REAL signed_p = ss < CU_REAL(0) ? -inverse_sqrt_two : inverse_sqrt_two;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        e0p = CUFFT_MUL(E0, make_cuFloatComplex(CU_REAL(0), signed_p));
#else
        e0p = CUFFT_MUL(E0, make_cuDoubleComplex(CU_REAL(0), signed_p));
#endif
    } else {
        CU_REAL p = sqrt(pp);
        CU_REAL s = sqrt(CU_REAL(1) - pp);
        if (fabs(pp - CU_REAL(3)) <= CU_REAL(3.0e-12)) {
            pp = (sin_phi * sin_phi) /
                 (cos_theta * cos_theta * cos_phi * cos_phi + sin_phi * sin_phi);
            p = sqrt(pp);
            s = sqrt(CU_REAL(1) - pp);
            if (phi_degrees < CU_REAL(0))
                phi_degrees += CU_REAL(360);
            if (phi_degrees >= CU_REAL(90) && phi_degrees < CU_REAL(180))
                s = -s;
            //-----------------------------------------------------------
            else if (phi_degrees >= CU_REAL(180) && phi_degrees < CU_REAL(270)) {
                p = -p;
                s = sqrt(CU_REAL(1) - pp);
            } else if (phi_degrees >= CU_REAL(270) && phi_degrees < CU_REAL(360))
                p = -p;
        } else if (fabs(pp - CU_REAL(2)) <= CU_REAL(2.0e-12)) {
            pp = (cos_phi * cos_phi) /
                 (cos_theta * cos_theta * sin_phi * sin_phi + cos_phi * cos_phi);
            p = sqrt(pp);
            s = -sqrt(CU_REAL(1) - pp);
            if (phi_degrees < CU_REAL(0))
                phi_degrees += CU_REAL(360);
            if (phi_degrees >= CU_REAL(0) && phi_degrees < CU_REAL(90)) {
            } else if (phi_degrees >= CU_REAL(90) && phi_degrees < CU_REAL(180)) {
                p = -p;
                s = -s;
            } else if (phi_degrees >= CU_REAL(180) && phi_degrees < CU_REAL(270)) {
                p = -p;
                s = sqrt(CU_REAL(1) - pp);
            } else if (phi_degrees >= CU_REAL(270) && phi_degrees < CU_REAL(360)) {
                s = -s;
            }
        }
        const CU_REAL sign_theta = theta < CU_REAL(0) ? CU_REAL(-1) : CU_REAL(1);
        e0s = CUFFT_MUL(E0, CUFFT_REAL(s * sign_theta));
        e0p = CUFFT_MUL(E0, CUFFT_REAL(p * sign_theta));
    }

    const CU_REAL kx = sin_theta * cos_phi;
    const CU_REAL ky = sin_theta * sin_phi;
    const CU_REAL kz = cos_theta;
    const CU_REAL angle = k0 * (kx * x + ky * y + kz * z);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cosf(angle), sinf(angle));
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(angle), sin(angle));
#endif
    CUCOMPLEX ex, ey, ez;
    if (beam_type == 1) {
        const CUCOMPLEX e0x_s = CUFFT_MUL(e0s, CUFFT_REAL(-sin_phi));
        const CUCOMPLEX e0y_s = CUFFT_MUL(e0s, CUFFT_REAL(cos_phi));
        const CUCOMPLEX e0x_p = CUFFT_MUL(e0p, CUFFT_REAL(cos_theta * cos_phi));
        const CUCOMPLEX e0y_p = CUFFT_MUL(e0p, CUFFT_REAL(cos_theta * sin_phi));
        const CUCOMPLEX e0z_p = CUFFT_MUL(e0p, CUFFT_REAL(-sin_theta));
        ex = CUFFT_MUL(CUFFT_ADD(e0x_s, e0x_p), phase);
        ey = CUFFT_MUL(CUFFT_ADD(e0y_s, e0y_p), phase);
        ez = CUFFT_MUL(e0z_p, phase);
    } else {
        ex = CUFFT_MUL(CUFFT_ADD(CUFFT_MUL(e0s, CUFFT_REAL(-sin_phi)),
                                 CUFFT_MUL(e0p, CUFFT_REAL(cos_theta * cos_phi))),
                       phase);
        ey = CUFFT_MUL(CUFFT_ADD(CUFFT_MUL(e0s, CUFFT_REAL(cos_phi)),
                                 CUFFT_MUL(e0p, CUFFT_REAL(cos_theta * sin_phi))),
                       phase);
        ez = CUFFT_MUL(CUFFT_MUL(e0p, CUFFT_REAL(-sin_theta)), phase);
    }
    return {ex, ey, ez};
}

//-----------------------------------------------------------
__global__ void beampropagation_initialize_first_plane_kernel(
    const CU_INT*first_plane_map, const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
    const CUCOMPLEX *epsilon, const CUCOMPLEX *incident, CUCOMPLEX *image_planes,
    CUCOMPLEX *local_field, int nx, int ny, int nfft2d, int nmax, int niso, int kkm, int jjm,
    CU_REAL xmin, CU_REAL ymin, CU_REAL first_z, CU_REAL aretecube, CU_REAL k0, CUCOMPLEX E0,
    CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees, CU_REAL phi_degrees, CU_REAL xgaus,
    CU_REAL ygaus, CU_REAL zgaus, CU_REAL w0, int beam_type) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft2d * nfft2d;
    if (index >= plane)
        return;
    const int ix = index % nfft2d;
    const int iy = index / nfft2d;
    const CU_REAL x = xmin + CU_REAL(ix - kkm) * aretecube;
    const CU_REAL y = ymin + CU_REAL(iy - jjm) * aretecube;
    const int map_value = first_plane_map[index].dat;
    if (map_value > 0) {
        const int dipole = map_value - 1;
        if (dipole >= nmax)
            return;
        const CUCOMPLEX background_inverse_phase =
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            make_cuFloatComplex(cosf(k0 * aretecube), -sinf(k0 * aretecube));
#else
            make_cuDoubleComplex(cos(k0 * aretecube), -sin(k0 * aretecube));
#endif
        for (int component = 0; component < 3; ++component) {
            const size_t eps_stride = size_t(nmax) * size_t(niso + 1);
            const size_t eps_index =
                size_t(dipole) + (niso == 1 ? size_t(0) : size_t(component) * eps_stride);
            const CUCOMPLEX material_phase =
                beampropagation_material_phase(epsilon[eps_index], k0 * aretecube);
            const CUCOMPLEX phase = CUFFT_MUL(material_phase, background_inverse_phase);
            const CUCOMPLEX value =
                CUFFT_MUL(incident[size_t(3) * size_t(dipole) + size_t(component)], phase);
            image_planes[size_t(component) * plane + size_t(index)] = value;
            local_field[size_t(3) * size_t(dipole) + size_t(component)] = value;
        }
    } else {
        const BeamPropagationIncidentField field =
            beampropagation_plane_field(E0, ss, pp, theta_degrees, phi_degrees, x, y, first_z, k0,
                                        xgaus, ygaus, zgaus, w0, beam_type);
        image_planes[index] = field.x;
        image_planes[size_t(plane) + size_t(index)] = field.y;
        image_planes[size_t(2) * size_t(plane) + size_t(index)] = field.z;
    }
    (void)nx;
    (void)ny;
}

//-----------------------------------------------------------
__global__ void beampropagation_map_first_plane_kernel(const CU_REAL *xs, const CU_REAL *ys,
                                                       const CU_REAL *zs, CU_INT*first_plane_map,
                                                       int nfft2d, int ndipole, int kkm, int jjm,
                                                       CU_REAL xmin, CU_REAL ymin, CU_REAL first_z,
                                                       CU_REAL aretecube) {
    const int dipole = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (dipole >= ndipole)
        return;
    const CU_REAL tolerance = aretecube / CU_REAL(1000);
    if (fabs(zs[dipole] - first_z) > tolerance)
        return;
    const long long x_offset = llround((xs[dipole] - xmin) / aretecube);
    const long long y_offset = llround((ys[dipole] - ymin) / aretecube);
    const long long ix = x_offset + kkm;
    const long long iy = y_offset + jjm;
    if (ix < 0 || ix >= nfft2d || iy < 0 || iy >= nfft2d)
        return;
    const CU_REAL expected_x = xmin + CU_REAL(x_offset) * aretecube;
    const CU_REAL expected_y = ymin + CU_REAL(y_offset) * aretecube;
    if (fabs(xs[dipole] - expected_x) > tolerance || fabs(ys[dipole] - expected_y) > tolerance)
        return;
    atomicExch(&first_plane_map[int(ix) + nfft2d * int(iy)].dat, dipole + 1);
}

//-----------------------------------------------------------
__global__ void beampropagation_update_plane_kernel(CUCOMPLEX *image_planes,
                                                    const CU_INT*first_plane_map,
                                                    const CUCOMPLEX *epsilon,
                                                    CUCOMPLEX *local_field, int nx, int ny,
                                                    int nfft2d, int nmax, int niso, int z_layer,
                                                    CU_REAL aretecube, CU_REAL k0) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft2d * nfft2d;
    if (index >= plane)
        return;
    const int xy = index % nfft2d + nfft2d * (index / nfft2d);
    const CU_REAL normalization = CU_REAL(1) / CU_REAL(plane);
    const CUCOMPLEX background_phase =
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        make_cuFloatComplex(cosf(k0 * aretecube), sinf(k0 * aretecube));
#else
        make_cuDoubleComplex(cos(k0 * aretecube), sin(k0 * aretecube));
#endif
    const int map_value = first_plane_map[xy].dat;
    if (map_value > 0) {
        const long long dipole = static_cast<long long>(nx) * ny * z_layer + map_value - 1;
        if (dipole < 0 || dipole >= nmax)
            return;
        const size_t eps_stride = size_t(nmax) * size_t(niso + 1);
        const size_t field_offset = size_t(3) * size_t(dipole);
        for (int component = 0; component < 3; ++component) {
            const size_t eps_index =
                size_t(dipole) + (niso == 1 ? size_t(0) : size_t(component) * eps_stride);
            const CUCOMPLEX phase =
                beampropagation_material_phase(epsilon[eps_index], k0 * aretecube);
            const CUCOMPLEX value =
                CUFFT_MUL(CUFFT_MUL(image_planes[size_t(component) * plane + index], phase),
                          CUFFT_REAL(normalization));
            local_field[field_offset + size_t(component)] = value;
            image_planes[size_t(component) * plane + index] = value;
        }
    } else {
        const CUCOMPLEX phase = CUFFT_MUL(background_phase, CUFFT_REAL(normalization));
        for (int component = 0; component < 3; ++component)
            image_planes[size_t(component) * plane + index] =
                CUFFT_MUL(image_planes[size_t(component) * plane + index], phase);
    }
    (void)ny;
}

//-----------------------------------------------------------
__global__ void beampropagation_local_to_macro_kernel(const CUCOMPLEX *epsilon,
                                                      CUCOMPLEX *local_field, int nmax, int niso,
                                                      int ndipole, CU_REAL aretecube, CU_REAL k0) {
    const int dipole = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (dipole >= ndipole)
        return;
    const CU_REAL constant = k0 * k0 * k0 * aretecube * aretecube * aretecube /
                             (CU_REAL(2.0) * CU_REAL(3.14159265358979323846));
    const size_t stride = size_t(nmax);
    const size_t tensor_stride = stride * size_t(niso);
    const size_t base = size_t(3) * size_t(dipole);
    CUCOMPLEX old_field[3] = {local_field[base], local_field[base + 1], local_field[base + 2]};
    CUCOMPLEX updated[3] = {CUFFT_ZERO, CUFFT_ZERO, CUFFT_ZERO};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            CUCOMPLEX eps_value = CUFFT_ZERO;
            if (niso == 1) {
                if (row == col)
                    eps_value = epsilon[dipole];
            } else {
                eps_value =
                    epsilon[size_t(dipole) + size_t(row) * stride + size_t(col) * tensor_stride];
            }
            const CU_REAL identity = row == col ? CU_REAL(1) : CU_REAL(0);
            const CUCOMPLEX diagonal_adjusted =
                CUFFT_ADD(eps_value, CUFFT_REAL(row == col ? CU_REAL(2) : CU_REAL(0)));
            const CUCOMPLEX contrast = CUFFT_SUB(eps_value, CUFFT_REAL(identity));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            const CUCOMPLEX radiation =
                make_cuFloatComplex(constant * contrast.y, -constant * contrast.x);
#else
            const CUCOMPLEX radiation =
                make_cuDoubleComplex(constant * contrast.y, -constant * contrast.x);
#endif
            const CUCOMPLEX matrix_value = CUFFT_ADD(diagonal_adjusted, radiation);
            updated[row] = CUFFT_ADD(updated[row], CUFFT_MUL(matrix_value, old_field[col]));
        }
        updated[row] = CUFFT_DIV(updated[row], CUFFT_REAL(CU_REAL(3)));
    }
    local_field[base] = updated[0];
    local_field[base + 1] = updated[1];
    local_field[base + 2] = updated[2];
}

//-----------------------------------------------------------
__global__ void gaussianfft_filter_kernel(CUCOMPLEX *image_planes, int nfft2d, CU_REAL aretecube,
                                          CU_REAL k0, int first_plane) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft2d * nfft2d;
    if (index >= plane)
        return;
    const int ix = index % nfft2d;
    const int iy = index / nfft2d;
    const int signed_ix = ix >= nfft2d / 2 ? ix - nfft2d : ix;
    const int signed_iy = iy >= nfft2d / 2 ? iy - nfft2d : iy;
    const CU_REAL delta_k =
        CU_REAL(2.0) * CU_REAL(3.14159265358979323846) / (CU_REAL(nfft2d) * aretecube);
    const CU_REAL kx = delta_k * CU_REAL(signed_ix);
    const CU_REAL ky = delta_k * CU_REAL(signed_iy);
    const CU_REAL kz2 = k0 * k0 - kx * kx - ky * ky;
    CUCOMPLEX factor = CUFFT_ZERO;
    const bool propagating = first_plane ? kz2 > CU_REAL(0) : kz2 >= CU_REAL(0);
    if (propagating) {
        const CU_REAL normalization = CU_REAL(1) / CU_REAL(plane);
        if (first_plane)
            factor = CUFFT_REAL(normalization);
        else {
            const CU_REAL angle = sqrt(kz2) * aretecube;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            factor = make_cuFloatComplex(cosf(angle) * normalization, sinf(angle) * normalization);
#else
            factor = make_cuDoubleComplex(cos(angle) * normalization, sin(angle) * normalization);
#endif
        }
    }
    for (int component = 0; component < 3; ++component) {
        CUCOMPLEX &value = image_planes[size_t(component) * size_t(plane) + size_t(index)];
        value = CUFFT_MUL(value, factor);
    }
}

//-----------------------------------------------------------
__global__ void gaussianfft_scatter_kernel(const CUCOMPLEX *image_planes, const CU_INT*plane_map,
                                           CUCOMPLEX *dipole_field, int nfft2d, int plane_count,
                                           int layer, size_t mesh_count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft2d * nfft2d;
    if (index >= plane)
        return;
    const int local_dipole = plane_map[index].dat - 1;
    if (local_dipole < 0)
        return;
    const size_t dipole = size_t(layer) * size_t(plane_count) + size_t(local_dipole);
    if (dipole >= mesh_count)
        return;
    for (int component = 0; component < 3; ++component)
        dipole_field[3 * dipole + size_t(component)] =
            image_planes[size_t(component) * size_t(plane) + size_t(index)];
}

//-----------------------------------------------------------
__global__ void gaussianfft_coordinates_kernel(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, int nfft2d,
                                               int nx, int ny, CU_REAL xmin, CU_REAL ymin,
                                               CU_REAL z, CU_REAL aretecube) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nfft2d * nfft2d;
    if (index >= count)
        return;
    const int column = index % nfft2d;
    const int row = index / nfft2d;
    xs[index] = xmin + CU_REAL(column - (nfft2d - nx) / 2) * aretecube;
    ys[index] = ymin + CU_REAL(row - (nfft2d - ny) / 2) * aretecube;
    zs[index] = z;
}

//-----------------------------------------------------------
__global__ void gaussianfft_build_plane_map_kernel(const CU_REAL *dipole_xs,
                                                   const CU_REAL *dipole_ys, CU_INT*plane_map,
                                                   int nfft2d, int dipole_count, int kkm, int jjm,
                                                   CU_REAL xmin, CU_REAL ymin, CU_REAL aretecube) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const CU_REAL tolerance = aretecube / CU_REAL(1000);
    int dipole = 0;
    const int plane = nfft2d * nfft2d;
    for (int index = 0; index < plane; ++index) {
        const int col = index % nfft2d;
        const int row = index / nfft2d;
        const CU_REAL x = xmin + CU_REAL(col - kkm) * aretecube;
        const CU_REAL y = ymin + CU_REAL(row - jjm) * aretecube;
        int mapped = 0;
        if (dipole < dipole_count && x >= dipole_xs[dipole] - tolerance &&
            x <= dipole_xs[dipole] + tolerance && y >= dipole_ys[dipole] - tolerance &&
            y <= dipole_ys[dipole] + tolerance) {
            mapped = dipole + 1;
            ++dipole;
        }
        plane_map[index].dat = mapped;
    }
}

//-----------------------------------------------------------
__global__ void gaussianfft_max_intensity_partials_kernel(const CUCOMPLEX *field, CU_REAL *partials,
                                                          int count) {
    __shared__ CU_REAL values[THREADS_PER_BLOCK];
    const int lane = int(threadIdx.x);
    const int index = int(blockIdx.x) * THREADS_PER_BLOCK + lane;
    CU_REAL intensity = CU_REAL(0);
    if (index < count)
        for (int component = 0; component < 3; ++component) {
            const CUCOMPLEX value = field[3 * index + component];
            intensity += value.x * value.x + value.y * value.y;
        }
    values[lane] = intensity;
    __syncthreads();
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (lane < stride)
            values[lane] = fmax(values[lane], values[lane + stride]);
        __syncthreads();
    }
    if (lane == 0)
        partials[blockIdx.x] = values[0];
}

//-----------------------------------------------------------
__global__ void gaussianfft_reduce_max_partials_kernel(const CU_REAL *partials, CU_REAL *maximum,
                                                       int partial_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    CU_REAL result = CU_REAL(0);
    for (int i = 0; i < partial_count; ++i)
        result = fmax(result, partials[i]);
    *maximum = result;
}

//-----------------------------------------------------------
__global__ void gaussianfft_blend_first_plane_kernel(const CUCOMPLEX *paraxial,
                                                     const CUCOMPLEX *rigorous,
                                                     const CU_REAL *max_intensity,
                                                     CU_REAL tolerance, int circular,
                                                     CUCOMPLEX *image_planes, int count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    CU_REAL intensity = CU_REAL(0);
    for (int component = 0; component < 3; ++component) {
        const CUCOMPLEX value = paraxial[3 * index + component];
        intensity += value.x * value.x + value.y * value.y;
    }
    const CU_REAL maximum = *max_intensity;
    // Fortran gaussianfft.f90: linear uses sqrt(I/Imax) >= tol;
    // circular uses 10*sqrt(I/Imax) >= tol.
    const CU_REAL multiplier = circular ? CU_REAL(10) : CU_REAL(1);
    const bool use_rigorous =
        maximum > CU_REAL(0) && sqrt(intensity / maximum) * multiplier >= tolerance;
    for (int component = 0; component < 3; ++component)
        image_planes[size_t(component) * size_t(count) + size_t(index)] =
            use_rigorous ? rigorous[3 * index + component] : paraxial[3 * index + component];
}

//-----------------------------------------------------------
__global__ void beampropagation_blend_gaussian_background_kernel(
    const CU_INT*first_plane_map, const CUCOMPLEX *paraxial, const CUCOMPLEX *rigorous,
    const CU_REAL *max_intensity, CU_REAL threshold, CUCOMPLEX *image_planes, int count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count || first_plane_map[index].dat != 0)
        return;
    CU_REAL intensity = CU_REAL(0);
    for (int component = 0; component < 3; ++component) {
        const CUCOMPLEX value = paraxial[3 * index + component];
        intensity += value.x * value.x + value.y * value.y;
    }
    const bool use_rigorous =
        *max_intensity > CU_REAL(0) && sqrt(intensity / *max_intensity) * CU_REAL(10) >= threshold;
    for (int component = 0; component < 3; ++component)
        image_planes[size_t(component) * size_t(count) + size_t(index)] =
            use_rigorous ? rigorous[3 * index + component] : paraxial[3 * index + component];
}

} // namespace

//-----------------------------------------------------------
extern "C" cudaError_t gaussianfft_build_coordinates_wrapper(CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
                                                             int nfft2d, int nx, int ny,
                                                             CU_REAL xmin, CU_REAL ymin, CU_REAL z,
                                                             CU_REAL aretecube) {
    if (!xs || !ys || !zs || nfft2d <= 0 || nfft2d > INT_MAX / nfft2d || nx <= 0 || ny <= 0 ||
        nx > nfft2d || ny > nfft2d || aretecube <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t count = size_t(nfft2d) * size_t(nfft2d);
    gaussianfft_coordinates_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        xs, ys, zs, nfft2d, nx, ny, xmin, ymin, z, aretecube);
    return finish_gpbicg_launch(134, "gaussianfft_coordinates_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gaussianfft_build_plane_map_wrapper(const CU_REAL *dipole_xs,
                                                           const CU_REAL *dipole_ys, CU_INT*plane_map,
                                                           int nfft2d, int dipole_count, int kkm,
                                                           int jjm, CU_REAL xmin, CU_REAL ymin,
                                                           CU_REAL aretecube) {
    if (!dipole_xs || !dipole_ys || !plane_map || nfft2d <= 0 || nfft2d > INT_MAX / nfft2d ||
        dipole_count <= 0 || kkm < 0 || jjm < 0 || kkm >= nfft2d || jjm >= nfft2d ||
        aretecube <= CU_REAL(0))
        return cudaErrorInvalidValue;
    gaussianfft_build_plane_map_kernel<<<1, 1>>>(dipole_xs, dipole_ys, plane_map, nfft2d,
                                                 dipole_count, kkm, jjm, xmin, ymin, aretecube);
    return finish_gpbicg_launch(138, "gaussianfft_build_plane_map_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gaussianfft_max_intensity_wrapper(const CUCOMPLEX *field, CU_REAL *partials,
                                                         CU_REAL *maximum, int count) {
    if (!field || !partials || !maximum || count <= 0)
        return cudaErrorInvalidValue;
    const int partial_count = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gaussianfft_max_intensity_partials_kernel<<<partial_count, THREADS_PER_BLOCK>>>(field, partials,
                                                                                    count);
    cudaError_t status = finish_gpbicg_launch(135, "gaussianfft_max_intensity_partials_kernel");
    if (status != cudaSuccess)
        return status;
    gaussianfft_reduce_max_partials_kernel<<<1, 1>>>(partials, maximum, partial_count);
    return finish_gpbicg_launch(136, "gaussianfft_reduce_max_partials_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gaussianfft_blend_first_plane_wrapper(const CUCOMPLEX *paraxial,
                                                             const CUCOMPLEX *rigorous,
                                                             const CU_REAL *max_intensity,
                                                             CU_REAL tolerance, int circular,
                                                             CUCOMPLEX *image_planes, int count) {
    if (!paraxial || !rigorous || !max_intensity || !image_planes || tolerance < CU_REAL(0) ||
        (circular != 0 && circular != 1) || count <= 0)
        return cudaErrorInvalidValue;
    gaussianfft_blend_first_plane_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        paraxial, rigorous, max_intensity, tolerance, circular, image_planes, count);
    return finish_gpbicg_launch(137, "gaussianfft_blend_first_plane_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t beampropagation_blend_gaussian_background_wrapper(
    const CU_INT*first_plane_map, const CUCOMPLEX *paraxial, const CUCOMPLEX *rigorous,
    const CU_REAL *max_intensity, CU_REAL threshold, CUCOMPLEX *image_planes, int count) {
    if (!first_plane_map || !paraxial || !rigorous || !max_intensity || !image_planes ||
        threshold < CU_REAL(0) || count <= 0)
        return cudaErrorInvalidValue;
    beampropagation_blend_gaussian_background_kernel<<<block_count(size_t(count)),
                                                       THREADS_PER_BLOCK>>>(
        first_plane_map, paraxial, rigorous, max_intensity, threshold, image_planes, count);
    return finish_gpbicg_launch(139, "beampropagation_blend_gaussian_background_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gaussianfft_filter_wrapper(CUCOMPLEX *image_planes, int nfft2d,
                                                  CU_REAL aretecube, CU_REAL k0, int first_plane) {
    if (!image_planes || nfft2d <= 0 || nfft2d > INT_MAX / nfft2d || aretecube <= CU_REAL(0) ||
        k0 <= CU_REAL(0) || (first_plane != 0 && first_plane != 1))
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft2d) * size_t(nfft2d);
    gaussianfft_filter_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        image_planes, nfft2d, aretecube, k0, first_plane);
    return finish_gpbicg_launch(118, "gaussianfft_filter_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t gaussianfft_scatter_wrapper(const CUCOMPLEX *image_planes,
                                                   const CU_INT*plane_map, CUCOMPLEX *dipole_field,
                                                   int nfft2d, int plane_count, int layer,
                                                   size_t mesh_count) {
    if (!image_planes || !plane_map || !dipole_field || nfft2d <= 0 || nfft2d > INT_MAX / nfft2d ||
        plane_count <= 0 || layer < 0 || mesh_count == 0)
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft2d) * size_t(nfft2d);
    gaussianfft_scatter_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        image_planes, plane_map, dipole_field, nfft2d, plane_count, layer, mesh_count);
    return finish_gpbicg_launch(119, "gaussianfft_scatter_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t beampropagation_spectral_wrapper(
    CUCOMPLEX *image_planes, const CU_INT*first_plane_map, const CUCOMPLEX *epsilon,
    CUCOMPLEX *local_field, int nx, int ny, int nz, int nfft2d, int nmax, int niso, int ndipole,
    cufftHandle plan, CU_REAL aretecube, CU_REAL k0, int convert_local_to_macro) {
    if (!image_planes || !first_plane_map || !epsilon || !local_field || nx <= 0 || ny <= 0 ||
        nz <= 1 || nfft2d <= 0 || nx > nfft2d || ny > nfft2d || nmax <= 0 || niso <= 0 ||
        nfft2d > INT_MAX / nfft2d || ndipole <= 0)
        return cudaErrorInvalidValue;
    const int plane = nfft2d * nfft2d;
    const CU_REAL delta_k =
        CU_REAL(2.0) * CU_REAL(3.14159265358979323846) / (CU_REAL(nfft2d) * aretecube);
    if (delta_k >= k0 / CU_REAL(2.0))
        return cudaErrorInvalidValue;

    if (!plan)
        return cudaErrorInvalidValue;
    cufftResult fft_status = CUFFT_SUCCESS;
    for (int z_layer = 1; z_layer < nz; ++z_layer) {
        for (int component = 0; component < 3; ++component) {
            CUCOMPLEX *image = image_planes + size_t(component) * plane;
            fft_status = CUFFT_EXECUTE(plan, image, image, CUFFT_INVERSE);
            if (fft_status != CUFFT_SUCCESS) {
                return cudaErrorUnknown;
            }
        }
        trace_beampropagation_layer(image_planes, size_t(plane), z_layer + 1, "inverse");
        beampropagation_spectral_kernel<<<block_count(size_t(plane)), THREADS_PER_BLOCK>>>(
            image_planes, nfft2d, aretecube, k0);
        cudaError_t status = finish_gpbicg_launch(115, "beampropagation_spectral_kernel");
        if (status != cudaSuccess) {
            return status;
        }
        trace_beampropagation_layer(image_planes, size_t(plane), z_layer + 1, "spectral");
        for (int component = 0; component < 3; ++component) {
            CUCOMPLEX *image = image_planes + size_t(component) * plane;
            fft_status = CUFFT_EXECUTE(plan, image, image, CUFFT_FORWARD);
            if (fft_status != CUFFT_SUCCESS) {
                return cudaErrorUnknown;
            }
        }
        trace_beampropagation_layer(image_planes, size_t(plane), z_layer + 1, "forward");
        beampropagation_update_plane_kernel<<<block_count(size_t(plane)), THREADS_PER_BLOCK>>>(
            image_planes, first_plane_map, epsilon, local_field, nx, ny, nfft2d, nmax, niso,
            z_layer, aretecube, k0);
        status = finish_gpbicg_launch(116, "beampropagation_update_plane_kernel");
        if (status != cudaSuccess) {
            return status;
        }
        trace_beampropagation_layer(image_planes, size_t(plane), z_layer + 1, "update");
    }
    if (convert_local_to_macro) {
        beampropagation_local_to_macro_kernel<<<block_count(size_t(ndipole)), THREADS_PER_BLOCK>>>(
            epsilon, local_field, nmax, niso, ndipole, aretecube, k0);
        const cudaError_t status =
            finish_gpbicg_launch(117, "beampropagation_local_to_macro_kernel");
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

//-----------------------------------------------------------
extern "C" cudaError_t beampropagation_initialize_plane_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CUCOMPLEX *epsilon,
    const CUCOMPLEX *incident, CUCOMPLEX *image_planes, CU_INT*first_plane_map,
    CUCOMPLEX *local_field, int nx, int ny, int nfft2d, int nmax, int niso, int ndipole, int kkm,
    int jjm, CU_REAL xmin, CU_REAL ymin, CU_REAL first_z, CU_REAL aretecube, CU_REAL k0,
    CUCOMPLEX E0, CU_REAL ss, CU_REAL pp, CU_REAL theta_degrees, CU_REAL phi_degrees, CU_REAL xgaus,
    CU_REAL ygaus, CU_REAL zgaus, CU_REAL w0, int beam_type) {
    if (!xs || !ys || !zs || !epsilon || !incident || !image_planes || !first_plane_map ||
        !local_field || nx <= 0 || ny <= 0 || nfft2d <= 0 || nx > nfft2d || ny > nfft2d ||
        nmax <= 0 || niso <= 0 || ndipole <= 0 || kkm < 0 || jjm < 0 || aretecube <= CU_REAL(0) ||
        k0 <= CU_REAL(0) || beam_type < 0 || beam_type > 3 || nfft2d > INT_MAX / nfft2d ||
        ((beam_type == 2 || beam_type == 3) && w0 <= CU_REAL(0)))
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft2d) * size_t(nfft2d);
    cudaError_t status = cudaMemset(first_plane_map, 0, plane * sizeof(int));
    if (status != cudaSuccess)
        return status;
    beampropagation_map_first_plane_kernel<<<block_count(size_t(ndipole)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, first_plane_map, nfft2d, ndipole, kkm, jjm, xmin, ymin, first_z, aretecube);
    status = finish_gpbicg_launch(118, "beampropagation_map_first_plane_kernel");
    if (status != cudaSuccess)
        return status;
    beampropagation_initialize_first_plane_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        first_plane_map, xs, ys, zs, epsilon, incident, image_planes, local_field, nx, ny, nfft2d,
        nmax, niso, kkm, jjm, xmin, ymin, first_z, aretecube, k0, E0, ss, pp, theta_degrees,
        phi_degrees, xgaus, ygaus, zgaus, w0, beam_type);
    return finish_gpbicg_launch(119, "beampropagation_initialize_first_plane_kernel");
}

namespace {
//-----------------------------------------------------------
__device__ CU_REAL pml_axis_mask(CU_REAL distance, CU_REAL lneps, CU_REAL lpml, CU_REAL dpml) {
    if (distance <= lneps)
        return CU_REAL(0);
    if (distance <= lpml)
        return CU_REAL(0.5) * (CU_REAL(1) - cos(CU_REAL(3.1415926535897932384626433832795) *
                                                (distance - lneps) / dpml));
    return CU_REAL(1);
}

//-----------------------------------------------------------
__global__ void pml_build_mask_candidates_kernel(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, CU_REAL *mask, CU_REAL *z_candidates,
    int dipole_count, CU_REAL xmin, CU_REAL xmax, CU_REAL ymin, CU_REAL ymax, CU_REAL zmin,
    CU_REAL zmax, CU_REAL aretecube, CU_REAL lneps, CU_REAL lpml, CU_REAL dpml) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= dipole_count)
        return;
    CU_REAL mask_x = CU_REAL(1);
    CU_REAL distance = abs(xs[i] - xmin) + aretecube / CU_REAL(2);
    if (distance <= lpml)
        mask_x = pml_axis_mask(distance, lneps, lpml, dpml);
    distance = abs(xmax - xs[i]) + aretecube / CU_REAL(2);
    if (distance <= lpml)
        mask_x = pml_axis_mask(distance, lneps, lpml, dpml);

    CU_REAL mask_y = CU_REAL(1);
    distance = abs(ys[i] - ymin) + aretecube / CU_REAL(2);
    if (distance <= lpml)
        mask_y = pml_axis_mask(distance, lneps, lpml, dpml);
    distance = abs(ymax - ys[i]) + aretecube / CU_REAL(2);
    if (distance <= lpml)
        mask_y = pml_axis_mask(distance, lneps, lpml, dpml);

    CU_REAL mask_z = CU_REAL(1);
    distance = abs(zs[i] - zmin) + aretecube / CU_REAL(2);
    if (distance <= lpml)
        mask_z = CU_REAL(0);
    distance = abs(zmax - zs[i]) + aretecube / CU_REAL(2);
    if (distance <= lneps)
        mask_z = CU_REAL(0);
    else if (distance <= lpml)
        mask_z = CU_REAL(0.5) * (CU_REAL(1) - cos(CU_REAL(3.1415926535897932384626433832795) *
                                                  (distance - lneps) / dpml));

    mask[i] = mask_x * mask_y * mask_z;
    z_candidates[i] = mask_z >= CU_REAL(0.999999) ? zs[i] : zmax;
}

//-----------------------------------------------------------
__global__ void pml_reduce_zinf_kernel(const CU_REAL *candidates, CU_REAL *zinf, int count,
                                       CU_REAL fallback) {
    __shared__ CU_REAL partial[THREADS_PER_BLOCK];
    CU_REAL local_min = fallback;
    for (int i = int(threadIdx.x); i < count; i += int(blockDim.x))
        local_min = fmin(local_min, candidates[i]);
    partial[threadIdx.x] = local_min;
    __syncthreads();
    for (int stride = int(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (int(threadIdx.x) < stride)
            partial[threadIdx.x] = fmin(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        zinf[0] = partial[0];
}

//-----------------------------------------------------------
__global__ void pml_apply_field_kernel(const CU_REAL *mask, CUCOMPLEX *field, int dipole_count,
                                       CU_REAL field_scale) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= dipole_count)
        return;
    const CU_REAL scale = mask[i] * field_scale;
    for (int component = 0; component < 3; ++component) {
        const int index = 3 * i + component;
        field[index].x *= scale;
        field[index].y *= scale;
    }
}

//-----------------------------------------------------------
__global__ void optical_force_gather_fft_kernel(const CUCOMPLEX *polarization,
                                                const CU_INT *dipole_map, CUCOMPLEX *fft_vectors,
                                                int nx, int ny, int nz, int nx2, int ny2, int nz2,
                                                int padded_count, int use_map) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= padded_count)
        return;
    const int plane = nx2 * ny2;
    const int z = index / plane;
    const int rem = index - z * plane;
    const int y = rem / nx2;
    const int x = rem - y * nx2;
    CUCOMPLEX vx = CUFFT_ZERO, vy = CUFFT_ZERO, vz = CUFFT_ZERO;
    if (x < nx && y < ny && z < nz) {
        const int physical = x + nx * (y + ny * z);
        const int dipole = use_map ? dipole_map[physical].dat - 1 : physical;
        if (dipole >= 0) {
            vx = polarization[3 * dipole];
            vy = polarization[3 * dipole + 1];
            vz = polarization[3 * dipole + 2];
        }
    }
    fft_vectors[index] = vx;
    fft_vectors[padded_count + index] = vy;
    fft_vectors[2 * padded_count + index] = vz;
}

//-----------------------------------------------------------
__global__ void optical_force_density_kernel(const CUCOMPLEX *polarization,
                                             const CUCOMPLEX *derivatives, CU_REAL *force_x,
                                             CU_REAL *force_y, CU_REAL *force_z, int dipole_count,
                                             int mesh_size, CU_REAL scale) {
    const int dipole = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (dipole >= dipole_count)
        return;
    const int vector = 3 * dipole;
    const int axis_stride = mesh_size;
    const int component_stride = 3 * mesh_size;
    const CUCOMPLEX px = CUFFT_CONJ(polarization[vector]);
    const CUCOMPLEX py = CUFFT_CONJ(polarization[vector + 1]);
    const CUCOMPLEX pz = CUFFT_CONJ(polarization[vector + 2]);
    for (int axis = 0; axis < 3; ++axis) {
        const size_t offset = size_t(axis) * size_t(axis_stride) + size_t(dipole);
        const CUCOMPLEX dx = derivatives[offset];
        const CUCOMPLEX dy = derivatives[size_t(component_stride) + offset];
        const CUCOMPLEX dz = derivatives[size_t(2 * component_stride) + offset];
        const CUCOMPLEX contraction =
            CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(px, dx), CUFFT_MUL(py, dy)), CUFFT_MUL(pz, dz));
        const CU_REAL value = CU_REAL(0.5) * contraction.x * scale;
        if (axis == 0)
            force_x[dipole] = value;
        else if (axis == 1)
            force_y[dipole] = value;
        else
            force_z[dipole] = value;
    }
}

//-----------------------------------------------------------
__global__ void optical_force_fill_reduction_ones_kernel(CU_REAL *values, int count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
        values[index] = CU_REAL(1);
}
} // namespace

namespace {
//-----------------------------------------------------------
__device__ CUCOMPLEX optical_derivative_weighted_sum(const CUCOMPLEX *field, int base, int stride,
                                                     int position, int axis_length,
                                                     CU_REAL inverse_scale) {
    int start;
    int point;
    if (position == 0) {
        start = 0;
        point = 0;
    } else if (position == 1) {
        start = 0;
        point = 1;
    } else if (position < axis_length - 2) {
        start = position - 2;
        point = 2;
    } else if (position == axis_length - 2) {
        start = axis_length - 5;
        point = 3;
    } else {
        start = axis_length - 5;
        point = 4;
    }

    const CUCOMPLEX f0 = field[base + (start + 0) * stride];
    const CUCOMPLEX f1 = field[base + (start + 1) * stride];
    const CUCOMPLEX f2 = field[base + (start + 2) * stride];
    const CUCOMPLEX f3 = field[base + (start + 3) * stride];
    const CUCOMPLEX f4 = field[base + (start + 4) * stride];
    CUCOMPLEX numerator;
    if (point == 0)
        numerator = CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(-50)), f0),
                                        CUFFT_MUL(CUFFT_REAL(CU_REAL(96)), f1)),
                              CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(-72)), f2),
                                        CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(32)), f3),
                                                  CUFFT_MUL(CUFFT_REAL(CU_REAL(-6)), f4))));
    else if (point == 1)
        numerator = CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(-6)), f0),
                                        CUFFT_MUL(CUFFT_REAL(CU_REAL(-20)), f1)),
                              CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(36)), f2),
                                        CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(-12)), f3),
                                                  CUFFT_MUL(CUFFT_REAL(CU_REAL(2)), f4))));
    else if (point == 2)
        numerator = CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(2)), f0),
                                        CUFFT_MUL(CUFFT_REAL(CU_REAL(-16)), f1)),
                              CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(16)), f3),
                                        CUFFT_MUL(CUFFT_REAL(CU_REAL(-2)), f4)));
    else if (point == 3)
        numerator = CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(-2)), f0),
                                        CUFFT_MUL(CUFFT_REAL(CU_REAL(12)), f1)),
                              CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(-36)), f2),
                                        CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(20)), f3),
                                                  CUFFT_MUL(CUFFT_REAL(CU_REAL(6)), f4))));
    else
        numerator = CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(6)), f0),
                                        CUFFT_MUL(CUFFT_REAL(CU_REAL(-32)), f1)),
                              CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(72)), f2),
                                        CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(-96)), f3),
                                                  CUFFT_MUL(CUFFT_REAL(CU_REAL(50)), f4))));
    return CUFFT_MUL(numerator, CUFFT_REAL(inverse_scale));
}

//-----------------------------------------------------------
__global__ void optical_field_derivative_kernel(const CUCOMPLEX *field, CUCOMPLEX *derivative,
                                                int nx, int ny, int nz, int axis,
                                                CU_REAL inverse_scale) {
    const int value = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nx * ny * nz * 3;
    if (value >= count)
        return;
    const int site = value / 3;
    const int component = value - site * 3;
    const int plane = nx * ny;
    const int z = site / plane;
    const int rem = site - z * plane;
    const int y = rem / nx;
    const int x = rem - y * nx;
    int position, length, stride;
    if (axis == 0) {
        position = x;
        length = nx;
        stride = 3;
    } else if (axis == 1) {
        position = y;
        length = ny;
        stride = 3 * nx;
    } else {
        position = z;
        length = nz;
        stride = 3 * plane;
    }
    derivative[value] = optical_derivative_weighted_sum(
        field, site * 3 + component - position * stride, stride, position, length, inverse_scale);
}
} // namespace

//-----------------------------------------------------------
extern "C" cudaError_t pml_build_mask_candidates_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, CU_REAL *mask, CU_REAL *z_candidates,
    int dipole_count, CU_REAL xmin, CU_REAL xmax, CU_REAL ymin, CU_REAL ymax, CU_REAL zmin,
    CU_REAL zmax, CU_REAL aretecube, CU_REAL lneps, CU_REAL lpml, CU_REAL dpml) {
    if (!xs || !ys || !zs || !mask || !z_candidates || dipole_count <= 0 ||
        aretecube <= CU_REAL(0) || dpml < CU_REAL(0))
        return cudaErrorInvalidValue;
    pml_build_mask_candidates_kernel<<<block_count(size_t(dipole_count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, mask, z_candidates, dipole_count, xmin, xmax, ymin, ymax, zmin, zmax, aretecube,
        lneps, lpml, dpml);
    return finish_gpbicg_launch(131, "pml_build_mask_candidates_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pml_reduce_zinf_wrapper(const CU_REAL *z_candidates, CU_REAL *zinf,
                                               int dipole_count, CU_REAL fallback_zinf) {
    if (!z_candidates || !zinf || dipole_count <= 0)
        return cudaErrorInvalidValue;
    pml_reduce_zinf_kernel<<<1, THREADS_PER_BLOCK>>>(z_candidates, zinf, dipole_count,
                                                     fallback_zinf);
    return finish_gpbicg_launch(132, "pml_reduce_zinf_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t pml_apply_field_wrapper(const CU_REAL *mask, CUCOMPLEX *field,
                                               int dipole_count, CU_REAL field_scale) {
    if (!mask || !field || dipole_count <= 0)
        return cudaErrorInvalidValue;
    pml_apply_field_kernel<<<block_count(size_t(dipole_count)), THREADS_PER_BLOCK>>>(
        mask, field, dipole_count, field_scale);
    return finish_gpbicg_launch(133, "pml_apply_field_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t optical_field_derivative_wrapper(const CUCOMPLEX *field,
                                                        CUCOMPLEX *derivative, int nx, int ny,
                                                        int nz, int axis, CU_REAL spacing) {
    if (!field || !derivative || nx < 5 || ny < 5 || nz < 5 || axis < 0 || axis > 2 ||
        spacing <= CU_REAL(0) || size_t(nx) * size_t(ny) * size_t(nz) > size_t(INT_MAX / 3))
        return cudaErrorInvalidValue;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz) * 3;
    optical_field_derivative_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        field, derivative, nx, ny, nz, axis, CU_REAL(1) / (CU_REAL(24) * spacing));
    return finish_gpbicg_launch(121, "optical_field_derivative_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t optical_force_gather_fft_wrapper(const CUCOMPLEX *polarization,
                                                        const CU_INT *dipole_map,
                                                        CUCOMPLEX *fft_vectors, int nx, int ny,
                                                        int nz, int nx2, int ny2, int nz2,
                                                        int padded_count, int use_map) {
    if (!polarization || !fft_vectors || (use_map && !dipole_map) || nx <= 0 || ny <= 0 ||
        nz <= 0 || nx2 < nx || ny2 < ny || nz2 < nz ||
        size_t(nx2) * size_t(ny2) * size_t(nz2) != size_t(padded_count) || padded_count <= 0)
        return cudaErrorInvalidValue;
    optical_force_gather_fft_kernel<<<block_count(size_t(padded_count)), THREADS_PER_BLOCK>>>(
        polarization, dipole_map, fft_vectors, nx, ny, nz, nx2, ny2, nz2, padded_count, use_map);
    return finish_gpbicg_launch(122, "optical_force_gather_fft_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t optical_force_build_green_derivative_wrapper(CUCOMPLEX *green_derivative,
                                                                    int nx, int ny, int nz, int nx2,
                                                                    int ny2, int nz2, int nxy2,
                                                                    int padded_count, int axis,
                                                                    CU_REAL spacing, CU_REAL k0) {
    if (!green_derivative || nx <= 0 || ny <= 0 || nz <= 0 || nx2 < nx || ny2 < ny || nz2 < nz ||
        nxy2 != nx2 * ny2 || size_t(nxy2) * size_t(nz2) != size_t(padded_count) || axis < 1 ||
        axis > 3 || spacing <= CU_REAL(0) || padded_count <= 0)
        return cudaErrorInvalidValue;
    build_green_derivative_tensor_kernel<<<block_count(size_t(padded_count)), THREADS_PER_BLOCK>>>(
        green_derivative, green_derivative + padded_count, green_derivative + 2 * padded_count,
        green_derivative + 3 * padded_count, green_derivative + 4 * padded_count,
        green_derivative + 5 * padded_count, nx, ny, nz, nx2, ny2, nz2, nxy2, padded_count, axis,
        spacing, k0);
    return finish_gpbicg_launch(123, "build_green_derivative_tensor_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t optical_force_derivative_spectral_product_wrapper(
    const CUCOMPLEX *fft_vectors, CUCOMPLEX *green_derivative, int padded_count) {
    if (!fft_vectors || !green_derivative || padded_count <= 0)
        return cudaErrorInvalidValue;
    derivative_spectral_product_kernel<<<block_count(size_t(padded_count)), THREADS_PER_BLOCK>>>(
        fft_vectors, fft_vectors + padded_count, fft_vectors + 2 * padded_count, green_derivative,
        green_derivative + padded_count, green_derivative + 2 * padded_count,
        green_derivative + 3 * padded_count, green_derivative + 4 * padded_count,
        green_derivative + 5 * padded_count, padded_count);
    return finish_gpbicg_launch(124, "derivative_spectral_product_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t optical_force_extract_green_derivative_wrapper(
    const CUCOMPLEX *green_derivative, CUCOMPLEX *output, int nx, int ny, int nz, int nx2, int nxy2,
    int padded_count, int physical_count) {
    if (!green_derivative || !output || nx <= 0 || ny <= 0 || nz <= 0 || nx2 < nx || nxy2 <= 0 ||
        padded_count <= 0 || physical_count <= 0 ||
        size_t(nx) * size_t(ny) * size_t(nz) != size_t(physical_count))
        return cudaErrorInvalidValue;
    extract_derivative_kernel<<<block_count(size_t(physical_count)), THREADS_PER_BLOCK>>>(
        green_derivative, green_derivative + 3 * padded_count, green_derivative + 5 * padded_count,
        output, nx, ny, nz, nx2, nxy2, physical_count, CU_REAL(padded_count));
    return finish_gpbicg_launch(125, "extract_derivative_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t optical_force_density_wrapper(const CUCOMPLEX *polarization,
                                                     const CUCOMPLEX *field_derivatives,
                                                     CU_REAL *force_x, CU_REAL *force_y,
                                                     CU_REAL *force_z, int dipole_count,
                                                     int mesh_size, CU_REAL scale) {
    if (!polarization || !field_derivatives || !force_x || !force_y || !force_z ||
        dipole_count < 0 || mesh_size < dipole_count)
        return cudaErrorInvalidValue;
    optical_force_density_kernel<<<block_count(size_t(dipole_count)), THREADS_PER_BLOCK>>>(
        polarization, field_derivatives, force_x, force_y, force_z, dipole_count, mesh_size, scale);
    return finish_gpbicg_launch(120, "optical_force_density_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t optical_force_fill_reduction_ones_wrapper(CU_REAL *ones, int count) {
    if (!ones || count <= 0)
        return cudaErrorInvalidValue;
    optical_force_fill_reduction_ones_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        ones, count);
    return finish_gpbicg_launch(126, "optical_force_fill_reduction_ones_kernel");
}

namespace {
//-----------------------------------------------------------
__global__ void optical_density_group_sum_kernel(const CU_INT*object_ids, const CU_REAL *x,
                                                 const CU_REAL *y, const CU_REAL *z,
                                                 CU_REAL *object_sums, int dipole_count,
                                                 int object_count) {
    const int object = int(blockIdx.x);
    const int thread = int(threadIdx.x);
    if (object >= object_count)
        return;
    extern __shared__ CU_REAL partial_sums[];
    CU_REAL *partial_x = partial_sums;
    CU_REAL *partial_y = partial_sums + blockDim.x;
    CU_REAL *partial_z = partial_sums + 2 * blockDim.x;
    CU_REAL sum_x = CU_REAL(0), sum_y = CU_REAL(0), sum_z = CU_REAL(0);
    const int object_id = object + 1;
    for (int dipole = thread; dipole < dipole_count; dipole += int(blockDim.x)) {
        if (object_ids[dipole].dat != object_id)
            continue;
        sum_x += x[dipole];
        sum_y += y[dipole];
        sum_z += z[dipole];
    }
    partial_x[thread] = sum_x;
    partial_y[thread] = sum_y;
    partial_z[thread] = sum_z;
    __syncthreads();
    for (int stride = int(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            partial_x[thread] += partial_x[thread + stride];
            partial_y[thread] += partial_y[thread + stride];
            partial_z[thread] += partial_z[thread + stride];
        }
        __syncthreads();
    }
    if (thread == 0) {
        object_sums[3 * object] = partial_x[0];
        object_sums[3 * object + 1] = partial_y[0];
        object_sums[3 * object + 2] = partial_z[0];
    }
}

//-----------------------------------------------------------
cudaError_t launch_optical_density_group_sum(const CU_INT*object_ids, const CU_REAL *x,
                                             const CU_REAL *y, const CU_REAL *z,
                                             CU_REAL *object_sums, int dipole_count,
                                             int object_count, int diagnostic_id,
                                             const char *diagnostic_name) {
    if (!object_ids || !x || !y || !z || !object_sums || dipole_count <= 0 || object_count <= 1)
        return cudaErrorInvalidValue;
    const size_t shared_bytes = size_t(3) * THREADS_PER_BLOCK * sizeof(CU_REAL);
    optical_density_group_sum_kernel<<<object_count, THREADS_PER_BLOCK, shared_bytes>>>(
        object_ids, x, y, z, object_sums, dipole_count, object_count);
    return finish_gpbicg_launch(diagnostic_id, diagnostic_name);
}
} // namespace

//-----------------------------------------------------------
extern "C" cudaError_t optical_force_group_sum_wrapper(const CU_INT*object_ids,
                                                       const CU_REAL *force_x,
                                                       const CU_REAL *force_y,
                                                       const CU_REAL *force_z, CU_REAL *object_sums,
                                                       int dipole_count, int object_count) {
    return launch_optical_density_group_sum(object_ids, force_x, force_y, force_z, object_sums,
                                            dipole_count, object_count, 129,
                                            "optical_force_group_sum_kernel");
}

//-----------------------------------------------------------
extern "C" cudaError_t optical_torque_group_sum_wrapper(
    const CU_INT*object_ids, const CU_REAL *torque_x, const CU_REAL *torque_y,
    const CU_REAL *torque_z, CU_REAL *object_sums, int dipole_count, int object_count) {
    return launch_optical_density_group_sum(object_ids, torque_x, torque_y, torque_z, object_sums,
                                            dipole_count, object_count, 130,
                                            "optical_torque_group_sum_kernel");
}

namespace {
//-----------------------------------------------------------
__global__ void optical_torque_density_kernel(const CUCOMPLEX *polarization,
                                              const CUCOMPLEX *local_field, const CU_REAL *xs,
                                              const CU_REAL *ys, const CU_REAL *zs,
                                              const CU_REAL *force_x, const CU_REAL *force_y,
                                              const CU_REAL *force_z, CU_REAL *torque_x,
                                              CU_REAL *torque_y, CU_REAL *torque_z,
                                              int dipole_count, CU_REAL k03, CU_REAL center_x,
                                              CU_REAL center_y, CU_REAL center_z, CU_REAL scale) {
    const int dipole = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (dipole >= dipole_count)
        return;
    const int base = 3 * dipole;
    const CUCOMPLEX px = CUFFT_CONJ(polarization[base]);
    const CUCOMPLEX py = CUFFT_CONJ(polarization[base + 1]);
    const CUCOMPLEX pz = CUFFT_CONJ(polarization[base + 2]);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX ik03 = make_cuFloatComplex(0.0f, CU_REAL(2.0 / 3.0) * k03);
#else
    const CUCOMPLEX ik03 = make_cuDoubleComplex(0.0, CU_REAL(2.0 / 3.0) * k03);
#endif
    const CUCOMPLEX ex = CUFFT_ADD(CUFFT_MUL(ik03, polarization[base]), local_field[base]);
    const CUCOMPLEX ey = CUFFT_ADD(CUFFT_MUL(ik03, polarization[base + 1]), local_field[base + 1]);
    const CUCOMPLEX ez = CUFFT_ADD(CUFFT_MUL(ik03, polarization[base + 2]), local_field[base + 2]);
    const CUCOMPLEX tx = CUFFT_SUB(CUFFT_MUL(py, ez), CUFFT_MUL(pz, ey));
    const CUCOMPLEX ty =
        CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(CU_REAL(-1)), CUFFT_MUL(px, ez)), CUFFT_MUL(pz, ex));
    const CUCOMPLEX tz = CUFFT_SUB(CUFFT_MUL(px, ey), CUFFT_MUL(py, ex));
    torque_x[dipole] = CU_REAL(0.5) * tx.x * scale + (ys[dipole] - center_y) * force_z[dipole] -
                       (zs[dipole] - center_z) * force_y[dipole];
    torque_y[dipole] = CU_REAL(0.5) * ty.x * scale + (zs[dipole] - center_z) * force_x[dipole] -
                       (xs[dipole] - center_x) * force_z[dipole];
    torque_z[dipole] = CU_REAL(0.5) * tz.x * scale + (xs[dipole] - center_x) * force_y[dipole] -
                       (ys[dipole] - center_y) * force_x[dipole];
}
} // namespace

//-----------------------------------------------------------
extern "C" cudaError_t optical_torque_density_wrapper(
    const CUCOMPLEX *polarization, const CUCOMPLEX *local_field, const CU_REAL *xs,
    const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *force_x, const CU_REAL *force_y,
    const CU_REAL *force_z, CU_REAL *torque_x, CU_REAL *torque_y, CU_REAL *torque_z,
    int dipole_count, CU_REAL k03, CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
    CU_REAL scale) {
    if (!polarization || !local_field || !xs || !ys || !zs || !force_x || !force_y || !force_z ||
        !torque_x || !torque_y || !torque_z || dipole_count < 0)
        return cudaErrorInvalidValue;
    optical_torque_density_kernel<<<block_count(size_t(dipole_count)), THREADS_PER_BLOCK>>>(
        polarization, local_field, xs, ys, zs, force_x, force_y, force_z, torque_x, torque_y,
        torque_z, dipole_count, k03, center_x, center_y, center_z, scale);
    return finish_gpbicg_launch(121, "optical_torque_density_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t rigorous_gaussian_incident_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, int count, double x0, double y0,
    double z0, double theta_degrees, double phi_degrees, double waist, double k0, double ss,
    double pp, COMPLEX e0, double tolerance, int circular, int nloin, const double *nodes,
    const double *weights, CUCOMPLEX *field, CU_INT*stop) {
    if (!xs || !ys || !zs || !nodes || !weights || !field || !stop || count <= 0 || waist <= 0.0 ||
        k0 <= 0.0 || tolerance <= 0.0 || (circular != 0 && circular != 1) || nloin < 0 ||
        nloin > 2 || (!circular && (pp < 0.0 || pp > 1.0)))
        return cudaErrorInvalidValue;
    rigorous_gaussian_incident_kernel<CU_REAL><<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        xs, ys, zs, count, x0, y0, z0, theta_degrees, phi_degrees, waist, k0, ss, pp,
        plane_wave_device_scalar(e0), tolerance,
        circular, nloin, nodes, weights, field, stop, {});
    return finish_kernel_launch(212, "rigorous_gaussian_incident_kernel");
}

extern "C" cudaError_t rigorous_gaussian_incident_grid_wrapper(
    IfddaIncidentGrid grid, double x0, double y0, double z0,
    double theta_degrees, double phi_degrees, double waist, double k0,
    double ss, double pp, COMPLEX e0, double tolerance, int circular, int nloin,
    const double* nodes, const double* weights, CUCOMPLEX* field, CU_INT* stop) {
    if (!grid.tabdip || !nodes || !weights || !field || !stop || grid.grid_count <= 0 ||
        grid.dipole_count <= 0 || waist <= 0 || k0 <= 0 || tolerance <= 0 ||
        (circular != 0 && circular != 1) || nloin < 0 || nloin > 2 ||
        (!circular && (pp < 0 || pp > 1))) return cudaErrorInvalidValue;
    rigorous_gaussian_incident_kernel<CU_REAL><<<block_count(size_t(grid.grid_count)),
                                                THREADS_PER_BLOCK>>>(
        nullptr, nullptr, nullptr, grid.dipole_count, x0, y0, z0,
        theta_degrees, phi_degrees, waist, k0, ss, pp,
        plane_wave_device_scalar(e0), tolerance,
        circular, nloin, nodes, weights, field, stop, grid);
    return finish_kernel_launch(212, "rigorous_gaussian_incident_grid_kernel");
}

// The original gaussianchamp/gaussianchampcirc routines suppress the rigorous
// field where the corresponding paraxial field satisfies
// sqrt(abs(Ex)+abs(Ey)+abs(Ez)) <= abs(E0)*1e-7. This CUDA post-pass is
// applied to gwave only, not to the separate gfftwave propagation pipeline.
//-------------------------------------------
__global__ void rigorous_gaussian_fortran_cutoff_kernel(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, int count,
    double x0, double y0, double z0, double theta_deg, double phi_deg,
    double waist, double k0, double ss, double pp, CUCOMPLEX e0,
    int circular, CUCOMPLEX* field, IfddaIncidentGrid grid)
{
    const int flat=int(blockIdx.x*blockDim.x+threadIdx.x);
    int id=flat;
    REAL pos_x=0,pos_y=0,pos_z=0;
    if(grid.tabdip) {
        if(!ifdda_incident_grid_fetch(grid,flat,&id,&pos_x,&pos_y,&pos_z))return;
    } else if(id<count) {
        pos_x=xs[id];pos_y=ys[id];pos_z=zs[id];
    } else return;
    using R=CU_REAL;
    const R theta=R(theta_deg*3.14159265358979323846/180.0);
    const R phi=R(phi_deg*3.14159265358979323846/180.0);
    const R st=sin(theta), ct=cos(theta), sp=sin(phi), cp=cos(phi);
    const R ux=st*cp,uy=st*sp,uz=ct;
    const R dx=pos_x-R(x0),dy=pos_y-R(y0),dz=pos_z-R(z0);
    R rx=dx,ry=dy,rz=dz;
    if(ux*ux+uy*uy>(sizeof(R)==4?R(1e-6):R(1e-12))) {
        const R inv=R(1)/(R(1)+uz);
        rx=(R(1)-ux*ux*inv)*dx-ux*uy*inv*dy-ux*dz;
        ry=-ux*uy*inv*dx+(R(1)-uy*uy*inv)*dy-uy*dz;
        rz=ux*dx+uy*dy+uz*dz;
    }
    const R w=R(waist),rayleigh=R(k0)*w*w;
    const R width2=R(2)*w*w*(R(1)+rz*rz/(rayleigh*rayleigh));
    const R gain=sqrt(R(2))*w/sqrt(width2)*exp(-(rx*rx+ry*ry)/width2);
    const R s=circular?R(1)/sqrt(R(2)):sqrt(R(1)-R(pp));
    const R p=circular?copysign(R(1)/sqrt(R(2)),R(ss)):sqrt(R(pp));
    const R sv[3]={-s*sp,s*cp,R(0)};
    const R pv[3]={p*ct*cp,p*ct*sp,-p*st};
    R sum=R(0);
    for(int i=0;i<3;++i)
        sum+=circular?hypot(sv[i],pv[i]):fabs(sv[i]+pv[i]);
    const R amplitude=hypot(e0.x,e0.y);
    if(sqrt(amplitude*gain*sum)<=amplitude*R(1e-7))
        for(int i=0;i<3;++i){field[3*id+i].x=R(0);field[3*id+i].y=R(0);}
}

//-------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t rigorous_gaussian_fortran_cutoff_wrapper(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, int count,
    double x0, double y0, double z0, double theta_deg, double phi_deg,
    double waist, double k0, double ss, double pp, COMPLEX e0,
    int circular, CUCOMPLEX* field)
{
    if(!xs||!ys||!zs||!field||count<=0||waist<=0||k0<=0)
        return cudaErrorInvalidValue;
    rigorous_gaussian_fortran_cutoff_kernel<<<block_count(size_t(count)),THREADS_PER_BLOCK>>>(
        xs,ys,zs,count,x0,y0,z0,theta_deg,phi_deg,waist,k0,ss,pp,
        plane_wave_device_scalar(e0),circular,field,{});
    return finish_kernel_launch(213,"rigorous_gaussian_fortran_cutoff_kernel");
}

extern "C" cudaError_t rigorous_gaussian_fortran_cutoff_grid_wrapper(
    IfddaIncidentGrid grid, double x0, double y0, double z0,
    double theta_deg, double phi_deg, double waist, double k0,
    double ss, double pp, COMPLEX e0, int circular, CUCOMPLEX* field) {
    if (!grid.tabdip || !field || grid.grid_count <= 0 ||
        grid.dipole_count <= 0 || waist <= 0 || k0 <= 0) return cudaErrorInvalidValue;
    rigorous_gaussian_fortran_cutoff_kernel<<<block_count(size_t(grid.grid_count)),
                                               THREADS_PER_BLOCK>>>(
        nullptr,nullptr,nullptr,grid.dipole_count,x0,y0,z0,theta_deg,phi_deg,
        waist,k0,ss,pp,plane_wave_device_scalar(e0),circular,field,grid);
    return finish_kernel_launch(213,"rigorous_gaussian_fortran_cutoff_grid_kernel");
}

//-----------------------------------------------------------
__global__ void rigorous_gaussian_derivative_kernel(CUCOMPLEX* fields, CU_REAL step) {
    CUCOMPLEX result[9];
    const CU_REAL scale = CU_REAL(0.5) / step;
    for (int axis = 0; axis < 3; ++axis) {
        const int plus_offset = 6 * axis;
        const int minus_offset = plus_offset + 3;
        for (int component = 0; component < 3; ++component) {
            const CUCOMPLEX plus = fields[plus_offset + component];
            const CUCOMPLEX minus = fields[minus_offset + component];
            result[component + 3 * axis].x = (plus.x - minus.x) * scale;
            result[component + 3 * axis].y = (plus.y - minus.y) * scale;
        }
    }
    for (int i = 0; i < 9; ++i)
        fields[i] = result[i];
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t rigorous_gaussian_derivative_wrapper(
    CUCOMPLEX* six_point_fields, CU_REAL step) {
    if (!six_point_fields || step <= CU_REAL(0))
        return cudaErrorInvalidValue;
    rigorous_gaussian_derivative_kernel<<<1, 1>>>(six_point_fields, step);
    return finish_kernel_launch(213, "rigorous_gaussian_derivative_kernel");
}

//-----------------------------------------------------------
__global__ void rigorous_gaussian_derivative_batch_kernel(
    const CUCOMPLEX* six_point_fields, CUCOMPLEX* derivatives,
    int point_count, CU_REAL step) {
    const int point = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (point >= point_count)
        return;
    const CU_REAL scale = CU_REAL(0.5) / step;
    const size_t field_base = size_t(point) * 18;
    const size_t derivative_base = size_t(point) * 9;
    for (int axis = 0; axis < 3; ++axis) {
        const size_t plus_base = field_base + size_t(6 * axis);
        const size_t minus_base = plus_base + 3;
        for (int component = 0; component < 3; ++component) {
            const CUCOMPLEX plus = six_point_fields[plus_base + size_t(component)];
            const CUCOMPLEX minus = six_point_fields[minus_base + size_t(component)];
            CUCOMPLEX value;
            value.x = (plus.x - minus.x) * scale;
            value.y = (plus.y - minus.y) * scale;
            derivatives[derivative_base + size_t(component + 3 * axis)] = value;
        }
    }
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t rigorous_gaussian_derivative_batch_wrapper(
    const CUCOMPLEX* six_point_fields, CUCOMPLEX* derivatives,
    int point_count, CU_REAL step) {
    if (!six_point_fields || !derivatives || point_count < 0 || step <= CU_REAL(0))
        return cudaErrorInvalidValue;
    if (point_count == 0)
        return cudaSuccess;
    rigorous_gaussian_derivative_batch_kernel<<<block_count(size_t(point_count)), THREADS_PER_BLOCK>>>(
        six_point_fields, derivatives, point_count, step);
    return finish_kernel_launch(264, "rigorous_gaussian_derivative_batch_kernel");
}

namespace {
//-----------------------------------------------------------
__global__ void direct_scattering_angles_kernel(
    const CUCOMPLEX *field, const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
    int dipole_count, int ntheta, int nphi, CU_REAL k0, CU_REAL theta0, CU_REAL phi0,
    CU_REAL dtheta, CU_REAL dphi, CU_REAL const_poynting, CU_REAL *poynting, CU_REAL *theta_values,
    CU_REAL *phi_values, CU_REAL *weighted_scattering, CU_REAL *weighted_asymmetry,
    CU_REAL *reduction_ones) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int angle_count = (ntheta + 1) * nphi;
    if (index >= angle_count)
        return;
    const int itheta = index / nphi;
    const int iphi = index % nphi;
    const CU_REAL theta = dtheta * CU_REAL(itheta);
    const CU_REAL phi = dphi * CU_REAL(iphi);
    const CU_REAL nx = sin(theta) * cos(phi);
    const CU_REAL ny = sin(theta) * sin(phi);
    const CU_REAL nz = cos(theta);
    CUCOMPLEX ex = CUFFT_ZERO, ey = CUFFT_ZERO, ez = CUFFT_ZERO;
    for (int dipole = 0; dipole < dipole_count; ++dipole) {
        const int base = 3 * dipole;
        const CU_REAL angle = k0 * (nx * xs[dipole] + ny * ys[dipole] + nz * zs[dipole]);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CUCOMPLEX phase = make_cuFloatComplex(cosf(angle), -sinf(angle));
#else
        const CUCOMPLEX phase = make_cuDoubleComplex(cos(angle), -sin(angle));
#endif
        const CUCOMPLEX longitudinal =
            CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(field[base], CUFFT_REAL(nx)),
                                CUFFT_MUL(field[base + 1], CUFFT_REAL(ny))),
                      CUFFT_MUL(field[base + 2], CUFFT_REAL(nz)));
        ex = CUFFT_ADD(
            ex, CUFFT_MUL(phase, CUFFT_SUB(field[base], CUFFT_MUL(longitudinal, CUFFT_REAL(nx)))));
        ey = CUFFT_ADD(ey, CUFFT_MUL(phase, CUFFT_SUB(field[base + 1],
                                                      CUFFT_MUL(longitudinal, CUFFT_REAL(ny)))));
        ez = CUFFT_ADD(ez, CUFFT_MUL(phase, CUFFT_SUB(field[base + 2],
                                                      CUFFT_MUL(longitudinal, CUFFT_REAL(nz)))));
    }
    const CU_REAL emod =
        ex.x * ex.x + ex.y * ex.y + ey.x * ey.x + ey.y * ey.y + ez.x * ez.x + ez.y * ez.y;
    const CU_REAL weight = dtheta * dphi * sin(theta) * emod;
    const CU_REAL direction =
        nx * sin(theta0) * cos(phi0) + ny * sin(theta0) * sin(phi0) + nz * cos(theta0);
    poynting[index] = emod * const_poynting;
    theta_values[index] = theta * CU_REAL(180.0) / CU_REAL(3.14159265358979323846);
    phi_values[index] = phi * CU_REAL(180.0) / CU_REAL(3.14159265358979323846);
    weighted_scattering[index] = weight;
    weighted_asymmetry[index] = weight * direction;
    reduction_ones[index] = CU_REAL(1);
}

//-----------------------------------------------------------
__global__ void direct_scattering_spectrum_kernel(const CUCOMPLEX *field, const CU_REAL *xs,
                                                  const CU_REAL *ys, const CU_REAL *zs,
                                                  int dipole_count, int ll, int np, CU_REAL k0,
                                                  CU_REAL delta, CU_REAL const_poynting,
                                                  CU_REAL *positive, CU_REAL *negative,
                                                  CU_REAL *kx_values) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= np * np)
        return;
    const int i = index % np - ll;
    const int j = index / np - ll;
    const CU_REAL kx = CU_REAL(i) * delta;
    const CU_REAL ky = CU_REAL(j) * delta;
    const CU_REAL nx = kx / k0;
    const CU_REAL ny = ky / k0;
    const CU_REAL kz2 = CU_REAL(1) - nx * nx - ny * ny;
    if (j == 0)
        kx_values[i + ll] = kx;
    if (kz2 <= CU_REAL(0)) {
        positive[index] = CU_REAL(0);
        negative[index] = CU_REAL(0);
        return;
    }
    const CU_REAL nz = sqrt(kz2);
    CUCOMPLEX px = CUFFT_ZERO, py = CUFFT_ZERO, pz = CUFFT_ZERO;
    CUCOMPLEX mx = CUFFT_ZERO, my = CUFFT_ZERO, mz = CUFFT_ZERO;
    for (int dipole = 0; dipole < dipole_count; ++dipole) {
        const int base = 3 * dipole;
        const CU_REAL transverse = nx * xs[dipole] + ny * ys[dipole];
        const CUCOMPLEX *v = field + base;
        const CUCOMPLEX lp =
            CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(v[0], CUFFT_REAL(nx)), CUFFT_MUL(v[1], CUFFT_REAL(ny))),
                      CUFFT_MUL(v[2], CUFFT_REAL(nz)));
        const CUCOMPLEX lm =
            CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(v[0], CUFFT_REAL(nx)), CUFFT_MUL(v[1], CUFFT_REAL(ny))),
                      CUFFT_MUL(v[2], CUFFT_REAL(-nz)));
        const CU_REAL ap = k0 * (transverse + nz * zs[dipole]);
        const CU_REAL am = k0 * (transverse - nz * zs[dipole]);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const CUCOMPLEX ep = make_cuFloatComplex(cosf(ap), -sinf(ap));
        const CUCOMPLEX em = make_cuFloatComplex(cosf(am), -sinf(am));
#else
        const CUCOMPLEX ep = make_cuDoubleComplex(cos(ap), -sin(ap));
        const CUCOMPLEX em = make_cuDoubleComplex(cos(am), -sin(am));
#endif
        px = CUFFT_ADD(px, CUFFT_MUL(ep, CUFFT_SUB(v[0], CUFFT_MUL(lp, CUFFT_REAL(nx)))));
        py = CUFFT_ADD(py, CUFFT_MUL(ep, CUFFT_SUB(v[1], CUFFT_MUL(lp, CUFFT_REAL(ny)))));
        pz = CUFFT_ADD(pz, CUFFT_MUL(ep, CUFFT_SUB(v[2], CUFFT_MUL(lp, CUFFT_REAL(nz)))));
        mx = CUFFT_ADD(mx, CUFFT_MUL(em, CUFFT_SUB(v[0], CUFFT_MUL(lm, CUFFT_REAL(nx)))));
        my = CUFFT_ADD(my, CUFFT_MUL(em, CUFFT_SUB(v[1], CUFFT_MUL(lm, CUFFT_REAL(ny)))));
        mz = CUFFT_ADD(mz, CUFFT_MUL(em, CUFFT_ADD(v[2], CUFFT_MUL(lm, CUFFT_REAL(nz)))));
    }
    positive[index] =
        (px.x * px.x + px.y * px.y + py.x * py.x + py.y * py.y + pz.x * pz.x + pz.y * pz.y) *
        const_poynting;
    negative[index] =
        (mx.x * mx.x + mx.y * mx.y + my.x * my.x + my.y * my.y + mz.x * mz.x + mz.y * mz.y) *
        const_poynting;
}
} // namespace

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t direct_scattering_angles_wrapper(
    const CUCOMPLEX *field, const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
    int dipole_count, int ntheta, int nphi, CU_REAL k0, CU_REAL theta0, CU_REAL phi0,
    CU_REAL dtheta, CU_REAL dphi, CU_REAL const_poynting, CU_REAL *poynting, CU_REAL *theta_values,
    CU_REAL *phi_values, CU_REAL *weighted_scattering, CU_REAL *weighted_asymmetry,
    CU_REAL *reduction_ones) {
    if (!field || !xs || !ys || !zs || !poynting || !theta_values || !phi_values ||
        !weighted_scattering || !weighted_asymmetry || !reduction_ones || dipole_count <= 0 ||
        ntheta <= 0 || nphi <= 0)
        return cudaErrorInvalidValue;
    const size_t count = size_t(ntheta + 1) * size_t(nphi);
    direct_scattering_angles_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        field, xs, ys, zs, dipole_count, ntheta, nphi, k0, theta0, phi0, dtheta, dphi,
        const_poynting, poynting, theta_values, phi_values, weighted_scattering, weighted_asymmetry,
        reduction_ones);
    return finish_kernel_launch(213, "direct_scattering_angles_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t direct_scattering_spectrum_wrapper(
    const CUCOMPLEX *field, const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
    int dipole_count, int ll, int np, CU_REAL k0, CU_REAL delta, CU_REAL const_poynting,
    CU_REAL *positive, CU_REAL *negative, CU_REAL *kx_values) {
    if (!field || !xs || !ys || !zs || !positive || !negative || !kx_values || dipole_count <= 0 ||
        ll < 0 || np != 2 * ll + 1)
        return cudaErrorInvalidValue;
    const size_t count = size_t(np) * size_t(np);
    direct_scattering_spectrum_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        field, xs, ys, zs, dipole_count, ll, np, k0, delta, const_poynting, positive, negative,
        kx_values);
    return finish_kernel_launch(214, "direct_scattering_spectrum_kernel");
}

namespace {
//-----------------------------------------------------------
__global__ void fourier_to_image_rearrange_kernel(CUCOMPLEX *fields, int nfft,
                                                  CU_REAL scale) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int half = nfft / 2;
    const int pair_count = nfft * half;
    if (index >= pair_count)
        return;

    const int i = index % nfft - half;
    const int j = index / nfft - half;
    const int x_source = i < 0 ? nfft + i : i;
    const int y_source = nfft + j;
    const int source = x_source + nfft * y_source;
    const int target = (i + half) + nfft * (j + half);
    const int plane = nfft * nfft;

    for (int component = 0; component < 6; ++component) {
        CUCOMPLEX *field = fields + size_t(component) * size_t(plane);
        const CUCOMPLEX temporary = field[target];
        field[target] = CUFFT_MUL(field[source], CUFFT_REAL(scale));
        field[source] = CUFFT_MUL(temporary, CUFFT_REAL(scale));
    }
}

//-----------------------------------------------------------
__device__ inline void bright_field_rotate_components(
    CUCOMPLEX& ex, CUCOMPLEX& ey, CUCOMPLEX& ez,
    CU_REAL u1, CU_REAL u2, CU_REAL sint, CU_REAL cost, CU_REAL scale) {
    const CUCOMPLEX tx = CUFFT_ADD(
        CUFFT_ADD(CUFFT_MUL(ex, CUFFT_REAL(u1 * u1 + (CU_REAL(1) - u1 * u1) * cost)),
                  CUFFT_MUL(ey, CUFFT_REAL(u1 * u2 * (CU_REAL(1) - cost)))),
        CUFFT_MUL(ez, CUFFT_REAL(u2 * sint)));
    const CUCOMPLEX ty = CUFFT_ADD(
        CUFFT_ADD(CUFFT_MUL(ex, CUFFT_REAL(u1 * u2 * (CU_REAL(1) - cost))),
                  CUFFT_MUL(ey, CUFFT_REAL(u2 * u2 + (CU_REAL(1) - u2 * u2) * cost))),
        CUFFT_MUL(ez, CUFFT_REAL(-u1 * sint)));
    const CUCOMPLEX tz = CUFFT_ADD(
        CUFFT_ADD(CUFFT_MUL(ex, CUFFT_REAL(-u2 * sint)),
                  CUFFT_MUL(ey, CUFFT_REAL(u1 * sint))),
        CUFFT_MUL(ez, CUFFT_REAL(cost)));
    ex = CUFFT_MUL(tx, CUFFT_REAL(scale));
    ey = CUFFT_MUL(ty, CUFFT_REAL(scale));
    ez = CUFFT_MUL(tz, CUFFT_REAL(scale));
}

//-----------------------------------------------------------
__global__ void bright_field_prepare_fourier_kernel(
    const CUCOMPLEX* diffraction, CUCOMPLEX* fourier_fields,
    int nfft, int max_index, CU_REAL delta_kx, CU_REAL delta_ky,
    CU_REAL k0, CU_REAL aperture, CU_REAL lens_z, CU_REAL gross,
    int side, int microscope_type, CU_REAL aperture_inner,
    int incident_index_x, int incident_index_y, CU_REAL incident_scale,
    CUCOMPLEX incident_x, CUCOMPLEX incident_y, CUCOMPLEX incident_z) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft * nfft;
    if (index >= plane)
        return;
    const int x = index % nfft;
    const int y = index / nfft;
    const int half = nfft / 2;
    const int ik = x < half ? x : x - nfft;
    const int jk = y < half ? y : y - nfft;
    const size_t base = size_t(index);
    const size_t field_stride = size_t(plane);
    CUCOMPLEX ex = CUFFT_ZERO;
    CUCOMPLEX ey = CUFFT_ZERO;
    CUCOMPLEX ez = CUFFT_ZERO;
    CUCOMPLEX eix = CUFFT_ZERO;
    CUCOMPLEX eiy = CUFFT_ZERO;
    CUCOMPLEX eiz = CUFFT_ZERO;
    if (abs(ik) <= max_index && abs(jk) <= max_index) {
        const CU_REAL kx = CU_REAL(ik) * delta_kx;
        const CU_REAL ky = CU_REAL(jk) * delta_ky;
        const CU_REAL transverse = kx * kx + ky * ky;
        if (transverse <= aperture * aperture && transverse < k0 * k0) {
            const CU_REAL kz = sqrt(k0 * k0 - transverse);
            const CU_REAL phase_angle = kz * lens_z;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            CUCOMPLEX focus = make_cuFloatComplex(cosf(phase_angle), sinf(phase_angle));
#else
            CUCOMPLEX focus = make_cuDoubleComplex(cos(phase_angle), sin(phase_angle));
#endif
            const size_t diffraction_index = size_t(max_index + ik) +
                size_t(nfft) * size_t(max_index + jk);
            ex = CUFFT_MUL(diffraction[diffraction_index], focus);
            ey = CUFFT_MUL(diffraction[field_stride + diffraction_index], focus);
            ez = CUFFT_MUL(diffraction[2u * field_stride + diffraction_index], focus);
            eix = ex;
            eiy = ey;
            eiz = ez;
            if (ik == incident_index_x && jk == incident_index_y) {
                CUCOMPLEX incident_phase = incident_scale == CU_REAL(0)
                    ? CUFFT_ZERO : CUFFT_REAL(incident_scale);
                if (microscope_type == 2) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
                    incident_phase = make_cuFloatComplex(-CU_REAL(0), incident_scale);
#else
                    incident_phase = make_cuDoubleComplex(-CU_REAL(0), incident_scale);
#endif
                }
                eix = CUFFT_ADD(eix, CUFFT_MUL(CUFFT_MUL(incident_x, focus), incident_phase));
                eiy = CUFFT_ADD(eiy, CUFFT_MUL(CUFFT_MUL(incident_y, focus), incident_phase));
                eiz = CUFFT_ADD(eiz, CUFFT_MUL(CUFFT_MUL(incident_z, focus), incident_phase));
            }
            if (microscope_type == 5) {
                const CU_REAL transverse_magnitude = sqrt(transverse);
                if (transverse_magnitude >= aperture_inner * k0 &&
                    transverse_magnitude <= aperture) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
                    const CUCOMPLEX imaginary_unit = make_cuFloatComplex(0.0f, 1.0f);
#else
                    const CUCOMPLEX imaginary_unit = make_cuDoubleComplex(0.0, 1.0);
#endif
                    ex = CUFFT_MUL(ex, imaginary_unit);
                    ey = CUFFT_MUL(ey, imaginary_unit);
                    ez = CUFFT_MUL(ez, imaginary_unit);
                    eix = CUFFT_MUL(eix, imaginary_unit);
                    eiy = CUFFT_MUL(eiy, imaginary_unit);
                    eiz = CUFFT_MUL(eiz, imaginary_unit);
                }
            }
            if (transverse != CU_REAL(0)) {
                const CU_REAL ux = kx / k0;
                const CU_REAL uy = ky / k0;
                const CU_REAL uz = CU_REAL(side) * sqrt(CU_REAL(1) - ux * ux - uy * uy);
                const CU_REAL vx = -kx / k0 / gross;
                const CU_REAL vy = -ky / k0 / gross;
                const CU_REAL vz = CU_REAL(side) * sqrt(CU_REAL(1) - vx * vx - vy * vy);
                CU_REAL u1 = uy * vz - uz * vy;
                CU_REAL u2 = -ux * vz + uz * vx;
                const CU_REAL cost = ux * vx + uy * vy + uz * vz;
                const CU_REAL sint = sqrt(u1 * u1 + u2 * u2);
                const CU_REAL rotation_scale = sqrt(uz / vz);
                u1 /= sint;
                u2 /= sint;
                bright_field_rotate_components(eix, eiy, eiz, u1, u2, sint, cost,
                                               rotation_scale);
                bright_field_rotate_components(ex, ey, ez, u1, u2, sint, cost,
                                               rotation_scale);
            }
        }
    }
    fourier_fields[base] = ex;
    fourier_fields[field_stride + base] = ey;
    fourier_fields[2u * field_stride + base] = ez;
    fourier_fields[3u * field_stride + base] = eix;
    fourier_fields[4u * field_stride + base] = eiy;
    fourier_fields[5u * field_stride + base] = eiz;
}

//-----------------------------------------------------------
__global__ void bright_field_image_accumulate_kernel(
    const CUCOMPLEX* fourier_fields, CU_REAL* image_intensity, size_t plane_size,
    int first_component, int component_count) {
    const size_t index = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (index >= plane_size)
        return;
    for (int component = first_component;
         component < first_component + component_count; ++component) {
        const CUCOMPLEX value = fourier_fields[size_t(component) * plane_size + index];
        image_intensity[size_t(component) * plane_size + index] +=
            value.x * value.x + value.y * value.y;
    }
}

//-----------------------------------------------------------
__global__ void microsconfocal_detector_prepare_kernel(
    const CUCOMPLEX* diffraction, CUCOMPLEX* fields, int nfft, int max_index,
    CU_REAL k0, CU_REAL aperture, CU_REAL delta_kx, CU_REAL delta_ky,
    CU_REAL gross, CU_REAL lens_z, int side, int incident_index_x,
    int incident_index_y, CU_REAL incident_scale, CUCOMPLEX incident_x,
    CUCOMPLEX incident_y, CUCOMPLEX incident_z, CU_INT* device_status) {
    const int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
    const int plane = nfft * nfft;
    if (index >= plane)
        return;
    const int ix = index % nfft;
    const int iy = index / nfft;
    const int ik = ix < nfft - max_index ? ix : ix - nfft;
    const int jk = iy < nfft - max_index ? iy : iy - nfft;
    const size_t stride = size_t(plane);
    CUCOMPLEX ex = CUFFT_ZERO;
    CUCOMPLEX ey = CUFFT_ZERO;
    CUCOMPLEX ez = CUFFT_ZERO;
    CUCOMPLEX eix = CUFFT_ZERO;
    CUCOMPLEX eiy = CUFFT_ZERO;
    CUCOMPLEX eiz = CUFFT_ZERO;
    if (abs(ik) <= max_index && abs(jk) <= max_index) {
        const CU_REAL kx = CU_REAL(ik) * delta_kx;
        const CU_REAL ky = CU_REAL(jk) * delta_ky;
        const CU_REAL transverse = kx * kx + ky * ky;
        if (transverse <= aperture * aperture && transverse < k0 * k0) {
            const CU_REAL kz = CU_REAL(side) * sqrt(k0 * k0 - transverse);
            const CU_REAL angle = kz * lens_z;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            const CUCOMPLEX focus = make_cuFloatComplex(cosf(angle), sinf(angle));
#else
            const CUCOMPLEX focus = make_cuDoubleComplex(cos(angle), sin(angle));
#endif
            const size_t source = size_t(max_index + ik) +
                                  size_t(nfft) * size_t(max_index + jk);
            ex = CUFFT_MUL(diffraction[source], focus);
            ey = CUFFT_MUL(diffraction[stride + source], focus);
            ez = CUFFT_MUL(diffraction[2u * stride + source], focus);
            eix = ex;
            eiy = ey;
            eiz = ez;
            if (side == 10 && ik == incident_index_x && jk == incident_index_y) {
                const CUCOMPLEX incident_factor = CUFFT_MUL(
                    focus, CUFFT_REAL(incident_scale));
                eix = CUFFT_ADD(eix, CUFFT_MUL(incident_x, incident_factor));
                eiy = CUFFT_ADD(eiy, CUFFT_MUL(incident_y, incident_factor));
                eiz = CUFFT_ADD(eiz, CUFFT_MUL(incident_z, incident_factor));
            }
            if (transverse != CU_REAL(0)) {
                const CU_REAL ux = kx / k0;
                const CU_REAL uy = ky / k0;
                const CU_REAL uz = CU_REAL(side) * sqrt(CU_REAL(1) - ux*ux - uy*uy);
                const CU_REAL vx = -kx / k0 / gross;
                const CU_REAL vy = -ky / k0 / gross;
                const CU_REAL vz = CU_REAL(side) * sqrt(CU_REAL(1) - vx*vx - vy*vy);
                CU_REAL u1 = uy*vz - uz*vy;
                CU_REAL u2 = -ux*vz + uz*vx;
                const CU_REAL cost = ux*vx + uy*vy + uz*vz;
                const CU_REAL sint = sqrt(u1*u1 + u2*u2);
                if (sint == CU_REAL(0)) {
                    atomicExch(&device_status->dat, 1);
                } else {
                    u1 /= sint;
                    u2 /= sint;
                    const CU_REAL scale = sqrt(uz / vz);
                    bright_field_rotate_components(ex, ey, ez, u1, u2, sint,
                                                   cost, scale);
                    bright_field_rotate_components(eix, eiy, eiz, u1, u2, sint,
                                                   cost, scale);
                }
            }
        }
    }
    fields[index] = eix;
    fields[stride + size_t(index)] = eiy;
    fields[2u * stride + size_t(index)] = eiz;
    fields[3u * stride + size_t(index)] = ex;
    fields[4u * stride + size_t(index)] = ey;
    fields[5u * stride + size_t(index)] = ez;
}

//-----------------------------------------------------------
__global__ void microsconfocal_detector_accumulate_kernel(
    const CUCOMPLEX* fields, CUCOMPLEX* image_stack, int nfft,
    int plane_index, int plane_count, CU_REAL pixel_size, CU_REAL kx_incident,
    CU_REAL ky_incident, CU_REAL kz_incident, CU_REAL lens_z) {
    const int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
    const int plane = nfft * nfft;
    if (index >= plane)
        return;
    const int ix = index % nfft - nfft/2;
    const int iy = index / nfft - nfft/2;
    const CU_REAL x = pixel_size * CU_REAL(ix);
    const CU_REAL y = pixel_size * CU_REAL(iy);
    const CU_REAL angle = kx_incident*x + ky_incident*y - kz_incident*lens_z;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phase = make_cuFloatComplex(cosf(angle), sinf(angle));
#else
    const CUCOMPLEX phase = make_cuDoubleComplex(cos(angle), sin(angle));
#endif
    const size_t stack_index = size_t(index) + size_t(plane)*size_t(plane_index);
    for (int component = 0; component < 3; ++component) {
        CUCOMPLEX* output = image_stack + size_t(component)*size_t(plane)*size_t(plane_count);
        output[stack_index] = CUFFT_ADD(output[stack_index],
            CUFFT_MUL(fields[size_t(component)*size_t(plane) + size_t(index)], phase));
    }
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
fourier_to_image_rearrange_wrapper(CUCOMPLEX *fields, int nfft, CU_REAL scale) {
    if (!fields || nfft <= 0 || (nfft & 1) != 0 || nfft > INT_MAX / nfft)
        return cudaErrorInvalidValue;
    const size_t pair_count = size_t(nfft) * size_t(nfft / 2);
    fourier_to_image_rearrange_kernel<<<block_count(pair_count), THREADS_PER_BLOCK>>>(
        fields, nfft, scale);
    return finish_kernel_launch(219, "fourier_to_image_rearrange_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
microsconfocal_detector_prepare_wrapper(
    const CUCOMPLEX* diffraction, CUCOMPLEX* fourier_fields, int nfft,
    int max_index, CU_REAL k0, CU_REAL aperture, CU_REAL delta_kx,
    CU_REAL delta_ky, CU_REAL gross, CU_REAL lens_z, int side,
    int incident_index_x, int incident_index_y, CU_REAL incident_scale,
    CUCOMPLEX incident_x, CUCOMPLEX incident_y, CUCOMPLEX incident_z,
    CU_INT* device_status) {
    if (!diffraction || !fourier_fields || !device_status || nfft <= 0 ||
        (nfft & 1) != 0 || nfft > INT_MAX/nfft || max_index < 0 ||
        max_index >= nfft/2 || k0 <= CU_REAL(0) || aperture < CU_REAL(0) ||
        delta_kx == CU_REAL(0) || delta_ky == CU_REAL(0) || gross == CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft)*size_t(nfft);
    microsconfocal_detector_prepare_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        diffraction, fourier_fields, nfft, max_index, k0, aperture, delta_kx,
        delta_ky, gross, lens_z, side, incident_index_x, incident_index_y,
        incident_scale, incident_x, incident_y, incident_z, device_status);
    return finish_kernel_launch(261, "microsconfocal_detector_prepare_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
microsconfocal_detector_accumulate_wrapper(
    const CUCOMPLEX* fourier_fields, CUCOMPLEX* image_stack, int nfft,
    int plane_index, int plane_count, CU_REAL pixel_size, CU_REAL kx_incident,
    CU_REAL ky_incident, CU_REAL kz_incident, CU_REAL lens_z) {
    if (!fourier_fields || !image_stack || nfft <= 0 || (nfft & 1) != 0 ||
        nfft > INT_MAX/nfft || plane_count <= 0 || plane_index < 0 ||
        plane_index >= plane_count)
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft)*size_t(nfft);
    microsconfocal_detector_accumulate_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        fourier_fields, image_stack, nfft, plane_index, plane_count, pixel_size,
        kx_incident, ky_incident, kz_incident, lens_z);
    return finish_kernel_launch(262, "microsconfocal_detector_accumulate_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
microsconfocal_detector_zero_wrapper(CUCOMPLEX* image_stack, size_t value_count) {
    if (!image_stack || value_count == 0 ||
        value_count > std::numeric_limits<size_t>::max()/sizeof(CUCOMPLEX))
        return cudaErrorInvalidValue;
    return cudaMemset(image_stack, 0, value_count*sizeof(CUCOMPLEX));
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
bright_field_prepare_fourier_wrapper(
    const CUCOMPLEX* diffraction, CUCOMPLEX* fourier_fields,
    int nfft, int max_index, CU_REAL delta_kx, CU_REAL delta_ky,
    CU_REAL k0, CU_REAL aperture, CU_REAL lens_z, CU_REAL gross,
    int side, int microscope_type, CU_REAL aperture_inner,
    int incident_index_x, int incident_index_y, CU_REAL incident_scale,
    CUCOMPLEX incident_x, CUCOMPLEX incident_y, CUCOMPLEX incident_z) {
    if (!diffraction || !fourier_fields || nfft <= 0 || (nfft & 1) != 0 ||
        nfft > INT_MAX / nfft || max_index < 0 || max_index >= nfft / 2 ||
        delta_kx == CU_REAL(0) || delta_ky == CU_REAL(0) || k0 <= CU_REAL(0) ||
        gross == CU_REAL(0) || (side != 1 && side != -1) || aperture < CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t plane_size = size_t(nfft) * size_t(nfft);
    bright_field_prepare_fourier_kernel<<<block_count(plane_size), THREADS_PER_BLOCK>>>(
        diffraction, fourier_fields, nfft, max_index, delta_kx, delta_ky,
        k0, aperture, lens_z, gross, side, microscope_type, aperture_inner,
        incident_index_x, incident_index_y, incident_scale,
        incident_x, incident_y, incident_z);
    return finish_kernel_launch(254, "bright_field_prepare_fourier_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
bright_field_image_accumulate_wrapper(
    const CUCOMPLEX* fourier_fields, CU_REAL* image_intensity, size_t plane_size,
    int first_component, int component_count) {
    if (!fourier_fields || !image_intensity || plane_size == 0 ||
        first_component < 0 || component_count <= 0 ||
        first_component > 6 - component_count)
        return cudaErrorInvalidValue;
    bright_field_image_accumulate_kernel<<<block_count(plane_size), THREADS_PER_BLOCK>>>(
        fourier_fields, image_intensity, plane_size, first_component, component_count);
    return finish_kernel_launch(255, "bright_field_image_accumulate_kernel");
}

//-----------------------------------------------------------
__global__ void quick_diffraction_gather_kernel(const CUCOMPLEX *field, CUCOMPLEX *images,
                                                const CU_INT*fft_map, int nx, int ny, int nxny,
                                                int nfft, int plane_index) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft * nfft;
    if (index >= plane)
        return;
    images[index] = CUFFT_ZERO;
    images[plane + index] = CUFFT_ZERO;
    images[2 * plane + index] = CUFFT_ZERO;
    const int x = index % nfft;
    const int y = index / nfft;
    if (x < nx && y < ny) {
        const int dipole = x + nx * y + nxny * plane_index;
        const int target = (fft_map[x].dat - 1) + nfft * (fft_map[y].dat - 1);
        images[target] = field[3 * dipole];
        images[plane + target] = field[3 * dipole + 1];
        images[2 * plane + target] = field[3 * dipole + 2];
    }
}

//-----------------------------------------------------------
__global__ void quick_diffraction_build_fft_map_kernel(CU_INT*fft_map, int nfft) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= nfft)
        return;
    const int half = nfft / 2;
    fft_map[index].dat = index - half >= 0 ? index - half + 1 : half + index + 1;
}

//-----------------------------------------------------------
__global__ void quick_diffraction_accumulate_kernel(const CUCOMPLEX *images, const CU_INT*fft_map,
                                                    const CU_REAL *zs, int nxny, int nfft, int imax,
                                                    int plane_index, CU_REAL k0, CU_REAL deltak,
                                                    CU_REAL phase_x, CU_REAL phase_y,
                                                    CUCOMPLEX *positive, CUCOMPLEX *negative) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = (2 * imax + 1) * (2 * imax + 1);
    if (index >= count)
        return;
    const int i = index % (2 * imax + 1) - imax;
    const int j = index / (2 * imax + 1) - imax;
    const CU_REAL kx = CU_REAL(i) * deltak;
    const CU_REAL ky = CU_REAL(j) * deltak;
    const CU_REAL radicand = k0 * k0 - kx * kx - ky * ky;
    if (radicand <= CU_REAL(0))
        return;
    const CU_REAL kz = sqrt(radicand);
    const int half = nfft / 2;
    const int source = (fft_map[i + half].dat - 1) + nfft * (fft_map[j + half].dat - 1);
    const int plane = nfft * nfft;
    const CU_REAL z = zs[nxny * plane_index];
    const CU_REAL phase = -(phase_x * CU_REAL(i) + phase_y * CU_REAL(j));
    const CU_REAL phase_positive = -kz * z + phase;
    const CU_REAL phase_negative = kz * z + phase;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX ep = make_cuFloatComplex(cosf(phase_positive), sinf(phase_positive));
    const CUCOMPLEX en = make_cuFloatComplex(cosf(phase_negative), sinf(phase_negative));
#else
    const CUCOMPLEX ep = make_cuDoubleComplex(cos(phase_positive), sin(phase_positive));
    const CUCOMPLEX en = make_cuDoubleComplex(cos(phase_negative), sin(phase_negative));
#endif
    const CUCOMPLEX ex = images[source];
    const CUCOMPLEX ey = images[plane + source];
    const CUCOMPLEX ez = images[2 * plane + source];
    const CUCOMPLEX dotp =
        CUFFT_MUL(CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(ex, CUFFT_REAL(kx)), CUFFT_MUL(ey, CUFFT_REAL(ky))),
                            CUFFT_MUL(ez, CUFFT_REAL(kz))),
                  CUFFT_REAL(CU_REAL(1) / k0));
    const CUCOMPLEX dotn =
        CUFFT_MUL(CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(ex, CUFFT_REAL(kx)), CUFFT_MUL(ey, CUFFT_REAL(ky))),
                            CUFFT_MUL(ez, CUFFT_REAL(-kz))),
                  CUFFT_REAL(CU_REAL(1) / k0));
    const int out = (imax + i) + nfft * (imax + j);
    positive[out] = CUFFT_ADD(positive[out],
                              CUFFT_MUL(ep, CUFFT_SUB(ex, CUFFT_MUL(dotp, CUFFT_REAL(kx / k0)))));
    positive[out + plane] = CUFFT_ADD(
        positive[out + plane], CUFFT_MUL(ep, CUFFT_SUB(ey, CUFFT_MUL(dotp, CUFFT_REAL(ky / k0)))));
    positive[out + 2 * plane] =
        CUFFT_ADD(positive[out + 2 * plane],
                  CUFFT_MUL(ep, CUFFT_SUB(ez, CUFFT_MUL(dotp, CUFFT_REAL(kz / k0)))));
    negative[out] = CUFFT_ADD(negative[out],
                              CUFFT_MUL(en, CUFFT_SUB(ex, CUFFT_MUL(dotn, CUFFT_REAL(kx / k0)))));
    negative[out + plane] = CUFFT_ADD(
        negative[out + plane], CUFFT_MUL(en, CUFFT_SUB(ey, CUFFT_MUL(dotn, CUFFT_REAL(ky / k0)))));
    negative[out + 2 * plane] =
        CUFFT_ADD(negative[out + 2 * plane],
                  CUFFT_MUL(en, CUFFT_ADD(ez, CUFFT_MUL(dotn, CUFFT_REAL(kz / k0)))));
}
} // namespace

//-------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
quick_diffraction_gather_wrapper(const CUCOMPLEX *field, CUCOMPLEX *images, const CU_INT*fft_map,
                                 int nx, int ny, int nxny, int nfft, int plane_index) {
    if (!field || !images || !fft_map || nx <= 0 || ny <= 0 || nxny <= 0 || nfft <= 0 ||
        plane_index < 0)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nfft) * size_t(nfft);
    quick_diffraction_gather_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        field, images, fft_map, nx, ny, nxny, nfft, plane_index);
    return finish_kernel_launch(215, "quick_diffraction_gather_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
quick_diffraction_build_fft_map_wrapper(CU_INT*fft_map, int nfft) {
    if (!fft_map || nfft <= 0)
        return cudaErrorInvalidValue;
    quick_diffraction_build_fft_map_kernel<<<block_count(size_t(nfft)), THREADS_PER_BLOCK>>>(
        fft_map, nfft);
    return finish_kernel_launch(256, "quick_diffraction_build_fft_map_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t quick_diffraction_accumulate_wrapper(
    const CUCOMPLEX *images, const CU_INT*fft_map, const CU_REAL *zs, int nxny, int nfft, int imax,
    int plane_index, CU_REAL k0, CU_REAL deltak, CU_REAL phase_x, CU_REAL phase_y,
    CUCOMPLEX *positive, CUCOMPLEX *negative) {
    if (!images || !fft_map || !zs || !positive || !negative || nxny <= 0 || nfft <= 0 ||
        imax < 0 || plane_index < 0 || k0 <= CU_REAL(0) || deltak <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t count = size_t(2 * imax + 1) * size_t(2 * imax + 1);
    quick_diffraction_accumulate_kernel<<<dim3(block_count(count), 1, 1), THREADS_PER_BLOCK>>>(
        images, fft_map, zs, nxny, nfft, imax, plane_index, k0, deltak, phase_x, phase_y, positive,
        negative);
    return finish_kernel_launch(216, "quick_diffraction_accumulate_kernel");
}

namespace {
//-----------------------------------------------------------
__device__ CU_REAL quick_field_norm(const CUCOMPLEX *field, int plane, int index) {
    const CUCOMPLEX x = field[index];
    const CUCOMPLEX y = field[plane + index];
    const CUCOMPLEX z = field[2 * plane + index];
    return x.x * x.x + x.y * x.y + y.x * y.x + y.y * y.y + z.x * z.x + z.y * z.y;
}

//-----------------------------------------------------------
__global__ void quick_diffraction_grid_kernel(
    const CUCOMPLEX *positive_field, const CUCOMPLEX *negative_field, int nfft, int imax,
    CU_REAL deltakx, CU_REAL deltaky, CU_REAL k0, CU_REAL theta0, CU_REAL phi0,
    CU_REAL const_poynting, int np, CU_REAL *positive, CU_REAL *negative, CU_REAL *kx_values,
    CU_REAL *weighted_scattering, CU_REAL *weighted_asymmetry, CU_REAL *reduction_ones) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = np * np;
    if (index >= count)
        return;
    const int ix = index % np;
    const int iy = index / np;
    const int i = ix - imax;
    const int j = iy - imax;
    const CU_REAL kx = CU_REAL(i) * deltakx;
    const CU_REAL ky = CU_REAL(j) * deltaky;
    if (iy == 0)
        kx_values[ix] = kx;
    const CU_REAL radicand = k0 * k0 - kx * kx - ky * ky;
    const int plane = nfft * nfft;
    const int source = (imax + i) + nfft * (imax + j);
    const CU_REAL epos = quick_field_norm(positive_field, plane, source);
    const CU_REAL eneg = quick_field_norm(negative_field, plane, source);
    positive[index] = epos * const_poynting;
    negative[index] = eneg * const_poynting;
    if (radicand <= CU_REAL(0)) {
        weighted_scattering[index] = CU_REAL(0);
        weighted_asymmetry[index] = CU_REAL(0);
        reduction_ones[index] = CU_REAL(1);
        return;
    }
    const CU_REAL kz = sqrt(radicand);
    const CU_REAL weight = deltakx * deltaky / kz;
    const CU_REAL st = sin(theta0), ct = cos(theta0);
    const CU_REAL cp = cos(phi0), sp = sin(phi0);
    const CU_REAL direction_positive = (kx * st * cp + ky * st * sp + kz * ct) / k0;
    const CU_REAL direction_negative = (kx * st * cp + ky * st * sp - kz * ct) / k0;
    weighted_scattering[index] = weight * (epos + eneg);
    weighted_asymmetry[index] = weight * (epos * direction_positive + eneg * direction_negative);
    reduction_ones[index] = CU_REAL(1);
}

//-----------------------------------------------------------
__global__ void quick_diffraction_interpolation_kernel(const CUCOMPLEX *positive_field,
                                                       const CUCOMPLEX *negative_field, int nfft,
                                                       int imax, CU_REAL deltakx, CU_REAL deltaky,
                                                       CU_REAL k0, int ntheta, int nphi,
                                                       CU_REAL const_poynting, CU_REAL *poynting,
                                                       CU_REAL *theta_values, CU_REAL *phi_values) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = (ntheta + 1) * nphi;
    if (index >= count)
        return;
    const int itheta = index / nphi;
    const int iphi = index % nphi;
    const CU_REAL pi = CU_REAL(3.14159265358979323846);
    const CU_REAL theta = pi * CU_REAL(itheta) / CU_REAL(ntheta);
    const CU_REAL phi = CU_REAL(2) * pi * CU_REAL(iphi) / CU_REAL(nphi);
    const CU_REAL kx = k0 * sin(theta) * cos(phi);
    const CU_REAL ky = k0 * sin(theta) * sin(phi);
    int i2 = int(floor(kx / deltakx));
    int j2 = int(floor(ky / deltakx));
    if (i2 < -imax)
        i2 = -imax;
    if (i2 >= imax)
        i2 = imax - 1;
    if (j2 < -imax)
        j2 = -imax;
    if (j2 >= imax)
        j2 = imax - 1;
    const int ii = imax + i2;
    const int jj = imax + j2;
    const int plane = nfft * nfft;
    const CUCOMPLEX *field = theta <= pi / CU_REAL(2) ? positive_field : negative_field;
    const CU_REAL e11 = quick_field_norm(field, plane, ii + nfft * jj);
    const CU_REAL e22 = quick_field_norm(field, plane, (ii + 1) + nfft * (jj + 1));
    const CU_REAL e21 = quick_field_norm(field, plane, (ii + 1) + nfft * jj);
    const CU_REAL e12 = quick_field_norm(field, plane, ii + nfft * (jj + 1));
    int zeros =
        (e11 == CU_REAL(0)) + (e22 == CU_REAL(0)) + (e21 == CU_REAL(0)) + (e12 == CU_REAL(0));
    CU_REAL emod;
    if (zeros == 0) {
        const CU_REAL dx = kx - deltakx * CU_REAL(i2);
        const CU_REAL dy = ky - deltakx * CU_REAL(j2);
        emod = (e21 - e11) * dx / deltakx + (e12 - e11) * dy / deltakx +
               (e11 + e22 - e12 - e21) * dx / deltakx * dy / deltakx + e11;
    } else
        emod = (e11 + e12 + e21 + e22) / CU_REAL(4 - zeros);
    poynting[index] = emod * const_poynting;
    theta_values[index] = theta * CU_REAL(180) / pi;
    phi_values[index] = phi * CU_REAL(180) / pi;
}
} // namespace

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t quick_diffraction_postprocess_wrapper(
    const CUCOMPLEX *positive_field, const CUCOMPLEX *negative_field, int nfft, int imax,
    CU_REAL deltakx, CU_REAL deltaky, CU_REAL k0, CU_REAL theta0, CU_REAL phi0,
    CU_REAL const_poynting, int ntheta, int nphi, CU_REAL *positive, CU_REAL *negative,
    CU_REAL *kx_values, CU_REAL *angular_poynting, CU_REAL *theta_values, CU_REAL *phi_values,
    CU_REAL *weighted_scattering, CU_REAL *weighted_asymmetry, CU_REAL *reduction_ones) {
    if (!positive_field || !negative_field || !positive || !negative || !kx_values ||
        !angular_poynting || !theta_values || !phi_values || !weighted_scattering ||
        !weighted_asymmetry || !reduction_ones || nfft <= 0 || imax < 0 || ntheta <= 0 || nphi <= 0)
        return cudaErrorInvalidValue;
    const int np = 2 * imax + 1;
    const size_t spectrum_count = size_t(np) * size_t(np);
    const size_t angle_count = size_t(ntheta + 1) * size_t(nphi);
    quick_diffraction_grid_kernel<<<block_count(spectrum_count), THREADS_PER_BLOCK>>>(
        positive_field, negative_field, nfft, imax, deltakx, deltaky, k0, theta0, phi0,
        const_poynting, np, positive, negative, kx_values, weighted_scattering, weighted_asymmetry,
        reduction_ones);
    cudaError_t status = finish_kernel_launch(217, "quick_diffraction_grid_kernel");
    if (status != cudaSuccess)
        return status;
    quick_diffraction_interpolation_kernel<<<block_count(angle_count), THREADS_PER_BLOCK>>>(
        positive_field, negative_field, nfft, imax, deltakx, deltaky, k0, ntheta, nphi,
        const_poynting, angular_poynting, theta_values, phi_values);
    return finish_kernel_launch(218, "quick_diffraction_interpolation_kernel");
}

namespace {
//-----------------------------------------------------------
__global__ void quick_energy_incident_wave_kernel(CUCOMPLEX *incident, int nfft, CU_REAL spacing,
                                                  CU_REAL scale, CU_REAL kx, CU_REAL ky,
                                                  CUCOMPLEX kz, CUCOMPLEX ex0, CUCOMPLEX ey0,
                                                  CUCOMPLEX ez0) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft * nfft;
    if (index >= plane)
        return;
    const int xindex = index % nfft;
    const int yindex = index / nfft;
    const int half = nfft / 2;
    const int ix = xindex < half ? xindex : xindex - nfft;
    const int iy = yindex < half ? yindex : yindex - nfft;
    const CU_REAL phase = kx * spacing * CU_REAL(ix) + ky * spacing * CU_REAL(iy);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX phasor = make_cuFloatComplex(cosf(phase), sinf(phase));
#else
    const CUCOMPLEX phasor = make_cuDoubleComplex(cos(phase), sin(phase));
#endif
    incident[index] = CUFFT_MUL(CUFFT_MUL(ex0, phasor), CUFFT_REAL(scale));
    incident[plane + index] = CUFFT_MUL(CUFFT_MUL(ey0, phasor), CUFFT_REAL(scale));
    incident[2 * plane + index] = CUFFT_MUL(CUFFT_MUL(ez0, phasor), CUFFT_REAL(scale));
    (void)kz;
}

//-----------------------------------------------------------
__global__ void quick_energy_scale_fields_kernel(CUCOMPLEX *positive, CUCOMPLEX *negative, int nfft,
                                                 int imax, CU_REAL dkx, CU_REAL dky, CU_REAL k0) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = (2 * imax + 1) * (2 * imax + 1);
    if (index >= count)
        return;
    const int i = index % (2 * imax + 1) - imax;
    const int j = index / (2 * imax + 1) - imax;
    const CU_REAL kx = CU_REAL(i) * dkx;
    const CU_REAL ky = CU_REAL(j) * dky;
    const CU_REAL radicand = k0 * k0 - kx * kx - ky * ky;
    if (radicand <= CU_REAL(0))
        return;
    const CU_REAL scale = k0 * k0 / (CU_REAL(2) * CU_REAL(3.14159265358979323846) * sqrt(radicand));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX factor = make_cuFloatComplex(0.0f, float(scale));
#else
    const CUCOMPLEX factor = make_cuDoubleComplex(0.0, scale);
#endif
    const int plane = nfft * nfft;
    const int target = (imax + i) + nfft * (imax + j);
    for (int component = 0; component < 3; ++component) {
        positive[component * plane + target] =
            CUFFT_MUL(positive[component * plane + target], factor);
        negative[component * plane + target] =
            CUFFT_MUL(negative[component * plane + target], factor);
    }
}

//-----------------------------------------------------------
__device__ CUCOMPLEX confocal_energy_make(CU_REAL real_part, CU_REAL imag_part) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real_part, imag_part);
#else
    return make_cuDoubleComplex(real_part, imag_part);
#endif
}

__global__ void confocal_energy_spectrum_kernel(
    CUCOMPLEX* incident, CUCOMPLEX* positive, CUCOMPLEX* negative,
    const CU_REAL* phase_positive, const CU_REAL* phase_negative,
    int nfft, int imax, CU_REAL k0, CU_REAL dkx, CU_REAL dky,
    CU_REAL kmax2, CU_REAL psi, CU_REAL x0, CU_REAL y0, CU_REAL z0,
    CUCOMPLEX e0, int nside, int beam_kind) {
    const int width = 2 * imax + 1;
    const int t = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= width * width)
        return;
    const int i = t % width - imax;
    const int j = t / width - imax;
    const CU_REAL kx = CU_REAL(i) * dkx;
    const CU_REAL ky = CU_REAL(j) * dky;
    const CU_REAL kp2 = kx * kx + ky * ky;
    if (kp2 > kmax2 * CU_REAL(0.999))
        return;
    const CU_REAL kz2 = k0 * k0 - kp2;
    if (kz2 <= CU_REAL(0))
        return;
    const CU_REAL kz = sqrt(kz2);
    const CU_REAL root_x = sqrt(kz2 + kx * kx);
    const CU_REAL root_y = sqrt(kz2 + ky * ky);
    if (root_x == CU_REAL(0) || root_y == CU_REAL(0))
        return;
    const CU_REAL cpsi = cos(psi);
    const CU_REAL spsi = sin(psi);
    const CU_REAL phasep = CU_REAL(6.2831853071795864769) * phase_positive[t];
    const CU_REAL phasen = CU_REAL(6.2831853071795864769) * phase_negative[t];
    const CU_REAL anglep = -(kx * x0 + ky * y0 + kz * z0) + phasep;
    const CU_REAL anglen = -(kx * x0 + ky * y0 - kz * z0) + phasen;
    const CU_REAL cx = cpsi * kz / root_x;
    const CU_REAL cy = spsi * kz / root_y;
    const CU_REAL cz = -cpsi * kx / root_x - spsi * ky / root_y;
    const CU_REAL ar = e0.x;
    const CU_REAL ai = e0.y;
    const CU_REAL cp = cos(anglep);
    const CU_REAL sp = sin(anglep);
    const CU_REAL cn = cos(anglen);
    const CU_REAL sn = sin(anglen);
    const CUCOMPLEX xp = confocal_energy_make((ar * cp - ai * sp) * cx,
                                               (ar * sp + ai * cp) * cx);
    const CUCOMPLEX yp = confocal_energy_make((ar * cp - ai * sp) * cy,
                                               (ar * sp + ai * cp) * cy);
    const CUCOMPLEX zp = confocal_energy_make((ar * cp - ai * sp) * cz,
                                               (ar * sp + ai * cp) * cz);
    CUCOMPLEX xn = confocal_energy_make(CU_REAL(0), CU_REAL(0));
    CUCOMPLEX yn = xn;
    CUCOMPLEX zn = xn;
    if (beam_kind == 0) {
        xn = confocal_energy_make((ar * cn - ai * sn) * cx,
                                  (ar * sn + ai * cn) * cx);
        yn = confocal_energy_make((ar * cn - ai * sn) * cy,
                                  (ar * sn + ai * cn) * cy);
        zn = confocal_energy_make(-(ar * cn - ai * sn) * cz,
                                  -(ar * sn + ai * cn) * cz);
    }
    const int xi = i >= 0 ? i : nfft + i;
    const int yi = j >= 0 ? j : nfft + j;
    const size_t index = size_t(xi) + size_t(nfft) * size_t(yi);
    const size_t plane = size_t(nfft) * size_t(nfft);
    positive[index] = xp;
    positive[index + plane] = yp;
    positive[index + 2 * plane] = zp;
    negative[index] = xn;
    negative[index + plane] = yn;
    negative[index + 2 * plane] = zn;
    if (nside == 1) {
        incident[index] = xp;
        incident[index + plane] = yp;
        incident[index + 2 * plane] = zp;
    } else if (nside == -1) {
        incident[index] = xn;
        incident[index + plane] = yn;
        incident[index + 2 * plane] = zn;
    }
}
//-------------------------------------------

__global__ void quick_energy_flux_kernel(const CUCOMPLEX *incident, const CUCOMPLEX *positive,
                                         const CUCOMPLEX *negative, int nfft, int imax, CU_REAL dkx,
                                         CU_REAL dky, CU_REAL k0, CU_REAL *flux_incident,
                                         CU_REAL *flux_reflected, CU_REAL *flux_transmitted,
                                         CU_REAL *energy_inside_na, CU_REAL *energy_total,
                                         CU_REAL *ones) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft * nfft;
    if (index >= plane)
        return;
    const int ix = index % nfft;
    const int iy = index / nfft;
    const int i = ix < nfft / 2 ? ix : ix - nfft;
    const int j = iy < nfft / 2 ? iy : iy - nfft;
    const CU_REAL kx = CU_REAL(i) * dkx;
    const CU_REAL ky = CU_REAL(j) * dky;
    const CU_REAL e2 = quick_field_norm(incident, plane, index);
    const CU_REAL radicand = k0 * k0 - kx * kx - ky * ky;
    const bool in_na = (i >= -imax && i <= imax && j >= -imax && j <= imax);
    energy_inside_na[index] = in_na && radicand > CU_REAL(0) ? e2 : CU_REAL(0);
    energy_total[index] = e2;
    ones[index] = CU_REAL(1);
    flux_incident[index] = CU_REAL(0);
    flux_reflected[index] = CU_REAL(0);
    flux_transmitted[index] = CU_REAL(0);
    if (radicand <= CU_REAL(0) || !in_na)
        return;
    const CU_REAL kz = sqrt(radicand);
    const int target = (imax + i) + nfft * (imax + j);
    const CU_REAL er = quick_field_norm(negative, plane, target);
    CU_REAL et = CU_REAL(0);
    for (int component = 0; component < 3; ++component) {
        const CUCOMPLEX total =
            CUFFT_ADD(positive[component * plane + target], incident[component * plane + index]);
        et += total.x * total.x + total.y * total.y;
    }
    flux_incident[index] = e2 * kz;
    flux_reflected[index] = er * kz;
    flux_transmitted[index] = et * kz;
}
//-------------------------------------------

__global__ void confocal_energy_flux_kernel(
    const CUCOMPLEX* incident_positive, const CUCOMPLEX* incident_negative,
    const CUCOMPLEX* scattered_positive, const CUCOMPLEX* scattered_negative,
    int nfft, int imax, CU_REAL dkx, CU_REAL dky, CU_REAL k0,
    CU_REAL kmax2, CU_REAL* flux_incident, CU_REAL* flux_reflected,
    CU_REAL* flux_transmitted) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft * nfft;
    if (index >= plane)
        return;
    const int ix = index % nfft;
    const int iy = index / nfft;
    const int i = ix < nfft / 2 ? ix : ix - nfft;
    const int j = iy < nfft / 2 ? iy : iy - nfft;
    flux_incident[index] = CU_REAL(0);
    flux_reflected[index] = CU_REAL(0);
    flux_transmitted[index] = CU_REAL(0);
    const CU_REAL kx = CU_REAL(i) * dkx;
    const CU_REAL ky = CU_REAL(j) * dky;
    const CU_REAL kz2 = k0 * k0 - kx * kx - ky * ky;
    if (i < -imax || i > imax || j < -imax || j > imax ||
        kx * kx + ky * ky > kmax2 * CU_REAL(0.999) || kz2 <= CU_REAL(0))
        return;
    const int target = (imax + i) + nfft * (imax + j);
    const CU_REAL kz = sqrt(kz2);
    const CU_REAL incident = quick_field_norm(incident_positive, plane, index) +
                             quick_field_norm(incident_negative, plane, index);
    CU_REAL reflected = CU_REAL(0);
    CU_REAL transmitted = CU_REAL(0);
    for (int component = 0; component < 3; ++component) {
        const CUCOMPLEX negative_total =
            CUFFT_ADD(scattered_negative[component * plane + target],
                      incident_negative[component * plane + index]);
        const CUCOMPLEX positive_total =
            CUFFT_ADD(scattered_positive[component * plane + target],
                      incident_positive[component * plane + index]);
        reflected += negative_total.x * negative_total.x + negative_total.y * negative_total.y;
        transmitted += positive_total.x * positive_total.x + positive_total.y * positive_total.y;
    }
    flux_incident[index] = incident * kz;
    flux_reflected[index] = reflected * kz;
    flux_transmitted[index] = transmitted * kz;
}

//-----------------------------------------------------------
__global__ void confocal_fluorescence_zero_kernel(
    CU_REAL* intensity,
    size_t count)
{
    const size_t index = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (index < count)
        intensity[index] = CU_REAL(0);
}

//-----------------------------------------------------------
__global__ void confocal_fluorescence_accumulate_kernel(
    const CUCOMPLEX* fields,
    CU_REAL* intensity,
    size_t plane_size,
    size_t plane_index)
{
    const size_t pixel = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (pixel >= plane_size)
        return;
    const CUCOMPLEX x = fields[pixel];
    const CUCOMPLEX y = fields[pixel + plane_size];
    const CUCOMPLEX z = fields[pixel + 2u * plane_size];
    const size_t output = pixel + plane_size * plane_index;
    intensity[output] += x.x * x.x + x.y * x.y +
        y.x * y.x + y.y * y.y + z.x * z.x + z.y * z.y;
}

//-----------------------------------------------------------
__global__ void confocal_fluorescence_square_kernel(
    CU_REAL* intensity,
    size_t count)
{
    const size_t index = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (index < count)
        intensity[index] *= intensity[index];
}

//-----------------------------------------------------------
__global__ void microscope_intensity_sqrt_kernel(
    CU_REAL* intensity,
    size_t count)
{
    const size_t index = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (index < count)
        intensity[index] = sqrt(intensity[index]);
}

//-----------------------------------------------------------
__global__ void confocal_fluorescence_extract_plane_kernel(
    const CU_REAL* intensity,
    CUCOMPLEX* output,
    size_t plane_size,
    size_t plane_index)
{
    const size_t pixel = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (pixel < plane_size)
    {
        output[pixel].x = intensity[pixel + plane_size * plane_index];
        output[pixel].y = CU_REAL(0);
    }
}

//-----------------------------------------------------------
__global__ void confocal_fluorescence_phase_kernel(
    CUCOMPLEX* fields, int fft_size, CU_REAL aretecube,
    CU_REAL kxinc, CU_REAL kyinc, CU_REAL kzinc, CU_REAL lens_z)
{
    const size_t pixel = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    const size_t plane_size = size_t(fft_size) * size_t(fft_size);
    if (pixel >= plane_size)
        return;
    const int half = fft_size / 2;
    const CU_REAL x = aretecube * CU_REAL(int(pixel % size_t(fft_size)) - half);
    const CU_REAL y = aretecube * CU_REAL(int(pixel / size_t(fft_size)) - half);
    const CU_REAL phase = kxinc * x + kyinc * y - kzinc * lens_z;
    const CU_REAL cosine = cos(phase);
    const CU_REAL sine = sin(phase);
    for (int component = 0; component < 3; ++component)
    {
        CUCOMPLEX& value = fields[size_t(component) * plane_size + pixel];
        const CU_REAL real_part = value.x * cosine - value.y * sine;
        value.y = value.x * sine + value.y * cosine;
        value.x = real_part;
    }
}

//-----------------------------------------------------------
__global__ void confocal_fluorescence_rotate_kernel(
    CUCOMPLEX* fields,
    int fft_size,
    int max_index,
    CU_REAL delta_kx,
    CU_REAL delta_ky,
    CU_REAL k0,
    CU_REAL gross,
    int side,
    CU_REAL aperture)
{
    const size_t index = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    const size_t plane_size = size_t(fft_size) * size_t(fft_size);
    if (index >= plane_size)
        return;
    const int grid_x = int(index % size_t(fft_size));
    const int grid_y = int(index / size_t(fft_size));
    const int i = grid_x <= fft_size / 2 ? grid_x : grid_x - fft_size;
    const int j = grid_y <= fft_size / 2 ? grid_y : grid_y - fft_size;
    if (i < -max_index || i > max_index || j < -max_index || j > max_index)
        return;
    const CU_REAL kx = delta_kx * CU_REAL(i);
    const CU_REAL ky = delta_ky * CU_REAL(j);
    if (kx * kx + ky * ky > aperture * aperture ||
        (kx == CU_REAL(0) && ky == CU_REAL(0)))
        return;
    const CU_REAL ux = kx / k0;
    const CU_REAL uy = ky / k0;
    const CU_REAL uz_squared = CU_REAL(1) - ux * ux - uy * uy;
    const CU_REAL vx = -kx / k0 / gross;
    const CU_REAL vy = -ky / k0 / gross;
    const CU_REAL vz_squared = CU_REAL(1) - vx * vx - vy * vy;
    if (uz_squared <= CU_REAL(0) || vz_squared <= CU_REAL(0))
        return;
    const CU_REAL uz = sqrt(uz_squared) * CU_REAL(side);
    const CU_REAL vz = sqrt(vz_squared) * CU_REAL(side);
    if (uz / vz <= CU_REAL(0))
        return;
    CU_REAL a = uy * vz - uz * vy;
    CU_REAL b = -ux * vz + uz * vx;
    const CU_REAL cost = ux * vx + uy * vy + uz * vz;
    const CU_REAL sint = sqrt(a * a + b * b);
    if (sint == CU_REAL(0) || vz == CU_REAL(0))
        return;
    a /= sint;
    b /= sint;
    const CU_REAL scale = sqrt(uz / vz);
    const CUCOMPLEX x = fields[index];
    const CUCOMPLEX y = fields[index + plane_size];
    const CUCOMPLEX z = fields[index + 2u * plane_size];
    const CU_REAL xx = (a * a + (CU_REAL(1) - a * a) * cost) * x.x +
        a * b * (CU_REAL(1) - cost) * y.x + b * sint * z.x;
    const CU_REAL xy = (a * a + (CU_REAL(1) - a * a) * cost) * x.y +
        a * b * (CU_REAL(1) - cost) * y.y + b * sint * z.y;
    const CU_REAL yx = a * b * (CU_REAL(1) - cost) * x.x +
        (b * b + (CU_REAL(1) - b * b) * cost) * y.x - a * sint * z.x;
    const CU_REAL yy = a * b * (CU_REAL(1) - cost) * x.y +
        (b * b + (CU_REAL(1) - b * b) * cost) * y.y - a * sint * z.y;
    const CU_REAL zx = -b * sint * x.x + a * sint * y.x + cost * z.x;
    const CU_REAL zy = -b * sint * x.y + a * sint * y.y + cost * z.y;
    fields[index] = CUFFT_REAL(0);
    fields[index].x = scale * xx;
    fields[index].y = scale * xy;
    fields[index + plane_size] = CUFFT_REAL(0);
    fields[index + plane_size].x = scale * yx;
    fields[index + plane_size].y = scale * yy;
    fields[index + 2u * plane_size] = CUFFT_REAL(0);
    fields[index + 2u * plane_size].x = scale * zx;
    fields[index + 2u * plane_size].y = scale * zy;
}

//-----------------------------------------------------------
__global__ void confocal_fluorescence_build_kernel(
    const CUCOMPLEX* diffraction,
    CUCOMPLEX* spectra,
    int fft_size,
    int max_index,
    CU_REAL delta_kx,
    CU_REAL delta_ky,
    CU_REAL k0,
    CU_REAL aperture,
    CU_REAL lens_z,
    int side)
{
    const size_t pixel = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    const size_t plane_size = size_t(fft_size) * size_t(fft_size);
    if (pixel >= plane_size)
        return;
    const size_t output = pixel;
    const size_t component_stride = plane_size;
    for (int component = 0; component < 6; ++component)
        spectra[size_t(component) * component_stride + output] = CUFFT_ZERO;
    const int grid_x = int(pixel % size_t(fft_size));
    const int grid_y = int(pixel / size_t(fft_size));
    const int i = grid_x <= fft_size / 2 ? grid_x : grid_x - fft_size;
    const int j = grid_y <= fft_size / 2 ? grid_y : grid_y - fft_size;
    if (i < -max_index || i > max_index || j < -max_index || j > max_index)
        return;
    const CU_REAL kx = delta_kx * CU_REAL(i);
    const CU_REAL ky = delta_ky * CU_REAL(j);
    const CU_REAL transverse = kx * kx + ky * ky;
    if (transverse > aperture * aperture)
        return;
    const CU_REAL kz_squared = k0 * k0 - transverse;
    if (kz_squared < CU_REAL(0))
        return;
    const CU_REAL phase = sqrt(kz_squared) * CU_REAL(side) * lens_z;
    const CU_REAL cosine = cos(phase);
    const CU_REAL sine = sin(phase);
    const size_t source = size_t(max_index + i) +
        size_t(fft_size) * size_t(max_index + j);
    if (source >= plane_size)
        return;
    for (int component = 0; component < 3; ++component)
    {
        const CUCOMPLEX value = diffraction[size_t(component) * plane_size + source];
        CUCOMPLEX rotated = CUFFT_ZERO;
        rotated.x = value.x * cosine - value.y * sine;
        rotated.y = value.x * sine + value.y * cosine;
        spectra[size_t(component) * plane_size + output] = rotated;
        spectra[size_t(component + 3) * plane_size + output] = rotated;
    }
}

//-----------------------------------------------------------
__global__ void microscope_expand_packed_field_kernel(
    const CUCOMPLEX* packed_field, const CU_INT* dipole_map,
    CUCOMPLEX* dense_interleaved_field, int cell_count, int dipole_count) {
    const int cell = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= cell_count)
        return;
    const int packed_index = dipole_map[cell].dat - 1;
    const size_t dense_base = size_t(cell) * size_t(3);
    if (packed_index < 0 || packed_index >= dipole_count) {
        dense_interleaved_field[dense_base] = CUFFT_ZERO;
        dense_interleaved_field[dense_base + 1] = CUFFT_ZERO;
        dense_interleaved_field[dense_base + 2] = CUFFT_ZERO;
        return;
    }
    const size_t packed_base = size_t(packed_index) * size_t(3);
    dense_interleaved_field[dense_base] = packed_field[packed_base];
    dense_interleaved_field[dense_base + 1] = packed_field[packed_base + 1];
    dense_interleaved_field[dense_base + 2] = packed_field[packed_base + 2];
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
microscope_expand_packed_field_wrapper(
    const CUCOMPLEX* packed_field, const CU_INT* dipole_map,
    CUCOMPLEX* dense_interleaved_field, int cell_count, int dipole_count) {
    if (!packed_field || !dipole_map || !dense_interleaved_field ||
        cell_count <= 0 || dipole_count <= 0)
        return cudaErrorInvalidValue;
    microscope_expand_packed_field_kernel<<<
        block_count(size_t(cell_count)), THREADS_PER_BLOCK>>>(
        packed_field, dipole_map, dense_interleaved_field,
        cell_count, dipole_count);
    return finish_kernel_launch(258, "microscope_expand_packed_field_kernel");
}

//-----------------------------------------------------------
__global__ void confocal_direct_diffraction_kernel(
    const CUCOMPLEX* dipole_field,
    const CU_REAL* xs,
    const CU_REAL* ys,
    const CU_REAL* zs,
    int dipole_count,
    int nfft,
    int max_index,
    int side,
    CU_REAL k0,
    CU_REAL aperture,
    CU_REAL delta_kx,
    CU_REAL delta_ky,
    CUCOMPLEX* diffraction)
{
    const int width = 2 * max_index + 1;
    const int index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (index >= width * width)
        return;

    const int i = index % width - max_index;
    const int j = index / width - max_index;
    const CU_REAL kx = CU_REAL(i) * delta_kx;
    const CU_REAL ky = CU_REAL(j) * delta_ky;
    const CU_REAL transverse = kx * kx + ky * ky;
    if (transverse > aperture * aperture)
        return;

    const CU_REAL kz_squared = k0 * k0 - transverse;
    if (kz_squared <= CU_REAL(0))
        return;

    const CU_REAL kz = sqrt(kz_squared);
    const CU_REAL nx = kx / k0;
    const CU_REAL ny = ky / k0;
    const CU_REAL nz = sqrt(kz_squared) / k0 * CU_REAL(side);
    CUCOMPLEX ex = CUFFT_ZERO;
    CUCOMPLEX ey = CUFFT_ZERO;
    CUCOMPLEX ez = CUFFT_ZERO;
    for (int dipole = 0; dipole < dipole_count; ++dipole)
    {
        const int base = 3 * dipole;
        const CUCOMPLEX fx = dipole_field[base];
        const CUCOMPLEX fy = dipole_field[base + 1];
        const CUCOMPLEX fz = dipole_field[base + 2];
        const CUCOMPLEX projection = CUFFT_ADD(
            CUFFT_ADD(CUFFT_MUL(fx, CUFFT_REAL(nx)),
                      CUFFT_MUL(fy, CUFFT_REAL(ny))),
            CUFFT_MUL(fz, CUFFT_REAL(nz)));
        const CU_REAL phase = -k0 *
            (nx * xs[dipole] + ny * ys[dipole] + nz * zs[dipole]);
        const CU_REAL cosine = cos(phase);
        const CU_REAL sine = sin(phase);
        CUCOMPLEX phase_factor = CUFFT_ZERO;
        phase_factor.x = cosine;
        phase_factor.y = sine;
        ex = CUFFT_ADD(ex, CUFFT_MUL(phase_factor,
            CUFFT_SUB(fx, CUFFT_MUL(projection, CUFFT_REAL(nx)))));
        ey = CUFFT_ADD(ey, CUFFT_MUL(phase_factor,
            CUFFT_SUB(fy, CUFFT_MUL(projection, CUFFT_REAL(ny)))));
        ez = CUFFT_ADD(ez, CUFFT_MUL(phase_factor,
            CUFFT_SUB(fz, CUFFT_MUL(projection, CUFFT_REAL(nz)))));
    }

    const CU_REAL scale = k0 * k0 / (CU_REAL(2) *
        CU_REAL(3.1415926535897932384626433832795) * kz);
    const size_t plane = size_t(nfft) * size_t(nfft);
    const size_t output = size_t(max_index + i) +
        size_t(nfft) * size_t(max_index + j);
    CUCOMPLEX* result = diffraction + output;
    result[0] = CUFFT_ZERO;
    result[plane] = CUFFT_ZERO;
    result[2u * plane] = CUFFT_ZERO;
    result[0].x = -scale * ex.y;
    result[0].y = scale * ex.x;
    result[plane].x = -scale * ey.y;
    result[plane].y = scale * ey.x;
    result[2u * plane].x = -scale * ez.y;
    result[2u * plane].y = scale * ez.x;
}
} // namespace

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t quick_energy_incident_wave_wrapper(
    CUCOMPLEX *incident, int nfft, CU_REAL spacing, CU_REAL scale, CU_REAL kx, CU_REAL ky,
    CUCOMPLEX kz, CUCOMPLEX ex0, CUCOMPLEX ey0, CUCOMPLEX ez0) {
    if (!incident || nfft <= 0 || spacing <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t count = size_t(nfft) * size_t(nfft);
    quick_energy_incident_wave_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        incident, nfft, spacing, scale, kx, ky, kz, ex0, ey0, ez0);
    return finish_kernel_launch(219, "quick_energy_incident_wave_kernel");
}

extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
quick_energy_finalize_fields_wrapper(CUCOMPLEX *positive, CUCOMPLEX *negative, int nfft, int imax,
                                     CU_REAL dkx, CU_REAL dky, CU_REAL k0) {
    if (!positive || !negative || nfft <= 0 || imax < 0 || dkx <= CU_REAL(0) || dky <= CU_REAL(0) ||
        k0 <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t count = size_t(2 * imax + 1) * size_t(2 * imax + 1);
    quick_energy_scale_fields_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        positive, negative, nfft, imax, dkx, dky, k0);
    return finish_kernel_launch(220, "quick_energy_scale_fields_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t quick_energy_flux_wrapper(
    const CUCOMPLEX *incident, const CUCOMPLEX *positive, const CUCOMPLEX *negative, int nfft,
    int imax, CU_REAL dkx, CU_REAL dky, CU_REAL k0, CU_REAL *flux_incident, CU_REAL *flux_reflected,
    CU_REAL *flux_transmitted, CU_REAL *energy_inside_na, CU_REAL *energy_total, CU_REAL *ones) {
    if (!incident || !positive || !negative || nfft <= 0 || imax < 0 || dkx <= CU_REAL(0) ||
        dky <= CU_REAL(0) || k0 <= CU_REAL(0) || !flux_incident || !flux_reflected ||
        !flux_transmitted || !energy_inside_na || !energy_total || !ones)
        return cudaErrorInvalidValue;
    const size_t count = size_t(nfft) * size_t(nfft);
    quick_energy_flux_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        incident, positive, negative, nfft, imax, dkx, dky, k0, flux_incident, flux_reflected,
        flux_transmitted, energy_inside_na, energy_total, ones);
    return finish_kernel_launch(221, "quick_energy_flux_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t confocal_energy_spectrum_wrapper(
    CUCOMPLEX* incident, CUCOMPLEX* positive, CUCOMPLEX* negative,
    const CU_REAL* phase_positive, const CU_REAL* phase_negative,
    int nfft, int imax, CU_REAL k0, CU_REAL deltakx, CU_REAL deltaky,
    CU_REAL kmax_squared, CU_REAL psi, CU_REAL x0, CU_REAL y0, CU_REAL z0,
    CUCOMPLEX e0, int nside, int beam_kind) {
    if (!incident || !positive || !negative || !phase_positive || !phase_negative ||
        nfft < 8 || imax < 0 || 2 * imax + 1 > nfft ||
        (nside != 1 && nside != -1) || beam_kind < 0 || beam_kind > 1)
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft) * size_t(nfft);
    const size_t bytes = 3 * plane * sizeof(CUCOMPLEX);
    cudaError_t status = cudaMemset(incident, 0, bytes);
    if (status == cudaSuccess)
        status = cudaMemset(positive, 0, bytes);
    if (status == cudaSuccess)
        status = cudaMemset(negative, 0, bytes);
    if (status != cudaSuccess)
        return status;
    const int count = (2 * imax + 1) * (2 * imax + 1);
    confocal_energy_spectrum_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        incident, positive, negative, phase_positive, phase_negative,
        nfft, imax, k0, deltakx, deltaky, kmax_squared, psi,
        x0, y0, z0, e0, nside, beam_kind);
    return finish_kernel_launch(222, "confocal_energy_spectrum_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t confocal_energy_flux_wrapper(
    const CUCOMPLEX* incident_positive, const CUCOMPLEX* incident_negative,
    const CUCOMPLEX* scattered_positive, const CUCOMPLEX* scattered_negative,
    int nfft, int imax, CU_REAL deltakx, CU_REAL deltaky, CU_REAL k0,
    CU_REAL kmax_squared, CU_REAL* flux_incident, CU_REAL* flux_reflected,
    CU_REAL* flux_transmitted) {
    if (!incident_positive || !incident_negative || !scattered_positive ||
        !scattered_negative || !flux_incident || !flux_reflected || !flux_transmitted ||
        nfft < 8 || imax < 0 || 2 * imax + 1 > nfft)
        return cudaErrorInvalidValue;
    const int count = nfft * nfft;
    confocal_energy_flux_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(
        incident_positive, incident_negative, scattered_positive, scattered_negative,
        nfft, imax, deltakx, deltaky, k0, kmax_squared,
        flux_incident, flux_reflected, flux_transmitted);
    return finish_kernel_launch(223, "confocal_energy_flux_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_fluorescence_zero_wrapper(CU_REAL* intensity, size_t count)
{
    if (!intensity || count == 0)
        return cudaErrorInvalidValue;
    confocal_fluorescence_zero_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        intensity,
        count);
    return finish_kernel_launch(231, "confocal_fluorescence_zero_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_fluorescence_accumulate_wrapper(
    const CUCOMPLEX* field_planes,
    CU_REAL* intensity,
    size_t plane_size,
    size_t plane_index)
{
    if (!field_planes || !intensity || plane_size == 0 || plane_index == SIZE_MAX ||
        plane_index + 1u > SIZE_MAX / plane_size)
        return cudaErrorInvalidValue;
    confocal_fluorescence_accumulate_kernel<<<block_count(plane_size), THREADS_PER_BLOCK>>>(
        field_planes,
        intensity,
        plane_size,
        plane_index);
    return finish_kernel_launch(232, "confocal_fluorescence_accumulate_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_fluorescence_square_wrapper(CU_REAL* intensity, size_t count)
{
    if (!intensity || count == 0)
        return cudaErrorInvalidValue;
    confocal_fluorescence_square_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        intensity,
        count);
    return finish_kernel_launch(233, "confocal_fluorescence_square_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
microscope_intensity_sqrt_wrapper(CU_REAL* intensity, size_t count)
{
    if (!intensity || count == 0)
        return cudaErrorInvalidValue;
    microscope_intensity_sqrt_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        intensity,
        count);
    return finish_kernel_launch(240, "microscope_intensity_sqrt_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_fluorescence_extract_plane_wrapper(
    const CU_REAL* intensity,
    CUCOMPLEX* output,
    size_t plane_size,
    size_t plane_index)
{
    if (!intensity || !output || plane_size == 0 || plane_index == SIZE_MAX ||
        plane_index + 1u > SIZE_MAX / plane_size)
        return cudaErrorInvalidValue;
    confocal_fluorescence_extract_plane_kernel<<<block_count(plane_size), THREADS_PER_BLOCK>>>(
        intensity,
        output,
        plane_size,
        plane_index);
    return finish_kernel_launch(234, "confocal_fluorescence_extract_plane_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_fluorescence_phase_wrapper(
    CUCOMPLEX* field_planes,
    int fft_size,
    CU_REAL aretecube,
    CU_REAL kxinc,
    CU_REAL kyinc,
    CU_REAL kzinc,
    CU_REAL lens_z)
{
    if (!field_planes || fft_size <= 0 || aretecube <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t plane_size = size_t(fft_size) * size_t(fft_size);
    confocal_fluorescence_phase_kernel<<<block_count(plane_size), THREADS_PER_BLOCK>>>(
        field_planes, fft_size, aretecube, kxinc, kyinc, kzinc, lens_z);
    return finish_kernel_launch(238, "confocal_fluorescence_phase_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_fluorescence_rotate_wrapper(
    CUCOMPLEX* field_planes,
    int fft_size,
    int max_index,
    CU_REAL delta_kx,
    CU_REAL delta_ky,
    CU_REAL k0,
    CU_REAL gross,
    int side,
    CU_REAL aperture)
{
    if (!field_planes || fft_size <= 0 || max_index < 0 ||
        max_index >= fft_size / 2 || delta_kx == CU_REAL(0) ||
        delta_ky == CU_REAL(0) || k0 <= CU_REAL(0) || gross == CU_REAL(0) ||
        (side != 1 && side != -1) || aperture < CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t plane_size = size_t(fft_size) * size_t(fft_size);
    confocal_fluorescence_rotate_kernel<<<block_count(plane_size), THREADS_PER_BLOCK>>>(
        field_planes,
        fft_size,
        max_index,
        delta_kx,
        delta_ky,
        k0,
        gross,
        side,
        aperture);
    return finish_kernel_launch(235, "confocal_fluorescence_rotate_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_fluorescence_build_wrapper(
    const CUCOMPLEX* diffraction,
    CUCOMPLEX* spectrum_planes,
    int fft_size,
    int max_index,
    CU_REAL delta_kx,
    CU_REAL delta_ky,
    CU_REAL k0,
    CU_REAL aperture,
    CU_REAL lens_z,
    int side)
{
    if (!diffraction || !spectrum_planes || fft_size <= 0 || max_index < 0 ||
        max_index >= fft_size / 2 || delta_kx == CU_REAL(0) ||
        delta_ky == CU_REAL(0) || k0 <= CU_REAL(0) ||
        aperture < CU_REAL(0) || (side != 1 && side != -1))
        return cudaErrorInvalidValue;
    const size_t plane_size = size_t(fft_size) * size_t(fft_size);
    confocal_fluorescence_build_kernel<<<block_count(plane_size), THREADS_PER_BLOCK>>>(
        diffraction,
        spectrum_planes,
        fft_size,
        max_index,
        delta_kx,
        delta_ky,
        k0,
        aperture,
        lens_z,
        side);
    return finish_kernel_launch(236, "confocal_fluorescence_build_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
confocal_direct_diffraction_wrapper(
    const CUCOMPLEX* dipole_field,
    const CU_REAL* xs,
    const CU_REAL* ys,
    const CU_REAL* zs,
    int dipole_count,
    int nfft,
    int max_index,
    int side,
    CU_REAL k0,
    CU_REAL aperture,
    CU_REAL delta_kx,
    CU_REAL delta_ky,
    CUCOMPLEX* diffraction)
{
    if (!dipole_field || !xs || !ys || !zs || !diffraction || dipole_count <= 0 ||
        nfft <= 0 || nfft > 16384 || max_index < 0 ||
        2LL * max_index + 1 > nfft || (side != 1 && side != -1) ||
        k0 <= CU_REAL(0) || aperture <= CU_REAL(0) ||
        delta_kx <= CU_REAL(0) || delta_ky <= CU_REAL(0))
        return cudaErrorInvalidValue;

    const int width = 2 * max_index + 1;
    const size_t plane = size_t(nfft) * size_t(nfft);
    cudaError_t status = cudaMemset(
        diffraction, 0, 3u * plane * sizeof(CUCOMPLEX));
    if (status != cudaSuccess)
        return status;

    confocal_direct_diffraction_kernel<<<
        block_count(size_t(width) * size_t(width)), THREADS_PER_BLOCK>>>(
        dipole_field, xs, ys, zs, dipole_count, nfft, max_index, side,
        k0, aperture, delta_kx, delta_ky, diffraction);
    return finish_kernel_launch(237, "confocal_direct_diffraction_kernel");
}

//------------------------------------------
// Arbitrary incident field interpolation
//------------------------------------------
__device__ static CUCOMPLEX arbitrary_scale(CUCOMPLEX value, CU_REAL scale) {
    return CUFFT_MUL(value, CUFFT_REAL(scale));
}

//-----------------------------------------------------------
__device__ static CUCOMPLEX arbitrary_lerp(CUCOMPLEX a, CUCOMPLEX b, CU_REAL t) {
    return CUFFT_ADD(arbitrary_scale(a, CU_REAL(1) - t), arbitrary_scale(b, t));
}

//-----------------------------------------------------------
__global__ static void arbitrary_illumination_interpolate_kernel(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, int dipole_count,
    const CUCOMPLEX *grid_x, const CUCOMPLEX *grid_y, const CUCOMPLEX *grid_z, int nx, int ny,
    int nz, CU_REAL xmin, CU_REAL ymin, CU_REAL zmin, CU_REAL dx, CU_REAL dy, CU_REAL dz,
    CUCOMPLEX *output) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= dipole_count)
        return;

    const CU_REAL rx = (xs[i] - xmin) / dx;
    const CU_REAL ry = (ys[i] - ymin) / dy;
    const CU_REAL rz = (zs[i] - zmin) / dz;
    const int ix = int(rx), iy = int(ry), iz = int(rz);
    const CU_REAL xd = rx - CU_REAL(ix);
    const CU_REAL yd = ry - CU_REAL(iy);
    const CU_REAL zd = rz - CU_REAL(iz);
    // Keep the legacy Fortran threshold exactly: the fractions are compared
    // with the physical mesh steps divided by 1000.
    const bool snap_x = xd <= dx / CU_REAL(1000);
    const bool snap_y = yd <= dy / CU_REAL(1000);
    const bool snap_z = zd <= dz / CU_REAL(1000);
    CUCOMPLEX vx[8], vy[8], vz[8];
    for (int corner = 0; corner < 8; ++corner) {
        const int xoff = ((corner & 1) && !snap_x) ? 1 : 0;
        const int yoff = ((corner & 2) && !snap_y) ? 1 : 0;
        const int zoff = ((corner & 4) && !snap_z) ? 1 : 0;
        const size_t grid_index = size_t(iz + zoff) * size_t(nx) * size_t(ny) +
                                  size_t(iy + yoff) * size_t(nx) + size_t(ix + xoff);
        vx[corner] = grid_x[grid_index];
        vy[corner] = grid_y[grid_index];
        vz[corner] = grid_z[grid_index];
    }

    CUCOMPLEX ex, ey, ez;
    if (snap_x && snap_y && snap_z) {
        ex = vx[0];
        ey = vy[0];
        ez = vz[0];
    } else if (snap_x && snap_y) {
        ex = arbitrary_lerp(vx[0], vx[4], zd);
        ey = arbitrary_lerp(vy[0], vy[4], zd);
        ez = arbitrary_lerp(vz[0], vz[4], zd);
    } else if (snap_x && snap_z) {
        ex = arbitrary_lerp(vx[0], vx[2], yd);
        ey = arbitrary_lerp(vy[0], vy[2], yd);
        ez = arbitrary_lerp(vz[0], vz[2], yd);
    } else if (snap_y && snap_z) {
        ex = arbitrary_lerp(vx[0], vx[1], xd);
        ey = arbitrary_lerp(vy[0], vy[1], xd);
        ez = arbitrary_lerp(vz[0], vz[1], xd);
    } else if (snap_x) {
        ex = arbitrary_lerp(arbitrary_lerp(vx[0], vx[2], yd), arbitrary_lerp(vx[4], vx[6], yd), zd);
        ey = arbitrary_lerp(arbitrary_lerp(vy[0], vy[2], yd), arbitrary_lerp(vy[4], vy[6], yd), zd);
        ez = arbitrary_lerp(arbitrary_lerp(vz[0], vz[2], yd), arbitrary_lerp(vz[4], vz[6], yd), zd);
    } else if (snap_y) {
        ex = arbitrary_lerp(arbitrary_lerp(vx[0], vx[1], xd), arbitrary_lerp(vx[4], vx[5], xd), zd);
        ey = arbitrary_lerp(arbitrary_lerp(vy[0], vy[1], xd), arbitrary_lerp(vy[4], vy[5], xd), zd);
        ez = arbitrary_lerp(arbitrary_lerp(vz[0], vz[1], xd), arbitrary_lerp(vz[4], vz[5], xd), zd);
    } else if (snap_z) {
        ex = arbitrary_lerp(arbitrary_lerp(vx[0], vx[1], xd), arbitrary_lerp(vx[2], vx[3], xd), yd);
        ey = arbitrary_lerp(arbitrary_lerp(vy[0], vy[1], xd), arbitrary_lerp(vy[2], vy[3], xd), yd);
        ez = arbitrary_lerp(arbitrary_lerp(vz[0], vz[1], xd), arbitrary_lerp(vz[2], vz[3], xd), yd);
    } else {
        ex = arbitrary_lerp(
            arbitrary_lerp(arbitrary_lerp(vx[0], vx[1], xd), arbitrary_lerp(vx[2], vx[3], xd), yd),
            arbitrary_lerp(arbitrary_lerp(vx[4], vx[5], xd), arbitrary_lerp(vx[6], vx[7], xd), yd),
            zd);
        ey = arbitrary_lerp(
            arbitrary_lerp(arbitrary_lerp(vy[0], vy[1], xd), arbitrary_lerp(vy[2], vy[3], xd), yd),
            arbitrary_lerp(arbitrary_lerp(vy[4], vy[5], xd), arbitrary_lerp(vy[6], vy[7], xd), yd),
            zd);
        ez = arbitrary_lerp(
            arbitrary_lerp(arbitrary_lerp(vz[0], vz[1], xd), arbitrary_lerp(vz[2], vz[3], xd), yd),
            arbitrary_lerp(arbitrary_lerp(vz[4], vz[5], xd), arbitrary_lerp(vz[6], vz[7], xd), yd),
            zd);
    }
    output[3 * size_t(i)] = ex;
    output[3 * size_t(i) + 1] = ey;
    output[3 * size_t(i) + 2] = ez;
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t arbitrary_illumination_interpolate_wrapper(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, int dipole_count,
    const CUCOMPLEX *grid_x, const CUCOMPLEX *grid_y, const CUCOMPLEX *grid_z, int nx, int ny,
    int nz, CU_REAL xmin, CU_REAL ymin, CU_REAL zmin, CU_REAL dx, CU_REAL dy, CU_REAL dz,
    CUCOMPLEX *output) {
    if (!xs || !ys || !zs || !grid_x || !grid_y || !grid_z || !output || dipole_count <= 0 ||
        nx <= 0 || ny <= 0 || nz <= 0 || dx <= CU_REAL(0) || dy <= CU_REAL(0) || dz <= CU_REAL(0))
        return cudaErrorInvalidValue;
    arbitrary_illumination_interpolate_kernel<<<block_count(size_t(dipole_count)),
                                                THREADS_PER_BLOCK>>>(
        xs, ys, zs, dipole_count, grid_x, grid_y, grid_z, nx, ny, nz, xmin, ymin, zmin, dx, dy, dz,
        output);
    return finish_kernel_launch(222, "arbitrary_illumination_interpolate_kernel");
}

//-----------------------------------------------------------
__global__ static void arbitrary_derivative_grid_pack_kernel(
    const CUCOMPLEX* interleaved, CUCOMPLEX* component_major, size_t site_count) {
    const size_t site = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (site >= site_count)
        return;
    component_major[site] = interleaved[3 * site];
    component_major[site_count + site] = interleaved[3 * site + 1];
    component_major[2 * site_count + site] = interleaved[3 * site + 2];
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t arbitrary_derivative_grid_pack_wrapper(
    const CUCOMPLEX* interleaved, CUCOMPLEX* component_major, size_t site_count) {
    if (!interleaved || !component_major || site_count == 0)
        return cudaErrorInvalidValue;
    arbitrary_derivative_grid_pack_kernel<<<block_count(site_count), THREADS_PER_BLOCK>>>(
        interleaved, component_major, site_count);
    return finish_kernel_launch(239, "arbitrary_derivative_grid_pack_kernel");
}

//-----------------------------------------------------------
__global__ static void speckle_embed_negative_kz_phase_kernel(
    const CUCOMPLEX* source_phases, CUCOMPLEX* dense_phases,
    int source_size, int nfft, int imax, size_t plane)
{
    const size_t index = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (index >= plane)
        return;
    const int ii = int(index % size_t(nfft));
    const int jj = int(index / size_t(nfft));
    const int i = ii - imax;
    const int j = jj - imax;
    if (i < -imax || i > imax || j < -imax || j > imax) {
        dense_phases[index] = CUFFT_ZERO;
        return;
    }
    const int source_x = (i % source_size + source_size) % source_size;
    const int source_y = (j % source_size + source_size) % source_size;
    dense_phases[index] = source_phases[size_t(source_x) +
        size_t(source_size) * size_t(source_y)];
}

//-----------------------------------------------------------
__global__ static void negative_kz_diffraction_spectrum_kernel(
    const CUCOMPLEX* random_phases, CUCOMPLEX* spectrum, int nfft, int imax,
    CU_REAL deltakx, CU_REAL deltakey, CU_REAL k0, CU_REAL numerical_aperture,
    CU_REAL polarization_angle, CU_REAL x0, CU_REAL y0, CU_REAL z0,
    CUCOMPLEX incident_amplitude, int illumination_kind, size_t plane) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= plane)
        return;
    spectrum[index] = CUFFT_ZERO;
    spectrum[plane + index] = CUFFT_ZERO;
    spectrum[2 * plane + index] = CUFFT_ZERO;
    if (illumination_kind == 0)
        return;

    const int ii = int(index % size_t(nfft));
    const int jj = int(index / size_t(nfft));
    const int i = ii - imax;
    const int j = jj - imax;
    if (i < -imax || i > imax || j < -imax || j > imax)
        return;

    const CU_REAL kx = CU_REAL(i) * deltakx;
    const CU_REAL ky = CU_REAL(j) * deltakey;
    const CU_REAL kp2 = kx * kx + ky * ky;
    const CU_REAL kmax = k0 * numerical_aperture;
    if (kp2 > kmax * kmax * CU_REAL(0.999) || kp2 > k0 * k0)
        return;

    const CU_REAL kz = -sqrt(k0 * k0 - kp2);
    const CU_REAL psi = polarization_angle * CU_REAL(3.14159265358979323846 / 180.0);
    const CU_REAL cpsi = cos(psi);
    const CU_REAL spsi = sin(psi);
    const CU_REAL phase_angle = -(kx * x0 + ky * y0 + kz * z0);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX propagation_phase = make_cuFloatComplex(cosf(phase_angle), sinf(phase_angle));
#else
    const CUCOMPLEX propagation_phase = make_cuDoubleComplex(cos(phase_angle), sin(phase_angle));
#endif
    CUCOMPLEX phase = propagation_phase;
    if (illumination_kind == 1)
        phase = CUFFT_MUL(phase, random_phases[index]);

    const CU_REAL norm_x = sqrt(kz * kz + kx * kx);
    const CU_REAL norm_y = sqrt(kz * kz + ky * ky);
    if (norm_x == CU_REAL(0) || norm_y == CU_REAL(0))
        return;
    const CU_REAL axp = kz / norm_x;
    const CU_REAL azp = -kx / norm_x;
    const CU_REAL ays = kz / norm_y;
    const CU_REAL azs = -ky / norm_y;
    const CUCOMPLEX amp_x = CUFFT_MUL(incident_amplitude, CUFFT_REAL(-cpsi * axp));
    const CUCOMPLEX amp_y = CUFFT_MUL(incident_amplitude, CUFFT_REAL(-spsi * ays));
    const CUCOMPLEX amp_z = CUFFT_MUL(incident_amplitude, CUFFT_REAL(-(cpsi * azp + spsi * azs)));
    spectrum[index] = CUFFT_MUL(amp_x, phase);
    spectrum[plane + index] = CUFFT_MUL(amp_y, phase);
    spectrum[2 * plane + index] = CUFFT_MUL(amp_z, phase);
}

//---------------------------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t negative_kz_diffraction_spectrum_wrapper(
    const CUCOMPLEX* random_phases, CUCOMPLEX* spectrum, int nfft, int imax,
    CU_REAL deltakx, CU_REAL deltakey, CU_REAL k0, CU_REAL numerical_aperture,
    CU_REAL polarization_angle, CU_REAL x0, CU_REAL y0, CU_REAL z0,
    CUCOMPLEX incident_amplitude, int illumination_kind) {
    if (!random_phases || !spectrum || nfft <= 0 || imax < 0 ||
        2 * imax + 1 > nfft || deltakx <= CU_REAL(0) || deltakey <= CU_REAL(0) ||
        k0 <= CU_REAL(0) || numerical_aperture < CU_REAL(0) ||
        illumination_kind < 0 || illumination_kind > 2)
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft) * size_t(nfft);
    negative_kz_diffraction_spectrum_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        random_phases, spectrum, nfft, imax, deltakx, deltakey, k0, numerical_aperture,
        polarization_angle, x0, y0, z0, incident_amplitude, illumination_kind, plane);
    return finish_kernel_launch(224, "negative_kz_diffraction_spectrum_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t
speckle_embed_negative_kz_phase_wrapper(
    const CUCOMPLEX* source_phases, CUCOMPLEX* dense_phases,
    int source_size, int nfft, int imax)
{
    if (!source_phases || !dense_phases || source_size <= 0 || nfft <= 0 ||
        imax < 0 || 2 * imax + 1 > nfft)
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft) * size_t(nfft);
    speckle_embed_negative_kz_phase_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        source_phases, dense_phases, source_size, nfft, imax, plane);
    return finish_kernel_launch(242, "speckle_embed_negative_kz_phase_kernel");
}

//-----------------------------------------------------------
__global__ static void lens_diffraction_clear_kernel(CUCOMPLEX* spectrum, int count) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
        spectrum[index] = CUFFT_ZERO;
}

//-----------------------------------------------------------
__global__ static void lens_diffraction_pack_layer_kernel(
    const CUCOMPLEX* green_field, CUCOMPLEX* layer_spectrum, const CU_INT* fft_indices,
    int nx, int ny, int nfft, int layer) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = nfft * nfft;
    if (index >= plane)
        return;
    // The caller clears the complete spectrum in a separate kernel launch.
    // Clearing here races against scatter writes by other GPU threads.
    const int i = index % nfft;
    const int j = index / nfft;
    if (i >= nx || j >= ny)
        return;
    const size_t source = size_t(i) + size_t(nx) * size_t(j) +
                          size_t(nx) * size_t(ny) * size_t(layer);
    const size_t destination = size_t(fft_indices[i].dat) +
                               size_t(nfft) * size_t(fft_indices[j].dat);
    layer_spectrum[destination] = green_field[3 * source];
    layer_spectrum[plane + destination] = green_field[3 * source + 1];
    layer_spectrum[2 * plane + destination] = green_field[3 * source + 2];
}

//-----------------------------------------------------------
__global__ static void lens_diffraction_accumulate_kernel(
    const CUCOMPLEX* layer_spectrum, CUCOMPLEX* output, const CU_INT* fft_indices,
    int nfft, int imax, CU_REAL deltakx, CU_REAL deltaky, CU_REAL k0,
    CU_REAL numerical_aperture, CU_REAL dside, CU_REAL var1, CU_REAL var2, CU_REAL z) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int width = 2 * imax + 1;
    const int count = width * width;
    if (index >= count)
        return;
    const int i = index % width - imax;
    const int j = index / width - imax;
    const CU_REAL kx = CU_REAL(i) * deltakx;
    const CU_REAL ky = CU_REAL(j) * deltaky;
    const CU_REAL kp2 = kx * kx + ky * ky;
    const CU_REAL k0sq = k0 * k0;
    if (kp2 >= numerical_aperture * numerical_aperture || kp2 >= k0sq)
        return;
    const CU_REAL kz = sqrt(k0sq - kp2);
    const int nfft_half = nfft / 2;
    const int source_x = fft_indices[i + nfft_half].dat;
    const int source_y = fft_indices[j + nfft_half].dat;
    const size_t source = size_t(source_x) + size_t(nfft) * size_t(source_y);
    const size_t plane = size_t(nfft) * size_t(nfft);
    const size_t destination = size_t(imax + i) + size_t(nfft) * size_t(imax + j);
    const CUCOMPLEX ex = layer_spectrum[source];
    const CUCOMPLEX ey = layer_spectrum[plane + source];
    const CUCOMPLEX ez = layer_spectrum[2 * plane + source];
    const CUCOMPLEX longitudinal = CUFFT_DIV(
        CUFFT_ADD(CUFFT_ADD(CUFFT_MUL(CUFFT_REAL(kx), ex),
                            CUFFT_MUL(CUFFT_REAL(ky), ey)),
                  CUFFT_MUL(CUFFT_REAL(kz * dside), ez)), CUFFT_REAL(k0));
    const CU_REAL phase = -(dside * kz * z + var1 * CU_REAL(i) + var2 * CU_REAL(j));
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX factor0 = make_cuFloatComplex(cosf(phase), sinf(phase));
    const CUCOMPLEX plus_i = make_cuFloatComplex(0.0f, 1.0f);
#else
    const CUCOMPLEX factor0 = make_cuDoubleComplex(cos(phase), sin(phase));
    const CUCOMPLEX plus_i = make_cuDoubleComplex(0.0, 1.0);
#endif
    const CUCOMPLEX factor = CUFFT_DIV(CUFFT_MUL(factor0, plus_i),
                                      CUFFT_REAL(CU_REAL(6.2831853071795864769) * kz));
    const CUCOMPLEX out_x = CUFFT_MUL(
        CUFFT_SUB(ex, CUFFT_DIV(CUFFT_MUL(CUFFT_REAL(kx), longitudinal), CUFFT_REAL(k0))), factor);
    const CUCOMPLEX out_y = CUFFT_MUL(
        CUFFT_SUB(ey, CUFFT_DIV(CUFFT_MUL(CUFFT_REAL(ky), longitudinal), CUFFT_REAL(k0))), factor);
    const CUCOMPLEX out_z = CUFFT_MUL(
        CUFFT_SUB(ez, CUFFT_DIV(CUFFT_MUL(CUFFT_REAL(kz * dside), longitudinal), CUFFT_REAL(k0))), factor);
    output[destination] = CUFFT_ADD(output[destination], CUFFT_MUL(CUFFT_REAL(k0sq), out_x));
    output[plane + destination] = CUFFT_ADD(output[plane + destination], CUFFT_MUL(CUFFT_REAL(k0sq), out_y));
    output[2 * plane + destination] = CUFFT_ADD(output[2 * plane + destination], CUFFT_MUL(CUFFT_REAL(k0sq), out_z));
}

//-----------------------------------------------------------
__global__ static void quicklens_direct_sum_kernel(
    const CUCOMPLEX* field, const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    const CUCOMPLEX* incident, CUCOMPLEX* scattered, CUCOMPLEX* combined,
    int dipole_count, int nfft, int half, int imax, CU_REAL dkx, CU_REAL k0,
    CU_REAL na, int side, CU_REAL zlens) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int width = 2 * imax + 1;
    if (index >= width * width) return;
    const int i = index % width - imax, j = index / width - imax;
    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);
    const CU_REAL kx = CU_REAL(i) * dkx, ky = CU_REAL(j) * dkx;
    const CU_REAL kp2 = kx*kx + ky*ky;
    if (kp2 > na*na || kp2 > k0*k0) return;
    const CU_REAL kz = sqrt(k0*k0-kp2);
    const CU_REAL ux=kx/k0, uy=ky/k0;
    const CU_REAL uz=CU_REAL(side)*sqrt(CU_REAL(1)-ux*ux-uy*uy);
    CU_REAL emxr=0, emxi=0, emyr=0, emyi=0, emzr=0, emzi=0;
    for (int d=0; d<dipole_count; ++d) {
        const int q=3*d;
        const CU_REAL angle=-k0*(ux*xs[d]+uy*ys[d]+uz*zs[d]);
        const CU_REAL cr=cos(angle), ci=sin(angle);
        const CUCOMPLEX fx=field[q], fy=field[q+1], fz=field[q+2];
        const CU_REAL dr=ux*fx.x+uy*fy.x+uz*fz.x;
        const CU_REAL di=ux*fx.y+uy*fy.y+uz*fz.y;
        const CU_REAL vxr=fx.x-ux*dr, vxi=fx.y-ux*di;
        const CU_REAL vyr=fy.x-uy*dr, vyi=fy.y-uy*di;
        const CU_REAL vzr=fz.x-uz*dr, vzi=fz.y-uz*di;
        emxr+=cr*vxr-ci*vxi; emxi+=cr*vxi+ci*vxr;
        emyr+=cr*vyr-ci*vyi; emyi+=cr*vyi+ci*vyr;
        emzr+=cr*vzr-ci*vzi; emzi+=cr*vzi+ci*vzr;
    }
    // scale = k0^2/(-2*pi*i*kz) * exp(i*kz*1e-9*zlens*side)
    const CU_REAL phase_angle=kz*CU_REAL(1.0e-9)*zlens*CU_REAL(side);
    const CU_REAL sr=-sin(phase_angle), si=cos(phase_angle);
    const CU_REAL scale=k0*k0/(CU_REAL(2)*pi*kz);
    const CU_REAL ar=scale*(sr*emxr-si*emxi), ai=scale*(sr*emxi+si*emxr);
    const CU_REAL br=scale*(sr*emyr-si*emyi), bi=scale*(sr*emyi+si*emyr);
    const CU_REAL cr=scale*(sr*emzr-si*emzi), ci=scale*(sr*emzi+si*emzr);
    const int plane=nfft*nfft;
    const int at=(i+half)+nfft*(j+half);
    const int incat=(i+imax)+nfft*(j+imax);
    if (at<0 || at>=plane) return;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    scattered[at]=make_cuFloatComplex(ar,ai);
    scattered[plane+at]=make_cuFloatComplex(br,bi);
    scattered[2*plane+at]=make_cuFloatComplex(cr,ci);
#else
    scattered[at]=make_cuDoubleComplex(ar,ai);
    scattered[plane+at]=make_cuDoubleComplex(br,bi);
    scattered[2*plane+at]=make_cuDoubleComplex(cr,ci);
#endif
    const CUCOMPLEX incx=incident[incat], incy=incident[plane+incat], incz=incident[2*plane+incat];
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX ph=make_cuFloatComplex(cos(phase_angle),sin(phase_angle));
    combined[at]=cuCaddf(scattered[at],cuCmulf(incx,ph));
    combined[plane+at]=cuCaddf(scattered[plane+at],cuCmulf(incy,ph));
    combined[2*plane+at]=cuCaddf(scattered[2*plane+at],cuCmulf(incz,ph));
#else
    const CUCOMPLEX ph=make_cuDoubleComplex(cos(phase_angle),sin(phase_angle));
    combined[at]=cuCadd(scattered[at],cuCmul(incx,ph));
    combined[plane+at]=cuCadd(scattered[plane+at],cuCmul(incy,ph));
    combined[2*plane+at]=cuCadd(scattered[2*plane+at],cuCmul(incz,ph));
#endif
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t lens_diffraction_clear_wrapper(
    CUCOMPLEX* spectrum, int count) {
    if (!spectrum || count <= 0)
        return cudaErrorInvalidValue;
    lens_diffraction_clear_kernel<<<block_count(size_t(count)), THREADS_PER_BLOCK>>>(spectrum, count);
    return finish_kernel_launch(225, "lens_diffraction_clear_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t lens_diffraction_pack_layer_wrapper(
    const CUCOMPLEX* green_field, CUCOMPLEX* layer_spectrum, const CU_INT* fft_indices,
    int nx, int ny, int nfft, int layer) {
    if (!green_field || !layer_spectrum || !fft_indices || nx <= 0 || ny <= 0 ||
        nx > nfft || ny > nfft || nfft <= 0 || layer < 0)
        return cudaErrorInvalidValue;
    const size_t plane = size_t(nfft) * size_t(nfft);
    lens_diffraction_pack_layer_kernel<<<block_count(plane), THREADS_PER_BLOCK>>>(
        green_field, layer_spectrum, fft_indices, nx, ny, nfft, layer);
    return finish_kernel_launch(226, "lens_diffraction_pack_layer_kernel");
}

//-----------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t lens_diffraction_accumulate_wrapper(
    const CUCOMPLEX* layer_spectrum, CUCOMPLEX* output, const CU_INT* fft_indices,
    int nfft, int imax, CU_REAL deltakx, CU_REAL deltaky, CU_REAL k0,
    CU_REAL numerical_aperture, CU_REAL dside, CU_REAL var1, CU_REAL var2, CU_REAL z) {
    if (!layer_spectrum || !output || !fft_indices || nfft <= 0 || imax < 0 ||
        2 * imax + 1 > nfft || deltakx <= CU_REAL(0) || deltaky <= CU_REAL(0) ||
        k0 <= CU_REAL(0) || numerical_aperture <= CU_REAL(0))
        return cudaErrorInvalidValue;
    const size_t count = size_t(2 * imax + 1) * size_t(2 * imax + 1);
    lens_diffraction_accumulate_kernel<<<block_count(count), THREADS_PER_BLOCK>>>(
        layer_spectrum, output, fft_indices, nfft, imax, deltakx, deltaky,
        k0, numerical_aperture, dside, var1, var2, z);
    return finish_kernel_launch(227, "lens_diffraction_accumulate_kernel");
}

//-------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t quicklens_direct_sum_wrapper(
    const CUCOMPLEX* field, const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs,
    const CUCOMPLEX* incident, CUCOMPLEX* scattered, CUCOMPLEX* combined,
    int dipole_count, int nfft, int half, int imax, CU_REAL dkx, CU_REAL k0,
    CU_REAL na, int side, CU_REAL zlens) {
    if (!field || !xs || !ys || !zs || !incident || !scattered || !combined ||
        dipole_count <= 0 || nfft <= 0 || imax < 0)
        return cudaErrorInvalidValue;
    const int count=(2*imax+1)*(2*imax+1);
    const int plane=nfft*nfft;
    lens_diffraction_clear_kernel<<<block_count(size_t(3*plane)),THREADS_PER_BLOCK>>>(scattered,3*plane);
    lens_diffraction_clear_kernel<<<block_count(size_t(3*plane)),THREADS_PER_BLOCK>>>(combined,3*plane);
    quicklens_direct_sum_kernel<<<block_count(size_t(count)),THREADS_PER_BLOCK>>>(
        field,xs,ys,zs,incident,scattered,combined,dipole_count,nfft,half,imax,
        dkx,k0,na,side,zlens);
    return finish_kernel_launch(251,"quicklens_direct_sum_kernel");
}

//---------------------------------------------------------------------------
extern "C" PRODUITMATVECTFFT_KERNEL_API cudaError_t mie_efficiencies_wrapper(
    CUCOMPLEX *recurrence, CU_REAL relative_index_real, CU_REAL relative_index_imag,
    CU_REAL x, CU_REAL radius, CU_REAL *outputs, int max_order) {
    if (!recurrence || !outputs ||
        (relative_index_real == CU_REAL(0) && relative_index_imag == CU_REAL(0)) ||
        x <= CU_REAL(0) || radius <= CU_REAL(0) ||
        max_order < 2)
        return cudaErrorInvalidValue;
    mie_efficiencies_kernel<<<1, 1>>>(recurrence, relative_index_real, relative_index_imag,
                                      x, radius, outputs, max_order);
    return finish_kernel_launch(223, "mie_efficiencies_kernel");
}
