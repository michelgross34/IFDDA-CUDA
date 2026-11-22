#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

namespace {

struct ChanPreconditionerDeviceState {
    cuda_data::ChanPreconditionerDeviceBuffers buffers{};
    int nx = 0, ny = 0, nz = 0, nxm = 0, nym = 0, nzm = 0;
    int matrix_order = 0, batch_count = 0;
    size_t field_count = 0, fft_count = 0, rhs_count = 0;
    size_t matrix_count = 0, pivot_count = 0;
    bool buffers_ready = false;
    bool matrices_ready = false;
    bool matrices_factorized = false;
};

ChanPreconditionerDeviceState state;

//-----------------------------------------------------------
cuda_data::ChanPreconditionerBufferSizes buffer_sizes_for_state() {
    cuda_data::ChanPreconditionerBufferSizes sizes{};
    sizes.vector_nx = state.nx;
    sizes.vector_ny = state.ny;
    sizes.vector_nz = state.nz;
    sizes.vector_field_count = state.field_count;
    sizes.vector_fft_count = state.fft_count;
    sizes.vector_rhs_count = state.rhs_count;
    sizes.vector_matrix_count = state.matrix_count;
    sizes.vector_map_count = size_t(state.nxm) * size_t(state.nym) * size_t(state.nzm);
    sizes.vector_pivot_count = state.pivot_count;
    sizes.vector_batch_count = size_t(state.batch_count);
    const size_t channels = size_t(6) * size_t(state.nx) * size_t(state.ny) * size_t(state.nz);
    sizes.vector_channel_count = channels;
    sizes.vector_green_count = channels;
    sizes.vector_chan_vectors_count = channels;
    return sizes;
}

//-----------------------------------------------------------
bool upload_chan_matrices(const CUCOMPLEX *source) {
    if (!source || !state.buffers.vector.matrices || state.batch_count <= 0 ||
        state.matrix_order <= 0 || state.nzm < state.nz)
        return false;

    // The Fortran sdetnn leading dimension is 3*nzm, whereas the resident
    // cuBLAS matrix has order 3*nz. Previously the padded source was
    // uploaded as a SECOND global GPU matrix and then packed with a kernel.
    // Pack bounded chunks on the CPU instead, and upload them directly into
    // the ONE resident LU array. The GPU-generated Chan path already writes
    // into this same array and does not execute this function.
    const size_t order = size_t(state.matrix_order);
    const size_t source_order = size_t(3) * size_t(state.nzm);
    if (order > std::numeric_limits<size_t>::max() / order ||
        source_order > std::numeric_limits<size_t>::max() / source_order)
        return false;
    const size_t packed_stride = order * order;
    const size_t source_stride = source_order * source_order;
    if (packed_stride > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return false;

    constexpr size_t staging_limit_bytes = size_t(8) * 1024 * 1024;
    const size_t matrices_per_chunk = std::max(
        size_t(1), staging_limit_bytes / (packed_stride * sizeof(CUCOMPLEX)));
    const size_t staging_batches =
        std::min(size_t(state.batch_count), matrices_per_chunk);
    std::vector<CUCOMPLEX> staging(staging_batches * packed_stride);

    for (size_t first = 0; first < size_t(state.batch_count); first += staging_batches) {
        const size_t batch_size =
            std::min(staging_batches, size_t(state.batch_count) - first);
        for (size_t batch = 0; batch < batch_size; ++batch) {
            const CUCOMPLEX *source_matrix =
                source + (first + batch) * source_stride;
            CUCOMPLEX *packed_matrix =
                staging.data() + batch * packed_stride;
            for (size_t column = 0; column < order; ++column)
                std::copy_n(source_matrix + column * source_order, order,
                            packed_matrix + column * order);
        }
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(
                    state.buffers.vector.matrices + first * packed_stride,
                    staging.data(), batch_size * packed_stride * sizeof(CUCOMPLEX),
                    cuda_utils::TransferCategory::Preconditioner),
                "Chan preconditioner", "upload compact Chan LU matrices"))
            return false;
    }
    return true;
}

//-----------------------------------------------------------
bool ensure_state(int nx, int ny, int nz, int nxm, int nym, int nzm) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || nxm < nx || nym < ny || nzm < nz)
        return false;
    const size_t plane = size_t(nx) * size_t(ny);
    const size_t batch = size_t(nx) * size_t(ny);
    const size_t order = size_t(3) * size_t(nz);
    const size_t map_count = size_t(nxm) * size_t(nym) * size_t(nzm);
    if (plane > size_t(std::numeric_limits<int>::max()) ||
        batch > size_t(std::numeric_limits<int>::max()) ||
        order > size_t(std::numeric_limits<int>::max()) ||
        map_count > size_t(std::numeric_limits<int>::max()) ||
        size_t(3) * plane * size_t(nz) > size_t(std::numeric_limits<int>::max()))
        return false;

    if (state.nx == nx && state.ny == ny && state.nz == nz && state.nxm == nxm &&
        state.nym == nym && state.nzm == nzm && state.buffers.vector.fft_plan != 0 &&
        state.buffers_ready)
        return true;

    state.matrices_ready = false;
    state.matrices_factorized = false;
    // The named buffers and dimension-keyed FFT plans persist across calls.
    state = {};

    state.nx = nx;
    state.ny = ny;
    state.nz = nz;
    state.nxm = nxm;
    state.nym = nym;
    state.nzm = nzm;
    state.matrix_order = int(order);
    state.batch_count = int(batch);
    state.field_count = size_t(3) * size_t(nxm) * size_t(nym) * size_t(nzm);
    state.fft_count = size_t(3) * size_t(nz) * plane;
    state.rhs_count = size_t(state.matrix_order) * batch;
    state.matrix_count = size_t(state.matrix_order) * size_t(state.matrix_order) * batch;
    state.pivot_count = size_t(state.matrix_order) * batch;

    const cuda_data::ChanPreconditionerBufferSizes buffer_sizes = buffer_sizes_for_state();
    if (!cuda_data::get_device_chan_preconditioner_buffers(buffer_sizes, &state.buffers))
        return false;
    state.buffers_ready = true;

    if (!cuda_utils::check_cuda(
            chan_preconditioner_identity_map_wrapper(state.buffers.vector.identity_map,
                                                     int(map_count)),
            "Chan preconditioner", "initialize dense identity map on GPU"))
        return false;
    return true;
}

//-----------------------------------------------------------
int apply_chan_preconditioner_cuda_(CUCOMPLEX *cpu_field, const int *cpu_dipole_map, const int *nx,
                                    const int *ny, const int *nz, const int *nxm, const int *nym,
                                    const int *nzm, bool dense_grid) {
    if (!cpu_field || (!dense_grid && !cpu_dipole_map) || !nx || !ny || !nz || !nxm || !nym ||
        !nzm || !ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm) || !state.matrices_ready ||
        !state.matrices_factorized)
        return 1;

    const size_t map_count = size_t(*nxm) * size_t(*nym) * size_t(*nzm);
    const size_t plane_count = size_t(*nx) * size_t(*ny);
    CU_INT *dipole_map = dense_grid ? state.buffers.vector.identity_map
                                 : get_device_tabdip(cpu_dipole_map, map_count);
    if (!dipole_map || !initialize_cuda_device())
        return 2;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(state.buffers.vector.field, cpu_field,
                                           state.field_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Preconditioner),
            "Chan preconditioner", "upload current product"))
        return 3;
    if (chan_preconditioner_pack_fft_wrapper(state.buffers.vector.field, dipole_map,
                                             state.buffers.vector.fft_planes, *nx, *ny, *nz, *nxm,
                                             *nym, *nzm, int(map_count)) != cudaSuccess)
        return 4;
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.vector.fft_plan, state.buffers.vector.fft_planes,
                                               state.buffers.vector.fft_planes, CUFFT_INVERSE),
                                 "Chan preconditioner", "backward 2-D FFT"))
        return 5;
    if (chan_preconditioner_fft_to_rhs_wrapper(state.buffers.vector.fft_planes,
                                               state.buffers.vector.rhs, *nx, *ny,
                                               *nz) != cudaSuccess)
        return 6;
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle)
        return 7;
    int solve_info = 0;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const cublasStatus_t solve_status = cublasCgetrsBatched(
        handle, CUBLAS_OP_N, state.matrix_order, 1, state.buffers.vector.matrix_ptrs,
        state.matrix_order, cuda_utils::device_integer_library_view(state.buffers.vector.pivots), state.buffers.vector.rhs_ptrs, state.matrix_order,
        &solve_info, state.batch_count);
#else
    const cublasStatus_t solve_status = cublasZgetrsBatched(
        handle, CUBLAS_OP_N, state.matrix_order, 1, state.buffers.vector.matrix_ptrs,
        state.matrix_order, cuda_utils::device_integer_library_view(state.buffers.vector.pivots), state.buffers.vector.rhs_ptrs,
        state.matrix_order, &solve_info, state.batch_count);
#endif
    if (!cuda_utils::check_cublas(solve_status, "Chan preconditioner", "batched getrs"))
        return 8;
    if (solve_info != 0) {
        std::fprintf(stderr, "Chan preconditioner: batched getrs info=%d\n", solve_info);
        return 9;
    }
    if (chan_preconditioner_rhs_to_fft_wrapper(state.buffers.vector.rhs,
                                               state.buffers.vector.fft_planes, *nx, *ny,
                                               *nz) != cudaSuccess)
        return 11;
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.vector.fft_plan, state.buffers.vector.fft_planes,
                                               state.buffers.vector.fft_planes, CUFFT_FORWARD),
                                 "Chan preconditioner", "forward 2-D FFT"))
        return 12;
    if (chan_preconditioner_unpack_fft_wrapper(
            state.buffers.vector.fft_planes, dipole_map, state.buffers.vector.field, *nx, *ny, *nz,
            *nxm, *nym, *nzm, int(map_count), CU_REAL(plane_count)) != cudaSuccess)
        return 13;
    // The CUDA inverse driver consumes FFprecon as its right-hand side. Keep
    // the result in the preconditioner buffer and give the resident solver a
    // temporary alias keyed by the corresponding Fortran array address.
    if (!register_gpbicg_device_b_alias(cpu_field, state.buffers.vector.field, state.field_count))
        return 14;
    return 0;
}

//-----------------------------------------------------------
int build_chan_blocks_cuda_(const CUCOMPLEX *chan_xx, const CUCOMPLEX *chan_xy,
                            const CUCOMPLEX *chan_xz, const CUCOMPLEX *chan_yy,
                            const CUCOMPLEX *chan_yz, const CUCOMPLEX *chan_zz, const int *nx,
                            const int *ny, const int *nz, const int *nxm, const int *nym,
                            const int *nzm) {
    if (!chan_xx || !chan_xy || !chan_xz || !chan_yy || !chan_yz || !chan_zz || !nx || !ny || !nz ||
        !nxm || !nym || !nzm || !ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm) ||
        !initialize_cuda_device())
        return 1;

    const size_t grid_count = size_t(*nx) * size_t(*ny) * size_t(*nz);
    std::vector<CUCOMPLEX> host_vectors(size_t(6) * grid_count);
    const CUCOMPLEX *source[6] = {chan_xx, chan_xy, chan_xz, chan_yy, chan_yz, chan_zz};
    for (size_t component = 0; component < 6; ++component)
        std::copy(source[component], source[component] + grid_count,
                  host_vectors.begin() + component * grid_count);

    if (host_vectors.size() != size_t(6) * grid_count ||
        !cuda_data::get_device_chan_preconditioner_buffers(buffer_sizes_for_state(),
                                                           &state.buffers))
        return 2;
    CUCOMPLEX *device_vectors = state.buffers.vector.channel_vectors;
    const bool uploaded = cuda_utils::check_cuda(
        cuda_data::copy_host_to_device(device_vectors, host_vectors.data(),
                                       host_vectors.size() * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Preconditioner),
        "Chan preconditioner", "upload channel vectors");
    if (!uploaded)
        return 3;
    const cudaError_t launch = chan_preconditioner_build_blocks_wrapper(
        device_vectors, state.buffers.vector.matrices, *nx, *ny, *nz);
    if (launch != cudaSuccess)
        return 4;
    const bool synchronized = cuda_utils::check_cuda(cudaDeviceSynchronize(), "Chan preconditioner",
                                                     "finish GPU block assembly");
    if (!synchronized)
        return 5;

    state.matrices_ready = true;
    state.matrices_factorized = false;
    std::fprintf(stderr, "[Chan preconditioner CUDA] assembled %d blocks of order %d on GPU\n",
                 state.batch_count, state.matrix_order);
    return 0;
}

//-----------------------------------------------------------
int factorize_chan_blocks_cuda_() {
    if (!state.matrices_ready || !initialize_cuda_device())
        return 1;
    const cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle)
        return 2;
    const size_t matrix_stride = size_t(state.matrix_order) * size_t(state.matrix_order);
    if (!cuda_utils::check_cuda(
            chan_preconditioner_init_batch_ptrs_wrapper(
                state.buffers.vector.matrices, state.buffers.vector.rhs,
                state.buffers.vector.matrix_ptrs, state.buffers.vector.rhs_ptrs,
                matrix_stride, size_t(state.matrix_order), state.batch_count),
            "Chan preconditioner", "initialize batched pointers on GPU"))
        return 3;

    cublasStatus_t status;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    status = cublasCgetrfBatched(
        handle, state.matrix_order, state.buffers.vector.matrix_ptrs, state.matrix_order,
        cuda_utils::device_integer_library_view(state.buffers.vector.pivots), cuda_utils::device_integer_library_view(state.buffers.vector.factor_info), state.batch_count);
#else
    status = cublasZgetrfBatched(
        handle, state.matrix_order, state.buffers.vector.matrix_ptrs,
        state.matrix_order, cuda_utils::device_integer_library_view(state.buffers.vector.pivots), cuda_utils::device_integer_library_view(state.buffers.vector.factor_info),
        state.batch_count);
#endif
    if (!cuda_utils::check_cublas(status, "Chan preconditioner", "batched GPU LU factorization"))
        return 4;
    std::vector<int> info(size_t(state.batch_count));
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(info.data(), state.buffers.vector.factor_info,
                                           info.size() * sizeof(int),
                                           cuda_utils::TransferCategory::Preconditioner),
            "Chan preconditioner", "download factorization status"))
        return 5;
    for (int batch = 0; batch < state.batch_count; ++batch) {
        if (info[size_t(batch)] != 0) {
            std::fprintf(stderr, "Chan preconditioner: GPU LU failed at block %d, info=%d\n", batch,
                         info[size_t(batch)]);
            state.matrices_factorized = false;
            return 6;
        }
    }
    state.matrices_factorized = true;
    std::fprintf(stderr, "[Chan preconditioner CUDA] factored %d blocks of order %d on GPU\n",
                 state.batch_count, state.matrix_order);
    return 0;
}

//-----------------------------------------------------------
int generate_chan_and_build_cuda_(const CUCOMPLEX *device_mean_polarizability,
                                  const CU_REAL *aretecube, const CU_REAL *k0, const int *nx,
                                  const int *ny, const int *nz, const int *nxm, const int *nym,
                                  const int *nzm) {
    if (!device_mean_polarizability || !aretecube || !k0 || !nx || !ny || !nz || !nxm || !nym ||
        !nzm || !ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm) || !initialize_cuda_device())
        return 1;
    if (*nz > std::numeric_limits<int>::max() / 6)
        return 2;

    const size_t plane = size_t(*nx) * size_t(*ny);
    const size_t grid_count = plane * size_t(*nz);
    const size_t vector_count = size_t(6) * grid_count;
    // Chan uses a separately generated tensor Green (six spatial components),
    // then six transformed planes and six permuted channels. Keep each GPU
    // allocation in cuda_data so repeated block builds reuse the same storage.
    cuda_data::ChanPreconditionerBufferSizes buffer_sizes = buffer_sizes_for_state();
    buffer_sizes.vector_green_count = vector_count;
    buffer_sizes.vector_chan_vectors_count = vector_count;
    if (!cuda_data::get_device_chan_preconditioner_buffers(buffer_sizes, &state.buffers))
        return 3;
    CUCOMPLEX *green_components = state.buffers.vector.green_components;
    CUCOMPLEX *fft_components = state.buffers.vector.fft_components;
    CUCOMPLEX *chan_vectors = state.buffers.vector.chan_vectors;
    if (!green_components || !fft_components || !chan_vectors) {
        return 3;
    }

    if (chan_preconditioner_green_tensor_wrapper(green_components, *aretecube, *k0, *nx, *ny,
                                                 *nz) != cudaSuccess ||
        chan_preconditioner_interpolate_wrapper(green_components, fft_components,
                                                device_mean_polarizability, *nx, *ny,
                                                *nz) != cudaSuccess) {
        return 4;
    }
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.vector.green_fft_plan,
                                               fft_components, fft_components, CUFFT_INVERSE),
                                 "Chan preconditioner", "inverse FFT of six Chan channels") ||
        chan_preconditioner_transpose_vectors_wrapper(fft_components, chan_vectors, *nx, *ny,
                                                      *nz) != cudaSuccess ||
        chan_preconditioner_build_blocks_wrapper(chan_vectors, state.buffers.vector.matrices, *nx,
                                                 *ny, *nz) != cudaSuccess) {
        return 5;
    }
    if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), "Chan preconditioner",
                                "finish GPU Green and Chan setup")) {
        return 6;
    }
    state.matrices_ready = true;
    state.matrices_factorized = false;
    std::fprintf(
        stderr,
        "[Chan preconditioner CUDA] generated Green tensor, six Chan FFTs and %d blocks on GPU\n",
        state.batch_count);
    return 0;
}

//-----------------------------------------------------------
int generate_chan_from_polarizabilities_cuda_(const CUCOMPLEX *polarizabilities,
                                              const int *dipole_count, const CU_REAL *aretecube,
                                              const CU_REAL *k0, const int *nx, const int *ny,
                                              const int *nz, const int *nxm, const int *nym,
                                              const int *nzm) {
    if (!polarizabilities || !dipole_count || *dipole_count <= 0 || !aretecube || !k0 || !nx ||
        !ny || !nz || !nxm || !nym || !nzm || !ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm) ||
        !initialize_cuda_device())
        return 1;

    cuda_data::ChanPreconditionerBufferSizes buffer_sizes = buffer_sizes_for_state();
    buffer_sizes.vector_polarizability_count = size_t(*dipole_count);
    if (!cuda_data::get_device_chan_preconditioner_buffers(buffer_sizes, &state.buffers))
        return 2;
    CUCOMPLEX *device_polarizabilities = state.buffers.vector.polarizabilities;
    CUCOMPLEX *device_sum = state.buffers.vector.polarizability_sum;
    CU_INT *device_nonzero_count = state.buffers.vector.nonzero_count;
    if (!device_polarizabilities || !device_sum || !device_nonzero_count ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(device_polarizabilities, polarizabilities,
                                           size_t(*dipole_count) * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Preconditioner),
            "Chan preconditioner", "upload scalar polarizabilities"))
        return 2;
    if (chan_preconditioner_mean_polarizability_wrapper(device_polarizabilities, *dipole_count,
                                                        device_sum,
                                                        device_nonzero_count) != cudaSuccess)
        return 3;

    int nonzero_count = 0;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(&nonzero_count, device_nonzero_count, sizeof(int),
                                           cuda_utils::TransferCategory::Preconditioner),
            "Chan preconditioner", "download nonzero polarization count"))
        return 4;
    if (nonzero_count == 0) {
        std::fprintf(stderr, "Chan preconditioner: no nonzero scalar polarizabilities\n");
        return 5;
    }
    if (chan_preconditioner_finalize_mean_polarizability_wrapper(
            device_sum, device_nonzero_count) != cudaSuccess)
        return 6;
    std::fprintf(stderr, "[Chan preconditioner CUDA] averaged %d scalar polarizabilities on GPU\n",
                 nonzero_count);
    const int build_status =
        generate_chan_and_build_cuda_(device_sum, aretecube, k0, nx, ny, nz, nxm, nym, nzm);
    return build_status;
}

} // namespace

//-----------------------------------------------------------
extern "C" void ifdda_reset_preconditionneur_apply_cuda_state() {
    state = {};
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_apply_dense_cuda_(CUCOMPLEX *cpu_field, const int *nx,
                                                     const int *ny, const int *nz, const int *nxm,
                                                     const int *nym, const int *nzm) {
    return apply_chan_preconditioner_cuda_(cpu_field, nullptr, nx, ny, nz, nxm, nym, nzm, true);
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_upload_cuda_(const CUCOMPLEX *cpu_matrices,
                                                const int *cpu_pivots, const int *nx, const int *ny,
                                                const int *nz, const int *nxm, const int *nym,
                                                const int *nzm) {
    if (!cpu_matrices || !cpu_pivots || !nx || !ny || !nz || !nxm || !nym || !nzm ||
        !ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm))
        return 1;

    const int matrix_order = 3 * *nz;
    const int batch_count = *nx * *ny;
    if (matrix_order != state.matrix_order || batch_count != state.batch_count)
        return 2;

    const size_t matrix_count = size_t(matrix_order) * size_t(matrix_order) * size_t(batch_count);
    const size_t pivot_count = size_t(matrix_order) * size_t(batch_count);
    if (!upload_chan_matrices(cpu_matrices))
        return 2;

    std::vector<int> pivots(pivot_count);
    for (size_t i = 0; i < pivot_count; ++i) {
        // Preserve the pivot convention used by the Fortran ZGETRF/ZGETRS
        // pair. cuBLAS getrsBatched accepts this LAPACK pivot sequence.
        pivots[i] = cpu_pivots[i];
        if (pivots[i] < 1 || pivots[i] > matrix_order)
            return 3;
    }
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(state.buffers.vector.pivots, pivots.data(),
                                           pivot_count * sizeof(int),
                                           cuda_utils::TransferCategory::Preconditioner),
            "Chan preconditioner", "upload Chan pivots"))
        return 4;

    const size_t matrix_stride = size_t(matrix_order) * size_t(matrix_order);
    if (!cuda_utils::check_cuda(
            chan_preconditioner_init_batch_ptrs_wrapper(
                state.buffers.vector.matrices, state.buffers.vector.rhs,
                state.buffers.vector.matrix_ptrs, state.buffers.vector.rhs_ptrs,
                matrix_stride, size_t(matrix_order), batch_count),
            "Chan preconditioner", "initialize uploaded Chan pointers on GPU"))
        return 5;
    state.matrices_ready = true;
    state.matrices_factorized = true;
    std::fprintf(stderr, "[Chan preconditioner CUDA] uploaded %d LU blocks of order %d\n",
                 batch_count, matrix_order);
    return 0;
}

// Upload the assembled (unfactored) Chan blocks and factor all blocks on the
// GPU. The Fortran reference path still performs LAPACK GETRF on the CPU.
//-----------------------------------------------------------
extern "C" int chan_preconditioner_factorize_cuda_(const CUCOMPLEX *cpu_matrices, const int *nx,
                                                   const int *ny, const int *nz, const int *nxm,
                                                   const int *nym, const int *nzm) {
    if (!cpu_matrices || !nx || !ny || !nz || !nxm || !nym || !nzm ||
        !ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm))
        return 1;
    if (!initialize_cuda_device() || !cuda_utils::get_cublas_handle())
        return 2;
    if (!upload_chan_matrices(cpu_matrices))
        return 3;
    const size_t matrix_stride = size_t(state.matrix_order) * size_t(state.matrix_order);
    if (!cuda_utils::check_cuda(
            chan_preconditioner_init_batch_ptrs_wrapper(
                state.buffers.vector.matrices, state.buffers.vector.rhs,
                state.buffers.vector.matrix_ptrs, state.buffers.vector.rhs_ptrs,
                matrix_stride, size_t(state.matrix_order), state.batch_count),
            "Chan preconditioner", "initialize factorization pointers on GPU"))
        return 3;
    const cublasHandle_t handle = cuda_utils::get_cublas_handle();
    cublasStatus_t status;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    status = cublasCgetrfBatched(
        handle, state.matrix_order, state.buffers.vector.matrix_ptrs, state.matrix_order,
        cuda_utils::device_integer_library_view(state.buffers.vector.pivots), cuda_utils::device_integer_library_view(state.buffers.vector.factor_info), state.batch_count);
#else
    status = cublasZgetrfBatched(
        handle, state.matrix_order, state.buffers.vector.matrix_ptrs,
        state.matrix_order, cuda_utils::device_integer_library_view(state.buffers.vector.pivots), cuda_utils::device_integer_library_view(state.buffers.vector.factor_info),
        state.batch_count);
#endif
    if (!cuda_utils::check_cublas(status, "Chan preconditioner", "batched GPU LU factorization"))
        return 4;
    std::vector<int> info(size_t(state.batch_count));
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(info.data(), state.buffers.vector.factor_info,
                                           info.size() * sizeof(int),
                                           cuda_utils::TransferCategory::Preconditioner),
            "Chan preconditioner", "download factorization status"))
        return 5;
    for (int batch = 0; batch < state.batch_count; ++batch) {
        if (info[size_t(batch)] != 0) {
            std::fprintf(stderr, "Chan preconditioner: GPU LU failed at block %d, info=%d\n", batch,
                         info[size_t(batch)]);
            state.matrices_ready = false;
            return 6;
        }
    }
    state.matrices_ready = true;
    state.matrices_factorized = true;
    std::fprintf(stderr, "[Chan preconditioner CUDA] factored %d blocks of order %d on GPU\n",
                 state.batch_count, state.matrix_order);
    return 0;
}

// The right-Chan driver first builds unfactorized blocks to transform its
// initial guess. Reuse those resident blocks for the later iterative solve
// instead of regenerating the same Green tensor and Chan matrices.
//-----------------------------------------------------------
extern "C" int chan_preconditioner_factorize_resident_cuda_(const int *nx, const int *ny,
                                                            const int *nz, const int *nxm,
                                                            const int *nym, const int *nzm) {
    if (!nx || !ny || !nz || !nxm || !nym || !nzm ||
        !ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm) || !state.matrices_ready ||
        state.matrices_factorized)
        return 1;
    return factorize_chan_blocks_cuda_();
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_build_cuda_(const CUCOMPLEX *chan_xx, const CUCOMPLEX *chan_xy,
                                               const CUCOMPLEX *chan_xz, const CUCOMPLEX *chan_yy,
                                               const CUCOMPLEX *chan_yz, const CUCOMPLEX *chan_zz,
                                               const int *nx, const int *ny, const int *nz,
                                               const int *nxm, const int *nym, const int *nzm) {
    return build_chan_blocks_cuda_(chan_xx, chan_xy, chan_xz, chan_yy, chan_yz, chan_zz, nx, ny, nz,
                                   nxm, nym, nzm);
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_build_factorize_cuda_(
    const CUCOMPLEX *chan_xx, const CUCOMPLEX *chan_xy, const CUCOMPLEX *chan_xz,
    const CUCOMPLEX *chan_yy, const CUCOMPLEX *chan_yz, const CUCOMPLEX *chan_zz, const int *nx,
    const int *ny, const int *nz, const int *nxm, const int *nym, const int *nzm) {
    const int build_status = build_chan_blocks_cuda_(chan_xx, chan_xy, chan_xz, chan_yy, chan_yz,
                                                     chan_zz, nx, ny, nz, nxm, nym, nzm);
    return build_status == 0 ? factorize_chan_blocks_cuda_() : build_status;
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_generate_build_cuda_(const CUCOMPLEX *mean_polarizability,
                                                        const CU_REAL *aretecube, const CU_REAL *k0,
                                                        const int *nx, const int *ny, const int *nz,
                                                        const int *nxm, const int *nym,
                                                        const int *nzm) {
    if (!mean_polarizability || !initialize_cuda_device())
        return 1;
    if (!ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm))
        return 2;
    cuda_data::ChanPreconditionerBufferSizes buffer_sizes = buffer_sizes_for_state();
    buffer_sizes.vector_polarizability_count = 1;
    if (!cuda_data::get_device_chan_preconditioner_buffers(buffer_sizes, &state.buffers))
        return 2;
    CUCOMPLEX *device_mean_polarizability = state.buffers.vector.mean_polarizability;
    const bool uploaded =
        cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                                   device_mean_polarizability, mean_polarizability,
                                   sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Preconditioner),
                               "Chan preconditioner", "upload mean polarizability");
    const int status = uploaded
                           ? generate_chan_and_build_cuda_(device_mean_polarizability, aretecube,
                                                           k0, nx, ny, nz, nxm, nym, nzm)
                           : 3;
    return status;
}

//-----------------------------------------------------------
extern "C" int chan_precon_pol_build_cuda_(const CUCOMPLEX *polarizabilities,
                                           const int *dipole_count, const CU_REAL *aretecube,
                                           const CU_REAL *k0, const int *nx, const int *ny,
                                           const int *nz, const int *nxm, const int *nym,
                                           const int *nzm) {
    return generate_chan_from_polarizabilities_cuda_(polarizabilities, dipole_count, aretecube, k0,
                                                     nx, ny, nz, nxm, nym, nzm);
}

//-----------------------------------------------------------
extern "C" int chan_precon_pol_build_lu_cuda_(const CUCOMPLEX *polarizabilities,
                                              const int *dipole_count, const CU_REAL *aretecube,
                                              const CU_REAL *k0, const int *nx, const int *ny,
                                              const int *nz, const int *nxm, const int *nym,
                                              const int *nzm) {
    const int build_status = generate_chan_from_polarizabilities_cuda_(
        polarizabilities, dipole_count, aretecube, k0, nx, ny, nz, nxm, nym, nzm);
    return build_status == 0 ? factorize_chan_blocks_cuda_() : build_status;
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_generate_build_factorize_cuda_(
    const CUCOMPLEX *mean_polarizability, const CU_REAL *aretecube, const CU_REAL *k0,
    const int *nx, const int *ny, const int *nz, const int *nxm, const int *nym, const int *nzm) {
    const int build_status = generate_chan_and_build_cuda_(mean_polarizability, aretecube, k0, nx,
                                                           ny, nz, nxm, nym, nzm);
    return build_status == 0 ? factorize_chan_blocks_cuda_() : build_status;
}

// Apply the un-factorized Chan matrix used only to create the initial guess of
// the right-preconditioned path. The matrix blocks have already been assembled
// on the GPU; FFTs, batched matrix-vector products and scatter also run there.
//-----------------------------------------------------------
static int apply_initial_chan_preconditioner_cuda_(
    CUCOMPLEX *cpu_field, const CUCOMPLEX *cpu_matrices, const int *cpu_dipole_map, const int *nx,
    const int *ny, const int *nz, const int *nxm, const int *nym, const int *nzm, bool dense_grid,
    COMPLEX *cpu_solution = nullptr, size_t solution_count = 0,
    bool solution_already_on_device = false) {
    if (!cpu_field || (!dense_grid && !cpu_dipole_map) || !nx || !ny || !nz || !nxm || !nym ||
        !nzm || !ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm) ||
        (!cpu_matrices && (!state.matrices_ready || state.matrices_factorized)))
        return 1;

    const size_t map_count = size_t(*nxm) * size_t(*nym) * size_t(*nzm);
    const size_t plane_count = size_t(*nx) * size_t(*ny);
    const bool publish_to_solver_vector =
        cpu_solution && solution_count > 0 && solution_count <= state.field_count;
    CUCOMPLEX *device_solution = publish_to_solver_vector
                                     ? get_gpbicg_device_xi_buffer(cpu_solution, solution_count)
                                     : nullptr;
    if (publish_to_solver_vector && std::getenv("IFDDA_TRACE_BPM_RESIDENT"))
        std::fprintf(stderr, "[Chan initial] host XI key=%p GPU XI=%p count=%zu current=%d\n",
                     static_cast<void *>(cpu_solution), static_cast<void *>(device_solution),
                     solution_count, solution_already_on_device ? 1 : 0);
    if (publish_to_solver_vector &&
        (!device_solution ||
         (!solution_already_on_device &&
          !cuda_utils::check_cuda(
              cuda_data::copy_host_to_device(device_solution, cpu_solution,
                                             solution_count * sizeof(CUCOMPLEX),
                                             cuda_utils::TransferCategory::SolverVector),
              "Chan initial preconditioner", "upload starting iterate"))))
        return 2;
    const CU_INT *dipole_map = dense_grid ? state.buffers.vector.identity_map
                                       : get_device_tabdip(cpu_dipole_map, map_count);
    if (!dipole_map || !initialize_cuda_device())
        return 2;
    if (cpu_matrices) {
        if (!upload_chan_matrices(cpu_matrices))
            return 3;
    }
    if (publish_to_solver_vector) {
        if (!cuda_utils::check_cuda(
                cudaMemset(state.buffers.vector.field, 0, state.field_count * sizeof(CUCOMPLEX)),
                "Chan initial preconditioner", "clear dense staging field") ||
            !cuda_utils::check_cuda(cuda_utils::tracked_memcpy(state.buffers.vector.field,
                                                               device_solution,
                                                               solution_count * sizeof(CUCOMPLEX),
                                                               cudaMemcpyDeviceToDevice),
                                    "Chan initial preconditioner", "seed dense field from iterate"))
            return 3;
    } else if (!cuda_utils::check_cuda(
                   cuda_data::copy_host_to_device(state.buffers.vector.field, cpu_field,
                                                  state.field_count * sizeof(CUCOMPLEX),
                                                  cuda_utils::TransferCategory::Preconditioner),
                   "Chan initial preconditioner", "upload initial guess"))
        return 3;
    const CUCOMPLEX *input_field = state.buffers.vector.field;

    if (chan_preconditioner_pack_fft_wrapper(input_field, dipole_map,
                                             state.buffers.vector.fft_planes, *nx, *ny, *nz, *nxm,
                                             *nym, *nzm, int(map_count)) != cudaSuccess ||
        !cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.vector.fft_plan, state.buffers.vector.fft_planes,
                                               state.buffers.vector.fft_planes, CUFFT_INVERSE),
                                 "Chan initial preconditioner", "backward 2-D FFT") ||
        chan_preconditioner_fft_to_rhs_wrapper(state.buffers.vector.fft_planes,
                                               state.buffers.vector.rhs, *nx, *ny,
                                               *nz) != cudaSuccess)
        return 4;

    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle)
        return 5;
    const CUCOMPLEX one = CUFFT_REAL(CU_REAL(1));
    const CUCOMPLEX zero = CUFFT_ZERO;
    for (int batch = 0; batch < state.batch_count; ++batch) {
        const size_t matrix_offset =
            size_t(batch) * size_t(state.matrix_order) * size_t(state.matrix_order);
        const size_t vector_offset = size_t(batch) * size_t(state.matrix_order);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const cublasStatus_t status = cublasCgemv(
            handle, CUBLAS_OP_N, state.matrix_order, state.matrix_order, &one,
            state.buffers.vector.matrices + matrix_offset, state.matrix_order,
            state.buffers.vector.rhs + vector_offset, 1, &zero,
            state.buffers.vector.rhs_output + vector_offset, 1);
#else
        const cublasStatus_t status = cublasZgemv(
            handle, CUBLAS_OP_N, state.matrix_order, state.matrix_order, &one,
            state.buffers.vector.matrices + matrix_offset, state.matrix_order,
            state.buffers.vector.rhs + vector_offset, 1, &zero,
            state.buffers.vector.rhs_output + vector_offset, 1);
#endif
        if (!cuda_utils::check_cublas(status, "Chan initial preconditioner",
                                      "matrix-vector product"))
            return 6;
    }

    if (chan_preconditioner_rhs_to_fft_wrapper(state.buffers.vector.rhs_output,
                                               state.buffers.vector.fft_planes, *nx, *ny,
                                               *nz) != cudaSuccess ||
        !cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.vector.fft_plan, state.buffers.vector.fft_planes,
                                               state.buffers.vector.fft_planes, CUFFT_FORWARD),
                                 "Chan initial preconditioner", "forward 2-D FFT") ||
        chan_preconditioner_unpack_fft_wrapper(
            state.buffers.vector.fft_planes, dipole_map, state.buffers.vector.field, *nx, *ny, *nz,
            *nxm, *nym, *nzm, int(map_count), CU_REAL(plane_count)) != cudaSuccess)
        return 7;
    if (publish_to_solver_vector) {
        if (!cuda_utils::check_cuda(
                cuda_utils::tracked_memcpy(device_solution, state.buffers.vector.field,
                                           solution_count * sizeof(CUCOMPLEX),
                                           cudaMemcpyDeviceToDevice),
                "Chan initial preconditioner", "publish packed initial iterate"))
            return 8;
        mark_gpbicg_device_xi_current(cpu_solution, true);
    } else if (!cuda_utils::check_cuda(
                   cuda_data::copy_device_to_host(cpu_field, state.buffers.vector.field,
                                                  state.field_count * sizeof(CUCOMPLEX),
                                                  cuda_utils::TransferCategory::Preconditioner),
                   "Chan initial preconditioner", "download initial guess"))
        return 8;
    return 0;
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_apply_initial_cuda_(CUCOMPLEX *cpu_field,
                                                       const CUCOMPLEX *cpu_matrices,
                                                       const int *cpu_dipole_map, const int *nx,
                                                       const int *ny, const int *nz, const int *nxm,
                                                       const int *nym, const int *nzm) {
    return apply_initial_chan_preconditioner_cuda_(cpu_field, cpu_matrices, cpu_dipole_map, nx, ny,
                                                   nz, nxm, nym, nzm, false);
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_apply_initial_device_cuda_(
    CUCOMPLEX *cpu_field, COMPLEX *cpu_solution, const int *dipole_count,
    const int *cpu_dipole_map, const int *nx, const int *ny, const int *nz, const int *nxm,
    const int *nym, const int *nzm) {
    if (!dipole_count || *dipole_count <= 0 ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / size_t(3))
        return 1;
    return apply_initial_chan_preconditioner_cuda_(cpu_field, nullptr, cpu_dipole_map, nx, ny, nz,
                                                   nxm, nym, nzm, false, cpu_solution,
                                                   size_t(3) * size_t(*dipole_count), true);
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_apply_initial_resident_cuda_(
    CUCOMPLEX *cpu_field, COMPLEX *cpu_solution, const int *dipole_count,
    const int *cpu_dipole_map, const int *nx, const int *ny, const int *nz, const int *nxm,
    const int *nym, const int *nzm) {
    if (!dipole_count || *dipole_count <= 0 ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / size_t(3))
        return 1;
    return apply_initial_chan_preconditioner_cuda_(cpu_field, nullptr, cpu_dipole_map, nx, ny, nz,
                                                   nxm, nym, nzm, false, cpu_solution,
                                                   size_t(3) * size_t(*dipole_count), true);
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_apply_initial_dense_cuda_(CUCOMPLEX *cpu_field,
                                                             const CUCOMPLEX *cpu_matrices,
                                                             const int *nx, const int *ny,
                                                             const int *nz, const int *nxm,
                                                             const int *nym, const int *nzm) {
    return apply_initial_chan_preconditioner_cuda_(cpu_field, cpu_matrices, nullptr, nx, ny, nz,
                                                   nxm, nym, nzm, true);
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_apply_initial_dense_device_cuda_(
    CUCOMPLEX *cpu_field, COMPLEX *cpu_solution, const int *dipole_count, const int *nx,
    const int *ny, const int *nz, const int *nxm, const int *nym, const int *nzm) {
    if (!dipole_count || *dipole_count <= 0 ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / size_t(3))
        return 1;
    return apply_initial_chan_preconditioner_cuda_(cpu_field, nullptr, nullptr, nx, ny, nz, nxm,
                                                   nym, nzm, true, cpu_solution,
                                                   size_t(3) * size_t(*dipole_count), true);
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_apply_initial_dense_resident_cuda_(
    CUCOMPLEX *cpu_field, COMPLEX *cpu_solution, const int *dipole_count, const int *nx,
    const int *ny, const int *nz, const int *nxm, const int *nym, const int *nzm) {
    if (!dipole_count || *dipole_count <= 0 ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / size_t(3))
        return 1;
    return apply_initial_chan_preconditioner_cuda_(cpu_field, nullptr, nullptr, nx, ny, nz, nxm,
                                                   nym, nzm, true, cpu_solution,
                                                   size_t(3) * size_t(*dipole_count), true);
}

// Build a Born start for the right-Chan solve without staging Xi through the
// host. The unfactorized Chan blocks were assembled by the preceding ntest=0
// call; apply their inverse action to the incident field on the device.
//-----------------------------------------------------------
extern "C" int chan_preconditioner_initialize_born_resident_cuda_(
    CUCOMPLEX *incident, COMPLEX *cpu_solution, const int *dipole_count,
    const int *cpu_dipole_map, const int *nx, const int *ny, const int *nz, const int *nxm,
    const int *nym, const int *nzm, const int *dense_grid) {
    if (!incident || !cpu_solution || !dipole_count || *dipole_count <= 0 || !dense_grid ||
        (*dense_grid != 0 && *dense_grid != 1) || (*dense_grid == 0 && !cpu_dipole_map) ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / size_t(3))
        return 1;

    const size_t count = size_t(3) * size_t(*dipole_count);
    CUCOMPLEX *device_solution = get_gpbicg_device_xi_buffer(cpu_solution, count);
    if (!device_solution ||
        cuda_data::copy_host_to_device(device_solution, incident, count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::SolverVector) != cudaSuccess)
        return 2;
    mark_gpbicg_device_xi_current(cpu_solution, true);

    const int status = apply_initial_chan_preconditioner_cuda_(
        incident, nullptr, cpu_dipole_map, nx, ny, nz, nxm, nym, nzm, *dense_grid != 0,
        cpu_solution, count, true);
    if (status == 0)
        std::fprintf(stderr, "[Chan preconditioner CUDA] initialized right Born iterate on GPU\n");
    return status;
}

//-----------------------------------------------------------
extern "C" int chan_preconditioner_apply_cuda_(CUCOMPLEX *cpu_field, const int *cpu_dipole_map,
                                               const int *nx, const int *ny, const int *nz,
                                               const int *nxm, const int *nym, const int *nzm) {
    if (!cpu_field || !cpu_dipole_map || !nx || !ny || !nz || !nxm || !nym || !nzm ||
        !ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm) || !state.matrices_ready)
        return 1;

    const size_t map_count = size_t(*nxm) * size_t(*nym) * size_t(*nzm);
    const size_t plane_count = size_t(*nx) * size_t(*ny);
    CU_INT *dipole_map = get_device_tabdip(cpu_dipole_map, map_count);
    if (!dipole_map || !initialize_cuda_device())
        return 2;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(state.buffers.vector.field, cpu_field,
                                           state.field_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Preconditioner),
            "Chan preconditioner", "upload current product"))
        return 3;
    if (chan_preconditioner_pack_fft_wrapper(cpu_field ? state.buffers.vector.field : nullptr,
                                             dipole_map, state.buffers.vector.fft_planes, *nx, *ny,
                                             *nz, *nxm, *nym, *nzm, int(map_count)) != cudaSuccess)
        return 4;
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.vector.fft_plan, state.buffers.vector.fft_planes,
                                               state.buffers.vector.fft_planes, CUFFT_INVERSE),
                                 "Chan preconditioner", "backward 2-D FFT"))
        return 5;
    if (chan_preconditioner_fft_to_rhs_wrapper(state.buffers.vector.fft_planes,
                                               state.buffers.vector.rhs, *nx, *ny,
                                               *nz) != cudaSuccess)
        return 6;
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle)
        return 7;
    int solve_info = 0;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const cublasStatus_t solve_status = cublasCgetrsBatched(
        handle, CUBLAS_OP_N, state.matrix_order, 1, state.buffers.vector.matrix_ptrs,
        state.matrix_order, cuda_utils::device_integer_library_view(state.buffers.vector.pivots), state.buffers.vector.rhs_ptrs,
        state.matrix_order,
        &solve_info, state.batch_count);
#else
    const cublasStatus_t solve_status = cublasZgetrsBatched(
        handle, CUBLAS_OP_N, state.matrix_order, 1, state.buffers.vector.matrix_ptrs,
        state.matrix_order, cuda_utils::device_integer_library_view(state.buffers.vector.pivots), state.buffers.vector.rhs_ptrs,
        state.matrix_order, &solve_info, state.batch_count);
#endif
    if (!cuda_utils::check_cublas(solve_status, "Chan preconditioner", "batched getrs")) {
        return 8;
    }
    if (solve_info != 0) {
        std::fprintf(stderr, "Chan preconditioner: batched getrs info=%d\n", solve_info);
        return 9;
    }

    if (chan_preconditioner_rhs_to_fft_wrapper(state.buffers.vector.rhs,
                                               state.buffers.vector.fft_planes, *nx, *ny,
                                               *nz) != cudaSuccess)
        return 11;
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.vector.fft_plan, state.buffers.vector.fft_planes,
                                               state.buffers.vector.fft_planes, CUFFT_FORWARD),
                                 "Chan preconditioner", "forward 2-D FFT"))
        return 12;
    if (chan_preconditioner_unpack_fft_wrapper(
            state.buffers.vector.fft_planes, dipole_map, state.buffers.vector.field, *nx, *ny, *nz,
            *nxm, *nym, *nzm, int(map_count), CU_REAL(plane_count)) != cudaSuccess)
        return 13;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(cpu_field, state.buffers.vector.field,
                                           state.field_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Preconditioner),
            "Chan preconditioner", "download preconditioned product"))
        return 14;
    return 0;
}

// Resident variant used by inverserig_CUDA.cpp. The input/output vector is
// already device-resident, so each iteration avoids a host round trip.
//-----------------------------------------------------------
extern "C" int chan_preconditioner_apply_device_cuda_(CUCOMPLEX *device_field,
                                                      const int *cpu_dipole_map, const int *nx,
                                                      const int *ny, const int *nz, const int *nxm,
                                                      const int *nym, const int *nzm) {
    if (!device_field || !cpu_dipole_map || !nx || !ny || !nz || !nxm || !nym || !nzm ||
        !ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm) || !state.matrices_ready)
        return 1;

    const size_t map_count = size_t(*nxm) * size_t(*nym) * size_t(*nzm);
    const size_t plane_count = size_t(*nx) * size_t(*ny);
    CU_INT *dipole_map = get_device_tabdip(cpu_dipole_map, map_count);
    if (!dipole_map || !initialize_cuda_device())
        return 2;
    static bool diagnostic_written = false;
    const bool trace = !diagnostic_written && std::getenv("IFDDA_TRACE_CHAN_PRODUCT") != nullptr;
    size_t packed_count = 0;
    if (trace) {
        for (size_t i = 0; i < map_count; ++i)
            if (cpu_dipole_map[i] > 0)
                packed_count = std::max(packed_count, static_cast<size_t>(cpu_dipole_map[i]));
        packed_count *= 3;
        std::vector<CUCOMPLEX> values(packed_count);
        if (packed_count != 0 && cuda_data::copy_device_to_host(
                                     values.data(), device_field, packed_count * sizeof(CUCOMPLEX),
                                     cuda_utils::TransferCategory::Diagnostic) == cudaSuccess) {
            double norm2 = 0.0;
            for (const CUCOMPLEX value : values)
                norm2 += double(value.x) * double(value.x) + double(value.y) * double(value.y);
            std::fprintf(stderr, "[CHAN CUDA input] (%.17g,%.17g) (%.17g,%.17g) %.17g\n",
                         double(values[0].x), double(values[0].y), double(values[1].x),
                         double(values[1].y), std::sqrt(norm2));
        }
    }
    if (chan_preconditioner_pack_fft_wrapper(device_field, dipole_map,
                                             state.buffers.vector.fft_planes, *nx, *ny, *nz, *nxm,
                                             *nym, *nzm, int(map_count)) != cudaSuccess)
        return 3;
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.vector.fft_plan, state.buffers.vector.fft_planes,
                                               state.buffers.vector.fft_planes, CUFFT_INVERSE),
                                 "Chan preconditioner", "backward 2-D FFT"))
        return 4;
    if (chan_preconditioner_fft_to_rhs_wrapper(state.buffers.vector.fft_planes,
                                               state.buffers.vector.rhs, *nx, *ny,
                                               *nz) != cudaSuccess)
        return 5;
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle)
        return 6;
    int solve_info = 0;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const cublasStatus_t solve_status = cublasCgetrsBatched(
        handle, CUBLAS_OP_N, state.matrix_order, 1, state.buffers.vector.matrix_ptrs,
        state.matrix_order, cuda_utils::device_integer_library_view(state.buffers.vector.pivots), state.buffers.vector.rhs_ptrs,
        state.matrix_order,
        &solve_info, state.batch_count);
#else
    const cublasStatus_t solve_status = cublasZgetrsBatched(
        handle, CUBLAS_OP_N, state.matrix_order, 1, state.buffers.vector.matrix_ptrs,
        state.matrix_order, cuda_utils::device_integer_library_view(state.buffers.vector.pivots), state.buffers.vector.rhs_ptrs,
        state.matrix_order, &solve_info, state.batch_count);
#endif
    const bool solve_ok =
        cuda_utils::check_cublas(solve_status, "Chan preconditioner", "resident batched getrs");
    if (!solve_ok || solve_info != 0) {
        std::fprintf(stderr,
                     "Chan preconditioner: resident batched solve failed (status %d, info %d)\n",
                     int(solve_status), solve_info);
        return 7;
    }
    if (chan_preconditioner_rhs_to_fft_wrapper(state.buffers.vector.rhs,
                                               state.buffers.vector.fft_planes, *nx, *ny,
                                               *nz) != cudaSuccess)
        return 8;
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.vector.fft_plan, state.buffers.vector.fft_planes,
                                               state.buffers.vector.fft_planes, CUFFT_FORWARD),
                                 "Chan preconditioner", "forward 2-D FFT"))
        return 9;
    if (chan_preconditioner_unpack_fft_wrapper(state.buffers.vector.fft_planes, dipole_map,
                                               device_field, *nx, *ny, *nz, *nxm, *nym, *nzm,
                                               int(map_count), CU_REAL(plane_count)) != cudaSuccess)
        return 10;
    if (trace && packed_count != 0) {
        std::vector<CUCOMPLEX> values(packed_count);
        if (cuda_data::copy_device_to_host(
                values.data(), device_field, packed_count * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::Diagnostic) == cudaSuccess) {
            double norm2 = 0.0;
            for (const CUCOMPLEX value : values)
                norm2 += double(value.x) * double(value.x) + double(value.y) * double(value.y);
            std::fprintf(stderr, "[CHAN CUDA after solve] (%.17g,%.17g) (%.17g,%.17g) %.17g\n",
                         double(values[0].x), double(values[0].y), double(values[1].x),
                         double(values[1].y), std::sqrt(norm2));
        }
        diagnostic_written = true;
    }
    return cuda_utils::check_cuda(cudaDeviceSynchronize(), "Chan preconditioner",
                                  "finish resident Chan application")
               ? 0
               : 11;
}
