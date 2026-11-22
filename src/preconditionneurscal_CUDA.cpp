#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

struct ScalarChanState {
    int nx = 0, ny = 0, nz = 0, nxm = 0, nym = 0, nzm = 0;
    size_t mesh_count = 0, field_count = 0, block_count = 0;
    cuda_data::ChanPreconditionerDeviceBuffers buffers{};
    bool buffers_ready = false;
    bool matrices_ready = false;
    bool factorized = false;
};

ScalarChanState state;
std::vector<int> dense_dipole_map;

//-----------------------------------------------------------
void set_error(int *nstop, char *infostr, size_t infostr_len, const char *message, int code) {
    if (nstop)
        *nstop = code;
    if (infostr && infostr_len) {
        std::memset(infostr, ' ', infostr_len);
        const size_t length =
            std::strlen(message) < infostr_len ? std::strlen(message) : infostr_len;
        std::memcpy(infostr, message, length);
    }
}

//-----------------------------------------------------------
void release_state() {
    // Named GPU buffers and dimension-keyed cuFFT plans persist in cuda_data.
    state = {};
}

//-----------------------------------------------------------
cuda_data::ChanPreconditionerBufferSizes buffer_sizes_for_state() {
    cuda_data::ChanPreconditionerBufferSizes sizes{};
    sizes.scalar_nx = state.nx;
    sizes.scalar_ny = state.ny;
    sizes.scalar_nz = state.nz;
    sizes.scalar_mesh_count = state.mesh_count;
    sizes.scalar_field_count = state.field_count;
    sizes.scalar_matrix_count = state.block_count * size_t(state.nz) * size_t(state.nz);
    sizes.scalar_pivot_count = state.block_count * size_t(state.nz);
    sizes.scalar_block_count = state.block_count;
    return sizes;
}

//-----------------------------------------------------------
bool ensure_state(int nx, int ny, int nz, int nxm, int nym, int nzm) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || nxm < nx || nym < ny || nzm < nz)
        return false;
    if (state.nx == nx && state.ny == ny && state.nz == nz && state.nxm == nxm &&
        state.nym == nym && state.nzm == nzm && state.buffers.scalar.fft_plan &&
        state.buffers_ready)
        return true;

    release_state();
    state.nx = nx;
    state.ny = ny;
    state.nz = nz;
    state.nxm = nxm;
    state.nym = nym;
    state.nzm = nzm;
    state.mesh_count = size_t(nx) * size_t(ny) * size_t(nz);
    state.field_count = size_t(3) * size_t(nxm) * size_t(nym) * size_t(nzm);
    state.block_count = size_t(nx) * size_t(ny);
    if (state.mesh_count > size_t(std::numeric_limits<int>::max()) ||
        state.block_count > size_t(std::numeric_limits<int>::max()) ||
        size_t(nz) * size_t(nz) > size_t(std::numeric_limits<int>::max())) {
        release_state();
        return false;
    }

    cuda_data::ChanPreconditionerBufferSizes buffer_sizes = buffer_sizes_for_state();
    if (!cuda_data::get_device_chan_preconditioner_buffers(buffer_sizes, &state.buffers)) {
        release_state();
        return false;
    }
    state.buffers_ready = true;

    const size_t matrix_stride = size_t(nz) * size_t(nz);
    return cuda_utils::check_cuda(
               chan_preconditioner_init_batch_ptrs_wrapper(
                   state.buffers.scalar.matrices, state.buffers.scalar.rhs,
                   state.buffers.scalar.matrix_ptrs, state.buffers.scalar.rhs_ptrs,
                   matrix_stride, size_t(nz), int(state.block_count)),
               "scalar Chan preconditioner", "initialize block pointers on GPU");
}

//-----------------------------------------------------------
bool factorize_blocks() {
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle)
        return false;
    cublasStatus_t status;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    status = cublasCgetrfBatched(handle, state.nz, state.buffers.scalar.matrix_ptrs,
        state.nz, cuda_utils::device_integer_library_view(state.buffers.scalar.pivots), cuda_utils::device_integer_library_view(state.buffers.scalar.factor_info),
        int(state.block_count));
#else
    status = cublasZgetrfBatched(handle, state.nz, state.buffers.scalar.matrix_ptrs, state.nz,
        cuda_utils::device_integer_library_view(state.buffers.scalar.pivots), cuda_utils::device_integer_library_view(state.buffers.scalar.factor_info), int(state.block_count));
#endif
    if (!cuda_utils::check_cublas(status, "scalar Chan preconditioner",
                                  "batched GPU LU factorization"))
        return false;
    std::vector<int> info(state.block_count);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(info.data(), state.buffers.scalar.factor_info,
                                           info.size() * sizeof(int),
                                           cuda_utils::TransferCategory::Preconditioner),
            "scalar Chan preconditioner", "read LU status"))
        return false;
    for (size_t block = 0; block < info.size(); ++block)
        if (info[block] != 0) {
            std::fprintf(stderr, "scalar Chan preconditioner: LU failed at block %zu, info=%d\n",
                         block, info[block]);
            return false;
        }
    state.factorized = true;
    return true;
}

//-----------------------------------------------------------
bool apply_scalar_preconditioner(CUCOMPLEX *field, const int *cpu_tabdip, bool solve_blocks) {
    if (!state.matrices_ready || (solve_blocks && !state.factorized) || !field || !cpu_tabdip)
        return false;
    const size_t map_count = size_t(state.nxm) * size_t(state.nym) * size_t(state.nzm);
    CU_INT *dipole_map = get_device_tabdip(cpu_tabdip, map_count);
    if (!dipole_map || !initialize_cuda_device())
        return false;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(state.buffers.scalar.field, field,
                                           state.field_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Preconditioner),
            "scalar Chan preconditioner", "upload field"))
        return false;
    if (scalar_preconditioner_gather_wrapper(
            state.buffers.scalar.field, dipole_map, state.buffers.scalar.fft_planes, state.nx,
            state.ny, state.nz, state.nxm, state.nym, state.nzm) != cudaSuccess)
        return false;
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.scalar.fft_plan, state.buffers.scalar.fft_planes,
                                               state.buffers.scalar.fft_planes, CUFFT_INVERSE),
                                 "scalar Chan preconditioner", "backward 2-D FFT") ||
        scalar_preconditioner_fft_to_rhs_wrapper(state.buffers.scalar.fft_planes,
                                                 state.buffers.scalar.rhs, state.nx, state.ny,
                                                 state.nz) != cudaSuccess)
        return false;

    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle)
        return false;
    if (!solve_blocks) {
        if (scalar_preconditioner_matvec_wrapper(
                state.buffers.scalar.matrices, state.buffers.scalar.rhs,
                state.buffers.scalar.rhs_output, state.nx, state.ny, state.nz) != cudaSuccess ||
            scalar_preconditioner_rhs_to_fft_wrapper(state.buffers.scalar.rhs_output,
                                                     state.buffers.scalar.fft_planes, state.nx,
                                                     state.ny, state.nz) != cudaSuccess ||
            !cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.scalar.fft_plan, state.buffers.scalar.fft_planes,
                                                   state.buffers.scalar.fft_planes, CUFFT_FORWARD),
                                     "scalar Chan preconditioner", "forward initialization FFT") ||
            scalar_preconditioner_scatter_wrapper(
                state.buffers.scalar.fft_planes, dipole_map, state.buffers.scalar.field, state.nx,
                state.ny, state.nz, state.nxm, state.nym, state.nzm,
                CU_REAL(1) / CU_REAL(state.nx * state.ny)) != cudaSuccess)
            return false;
        return cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(field, state.buffers.scalar.field,
                                           state.field_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Preconditioner),
            "scalar Chan preconditioner", "download initialization product");
    }

    int solve_info = 0;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const cublasStatus_t status = cublasCgetrsBatched(
        handle, CUBLAS_OP_N, state.nz, 1, state.buffers.scalar.matrix_ptrs, state.nz,
        cuda_utils::device_integer_library_view(state.buffers.scalar.pivots), state.buffers.scalar.rhs_ptrs, state.nz, &solve_info,
        int(state.block_count));
#else
    const cublasStatus_t status = cublasZgetrsBatched(
        handle, CUBLAS_OP_N, state.nz, 1, state.buffers.scalar.matrix_ptrs, state.nz,
        cuda_utils::device_integer_library_view(state.buffers.scalar.pivots), state.buffers.scalar.rhs_ptrs, state.nz, &solve_info,
        int(state.block_count));
#endif
    if (!cuda_utils::check_cublas(status, "scalar Chan preconditioner", "batched GPU solve") ||
        solve_info != 0)
        return false;
    if (scalar_preconditioner_rhs_to_fft_wrapper(state.buffers.scalar.rhs,
                                                 state.buffers.scalar.fft_planes, state.nx,
                                                 state.ny, state.nz) != cudaSuccess ||
        !cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.scalar.fft_plan, state.buffers.scalar.fft_planes,
                                               state.buffers.scalar.fft_planes, CUFFT_FORWARD),
                                 "scalar Chan preconditioner", "forward 2-D FFT") ||
        scalar_preconditioner_scatter_wrapper(
            state.buffers.scalar.fft_planes, dipole_map, state.buffers.scalar.field, state.nx,
            state.ny, state.nz, state.nxm, state.nym, state.nzm,
            CU_REAL(1) / CU_REAL(state.nx * state.ny)) != cudaSuccess)
        return false;
    return cuda_utils::check_cuda(
        cuda_data::copy_device_to_host(field, state.buffers.scalar.field,
                                       state.field_count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Preconditioner),
        "scalar Chan preconditioner", "download preconditioned field");
}

//-----------------------------------------------------------
bool apply_scalar_preconditioner_resident(CUCOMPLEX *field, size_t field_count,
                                          const int *cpu_tabdip, bool solve_blocks) {
    if (!state.matrices_ready || (solve_blocks && !state.factorized) || !field || !cpu_tabdip ||
        field_count == 0 || field_count > state.field_count)
        return false;
    const size_t map_count = size_t(state.nxm) * size_t(state.nym) * size_t(state.nzm);
    CU_INT *dipole_map = get_device_tabdip(cpu_tabdip, map_count);
    if (!dipole_map || !initialize_cuda_device() ||
        !cuda_utils::check_cuda(
            cudaMemset(state.buffers.scalar.field, 0, state.field_count * sizeof(CUCOMPLEX)),
            "scalar Chan preconditioner", "clear resident field workspace") ||
        !cuda_utils::check_cuda(cuda_utils::tracked_memcpy(state.buffers.scalar.field, field,
                                                           field_count * sizeof(CUCOMPLEX),
                                                           cudaMemcpyDeviceToDevice),
                                "scalar Chan preconditioner", "stage resident field"))
        return false;
    if (scalar_preconditioner_gather_wrapper(
            state.buffers.scalar.field, dipole_map, state.buffers.scalar.fft_planes, state.nx,
            state.ny, state.nz, state.nxm, state.nym, state.nzm) != cudaSuccess ||
        !cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.scalar.fft_plan, state.buffers.scalar.fft_planes,
                                               state.buffers.scalar.fft_planes, CUFFT_INVERSE),
                                 "scalar Chan preconditioner", "backward 2-D FFT") ||
        scalar_preconditioner_fft_to_rhs_wrapper(state.buffers.scalar.fft_planes,
                                                 state.buffers.scalar.rhs, state.nx, state.ny,
                                                 state.nz) != cudaSuccess)
        return false;

    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle)
        return false;
    if (!solve_blocks) {
        if (scalar_preconditioner_matvec_wrapper(
                state.buffers.scalar.matrices, state.buffers.scalar.rhs,
                state.buffers.scalar.rhs_output, state.nx, state.ny, state.nz) != cudaSuccess ||
            scalar_preconditioner_rhs_to_fft_wrapper(state.buffers.scalar.rhs_output,
                                                     state.buffers.scalar.fft_planes, state.nx,
                                                     state.ny, state.nz) != cudaSuccess ||
            !cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.scalar.fft_plan, state.buffers.scalar.fft_planes,
                                                   state.buffers.scalar.fft_planes, CUFFT_FORWARD),
                                     "scalar Chan preconditioner", "forward initialization FFT") ||
            scalar_preconditioner_scatter_wrapper(
                state.buffers.scalar.fft_planes, dipole_map, state.buffers.scalar.field, state.nx,
                state.ny, state.nz, state.nxm, state.nym, state.nzm,
                CU_REAL(1) / CU_REAL(state.nx * state.ny)) != cudaSuccess)
            return false;
    } else {
        int solve_info = 0;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const cublasStatus_t status = cublasCgetrsBatched(
            handle, CUBLAS_OP_N, state.nz, 1, state.buffers.scalar.matrix_ptrs, state.nz,
            cuda_utils::device_integer_library_view(state.buffers.scalar.pivots), state.buffers.scalar.rhs_ptrs, state.nz,
            &solve_info, int(state.block_count));
#else
        const cublasStatus_t status = cublasZgetrsBatched(
            handle, CUBLAS_OP_N, state.nz, 1, state.buffers.scalar.matrix_ptrs, state.nz,
            cuda_utils::device_integer_library_view(state.buffers.scalar.pivots), state.buffers.scalar.rhs_ptrs, state.nz, &solve_info,
            int(state.block_count));
#endif
        if (!cuda_utils::check_cublas(status, "scalar Chan preconditioner",
                                      "resident batched GPU solve") ||
            solve_info != 0 ||
            scalar_preconditioner_rhs_to_fft_wrapper(state.buffers.scalar.rhs,
                                                     state.buffers.scalar.fft_planes, state.nx,
                                                     state.ny, state.nz) != cudaSuccess ||
            !cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.scalar.fft_plan, state.buffers.scalar.fft_planes,
                                                   state.buffers.scalar.fft_planes, CUFFT_FORWARD),
                                     "scalar Chan preconditioner", "forward 2-D FFT") ||
            scalar_preconditioner_scatter_wrapper(
                state.buffers.scalar.fft_planes, dipole_map, state.buffers.scalar.field, state.nx,
                state.ny, state.nz, state.nxm, state.nym, state.nzm,
                CU_REAL(1) / CU_REAL(state.nx * state.ny)) != cudaSuccess)
            return false;
    }
    if (!cuda_utils::check_cuda(
            cuda_utils::tracked_memcpy(field, state.buffers.scalar.field,
                                       field_count * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice),
            "scalar Chan preconditioner", "publish resident preconditioned field"))
        return false;
    return true;
}

//-----------------------------------------------------------
bool build_scalar_preconditioner(const CUCOMPLEX *polarizabilities, int nbsphere,
                                 const CUCOMPLEX *uincx, const CUCOMPLEX *uincy,
                                 const CUCOMPLEX *uincz, const CU_REAL *aretecube,
                                 const CU_REAL *k0, const int *nrig, int nx, int ny, int nz,
                                 int nxm, int nym, int nzm, CUCOMPLEX *chanxxvect,
                                 CUCOMPLEX *sdetnn, int *ipvtnn, bool factorize,
                                 bool polarizabilities_on_device = false) {
    if (!polarizabilities || !uincx || !uincy || !uincz || !aretecube || !k0 || !nrig ||
        (*nrig != 8 && *nrig != 9) || nbsphere <= 0 || !ensure_state(nx, ny, nz, nxm, nym, nzm) ||
        !initialize_cuda_device())
        return false;
    state.factorized = false;
    state.matrices_ready = false;

    cuda_data::ChanPreconditionerBufferSizes buffer_sizes = buffer_sizes_for_state();
    buffer_sizes.scalar_polarizability_count = size_t(nbsphere);
    if (!cuda_data::get_device_chan_preconditioner_buffers(buffer_sizes, &state.buffers))
        return false;
    CUCOMPLEX *device_polarizabilities = state.buffers.scalar.polarizabilities;
    CUCOMPLEX *device_sum = state.buffers.scalar.polarizability_sum;
    CU_INT *device_nonzero = state.buffers.scalar.nonzero_count;
    if (!device_polarizabilities || !device_sum || !device_nonzero ||
        !cuda_utils::check_cuda(
            polarizabilities_on_device
                ? cuda_utils::tracked_memcpy(device_polarizabilities, polarizabilities,
                                             size_t(nbsphere) * sizeof(CUCOMPLEX),
                                             cudaMemcpyDeviceToDevice,
                                             cuda_utils::TransferCategory::Preconditioner)
                : cuda_data::copy_host_to_device(device_polarizabilities, polarizabilities,
                                                 size_t(nbsphere) * sizeof(CUCOMPLEX),
                                                 cuda_utils::TransferCategory::Preconditioner),
            "scalar Chan preconditioner", "copy polarizabilities") ||
        !cuda_utils::check_cuda(cudaMemset(device_sum, 0, sizeof(CUCOMPLEX)),
                                "scalar Chan preconditioner", "zero polarization sum") ||
        !cuda_utils::check_cuda(cudaMemset(device_nonzero, 0, sizeof(int)),
                                "scalar Chan preconditioner", "zero polarization count")) {
        return false;
    }
    if (chan_preconditioner_mean_polarizability_wrapper(
            device_polarizabilities, nbsphere, device_sum, device_nonzero) != cudaSuccess) {
        return false;
    }
    CUCOMPLEX sum = CUFFT_ZERO;
    int nonzero = 0;
    const bool reduced =
        cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(&sum, device_sum, sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Preconditioner),
            "scalar Chan preconditioner", "download polarization sum") &&
        cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(&nonzero, device_nonzero, sizeof(int),
                                           cuda_utils::TransferCategory::Preconditioner),
            "scalar Chan preconditioner", "download polarization count");
    if (!reduced || nonzero == 0)
        return false;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX mean = make_cuFloatComplex(sum.x / CU_REAL(nonzero), sum.y / CU_REAL(nonzero));
#else
    const CUCOMPLEX mean = make_cuDoubleComplex(sum.x / CU_REAL(nonzero), sum.y / CU_REAL(nonzero));
#endif
    if (scalar_chan_build_wrapper(state.buffers.scalar.channel, mean, *uincx, *uincy, *uincz,
                                  *aretecube, *k0, *nrig, nx, ny, nz) != cudaSuccess ||
        !cuda_utils::check_cufft(CUFFT_EXECUTE(state.buffers.scalar.fft_plan, state.buffers.scalar.channel,
                                               state.buffers.scalar.channel, CUFFT_INVERSE),
                                 "scalar Chan preconditioner", "inverse Chan FFT") ||
        scalar_chan_transpose_wrapper(state.buffers.scalar.channel, state.buffers.scalar.vectors,
                                      nx, ny, nz) != cudaSuccess ||
        scalar_chan_build_blocks_wrapper(state.buffers.scalar.vectors,
                                         state.buffers.scalar.matrices, nx, ny, nz) != cudaSuccess)
        return false;
    state.matrices_ready = true;
    if (factorize && !factorize_blocks())
        return false;

    const size_t channel_count = state.mesh_count;
    if (chanxxvect &&
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(chanxxvect, state.buffers.scalar.vectors,
                                           channel_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Preconditioner),
            "scalar Chan preconditioner", "download Chan vectors"))
        return false;
    if (sdetnn) {
        const size_t matrix_count = state.block_count * size_t(nz) * size_t(nz);
        if (!cuda_utils::check_cuda(
                cuda_data::copy_device_to_host(sdetnn, state.buffers.scalar.matrices,
                                               matrix_count * sizeof(CUCOMPLEX),
                                               cuda_utils::TransferCategory::Preconditioner),
                "scalar Chan preconditioner", "download factored blocks"))
            return false;
    }
    if (ipvtnn) {
        const size_t pivot_count = state.block_count * size_t(nz);
        if (!cuda_utils::check_cuda(
                cuda_data::copy_device_to_host(ipvtnn, state.buffers.scalar.pivots,
                                               pivot_count * sizeof(int),
                                               cuda_utils::TransferCategory::Preconditioner),
                "scalar Chan preconditioner", "download block pivots"))
            return false;
    }
    std::fprintf(
        stderr,
        "[scalar Chan preconditioner CUDA] generated and factored %zu blocks of order %d on GPU\n",
        state.block_count, nz);
    return true;
}

const int *identity_dipole_map(size_t count) {
    dense_dipole_map.resize(count);
    for (size_t index = 0; index < count; ++index)
        dense_dipole_map[index] = int(index + 1);
    return dense_dipole_map.data();
}

//-----------------------------------------------------------
void run_dense_scalar_preconditioner(CUCOMPLEX *ff, const CUCOMPLEX *polarisa, const int *nbsphere,
                                     const CUCOMPLEX *uincx, const CUCOMPLEX *uincy,
                                     const CUCOMPLEX *uincz, const int *nx, const int *ny,
                                     const int *nz, const int *nxm, const int *nym, const int *nzm,
                                     const CU_REAL *aretecube, const CU_REAL *k0, const int *ntest,
                                     const int *nrig, CUCOMPLEX *chanxxvect, CUCOMPLEX *sdetnn,
                                     int *ipvtnn, bool solve_blocks, char *infostr, int *nstop,
                                     size_t infostr_len) {
    if (!ff || !polarisa || !nbsphere || !uincx || !uincy || !uincz || !nx || !ny || !nz || !nxm ||
        !nym || !nzm || !aretecube || !k0 || !ntest || !nrig || !nstop) {
        set_error(nstop, infostr, infostr_len, "invalid dense scalar preconditioner arguments", 1);
        return;
    }
    if (*ntest == 0) {
        if (!build_scalar_preconditioner(polarisa, *nbsphere, uincx, uincy, uincz, aretecube, k0,
                                         nrig, *nx, *ny, *nz, *nxm, *nym, *nzm, chanxxvect, sdetnn,
                                         ipvtnn, solve_blocks)) {
            set_error(nstop, infostr, infostr_len,
                      solve_blocks ? "GPU dense scalar Chan setup failed"
                                   : "GPU dense scalar initialization setup failed",
                      2);
            return;
        }
    } else if (*ntest == 1) {
        const size_t map_count = size_t(*nxm) * size_t(*nym) * size_t(*nzm);
        if (!ensure_state(*nx, *ny, *nz, *nxm, *nym, *nzm) ||
            !apply_scalar_preconditioner(ff, identity_dipole_map(map_count), solve_blocks)) {
            set_error(nstop, infostr, infostr_len,
                      solve_blocks ? "GPU dense scalar Chan application failed"
                                   : "GPU dense scalar initialization product failed",
                      3);
            return;
        }
    } else {
        set_error(nstop, infostr, infostr_len, "invalid dense scalar preconditioner stage", 4);
        return;
    }
    *nstop = 0;
    if (infostr && infostr_len)
        std::memset(infostr, ' ', infostr_len);
}

} // namespace

//-----------------------------------------------------------
extern "C" void ifdda_reset_preconditionneurscal_cuda_state() {
    state = {};
    dense_dipole_map.clear();
}

//-----------------------------------------------------------
extern "C" void preconditionneurscal_(
    CUCOMPLEX *ff, CUCOMPLEX *xi, CUCOMPLEX *xr, const int *tabdip, const CUCOMPLEX *polarisa,
    const int *niso, CUCOMPLEX *chanxxvect, const CUCOMPLEX *uincx, const CUCOMPLEX *uincy,
    const CUCOMPLEX *uincz, const int *nx, const int *ny, const int *nz, const int *nbsphere,
    const int *ndipole, const int *nxm, const int *nym, const int *nzm, const CU_REAL *aretecube,
    const CU_REAL *k0, const int *ntest, CUCOMPLEX *sdetnn, int *ipvtnn, CUCOMPLEX *vectxx,
    const long long *plan2b, const long long *plan2f, const int *nrig, char *infostr, int *nstop,
    size_t infostr_len) {
    (void)xi;
    (void)xr;
    (void)niso;
    (void)ndipole;
    (void)vectxx;
    (void)plan2b;
    (void)plan2f;
    if (!ff || !tabdip || !polarisa || !uincx || !uincy || !uincz || !nx || !ny || !nz ||
        !nbsphere || !nxm || !nym || !nzm || !aretecube || !k0 || !ntest || !nstop) {
        set_error(nstop, infostr, infostr_len, "invalid scalar preconditioner arguments", 1);
        return;
    }
    if (*ntest == 0) {
        if (!build_scalar_preconditioner(polarisa, *nbsphere, uincx, uincy, uincz, aretecube, k0,
                                         nrig, *nx, *ny, *nz, *nxm, *nym, *nzm, chanxxvect, sdetnn,
                                         ipvtnn, true)) {
            set_error(nstop, infostr, infostr_len, "GPU scalar Chan setup failed", 2);
            return;
        }
    } else if (*ntest == 1) {
        if (!apply_scalar_preconditioner(ff, tabdip, true)) {
            set_error(nstop, infostr, infostr_len, "GPU scalar Chan application failed", 3);
            return;
        }
    } else {
        set_error(nstop, infostr, infostr_len, "invalid scalar preconditioner stage", 4);
        return;
    }
    *nstop = 0;
    if (infostr && infostr_len)
        std::memset(infostr, ' ', infostr_len);
}

//-----------------------------------------------------------
extern "C" void preconditionneurinitscal2_(
    CUCOMPLEX *ff, CUCOMPLEX *xi, CUCOMPLEX *xr, const int *tabdip, const CUCOMPLEX *polarisa,
    const int *niso, CUCOMPLEX *chanxxvect, const CUCOMPLEX *uincx, const CUCOMPLEX *uincy,
    const CUCOMPLEX *uincz, const int *nx, const int *ny, const int *nz, const int *nbsphere,
    const int *nxm, const int *nym, const int *nzm, const CU_REAL *aretecube, const CU_REAL *k0,
    const int *ntest, CUCOMPLEX *sdetnn, CUCOMPLEX *vectxx, const long long *plan2b,
    const long long *plan2f, const int *nrig, char *infostr, int *nstop, size_t infostr_len) {
    (void)xi;
    (void)xr;
    (void)niso;
    (void)vectxx;
    (void)plan2b;
    (void)plan2f;
    if (!ff || !tabdip || !polarisa || !uincx || !uincy || !uincz || !nx || !ny || !nz ||
        !nbsphere || !nxm || !nym || !nzm || !aretecube || !k0 || !ntest || !nrig || !nstop) {
        set_error(nstop, infostr, infostr_len, "invalid scalar initialization arguments", 1);
        return;
    }
    if (*ntest == 0) {
        if (!build_scalar_preconditioner(polarisa, *nbsphere, uincx, uincy, uincz, aretecube, k0,
                                         nrig, *nx, *ny, *nz, *nxm, *nym, *nzm, chanxxvect, sdetnn,
                                         nullptr, false)) {
            set_error(nstop, infostr, infostr_len, "GPU scalar initialization setup failed", 2);
            return;
        }
    } else if (*ntest == 1) {
        if (!apply_scalar_preconditioner(ff, tabdip, false)) {
            set_error(nstop, infostr, infostr_len, "GPU scalar initialization product failed", 3);
            return;
        }
    } else {
        set_error(nstop, infostr, infostr_len, "invalid scalar initialization stage", 4);
        return;
    }
    *nstop = 0;
    if (infostr && infostr_len)
        std::memset(infostr, ' ', infostr_len);
}

//-----------------------------------------------------------
extern "C" void preconditionneurscalopt_(
    CUCOMPLEX *ff, CUCOMPLEX *xi, CUCOMPLEX *xr, const CUCOMPLEX *polarisa, const int *niso,
    CUCOMPLEX *chanxxvect, const CUCOMPLEX *uincx, const CUCOMPLEX *uincy, const CUCOMPLEX *uincz,
    const int *nx, const int *ny, const int *nz, const int *nbsphere, const int *nxm,
    const int *nym, const int *nzm, const CU_REAL *aretecube, const CU_REAL *k0, const int *ntest,
    CUCOMPLEX *sdetnn, int *ipvtnn, CUCOMPLEX *vectxx, const long long *plan2b,
    const long long *plan2f, const int *nrig, char *infostr, int *nstop, size_t infostr_len) {
    (void)xi;
    (void)xr;
    (void)niso;
    (void)vectxx;
    (void)plan2b;
    (void)plan2f;
    run_dense_scalar_preconditioner(ff, polarisa, nbsphere, uincx, uincy, uincz, nx, ny, nz, nxm,
                                    nym, nzm, aretecube, k0, ntest, nrig, chanxxvect, sdetnn,
                                    ipvtnn, true, infostr, nstop, infostr_len);
}

//-----------------------------------------------------------
extern "C" void preconditionneurinitscal_(
    CUCOMPLEX *ff, CUCOMPLEX *xi, CUCOMPLEX *xr, const CUCOMPLEX *polarisa, const int *niso,
    CUCOMPLEX *chanxxvect, const CUCOMPLEX *uincx, const CUCOMPLEX *uincy, const CUCOMPLEX *uincz,
    const int *nx, const int *ny, const int *nz, const int *nbsphere, const int *nxm,
    const int *nym, const int *nzm, const CU_REAL *aretecube, const CU_REAL *k0, const int *ntest,
    CUCOMPLEX *sdetnn, CUCOMPLEX *vectxx, const long long *plan2b, const long long *plan2f,
    const int *nrig, char *infostr, int *nstop, size_t infostr_len) {
    (void)xi;
    (void)xr;
    (void)niso;
    (void)vectxx;
    (void)plan2b;
    (void)plan2f;
    run_dense_scalar_preconditioner(ff, polarisa, nbsphere, uincx, uincy, uincz, nx, ny, nz, nxm,
                                    nym, nzm, aretecube, k0, ntest, nrig, chanxxvect, sdetnn,
                                    nullptr, false, infostr, nstop, infostr_len);
}

//-----------------------------------------------------------
extern "C" int ifdda_scalar_chan_prepare_resident(const CUCOMPLEX *polarizabilities, int nbsphere,
                                                  const CUCOMPLEX *uincx, const CUCOMPLEX *uincy,
                                                  const CUCOMPLEX *uincz, CU_REAL aretecube,
                                                  CU_REAL k0, int nrig, int nx, int ny, int nz,
                                                  int nxm, int nym, int nzm, int factorize) {
    return build_scalar_preconditioner(polarizabilities, nbsphere, uincx, uincy, uincz, &aretecube,
                                       &k0, &nrig, nx, ny, nz, nxm, nym, nzm, nullptr, nullptr,
                                       nullptr, factorize != 0, true)
               ? 1
               : 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_scalar_chan_apply_resident(CUCOMPLEX *field, int field_count,
                                                const int *tabdip, int solve_blocks) {
    return apply_scalar_preconditioner_resident(field, size_t(field_count), tabdip,
                                                solve_blocks != 0)
               ? 1
               : 0;
}
