#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using HostComplex = std::complex<CU_REAL>;

namespace {

thread_local bool keep_lens_result_on_device = false;
thread_local const CUCOMPLEX* resident_lens_input = nullptr;
thread_local CUCOMPLEX* last_lens_result_on_device = nullptr;

//-----------------------------------------------------------
void set_error(int* nstop, char* infostr, size_t length, const char* message) {
    if (nstop)
        *nstop = 1;
    if (!infostr || length == 0)
        return;
    std::memset(infostr, ' ', length);
    std::memcpy(infostr, message, std::min(length, std::strlen(message)));
}

//-----------------------------------------------------------
} // namespace

//-----------------------------------------------------------
extern "C" void diffractefft2dlens_set_resident_output(int enabled) {
    keep_lens_result_on_device = enabled != 0;
    if (enabled)
        last_lens_result_on_device = nullptr;
}

//-----------------------------------------------------------
extern "C" int ifdda_quicklens_direct_cuda_(
    const COMPLEX* field, const REAL* xs, const REAL* ys, const REAL* zs,
    const COMPLEX* incident, int dipole_count, int nfft, int half, int imax,
    REAL dkx, REAL k0, REAL na, int side, REAL zlens,
    COMPLEX* fx, COMPLEX* fy, COMPLEX* fz,
    COMPLEX* fix, COMPLEX* fiy, COMPLEX* fiz) {
    if (!field || !xs || !ys || !zs || !incident || !fx || !fy || !fz ||
        !fix || !fiy || !fiz || dipole_count <= 0 || nfft <= 0 || imax < 0 ||
        2LL * imax + 1 > nfft || k0 <= REAL(0) || dkx <= REAL(0)) return 1;
    size_t plane=0, field_count=0;
    if (!cuda_utils::checked_multiply(size_t(nfft),size_t(nfft),&plane) ||
        !cuda_utils::checked_multiply(size_t(dipole_count),size_t(3),&field_count) ||
        plane > size_t(std::numeric_limits<int>::max())/3 ||
        field_count > std::numeric_limits<size_t>::max()/sizeof(CUCOMPLEX)) return 1;
    if (!initialize_cuda_device()) return 1;
    cuda_data::LensDiffractionDeviceBuffers buffers{};
    if (!cuda_data::get_device_lens_diffraction_buffers(plane,size_t(dipole_count),&buffers)) return 1;
    const size_t plane_bytes=plane*3*sizeof(CUCOMPLEX);
    std::vector<std::complex<CU_REAL>> host_field(field_count),host_incident(plane*3);
    for(size_t q=0;q<field_count;++q) host_field[q]=field[q];
    for(size_t q=0;q<plane*3;++q) host_incident[q]=incident[q];
    const CUCOMPLEX* device_field=get_device_result_vector_alias(field,field_count);
    bool ok=true;
    if(device_field) {
        std::printf("CUDA_ALIAS_REUSE: quicklens direct field (%zu complex values)\n",field_count);
    } else {
        ok=cuda_utils::check_cuda(cuda_data::copy_host_complex_to_device(
            buffers.green_field,host_field.data(),field_count,cuda_utils::TransferCategory::SolverVector),
            "Quick Lens direct field upload", "cudaMemcpy HostToDevice");
        device_field=buffers.green_field;
    }
    ok=ok &&
        cuda_utils::check_cuda(cuda_data::copy_host_to_device(
        buffers.incident_spectrum,host_incident.data(),plane_bytes,cuda_utils::TransferCategory::IncidentField),
        "Quick Lens incident upload", "cudaMemcpy HostToDevice") &&
        cuda_utils::check_cuda(cuda_data::copy_host_to_device(
        buffers.dipole_x,xs,size_t(dipole_count)*sizeof(REAL),cuda_utils::TransferCategory::Geometry),
        "Quick Lens x coordinate upload", "cudaMemcpy HostToDevice") &&
        cuda_utils::check_cuda(cuda_data::copy_host_to_device(
        buffers.dipole_y,ys,size_t(dipole_count)*sizeof(REAL),cuda_utils::TransferCategory::Geometry),
        "Quick Lens y coordinate upload", "cudaMemcpy HostToDevice") &&
        cuda_utils::check_cuda(cuda_data::copy_host_to_device(
        buffers.dipole_z,zs,size_t(dipole_count)*sizeof(REAL),cuda_utils::TransferCategory::Geometry),
        "Quick Lens z coordinate upload", "cudaMemcpy HostToDevice");
    if (!ok) return 1;
    if (!cuda_utils::check_cuda(quicklens_direct_sum_wrapper(
        device_field,buffers.dipole_x,buffers.dipole_y,buffers.dipole_z,
        buffers.incident_spectrum,buffers.accumulated_spectrum,buffers.layer_spectrum,
        dipole_count,nfft,half,imax,dkx,k0,na,side,zlens),"Quick Lens direct sum", "kernel launch")) return 1;
    std::vector<std::complex<CU_REAL>> host_scattered(plane*3),host_combined(plane*3);
    if (!cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
        host_scattered.data(),buffers.accumulated_spectrum,plane*3,cuda_utils::TransferCategory::Result),
        "Quick Lens scattered spectrum download", "cudaMemcpy DeviceToHost") ||
        !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
        host_combined.data(),buffers.layer_spectrum,plane*3,cuda_utils::TransferCategory::Result),
        "Quick Lens combined spectrum download", "cudaMemcpy DeviceToHost")) return 1;
    for(size_t q=0;q<plane;++q) {
        fx[q]=host_scattered[q];
        fy[q]=host_scattered[plane+q];
        fz[q]=host_scattered[2*plane+q];
        fix[q]=host_combined[q];
        fiy[q]=host_combined[plane+q];
        fiz[q]=host_combined[2*plane+q];
    }
    return 0;
}

//-----------------------------------------------------------
extern "C" void diffractefft2dlens_set_resident_input(
    const CUCOMPLEX* device_input) {
    resident_lens_input = device_input;
}

//-----------------------------------------------------------
extern "C" const CUCOMPLEX* diffractefft2dlens_get_resident_output() {
    return last_lens_result_on_device;
}

//-----------------------------------------------------------
extern "C" void diffractefft2dlens_(
    const int* nx, const int* ny, const int* nz, const int* nxm, const int* nym,
    const int* nzm, const int* nfft2d, int* tabfft2, const CU_REAL* k0,
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, const CU_REAL* aretecube,
    HostComplex* eloinx, HostComplex* eloiny, HostComplex* eloinz,
    const HostComplex* ff, int* imax, CU_REAL* deltakx, CU_REAL* deltaky,
    HostComplex* ediffkzpos, const CU_REAL* numaper, const int* nside,
    const IFDDA_PLAN_INT64* plan2f, const IFDDA_PLAN_INT64* plan2b, int* nstop,
    char infostr[64], size_t infostr_length) {
    (void)plan2f;
    (void)plan2b;
    if (nstop)
        *nstop = 0;
    last_lens_result_on_device = nullptr;
    if (infostr && infostr_length)
        std::memset(infostr, ' ', infostr_length);
    if (!nx || !ny || !nz || !nxm || !nym || !nzm || !nfft2d || !tabfft2 || !k0 ||
        !xs || !ys || !zs || !aretecube || !eloinx || !eloiny || !eloinz || !ff ||
        !imax || !deltakx || !deltaky || !ediffkzpos || !numaper || !nside || !nstop ||
        *nx <= 0 || *ny <= 0 || *nz <= 0 || *nxm < *nx || *nym < *ny || *nzm < *nz ||
        *nfft2d <= 0 || *nfft2d > 16384 || *k0 <= CU_REAL(0) || *aretecube <= CU_REAL(0) ||
        *numaper <= CU_REAL(0)) {
        set_error(nstop, infostr, infostr_length, "Invalid lens diffraction arguments");
        return;
    }

    size_t mesh_count = 0;
    size_t plane_size = 0;
    size_t total_mesh = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx), size_t(*ny), &mesh_count) ||
        !cuda_utils::checked_multiply(mesh_count, size_t(*nz), &total_mesh) ||
        !cuda_utils::checked_multiply(size_t(*nfft2d), size_t(*nfft2d), &plane_size) ||
        plane_size > size_t(std::numeric_limits<int>::max()) / 3 ||
        total_mesh > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX))) {
        set_error(nstop, infostr, infostr_length, "Lens diffraction dimensions overflow");
        return;
    }

    const CU_REAL pi = CU_REAL(3.1415926535897932384626433832795L);
    const int half = *nfft2d / 2;
    *deltakx = CU_REAL(2) * pi / (CU_REAL(*nfft2d) * *aretecube);
    *deltaky = *deltakx;
    if (*deltakx >= *numaper) {
        set_error(nstop, infostr, infostr_length, "In FFT lens nfft2d too small");
        return;
    }
    *imax = int(std::floor(*k0 / *deltakx + CU_REAL(0.5))) + 1;
    if (2LL * *imax + 1 > *nfft2d) {
        set_error(nstop, infostr, infostr_length,
                  "In FFT diffraction for lens nfft2d too small!");
        return;
    }
    if (2 * *imax + 1 < 7) {
        set_error(nstop, infostr, infostr_length, "In FFT diffract nfft2d too small");
        return;
    }

    std::vector<int> host_fft_indices(static_cast<size_t>(*nfft2d));
    for (int i = 0; i < *nfft2d; ++i)
        host_fft_indices[size_t(i)] = i - half >= 0 ? i - half : half + i;
    for (int i = 0; i < *nfft2d; ++i)
        tabfft2[i] = host_fft_indices[size_t(i)] + 1;

    std::vector<CUCOMPLEX> host_green;
    if (!resident_lens_input) {
        host_green.resize(total_mesh * 3);
        for (size_t i = 0; i < total_mesh * 3; ++i)
            host_green[i] = cuda_utils::device_value(ff[i]);
    }

    if (!initialize_cuda_device()) {
        set_error(nstop, infostr, infostr_length, "CUDA device initialization failed");
        return;
    }
    cuda_data::LensDiffractionDeviceBuffers buffers{};
    if (!cuda_data::get_device_lens_diffraction_buffers(plane_size, total_mesh, &buffers)) {
        set_error(nstop, infostr, infostr_length, "Could not acquire lens diffraction buffers");
        return;
    }

    const size_t spectrum_bytes = plane_size * 3 * sizeof(CUCOMPLEX);
    const size_t green_bytes = total_mesh * 3 * sizeof(CUCOMPLEX);
    bool ok = resident_lens_input
        ? cuda_utils::check_cuda(cuda_data::copy_device_to_device(
              buffers.green_field, resident_lens_input, green_bytes),
              "lens diffraction", "reuse resident scattered field")
        : cuda_utils::check_cuda(
              cuda_data::copy_host_to_device(buffers.green_field, host_green.data(), green_bytes,
                                             cuda_utils::TransferCategory::SolverVector),
              "lens diffraction", "copy Green field to GPU");
    ok = ok &&
        cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.fft_indices, host_fft_indices.data(),
                                           size_t(*nfft2d) * sizeof(int),
                                           cuda_utils::TransferCategory::Control),
            "lens diffraction", "copy FFT index map to GPU") &&
        cuda_utils::check_cuda(
            lens_diffraction_clear_wrapper(buffers.accumulated_spectrum,
                                           int(3 * plane_size)),
            "lens diffraction", "clear output spectrum");
    if (!ok) {
        set_error(nstop, infostr, infostr_length, "CUDA lens diffraction setup failed");
        return;
    }

    const CU_REAL direction = CU_REAL(*nside);
    const CU_REAL var1 = (xs[0] + CU_REAL(half) * *aretecube) * *deltakx;
    const CU_REAL var2 = (ys[0] + CU_REAL(half) * *aretecube) * *deltaky;
    for (int layer = 0; layer < *nz; ++layer) {
        if (!cuda_utils::check_cuda(
                lens_diffraction_clear_wrapper(buffers.layer_spectrum,
                    int(3 * plane_size)),
                "lens diffraction", "clear layer FFT spectrum") ||
            !cuda_utils::check_cuda(
                lens_diffraction_pack_layer_wrapper(buffers.green_field,
                    buffers.layer_spectrum, buffers.fft_indices, *nx, *ny, *nfft2d, layer),
                "lens diffraction", "pack Green field layer") ||
            !cuda_utils::check_cufft(
                CUFFT_EXECUTE(buffers.fft_plan, buffers.layer_spectrum,
                              buffers.layer_spectrum, CUFFT_FORWARD),
                "lens diffraction", "transform Green field layer") ||
            !cuda_utils::check_cuda(
                lens_diffraction_accumulate_wrapper(buffers.layer_spectrum,
                    buffers.accumulated_spectrum, buffers.fft_indices, *nfft2d, *imax,
                    *deltakx, *deltaky, *k0, *numaper, direction, var1, var2,
                    zs[size_t(layer) * mesh_count]),
                "lens diffraction", "accumulate far field")) {
            set_error(nstop, infostr, infostr_length, "CUDA lens diffraction failed");
            return;
        }
    }

    last_lens_result_on_device = buffers.accumulated_spectrum;
    if (!keep_lens_result_on_device) {
        std::vector<CUCOMPLEX> host_output(plane_size * 3);
        if (!cuda_utils::check_cuda(
                cuda_data::copy_device_to_host(host_output.data(), buffers.accumulated_spectrum,
                    spectrum_bytes, cuda_utils::TransferCategory::Result),
                "lens diffraction", "copy diffraction spectrum to CPU")) {
            set_error(nstop, infostr, infostr_length,
                      "Could not copy diffraction result to CPU");
            return;
        }
        for (size_t i = 0; i < plane_size * 3; ++i)
            ediffkzpos[i] = cuda_utils::host_value(host_output[i]);

        // Preserve the Fortran routine's inout workspace contract.
        std::vector<CUCOMPLEX> host_last_layer(plane_size * 3);
        if (!cuda_utils::check_cuda(
                cuda_data::copy_device_to_host(host_last_layer.data(), buffers.layer_spectrum,
                    spectrum_bytes, cuda_utils::TransferCategory::Result),
                "lens diffraction", "copy final FFT workspace")) {
            set_error(nstop, infostr, infostr_length,
                      "Could not copy final FFT workspace to CPU");
            return;
        }
        for (size_t i = 0; i < plane_size; ++i) {
            eloinx[i] = cuda_utils::host_value(host_last_layer[i]);
            eloiny[i] = cuda_utils::host_value(host_last_layer[plane_size + i]);
            eloinz[i] = cuda_utils::host_value(host_last_layer[2 * plane_size + i]);
        }
    }
}
