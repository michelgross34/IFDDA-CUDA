#include "cdmlib_cpp_abi.hpp"
#include "cuda_data.h"
#define IFDDA_CUDA_CALLERS_CALCULWIDEFIELD_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_CALCULWIDEFIELD_CUDA
#include "widefield_CUDA.h"
#include "cuda_utils.h"
#include "kernel.h"

#undef USE_FFTW // Wide-field plans are owned by the CUDA/cuFFT backend.
/* C99 translation of cdmlib/calculwidefield.f90.
 * The original Fortran source remains in cdmlib as the scientific reference.
 */

#include <cmath>
#include <algorithm>
#include <cstring>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#define I (COMPLEX(0.0, 1.0))
#ifdef USE_FFTW
#include "cdmlib_cpp_fftw_precision.hpp"
#endif
























static int has_prefix(const char *s, size_t n, const char *prefix, size_t m)
{
    return n >= m && memcmp(s, prefix, m) == 0;
}

static FILE *open_optional_output(const char *name)
{
    FILE *f = fopen(name, "w");
    return f;
}

//-----------------------------------------------------------
extern "C" void ondeplanemulti_(REAL *x, REAL *y, REAL *z, REAL *k0,
    const COMPLEX *amplitudes, const REAL *s_polarization,
    const REAL *p_polarization, const REAL *theta_degrees,
    const REAL *phi_degrees, const int *wave_count,
    COMPLEX *ex, COMPLEX *ey, COMPLEX *ez, int *nstop,
    char *infostr, size_t infostr_len)
{
    if (!x || !y || !z || !k0 || !amplitudes || !s_polarization ||
        !p_polarization || !theta_degrees || !phi_degrees || !wave_count ||
        !ex || !ey || !ez || !nstop || !infostr || *wave_count < 0 ||
        *wave_count > 10) {
        if (nstop)
            *nstop = 1;
        if (infostr && infostr_len)
            std::memset(infostr, ' ', infostr_len);
        return;
    }

    std::vector<COMPLEX> host_amplitudes(amplitudes, amplitudes + 3 * size_t(*wave_count));

    const int point_count = 1;
    COMPLEX output[3]{};
    const COMPLEX zero_amplitude{};
    int status = 1;
    ifdda_multi_plane_wave_field_cuda_(
        x, y, z, &point_count, k0, &zero_amplitude, host_amplitudes.data(),
        s_polarization, p_polarization, theta_degrees, phi_degrees,
        wave_count, output, &status);
    *nstop = status;
    if (status == 0) {
        *ex = output[0];
        *ey = output[1];
        *ez = output[2];
        return;
    }
    if (infostr_len) {
        const char message[] = "CUDA multi-plane-wave calculation failed";
        std::memset(infostr, ' ', infostr_len);
        std::memcpy(infostr, message,
            std::min(infostr_len, sizeof(message) - 1));
    }
}

//-------------------------------------------
extern "C" void calculwidefield_(
    int *nmat, int *nrig, int *nx, int *ny, int *nz, int *nx2, int *ny2,
    int *nz2, int *nbsphere, int *subunit, int *nmaxpp, int *nxmp, int *nymp,
    int *nzmp, int *nquad, int *nxm, int *nym, int *nzm, int *nfft2d,
    int *ntotalm, int *nmax, int *speckseed, REAL *irra, REAL *numaperil,
    REAL *aretecube, REAL *k0, REAL *lambda, REAL *theta, REAL *phi,
    REAL *pp, REAL *ss, REAL *P0, REAL *w0, REAL *xdip,
    REAL *ydip, REAL *zdip, REAL *xgaus, REAL *ygaus, REAL *zgaus,
    REAL *tolinit, COMPLEX *E0, COMPLEX *E0m, REAL *ssm,
    REAL *ppm, REAL *thetam, REAL *phim, int *nbinc, REAL *xs,
    REAL *ys, REAL *zs, REAL *xswf, REAL *yswf, REAL *zswf,
    COMPLEX *xi, COMPLEX *xr, COMPLEX *FFloc,
    COMPLEX *FF, COMPLEX *FF0, COMPLEX *incidentfieldx,
    COMPLEX *incidentfieldy, COMPLEX *incidentfieldz,
    REAL *incidentfield, COMPLEX *Eimagex, COMPLEX *Eimagey,
    COMPLEX *Eimagez, COMPLEX *polarisa, int *niso,
    COMPLEX *epsilon, char *beam, char *infostr, char *namefileinc,
    COMPLEX *FFTTENSORxx, COMPLEX *FFTTENSORxy,
    COMPLEX *FFTTENSORxz, COMPLEX *FFTTENSORyy,
    COMPLEX *FFTTENSORyz, COMPLEX *FFTTENSORzz,
    int64_t *plan2f, int64_t *plan2b, int64_t *group_idnf, int *nstop,
    size_t beam_len, size_t infostr_len, size_t namefileinc_len)
{
    (void)infostr_len;
    const int nxmpp = *nx + 2 * *nxmp;
    const int nympp = *ny + 2 * *nymp;
    const int nzmpp = *nz + 2 * *nzmp;
    const size_t ngrid = (size_t)nxmpp * (size_t)nympp * (size_t)nzmpp;
    const int nxm2 = 2 * nxmpp, nym2 = 2 * nympp, nzm2 = 2 * nzmpp;
    const int nxym2 = nxm2 * nym2;
    const size_t nfft = (size_t)*nfft2d * (size_t)*nfft2d;
    const COMPLEX one = 1.0 + 0.0 * I;
    const int special_beam = has_prefix(beam, beam_len, "gfftwave", 8) ||
        has_prefix(beam, beam_len, "speckle", 7) ||
        has_prefix(beam, beam_len, "demispeckle", 11) ||
        has_prefix(beam, beam_len, "confocal", 8) ||
        has_prefix(beam, beam_len, "demiconfocal", 12);
    bool incident_grid_is_resident = false;
    const bool trace_widefield = std::getenv("IFDDA_TRACE_WIDEFIELD") != nullptr;
    REAL xmin = HUGE_VAL;
    REAL ymin = HUGE_VAL;
    REAL zmin = HUGE_VAL;
    COMPLEX eps0 = one;
    int i, j, k, sub = 0;
    int ntotal;
    int64_t planfn = 0, planbn = 0;
#ifdef USE_FFTW
    cdmlib_fftw_plan plan_forward = NULL, plan_backward = NULL;
#endif
    FILE *fx = NULL, *fy = NULL, *fz = NULL;
    FILE *fix = NULL, *fiy = NULL, *fiz = NULL, *fim = NULL;
    const size_t nmaxpp_size = ngrid;

    (void)nx2; (void)ny2; (void)nz2; (void)nmax; (void)xi; (void)polarisa;
    (void)Eimagex;
    (void)Eimagey; (void)Eimagez;
    *nmaxpp = (int)nmaxpp_size;
#ifdef USE_FFTW
    {
        const unsigned flags = ngrid <= 1000000 ? FFTW_ESTIMATE : FFTW_MEASURE;
        plan_backward = CDMLIB_FFTW_PLAN_DFT_3D(nzm2, nym2, nxm2,
            nullptr, nullptr, FFTW_BACKWARD, flags);
        plan_forward = CDMLIB_FFTW_PLAN_DFT_3D(nzm2, nym2, nxm2,
            nullptr, nullptr, FFTW_FORWARD, flags);
        planbn = (int64_t)(uintptr_t)plan_backward;
        planfn = (int64_t)(uintptr_t)plan_forward;
    }
#endif

    const void* tensor_keys[6] = {
        FFTTENSORxx, FFTTENSORxy, FFTTENSORxz,
        FFTTENSORyy, FFTTENSORyz, FFTTENSORzz};
    GreenTensorDevicePointers device_green{};
    const int green_status = ifdda_widefield_generate_green_c99(
        tensor_keys, nxmpp, nympp, nzmpp, *nxm, *nym, *nzm,
        nxm2, nym2, nzm2, *aretecube, *k0, *nquad, &device_green);
    if (green_status != 0) {
        *nstop = 1;
        if (infostr_len) snprintf(infostr, infostr_len,
            "CUDA wide-field Green tensor generation failed");
        return;
    }

    for (i = 0; i < *nbsphere; ++i) {
        if (xs[i] < xmin) xmin = xs[i];
        if (ys[i] < ymin) ymin = ys[i];
        if (zs[i] < zmin) zmin = zs[i];
    }

    cuda_data::WideFieldDeviceBuffers widefield_buffers{};
    const int grid_status = ifdda_widefield_prepare_grid_c99(
        xs, ys, zs, *nbsphere, nxmpp, nympp, nzmpp,
        *nxmp, *nymp, *nzmp, static_cast<CU_REAL>(xmin),
        static_cast<CU_REAL>(ymin), static_cast<CU_REAL>(zmin),
        static_cast<CU_REAL>(*aretecube),
        static_cast<CU_REAL>(*lambda / REAL(1000)), &widefield_buffers);
    if (grid_status != 0) {
        *nstop = 1;
        if (infostr_len) {
            constexpr char message[] = "CUDA wide-field grid setup failed";
            std::memset(infostr, ' ', infostr_len);
            std::memcpy(infostr, message,
                std::min(infostr_len, sizeof(message) - 1));
        }
        return;
    }

    if (*nmat == 2) {
        int dims[4] = {nzmpp, *nmax, 1, 1};
        for (i = 0; i < nzmpp; ++i)
            zswf[i] = zmin + (REAL)(i - *nzmp) * *aretecube;
        for (i = 0; i < nympp; ++i)
            yswf[i] = ymin + (REAL)(i - *nymp) * *aretecube;
        for (i = 0; i < nxmpp; ++i)
            xswf[i] = xmin + (REAL)(i - *nxmp) * *aretecube;
        hdf5write1d_i8_(group_idnf, "zwf", zswf, dims, 3);
        dims[0] = nympp;
        hdf5write1d_i8_(group_idnf, "ywf", yswf, dims, 3);
        dims[0] = nxmpp;
        hdf5write1d_i8_(group_idnf, "xwf", xswf, dims, 3);
    }

    if (special_beam) {
        if (has_prefix(beam, beam_len, "gfftwave", 8)) {
            const CU_REAL grid_origin_x = static_cast<CU_REAL>(xmin) -
                CU_REAL(*nxmp) * static_cast<CU_REAL>(*aretecube);
            const CU_REAL grid_origin_y = static_cast<CU_REAL>(ymin) -
                CU_REAL(*nymp) * static_cast<CU_REAL>(*aretecube);
            const CU_REAL grid_origin_z = static_cast<CU_REAL>(zmin) -
                CU_REAL(*nzmp) * static_cast<CU_REAL>(*aretecube);
            const int circular = has_prefix(beam, beam_len, "gfftwavecircular", 16) ? 1 : 0;
            const COMPLEX e0_cuda = *E0;
            CUCOMPLEX* resident_gaussian_grid = nullptr;
            int gaussian_status = 1;
            ifdda_gaussianfft_cuda_resident_grid_origin_(
                nfft2d, &nxmpp, &nympp, &nzmpp, nmax, nmaxpp,
                widefield_buffers.grid_x, widefield_buffers.grid_y,
                &grid_origin_x, &grid_origin_y, &grid_origin_z,
                aretecube, xgaus, ygaus, zgaus, theta, phi, w0, k0,
                ss, pp, &e0_cuda, tolinit, &circular,
                &resident_gaussian_grid, &gaussian_status);
            if (gaussian_status == 0 && resident_gaussian_grid &&
                widefield_buffers.scattered_field) {
                const cudaError_t copy_status = cuda_data::copy_device_to_device(
                    widefield_buffers.scattered_field, resident_gaussian_grid,
                    size_t(3) * ngrid * sizeof(CUCOMPLEX));
                incident_grid_is_resident = copy_status == cudaSuccess;
                if (!incident_grid_is_resident)
                    gaussian_status = static_cast<int>(copy_status);
            }
            if (gaussian_status != 0 || !incident_grid_is_resident) {
                *nstop = 1;
                if (infostr_len) {
                    constexpr char message[] = "CUDA Gaussian FFT illumination failed";
                    std::memset(infostr, ' ', infostr_len);
                    std::memcpy(infostr, message,
                        std::min(infostr_len, sizeof(message) - 1));
                }
                return;
            }
        } else {
            const bool is_speckle = has_prefix(beam, beam_len, "speckle", 7) ||
                has_prefix(beam, beam_len, "demispeckle", 11);
            const bool bidirectional = has_prefix(beam, beam_len, "speckle", 7) ||
                has_prefix(beam, beam_len, "confocal", 8);
            const int dims[3] = {nxmpp, nympp, nzmpp};
            const int count = static_cast<int>(ngrid);
            const int nfft_value = *nfft2d;
            COMPLEX e0_cuda = *E0;
            CU_REAL irradiance_cuda = static_cast<CU_REAL>(*irra);
            int status = 1;
            CUCOMPLEX* resident_incident_grid = nullptr;
            const CU_REAL k0_cuda = static_cast<CU_REAL>(*k0);
            const CU_REAL x0_cuda = static_cast<CU_REAL>(*xgaus);
            const CU_REAL y0_cuda = static_cast<CU_REAL>(*ygaus);
            const CU_REAL z0_cuda = static_cast<CU_REAL>(*zgaus);
            const CU_REAL psi_cuda = static_cast<CU_REAL>(*pp);
            const CU_REAL spacing_cuda = static_cast<CU_REAL>(*aretecube);
            const CU_REAL power_cuda = static_cast<CU_REAL>(*P0);
            const CU_REAL aperture_cuda = static_cast<CU_REAL>(*numaperil);
            const int bidirectional_flag = bidirectional ? 1 : 0;
            const CU_REAL grid_origin_x = static_cast<CU_REAL>(xmin) -
                CU_REAL(*nxmp) * static_cast<CU_REAL>(*aretecube);
            const CU_REAL grid_origin_y = static_cast<CU_REAL>(ymin) -
                CU_REAL(*nymp) * static_cast<CU_REAL>(*aretecube);
            const CU_REAL grid_origin_z = static_cast<CU_REAL>(zmin) -
                CU_REAL(*nzmp) * static_cast<CU_REAL>(*aretecube);

            if (is_speckle) {
                const CUCOMPLEX *phase_positive_device = nullptr;
                const CUCOMPLEX *phase_negative_device = nullptr;
                const int seed = *speckseed;
                const int phase_bidirectional = bidirectional ? 1 : 0;
                status = ifdda_prepare_speckle_phases_device_(
                    &nfft_value, &nfft_value, &k0_cuda, &aperture_cuda,
                    &spacing_cuda, &seed, &phase_bidirectional,
                    &phase_positive_device, &phase_negative_device);
                if (status == 0)
                    ifdda_speckle_field_cuda_resident_grid_origin_(dims, dims + 1, dims + 2, &count,
                    &nfft_value, &k0_cuda, &x0_cuda, &y0_cuda, &z0_cuda,
                    &psi_cuda, &spacing_cuda, &power_cuda, &aperture_cuda,
                    &e0_cuda, phase_positive_device, phase_negative_device,
                    &grid_origin_x, &grid_origin_y, &grid_origin_z,
                    &bidirectional_flag, &resident_incident_grid,
                    &irradiance_cuda, &status);
                if (status == 0 && widefield_buffers.scattered_field) {
                    status = cuda_data::copy_device_to_device(
                        widefield_buffers.scattered_field, resident_incident_grid,
                        size_t(3) * ngrid * sizeof(CUCOMPLEX));
                    incident_grid_is_resident = status == cudaSuccess;
                    if (status != cudaSuccess)
                        status = 1;
                }
            } else {
                ifdda_confocal_field_cuda_resident_grid_origin_(dims, dims + 1, dims + 2, &count,
                    &nfft_value, &k0_cuda, &x0_cuda, &y0_cuda, &z0_cuda,
                    &psi_cuda, &spacing_cuda, &power_cuda, &aperture_cuda,
                    &e0_cuda, &grid_origin_x, &grid_origin_y, &grid_origin_z,
                    &bidirectional_flag,
                    &resident_incident_grid, &irradiance_cuda, &status);
                if (status == 0 && widefield_buffers.scattered_field) {
                    status = cuda_data::copy_device_to_device(
                        widefield_buffers.scattered_field, resident_incident_grid,
                        size_t(3) * ngrid * sizeof(CUCOMPLEX));
                    incident_grid_is_resident = status == cudaSuccess;
                    if (status != cudaSuccess)
                        status = 1;
                }
            }
            if (status != 0) {
                if (trace_widefield)
                    std::fprintf(stderr,
                        "WIDEFIELD_TRACE illumination failed beam=%.*s status=%d\n",
                        static_cast<int>(beam_len), beam, status);
                *nstop = 1;
                const char *message = is_speckle ?
                    "CUDA speckle illumination failed" :
                    "CUDA confocal illumination failed";
                if (infostr_len) {
                    std::memset(infostr, ' ', infostr_len);
                    std::memcpy(infostr, message,
                        std::min(infostr_len, std::strlen(message)));
                }
                return;
            }
            *E0 = e0_cuda;
            *irra = static_cast<REAL>(irradiance_cuda);
            if (!incident_grid_is_resident) {
                if (trace_widefield)
                    std::fprintf(stderr,
                        "WIDEFIELD_TRACE incident field is not resident beam=%.*s\n",
                        static_cast<int>(beam_len), beam);
                *nstop = 1;
                if (infostr_len) {
                    constexpr char message[] =
                        "CUDA special illumination did not return a resident field";
                    std::memset(infostr, ' ', infostr_len);
                    std::memcpy(infostr, message,
                        std::min(infostr_len, sizeof(message) - 1));
                }
                return;
            }
        }
    }

    if (!special_beam) {
        const int count = static_cast<int>(ngrid);
        const CUCOMPLEX amplitude{
            static_cast<CU_REAL>(std::real(*E0)),
            static_cast<CU_REAL>(std::imag(*E0))};
        const CUCOMPLEX medium{
            static_cast<CU_REAL>(std::real(eps0)),
            static_cast<CU_REAL>(std::imag(eps0))};
        int status = 1;
        const bool linear_plane = has_prefix(beam, beam_len, "pwavelinear", 11);
        const bool circular_plane = has_prefix(beam, beam_len, "pwavecircular", 13);
        const bool multi_plane = has_prefix(beam, beam_len, "wavelinearmulti", 15);
        const bool antenna = has_prefix(beam, beam_len, "antenna", 7);
        const bool paraxial_linear = has_prefix(beam, beam_len, "gparawavelinear", 15);
        const bool paraxial_circular = has_prefix(beam, beam_len, "gparawavecircular", 17);
        const bool rigorous_linear = has_prefix(beam, beam_len, "gwavelinear", 11);
        const bool rigorous_circular = has_prefix(beam, beam_len, "gwavecircular", 13);
        const bool arbitrary = has_prefix(beam, beam_len, "arbitrary", 9);
        if (linear_plane || circular_plane) {
            const int circular = circular_plane ? 1 : 0;
            const COMPLEX medium_host = eps0;
            const COMPLEX amplitude_host = *E0;
            ifdda_plane_wave_field_cuda_device_coordinates_(
                widefield_buffers.grid_x, widefield_buffers.grid_y,
                widefield_buffers.grid_z, &count, k0, &medium_host, &amplitude_host,
                ss, pp, theta, phi, &circular,
                widefield_buffers.scattered_field, &status);
            incident_grid_is_resident = status == 0;
        } else if (multi_plane) {
            const COMPLEX zero{};
            ifdda_multi_plane_wave_field_cuda_device_coordinates_(
                widefield_buffers.grid_x, widefield_buffers.grid_y,
                widefield_buffers.grid_z, &count, k0, &zero,
                E0m, ssm, ppm, thetam, phim, nbinc,
                widefield_buffers.scattered_field, &status);
            incident_grid_is_resident = status == 0;
        } else if (antenna) {
            COMPLEX reference_field[3]{};
            const COMPLEX amplitude_host = *E0;
            ifdda_antenna_field_cuda_device_coordinates_(
                widefield_buffers.grid_x, widefield_buffers.grid_y,
                widefield_buffers.grid_z, &count, xdip, ydip, zdip,
                aretecube, k0, &amplitude_host, theta, phi, xgaus, ygaus,
                zgaus, reference_field, &status,
                widefield_buffers.scattered_field);
            incident_grid_is_resident = status == 0;
        } else if (paraxial_linear || paraxial_circular) {
            const int circular = paraxial_circular ? 1 : 0;
            const COMPLEX amplitude_host = *E0;
            const COMPLEX medium_host = eps0;
            ifdda_paraxial_gaussian_field_cuda_device_coordinates_(
                widefield_buffers.grid_x, widefield_buffers.grid_y,
                widefield_buffers.grid_z, &count, &amplitude_host, &medium_host, ss, pp,
                theta, phi, xgaus, ygaus, zgaus, w0, k0, &circular,
                widefield_buffers.scattered_field, &status);
            incident_grid_is_resident = status == 0;
        } else if (rigorous_linear || rigorous_circular) {
            const int circular = rigorous_circular ? 1 : 0;
            const int nloin = 0;
            const COMPLEX amplitude_host = *E0;
            ifdda_rigorous_gaussian_field_cuda_device_coordinates_(
                widefield_buffers.grid_x, widefield_buffers.grid_y,
                widefield_buffers.grid_z, &count, xgaus, ygaus, zgaus,
                theta, phi, w0, k0, ss, pp, &amplitude_host, tolinit,
                &circular, &nloin, widefield_buffers.scattered_field, &status);
            incident_grid_is_resident = status == 0;
        } else if (arbitrary) {
            char arbitrary_info[64]{};
            const CU_REAL grid_x_min = static_cast<CU_REAL>(xmin) -
                CU_REAL(*nxmp) * static_cast<CU_REAL>(*aretecube);
            const CU_REAL grid_x_max = grid_x_min +
                CU_REAL(nxmpp - 1) * static_cast<CU_REAL>(*aretecube);
            const CU_REAL grid_y_min = static_cast<CU_REAL>(ymin) -
                CU_REAL(*nymp) * static_cast<CU_REAL>(*aretecube);
            const CU_REAL grid_y_max = grid_y_min +
                CU_REAL(nympp - 1) * static_cast<CU_REAL>(*aretecube);
            const CU_REAL grid_z_min = static_cast<CU_REAL>(zmin) -
                CU_REAL(*nzmp) * static_cast<CU_REAL>(*aretecube);
            const CU_REAL grid_z_max = grid_z_min +
                CU_REAL(nzmpp - 1) * static_cast<CU_REAL>(*aretecube);
            ifdda_incident_arbitrary_cuda_device_grid_bounds_(
                &grid_x_min, &grid_x_max, &grid_y_min, &grid_y_max,
                &grid_z_min, &grid_z_max, widefield_buffers.grid_x,
                widefield_buffers.grid_y, widefield_buffers.grid_z,
                aretecube, widefield_buffers.scattered_field,
                nxm, nym, nzm, &count, nstop, namefileinc, arbitrary_info);
            status = *nstop == 0 ? 0 : 1;
            incident_grid_is_resident = status == 0;
        }

        if (status != 0) {
            *nstop = 1;
            if (infostr_len) {
                constexpr char message[] = "CUDA wide-field illumination failed";
                std::memset(infostr, ' ', infostr_len);
                std::memcpy(infostr, message,
                    std::min(infostr_len, sizeof(message) - 1));
            }
            return;
        }
    }

    if (*nmat == 0) {
        fx=open_optional_output("xwf.mat");
        fy=open_optional_output("ywf.mat");
        fz=open_optional_output("zwf.mat");
        fix=open_optional_output("incidentfieldxwf.mat");
        fiy=open_optional_output("incidentfieldywf.mat");
        fiz=open_optional_output("incidentfieldzwf.mat");
        fim=open_optional_output("incidentfieldwf.mat");
    }

    for (i = 0; i < nzmpp; ++i)
        for (j = 0; j < nympp; ++j)
            for (k = 0; k < nxmpp; ++k) {
                const REAL x=xmin+(REAL)(k-*nxmp)* *aretecube;
                const REAL y=ymin+(REAL)(j-*nymp)* *aretecube;
                const REAL z=zmin+(REAL)(i-*nzmp)* *aretecube;
                ++sub;
                if (j==0 && i==0 && fx) fprintf(fx,"%.17g\n",x);
                if (i==0 && k==0 && fy) fprintf(fy,"%.17g\n",y);
                if (j==0 && k==0 && fz) fprintf(fz,"%.17g\n",z);
            }
    *subunit=sub;
    if (fx) fclose(fx); if (fy) fclose(fy); if (fz) fclose(fz);

    const size_t dipole_field_count = size_t(3) * size_t(*nbsphere);
    const CUCOMPLEX* resident_scattered = special_beam
        ? get_device_result_vector_alias(FF, dipole_field_count)
        : nullptr;
    static_assert(sizeof(COMPLEX) == sizeof(CUCOMPLEX),
                  "Host and CUDA complex element sizes must match");

    bool scatter_ok = widefield_buffers.scattered_field != nullptr;
    if (trace_widefield)
        std::fprintf(stderr,
            "WIDEFIELD_TRACE scatter begin resident_incident=%d resident_scattered=%d FF=%p epsilon=%p dipoles=%d grid=%zu\n",
            incident_grid_is_resident ? 1 : 0, resident_scattered ? 1 : 0,
            static_cast<const void *>(FF), static_cast<const void *>(epsilon),
            *nbsphere, ngrid);
    if (scatter_ok && !incident_grid_is_resident)
        scatter_ok = cuda_data::copy_host_complex_to_device(
            widefield_buffers.scattered_field, xr, size_t(3) * ngrid,
            cuda_utils::TransferCategory::IncidentField) == cudaSuccess;
    if (scatter_ok) {
        if (resident_scattered)
            scatter_ok = ifdda_widefield_scatter_dipoles_device_c99(
                resident_scattered, epsilon, *nmax, *nbsphere,
                static_cast<int>(ngrid), *niso, &widefield_buffers) == 0;
        else {
            const int scatter_status = ifdda_widefield_scatter_dipoles_host_c99(
                FF, special_beam ? nullptr : FF0, epsilon, *nmax,
                *nbsphere, static_cast<int>(ngrid), *niso,
                special_beam ? 1 : 0, &widefield_buffers);
            scatter_ok = scatter_status == 0;
            if (trace_widefield)
                std::fprintf(stderr,
                    "WIDEFIELD_TRACE host scatter status=%d\n", scatter_status);
        }
    }
    if (scatter_ok)
        scatter_ok = cuda_utils::check_cuda(
            widefield_incident_magnitude_wrapper(
                widefield_buffers.scattered_field,
                widefield_buffers.incident_magnitude, static_cast<int>(ngrid)),
            "CUDA wide-field incident field", "compute GPU magnitude");
    static_assert(sizeof(REAL) == sizeof(CU_REAL),
                  "Fortran and CUDA real element sizes must match");
    if (scatter_ok)
        scatter_ok = cuda_data::copy_device_to_host(
            incidentfield, widefield_buffers.incident_magnitude,
            ngrid * sizeof(CU_REAL),
            cuda_utils::TransferCategory::IncidentField) == cudaSuccess;
    if (scatter_ok)
        scatter_ok = cuda_data::copy_device_complex_to_host(
            xr, widefield_buffers.scattered_field, size_t(3) * ngrid,
            cuda_utils::TransferCategory::IncidentField) == cudaSuccess;
    if (!scatter_ok) {
        if (trace_widefield)
            std::fprintf(stderr, "WIDEFIELD_TRACE scatter pipeline failed\n");
        *nstop = 1;
        if (infostr_len) {
            constexpr char message[] = "CUDA wide-field dipole scatter failed";
            std::memset(infostr, ' ', infostr_len);
            std::memcpy(infostr, message,
                std::min(infostr_len, sizeof(message) - 1));
        }
        return;
    }
    for (i=0;i<sub;++i) {
        incidentfieldx[i]=xr[3*i]; incidentfieldy[i]=xr[3*i+1];
        incidentfieldz[i]=xr[3*i+2];
        if(fix) fprintf(fix,"%.17g %.17g\n",std::real(xr[3*i]),std::imag(xr[3*i]));
        if(fiy) fprintf(fiy,"%.17g %.17g\n",std::real(xr[3*i+1]),std::imag(xr[3*i+1]));
        if(fiz) fprintf(fiz,"%.17g %.17g\n",std::real(xr[3*i+2]),std::imag(xr[3*i+2]));
        if(fim) fprintf(fim,"%.17g\n",incidentfield[i]);
    }
    if(fix) fclose(fix); if(fiy) fclose(fiy); if(fiz) fclose(fiz); if(fim) fclose(fim);

    ntotal=8* *nmaxpp;
    {
        bool ok = bool(device_green) && widefield_buffers.initial_field &&
            widefield_buffers.scattered_field;
        if (ok) {
            const long long cuda_planfn = static_cast<long long>(planfn);
            const long long cuda_planbn = static_cast<long long>(planbn);
            produitfftmatvectoptresident_(green_view(device_green),
                widefield_buffers.initial_field, widefield_buffers.scattered_field, ntotalm,
                &ntotal, nmaxpp, nxm, nym, nzm, &nxmpp, &nympp, &nzmpp,
                &nxm2, &nym2, &nxym2, &nzm2, &cuda_planfn, &cuda_planbn);
            if (*nrig != 0) {
                ok = cuda_data::copy_host_complex_to_device(
                    widefield_buffers.dipole_scattered, FFloc,
                    size_t(3) * size_t(*nbsphere),
                    cuda_utils::TransferCategory::SolverVector) == cudaSuccess;
                if (ok)
                    ok = cuda_utils::check_cuda(
                        widefield_restore_material_local_field_wrapper(
                            widefield_buffers.dipole_map, static_cast<int>(ngrid),
                            *nbsphere, *niso, widefield_buffers.dipole_epsilon,
                            widefield_buffers.dipole_scattered,
                            widefield_buffers.scattered_field),
                        "wide-field result", "restore local field in material");
            }
        }
        if (ok) {
            ok = cuda_data::copy_device_complex_to_host(
                xr, widefield_buffers.scattered_field, size_t(3) * ngrid,
                cuda_utils::TransferCategory::Result) == cudaSuccess;
        }
        if (!ok) {
            *nstop = 1;
            if (infostr_len) {
                constexpr char message[] = "CUDA wide-field Green product failed";
                std::memset(infostr, ' ', infostr_len);
                std::memcpy(infostr, message,
                    std::min(infostr_len, sizeof(message) - 1));
            }
            return;
        }
    }
}
