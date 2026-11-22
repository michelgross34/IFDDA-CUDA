#include "cdmlib_cpp_abi.hpp"
#define IFDDA_CUDA_CALLERS_MICROSDF_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_MICROSDF_CUDA
#include "cuda_data.h"
#include "cuda_solver_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"
#include "microsdf_fields_CUDA.h"
#include <algorithm>
#include <new>
#define I (COMPLEX(0.0, 1.0))
/*
 * C99 implementation of cdmlib/microsdf.f90.
 * All Fortran arguments are passed by address; hidden string lengths follow
 * the explicit arguments according to the GNU Fortran ABI.
 */

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>













static void microsdf_set_info(char *dst, size_t length, const char *message)
{
    size_t n = strlen(message);
    if (n > length) n = length;
    memset(dst, ' ', length);
    memcpy(dst, message, n);
}

//-----------------------------------------------------------
extern "C" void deltakroutine_(const REAL *kxinc, const REAL *kyinc,
    const REAL *deltakx, const REAL *deltaky, const REAL *numaper,
    int *i, int *j)
{
    if (!kxinc || !kyinc || !deltakx || !deltaky || !numaper || !i || !j ||
        *deltakx == REAL(0) || *deltaky == REAL(0))
        return;

    *i = static_cast<int>(std::round(*kxinc / *deltakx));
    *j = static_cast<int>(std::round(*kyinc / *deltaky));

    const REAL kx = *deltakx * static_cast<REAL>(*i);
    const REAL ky = *deltaky * static_cast<REAL>(*j);
    if (kx * kx + ky * ky <= *numaper * *numaper)
        return;

    // Match the Fortran fallback, which repeats the same nearest-grid choice.
    *i = static_cast<int>(std::round(*kxinc / *deltakx));
    *j = static_cast<int>(std::round(*kyinc / *deltaky));
}

/* Every explicit Fortran argument is passed by address.  The three hidden
 * CHARACTER lengths follow all explicit arguments in GNU Fortran's ABI. */
//-------------------------------------------
extern "C" void microsdf_(
    COMPLEX *FFTTENSORxx, COMPLEX *FFTTENSORxy, COMPLEX *FFTTENSORxz,
    COMPLEX *FFTTENSORyy, COMPLEX *FFTTENSORyz, COMPLEX *FFTTENSORzz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    int *ntotalm, int *ntotal, int *ldabi, int *nlar, int *nmax,
    int *nxm, int *nym, int *nzm, int *nx, int *ny, int *nz,
    int *nx2, int *ny2, int *nxy2, int *nz2, int *ndipole,
    int *nbsphere, int *nbsphere3, int *nproche, int *nrig,
    int *nfft2d, int *tabfft2, int *tabdip, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *wrk, COMPLEX *FF, COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *polarisa,
    int *niso, COMPLEX *epsilon, char *methodeit, char *trope,
    REAL *tolinit, REAL *tol1,
    int *nloop, int *ncompte, int *nlim,
    REAL *xs, REAL *ys, REAL *zs, REAL *aretecube, REAL *numaper,
    REAL *numaperinc, int *npolainc, int *nquicklens, COMPLEX *eps0,
    REAL *k0, REAL *P0, REAL *irra, REAL *w0, REAL *gross, REAL *zlens,
    COMPLEX *Eimagex, COMPLEX *Eimagey, COMPLEX *Eimagez,
    COMPLEX *Efourierx, COMPLEX *Efouriery, COMPLEX *Efourierz,
    COMPLEX *Eimageincx, COMPLEX *Eimageincy, COMPLEX *Eimageincz,
    COMPLEX *Efourierincx, COMPLEX *Efourierincy, COMPLEX *Efourierincz,
    COMPLEX *Ediffkzpos, COMPLEX *Ediffkzneg, REAL *kxy, REAL *xy, int *nside,
    int64_t *planXf, int64_t *planXb, int64_t *planYf, int64_t *planYb,
    int64_t *planZf, int64_t *planZb, int64_t *planf, int64_t *planb,
    int64_t *plan2f, int64_t *plan2b, int *nmat, int64_t *file_id,
    int64_t *group_idmic, int *nstop, char *infostr,
    size_t methodeit_len, size_t trope_len, size_t infostr_len)
{
    int *const p_npolainc = npolainc;
    int *const p_nstop = nstop;
    const int *const p_nrig = nrig;
    const int *const p_nfft2d = nfft2d;
    const REAL *const p_numaper = numaper;
    const REAL *const p_numaperinc = numaperinc;

    /* These are referenced by later stages of the translation. */
    (void)FFTTENSORxx; (void)FFTTENSORxy; (void)FFTTENSORxz;
    (void)FFTTENSORyy; (void)FFTTENSORyz; (void)FFTTENSORzz;
    (void)vectx; (void)vecty; (void)vectz;
    (void)ntotalm; (void)ntotal; (void)ldabi; (void)nlar; (void)nmax;
    (void)nxm; (void)nym; (void)nzm; (void)nx; (void)ny; (void)nz;
    (void)nx2; (void)ny2; (void)nxy2; (void)nz2; (void)ndipole;
    (void)nbsphere; (void)nbsphere3; (void)nproche;
    (void)tabfft2; (void)tabdip; (void)XI; (void)XR; (void)wrk;
    (void)FF; (void)FF0; (void)FFloc; (void)polarisa; (void)niso;
    (void)epsilon; (void)methodeit; (void)trope; (void)tolinit; (void)tol1;
    (void)nloop; (void)ncompte; (void)nlim; (void)xs; (void)ys; (void)zs;
    (void)eps0;
    (void)irra; (void)w0; (void)gross; (void)zlens;
    (void)Eimagex; (void)Eimagey; (void)Eimagez;
    (void)Efourierx; (void)Efouriery; (void)Efourierz;
    (void)Eimageincx; (void)Eimageincy; (void)Eimageincz;
    (void)Efourierincx; (void)Efourierincy; (void)Efourierincz;
    (void)Ediffkzpos; (void)Ediffkzneg; (void)kxy; (void)xy; (void)nside;
    (void)planXf; (void)planXb; (void)planYf; (void)planYb;
    (void)planZf; (void)planZb; (void)planf; (void)planb;
    (void)plan2f; (void)plan2b; (void)nmat; (void)file_id;
    (void)group_idmic; (void)methodeit_len; (void)trope_len;

    printf("***** Dark field and phase microscope *****\n");
    *p_npolainc = 0;
    if (*p_numaperinc >= 1.0 || *p_numaper <= 0.0) {
        microsdf_set_info(infostr, infostr_len,
                          "NA inc strictly between 0 and 1");
        *p_nstop = 1;
        return;
    }

    /* Shared spatial-frequency and incidence-grid setup (Fortran lines 77-171). */
    {
        const REAL pi = 3.141592653589793238462643383279502884;
        const REAL *x = xs, *y = ys;
        const REAL *z = zs;
        const REAL arete = *aretecube;
        const REAL kval = *k0;
        const REAL k02 = kval * kval;
        const int nfft = *p_nfft2d, half = nfft / 2;
        const int nbs = *nbsphere;
        const int quick = *nquicklens;
        REAL xmax = -CDMLIB_MAX / 100.0, xmin = CDMLIB_MAX / 100.0;
        REAL ymax = -CDMLIB_MAX / 100.0, ymin = CDMLIB_MAX / 100.0;
        REAL deltakm, deltakx, deltak, deltak_y, deltatheta;
        int imaxk0 = 0, imaxinc, ideltam, k = 0;
        int i, npol = 2, niter;
        REAL phi, theta, kxinc, kyinc, ss, pp;
        REAL xzero=0.0, yzero=0.0, zzero=0.0;
        COMPLEX E0 = 0.0 + 0.0 * I;
        REAL *const xsout = xs;
        REAL *const ysout = ys;
        REAL *const zsout = zs;
        COMPLEX *const imageix = Eimageincx;
        COMPLEX *const imageiy = Eimageincy;
        COMPLEX *const imageiz = Eimageincz;
        COMPLEX *const imagex = Eimagex;
        COMPLEX *const imagey = Eimagey;
        COMPLEX *const imagez = Eimagez;
        COMPLEX *const fourierx = Efourierx;
        COMPLEX *const fouriery = Efouriery;
        COMPLEX *const fourierz = Efourierz;
        COMPLEX *const fourierix = Efourierincx;
        COMPLEX *const fourieriy = Efourierincy;
        COMPLEX *const fourieriz = Efourierincz;
        COMPLEX *const diffpos = Ediffkzpos;
        COMPLEX *const incident = FF0;
        COMPLEX *const initial = XI;
        COMPLEX *const real_work = XR;
        COMPLEX *const polar = polarisa;
        COMPLEX *const local = FFloc;
        COMPLEX *const scattered = FF;
        REAL irradiance;
        REAL *const p_irra = irra;
        REAL *const p_P0 = P0;
        REAL *const p_w0 = w0;
        const int *const p_niso = niso;
        const int *const p_nmax = nmax;
        REAL *const p_tolinit = tolinit;
        REAL *const p_tol1 = tol1;
        int *const p_nloop = nloop;
        int *const p_ncompte = ncompte;
        int *const p_nlim = nlim;
        const int negative_side = *nside < 0;
        char beam[64];
        (void)z; (void)half; (void)k02; (void)imaxk0;
        (void)xsout; (void)ysout; (void)zsout; (void)p_tolinit;
        (void)p_tol1; (void)p_nloop; (void)p_ncompte; (void)p_nlim;
        (void)beam;

        for (i = 0; i < nbs; ++i) {
            if (x[i] > xmax) xmax = x[i];
            if (x[i] < xmin) xmin = x[i];
            if (y[i] > ymax) ymax = y[i];
            if (y[i] < ymin) ymin = y[i];
        }
        deltakm = pi / fmax(xmax - xmin, ymax - ymin);
        if (quick == 1) {
            const REAL numaperk = kval * *p_numaperinc;
            REAL ratio;
            deltakx = 2.0 * pi / (arete * (REAL)nfft);
            if (deltakx >= numaperk) {
                *p_nstop = 1;
                microsdf_set_info(infostr, infostr_len,
                    "In FFT lens nfft2d too small:Increase size of FFT 1");
                return;
            }
            ratio = deltakm / deltakx;
            i = (int)ratio;
            if (i == 0) {
                *p_nstop = 1;
                microsdf_set_info(infostr, infostr_len,
                    "In FFT lens nfft2d too small:Increase size of FFT 2");
                return;
            }
            for (;;) {
                deltak = deltakx * (REAL)i;
                imaxinc = (int)lround(kval / deltak) + 1;
                if (imaxinc > 3) break;
                --i;
                if (i == 0) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "In FFT lens nfft2d too small:Increase size of FFT 3");
                    return;
                }
            }

        } else {
            for (k = 0;; ++k) {
                deltakx = 2.0 * pi / ((REAL)nfft * arete) / ldexp(1.0, k);
                imaxk0 = (int)lround(kval / deltakx) + 1;
                if (imaxk0 > 5) break;
            }
            for (k = 0;; ++k) {
                const REAL q = deltakm / deltakx;
                deltak = deltakx * (REAL)lround(q) / ldexp(1.0, k);
                imaxinc = (int)lround(kval / deltak) + 1;
                if (imaxinc > 2) break;
            }
        }
        deltak_y = deltakx;
        deltatheta = deltak / kval * *p_numaperinc;
        ideltam = (int)(2.0 * pi / deltatheta) + 1;
        if (ideltam < 8) ideltam = 8;
#if defined(IFDDA_TEST_MAX_ANGLE_POINTS) && IFDDA_TEST_MAX_ANGLE_POINTS > 0
        if (ideltam > IFDDA_TEST_MAX_ANGLE_POINTS)
            ideltam = IFDDA_TEST_MAX_ANGLE_POINTS;
#endif
        if (imaxk0 == 0) imaxk0 = (int)lround(kval / deltakx) + 1;
        std::fill_n(imageix, (size_t)nfft * nfft, COMPLEX{});
        std::fill_n(imageiy, (size_t)nfft * nfft, COMPLEX{});
        std::fill_n(imageiz, (size_t)nfft * nfft, COMPLEX{});
        std::fill_n(imagex, (size_t)nfft * nfft, COMPLEX{});
        std::fill_n(imagey, (size_t)nfft * nfft, COMPLEX{});
        std::fill_n(imagez, (size_t)nfft * nfft, COMPLEX{});
        niter = npol * ideltam;
        const size_t microscope_plane_count = size_t(nfft) * size_t(nfft);
        size_t microscope_xy_cell_count = 0;
        size_t microscope_mesh_cell_count = 0;
        if (*nx <= 0 || *ny <= 0 || *nz <= 0 ||
            !cuda_utils::checked_multiply(size_t(*nx), size_t(*ny),
                &microscope_xy_cell_count) ||
            !cuda_utils::checked_multiply(microscope_xy_cell_count,
                size_t(*nz), &microscope_mesh_cell_count) ||
            microscope_mesh_cell_count > size_t(INT_MAX) / size_t(3)) {
            *p_nstop = 1;
            microsdf_set_info(infostr, infostr_len,
                "Invalid CUDA dark-field mesh dimensions");
            return;
        }
        const size_t microscope_dense_field_count =
            size_t(3) * microscope_mesh_cell_count;
        cuda_data::BrightFieldMicroscopeDeviceBuffers dark_image_buffers{};
        const bool use_cuda_image_pipeline = true;
        if (use_cuda_image_pipeline &&
            (!cuda_data::get_device_bright_field_microscope_buffers(
                size_t(nbs), microscope_plane_count, nfft, &dark_image_buffers) ||
             cudaMemset(dark_image_buffers.image_intensity, 0,
                6u * microscope_plane_count * sizeof(CU_REAL)) != cudaSuccess)) {
            *p_nstop = 1;
            microsdf_set_info(infostr, infostr_len,
                "Unable to initialize CUDA dark-field image buffers");
            return;
        }
        *p_P0 /= (REAL)niter;
        irradiance_(p_P0, p_w0, &E0, &irradiance);
        *p_irra = irradiance;
        printf("Magnitude of each illumination: (% .16e,% .16e)\n",
               std::real(E0), std::imag(E0));

        cuda_data::LocalMacroDeviceBuffers nrig5_macro_buffers{};
        cuda_data::DarkFieldMicroscopeDeviceBuffers dark_field_device_buffers{};
        const size_t microscope_field_count = 3u * size_t(nbs);
        if (!cuda_data::get_device_dark_field_microscope_buffers(
                size_t(nbs), 0, 0, 0,
                3u * microscope_plane_count, 0, 0, 0,
                std::max(microscope_field_count, microscope_dense_field_count),
                &dark_field_device_buffers) ||
            !dark_field_device_buffers.spectrum_planes ||
            !dark_field_device_buffers.image_scratch) {
            *p_nstop = 1;
            microsdf_set_info(infostr, infostr_len,
                "CUDA dark-field diffraction buffers could not be initialized");
            return;
        }
        CU_INT *const device_microscope_dipole_map =
            get_device_tabdip(tabdip, microscope_mesh_cell_count);
        if (!device_microscope_dipole_map) {
            *p_nstop = 1;
            microsdf_set_info(infostr, infostr_len,
                "CUDA dark-field dipole map allocation failed");
            return;
        }
        bool nrig5_buffers_ready = false;
        if (*p_nrig == 3 || *p_nrig == 4 || *p_nrig == 5) {
            const size_t dipole_count = size_t(nbs);
            const size_t field_count = size_t(3) * dipole_count;
            const size_t material_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
            try {
                std::vector<CUCOMPLEX> epsilon_cuda(material_count);
                std::vector<CUCOMPLEX> polarization_cuda(material_count);
                for (size_t index = 0; index < material_count; ++index) {
                    epsilon_cuda[index] = CUCOMPLEX{
                        CU_REAL(std::real(epsilon[index])),
                        CU_REAL(std::imag(epsilon[index]))};
                    polarization_cuda[index] = CUCOMPLEX{
                        CU_REAL(std::real(polar[index])),
                        CU_REAL(std::imag(polar[index]))};
                }
                const bool dark_field_buffers_ready =
                    cuda_data::get_device_dark_field_microscope_buffers(
                        dipole_count, material_count, 0, 0,
                        3u * microscope_plane_count, 0, 4 * field_count, 0,
                        microscope_field_count,
                        &dark_field_device_buffers) &&
                    cuda_data::copy_host_to_device(dark_field_device_buffers.polarization,
                        polarization_cuda.data(), material_count * sizeof(CUCOMPLEX),
                        cuda_utils::TransferCategory::Material) == cudaSuccess;
                const bool local_macro_buffers_ready = *p_nrig != 5 ||
                    (cuda_data::get_device_local_macro_buffers(
                        material_count, dipole_count, &nrig5_macro_buffers) &&
                     cuda_data::copy_host_to_device(nrig5_macro_buffers.epsilon,
                        epsilon_cuda.data(), material_count * sizeof(CUCOMPLEX),
                        cuda_utils::TransferCategory::Material) == cudaSuccess);
                if (!dark_field_buffers_ready || !local_macro_buffers_ready) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA Rytov microscope buffers could not be initialized");
                    return;
                }
                nrig5_buffers_ready = true;
            } catch (...) {
                *p_nstop = 1;
                microsdf_set_info(infostr, infostr_len,
                    "CUDA Rytov microscope allocation failed");
                return;
            }
        }

        const auto process_cuda_image = [&](const CUCOMPLEX *spectrum) -> bool {
            if (!spectrum) {
                *p_nstop = 1;
                microsdf_set_info(infostr, infostr_len,
                    "CUDA dark-field spectrum is unavailable");
                return false;
            }
            int ikxinc = 0, jkyinc = 0;
            deltakroutine_(&kxinc, &kyinc, &deltakx, &deltak_y,
                           &kval, &ikxinc, &jkyinc);
            const REAL ikx = deltakx * REAL(ikxinc);
            const REAL jky = deltak_y * REAL(jkyinc);
            const REAL transverse = ikx * ikx + jky * jky;
            COMPLEX incident_x = COMPLEX(0.0, 0.0);
            COMPLEX incident_y = COMPLEX(0.0, 0.0);
            COMPLEX incident_z = COMPLEX(0.0, 0.0);
            if (std::abs(ikxinc) <= imaxk0 &&
                std::abs(jkyinc) <= imaxk0 &&
                transverse <= *numaper * *numaper && transverse < k02) {
                ondeplanekxky_(&xzero, &yzero, &zzero, &kval, &E0,
                    &ss, &pp, &ikx, &jky, &incident_x, &incident_y,
                    &incident_z, p_nstop, infostr, infostr_len);
                if (*p_nstop != 0)
                    return false;
            }
            const REAL incident_scale = deltakx == REAL(0)
                ? REAL(0) : REAL(1) / (deltakx * deltakx);
            const cudaError_t prepare_status = bright_field_prepare_fourier_wrapper(
                spectrum, dark_image_buffers.fourier_fields, nfft, imaxk0,
                CU_REAL(deltakx), CU_REAL(deltak_y), CU_REAL(kval),
                CU_REAL(*numaper), CU_REAL(*zlens), CU_REAL(*gross),
                negative_side ? -1 : 1, 2, CU_REAL(0), ikxinc, jkyinc,
                CU_REAL(incident_scale),
                CUCOMPLEX{CU_REAL(std::real(incident_x)),
                          CU_REAL(std::imag(incident_x))},
                CUCOMPLEX{CU_REAL(std::real(incident_y)),
                          CU_REAL(std::imag(incident_y))},
                CUCOMPLEX{CU_REAL(std::real(incident_z)),
                          CU_REAL(std::imag(incident_z))});
            if (!cuda_utils::check_cuda(prepare_status,
                    "dark-field image", "prepare Fourier planes") ||
                (!negative_side && !cuda_utils::check_cuda(
                    bright_field_image_accumulate_wrapper(
                        dark_image_buffers.fourier_fields,
                        dark_image_buffers.image_intensity,
                        microscope_plane_count, 3, 3),
                    "dark-field image", "accumulate incident intensity"))) {
                *p_nstop = 1;
                microsdf_set_info(infostr, infostr_len,
                    "CUDA dark-field Fourier preparation failed");
                return false;
            }
            const cufftResult fft_status = CUFFT_EXECUTE(
                dark_image_buffers.fft_plan,
                dark_image_buffers.fourier_fields,
                dark_image_buffers.fourier_fields, CUFFT_FORWARD);
            const REAL image_scale = deltakx * deltak_y / *gross;
            if (!cuda_utils::check_cufft(fft_status, "dark-field image",
                    "execute batched 2-D FFT") ||
                !cuda_utils::check_cuda(fourier_to_image_rearrange_wrapper(
                    dark_image_buffers.fourier_fields, nfft,
                    CU_REAL(image_scale)), "dark-field image",
                    "rearrange Fourier planes") ||
                !cuda_utils::check_cuda(bright_field_image_accumulate_wrapper(
                    dark_image_buffers.fourier_fields,
                    dark_image_buffers.image_intensity,
                    microscope_plane_count, 0, negative_side ? 3 : 6),
                    "dark-field image", "accumulate image intensities")) {
                *p_nstop = 1;
                microsdf_set_info(infostr, infostr_len,
                    "CUDA dark-field image FFT failed");
                return false;
            }
            return true;
        };
        const CUCOMPLEX *final_device_spectrum = nullptr;

        /* Sample all plane-wave incidences and build FF0/XI exactly as the
         * Fortran pre-solver loop does. The solver and detector stages follow. */
        for (i = 0; i < niter; ++i) {
            const int ipol = i / ideltam;
            const int idelta = i % ideltam;
            const REAL twopi = 2.0 * pi;
            ss = ipol == 0 ? 1.0 : 0.0;
            pp = ipol == 0 ? 0.0 : 1.0;
            phi = (REAL)idelta * twopi / (REAL)ideltam;
            theta = asin(*p_numaperinc);
            kxinc = cos(phi) * kval * *p_numaperinc;
            kyinc = sin(phi) * kval * *p_numaperinc;
            (void)theta;
            cuda_data::MicroscopeIncidentDeviceBuffers incident_device{};
            const int incident_status = ifdda_microsdf_plane_wave_device_c99(
                incident, x, y, z, size_t(nbs), CU_REAL(kval), CU_REAL(kxinc),
                CU_REAL(kyinc),
                std::complex<CU_REAL>(CU_REAL(std::real(E0)), CU_REAL(std::imag(E0))),
                CU_REAL(pp), &incident_device);
            const size_t incident_bytes = size_t(3) * size_t(nbs) * sizeof(CUCOMPLEX);
            const bool host_incident_needed =
                *p_nrig != 3 && *p_nrig != 4 && *p_nrig != 5;
            if (incident_status != 0 || !incident_device.incident ||
                !incident_device.initial ||
                (host_incident_needed && cuda_data::copy_device_to_host(
                    incident, incident_device.incident, incident_bytes,
                    cuda_utils::TransferCategory::IncidentField) != cudaSuccess) ||
                (host_incident_needed && cuda_data::copy_device_to_host(
                    initial, incident_device.initial, incident_bytes,
                    cuda_utils::TransferCategory::IncidentField) != cudaSuccess)) {
                *p_nstop = 1;
                microsdf_set_info(infostr, infostr_len,
                    "CUDA dark-field incident field generation failed");
                return;
            }
            const CUCOMPLEX *device_scattered_for_diffraction = nullptr;
            if (*p_nrig < 0 || *p_nrig > 9) {
                release_gpbicg_device_b_alias(incident);
                microsdf_set_info(infostr, infostr_len,
                                  "Method not implemented for microscopy");
                *p_nstop = 1;
                return;
            }
            if (*p_nrig == 0) {
                REAL tol = *p_tolinit;
                const int *p_ntotalm = ntotalm;
                const int *p_ntotal = ntotal;
                const int *p_ldabi = ldabi;
                const int *p_nlar = nlar;
                const int *p_nmax = nmax;
                const int *p_ndipole = ndipole;
                const int *p_nxm = nxm, *p_nym = nym;
                const int *p_nzm = nzm, *p_nx = nx;
                const int *p_ny = ny, *p_nz = nz;
                const int *p_nx2 = nx2, *p_ny2 = ny2;
                const int *p_nxy2 = nxy2, *p_nz2 = nz2;
                const int *p_nbs3 = nbsphere3;
                const int *p_niso = niso;
                const int *p_nlim = nlim;
                REAL *p_tol1 = tol1;
                int *p_nloop = nloop, *p_ncompte = ncompte;
                int *p_nstop = nstop;
                char *method = methodeit;
                char *info = infostr;
                COMPLEX *tensors[6] = {FFTTENSORxx,FFTTENSORxy,FFTTENSORxz,
                                    FFTTENSORyy,FFTTENSORyz,FFTTENSORzz};
                int64_t *planxf = planXf, *planxb = planXb;
                int64_t *planyf = planYf, *planyb = planYb;
                int64_t *planzf = planZf, *planzb = planZb;
                int64_t *planf64 = planf, *planb64 = planb;
                *p_nloop = 0;
                *p_ncompte = 0;
                // Native C++ microscopy calls the resident solver directly.
                // Keep inverserig_ / inverserigopt_ only for Fortran ABI callers.
                if (!p_ntotal || *p_ntotal <= 0 ||
                    size_t(*p_ntotal) > SIZE_MAX / 3u) {
                    *p_nstop = 1;
                    microsdf_set_info(info, infostr_len,
                        "Invalid dark-field GPU FFT workspace size");
                    return;
                }
                CUCOMPLEX *fft_work = get_device_vectors(
                    size_t(3) * size_t(*p_ntotal));
                if (!fft_work) {
                    *p_nstop = 1;
                    microsdf_set_info(info, infostr_len,
                        "Unable to acquire dark-field GPU FFT workspace");
                    return;
                }
                const int optimized_product = *nproche == -1 ? 0 : 1;
                const int initial_guess_mode = optimized_product ? 1 : -1;
                const int no_preconditioner = 0;
                const long long plans[8] = {
                    static_cast<long long>(*planxf),
                    static_cast<long long>(*planxb),
                    static_cast<long long>(*planyf),
                    static_cast<long long>(*planyb),
                    static_cast<long long>(*planzf),
                    static_cast<long long>(*planzb),
                    static_cast<long long>(*planf64),
                    static_cast<long long>(*planb64)
                };
                inverserig_all_cuda_driver_(
                    tensors[0],tensors[1],tensors[2],tensors[3],
                    tensors[4],tensors[5],
                    fft_work,fft_work+*p_ntotal,
                    fft_work+size_t(2)*size_t(*p_ntotal),
                    tabdip,p_ntotalm,p_ntotal,p_ldabi,p_nlar,p_nmax,p_ndipole,
                    p_nxm,p_nym,p_nzm,p_nx,p_ny,p_nz,p_nx2,p_ny2,p_nxy2,p_nz2,
                    &nbs,p_nbs3,initial,real_work,wrk,scattered,incident,
                    local,polar,p_niso,method,&tol,p_tol1,p_nloop,p_ncompte,p_nlim,
                    &plans[0],&plans[1],&plans[2],&plans[3],
                    &plans[4],&plans[5],&plans[6],&plans[7],
                    &optimized_product,&initial_guess_mode,&no_preconditioner,
                    p_nstop,info,methodeit_len,infostr_len);
                if (*p_nstop == 1) return;
            } else if (*p_nrig == 1) {
                release_gpbicg_device_b_alias(incident);
                std::memcpy(local, incident,
                    size_t(*nbsphere3) * sizeof(COMPLEX));
                if (ifdda_produitpolavect_device_input_c99(
                        incident_device.incident, polar, niso, nmax, &nbs,
                        scattered) != 0) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA dark-field polarizability calculation failed");
                    return;
                }
            } else if (*p_nrig == 2) {
                release_gpbicg_device_b_alias(incident);
                const int nsens = -1;
                try {
                    const size_t dipole_count = size_t(nbs);
                    const size_t field_count = size_t(3) * dipole_count;
                    const size_t epsilon_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
                    std::vector<CUCOMPLEX> epsilon_cuda(epsilon_count);
                    std::vector<CUCOMPLEX> incident_cuda(field_count);
                    std::vector<CUCOMPLEX> output_x(dipole_count);
                    std::vector<CUCOMPLEX> output_y(dipole_count);
                    std::vector<CUCOMPLEX> output_z(dipole_count);
                    std::vector<CU_REAL> field_magnitude(dipole_count);
                    for (size_t index = 0; index < epsilon_count; ++index) {
                        epsilon_cuda[index] = CUCOMPLEX{
                            CU_REAL(std::real(epsilon[index])),
                            CU_REAL(std::imag(epsilon[index]))};
                    }
                    for (size_t index = 0; index < field_count; ++index) {
                        incident_cuda[index] = CUCOMPLEX{
                            CU_REAL(std::real(incident[index])),
                            CU_REAL(std::imag(incident[index]))};
                    }
                    int macro_status = 1;
                    ifdda_local_macro_batch_cuda_(epsilon_cuda.data(), incident_cuda.data(),
                        output_x.data(), output_y.data(), output_z.data(),
                        field_magnitude.data(), nmax, niso, &nbs, aretecube, k0,
                        &nsens, &macro_status);
                    if (macro_status != 0) {
                        *p_nstop = 1;
                        microsdf_set_info(infostr, infostr_len,
                            "CUDA local macro field calculation failed");
                        return;
                    }
                    for (size_t dipole = 0; dipole < dipole_count; ++dipole) {
                        const size_t base = 3u * dipole;
                        local[base] = COMPLEX(output_x[dipole].x, output_x[dipole].y);
                        local[base + 1] = COMPLEX(output_y[dipole].x, output_y[dipole].y);
                        local[base + 2] = COMPLEX(output_z[dipole].x, output_z[dipole].y);
                    }
                } catch (...) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA local macro field allocation failed");
                    return;
                }
                if (microsbf_polarization_fields_cuda(
                        &nbs, nmax, niso, polar, local, scattered) != 0) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA polarization failed");
                    return;
                }
            } else if (*p_nrig == 3 || *p_nrig == 4 || *p_nrig == 5) {
                const size_t dipole_count = size_t(nbs);
                const size_t field_count = 3u * dipole_count;
                CUCOMPLEX *device_initial = dark_field_device_buffers.solver_fields;
                CUCOMPLEX *device_delta = device_initial + field_count;
                CUCOMPLEX *device_local = device_delta + field_count;
                CUCOMPLEX *device_scattered = device_local + field_count;
                const CUCOMPLEX *device_reference = incident_device.incident;
                const size_t field_bytes = field_count * sizeof(CUCOMPLEX);
                if (*p_nrig == 5) {
                    cudaError_t macro_error = cudaErrorInvalidValue;
                    if (nrig5_buffers_ready &&
                        cuda_data::copy_device_to_device(
                            nrig5_macro_buffers.input_field, incident_device.incident,
                            field_bytes) == cudaSuccess) {
                        macro_error = local_macro_batch_wrapper(
                            nrig5_macro_buffers.epsilon,
                            nrig5_macro_buffers.input_field,
                            nrig5_macro_buffers.output_field,
                            nrig5_macro_buffers.field_magnitude,
                            nrig5_macro_buffers.status, *niso, *nmax, nbs,
                            *aretecube, kval, -1);
                    }
                    if (macro_error != cudaSuccess) {
                        *p_nstop = 1;
                        microsdf_set_info(infostr, infostr_len,
                            "CUDA dark-field local macro field failed");
                        return;
                    }
                    device_reference = nrig5_macro_buffers.output_field;
                }
                const size_t tensor_count = (size_t(*nxm) + 1u) *
                    (size_t(*nym) + 1u) * (size_t(*nzm) + 1u);
                const void *host_tensor_keys[6] = {
                    FFTTENSORxx, FFTTENSORxy, FFTTENSORxz,
                    FFTTENSORyy, FFTTENSORyz, FFTTENSORzz};
                GreenTensorDevicePointers device_green = find_device_green_for_host_keys(
                    host_tensor_keys, tensor_count);
                if (!device_green) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA dark-field Green tensor lookup failed");
                    return;
                }
                if (cudaMemcpy(device_initial, device_reference,
                        field_count * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice) != cudaSuccess ||
                    polarization_inplace_wrapper(dark_field_device_buffers.polarization,
                        device_initial, *p_niso, *p_nmax, nbs) != cudaSuccess ||
                    complex_vector_scale_wrapper(device_initial, device_initial,
                        CUCOMPLEX{CU_REAL(-1), CU_REAL(0)}, int(field_count)) != cudaSuccess) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA dark-field initial polarization failed");
                    return;
                }
                cudaError_t delta_initialization = cudaSuccess;
                if (*p_nrig == 3) {
                    delta_initialization = cudaMemcpy(device_delta, device_reference,
                        field_bytes, cudaMemcpyDeviceToDevice);
                } else if (*p_nrig == 4) {
                    delta_initialization = cudaMemset(device_delta, 0, field_bytes);
                } else {
                    const CUCOMPLEX background_epsilon{
                        CU_REAL(std::real(*eps0)), CU_REAL(std::imag(*eps0))};
                    delta_initialization = rytov_macro_delta_wrapper(
                        incident_device.incident, nrig5_macro_buffers.epsilon,
                        device_delta, background_epsilon, *p_niso, *p_nmax, nbs);
                }
                if (delta_initialization != cudaSuccess) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA dark-field Green input initialization failed");
                    return;
                }
                const int *p_ntotalm = ntotalm;
                const int *p_ntotal = ntotal;
                const int *p_ndipole = ndipole;
                const int *p_nxm = nxm;
                const int *p_nym = nym;
                const int *p_nzm = nzm;
                const int *p_nx = nx;
                const int *p_ny = ny;
                const int *p_nz = nz;
                const int *p_nx2 = nx2;
                const int *p_ny2 = ny2;
                const int *p_nxy2 = nxy2;
                const int *p_nz2 = nz2;
                const int64_t *p_planf = planf;
                const int64_t *p_planb = planb;
                const CUCOMPLEX *tensor_xx = device_green[0];
                const CUCOMPLEX *tensor_xy = device_green[1];
                const CUCOMPLEX *tensor_xz = device_green[2];
                const CUCOMPLEX *tensor_yy = device_green[3];
                const CUCOMPLEX *tensor_yz = device_green[4];
                const CUCOMPLEX *tensor_zz = device_green[5];
                CUCOMPLEX *dummy_vectors = get_device_vectors(1);
                if (!dummy_vectors) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA dark-field FFT workspace allocation failed");
                    return;
                }
                CUCOMPLEX *vx = dummy_vectors;
                CUCOMPLEX *vy = dummy_vectors;
                CUCOMPLEX *vz = dummy_vectors;
                const int *dipmap = tabdip;
                const long long cuda_planb = static_cast<long long>(*p_planb);
                const long long cuda_planf = static_cast<long long>(*p_planf);
                produitfftmatvect3host_(tensor_xx,tensor_xy,tensor_xz,tensor_yy,
                    tensor_yz,tensor_zz,vx,vy,vz,dipmap,p_ntotalm,p_ntotal,
                    p_nmax,p_ndipole,p_nxm,p_nym,p_nzm,p_nx,p_ny,p_nz,p_nx2,
                    p_ny2,p_nxy2,p_nz2,device_initial,device_delta,&cuda_planb,&cuda_planf);
                cudaError_t local_status = cudaSuccess;
                if (*p_nrig == 3) {
                    local_status = cudaMemcpy(device_local, device_delta,
                        field_count * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice);
                } else {
                    local_status = rytov_reference_local_wrapper(device_reference,
                        device_delta, device_local, nbs);
                }
                if (local_status == cudaSuccess)
                    local_status = cudaMemcpy(device_scattered, device_local,
                        field_count * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice);
                if (local_status == cudaSuccess)
                    local_status = polarization_inplace_wrapper(
                        dark_field_device_buffers.polarization,
                        device_scattered, *p_niso, *p_nmax, nbs);
                if (local_status == cudaSuccess)
                    device_scattered_for_diffraction = device_scattered;
                if (local_status != cudaSuccess ||
                    cuda_data::copy_device_to_host(initial, device_initial,
                        field_count * sizeof(CUCOMPLEX),
                        cuda_utils::TransferCategory::Result) != cudaSuccess ||
                    cuda_data::copy_device_to_host(local, device_local,
                        field_count * sizeof(CUCOMPLEX),
                        cuda_utils::TransferCategory::Result) != cudaSuccess ||
                    cuda_data::copy_device_to_host(scattered, device_scattered,
                        field_count * sizeof(CUCOMPLEX),
                        cuda_utils::TransferCategory::Result) != cudaSuccess) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA dark-field Green/polarization stage failed");
                    return;
                }
                release_gpbicg_device_b_alias(incident);
            } else if (*p_nrig == 6 || *p_nrig == 7) {
                release_gpbicg_device_b_alias(incident);
                const int *p_niso = niso;
                const int *p_ndipole = ndipole;
                const int *p_nx = nx, *p_ny = ny;
                const int *p_nz = nz, *p_nxm = nxm;
                const int *p_nym = nym, *p_nzm = nzm;
                const int *p_nmax = nmax;
                const int *p_nfft = nfft2d;
                const int resident_mode = 0;
                const int64_t *p_plan2f = plan2f;
                const int64_t *p_plan2b = plan2b;
                REAL xgaus = 0.0, ygaus = 0.0, zgaus = 0.0;
                const REAL theta_arg = theta;
                const REAL phi_arg = phi;
                const COMPLEX *eps = epsilon;
                COMPLEX *fourierx = Efourierx;
                COMPLEX *fouriery = Efouriery;
                COMPLEX *fourierz = Efourierz;
                char beamname[64];
                memset(beamname, ' ', sizeof beamname);
                memcpy(beamname, "pwavelinear", 11);
                if (!negative_side)
                    for (int j = 0; j < 3 * *p_nmax; ++j) incident[j] = 0.0;
                if (*p_nrig == 6) {
                    beampropagationmacro_(xs,ys,zs,&arete,&kval,p_w0,&E0,&ss,&pp,
                        &theta_arg,&phi_arg,&xgaus,&ygaus,&zgaus,beamname,eps,
                        p_niso,p_ndipole,p_nx,p_ny,p_nz,p_nxm,p_nym,p_nzm,p_nmax,
                        p_nfft,fourierx,fouriery,fourierz,incident,local,scattered,
                        p_plan2f,p_plan2b,p_nstop,infostr,64,infostr_len);
                } else {
                    beampropagation_(xs,ys,zs,&arete,&kval,p_w0,&E0,&ss,&pp,
                        &theta_arg,&phi_arg,&xgaus,&ygaus,&zgaus,beamname,eps,
                        p_niso,p_ndipole,p_nx,p_ny,p_nz,p_nxm,p_nym,p_nzm,p_nmax,
                        p_nfft,fourierx,fouriery,fourierz,incident,local,scattered,
                        p_plan2f,p_plan2b,p_nstop,infostr,&resident_mode,64,infostr_len);
                }
                if (*p_nstop != 0) return;
                if (microsbf_polarization_fields_cuda(
                        &nbs,p_nmax,p_niso,polar,local,scattered) != 0) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA polarization failed");
                    return;
                }
            } else if (*p_nrig == 8 || *p_nrig == 9) {
                release_gpbicg_device_b_alias(incident);
                REAL x0 = 0.0, y0 = 0.0, z0 = 0.0;
                COMPLEX uinc[3];
                REAL emod;
                REAL tol = *p_tolinit;
                const REAL *p_k0 = &kval;
                const int *p_niso = niso;
                const int *p_nmax = nmax;
                const int *p_ntotalm = ntotalm;
                const int *p_ntotal = ntotal;
                const int *p_ldabi = ldabi;
                const int *p_nlar = nlar;
                const int *p_nxm = nxm, *p_nym = nym;
                const int *p_nzm = nzm, *p_nx = nx;
                const int *p_ny = ny, *p_nz = nz;
                const int *p_nx2 = nx2, *p_ny2 = ny2;
                const int *p_nxy2 = nxy2, *p_nz2 = nz2;
                const int *p_nlim = nlim;
                REAL *p_tol1 = tol1;
                int *p_nloop = nloop, *p_ncompte = ncompte;
                char *method = methodeit;
                const int64_t *p_planf = planf;
                const int64_t *p_planb = planb;
                COMPLEX *tensor_xx = FFTTENSORxx;
                COMPLEX *tensor_yy = FFTTENSORyy;
                COMPLEX *tensor_zz = FFTTENSORzz;
                COMPLEX *vx = vectx;
                ondeplanekxky_(&x0,&y0,&z0,p_k0,&E0,&ss,&pp,&kxinc,&kyinc,
                    &uinc[0],&uinc[1],&uinc[2],p_nstop,infostr,infostr_len);
                if (*p_nstop != 0) return;
                emod = sqrt(pow(std::abs(uinc[0]),2.0)+pow(std::abs(uinc[1]),2.0)+pow(std::abs(uinc[2]),2.0));
                for (int comp = 0; comp < 3; ++comp) uinc[comp] /= emod;
                *p_tol1 = 0.0;
                *ncompte = 0;
                *nloop = 0;
                if (*nproche == -1) {
                    COMPLEX *scalar_ff = new (std::nothrow) COMPLEX[(size_t)nbs]();
                    COMPLEX *scalar_local = new (std::nothrow) COMPLEX[(size_t)nbs]();
                    if (scalar_ff == NULL || scalar_local == NULL) {
                        delete[] scalar_ff; delete[] scalar_local;
                        microsdf_set_info(infostr, infostr_len, "allocation failure");
                        *p_nstop = 1;
                        return;
                    }
                    inverserigscalaire_(tensor_xx,vx,tabdip,scalar_ff,scalar_local,
                        &uinc[0],&uinc[1],&uinc[2],p_ntotalm,p_ntotal,p_ldabi,
                        p_nlar,p_nmax,ndipole,p_nxm,p_nym,p_nzm,
                        p_nx,p_ny,p_nz,p_nx2,p_ny2,p_nxy2,p_nz2,&nbs,initial,
                        real_work,wrk,scattered,incident,local,
                        polar,p_niso,method,&tol,p_tol1,p_nloop,p_ncompte,p_nlim,
                        p_planf,p_planb,p_nstop,infostr,methodeit_len,infostr_len);
                    delete[] scalar_ff; delete[] scalar_local;
                } else {
                    inverserigscalaire_(tensor_xx,vx,tabdip,tensor_yy,tensor_zz,
                        &uinc[0],&uinc[1],&uinc[2],p_ntotalm,p_ntotal,p_ldabi,
                        p_nlar,p_nmax,ndipole,p_nxm,p_nym,p_nzm,p_nx,p_ny,p_nz,
                        p_nx2,p_ny2,p_nxy2,p_nz2,&nbs,initial,real_work,wrk,
                        scattered,incident,local,polar,p_niso,method,&tol,p_tol1,
                        p_nloop,p_ncompte,p_nlim,p_planf,p_planb,p_nstop,infostr,
                        methodeit_len,infostr_len);
                }
                if (*p_nstop == 1) return;
            }
        /* Diffraction and image accumulation for this illumination. */
        if (quick == 1) {
            int imax = imaxk0;
            REAL dkx = deltakx, dky = deltakx;
            const CUCOMPLEX *lens_input = nullptr;
            if (device_scattered_for_diffraction) {
                if (!cuda_utils::check_cuda(microscope_expand_packed_field_wrapper(
                        device_scattered_for_diffraction,
                        device_microscope_dipole_map,
                        dark_field_device_buffers.image_scratch,
                        int(microscope_mesh_cell_count), nbs),
                        "dark-field lens diffraction",
                        "expand packed scattered field")) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA dark-field lens field expansion failed");
                    return;
                }
                lens_input = dark_field_device_buffers.image_scratch;
            }
            diffractefft2dlens_set_resident_input(lens_input);
            diffractefft2dlens_set_resident_output(1);
            diffractefft2dlens_(nx,ny,nz,
                nxm,nym,nzm,p_nfft2d,
                tabfft2,&kval,xs,ys,
                zs,&arete,fourierx,fouriery,fourierz,scattered,
                &imax,&dkx,&dky,diffpos,p_numaper,nside,
                plan2f,plan2b,p_nstop,
                infostr,infostr_len);
            diffractefft2dlens_set_resident_input(nullptr);
            diffractefft2dlens_set_resident_output(0);
            if (*p_nstop == 1) return;
            deltakx = dkx;
            deltak_y = dky;
            deltak = dkx;
            imaxk0 = imax;
            final_device_spectrum = diffractefft2dlens_get_resident_output();
            if (!final_device_spectrum) {
                *p_nstop = 1;
                microsdf_set_info(infostr, infostr_len,
                    "CUDA dark-field lens spectrum is unavailable");
                return;
            }
        } else {
            const CUCOMPLEX *diffraction_input = device_scattered_for_diffraction;
            if (!diffraction_input) {
                std::vector<CUCOMPLEX> host_scattered;
                try {
                    host_scattered.resize(microscope_field_count);
                } catch (...) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA dark-field field staging allocation failed");
                    return;
                }
                for (size_t index = 0; index < microscope_field_count; ++index)
                    host_scattered[index] = CUCOMPLEX{
                        CU_REAL(std::real(scattered[index])),
                        CU_REAL(std::imag(scattered[index]))};
                if (cuda_data::copy_host_to_device(dark_field_device_buffers.image_scratch,
                        host_scattered.data(), microscope_field_count * sizeof(CUCOMPLEX),
                        cuda_utils::TransferCategory::Result) != cudaSuccess) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA dark-field field upload failed");
                    return;
                }
                diffraction_input = dark_field_device_buffers.image_scratch;
            }
            if (!cuda_utils::check_cuda(confocal_direct_diffraction_wrapper(
                    diffraction_input,
                    incident_device.xs, incident_device.ys, incident_device.zs,
                    nbs, nfft, imaxk0, negative_side ? -1 : 1,
                    CU_REAL(kval), CU_REAL(*numaper), CU_REAL(deltakx),
                    CU_REAL(deltak_y), dark_field_device_buffers.spectrum_planes),
                    "dark-field diffraction", "compute dipole spectrum")) {
                *p_nstop = 1;
                microsdf_set_info(infostr, infostr_len,
                    "CUDA dark-field direct diffraction failed");
                return;
            }
            final_device_spectrum = dark_field_device_buffers.spectrum_planes;
        }
        if (use_cuda_image_pipeline && !process_cuda_image(final_device_spectrum))
            return;
        if (*p_nstop == -1) {
            microsdf_set_info(infostr, infostr_len,
                "Calculation cancelled during iterative method");
            return;
        }
        }

        /* Keep array aliases live for the next conversion pass. */
        (void)p_nmax; (void)p_niso;
        (void)wrk; (void)tol1; (void)tolinit; (void)ncompte;
        (void)nloop; (void)nlim; (void)planf; (void)planb;
        (void)Ediffkzpos; (void)Ediffkzneg; (void)kxy; (void)xy;
        (void)nside; (void)planXf; (void)planXb; (void)planYf;
        (void)planYb; (void)planZf; (void)planZb; (void)plan2f;
        (void)plan2b; (void)nmat; (void)file_id; (void)group_idmic;
        (void)FFTTENSORxx; (void)FFTTENSORxy; (void)FFTTENSORxz;
        (void)FFTTENSORyy; (void)FFTTENSORyz; (void)FFTTENSORzz;
        (void)vectx; (void)vecty; (void)vectz; (void)ntotalm; (void)ntotal;
        (void)ldabi; (void)nlar; (void)nxm; (void)nym; (void)nzm;
        (void)nx; (void)ny; (void)nz; (void)nx2; (void)ny2; (void)nxy2;
        (void)nz2; (void)ndipole; (void)nbsphere3; (void)nproche;
        (void)tabfft2; (void)tabdip; (void)XR; (void)FF;
        (void)epsilon; (void)methodeit; (void)trope;
        (void)nquicklens; (void)gross; (void)zlens; (void)Efourierx;
        (void)Efouriery; (void)Efourierz; (void)Efourierincx;
        (void)Efourierincy; (void)Efourierincz;

    if (use_cuda_image_pipeline) {
        try {
            std::vector<CU_REAL> host_intensity(6u * microscope_plane_count);
            if (microscope_intensity_sqrt_wrapper(
                    dark_image_buffers.image_intensity,
                    host_intensity.size()) != cudaSuccess ||
                cuda_data::copy_device_to_host(host_intensity.data(),
                    dark_image_buffers.image_intensity,
                    host_intensity.size() * sizeof(CU_REAL),
                    cuda_utils::TransferCategory::Result) != cudaSuccess) {
                *p_nstop = 1;
                microsdf_set_info(infostr, infostr_len,
                    "CUDA dark-field image download failed");
                return;
            }
            for (size_t pixel = 0; pixel < microscope_plane_count; ++pixel) {
                imagex[pixel] = COMPLEX(REAL(host_intensity[pixel]), 0);
                imagey[pixel] = COMPLEX(REAL(host_intensity[
                    microscope_plane_count + pixel]), 0);
                imagez[pixel] = COMPLEX(REAL(host_intensity[
                    2u * microscope_plane_count + pixel]), 0);
                imageix[pixel] = COMPLEX(REAL(host_intensity[
                    3u * microscope_plane_count + pixel]), 0);
                imageiy[pixel] = COMPLEX(REAL(host_intensity[
                    4u * microscope_plane_count + pixel]), 0);
                imageiz[pixel] = COMPLEX(REAL(host_intensity[
                    5u * microscope_plane_count + pixel]), 0);
            }
            if (*nmat == 2) {
                std::vector<CUCOMPLEX> host_spectrum(3u * microscope_plane_count);
                if (cuda_data::copy_device_to_host(host_spectrum.data(),
                        final_device_spectrum,
                        host_spectrum.size() * sizeof(CUCOMPLEX),
                        cuda_utils::TransferCategory::Result) != cudaSuccess) {
                    *p_nstop = 1;
                    microsdf_set_info(infostr, infostr_len,
                        "CUDA dark-field spectrum download failed");
                    return;
                }
                for (size_t index = 0; index < host_spectrum.size(); ++index)
                    diffpos[index] = COMPLEX(host_spectrum[index].x,
                                             host_spectrum[index].y);
            }
        } catch (...) {
            *p_nstop = 1;
            microsdf_set_info(infostr, infostr_len,
                "CUDA dark-field image staging allocation failed");
            return;
        }
    }
    {
        const REAL deltax=2.0*pi/(REAL)nfft/deltakx;
        const int *p_nside=nside;
        REAL *const kxout=kxy;
        REAL *const xyout=xy;
        (void)p_nside;
        for (int ix=-half;ix<half;++ix) {
            const int out=ix+half;
            xyout[out]=deltax*(REAL)ix*(*gross);
            kxout[out]=deltakx*(REAL)ix/kval;
        }
    }
    if (*nmat == 0) {
        FILE *f_k = fopen("kxfourier.mat","w");
        FILE *f_x = fopen("ximage.mat","w");
        FILE *f_kx = fopen("kxincidentdf.mat","w");
        FILE *f_ky = fopen("kyincidentdf.mat","w");
        const char *names[8] = {"imagedf.mat","imagedfx.mat","imagedfy.mat",
            "imagedfz.mat","imageincdf.mat","imageincdfx.mat",
            "imageincdfy.mat","imageincdfz.mat"};
        FILE *images[8];
        const int image_file_count=negative_side?4:8;
        if (f_k==NULL || f_x==NULL || f_kx==NULL || f_ky==NULL) {
            if(f_k) fclose(f_k);
            if(f_x) fclose(f_x);
            if(f_kx) fclose(f_kx);
            if(f_ky) fclose(f_ky);
            microsdf_set_info(infostr,infostr_len,"Unable to open microscopy output file");
            *p_nstop=1;
            return;
        }
        for(int q=0;q<image_file_count;++q) {
            images[q]=fopen(names[q],"w");
            if(images[q]==NULL) {
                for(int j=0;j<q;++j)fclose(images[j]);
                fclose(f_k);fclose(f_x);fclose(f_kx);fclose(f_ky);
                microsdf_set_info(infostr,infostr_len,"Unable to open microscopy output file");
                *p_nstop=1;
                return;
            }
        }
        for(int ix=-half;ix<half;++ix)fprintf(f_x," % .16E\n",(xy)[ix+half]);
        for(int ix=-imaxk0;ix<=imaxk0;++ix)fprintf(f_k," % .16E\n",deltakx*(REAL)ix/kval);
        for(int angle=0;angle<ideltam;++angle) {
            const REAL phi=2.0*pi*(REAL)angle/(REAL)ideltam;
            fprintf(f_kx," % .16E\n",cos(phi)* *p_numaperinc);
            fprintf(f_ky," % .16E\n",sin(phi)* *p_numaperinc);
        }
        for(int pixel=0;pixel<nfft*nfft;++pixel) {
            const COMPLEX a=imagex[pixel],b=imagey[pixel],c=imagez[pixel];
            const REAL intensity=sqrt(std::real(a*a+b*b+c*c));
            fprintf(images[0]," % .16E\n",intensity);
            fprintf(images[1]," % .16E\n",std::real(a));
            fprintf(images[2]," % .16E\n",std::real(b));
            fprintf(images[3]," % .16E\n",std::real(c));
            if(!negative_side) {
                const COMPLEX ai=imageix[pixel],bi=imageiy[pixel],ci=imageiz[pixel];
                const REAL incintensity=sqrt(std::real(ai*ai+bi*bi+ci*ci));
                fprintf(images[4]," % .16E\n",incintensity);
                fprintf(images[5]," % .16E\n",std::real(ai));
                fprintf(images[6]," % .16E\n",std::real(bi));
                fprintf(images[7]," % .16E\n",std::real(ci));
            }
        }
        fclose(f_k);fclose(f_x);fclose(f_kx);fclose(f_ky);
        for(int q=0;q<image_file_count;++q)fclose(images[q]);
    } else if (*nmat == 2) {
        int dims[4]={nfft,nfft,0,0};
        int zero=0;
        char dataset[100];
        memset(dataset,' ',sizeof dataset);memcpy(dataset,"x Image",7);
        hdf5write1d_i8_(group_idmic,dataset,xy,dims,(size_t)100);
        memset(dataset,' ',sizeof dataset);memcpy(dataset,negative_side?"Image dark field kz<0":"Image dark field kz>0",21);
        writehdf5mic_(Eimagex,Eimagey,Eimagez,nfft2d,&imaxk0,diffpos,&zero,dataset,group_idmic,(size_t)100);
        if(!negative_side) {
            memset(dataset,' ',sizeof dataset);memcpy(dataset,"Image+incident dark field kz>0",30);
            writehdf5mic_(Eimageincx,Eimageincy,Eimageincz,nfft2d,&imaxk0,diffpos,&zero,dataset,group_idmic,(size_t)100);
        }
        for(int angle=0;angle<ideltam;++angle) {
            const REAL phi=2.0*pi*(REAL)angle/(REAL)ideltam;
            (kxy)[angle]=cos(phi)* *p_numaperinc;
        }
        dims[0]=ideltam; dims[1]=nfft;
        memset(dataset,' ',sizeof dataset);memcpy(dataset,"kx incident df",14);
        hdf5write1d_i8_(group_idmic,dataset,kxy,dims,(size_t)100);
        for(int angle=0;angle<ideltam;++angle) {
            const REAL phi=2.0*pi*(REAL)angle/(REAL)ideltam;
            (kxy)[angle]=sin(phi)* *p_numaperinc;
        }
        memset(dataset,' ',sizeof dataset);memcpy(dataset,"ky incident df",14);
        hdf5write1d_i8_(group_idmic,dataset,kxy,dims,(size_t)100);
    }
    }

}

/* The negative-z entry omits the six incident-image buffers.  It shares the
 * C99 numerical core while selecting the negative hemisphere. */
//-------------------------------------------
extern "C" void microsdfneg_(
    COMPLEX *FFTTENSORxx, COMPLEX *FFTTENSORxy, COMPLEX *FFTTENSORxz,
    COMPLEX *FFTTENSORyy, COMPLEX *FFTTENSORyz, COMPLEX *FFTTENSORzz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    int *ntotalm, int *ntotal, int *ldabi, int *nlar, int *nmax,
    int *nxm, int *nym, int *nzm, int *nx, int *ny, int *nz,
    int *nx2, int *ny2, int *nxy2, int *nz2, int *ndipole,
    int *nbsphere, int *nbsphere3, int *nproche, int *nrig,
    int *nfft2d, int *tabfft2, int *tabdip, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *wrk, COMPLEX *FF, COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *polarisa,
    int *niso, COMPLEX *epsilon, char *methodeit, char *trope,
    REAL *tolinit, REAL *tol1, int *nloop, int *ncompte, int *nlim,
    REAL *xs, REAL *ys, REAL *zs, REAL *aretecube, REAL *numaper,
    REAL *numaperinc, int *npolainc_in, int *nquicklens, COMPLEX *eps0,
    REAL *k0, REAL *P0, REAL *irra, REAL *w0, REAL *gross, REAL *zlens,
    COMPLEX *Eimagex, COMPLEX *Eimagey, COMPLEX *Eimagez,
    COMPLEX *Efourierx, COMPLEX *Efouriery, COMPLEX *Efourierz,
    COMPLEX *Ediffkzpos, COMPLEX *Ediffkzneg, REAL *kxy, REAL *xy, int *nside,
    int64_t *planXf, int64_t *planXb, int64_t *planYf, int64_t *planYb,
    int64_t *planZf, int64_t *planZb, int64_t *planf, int64_t *planb,
    int64_t *plan2f, int64_t *plan2b, int *nmat, int64_t *file_id,
    int64_t *group_idmic, int *nstop, char *infostr,
    size_t methodeit_len, size_t trope_len, size_t infostr_len)
{
    const int nfft = *nfft2d;
    const size_t nimage = (size_t)nfft * (size_t)nfft;
    COMPLEX *scratch = new (std::nothrow) COMPLEX[6u * nimage]();
    int negative_nside = -1;
    int npolainc = *npolainc_in;
    (void)nside;
    if (scratch == NULL) {
        *nstop = 1;
        microsdf_set_info(infostr, infostr_len,
                          "Unable to allocate microscopy buffers");
        return;
    }
    microsdf_(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz,FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,
        vectx,vecty,vectz,ntotalm,ntotal,ldabi,nlar,nmax,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,ndipole,
        nbsphere,nbsphere3,nproche,nrig,nfft2d,tabfft2,tabdip,XI,XR,wrk,FF,FF0,FFloc,polarisa,niso,epsilon,
        methodeit,trope,tolinit,tol1,nloop,ncompte,nlim,xs,ys,zs,aretecube,numaper,numaperinc,&npolainc,nquicklens,
        eps0,k0,P0,irra,w0,gross,zlens,Eimagex,Eimagey,Eimagez,Efourierx,Efouriery,Efourierz,
        scratch,scratch+nimage,scratch+2u*nimage,scratch+3u*nimage,scratch+4u*nimage,scratch+5u*nimage,
        Ediffkzpos,Ediffkzneg,kxy,xy,&negative_nside,planXf,planXb,planYf,planYb,planZf,planZb,
        planf,planb,plan2f,plan2b,nmat,file_id,group_idmic,nstop,infostr,
        methodeit_len,trope_len,infostr_len);
    delete[] scratch;
}
