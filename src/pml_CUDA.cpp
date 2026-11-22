#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <complex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <vector>

//-----------------------------------------------------------
// Scalar PML grid setup stays on the host; the PML mask and field operations
// are evaluated by the CUDA kernels below.
extern "C" void initnrbc_(const REAL *k0, REAL *lcos, REAL *lnrbc,
                          REAL *lpml, REAL *taillemax, const int *nnnr,
                          int *nrbc, int *npml, int *pmlwrk, REAL *aretecube) {
    if (!k0 || !lcos || !lnrbc || !lpml || !taillemax || !nnnr || !nrbc ||
        !npml || !pmlwrk || !aretecube || *k0 <= CU_REAL(0) || *nnnr <= 0)
        return;

    constexpr CU_REAL pi = CU_REAL(3.1415926535897932384626433832795);
    const CU_REAL lambda = CU_REAL(2) * pi / *k0;
    *lcos *= lambda;
    *lnrbc *= lambda;
    *lpml = *lcos + *lnrbc;
    const CU_REAL border = *lpml;
    *taillemax = CU_REAL(2) * border + *taillemax;
    *aretecube = *taillemax / CU_REAL(*nnnr);
    const auto fortran_nint = [](CU_REAL value) {
        return static_cast<int>(std::floor(value + CU_REAL(0.5)));
    };
    *npml = fortran_nint(*lpml / *aretecube);
    *nrbc = fortran_nint(*lnrbc / *aretecube);

    std::fprintf(stdout, " object in absorbing background : compute PML\n");
    std::fprintf(stdout, " total size   : %g %g lambda\n", double(*taillemax),
                 double(*taillemax / lambda));
    std::fprintf(stdout, " size PML     : %g %g lambda\n", double(*lpml),
                 double(*lpml / lambda));
    std::fprintf(stdout, " size NRBC    : %g %g lambda\n",
                 double(CU_REAL(*nrbc) * *aretecube),
                 double(CU_REAL(*nrbc) * *aretecube / lambda));
    std::fprintf(stdout, " size COSINUS : %g %g lambda\n",
                 double(CU_REAL(*npml - *nrbc) * *aretecube),
                 double(CU_REAL(*npml - *nrbc) * *aretecube / lambda));
    std::fprintf(stdout, " size edge    : %g\n", double(border));

    std::ofstream pml_file("pml.dat", std::ios::out | std::ios::trunc);
    if (pml_file)
        pml_file << *npml << '\n';

    pmlwrk[0] = *nrbc;
    pmlwrk[1] = *npml;
    pmlwrk[2] = fortran_nint(border / *aretecube);
    pmlwrk[3] = *nnnr;
}

//-----------------------------------------------------------
extern "C" void ifdda_apply_pml_mask_cuda_(const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
                                           const COMPLEX *epsilon, const CU_REAL *k0,
                                           const int *pml_work, const int *mask_type,
                                           const int *dipole_count, const CU_REAL *spacing,
                                           COMPLEX *field, CU_REAL *mask, int *status_out) {
    if (status_out)
        *status_out = 1;
    if (!xs || !ys || !zs || !epsilon || !k0 || !pml_work || !mask_type || !dipole_count ||
        !spacing || !field || !mask || !status_out || *dipole_count <= 0 ||
        *spacing <= CU_REAL(0) || *mask_type != 1 || pml_work[0] < 0 || pml_work[1] < pml_work[0])
        return;

    const size_t count = size_t(*dipole_count);
    if (count > size_t(std::numeric_limits<int>::max()) ||
        count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)))
        return;
    if (!initialize_cuda_device())
        return;

    const CU_REAL lneps = CU_REAL(pml_work[0]) * *spacing;
    const CU_REAL lpml = CU_REAL(pml_work[1]) * *spacing;
    const CU_REAL dpml = lpml - lneps;
    cuda_data::PmlDeviceBuffers buffers{};
    if (!cuda_data::get_device_pml_buffers(count, &buffers))
        return;
    const char *validate_env = std::getenv("IFDDA_VALIDATE_PML_MASK");
    const bool validate = validate_env && *validate_env && *validate_env != '0';
    std::vector<COMPLEX> original_field;
    if (validate)
        original_field.assign(field, field + count * 3);

    using cuda_utils::TransferCategory;
    cudaError_t status = cuda_data::copy_host_to_device(buffers.xs, xs, count * sizeof(CU_REAL),
                                                        TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(buffers.ys, ys, count * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(buffers.zs, zs, count * sizeof(CU_REAL),
                                                TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_complex_to_device(
            buffers.field, field, count * 3, TransferCategory::IncidentField);
    if (status != cudaSuccess)
        return;

    if (!cuda_utils::check_cuda(pml_build_mask_candidates_wrapper(
                                    buffers.xs, buffers.ys, buffers.zs, buffers.mask,
                                    buffers.z_candidates, *dipole_count, xs[0],
                                    xs[*dipole_count - 1], ys[0], ys[*dipole_count - 1], zs[0],
                                    zs[*dipole_count - 1], *spacing, lneps, lpml, dpml),
                                "PML mask", "build mask") ||
        !cuda_utils::check_cuda(pml_reduce_zinf_wrapper(buffers.z_candidates, buffers.zinf,
                                                        *dipole_count, zs[*dipole_count - 1]),
                                "PML mask", "reduce zinf"))
        return;

    CU_REAL zinf = CU_REAL(0);
    if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(&zinf, buffers.zinf, sizeof(CU_REAL),
                                                               TransferCategory::Result),
                                "PML mask", "download zinf"))
        return;

    const COMPLEX epsilon_value = *epsilon;
    const CU_REAL exponent =
        -std::imag(std::sqrt(epsilon_value)) * (zinf - *spacing / CU_REAL(2)) * *k0;
    const CU_REAL field_scale = std::exp(-exponent);
    if (!cuda_utils::check_cuda(
            pml_apply_field_wrapper(buffers.mask, buffers.field, *dipole_count, field_scale),
            "PML mask", "apply mask to incident field") ||
        !cuda_utils::check_cuda(cuda_data::copy_device_to_host(mask, buffers.mask,
                                                               count * sizeof(CU_REAL),
                                                               TransferCategory::Geometry),
                                "PML mask", "download mask") ||
        !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
                                   field, buffers.field, count * 3,
                                   TransferCategory::IncidentField),
                                "PML mask", "download masked incident field"))
        return;

    const std::complex<CU_REAL> e_expo = std::exp(exponent);
    std::fprintf(stdout, " zinf box %g (%g,%g)\n", double(zinf), double(std::real(e_expo)),
                 double(std::imag(e_expo)));
    if (validate) {
        const CU_REAL xmin = xs[0], xmax = xs[*dipole_count - 1];
        const CU_REAL ymin = ys[0], ymax = ys[*dipole_count - 1];
        const CU_REAL zmin = zs[0], zmax = zs[*dipole_count - 1];
        auto axis_mask = [lneps, lpml, dpml](CU_REAL distance) {
            if (distance <= lneps)
                return CU_REAL(0);
            if (distance <= lpml)
                return CU_REAL(0.5) *
                       (CU_REAL(1) - std::cos(CU_REAL(3.1415926535897932384626433832795) *
                                              (distance - lneps) / dpml));
            return CU_REAL(1);
        };
        double max_mask_error = 0.0;
        double max_field_error = 0.0;
        for (size_t i = 0; i < count; ++i) {
            CU_REAL mx = axis_mask(std::abs(xs[i] - xmin) + *spacing / CU_REAL(2));
            const CU_REAL right_x = std::abs(xmax - xs[i]) + *spacing / CU_REAL(2);
            if (right_x <= lpml)
                mx = axis_mask(right_x);
            CU_REAL my = axis_mask(std::abs(ys[i] - ymin) + *spacing / CU_REAL(2));
            const CU_REAL right_y = std::abs(ymax - ys[i]) + *spacing / CU_REAL(2);
            if (right_y <= lpml)
                my = axis_mask(right_y);
            CU_REAL mz =
                std::abs(zs[i] - zmin) + *spacing / CU_REAL(2) <= lpml ? CU_REAL(0) : CU_REAL(1);
            const CU_REAL upper_z = std::abs(zmax - zs[i]) + *spacing / CU_REAL(2);
            if (upper_z <= lneps)
                mz = CU_REAL(0);
            else if (upper_z <= lpml)
                mz = axis_mask(upper_z);
            const CU_REAL expected_mask = mx * my * mz;
            max_mask_error = std::max(max_mask_error, std::abs(double(mask[i] - expected_mask)));
            for (size_t component = 0; component < 3; ++component) {
                const size_t index = 3 * i + component;
                const CU_REAL expected_scale = expected_mask * field_scale;
                const double dx = double(std::real(field[index])) -
                                  double(std::real(original_field[index]) * expected_scale);
                const double dy = double(std::imag(field[index])) -
                                  double(std::imag(original_field[index]) * expected_scale);
                max_field_error = std::max(max_field_error, std::hypot(dx, dy));
            }
        }
        std::fprintf(stdout, "PML_MASK_VALIDATE max_mask_error=%.17g max_field_error=%.17g\n",
                     max_mask_error, max_field_error);
    }
    *status_out = 0;
}
