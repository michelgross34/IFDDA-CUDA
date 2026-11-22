#include "kernel.h"
#define IFDDA_CUDA_CALLERS_SAVEMACROSCOPICFIELD_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_SAVEMACROSCOPICFIELD_CUDA

#include <cstdio>
#include <cstring>
#include <vector>





namespace {

//-----------------------------------------------------------
void write_complex_record(FILE *file, CUCOMPLEX value)
{
    if (file)
    {
        std::fprintf(file, " %.17g %.17g\n", double(CU_REAL(value.x)),
                     double(CU_REAL(value.y)));
    }
}

//-----------------------------------------------------------
void write_real_record(FILE *file, CU_REAL value)
{
    if (file)
    {
        std::fprintf(file, " %.17g\n", double(value));
    }
}

//-----------------------------------------------------------
void write_hdf5_real(
    const int *group_id,
    const char *name,
    const CU_REAL *data,
    const int *dimensions)
{
    char fortran_name[100];
    std::memset(fortran_name, ' ', sizeof(fortran_name));
    const size_t length = std::strlen(name);
    std::memcpy(fortran_name, name, length);
    hdf5write1d_(group_id, fortran_name, data, dimensions, sizeof(fortran_name));
}

//-----------------------------------------------------------
}

//-----------------------------------------------------------
extern "C" void savemacroscopicfield_(
    const int *nmat,
    const int *ndipole,
    const int *nmax,
    const int *nlar,
    const int *testgreen,
    const int *tabdip,
    const CU_REAL *aretecube,
    const CU_REAL *k0,
    const CUCOMPLEX *FFloc,
    const CUCOMPLEX *epsilon,
    const int *niso,
    CU_REAL *macroscopicfield,
    CUCOMPLEX *macroscopicfieldx,
    CUCOMPLEX *macroscopicfieldy,
    CUCOMPLEX *macroscopicfieldz,
    CUCOMPLEX *wrk,
    const int *group_idnf)
{
    if (!nmat || !ndipole || !nmax || !nlar || !testgreen || !tabdip ||
        !aretecube || !k0 || !FFloc || !epsilon || !niso ||
        !macroscopicfield || !macroscopicfieldx || !macroscopicfieldy ||
        !macroscopicfieldz || !wrk || !group_idnf || *nmax < 1 ||
        *ndipole < 0 || *ndipole > *nmax || (*niso != 1 && *niso != 3))
    {
        return;
    }

    std::vector<CUCOMPLEX> field_x(static_cast<size_t>(*nmax));
    std::vector<CUCOMPLEX> field_y(static_cast<size_t>(*nmax));
    std::vector<CUCOMPLEX> field_z(static_cast<size_t>(*nmax));
    std::vector<CU_REAL> magnitude(static_cast<size_t>(*nmax));
    int status = 1;
    int count = *nmax;
    int sensitivity = 1;
    if (count > 0)
    {
        ifdda_local_macro_batch_cuda_(
            epsilon, FFloc, field_x.data(), field_y.data(), field_z.data(),
            magnitude.data(), nmax, niso, &count, aretecube, k0,
            &sensitivity, &status);
    }
    if (status != 0)
    {
        return;
    }

    const int green_kind = *testgreen;
    const bool ascii_field = (*nmat == 0 && green_kind == 0);
    const bool ascii_green = (*nmat == 0 && green_kind > 1);
    if (*nmat == 0 && !ascii_field && !ascii_green)
    {
        return;
    }
    FILE *out_x = nullptr;
    FILE *out_y = nullptr;
    FILE *out_z = nullptr;
    FILE *out_magnitude = nullptr;
    if (ascii_field)
    {
        out_x = std::fopen("macroscopicfieldx.mat", "a");
        out_y = std::fopen("macroscopicfieldy.mat", "a");
        out_z = std::fopen("macroscopicfieldz.mat", "a");
        out_magnitude = std::fopen("macroscopicfield.mat", "a");
    }
    if (ascii_green)
    {
        const int x_unit = green_kind == 2 ? 70 : green_kind == 3 ? 73 : 76;
        const int y_unit = x_unit + 1;
        const int z_unit = x_unit + 2;
        const char *x_file = x_unit == 70 ? "greenxx.mat" :
                             x_unit == 73 ? "greenxy.mat" : "greenxz.mat";
        const char *y_file = y_unit == 71 ? "greenyx.mat" :
                             y_unit == 74 ? "greenyy.mat" : "greenyz.mat";
        const char *z_file = z_unit == 72 ? "greenzx.mat" :
                             z_unit == 75 ? "greenzy.mat" : "greenzz.mat";
        out_x = std::fopen(x_file, "a");
        out_y = std::fopen(y_file, "a");
        out_z = std::fopen(z_file, "a");
    }

    for (int i = 0; i < *ndipole; ++i)
    {
        const int k = tabdip[i] - 1;
        if (k < 0 || k >= *nmax)
        {
            if (ascii_field)
            {
                write_complex_record(out_x, CUCOMPLEX{CU_REAL(0), CU_REAL(0)});
                write_complex_record(out_y, CUCOMPLEX{CU_REAL(0), CU_REAL(0)});
                write_complex_record(out_z, CUCOMPLEX{CU_REAL(0), CU_REAL(0)});
                write_real_record(out_magnitude, CU_REAL(0));
            }
            if (ascii_green)
            {
                write_complex_record(out_x, CUCOMPLEX{CU_REAL(0), CU_REAL(0)});
                write_complex_record(out_y, CUCOMPLEX{CU_REAL(0), CU_REAL(0)});
                write_complex_record(out_z, CUCOMPLEX{CU_REAL(0), CU_REAL(0)});
            }
            continue;
        }

        macroscopicfieldx[k] = field_x[size_t(k)];
        macroscopicfieldy[k] = field_y[size_t(k)];
        macroscopicfieldz[k] = field_z[size_t(k)];
        macroscopicfield[k] = magnitude[size_t(k)];

        if (ascii_field || ascii_green)
        {
            write_complex_record(out_x, field_x[size_t(k)]);
            write_complex_record(out_y, field_y[size_t(k)]);
            write_complex_record(out_z, field_z[size_t(k)]);
        }
        if (ascii_field)
        {
            write_real_record(out_magnitude, magnitude[size_t(k)]);
        }
    }

    if (out_x)
    {
        std::fclose(out_x);
    }
    if (out_y)
    {
        std::fclose(out_y);
    }
    if (out_z)
    {
        std::fclose(out_z);
    }
    if (out_magnitude)
    {
        std::fclose(out_magnitude);
    }

    if (*nmat != 0)
    {
        if (*nmat == 2)
        {
            for (int i = 0; i < *ndipole; ++i)
            {
                const int k = tabdip[i] - 1;
                if (k < 0 || k >= *nmax)
                {
                    wrk[i] = CUCOMPLEX{CU_REAL(0), CU_REAL(0)};
                    wrk[size_t(3) * size_t(*nmax) + size_t(i)] =
                        CUCOMPLEX{CU_REAL(0), CU_REAL(0)};
                    wrk[size_t(6) * size_t(*nmax) + size_t(i)] =
                        CUCOMPLEX{CU_REAL(0), CU_REAL(0)};
                    wrk[size_t(9) * size_t(*nmax) + size_t(i)] =
                        CUCOMPLEX{CU_REAL(0), CU_REAL(0)};
                    continue;
                }
                const CUCOMPLEX ex = field_x[size_t(k)];
                const CUCOMPLEX ey = field_y[size_t(k)];
                const CUCOMPLEX ez = field_z[size_t(k)];
                wrk[i] = ex;
                wrk[size_t(3) * size_t(*nmax) + size_t(i)] = ey;
                wrk[size_t(6) * size_t(*nmax) + size_t(i)] = ez;
                wrk[size_t(9) * size_t(*nmax) + size_t(i)] =
                    CUCOMPLEX{magnitude[size_t(k)], CU_REAL(0)};
            }

            int dimensions[2] = {*ndipole, *nmax * 3};
            std::vector<CU_REAL> real_data(static_cast<size_t>(*ndipole));
            const char *names[7] = {};
            if (green_kind == 0)
            {
                const char *macro_names[7] = {
                    "Macroscopic field modulus",
                    "Macroscopic field x component real part",
                    "Macroscopic field x component imaginary part",
                    "Macroscopic field y component real part",
                    "Macroscopic field y component imaginary part",
                    "Macroscopic field z component real part",
                    "Macroscopic field z component imaginary part"};
                std::memcpy(names, macro_names, sizeof(names));
            }
            else if (green_kind == 2)
            {
                const char *green_names[7] = {
                    "",
                    "Green tensor xx component real part",
                    "Green tensor xx component imaginary part",
                    "Green tensor yx component real part",
                    "Green tensor yx component imaginary part",
                    "Green tensor zx component real part",
                    "Green tensor zx component imaginary part"};
                std::memcpy(names, green_names, sizeof(names));
            }
            else if (green_kind == 3)
            {
                const char *green_names[7] = {
                    "",
                    "Green tensor xy component real part",
                    "Green tensor xy component imaginary part",
                    "Green tensor yy component real part",
                    "Green tensor yy component imaginary part",
                    "Green tensor zy component real part",
                    "Green tensor zy component imaginary part"};
                std::memcpy(names, green_names, sizeof(names));
            }
            else if (green_kind == 4)
            {
                const char *green_names[7] = {
                    "",
                    "Green tensor xz component real part",
                    "Green tensor xz component imaginary part",
                    "Green tensor yz component real part",
                    "Green tensor yz component imaginary part",
                    "Green tensor zz component real part",
                    "Green tensor zz component imaginary part"};
                std::memcpy(names, green_names, sizeof(names));
            }
            for (int part = 0; part < 7; ++part)
            {
                if (names[part] == nullptr || names[part][0] == '\0')
                {
                    continue;
                }
                std::vector<CU_REAL> real_data(static_cast<size_t>(*ndipole));
                for (int i = 0; i < *ndipole; ++i)
                {
                    const int k = tabdip[i] - 1;
                    CUCOMPLEX value{CU_REAL(0), CU_REAL(0)};
                    if (k >= 0 && k < *nmax)
                    {
                        if (part == 0)
                        {
                            value.x = magnitude[size_t(k)];
                        }
                        else
                        {
                            const int component = (part - 1) / 2;
                            value = component == 0 ? field_x[size_t(k)] :
                                    component == 1 ? field_y[size_t(k)] :
                                                     field_z[size_t(k)];
                        }
                    }
                    real_data[size_t(i)] =
                        part == 0 || (part % 2 == 1) ? value.x : value.y;
                }
                write_hdf5_real(group_idnf, names[part], real_data.data(), dimensions);
            }
        }
    }

    (void)nlar;
}
