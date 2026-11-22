#include "mod_precision.h"
#include "cdmlib_cpp_abi.hpp"
#define IFDDA_CDMLIB_PRECISION_C99_H
/* C++17 test0 driver translated from the C99 port of tests/test0/main.f90. */
/* ABI aliases select the configured precision; scalars by address, column-major contiguous arrays,
   and hidden CHARACTER lengths (size_t) appended in argument order. */
#include <complex>
#include <vector>
#include <ctype.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
extern "C" {
#include "cdmlib_abi.h"
}
#ifndef IFDDA_TEST_COMPLEX
#define IFDDA_TEST_COMPLEX(real_value, imag_value) cdmlib_complex((real_value), (imag_value))
#endif

static void *alloc_array(size_t n, size_t element) {
    void *p = calloc(n ? n : 1, element);
    if (!p) { fputs("Out of memory\n", stderr); exit(EXIT_FAILURE); }
    return p;
}
static void fstring(char *s, size_t n, const char *v) {
    size_t k = strlen(v); if (k > n) k = n;
    memset(s, 32, n); memcpy(s, v, k);
}

static int solver_name_equal(const char *lhs, const char *rhs);

static const char *const ifdda_test_geometry_names[] = {
    "concentricsphere", "sphere", "cube", "cuboid1", "cuboid2",
    "inhomosphere", "inhomocuboid1", "inhomocuboid2", "ellipsoid",
    "nspheres", "cylinder", "randomsphere1", "randomsphere2", "arbitrary",
    "randomspheremulti1"
};

static int ifdda_test_geometry_case(const char *name) {
    char *end = NULL;
    long id = strtol(name,&end,10);
    if(end!=name && *end=='\0' && id>=0 && id<15) return (int)id;
    if (solver_name_equal(name, "0") || solver_name_equal(name, "concentric")) return 0;
    if (solver_name_equal(name, "randommulti")) return 14;
    for (size_t i=0;i<sizeof(ifdda_test_geometry_names)/sizeof(ifdda_test_geometry_names[0]);++i)
        if (solver_name_equal(name, ifdda_test_geometry_names[i])) return (int)i;
    return -1;
}

static void ifdda_test_geometry_usage(void) {
    puts("Geometries: concentricsphere sphere cube cuboid1 cuboid2 inhomosphere");
    puts("            inhomocuboid1 inhomocuboid2 ellipsoid nspheres cylinder");
    puts("            randomsphere1 randomsphere2 arbitrary randomspheremulti1");
}

static int solver_name_equal(const char *lhs, const char *rhs) {
    while (*lhs && *rhs) {
        if (tolower((unsigned char)*lhs) != tolower((unsigned char)*rhs)) {
            return 0;
        }
        ++lhs;
        ++rhs;
    }
    return *lhs == '\0' && *rhs == '\0';
}

static const char *solver_method_from_name(const char *name) {
    static const struct {
        const char *name;
        const char *method;
    } solvers[] = {
        {"gpbicg", "GPBICG1"},
        {"gpbicg1", "GPBICG1"},
        {"bornserie", "BORNSERIE"},
        {"gpbicg2", "GPBICG2"},
        {"gpbicgsafe", "GPBICGsafe"},
        {"gpbicgplus", "GPBICGplus"},
        {"gpbicgar", "GPBICGAR"},
        {"gpbicgar2", "GPBICGAR2"},
        {"ssbicgsafe2", "SSBICGSAFE2"},
        {"gpbicgstarplus", "BICGSTARPLUS"},
        {"bicgstarplus", "BICGSTARPLUS"},
        {"tfqmr", "TFQMR"},
        {"zcg", "CG"},
        {"pimzbicgstab", "BICGSTAB"},
        {"qmrbicgstab", "QMRBICGSTAB1"},
        {"qmrbicgstab1", "QMRBICGSTAB1"},
        {"qmrbicgstab2", "QMRBICGSTAB2"},
        {"qmrcla", "QMRCLA"},
        {"pimzqmr", "QMRCLA"},
        {"gpbicor", "GPBICOR"},
        {"gcors2", "GCORS2"},
        {"cors", "CORS"},
        {"idrs2", "IDRS2"},
        {"idrs4", "IDRS4"},
        {"idrs8", "IDRS8"},
        {"bicglstab", "BICGLSTAB"},
        {"bicglstab2", "BICGLSTAB2"},
        {"bicglstab4", "BICGLSTAB4"},
        {"bicglstab8", "BICGLSTAB8"},
        {"gpbicgstabl2", "GPBICGSTABL2"},
        {"gpbicgstabl4", "GPBICGSTABL4"},
        {"gpbicgstabl8", "GPBICGSTABL8"}
    };
    size_t i;
    for (i = 0; i < sizeof(solvers) / sizeof(solvers[0]); ++i) {
        if (solver_name_equal(name, solvers[i].name)) {
            return solvers[i].method;
        }
    }
    return NULL;
}

static const char *illumination_name_from_cli(const char *name) {
    static const char *const names[] = {
        "pwavelinear", "pwavecircular", "wavelinearmulti", "antenna",
        "gparawavelinear", "gparawavecircular", "gwavelinear", "gwavecircular",
        "arbitrary", "gfftwavelinear", "gfftwavecircular", "greentensor",
        "speckle", "demispeckle", "confocal", "demiconfocal"
    };
    size_t i;
    for (i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        if (solver_name_equal(name, names[i])) return names[i];
    }
    return NULL;
}

static int cli_grid_size = 0;
static int cli_nfft2d = 0;
static int cli_nrig_value = -1;
static int cli_nquad_value = -1;
static int cli_preconditioner_value = -1;
static int cli_init_value = -1;
static int cli_local_value = -1;
static int cli_macro_value = -1;
static int cli_force_value = -1;
static int cli_force_density_value = -1;
static int cli_torque_value = -1;
static int cli_torque_density_value = -1;
static int cli_energy_value = -1;
static int cli_near_domain_value = -1;
static bool cli_near_domain_set = false;
static cdmlib_real cli_background_epsilon_value = 0;
static bool cli_background_epsilon_set = false;
static int cli_abc_type_value = 0;
static cdmlib_real cli_abc_nrbc_value = 0, cli_abc_cos_value = 0;
static bool cli_abc_type_set = false, cli_abc_nrbc_set = false, cli_abc_cos_set = false;
static int cli_quick_lens_value = -1;
static int cli_quick_diffraction_value = -1;
static int cli_scattering_value = -1;
static int cli_scattering_theta_points = -1;
static int cli_scattering_phi_points = -1;
static int cli_max_iterations = 0;
static int cli_speckle_seed = 0;
static int cli_cross_section = -1;
static cdmlib_real cli_radius_value = 0;
static cdmlib_real cli_illumination_na = 0;
static cdmlib_real cli_wavelength_value = 0;
static cdmlib_real cli_power_value = 0;
static cdmlib_real cli_tolerance_value = 0;
static cdmlib_real cli_theta_value = 0;
static cdmlib_real cli_phi_value = 0;
static cdmlib_real cli_pp_value = 0;
static cdmlib_real cli_ss_value = 0;
static cdmlib_real cli_waist_value = 0;
static cdmlib_real cli_epsilon_real = 0;
static cdmlib_real cli_epsilon_imag = 0;
static int cli_epsilon_set = 0;
static cdmlib_real cli_geometry_angles[3], cli_geometry_dimensions[3];
static int cli_geometry_angles_set = 0, cli_geometry_dimensions_set = 0;
static cdmlib_real cli_geometry_density = 0, cli_cell_size = 0;
static int cli_geometry_density_set = 0, cli_cell_size_set = 0;
static int cli_microscope = -1, cli_microscope_type = -1, cli_microscope_side = -1;
static cdmlib_real cli_na = 0, cli_magnification = 0, cli_lens_z = 0;
static int cli_na_set = 0, cli_magnification_set = 0, cli_lens_z_set = 0;
static int cli_grid_dimensions[3], cli_grid_dimensions_set = 0;
static int cli_padding[3], cli_padding_set = 0;
static cdmlib_real cli_geometry_center[3], cli_gaussian_center[3];
static int cli_geometry_center_set = 0, cli_gaussian_center_set = 0;
static cdmlib_real cli_condenser_na = 0, cli_inner_na = 0, cli_na_center[2];
static int cli_condenser_na_set = 0, cli_inner_na_set = 0, cli_na_center_set = 0;
static cdmlib_real cli_side_length = 0;
static int cli_side_length_set = 0, cli_geometry_only = -1, cli_geometry_only_set = 0;
static int cli_save_mode = -1, cli_save_mode_set = 0, cli_read_field = -1, cli_read_field_set = 0;
static char cli_polarizability[3] = "RR", cli_isotropy[4] = "iso";
static int cli_polarizability_set = 0, cli_isotropy_set = 0;
static char cli_field_file[64], cli_object_file[64], cli_hdf5_file[100], cli_arbitrary_file[64];
static int cli_field_file_set = 0, cli_object_file_set = 0, cli_hdf5_file_set = 0, cli_arbitrary_file_set = 0;
static int cli_method_case = -1, cli_option_case = -1;

static int parse_triplet(const char *text, cdmlib_real out[3]) {
    double a, b, c;
    char extra;
    if (sscanf(text, "%lf,%lf,%lf %c", &a, &b, &c, &extra) != 3) return 0;
    out[0] = (cdmlib_real)a;
    out[1] = (cdmlib_real)b;
    out[2] = (cdmlib_real)c;
    return 1;
}

static int parse_int_value(const char *text, int *out) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < INT_MIN || value > INT_MAX)
        return 0;
    *out = (int)value;
    return 1;
}

static int parse_real_value(const char *text, cdmlib_real *out) {
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == text || *end != '\0')
        return 0;
    *out = (cdmlib_real)value;
    return 1;
}

static int parse_csv_values(const char *text, cdmlib_real *out, size_t count) {
    const char *p = text;
    for (size_t i = 0; i < count; ++i) {
        char *end = NULL;
        double v = strtod(p, &end);
        if (end == p) return 0;
        out[i] = (cdmlib_real)v;
        if (i + 1 < count) { if (*end != ',') return 0; p = end + 1; }
        else if (*end != '\0') return 0;
    }
    return 1;
}

static int copy_cli_filename(char *dst, size_t capacity, const char *src) {
    size_t n = strlen(src);
    if (n == 0 || n >= capacity) return 0;
    memcpy(dst, src, n + 1);
    return 1;
}

static int parse_solver_option(int argc, char **argv, const char **method,
                               int *geometry_case, const char **illumination) {
    int i, option_start = 1;
    int solver_was_named = 0;
    *method = "GPBICG1";
    *geometry_case = -1;
    *illumination = "pwavelinear";
    cli_grid_size = cli_nfft2d = cli_nrig_value = cli_nquad_value =
        cli_max_iterations = cli_speckle_seed = 0;
    cli_preconditioner_value = cli_init_value = -1;
    cli_local_value = cli_macro_value = -1;
    cli_force_value = cli_force_density_value = -1;
    cli_torque_value = cli_torque_density_value = -1;
    cli_energy_value = cli_near_domain_value = -1;
    cli_near_domain_set = false;
    cli_background_epsilon_set = false;
    cli_abc_type_set = cli_abc_nrbc_set = cli_abc_cos_set = false;
    cli_quick_lens_value = cli_quick_diffraction_value = -1;
    cli_scattering_value = -1;
    cli_scattering_theta_points = cli_scattering_phi_points = -1;
    cli_cross_section = -1;
    cli_nrig_value = cli_nquad_value = -1;
    cli_epsilon_set = 0;
    cli_geometry_angles_set = cli_geometry_dimensions_set = 0;
    cli_geometry_density_set = cli_cell_size_set = 0;
    cli_microscope = cli_microscope_type = cli_microscope_side = -1;
    cli_na_set = cli_magnification_set = cli_lens_z_set = 0;
    cli_grid_dimensions_set = cli_padding_set = 0;
    cli_geometry_center_set = cli_gaussian_center_set = 0;
    cli_condenser_na_set = cli_inner_na_set = cli_na_center_set = 0;
    cli_side_length_set = cli_geometry_only_set = cli_save_mode_set = 0;
    cli_read_field_set = cli_polarizability_set = cli_isotropy_set = 0;
    cli_field_file_set = cli_object_file_set = cli_hdf5_file_set = cli_arbitrary_file_set = 0;
    cli_method_case = cli_option_case = -1;
    for (int j = 1; j < argc; ++j) {
        if (strncmp(argv[j], "--", 2) == 0) { option_start=j; break; }
        option_start=j+1;
        int v = 0;
        if (j == 0) continue;
        if (j == 1) { if (!parse_int_value(argv[j], &v) || v < 0 || v > 14) return -1; *geometry_case = v; }
        else if (j == 2) { if (!parse_int_value(argv[j], &v)) return -1; static const char *const beams[] = {"pwavelinear","pwavecircular","wavelinearmulti","antenna","gparawavelinear","gparawavecircular","gwavelinear","gwavecircular","arbitrary","gfftwavelinear","gfftwavecircular","greentensor","speckle","demispeckle","confocal","demiconfocal"}; if(v<200||v>215)return -1; *illumination=beams[v-200]; }
        else if (j == 3) { if (!parse_int_value(argv[j], &v) || v < 0 || v > 15) return -1; cli_nrig_value=v; }
        else if (j == 4) { if (!parse_int_value(argv[j], &v) || v < 0 || v > 24) return -1; cli_method_case=v; }
        else if (j == 5) { if (!parse_int_value(argv[j], &v) || v < 0 || v > 2) return -1; cli_preconditioner_value=v; }
        else if (j == 6) { if (!parse_int_value(argv[j], &v) || v < 0 || v > 5) return -1; cli_init_value=v; }
        else if (j == 7) { if (!parse_int_value(argv[j], &v) || v < 0 || v > 28) return -1; cli_option_case=v; }
        else { fprintf(stderr, "Too many positional arguments: %s\n", argv[j]); return -1; }
    }
    for (i = option_start; i < argc; ++i) {
        const char *name = NULL;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            puts("Usage: test0 [--solver NAME] [--geometry NAME] [--illumination NAME]");
            puts("Illuminations: pwavelinear pwavecircular wavelinearmulti antenna");
            puts("               gparawavelinear gparawavecircular gwavelinear gwavecircular");
            puts("               gfftwavelinear gfftwavecircular arbitrary greentensor");
            puts("               speckle demispeckle confocal demiconfocal");
            puts("Options: --preconditioner N --force N --force-density N --torque N");
            puts("         --torque-density N --energy N --near-domain N");
            puts("         --background-epsilon X (imaginary background permittivity)");
            puts("         --abc-type N --abc-nrbc X --abc-cos X (PML settings)");
            puts("         --quick-lens N --quick-diffraction N --local N --macro N");
            puts("         --power W --scattering-theta-points N --scattering-phi-points N");
            puts("         --method-case N --option-case N --grid-dimensions NX,NY,NZ");
            puts("         --geometry-center X,Y,Z --gaussian-center X,Y,Z --near-padding X,Y,Z");
            puts("         --side-length X --geometry-only N --save-mode N --read-field N");
            puts("         --polarizability CODE --isotropy iso|ani --field-file FILE --object-file FILE");
            puts("         --hdf5-file FILE --arbitrary-file FILE --condenser-na X --inner-na X --na-center X,Y");
            puts("         Legacy positional form: GEOMETRY_ID BEAM_ID NRIG_ID METHOD_ID PRECONDITIONER_ID INITIAL_GUESS_ID OPTION_ID");
            ifdda_test_geometry_usage();
            puts("Solver names: gpbicg, gpbicg2, gpbicgsafe, gpbicgplus,");
            puts("  gpbicgar, gpbicgar2, ssbicgsafe2, gpbicgstarplus, tfqmr, zcg,");
            puts("  pimzbicgstab, qmrbicgstab[1|2], qmrcla, gpbicor, gcors2, cors, idrs[2|4|8],");
            puts("  bicglstab[|2|4|8], gpbicgstabl[2|4|8]");
            return 1;
        }
        if (strcmp(argv[i], "--solver") == 0) {
            if (i + 1 >= argc) {
                fputs("Missing solver name after --solver. Use --help for the list.\n", stderr);
                return -1;
            }
            name = argv[++i];
        } else if (strncmp(argv[i], "--solver=", 9) == 0) {
            name = argv[i] + 9;
        } else if (strcmp(argv[i], "--illumination") == 0 ||
                   strncmp(argv[i], "--illumination=", 15) == 0) {
            const char *value;
            if (strcmp(argv[i], "--illumination") == 0) {
                if (i + 1 >= argc) {
                    fputs("Missing illumination after --illumination.\n", stderr);
                    return -1;
                }
                value = argv[++i];
            } else {
                value = argv[i] + 15;
            }
            *illumination = illumination_name_from_cli(value);
            if (*illumination == NULL) {
                fprintf(stderr, "Unsupported illumination: %s. Use --help.\n", value);
                return -1;
            }
            continue;
        } else if (strcmp(argv[i], "--geometry") == 0 ||
                   strncmp(argv[i], "--geometry=", 11) == 0) {
            const char *geometry_name;
            if (strcmp(argv[i], "--geometry") == 0) {
                if (i + 1 >= argc) {
                    fputs("Missing geometry after --geometry.\n",stderr);
                    return -1;
                }
                geometry_name = argv[++i];
            } else {
                geometry_name = argv[i] + 11;
            }
            *geometry_case = ifdda_test_geometry_case(geometry_name);
            if (*geometry_case < 0) {
                fprintf(stderr, "Unknown geometry: %s. Use --help.\n",
                        geometry_name);
                return -1;
            }
            continue;
        } else if (strncmp(argv[i], "--", 2) == 0) {
            const char *key = argv[i] + 2;
            const char *value = strchr(key, '=');
            char buffer[128];
            if (value) {
                size_t length = (size_t)(value - key);
                if (length >= sizeof buffer) return -1;
                memcpy(buffer, key, length); buffer[length] = '\0';
                value++;
            } else if (i + 1 < argc) {
                value = argv[++i];
                strncpy(buffer, key, sizeof buffer - 1);
                buffer[sizeof buffer - 1] = '\0';
                key = buffer;
            } else {
                fprintf(stderr, "Missing value after %s\n", argv[i]);
                return -1;
            }
            if (strcmp(key, "grid-size") == 0) {
                if (!parse_int_value(value, &cli_grid_size)) return -1;
            } else if (strcmp(key, "grid-dimensions") == 0) {
                cdmlib_real dims[3];
                if (!parse_csv_values(value,dims,3)) return -1;
                for(int d=0;d<3;++d) { if(dims[d]<8 || dims[d]>256 || (int)dims[d]!=dims[d]) return -1; cli_grid_dimensions[d]=(int)dims[d]; }
                cli_grid_dimensions_set=1;
            } else if (strcmp(key, "nfft2d") == 0) {
                if (!parse_int_value(value, &cli_nfft2d)) return -1;
            } else if (strcmp(key, "nrig") == 0) {
                if (!parse_int_value(value, &cli_nrig_value)) return -1;
            } else if (strcmp(key, "nquad") == 0) {
                if (!parse_int_value(value, &cli_nquad_value)) return -1;
            } else if (strcmp(key, "preconditioner") == 0) {
                if (!parse_int_value(value, &cli_preconditioner_value)) return -1;
            } else if (strcmp(key, "initial-guess") == 0) {
                if (!parse_int_value(value, &cli_init_value)) return -1;
            } else if (strcmp(key, "local") == 0 || strcmp(key, "local-field") == 0) {
                if (!parse_int_value(value, &cli_local_value)) return -1;
            } else if (strcmp(key, "macro") == 0 || strcmp(key, "macro-field") == 0) {
                if (!parse_int_value(value, &cli_macro_value)) return -1;
            } else if (strcmp(key, "force") == 0) {
                if (!parse_int_value(value, &cli_force_value)) return -1;
            } else if (strcmp(key, "force-density") == 0) {
                if (!parse_int_value(value, &cli_force_density_value)) return -1;
            } else if (strcmp(key, "torque") == 0) {
                if (!parse_int_value(value, &cli_torque_value)) return -1;
            } else if (strcmp(key, "torque-density") == 0) {
                if (!parse_int_value(value, &cli_torque_density_value)) return -1;
            } else if (strcmp(key, "energy") == 0) {
                if (!parse_int_value(value, &cli_energy_value)) return -1;
            } else if (strcmp(key, "near-domain") == 0) {
                if (!parse_int_value(value, &cli_near_domain_value)) return -1;
                cli_near_domain_set = true;
            } else if (strcmp(key, "background-epsilon") == 0) {
                if (!parse_real_value(value, &cli_background_epsilon_value)) return -1;
                cli_background_epsilon_set = true;
            } else if (strcmp(key, "abc-type") == 0) {
                if (!parse_int_value(value, &cli_abc_type_value)) return -1;
                cli_abc_type_set = true;
            } else if (strcmp(key, "abc-nrbc") == 0) {
                if (!parse_real_value(value, &cli_abc_nrbc_value) || cli_abc_nrbc_value < 0) return -1;
                cli_abc_nrbc_set = true;
            } else if (strcmp(key, "abc-cos") == 0) {
                if (!parse_real_value(value, &cli_abc_cos_value) || cli_abc_cos_value < 0) return -1;
                cli_abc_cos_set = true;
            } else if (strcmp(key, "quick-lens") == 0) {
                if (!parse_int_value(value, &cli_quick_lens_value)) return -1;
            } else if (strcmp(key, "quick-diffraction") == 0) {
                if (!parse_int_value(value, &cli_quick_diffraction_value)) return -1;
            } else if (strcmp(key, "scattering") == 0) {
                if (!parse_int_value(value, &cli_scattering_value)) return -1;
            } else if (strcmp(key, "scattering-theta-points") == 0) {
                if (!parse_int_value(value, &cli_scattering_theta_points)) return -1;
            } else if (strcmp(key, "scattering-phi-points") == 0) {
                if (!parse_int_value(value, &cli_scattering_phi_points)) return -1;
            } else if (strcmp(key, "max-iterations") == 0) {
                if (!parse_int_value(value, &cli_max_iterations)) return -1;
            } else if (strcmp(key, "speckle-seed") == 0) {
                if (!parse_int_value(value, &cli_speckle_seed)) return -1;
            } else if (strcmp(key, "cross-section") == 0) {
                if (!parse_int_value(value, &cli_cross_section)) return -1;
            } else if (strcmp(key, "wavelength") == 0) {
                if (!parse_real_value(value, &cli_wavelength_value)) return -1;
            } else if (strcmp(key, "power") == 0) {
                if (!parse_real_value(value, &cli_power_value)) return -1;
            } else if (strcmp(key, "tolerance") == 0) {
                if (!parse_real_value(value, &cli_tolerance_value)) return -1;
            } else if (strcmp(key, "theta") == 0) {
                if (!parse_real_value(value, &cli_theta_value)) return -1;
            } else if (strcmp(key, "phi") == 0) {
                if (!parse_real_value(value, &cli_phi_value)) return -1;
            } else if (strcmp(key, "pp") == 0) {
                if (!parse_real_value(value, &cli_pp_value)) return -1;
            } else if (strcmp(key, "ss") == 0) {
                if (!parse_real_value(value, &cli_ss_value)) return -1;
            } else if (strcmp(key, "waist") == 0) {
                if (!parse_real_value(value, &cli_waist_value)) return -1;
            } else if (strcmp(key, "radius") == 0) {
                if (!parse_real_value(value, &cli_radius_value)) return -1;
            } else if (strcmp(key, "geometry-angles") == 0) {
                if (!parse_triplet(value, cli_geometry_angles)) return -1;
                cli_geometry_angles_set = 1;
            } else if (strcmp(key, "geometry-dimensions") == 0) {
                if (!parse_triplet(value, cli_geometry_dimensions)) return -1;
                cli_geometry_dimensions_set = 1;
            } else if (strcmp(key, "geometry-density") == 0) {
                if (!parse_real_value(value, &cli_geometry_density) || cli_geometry_density <= 0) return -1;
                cli_geometry_density_set = 1;
            } else if (strcmp(key, "cell-size") == 0) {
                if (!parse_real_value(value, &cli_cell_size) || cli_cell_size <= 0) return -1;
                cli_cell_size_set = 1;
            } else if (strcmp(key, "microscope") == 0) {
                if (!parse_int_value(value, &cli_microscope)) return -1;
            } else if (strcmp(key, "microscope-type") == 0) {
                if (!parse_int_value(value, &cli_microscope_type)) return -1;
            } else if (strcmp(key, "microscope-side") == 0) {
                if (!parse_int_value(value, &cli_microscope_side)) return -1;
            } else if (strcmp(key, "na") == 0) {
                if (!parse_real_value(value, &cli_na)) return -1;
                cli_na_set = 1;
            } else if (strcmp(key, "magnification") == 0) {
                if (!parse_real_value(value, &cli_magnification)) return -1;
                cli_magnification_set = 1;
            } else if (strcmp(key, "lens-z") == 0) {
                if (!parse_real_value(value, &cli_lens_z)) return -1;
                cli_lens_z_set = 1;
            } else if (strcmp(key, "illumination-na") == 0) {
                if (!parse_real_value(value, &cli_illumination_na)) return -1;
            } else if (strcmp(key, "method-case") == 0) {
                if (!parse_int_value(value,&cli_method_case) || cli_method_case<0 || cli_method_case>24) return -1;
            } else if (strcmp(key, "option-case") == 0) {
                if (!parse_int_value(value,&cli_option_case) || cli_option_case<0 || cli_option_case>28) return -1;
            } else if (strcmp(key, "geometry-center") == 0) {
                if(!parse_csv_values(value,cli_geometry_center,3)) return -1; cli_geometry_center_set=1;
            } else if (strcmp(key, "gaussian-center") == 0) {
                if(!parse_csv_values(value,cli_gaussian_center,3)) return -1; cli_gaussian_center_set=1;
            } else if (strcmp(key, "near-padding") == 0) {
                cdmlib_real p[3]; if(!parse_csv_values(value,p,3)) return -1;
                for(int d=0;d<3;++d) { if(p[d]<0 || (int)p[d]!=p[d]) return -1; cli_padding[d]=(int)p[d]; }
                cli_padding_set=1;
            } else if (strcmp(key, "side-length") == 0) {
                if(!parse_real_value(value,&cli_side_length) || cli_side_length<=0) return -1; cli_side_length_set=1;
            } else if (strcmp(key, "geometry-only") == 0) {
                if(!parse_int_value(value,&cli_geometry_only)) return -1; cli_geometry_only_set=1;
            } else if (strcmp(key, "save-mode") == 0) {
                if(!parse_int_value(value,&cli_save_mode)) return -1; cli_save_mode_set=1;
            } else if (strcmp(key, "read-field") == 0) {
                if(!parse_int_value(value,&cli_read_field)) return -1; cli_read_field_set=1;
            } else if (strcmp(key, "polarizability") == 0) {
                static const char *const valid[] = {"CM","RR","PS","GB","LA","LR","LS","FG"};
                int valid_code=0;
                for(size_t p=0;p<sizeof(valid)/sizeof(valid[0]);++p) if(strcmp(value,valid[p])==0) valid_code=1;
                if(!valid_code) return -1;
                memcpy(cli_polarizability,value,3); cli_polarizability_set=1;
            } else if (strcmp(key, "isotropy") == 0) {
                if(strcmp(value,"iso") && strcmp(value,"ani")) return -1; strcpy(cli_isotropy,value); cli_isotropy_set=1;
            } else if (strcmp(key, "condenser-na") == 0) {
                if(!parse_real_value(value,&cli_condenser_na)) return -1; cli_condenser_na_set=1;
            } else if (strcmp(key, "inner-na") == 0) {
                if(!parse_real_value(value,&cli_inner_na)) return -1; cli_inner_na_set=1;
            } else if (strcmp(key, "na-center") == 0) {
                if(!parse_csv_values(value,cli_na_center,2)) return -1; cli_na_center_set=1;
            } else if (strcmp(key, "field-file") == 0) {
                if(!copy_cli_filename(cli_field_file,sizeof cli_field_file,value)) return -1; cli_field_file_set=1;
            } else if (strcmp(key, "object-file") == 0) {
                if(!copy_cli_filename(cli_object_file,sizeof cli_object_file,value)) return -1; cli_object_file_set=1;
            } else if (strcmp(key, "hdf5-file") == 0) {
                if(!copy_cli_filename(cli_hdf5_file,sizeof cli_hdf5_file,value)) return -1; cli_hdf5_file_set=1;
            } else if (strcmp(key, "arbitrary-file") == 0) {
                if(!copy_cli_filename(cli_arbitrary_file,sizeof cli_arbitrary_file,value)) return -1; cli_arbitrary_file_set=1;
            } else if (strcmp(key, "epsilon") == 0) {
                double real_value = 0;
                double imag_value = 0;
                if (sscanf(value, "%lf,%lf", &real_value, &imag_value) != 2)
                    return -1;
                cli_epsilon_real = (cdmlib_real)real_value;
                cli_epsilon_imag = (cdmlib_real)imag_value;
                cli_epsilon_set = 1;
            } else {
                fprintf(stderr, "Unknown argument: --%s\n", key);
                return -1;
            }
            continue;
        } else {
            fprintf(stderr, "Unknown argument: %s. Use --help for usage.\n", argv[i]);
            return -1;
        }
        *method = solver_method_from_name(name);
        if (*method == NULL) {
            fprintf(stderr, "Unsupported solver: %s. Use --help for the list.\n", name);
            return -1;
        }
        solver_was_named = 1;
    }
    if (!solver_was_named && cli_method_case >= 0) {
        static const char *const methods[25] = {"GPBICG1","GPBICG1","GPBICG2","GPBICGplus","GPBICGsafe","GPBICGAR","GPBICGAR2","BICGSTARPLUS","GPBICOR","CORS","QMRCLA","TFQMR","CG","BICGSTAB","QMRBICGSTAB1","QMRBICGSTAB2","IDRS2","IDRS4","IDRS8","BICGLSTAB2","BICGLSTAB4","BICGLSTAB8","GPBICGSTABL2","GPBICGSTABL4","GPBICG1"};
        *method = methods[cli_method_case];
    }
    return 0;
}

static int run_one_calculation(int argc, char **argv) {
    const char *selected_solver_method = NULL;
    const char *selected_illumination = NULL;
    int selected_geometry_case = -1; // No CLI override: original sphere test.
    int option_status = parse_solver_option(argc, argv,
        &selected_solver_method, &selected_geometry_case, &selected_illumination);
    if (option_status != 0) {
        return option_status > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    /* The ABI typedef selects the complex layout for the linked precision. */
    cdmlib_real lambda = 0;
    cdmlib_real eps0i = 0;
    int ntypeabc = 0;
    cdmlib_real lnrbc = 0;
    cdmlib_real lcos = 0;
    char beam[64];
    char object[64];
    char trope[3];
    int niso = 0;
    char materiaumulti[128];
    int nnnr = 0;
    cdmlib_real tolinit = 0;
    int nlim = 0;
    char methodeit[12];
    char polarizability[2];
    int nprecon = 0;
    int ninitest = 0;
    int nquad = 0;
    int nlecture = 0;
    char filereread[64];
    int nmat = 0;
    char h5file[100];
    int nlocal = 0;
    int nmacro = 0;
    int nsection = 0;
    int nsectionsca = 0;
    int nquickdiffracte = 0;
    int nrig = 0;
    int nforce = 0;
    int nforced = 0;
    int ntorque = 0;
    int ntorqued = 0;
    int nproche = 0;
    int nlentille = 0;
    int nquicklens = 0;
    int nenergie = 0;
    int nobjet = 0;
    cdmlib_real density = 0;
    cdmlib_real side = 0;
    cdmlib_real sidex = 0;
    cdmlib_real sidey = 0;
    cdmlib_real sidez = 0;
    cdmlib_real hauteur = 0;
    int numberobjet = 0;
    cdmlib_real *rayonmulti;
    cdmlib_real *xgmulti;
    cdmlib_real *ygmulti;
    cdmlib_real *zgmulti;
    cdmlib_complex *epsmulti;
    cdmlib_complex *epsanimulti;
    cdmlib_real lc = 0;
    cdmlib_real hc = 0;
    int ng = 0;
    cdmlib_real demiaxea = 0;
    cdmlib_real demiaxeb = 0;
    cdmlib_real demiaxec = 0;
    cdmlib_real thetaobj = 0;
    cdmlib_real phiobj = 0;
    cdmlib_real psiobj = 0;
    char namefileobj[64];
    cdmlib_real theta = 0;
    cdmlib_real phi = 0;
    cdmlib_real pp = 0;
    cdmlib_real ss = 0;
    cdmlib_real P0 = 0;
    cdmlib_real w0 = 0;
    cdmlib_real xgaus = 0;
    cdmlib_real ygaus = 0;
    cdmlib_real zgaus = 0;
    int speckseed = 0;
    char namefileinc[64];
    cdmlib_real *thetam;
    cdmlib_real *phim;
    cdmlib_real *ppm;
    cdmlib_real *ssm;
    cdmlib_complex *E0m;
    int nbinc = 0;
    char infostr[64];
    int nstop = 0;
    int nbsphere = 0;
    int ndipole = 0;
    cdmlib_real aretecube = 0;
    cdmlib_real lambda10n = 0;
    cdmlib_real k0 = 0;
    cdmlib_real tol1 = 0;
    cdmlib_real tempsreelmvp = 0;
    cdmlib_real tempsreeltotal = 0;
    int ncompte = 0;
    int nloop = 0;
    cdmlib_real efficacite = 0;
    cdmlib_real efficaciteref = 0;
    cdmlib_real efficacitetrans = 0;
    cdmlib_real Cext = 0;
    cdmlib_real Cabs = 0;
    cdmlib_real Csca = 0;
    cdmlib_real Cscai = 0;
    cdmlib_real gasym = 0;
    cdmlib_real irra = 0;
    cdmlib_complex E0 = 0;
    cdmlib_real *forcet;
    cdmlib_real forcem = 0;
    cdmlib_real *couplet;
    cdmlib_real couplem = 0;
    int nxm = 0;
    int nym = 0;
    int nzm = 0;
    int nxmp = 0;
    int nymp = 0;
    int nzmp = 0;
    int nmaxpp = 0;
    int nxym1 = 0;
    int nzm1 = 0;
    int nzms1 = 0;
    cdmlib_real *incidentfield;
    cdmlib_real *localfield;
    cdmlib_real *macroscopicfield;
    cdmlib_real *xs;
    cdmlib_real *ys;
    cdmlib_real *zs;
    cdmlib_real *xswf;
    cdmlib_real *yswf;
    cdmlib_real *zswf;
    int ntheta = 0;
    int nphi = 0;
    cdmlib_real *thetafield;
    cdmlib_real *phifield;
    cdmlib_real *poyntingfield;
    cdmlib_real *kxypoynting;
    cdmlib_real *poyntingfieldpos;
    cdmlib_real *poyntingfieldneg;
    cdmlib_real *forcex;
    cdmlib_real *forcey;
    cdmlib_real *forcez;
    cdmlib_real *forcexmulti;
    cdmlib_real *forceymulti;
    cdmlib_real *forcezmulti;
    cdmlib_real *torquex;
    cdmlib_real *torquey;
    cdmlib_real *torquez;
    cdmlib_real *torquexmulti;
    cdmlib_real *torqueymulti;
    cdmlib_real *torquezmulti;
    cdmlib_complex *incidentfieldx;
    cdmlib_complex *incidentfieldy;
    cdmlib_complex *incidentfieldz;
    cdmlib_complex *localfieldx;
    cdmlib_complex *localfieldy;
    cdmlib_complex *localfieldz;
    cdmlib_complex *macroscopicfieldx;
    cdmlib_complex *macroscopicfieldy;
    cdmlib_complex *macroscopicfieldz;
    cdmlib_complex *polarisa;
    cdmlib_complex *epsilon;
    int nfft2d = 0;
    int npoynting = 0;
    cdmlib_real deltapoyntingx = 0;
    cdmlib_real deltapoyntingy = 0;
    cdmlib_complex *Eimagex;
    cdmlib_complex *Eimagey;
    cdmlib_complex *Eimagez;
    cdmlib_complex *Eimageincx;
    cdmlib_complex *Eimageincy;
    cdmlib_complex *Eimageincz;
    cdmlib_complex *Efourierx;
    cdmlib_complex *Efouriery;
    cdmlib_complex *Efourierz;
    cdmlib_complex *Efourierincx;
    cdmlib_complex *Efourierincy;
    cdmlib_complex *Efourierincz;
    cdmlib_real *kxy;
    cdmlib_real *xy;
    cdmlib_real numaper = 0;
    cdmlib_real numaperinc = 0;
    cdmlib_real numaperinc2 = 0;
    cdmlib_real kcnax = 0;
    cdmlib_real kcnay = 0;
    cdmlib_real numaperil = 0;
    cdmlib_real gross = 0;
    cdmlib_real zlens = 0;
    cdmlib_real psiinc = 0;
    int ntypemic = 0;
    int nside = 0;
    cdmlib_complex *FF;
    cdmlib_complex *FF0;
    cdmlib_complex *FFloc;
    cdmlib_complex *xr;
    cdmlib_complex *xi;
    cdmlib_complex *FFprecon;
    cdmlib_complex *wrk;
    int nlar = 0;
    cdmlib_complex *FFTTENSORxx;
    cdmlib_complex *FFTTENSORxy;
    cdmlib_complex *FFTTENSORxz;
    cdmlib_complex *FFTTENSORyy;
    cdmlib_complex *FFTTENSORyz;
    cdmlib_complex *FFTTENSORzz;
    cdmlib_complex *Ediffkzpos;
    cdmlib_complex *Ediffkzneg;
    cdmlib_complex *sdetnn;
    int *Tabdip;
    int *Tabmulti;
    int *tabfft2;

    /* Physical parameters from the Fortran sphere test. */
    lambda=632.8; P0=1.; w0=lambda*10.;
    nnnr=nxm=nym=nzm=32; numberobjet=1; niso=1; nfft2d=128; nlar=42;
    tolinit=1.e-4; nlim=1000; ninitest=1;
    if (cli_grid_size > 0) nnnr=nxm=nym=nzm=cli_grid_size;
    if (cli_grid_dimensions_set) { nxm=cli_grid_dimensions[0]; nym=cli_grid_dimensions[1]; nzm=cli_grid_dimensions[2]; nnnr=nxm<nym?(nxm<nzm?nxm:nzm):(nym<nzm?nym:nzm); }
    if (cli_isotropy_set && strcmp(cli_isotropy,"ani")==0) niso=3;
    if (cli_nfft2d > 0) nfft2d=cli_nfft2d;
    if (cli_nrig_value >= 0) nrig=cli_nrig_value;
    if (cli_nquad_value >= 0) nquad=cli_nquad_value;
    if (cli_preconditioner_value >= 0) nprecon=cli_preconditioner_value;
    if (cli_init_value >= 0) ninitest=cli_init_value;
    const char* geometry_ani=getenv("IFDDA_TEST_GEOMETRY_ANIS");
    if(geometry_ani && strcmp(geometry_ani,"1")==0)niso=3;
    if(cli_isotropy_set) niso=(strcmp(cli_isotropy,"ani")==0)?3:1;
    /* The Fortran ABI requires full 3*nzm block matrices for vector
       Chan preconditioning.  Match tests/test0/main.f90 exactly. */
    if (nprecon == 0) {
        nxym1=nzm1=nzms1=1;
    } else {
        nxym1=nxm*nym;
        nzm1=nzm;
        nzms1=(nrig==0 ? 3*nzm : nzm);
    }
    if (cli_max_iterations > 0) nlim=cli_max_iterations;
    if (cli_speckle_seed != 0) speckseed=cli_speckle_seed;
    if (cli_cross_section >= 0) nsection=cli_cross_section;
    if (cli_wavelength_value != 0) lambda=cli_wavelength_value;
    if (cli_power_value != 0) P0=cli_power_value;
    if (cli_tolerance_value != 0) tolinit=cli_tolerance_value;
    if (cli_waist_value != 0) w0=cli_waist_value;
    /* A validation-only cap for slow solvers such as CG.  The default
       remains the historical 1000 iterations when unset. */
    if (const char *maxit_text = getenv("IFDDA_TEST_MAXIT")) {
        char *end = nullptr;
        errno = 0;
        long candidate = strtol(maxit_text, &end, 10);
        if (errno != 0 || end == maxit_text || *end != '\0' ||
            candidate < 2 || candidate > INT_MAX) {
            fprintf(stderr, "Invalid IFDDA_TEST_MAXIT: expected integer 2..%d\n", INT_MAX);
            return EXIT_FAILURE;
        }
        nlim = static_cast<int>(candidate);
        fprintf(stderr, "[IF-DDA] Validation maximum iterations: %d\n", nlim);
    }
    nlocal=1;
    nmacro=1;
    nsection=1;
    nproche=1;
    nmat=1;
    /* Match main0.f90: named geometries use the enclosing-box domain (1),
       while the historical no-argument calculation keeps compact domain (0).
       In particular, --geometry sphere must not silently force nproche=0. */
    nproche = (selected_geometry_case >= 0) ? 1 : 0;
    if (cli_local_value >= 0) nlocal=cli_local_value;
    if (cli_macro_value >= 0) nmacro=cli_macro_value;
    if (cli_force_value >= 0) nforce=cli_force_value;
    if (cli_force_density_value >= 0) nforced=cli_force_density_value;
    if (cli_torque_value >= 0) ntorque=cli_torque_value;
    if (cli_torque_density_value >= 0) ntorqued=cli_torque_density_value;
    if (cli_energy_value >= 0) nenergie=cli_energy_value;
    if (cli_near_domain_set) nproche=cli_near_domain_value;
    if (cli_background_epsilon_set) eps0i=cli_background_epsilon_value;
    if (cli_abc_type_set) ntypeabc=cli_abc_type_value;
    if (cli_abc_nrbc_set) lnrbc=cli_abc_nrbc_value;
    if (cli_abc_cos_set) lcos=cli_abc_cos_value;
    if (cli_quick_lens_value >= 0) nquicklens=cli_quick_lens_value;
    if (cli_quick_diffraction_value >= 0) nquickdiffracte=cli_quick_diffraction_value;
    if (cli_scattering_value >= 0) nsectionsca=cli_scattering_value;
    if (cli_scattering_theta_points >= 0) ntheta=cli_scattering_theta_points;
    if (cli_scattering_phi_points >= 0) nphi=cli_scattering_phi_points;
    if(const char* matfiles=getenv("IFDDA_TEST_GEOMETRY_ASCII"))
        if(strcmp(matfiles,"1")==0)nmat=0;
    /* IFDDA_TEST_GEOMETRY_ASCII above remains the explicit output-mode override. */
    switch(cli_option_case) {
    case 1: nlocal=0; break; case 2: nmacro=0; break; case 3: nsection=0; break;
    case 4: nsectionsca=1; ntheta=12; nphi=24; break;
    case 5: nsectionsca=1; nquickdiffracte=1; break; case 6: nenergie=1; break;
    case 7: nquad=1; break; case 8: nproche=1; break;
    case 9: nproche=2; nnnr=6; nxmp=nymp=nzmp=1; break;
    case 10: nforce=1; break; case 11: nforce=ntorque=1; break;
    case 12: nlentille=1; ntypemic=0; nquicklens=0; break;
    case 13: nlentille=1; ntypemic=0; nquicklens=1; break;
    case 14: nlentille=1; ntypemic=0; nquicklens=0; nside=1; break;
    case 15: case 16: case 17: case 18: case 19: case 20:
        nlentille=1; ntypemic=cli_option_case-14; break;
    case 21: nlentille=1; ntypemic=7; break;
    case 22: nquad=2; break; case 23: nquad=3; break;
    case 24: nquad=4; break; case 25: nquad=5; break;
    case 26: nforce=nforced=1; break; case 27: nforce=ntorque=ntorqued=1; break;
    case 28: nlecture=1; fstring(filereread,64,"coverage-nlecture"); break;
    }
    if(cli_local_value>=0)nlocal=cli_local_value;
    if(cli_macro_value>=0)nmacro=cli_macro_value;
    if(cli_force_value>=0)nforce=cli_force_value;
    if(cli_force_density_value>=0)nforced=cli_force_density_value;
    if(cli_torque_value>=0)ntorque=cli_torque_value;
    if(cli_torque_density_value>=0)ntorqued=cli_torque_density_value;
    if(cli_energy_value>=0)nenergie=cli_energy_value;
    if(cli_near_domain_set)nproche=cli_near_domain_value;
    if(cli_padding_set && nproche!=2) { fputs("--near-padding requires --near-domain 2\n",stderr); return EXIT_FAILURE; }
    if(cli_padding_set) { nxmp=cli_padding[0]; nymp=cli_padding[1]; nzmp=cli_padding[2]; }
    if(cli_geometry_only_set)nobjet=cli_geometry_only;
    if(cli_save_mode_set)nmat=cli_save_mode;
    printf("[IF-DDA] Vector solver driver: inverserigopt (nproche=%d)\n",
           nproche);
    numaper=0.9; numaperinc=0.9; zlens=200.; gross=100.;
    fstring(beam,64,selected_illumination);
    printf("[IF-DDA] Illumination selected: %s\n", selected_illumination);

    fstring(object,64,"sphere");

    fstring(trope,3,"iso");
    if(niso==3)fstring(trope,3,"ani");

    fstring(methodeit,12,selected_solver_method);
    printf("[IF-DDA] Solver selected: %s\n", selected_solver_method);
    fstring(polarizability,2,"RR"); fstring(filereread,64,"toto");
    if(const char* pol=getenv("IFDDA_TEST_POLARIZABILITY")) {
        if(strcmp(pol,"CM")!=0 && strcmp(pol,"RR")!=0 && strcmp(pol,"LR")!=0){
            fputs("IFDDA_TEST_POLARIZABILITY must be CM, RR or LR\n",stderr);
            return EXIT_FAILURE;
        }
        fstring(polarizability,2,pol);
    }
    fstring(h5file,100,"ifdda.h5"); fstring(namefileobj,64,"");
    fstring(namefileinc,64,""); memset(infostr,32,64);
    if (cli_polarizability_set) fstring(polarizability,2,cli_polarizability);
    if (cli_hdf5_file_set) fstring(h5file,100,cli_hdf5_file);
    if (cli_object_file_set) fstring(namefileobj,64,cli_object_file);
    if (cli_field_file_set) fstring(filereread,64,cli_field_file);
    if (cli_arbitrary_file_set) fstring(namefileinc,64,cli_arbitrary_file);
    if (cli_read_field_set) nlecture=cli_read_field;
    if (cli_save_mode_set) nmat=cli_save_mode;
    if (cli_option_case==28) { nlecture=1; fstring(filereread,64,"coverage-nlecture"); }
    if (cli_isotropy_set) fstring(trope,3,cli_isotropy);
    memset(materiaumulti,32,sizeof materiaumulti);
    fstring(materiaumulti,64,"xx");
    size_t cells=(size_t)nxm*nym*nzm, fft=(size_t)nfft2d*nfft2d;
    size_t tensor=(size_t)(nxm+1)*(nym+1)*(nzm+1);
    rayonmulti = static_cast<cdmlib_real *>(alloc_array(2, sizeof(cdmlib_real)));
    xgmulti = static_cast<cdmlib_real *>(alloc_array(2, sizeof(cdmlib_real)));
    ygmulti = static_cast<cdmlib_real *>(alloc_array(2, sizeof(cdmlib_real)));
    zgmulti = static_cast<cdmlib_real *>(alloc_array(2, sizeof(cdmlib_real)));
    epsmulti = static_cast<cdmlib_complex *>(alloc_array(2, sizeof(cdmlib_complex)));
    epsanimulti = static_cast<cdmlib_complex *>(alloc_array(18, sizeof(cdmlib_complex)));
    thetam = static_cast<cdmlib_real *>(alloc_array(10, sizeof(cdmlib_real)));
    phim = static_cast<cdmlib_real *>(alloc_array(10, sizeof(cdmlib_real)));
    ppm = static_cast<cdmlib_real *>(alloc_array(10, sizeof(cdmlib_real)));
    ssm = static_cast<cdmlib_real *>(alloc_array(10, sizeof(cdmlib_real)));
    E0m = static_cast<cdmlib_complex *>(alloc_array(10, sizeof(cdmlib_complex)));
    if (solver_name_equal(selected_illumination,"pwavelinear")) {
        /* Match main0.f90: its default linear plane wave has pp=0, ss=1. */
        theta=0.; phi=0.; pp=0.; ss=1.;
    } else if (solver_name_equal(selected_illumination,"pwavecircular")) {
        ss=1.;
    } else if (solver_name_equal(selected_illumination,"wavelinearmulti")) {
        nbinc=1;
        thetam[0]=0.; phim[0]=0.; ppm[0]=0.; ssm[0]=1.; E0m[0]=1.;
    } else if (solver_name_equal(selected_illumination,"antenna")) {
        xgaus=-300.; ygaus=0.; zgaus=0.; theta=0.; phi=0.;
    } else if (solver_name_equal(selected_illumination,"arbitrary")) {
        fstring(namefileinc,64,"incarbitrary.in");
    } else if (solver_name_equal(selected_illumination,"greentensor")) {
        xgaus=0.; ygaus=0.; zgaus=0.;
    } else if (solver_name_equal(selected_illumination,"speckle") ||
               solver_name_equal(selected_illumination,"demispeckle") ||
               solver_name_equal(selected_illumination,"confocal") ||
               solver_name_equal(selected_illumination,"demiconfocal")) {
        xgaus=0.; ygaus=0.; zgaus=0.;
        numaperil=1.;
    } else if (solver_name_equal(selected_illumination,"gparawavelinear") ||
               solver_name_equal(selected_illumination,"gwavelinear") ||
               solver_name_equal(selected_illumination,"gfftwavelinear")) {
        ss=1.; pp=0.; xgaus=0.; ygaus=0.; zgaus=0.;
    } else if (solver_name_equal(selected_illumination,"gparawavecircular") ||
               solver_name_equal(selected_illumination,"gwavecircular") ||
               solver_name_equal(selected_illumination,"gfftwavecircular")) {
        ss=1.; xgaus=0.; ygaus=0.; zgaus=0.;
    }
    if (!solver_name_equal(selected_illumination,"pwavelinear") &&
        !solver_name_equal(selected_illumination,"pwavecircular")) {
        nsection=0;
    }
    if (cli_theta_value != 0) theta=cli_theta_value;
    if (cli_phi_value != 0) phi=cli_phi_value;
    if (cli_pp_value != 0) pp=cli_pp_value;
    if (cli_ss_value != 0) ss=cli_ss_value;
    if (cli_illumination_na != 0) numaperil=cli_illumination_na;
    if (cli_arbitrary_file_set) fstring(namefileinc,64,cli_arbitrary_file);
    if (cli_gaussian_center_set) { xgaus=cli_gaussian_center[0]; ygaus=cli_gaussian_center[1]; zgaus=cli_gaussian_center[2]; }
    forcet = static_cast<cdmlib_real *>(alloc_array(3, sizeof(cdmlib_real)));
    couplet = static_cast<cdmlib_real *>(alloc_array(3, sizeof(cdmlib_real)));
    forcexmulti = static_cast<cdmlib_real *>(alloc_array(2, sizeof(cdmlib_real)));
    forceymulti = static_cast<cdmlib_real *>(alloc_array(2, sizeof(cdmlib_real)));
    forcezmulti = static_cast<cdmlib_real *>(alloc_array(2, sizeof(cdmlib_real)));
    torquexmulti = static_cast<cdmlib_real *>(alloc_array(2, sizeof(cdmlib_real)));
    torqueymulti = static_cast<cdmlib_real *>(alloc_array(2, sizeof(cdmlib_real)));
    torquezmulti = static_cast<cdmlib_real *>(alloc_array(2, sizeof(cdmlib_real)));
    incidentfield = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    localfield = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    macroscopicfield = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    xs = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    ys = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    zs = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    xswf = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    yswf = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    zswf = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    thetafield = static_cast<cdmlib_real *>(alloc_array(fft, sizeof(cdmlib_real)));
    phifield = static_cast<cdmlib_real *>(alloc_array(fft, sizeof(cdmlib_real)));
    poyntingfield = static_cast<cdmlib_real *>(alloc_array(fft, sizeof(cdmlib_real)));
    kxypoynting = static_cast<cdmlib_real *>(alloc_array(nfft2d, sizeof(cdmlib_real)));
    poyntingfieldpos = static_cast<cdmlib_real *>(alloc_array(fft, sizeof(cdmlib_real)));
    poyntingfieldneg = static_cast<cdmlib_real *>(alloc_array(fft, sizeof(cdmlib_real)));
    forcex = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    forcey = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    forcez = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    torquex = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    torquey = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    torquez = static_cast<cdmlib_real *>(alloc_array(cells, sizeof(cdmlib_real)));
    incidentfieldx = static_cast<cdmlib_complex *>(alloc_array(cells, sizeof(cdmlib_complex)));
    incidentfieldy = static_cast<cdmlib_complex *>(alloc_array(cells, sizeof(cdmlib_complex)));
    incidentfieldz = static_cast<cdmlib_complex *>(alloc_array(cells, sizeof(cdmlib_complex)));
    localfieldx = static_cast<cdmlib_complex *>(alloc_array(cells, sizeof(cdmlib_complex)));
    localfieldy = static_cast<cdmlib_complex *>(alloc_array(cells, sizeof(cdmlib_complex)));
    localfieldz = static_cast<cdmlib_complex *>(alloc_array(cells, sizeof(cdmlib_complex)));
    macroscopicfieldx = static_cast<cdmlib_complex *>(alloc_array(cells, sizeof(cdmlib_complex)));
    macroscopicfieldy = static_cast<cdmlib_complex *>(alloc_array(cells, sizeof(cdmlib_complex)));
    macroscopicfieldz = static_cast<cdmlib_complex *>(alloc_array(cells, sizeof(cdmlib_complex)));
    polarisa = static_cast<cdmlib_complex *>(alloc_array(cells*niso*niso, sizeof(cdmlib_complex)));
    epsilon = static_cast<cdmlib_complex *>(alloc_array(cells*niso*niso, sizeof(cdmlib_complex)));
    Eimagex = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Eimagey = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Eimagez = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Eimageincx = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Eimageincy = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Eimageincz = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Efourierx = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Efouriery = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Efourierz = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Efourierincx = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Efourierincy = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    Efourierincz = static_cast<cdmlib_complex *>(alloc_array(fft, sizeof(cdmlib_complex)));
    kxy = static_cast<cdmlib_real *>(alloc_array(nfft2d, sizeof(cdmlib_real)));
    xy = static_cast<cdmlib_real *>(alloc_array(nfft2d, sizeof(cdmlib_real)));
    FF = static_cast<cdmlib_complex *>(alloc_array(3*cells, sizeof(cdmlib_complex)));
    FF0 = static_cast<cdmlib_complex *>(alloc_array(3*cells, sizeof(cdmlib_complex)));
    FFloc = static_cast<cdmlib_complex *>(alloc_array(3*cells, sizeof(cdmlib_complex)));
    xr = static_cast<cdmlib_complex *>(alloc_array(3*cells, sizeof(cdmlib_complex)));
    xi = static_cast<cdmlib_complex *>(alloc_array(3*cells, sizeof(cdmlib_complex)));
    FFprecon = static_cast<cdmlib_complex *>(alloc_array(3*(size_t)nxym1*nzm1, sizeof(cdmlib_complex)));
    wrk = static_cast<cdmlib_complex *>(alloc_array(3*cells*nlar, sizeof(cdmlib_complex)));
    FFTTENSORxx = static_cast<cdmlib_complex *>(alloc_array(tensor, sizeof(cdmlib_complex)));
    FFTTENSORxy = static_cast<cdmlib_complex *>(alloc_array(tensor, sizeof(cdmlib_complex)));
    FFTTENSORxz = static_cast<cdmlib_complex *>(alloc_array(tensor, sizeof(cdmlib_complex)));
    FFTTENSORyy = static_cast<cdmlib_complex *>(alloc_array(tensor, sizeof(cdmlib_complex)));
    FFTTENSORyz = static_cast<cdmlib_complex *>(alloc_array(tensor, sizeof(cdmlib_complex)));
    FFTTENSORzz = static_cast<cdmlib_complex *>(alloc_array(tensor, sizeof(cdmlib_complex)));
    Ediffkzpos = static_cast<cdmlib_complex *>(alloc_array(3*fft, sizeof(cdmlib_complex)));
    Ediffkzneg = static_cast<cdmlib_complex *>(alloc_array(3*fft, sizeof(cdmlib_complex)));
    sdetnn = static_cast<cdmlib_complex *>(alloc_array((size_t)nzms1*nzms1*nxym1, sizeof(cdmlib_complex)));
    Tabdip = static_cast<int *>(alloc_array(cells, sizeof(int)));
    Tabmulti = static_cast<int *>(alloc_array(cells, sizeof(int)));
    tabfft2 = static_cast<int *>(alloc_array(nfft2d, sizeof(int)));
    rayonmulti[0]=125.; epsmulti[0]=COMPLEX(2.0,0.0);
    if (selected_geometry_case >= 0) {
        fstring(object,64,ifdda_test_geometry_names[selected_geometry_case]);
        /* Common benchmark fixture. All dimensions in nm on entry to cdmlib.
           The default (no --geometry) remains byte-for-byte historical. */
        numberobjet=1;
        nnnr=32;
        fstring(materiaumulti+64,64,"xx");
        epsmulti[0]=IFDDA_TEST_COMPLEX(2.0,0.0);
        switch (selected_geometry_case) {
        case 0: /* concentricsphere */
            numberobjet=2; rayonmulti[0]=2.5; rayonmulti[1]=5.;
            epsmulti[0]=IFDDA_TEST_COMPLEX(2.25,0.);
            epsmulti[1]=IFDDA_TEST_COMPLEX(1.25,0.); break;
        case 1: /* sphere: match the Fortran coverage driver's benchmark */
            rayonmulti[0]=100.;
            epsmulti[0]=IFDDA_TEST_COMPLEX(1.6,0.); break;
        case 2: /* cube */
            side=100.; nnnr=10;
            epsmulti[0]=IFDDA_TEST_COMPLEX(2.,0.5); break;
        case 3: /* cuboid1 */
            sidex=20.; sidey=40.; sidez=10.; nnnr=20;
            epsmulti[0]=IFDDA_TEST_COMPLEX(2.,0.5); break;
        case 4: /* cuboid2 */
            aretecube=25.; epsmulti[0]=IFDDA_TEST_COMPLEX(2.,0.5); break;
        case 5: /* inhomosphere */
            rayonmulti[0]=1000.; lc=lambda; hc=0.2; ng=2; nnnr=20;
            epsmulti[0]=IFDDA_TEST_COMPLEX(2.25,0.); break;
        case 6: /* inhomocuboid1 */
            sidex=2000.; sidey=2000.; sidez=2000.;
            lc=lambda; hc=0.2; ng=2; nnnr=20;
            epsmulti[0]=IFDDA_TEST_COMPLEX(2.25,0.); break;
        case 7: /* inhomocuboid2 */
            aretecube=25.; lc=lambda; hc=0.2; ng=2;
            epsmulti[0]=IFDDA_TEST_COMPLEX(2.25,0.); break;
        case 8: /* ellipsoid, rotated */
            demiaxea=10.; demiaxeb=20.; demiaxec=40.;
            thetaobj=45.; phiobj=45.; nnnr=20;
            epsmulti[0]=IFDDA_TEST_COMPLEX(2.25,0.); break;
        case 9: /* nspheres */
            numberobjet=2; rayonmulti[0]=10.; rayonmulti[1]=15.;
            xgmulti[1]=30.; nnnr=20;
            epsmulti[0]=IFDDA_TEST_COMPLEX(2.25,0.);
            epsmulti[1]=IFDDA_TEST_COMPLEX(1.25,0.); break;
        case 10: /* cylinder, rotated */
            rayonmulti[0]=10.; hauteur=20.; thetaobj=45.; phiobj=45.;
            nnnr=20; epsmulti[0]=IFDDA_TEST_COMPLEX(2.25,0.); break;
        case 11: /* randomsphere1 */
            rayonmulti[0]=120.; density=0.2;
            sidex=2000.; sidey=2000.; sidez=2000.;
            nnnr=20; ng=2;
            epsmulti[0]=IFDDA_TEST_COMPLEX(2.25,0.05); break;
        case 12: /* randomsphere2 */
            rayonmulti[0]=100.; density=0.2; aretecube=25.;
            ng=2; nnnr=20;
            epsmulti[0]=IFDDA_TEST_COMPLEX(2.25,0.05); break;
        case 13: /* arbitrary: arbit.in required in working directory */
            fstring(namefileobj,64,"arbit.in"); break;
        case 14: /* randomspheremulti1 */
            numberobjet=2; rayonmulti[0]=120.; rayonmulti[1]=120.;
            xgmulti[0]=-250.; xgmulti[1]=250.;
            density=0.2; sidex=2000.; sidey=2000.; sidez=2000.;
            ng=2; nnnr=20;
            epsmulti[0]=IFDDA_TEST_COMPLEX(1.2,0.05);
            epsmulti[1]=IFDDA_TEST_COMPLEX(2.25,0.); break;
        }
        printf("[IF-DDA] Geometry selected: %s (ID=%d)\n",
               ifdda_test_geometry_names[selected_geometry_case],
               selected_geometry_case);
    }
    if (cli_grid_size > 0 && !cli_grid_dimensions_set) nnnr=nxm=nym=nzm=cli_grid_size;
    else if (cli_grid_dimensions_set) { nxm=cli_grid_dimensions[0]; nym=cli_grid_dimensions[1]; nzm=cli_grid_dimensions[2]; nnnr=nxm<nym?(nxm<nzm?nxm:nzm):(nym<nzm?nym:nzm); }
    /* Match the Fortran coverage driver: --epsilon overrides every object,
       including both shells of concentricsphere. */
    if (cli_epsilon_set)
        for (int ii=0; ii<numberobjet; ++ii)
            epsmulti[ii]=cdmlib_complex(cli_epsilon_real,cli_epsilon_imag);
    if (cli_radius_value != 0) rayonmulti[0]=cli_radius_value;
    if (cli_geometry_angles_set) {
        thetaobj=cli_geometry_angles[0]; phiobj=cli_geometry_angles[1]; psiobj=cli_geometry_angles[2];
    }
    if (cli_side_length_set) side=sidex=sidey=sidez=cli_side_length;
    if (cli_geometry_dimensions_set) {
        const cdmlib_real *d=cli_geometry_dimensions;
        if (d[0]<=0 || d[1]<=0 || d[2]<=0) {
            fputs("Invalid geometry dimensions\n",stderr); return EXIT_FAILURE;
        }
        switch(selected_geometry_case) {
        case 3: case 6: sidex=d[0]; sidey=d[1]; sidez=d[2]; break;
        case 8: demiaxea=d[0]; demiaxeb=d[1]; demiaxec=d[2]; break;
        case 10: rayonmulti[0]=d[0]; hauteur=d[1]; break;
        default: fputs("--geometry-dimensions not valid for this geometry\n",stderr); return EXIT_FAILURE;
        }
    }
    if (cli_geometry_density_set) density=cli_geometry_density;
    if (cli_cell_size_set) aretecube=cli_cell_size;
    if (cli_geometry_center_set) { xgmulti[0]=cli_geometry_center[0]; ygmulti[0]=cli_geometry_center[1]; zgmulti[0]=cli_geometry_center[2]; }
    if (cli_geometry_only_set) nobjet=cli_geometry_only;
    if (cli_padding_set) { nxmp=cli_padding[0]; nymp=cli_padding[1]; nzmp=cli_padding[2]; }
    if (cli_condenser_na_set) numaperinc=cli_condenser_na;
    if (cli_inner_na_set) numaperinc2=cli_inner_na;
    if (cli_na_center_set) { kcnax=cli_na_center[0]; kcnay=cli_na_center[1]; }
    if (cli_object_file_set) fstring(namefileobj,64,cli_object_file);
    if (cli_microscope>=0) nlentille=cli_microscope;
    if (cli_microscope_type>=0) ntypemic=cli_microscope_type;
    if (cli_microscope_side>=0) nside=cli_microscope_side;
    if (cli_na_set) numaper=cli_na;
    if (cli_magnification_set) gross=cli_magnification;
    if (cli_lens_z_set) zlens=cli_lens_z;
    /* Geometry-only cross-backend verification: avoid conflating a GPU solver
       failure with the GPU geometry construction. Ordinary runs still use
       the single GPBICG1 iterative solver. */
    if(niso==3)for(int q=0;q<9;++q)
        epsanimulti[q]=(q%3==q/3)?epsmulti[0]:IFDDA_TEST_COMPLEX(0.,0.);
    const char *geometry_only = getenv("IFDDA_TEST_GEOMETRY_ONLY");
    if (geometry_only && strcmp(geometry_only,"1")==0) {
        nobjet=1;
        puts("[IF-DDA] Geometry-only verification (no iterative solve)");
    }
    if (cli_geometry_only_set) nobjet=cli_geometry_only;
    /* Append the GNU Fortran hidden string lengths after all explicit arguments. */
    cdmlib_(
        &lambda, &eps0i, &ntypeabc, &lnrbc, &lcos, beam,
        object, trope, &niso, materiaumulti, &nnnr, &tolinit,
        &nlim, methodeit, polarizability, &nprecon, &ninitest, &nquad,
        &nlecture, filereread, &nmat, h5file, &nlocal, &nmacro,
        &nsection, &nsectionsca, &nquickdiffracte, &nrig, &nforce, &nforced,
        &ntorque, &ntorqued, &nproche, &nlentille, &nquicklens, &nenergie,
        &nobjet, &density, &side, &sidex, &sidey, &sidez,
        &hauteur, &numberobjet, rayonmulti, xgmulti, ygmulti, zgmulti,
        epsmulti, epsanimulti, &lc, &hc, &ng, &demiaxea,
        &demiaxeb, &demiaxec, &thetaobj, &phiobj, &psiobj, namefileobj,
        &theta, &phi, &pp, &ss, &P0, &w0,
        &xgaus, &ygaus, &zgaus, &speckseed, namefileinc, thetam,
        phim, ppm, ssm, E0m, &nbinc, infostr,
        &nstop, &nbsphere, &ndipole, &aretecube, &lambda10n, &k0,
        &tol1, &tempsreelmvp, &tempsreeltotal, &ncompte, &nloop, &efficacite,
        &efficaciteref, &efficacitetrans, &Cext, &Cabs, &Csca, &Cscai,
        &gasym, &irra, &E0, forcet, &forcem, couplet,
        &couplem, &nxm, &nym, &nzm, &nxmp, &nymp,
        &nzmp, &nmaxpp, &nxym1, &nzm1, &nzms1, incidentfield,
        localfield, macroscopicfield, xs, ys, zs, xswf,
        yswf, zswf, &ntheta, &nphi, thetafield, phifield,
        poyntingfield, kxypoynting, poyntingfieldpos, poyntingfieldneg, forcex, forcey,
        forcez, forcexmulti, forceymulti, forcezmulti, torquex, torquey,
        torquez, torquexmulti, torqueymulti, torquezmulti, incidentfieldx, incidentfieldy,
        incidentfieldz, localfieldx, localfieldy, localfieldz, macroscopicfieldx, macroscopicfieldy,
        macroscopicfieldz, polarisa, epsilon, &nfft2d, &npoynting, &deltapoyntingx,
        &deltapoyntingy, Eimagex, Eimagey, Eimagez, Eimageincx, Eimageincy,
        Eimageincz, Efourierx, Efouriery, Efourierz, Efourierincx, Efourierincy,
        Efourierincz, kxy, xy, &numaper, &numaperinc, &numaperinc2,
        &kcnax, &kcnay, &numaperil, &gross, &zlens, &psiinc,
        &ntypemic, &nside, FF, FF0, FFloc, xr,
        xi, FFprecon, wrk, &nlar, FFTTENSORxx, FFTTENSORxy,
        FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, Ediffkzpos, Ediffkzneg,
        sdetnn, Tabdip, Tabmulti, tabfft2,
        (size_t)64, (size_t)64, (size_t)3, (size_t)64, (size_t)12, (size_t)2, (size_t)64, (size_t)100, (size_t)64, (size_t)64, (size_t)64);
    printf("CDMLIB_GEOMETRY_RESULT %d %d\n",nbsphere,ndipole);
    size_t occupied=0;
    unsigned long long geometry_mask_hash=0;
    for(size_t p=0;p<(size_t)nbsphere && p<cells;++p) {
        const bool in_material=epsilon[p]!=cdmlib_complex(0.0,0.0) &&
            epsilon[p]!=cdmlib_complex(1.0,0.0);
        if(in_material)++occupied;
        geometry_mask_hash=(geometry_mask_hash*131ULL+
            (in_material?1ULL:0ULL))%1000000007ULL;
    }
    printf("CDMLIB_MATERIAL_RESULT %zu\n",occupied);
    printf("CDMLIB_MATERIAL_HASH %llu\n",geometry_mask_hash);
    if (nstop == 0) puts("Computation finished without problem:");
    else { fprintf(stderr,"Programm finished with problem: %.64s\n",infostr); }
    printf("C++ results: Cext=%.16e Csca=%.16e Cabs=%.16e\n",Cext,Csca,Cabs);
    printf("C++ results: nstop=%d ncompte=%d nloop=%d\n",nstop,ncompte,nloop);
    cdmlib_real fieldmax = 0;
    cdmlib_real localmax = 0;
    cdmlib_real forcearrmax = 0;
    cdmlib_real torquearrmax = 0;
    for (int p = 0; p < ndipole; ++p) {
        if (incidentfieldx && incidentfieldy && incidentfieldz) {
            fieldmax = std::max(fieldmax, (cdmlib_real)std::abs(incidentfieldx[p]));
            fieldmax = std::max(fieldmax, (cdmlib_real)std::abs(incidentfieldy[p]));
            fieldmax = std::max(fieldmax, (cdmlib_real)std::abs(incidentfieldz[p]));
        }
        if (localfieldx && localfieldy && localfieldz) {
            localmax = std::max(localmax, (cdmlib_real)std::abs(localfieldx[p]));
            localmax = std::max(localmax, (cdmlib_real)std::abs(localfieldy[p]));
            localmax = std::max(localmax, (cdmlib_real)std::abs(localfieldz[p]));
        }
        if (forcex && forcey && forcez) {
            forcearrmax = std::max(forcearrmax, (cdmlib_real)std::abs(forcex[p]));
            forcearrmax = std::max(forcearrmax, (cdmlib_real)std::abs(forcey[p]));
            forcearrmax = std::max(forcearrmax, (cdmlib_real)std::abs(forcez[p]));
        }
        if (torquex && torquey && torquez) {
            torquearrmax = std::max(torquearrmax, (cdmlib_real)std::abs(torquex[p]));
            torquearrmax = std::max(torquearrmax, (cdmlib_real)std::abs(torquey[p]));
            torquearrmax = std::max(torquearrmax, (cdmlib_real)std::abs(torquez[p]));
        }
    }
    printf("CDMLIB_RESULT %d %d %d %.16e %.16e %.16e %.16e %d %.16e %.16e %.16e %.16e %.16e %.16e %.16e\n",
           nstop, ncompte, nloop, Cext, Csca, Cabs, tol1, ndipole,
           fieldmax, localmax, forcem, couplem, efficacite, forcearrmax, torquearrmax);
    free(rayonmulti);
    free(xgmulti);
    free(ygmulti);
    free(zgmulti);
    free(epsmulti);
    free(epsanimulti);
    free(thetam);
    free(phim);
    free(ppm);
    free(ssm);
    free(E0m);
    free(forcet);
    free(couplet);
    free(forcexmulti);
    free(forceymulti);
    free(forcezmulti);
    free(torquexmulti);
    free(torqueymulti);
    free(torquezmulti);
    free(incidentfield);
    free(localfield);
    free(macroscopicfield);
    free(xs);
    free(ys);
    free(zs);
    free(xswf);
    free(yswf);
    free(zswf);
    free(thetafield);
    free(phifield);
    free(poyntingfield);
    free(kxypoynting);
    free(poyntingfieldpos);
    free(poyntingfieldneg);
    free(forcex);
    free(forcey);
    free(forcez);
    free(torquex);
    free(torquey);
    free(torquez);
    free(incidentfieldx);
    free(incidentfieldy);
    free(incidentfieldz);
    free(localfieldx);
    free(localfieldy);
    free(localfieldz);
    free(macroscopicfieldx);
    free(macroscopicfieldy);
    free(macroscopicfieldz);
    free(polarisa);
    free(epsilon);
    free(Eimagex);
    free(Eimagey);
    free(Eimagez);
    free(Eimageincx);
    free(Eimageincy);
    free(Eimageincz);
    free(Efourierx);
    free(Efouriery);
    free(Efourierz);
    free(Efourierincx);
    free(Efourierincy);
    free(Efourierincz);
    free(kxy);
    free(xy);
    free(FF);
    free(FF0);
    free(FFloc);
    free(xr);
    free(xi);
    free(FFprecon);
    free(wrk);
    free(FFTTENSORxx);
    free(FFTTENSORxy);
    free(FFTTENSORxz);
    free(FFTTENSORyy);
    free(FFTTENSORyz);
    free(FFTTENSORzz);
    free(Ediffkzpos);
    free(Ediffkzneg);
    free(sdetnn);
    free(Tabdip);
    free(Tabmulti);
    free(tabfft2);
    return nstop == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Test-only driver: exercise the full cdmlib_ entry point twice within the
// same process. Each invocation resets the CUDA context before geometry setup.
// An optional larger second grid tests that the old GPU capacities were reset.
int main(int argc, char **argv) {
    int repeat = 1;
    const char* second_grid = nullptr;
    std::vector<char*> args;
    args.reserve(size_t(argc) + 2);
    args.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--repeat-calculation") == 0) {
            if (++i >= argc || !parse_int_value(argv[i], &repeat) ||
                repeat < 1 || repeat > 10) {
                fputs("--repeat-calculation requires an integer between 1 and 10\n", stderr);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--second-grid-size") == 0) {
            if (++i >= argc) {
                fputs("--second-grid-size requires a grid dimension\n", stderr);
                return EXIT_FAILURE;
            }
            second_grid = argv[i];
        } else {
            args.push_back(argv[i]);
        }
    }
    if (second_grid && repeat < 2) {
        fputs("--second-grid-size requires --repeat-calculation 2 or more\n", stderr);
        return EXIT_FAILURE;
    }
    for (int run = 0; run < repeat; ++run) {
        std::vector<char*> current_args = args;
        if (run > 0 && second_grid) {
            current_args.push_back(const_cast<char*>("--grid-size"));
            current_args.push_back(const_cast<char*>(second_grid));
        }
        if (repeat > 1)
            printf("IFDDA_REPEAT_CALCULATION %d/%d\n", run + 1, repeat);
        if (run_one_calculation(int(current_args.size()), current_args.data()) != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
