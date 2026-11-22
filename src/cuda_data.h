#ifndef PRODUITMATVECTFFT_CUDA_DATA_H
#define PRODUITMATVECTFFT_CUDA_DATA_H

#include "cdmlib_cpp_precision.hpp"
#include "kernel.h"

#include <complex>
#include <cstddef>

// The shared CPU/Fortran precision must agree with the CUDA build precision.
static_assert(std::is_same<REAL, CU_REAL>::value, "CPU and CUDA real precisions differ");
static_assert(std::is_same<COMPLEX, std::complex<CU_REAL>>::value,
              "CPU and CUDA complex precisions differ");

namespace cuda_utils {
enum class TransferCategory : int;
}

// Central entry points for every host/device transfer. Device-to-device work
// stays with the solver/kernel that owns it and is intentionally not counted.
namespace cuda_data {
cudaError_t copy_host_to_device(void* device_destination,
                                const void* host_source,
                                size_t bytes,
                                cuda_utils::TransferCategory category);
cudaError_t copy_device_to_host(void* host_destination,
                                const void* device_source,
                                size_t bytes,
                                cuda_utils::TransferCategory category);
cudaError_t copy_host_complex_to_device(CUCOMPLEX* device_destination,
                                        const std::complex<CU_REAL>* host_source,
                                        size_t count,
                                        cuda_utils::TransferCategory category);
cudaError_t copy_device_complex_to_host(std::complex<CU_REAL>* host_destination,
                                        const CUCOMPLEX* device_source,
                                        size_t count,
                                        cuda_utils::TransferCategory category);
cudaError_t
copy_device_to_device(void* device_destination, const void* device_source, size_t bytes);

typedef struct CubeGeometryDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CU_INT* dipole_map = nullptr;
    CU_REAL* shape_parameters = nullptr;
} CubeGeometryDeviceBuffers;

typedef struct SphereGeometryCompactDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CU_INT* prefix = nullptr;
    CU_INT* packed_count = nullptr;
    void* scan_temporary = nullptr;
    size_t scan_temporary_bytes = 0;
} SphereGeometryCompactDeviceBuffers;

bool get_device_sphere_geometry_compact_buffers(size_t count,
                                                size_t scan_temporary_bytes,
                                                SphereGeometryCompactDeviceBuffers* buffers);
// Shared fixed GPU arrays: declarations are centralized in cuda_interfaces.h.
bool register_device_geometry_coordinate_aliases(const REAL* host_xs,
                                                 const REAL* host_ys,
                                                 const REAL* host_zs,
                                                 size_t count,
                                                 const CU_REAL* device_xs,
                                                 const CU_REAL* device_ys,
                                                 const CU_REAL* device_zs);
bool get_device_geometry_coordinate_aliases(const REAL* host_xs,
                                            const REAL* host_ys,
                                            const REAL* host_zs,
                                            size_t count,
                                            const CU_REAL** device_xs,
                                            const CU_REAL** device_ys,
                                            const CU_REAL** device_zs);
void clear_device_geometry_coordinate_aliases();

// Host-only descriptor of a regular lattice. Its pointers refer to the
// original Fortran ABI outputs; no GPU coordinate buffers are allocated.
// The shared object builder publishes this after constructing tabdip, and
// Fortran illumination may use it to access resident tabdip directly.
struct HostRegularGeometry {
    const REAL* xs = nullptr;
    const REAL* ys = nullptr;
    const REAL* zs = nullptr;
    const int* tabdip = nullptr;
    int nx = 0, ny = 0, nz = 0;
    int compact_count = 0;
    REAL cell_size = REAL(0);
};
bool register_host_regular_geometry(const HostRegularGeometry& geometry);
bool get_host_regular_geometry(const REAL* xs, const REAL* ys, const REAL* zs,
                              int compact_count, HostRegularGeometry* geometry);
void clear_host_regular_geometry();

typedef struct DipoleInsideDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CU_INT* first_match = nullptr;
} DipoleInsideDeviceBuffers;

bool get_device_dipole_inside_buffers(size_t count, DipoleInsideDeviceBuffers* buffers);

typedef struct PolarizabilityInitDeviceBuffers {
    CU_INT* dipole_map = nullptr;
    CUCOMPLEX* polarizability = nullptr;
    CUCOMPLEX* epsilon = nullptr;
} PolarizabilityInitDeviceBuffers;

bool get_device_polarizability_init_buffers(size_t dipole_count,
                                            size_t tensor_count,
                                            PolarizabilityInitDeviceBuffers* buffers);

typedef struct ScalarPolarizabilityDeviceBuffers {
    CUCOMPLEX* value = nullptr;
    CU_REAL* sldr = nullptr;
} ScalarPolarizabilityDeviceBuffers;

bool get_device_scalar_polarizability_buffers(ScalarPolarizabilityDeviceBuffers* buffers);

typedef struct PolarizabilityBatchDeviceBuffers {
    CUCOMPLEX* epsilon = nullptr;
    CUCOMPLEX* values = nullptr;
    CU_INT* inside = nullptr;
    CU_REAL* sldr = nullptr;
} PolarizabilityBatchDeviceBuffers;

bool get_device_polarizability_batch_buffers(size_t count,
                                             PolarizabilityBatchDeviceBuffers* buffers);

typedef struct TensorPolarizabilityDeviceBuffers {
    CUCOMPLEX* epsilon = nullptr;
    CUCOMPLEX* value = nullptr;
    CU_INT* status = nullptr;
} TensorPolarizabilityDeviceBuffers;

bool get_device_tensor_polarizability_buffers(TensorPolarizabilityDeviceBuffers* buffers);

typedef struct GaussianPowerDeviceBuffers {
    CUCOMPLEX* field = nullptr;
    CU_REAL* irradiance = nullptr;
} GaussianPowerDeviceBuffers;

bool get_device_gaussian_power_buffers(GaussianPowerDeviceBuffers* buffers);

typedef struct MaterialInterpolationDeviceBuffers {
    CU_REAL* result = nullptr;
} MaterialInterpolationDeviceBuffers;

bool get_device_material_interpolation_buffers(MaterialInterpolationDeviceBuffers* buffers);

typedef struct MicroscopeMaskDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* extrema = nullptr;
    CUCOMPLEX* mask = nullptr;
    CU_INT* selected_count = nullptr;
} MicroscopeMaskDeviceBuffers;

bool get_device_microscope_mask_buffers(size_t coordinate_count,
                                        size_t mask_capacity,
                                        MicroscopeMaskDeviceBuffers* buffers);

// Persistent storage for the dark-field microscopy path. Zero-sized request
// fields disable optional allocations so each microscope mode asks only for
// the planes and solver buffers it actually uses.
typedef struct DarkFieldMicroscopeDeviceBuffers {
    CU_INT* dipole_map = nullptr;
    CUCOMPLEX* polarization = nullptr;
    CUCOMPLEX* scalar_local = nullptr;
    CUCOMPLEX* image_planes = nullptr;
    CUCOMPLEX* spectrum_planes = nullptr;
    CUCOMPLEX* detector_planes = nullptr;
    CUCOMPLEX* solver_fields = nullptr;
    CUCOMPLEX* scalar_solution = nullptr;
    CUCOMPLEX* image_scratch = nullptr;
} DarkFieldMicroscopeDeviceBuffers;

bool get_device_dark_field_microscope_buffers(size_t dipole_count,
                                              size_t polarization_count,
                                              size_t scalar_local_count,
                                              size_t image_value_count,
                                              size_t spectrum_value_count,
                                              size_t detector_value_count,
                                              size_t solver_field_count,
                                              size_t scalar_solution_count,
                                              size_t image_scratch_count,
                                              DarkFieldMicroscopeDeviceBuffers* buffers);

typedef struct MicroscopeIncidentDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CUCOMPLEX* incident = nullptr;
    CUCOMPLEX* initial = nullptr;
} MicroscopeIncidentDeviceBuffers;

bool get_device_microscope_incident_buffers(size_t dipole_count,
                                            MicroscopeIncidentDeviceBuffers* buffers);

typedef struct BrightFieldMicroscopeDeviceBuffers {
    CUCOMPLEX* scattered_field = nullptr;
    CUCOMPLEX* diffraction_field = nullptr;
    CUCOMPLEX* fourier_fields = nullptr;
    CU_REAL* image_intensity = nullptr;
    cufftHandle fft_plan = 0;
} BrightFieldMicroscopeDeviceBuffers;

bool get_device_bright_field_microscope_buffers(size_t dipole_count,
                                                size_t fft_plane_count,
                                                int nfft,
                                                BrightFieldMicroscopeDeviceBuffers* buffers);

// Persistent buffers for confocal fluorescence microscopy. Each requested
// element count may be zero when that operation is not used by the caller.
typedef struct ConfocalFluorescenceDeviceBuffers {
    CUCOMPLEX* field_planes = nullptr;
    CU_REAL* intensity_stack = nullptr;
    CUCOMPLEX* diffraction = nullptr;
    CUCOMPLEX* spectrum_planes = nullptr;
} ConfocalFluorescenceDeviceBuffers;

bool get_device_confocal_fluorescence_buffers(size_t field_value_count,
                                              size_t intensity_value_count,
                                              size_t diffraction_value_count,
                                              size_t spectrum_value_count,
                                              ConfocalFluorescenceDeviceBuffers* buffers);

typedef struct ConfocalMicroscopeDeviceBuffers {
    CUCOMPLEX* dipole_field = nullptr;
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CUCOMPLEX* diffraction = nullptr;
    CUCOMPLEX* dense_field = nullptr;
} ConfocalMicroscopeDeviceBuffers;

bool get_device_confocal_microscope_buffers(size_t dipole_count,
                                            size_t diffraction_value_count,
                                            size_t dense_field_value_count,
                                            ConfocalMicroscopeDeviceBuffers* buffers);

typedef struct ConfocalDetectorDeviceBuffers {
    CUCOMPLEX* diffraction = nullptr;
    CUCOMPLEX* fourier_fields = nullptr;
    CUCOMPLEX* image_stack = nullptr;
    CU_INT* status = nullptr;
} ConfocalDetectorDeviceBuffers;

bool get_device_confocal_detector_buffers(size_t plane_value_count,
                                          size_t image_value_count,
                                          ConfocalDetectorDeviceBuffers* buffers);

// Persistent storage shared by the wide-field near-field calculation. Counts
// are supplied as a request so unused stages do not allocate their buffers.
typedef struct WideFieldBufferRequest {
    size_t grid_count = 0;
    size_t dipole_count = 0;
    size_t dipole_field_count = 0;
    size_t dipole_incident_count = 0;
    size_t dipole_material_count = 0;
    size_t dipole_material_source_count = 0;
    size_t grid_material_count = 0;
    size_t fft_vector_count = 0;
    size_t green_tensor_count = 0;
} WideFieldBufferRequest;

typedef struct WideFieldDeviceBuffers {
    CU_REAL* dipole_x = nullptr;
    CU_REAL* dipole_y = nullptr;
    CU_REAL* dipole_z = nullptr;
    CU_REAL* grid_x = nullptr;
    CU_REAL* grid_y = nullptr;
    CU_REAL* grid_z = nullptr;
    CUCOMPLEX* incident = nullptr;
    CU_REAL* incident_magnitude = nullptr;
    CUCOMPLEX* initial_field = nullptr;
    CUCOMPLEX* scattered_field = nullptr;
    CUCOMPLEX* local_field = nullptr;
    CUCOMPLEX* polarization = nullptr;
    CUCOMPLEX* epsilon = nullptr;
    CUCOMPLEX* dipole_scattered = nullptr;
    CUCOMPLEX* dipole_incident = nullptr;
    CUCOMPLEX* dipole_epsilon = nullptr;
    CUCOMPLEX* dipole_epsilon_source = nullptr;
    CUCOMPLEX* fft_vectors = nullptr;
    CUCOMPLEX* green_tensors = nullptr;
    CU_INT* dipole_map = nullptr;
} WideFieldDeviceBuffers;

bool get_device_widefield_buffers(const WideFieldBufferRequest& request,
                                  WideFieldDeviceBuffers* buffers);

typedef struct AbcChiDeviceBuffers {
    CUCOMPLEX* epsilon_output = nullptr;
    CUCOMPLEX* contrast_output = nullptr;
} AbcChiDeviceBuffers;

bool get_device_abcchi_buffers(AbcChiDeviceBuffers* buffers);

bool get_device_cube_geometry_buffers(size_t count,
                                      CubeGeometryDeviceBuffers* buffers,
                                      size_t shape_parameter_count = 0);

typedef struct InhomogeneousMaterialDeviceBuffers {
    CUCOMPLEX* spectrum = nullptr;
    CU_REAL* phases = nullptr;
    CU_REAL* ones = nullptr;
    CU_REAL* real_values = nullptr;
    CU_REAL* magnitude_squares = nullptr;
    cufftHandle fft_plan = 0;
} InhomogeneousMaterialDeviceBuffers;

bool get_device_inhomogeneous_material_buffers(int nx,
                                               int ny,
                                               int nz,
                                               InhomogeneousMaterialDeviceBuffers* buffers);
// Named, process-lifetime scratch buffers for CUDA drivers and Chan setup.
// A buffer is grown only when needed and is never released during a solve.
void* get_device_workspace_buffer_bytes(const char* name, size_t bytes);

typedef struct MieReferenceDeviceBuffers {
    CUCOMPLEX* recurrence = nullptr;
    CU_REAL* outputs = nullptr;
} MieReferenceDeviceBuffers;

bool get_device_mie_reference_buffers(size_t recurrence_count, MieReferenceDeviceBuffers* buffers);

// All persistent Chan-preconditioner GPU workspaces are returned together.
// Zero counts leave the corresponding pointer null; a false return means that
// at least one requested buffer could not be obtained.
typedef struct ChanPreconditionerBufferSizes {
    int scalar_nx = 0;
    int scalar_ny = 0;
    int scalar_nz = 0;
    size_t scalar_mesh_count = 0;
    size_t scalar_field_count = 0;
    size_t scalar_matrix_count = 0;
    size_t scalar_pivot_count = 0;
    size_t scalar_block_count = 0;
    size_t scalar_polarizability_count = 0;
    size_t vector_field_count = 0;
    size_t vector_fft_count = 0;
    size_t vector_rhs_count = 0;
    size_t vector_matrix_count = 0;
    size_t vector_map_count = 0;
    size_t vector_pivot_count = 0;
    size_t vector_batch_count = 0;
    size_t vector_channel_count = 0;
    size_t vector_polarizability_count = 0;
    size_t vector_green_count = 0;
    size_t vector_chan_vectors_count = 0;
    int vector_nx = 0;
    int vector_ny = 0;
    int vector_nz = 0;
} ChanPreconditionerBufferSizes;

typedef struct ChanPreconditionerDeviceBuffers {
    struct {
        cufftHandle fft_plan = 0;
        CUCOMPLEX* channel = nullptr;
        CUCOMPLEX* vectors = nullptr;
        CUCOMPLEX* field = nullptr;
        CUCOMPLEX* matrices = nullptr;
        CUCOMPLEX* fft_planes = nullptr;
        CUCOMPLEX* rhs = nullptr;
        CUCOMPLEX* rhs_output = nullptr;
        CUCOMPLEX* polarizabilities = nullptr;
        CUCOMPLEX* polarizability_sum = nullptr;

        CU_INT* nonzero_count = nullptr;
        CU_INT* pivots = nullptr;
        CU_INT* factor_info = nullptr;
        CUCOMPLEX** matrix_ptrs = nullptr;
        CUCOMPLEX** rhs_ptrs = nullptr;
    } scalar;

    struct {
        cufftHandle fft_plan = 0;
        cufftHandle green_fft_plan = 0;
        CUCOMPLEX* field = nullptr;
        CUCOMPLEX* fft_planes = nullptr;
        CUCOMPLEX* rhs = nullptr;
        CUCOMPLEX* rhs_output = nullptr;
        CUCOMPLEX* matrices = nullptr;
        CUCOMPLEX* channel_vectors = nullptr;
        CUCOMPLEX* green_components = nullptr;
        CUCOMPLEX* fft_components = nullptr;
        CUCOMPLEX* chan_vectors = nullptr;
        CUCOMPLEX* polarizabilities = nullptr;
        CUCOMPLEX* polarizability_sum = nullptr;
        CUCOMPLEX* mean_polarizability = nullptr;
        CU_INT* identity_map = nullptr;
        CU_INT* nonzero_count = nullptr;
        CU_INT* pivots = nullptr;
        CU_INT* factor_info = nullptr;
        CUCOMPLEX** matrix_ptrs = nullptr;
        CUCOMPLEX** rhs_ptrs = nullptr;
    } vector;
} ChanPreconditionerDeviceBuffers;

bool get_device_chan_preconditioner_buffers(const ChanPreconditionerBufferSizes& sizes,
                                            ChanPreconditionerDeviceBuffers* buffers);

// Persistent scratch for the scalar iterative CUDA driver. Zero-sized requests
// leave their pointers null, so guess or FFT buffers are acquired only when
// that operation needs them.
typedef struct ScalarDriverDeviceBuffers {
    CUCOMPLEX* incident = nullptr;
    CUCOMPLEX* polarizability = nullptr;
    CUCOMPLEX* fft_work = nullptr;
    CUCOMPLEX* scalar_solution = nullptr;
    CUCOMPLEX* local_solution = nullptr;
    CUCOMPLEX* vector_field = nullptr;
    CUCOMPLEX* vector_local_field = nullptr;
} ScalarDriverDeviceBuffers;

bool get_device_scalar_driver_buffers(size_t scalar_count,
                                      size_t fft_count,
                                      size_t vector_capacity_count,
                                      ScalarDriverDeviceBuffers* buffers);

typedef struct InverseDriverGammaBuffers {
    CUCOMPLEX* input = nullptr;
    CUCOMPLEX* output = nullptr;
    CUCOMPLEX* incident = nullptr;
    CUCOMPLEX* solution = nullptr;
} InverseDriverGammaBuffers;

bool get_device_inverse_driver_gamma_buffers(size_t count, InverseDriverGammaBuffers* buffers);

typedef struct InverseDriverBornBuffers {
    CUCOMPLEX* input = nullptr;
    CUCOMPLEX* output = nullptr;
} InverseDriverBornBuffers;

bool get_device_inverse_driver_born_buffers(size_t count, InverseDriverBornBuffers* buffers);

typedef struct InverseDriverRytovBuffers {
    CU_INT* error = nullptr;
} InverseDriverRytovBuffers;

bool get_device_inverse_driver_rytov_buffers(InverseDriverRytovBuffers* buffers);

typedef struct RytovFieldDeviceBuffers {
    CUCOMPLEX* reference = nullptr;
    CUCOMPLEX* delta = nullptr;
    CUCOMPLEX* result = nullptr;
    CU_INT* error = nullptr;
} RytovFieldDeviceBuffers;

bool get_device_rytov_field_buffers(size_t vector_count, RytovFieldDeviceBuffers* buffers);

// Persistent typed workspaces shared by the CUDA incident-field families.
// The request sizes are operation-specific; zero sizes do not allocate.
typedef struct IncidentFieldBufferRequest {
    size_t coordinate_count = 0;
    size_t field_count = 0;
    size_t parameter_count = 0;
    size_t real_parameter_count = 0;
    size_t antenna_coordinate_count = 0;
    size_t antenna_field_count = 0;
    size_t antenna_reference_count = 0;
} IncidentFieldBufferRequest;

typedef struct IncidentFieldDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CUCOMPLEX* field = nullptr;
    CUCOMPLEX* parameters = nullptr;
    CU_REAL* real_parameters = nullptr;
    CU_REAL* antenna_xs = nullptr;
    CU_REAL* antenna_ys = nullptr;
    CU_REAL* antenna_zs = nullptr;
    CUCOMPLEX* antenna_field = nullptr;
    CUCOMPLEX* antenna_reference = nullptr;
} IncidentFieldDeviceBuffers;

bool get_device_incident_field_buffers(const IncidentFieldBufferRequest& request,
                                       IncidentFieldDeviceBuffers* buffers);

typedef struct DipoleDerivativeDeviceBuffers {
    CUCOMPLEX* derivative = nullptr;
} DipoleDerivativeDeviceBuffers;

bool get_device_dipole_derivative_buffers(DipoleDerivativeDeviceBuffers* buffers);

typedef struct BeamPropagationBufferRequest {
    size_t coordinate_count = 0;
    size_t epsilon_count = 0;
    size_t incident_count = 0;
    size_t image_count = 0;
    size_t map_count = 0;
    size_t local_field_count = 0;
} BeamPropagationBufferRequest;

typedef struct BeamPropagationDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CUCOMPLEX* epsilon = nullptr;
    CUCOMPLEX* incident = nullptr;
    CUCOMPLEX* images = nullptr;
    CU_INT* map = nullptr;
    CUCOMPLEX* local_field = nullptr;
} BeamPropagationDeviceBuffers;

bool get_device_beam_propagation_buffers(const BeamPropagationBufferRequest& request,
                                         BeamPropagationDeviceBuffers* buffers);

// Workspace for the rigorous Gaussian correction used by the BPM first plane.
typedef struct BeamPropagationGaussianBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CUCOMPLEX* paraxial_field = nullptr;
    CUCOMPLEX* rigorous_field = nullptr;
    CU_REAL* max_partials = nullptr;
    CU_REAL* max_intensity = nullptr;
    CU_INT* convergence = nullptr;
    double* gkp_nodes = nullptr;
    double* gkp_weights = nullptr;
} BeamPropagationGaussianBuffers;

bool get_device_beam_propagation_gaussian_buffers(size_t plane_count,
                                                  BeamPropagationGaussianBuffers* buffers);

typedef struct SmallDenseSystemDeviceBuffers {
    CUCOMPLEX* matrix = nullptr;
    CUCOMPLEX* rhs = nullptr;
    CU_INT* info = nullptr;
} SmallDenseSystemDeviceBuffers;

bool get_device_small_dense_system_buffers(SmallDenseSystemDeviceBuffers* buffers);

typedef struct LocalMacroDeviceBuffers {
    CUCOMPLEX* epsilon = nullptr;
    CUCOMPLEX* input_field = nullptr;
    CUCOMPLEX* output_field = nullptr;
    CU_REAL* field_magnitude = nullptr;
    CU_INT* status = nullptr;
} LocalMacroDeviceBuffers;

bool get_device_local_macro_buffers(size_t tensor_count,
                                    size_t dipole_count,
                                    LocalMacroDeviceBuffers* buffers);

typedef struct GreenTensorDeviceBuffers {
    GreenTensorDevicePointers spatial{};
    GreenTensorDevicePointers spectra{};
} GreenTensorDeviceBuffers;

typedef struct MultilayerFftPlans {
    cufftHandle layer = 0;
    cufftHandle image = 0;
} MultilayerFftPlans;

bool get_device_multilayer_fft_plans(int layer_side, int image_side, MultilayerFftPlans* plans);

bool get_device_vector_green_buffers(size_t tensor_count,
                                     size_t fft_count,
                                     GreenTensorDeviceBuffers* buffers);
// Green generation scratch only: synchronizes and releases both work arrays
// immediately after the tensor FFT. The six output tensors stay resident.
bool release_device_vector_green_buffers();

typedef struct OpticalForceDeviceBuffers {
    CUCOMPLEX* polarization = nullptr;
    CUCOMPLEX* field_derivatives = nullptr;
    CUCOMPLEX* incident_grid = nullptr;
    CUCOMPLEX* incident_derivative = nullptr;
    CUCOMPLEX* local_field = nullptr;
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CU_REAL* force_x = nullptr;
    CU_REAL* force_y = nullptr;
    CU_REAL* force_z = nullptr;
    CU_REAL* reduction_ones = nullptr;
    CU_REAL* reduction_results = nullptr;
    CU_REAL* torque_x = nullptr;
    CU_REAL* torque_y = nullptr;
    CU_REAL* torque_z = nullptr;

    struct {
        CUCOMPLEX* vectors = nullptr;
        CUCOMPLEX* green_derivatives = nullptr;
        CUCOMPLEX* output = nullptr;
    } fft;

    struct {
        CU_INT* object_ids = nullptr;
        CU_REAL* force_sums = nullptr;
        CU_REAL* torque_sums = nullptr;
    } multi_object;
} OpticalForceDeviceBuffers;

typedef struct OpticalForceBufferRequest {
    size_t dipole_count = 0;
    size_t mesh_size = 0;
    size_t padded_count = 0;
    size_t object_count = 0;
    bool incident_derivative = false;
    bool force_density = false;
    bool torque_density = false;
    bool reduction = false;
    bool force_fft = false;
    bool group_force = false;
    bool group_torque = false;
} OpticalForceBufferRequest;

// Persistent buffers used by the post-solve scattering/Poynting calculation.
// Only the requested output sizes are allocated; this bundle is independent
// of force/torque buffers and is reused across repeated calculations.
typedef struct ScatteringResultDeviceBuffers {
    CUCOMPLEX* dipole_field = nullptr;
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CU_REAL* angular_poynting = nullptr;
    CU_REAL* theta_values = nullptr;
    CU_REAL* phi_values = nullptr;
    CU_REAL* positive_spectrum = nullptr;
    CU_REAL* negative_spectrum = nullptr;
    CU_REAL* kx_values = nullptr;
    CU_REAL* weighted_scattering = nullptr;
    CU_REAL* weighted_asymmetry = nullptr;
    CU_REAL* reduction_ones = nullptr;
    CUCOMPLEX* quick_fft_images = nullptr;
    CUCOMPLEX* quick_positive_field = nullptr;
    CUCOMPLEX* quick_negative_field = nullptr;
    CU_INT* quick_fft_map = nullptr;
} ScatteringResultDeviceBuffers;

// Persistent storage for the six complex image planes transformed by
// fouriertoimage. The planes are packed consecutively for batched cuFFT.
typedef struct FourierImageDeviceBuffers {
    CUCOMPLEX* fields = nullptr;
} FourierImageDeviceBuffers;

bool get_device_fourier_image_buffers(size_t plane_count, FourierImageDeviceBuffers* buffers);

bool get_device_scattering_result_buffers(size_t dipole_count,
                                          size_t angular_count,
                                          size_t spectrum_count,
                                          size_t fft_plane_count,
                                          size_t fft_map_count,
                                          ScatteringResultDeviceBuffers* buffers);

typedef struct QuickDiffractionFftDeviceBuffers {
    CUCOMPLEX* dipole_field = nullptr;
    CUCOMPLEX* fft_images = nullptr;
    CU_INT* fft_map = nullptr;
    cufftHandle fft_plan = 0;
} QuickDiffractionFftDeviceBuffers;

bool get_device_quick_diffraction_fft_buffers(size_t dipole_count,
                                              int nfft,
                                              QuickDiffractionFftDeviceBuffers* buffers);

// Persistent workspace for the plane-wave energy-flux calculation. Every
// requested buffer is operation-specific and reused across calls.
typedef struct EnergyConservationDeviceBuffers {
    CUCOMPLEX* dipole_field = nullptr;
    CU_REAL* z_planes = nullptr;
    CU_INT* fft_map = nullptr;
    CUCOMPLEX* fft_images = nullptr;
    CUCOMPLEX* incident_spectrum = nullptr;
    CUCOMPLEX* positive_field = nullptr;
    CUCOMPLEX* negative_field = nullptr;
    CU_REAL* flux_incident = nullptr;
    CU_REAL* flux_reflected = nullptr;
    CU_REAL* flux_transmitted = nullptr;
    CU_REAL* energy_inside_na = nullptr;
    CU_REAL* energy_total = nullptr;
    CU_REAL* reduction_ones = nullptr;
} EnergyConservationDeviceBuffers;

bool get_device_energy_conservation_buffers(size_t dipole_count,
                                            size_t fft_plane_count,
                                            size_t fft_map_count,
                                            EnergyConservationDeviceBuffers* buffers);

bool get_device_optical_force_buffers(const OpticalForceBufferRequest& request,
                                      OpticalForceDeviceBuffers* buffers);

typedef struct PmlDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CU_INT* inside_map = nullptr;
    CU_REAL* material = nullptr;
    CU_REAL* mask = nullptr;
    CU_REAL* z_candidates = nullptr;
    CU_REAL* zinf = nullptr;
    CUCOMPLEX* field = nullptr;
    CUCOMPLEX* epsilon_output = nullptr;
    CUCOMPLEX* contrast_output = nullptr;
} PmlDeviceBuffers;

bool get_device_pml_buffers(size_t dipole_count,
                            PmlDeviceBuffers* buffers,
                            size_t material_count = 0);

typedef struct SpectralFieldDeviceBuffers {
    CUCOMPLEX* phase_positive = nullptr;
    CUCOMPLEX* phase_negative = nullptr;
    CUCOMPLEX* spectrum = nullptr;
    CUCOMPLEX* field = nullptr;
    CU_REAL* normalization = nullptr;
} SpectralFieldDeviceBuffers;

typedef struct SpecklePhaseDeviceBuffers {
    CU_REAL* uniforms_positive = nullptr;
    CU_REAL* uniforms_negative = nullptr;
    CUCOMPLEX* phase_positive = nullptr;
    CUCOMPLEX* phase_negative = nullptr;
} SpecklePhaseDeviceBuffers;

bool get_device_speckle_phase_buffers(size_t phase_count,
                                      bool bidirectional,
                                      SpecklePhaseDeviceBuffers* buffers);

bool get_device_spectral_field_buffers(size_t plane_size,
                                       size_t field_count,
                                       SpectralFieldDeviceBuffers* buffers);

typedef struct NegativeKzDiffractionDeviceBuffers {
    CUCOMPLEX* random_phases = nullptr;
    CUCOMPLEX* spectrum = nullptr;
} NegativeKzDiffractionDeviceBuffers;

bool get_device_negative_kz_diffraction_buffers(size_t plane_size,
                                                NegativeKzDiffractionDeviceBuffers* buffers);

typedef struct LensDiffractionDeviceBuffers {
    CUCOMPLEX* green_field = nullptr;
    CUCOMPLEX* layer_spectrum = nullptr;
    CUCOMPLEX* accumulated_spectrum = nullptr;
    CUCOMPLEX* incident_spectrum = nullptr;
    CU_REAL* dipole_x = nullptr;
    CU_REAL* dipole_y = nullptr;
    CU_REAL* dipole_z = nullptr;
    CU_INT* fft_indices = nullptr;
    cufftHandle fft_plan = 0;
} LensDiffractionDeviceBuffers;

bool get_device_lens_diffraction_buffers(size_t plane_size,
                                         size_t mesh_count,
                                         LensDiffractionDeviceBuffers* buffers);

typedef struct DiffractionEnergyDeviceBuffers {
    CUCOMPLEX* green_field = nullptr;
    CUCOMPLEX* incident_spectrum = nullptr;
    CUCOMPLEX* incident_positive_spectrum = nullptr;
    CUCOMPLEX* incident_negative_spectrum = nullptr;
    CUCOMPLEX* positive_spectrum = nullptr;
    CUCOMPLEX* negative_spectrum = nullptr;
    CUCOMPLEX* layer_spectrum = nullptr;
    CU_REAL* phase_positive = nullptr;
    CU_REAL* phase_negative = nullptr;
    CU_REAL* z_coordinates = nullptr;
    CU_REAL* reduction_partials = nullptr;
    CU_INT* fft_indices = nullptr;
    cufftHandle fft_plan = 0;
} DiffractionEnergyDeviceBuffers;

bool get_device_diffraction_energy_buffers(size_t plane_size,
                                           size_t mesh_count,
                                           DiffractionEnergyDeviceBuffers* buffers);

typedef struct DiffractionGaussianDeviceBuffers {
    CU_REAL* x_coordinates = nullptr;
    CU_REAL* y_coordinates = nullptr;
    CU_REAL* z_coordinates = nullptr;
    CUCOMPLEX* paraxial_field = nullptr;
    CUCOMPLEX* rigorous_field = nullptr;
    CU_REAL* maximum_intensity = nullptr;
    CU_REAL* maximum_partials = nullptr;
    CU_INT* convergence = nullptr;
    double* gkp_nodes = nullptr;
    double* gkp_weights = nullptr;
} DiffractionGaussianDeviceBuffers;

bool get_device_diffraction_gaussian_buffers(size_t plane_size,
                                             bool include_rigorous,
                                             DiffractionGaussianDeviceBuffers* buffers);

// Persistent workspaces for the incident spectrum assembled by
// diffractefft2dinc. Parameters are packed as five CUCOMPLEX values per
// plane wave; random phases are indexed on the nfft-by-nfft Fourier grid.
typedef struct DiffractionIncidentDeviceBuffers {
    CUCOMPLEX* spectrum = nullptr;
    CUCOMPLEX* plane_parameters = nullptr;
    CUCOMPLEX* random_phases = nullptr;
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CUCOMPLEX* paraxial_field = nullptr;
    CUCOMPLEX* rigorous_field = nullptr;
    CUCOMPLEX* fft_planes = nullptr;
    CUCOMPLEX* antenna_field = nullptr;
    CUCOMPLEX* antenna_reference = nullptr;
    CU_REAL* max_intensity = nullptr;
    CU_REAL* max_partials = nullptr;
    CU_INT* convergence = nullptr;
    double* gkp_nodes = nullptr;
    double* gkp_weights = nullptr;
    cufftHandle fft_plan = 0;
} DiffractionIncidentDeviceBuffers;

bool get_device_diffraction_incident_buffers(size_t plane_size,
                                             size_t plane_count,
                                             bool gaussian_path,
                                             bool antenna_path,
                                             DiffractionIncidentDeviceBuffers* buffers);

typedef struct RigorousGaussianDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    double* gkp_nodes = nullptr;
    double* gkp_weights = nullptr;
    CUCOMPLEX* field = nullptr;
    CU_INT* convergence = nullptr;
} RigorousGaussianDeviceBuffers;

bool get_device_rigorous_gaussian_buffers(size_t dipole_count,
                                          RigorousGaussianDeviceBuffers* buffers);
bool get_device_rigorous_gaussian_buffers(size_t dipole_count,
                                          bool need_coordinates,
                                          bool need_field,
                                          RigorousGaussianDeviceBuffers* buffers);

// Persistent storage for the rigorous Gaussian illumination propagated by
// the FFT method. The FFT planes and dipole result remain resident while the
// illumination is constructed; callers request only this operation's bundle.
typedef struct GaussianFftDeviceBuffers {
    CU_INT* plane_map = nullptr;
    CU_REAL* dipole_xs = nullptr;
    CU_REAL* dipole_ys = nullptr;
    CUCOMPLEX* image_planes = nullptr;
    CUCOMPLEX* dipole_field = nullptr;
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CUCOMPLEX* paraxial_field = nullptr;
    CUCOMPLEX* rigorous_field = nullptr;
    CU_REAL* max_intensity = nullptr;
    CU_REAL* max_partials = nullptr;
    CU_INT* convergence = nullptr;
    double* gkp_nodes = nullptr;
    double* gkp_weights = nullptr;
} GaussianFftDeviceBuffers;

bool get_device_gaussian_fft_buffers(size_t plane_size,
                                     size_t mesh_count,
                                     GaussianFftDeviceBuffers* buffers,
                                     bool need_dipole_coordinates = true);

// Persistent buffers for arbitrary-field interpolation. File decoding stays
// on the host; grid sampling and interpolation run on the GPU.
typedef struct ArbitraryIlluminationDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CUCOMPLEX* grid_x = nullptr;
    CUCOMPLEX* grid_y = nullptr;
    CUCOMPLEX* grid_z = nullptr;
    CUCOMPLEX* field = nullptr;
} ArbitraryIlluminationDeviceBuffers;

bool get_device_arbitrary_illumination_buffers(size_t grid_count,
                                               size_t dipole_count,
                                               ArbitraryIlluminationDeviceBuffers* buffers);
bool get_device_arbitrary_illumination_buffers(size_t grid_count,
                                               size_t dipole_count,
                                               bool need_coordinates,
                                               bool need_field,
                                               ArbitraryIlluminationDeviceBuffers* buffers);

typedef struct ArbitraryDerivativeDeviceBuffers {
    CU_REAL* xs = nullptr;
    CU_REAL* ys = nullptr;
    CU_REAL* zs = nullptr;
    CUCOMPLEX* field = nullptr;
    CUCOMPLEX* derivative_scratch = nullptr;
    CUCOMPLEX* derivative_grids = nullptr;
    CUCOMPLEX* result = nullptr;
} ArbitraryDerivativeDeviceBuffers;

bool get_device_arbitrary_derivative_buffers(size_t grid_count,
                                             size_t dipole_count,
                                             ArbitraryDerivativeDeviceBuffers* buffers);

// Reuses small GPU buffers for the GPBICGSTABL dense least-squares system;
// the host/device transfers are centralized and counted here.
cudaError_t solve_small_dense_system_host(CUCOMPLEX* matrix_host,
                                          CUCOMPLEX* rhs_host,
                                          int order,
                                          int* info_host);
} // namespace cuda_data

// Selects the CUDA device and prints its properties once per process.
extern "C" bool initialize_cuda_device();
// Reset the CUDA context and invalidate every cached device pointer/plan.
extern "C" bool ifdda_reset_cuda_device_state();

// Returns a process-owned GPU buffer with room for `count` complex elements.
// The same pointer is reused while its capacity is sufficient.
extern "C" CUCOMPLEX* get_device_vectors(size_t count);
// Dedicated, reusable contiguous workspace for the three component-wise
// produitfftmatvect CUDA paths. Returns 3 * vector_count elements.
extern "C" CUCOMPLEX* get_device_product_fft_vectors(size_t vector_count);
// Persistent auxiliary vector distinct from the FFT workspace.
extern "C" CUCOMPLEX* get_device_auxiliary_vector(size_t count);
// Separate reusable staging area for post-solve result reductions.
extern "C" CUCOMPLEX* get_device_result_vectors(size_t count);
// Computes extinction, absorption, and scattering from host ABI arrays. The
// implementation reuses registered device aliases when present, otherwise
// stages the COMPLEX arrays to the GPU and returns REAL scalars to the host.
extern "C" int ifdda_compute_cross_sections_cuda_(const COMPLEX* incident,
                                                  const COMPLEX* polarization,
                                                  const COMPLEX* local_field,
                                                  const int* dipole_count,
                                                  const REAL* k0,
                                                  const REAL* incident_intensity,
                                                  REAL* extinction,
                                                  REAL* absorption,
                                                  REAL* scattering);
// Direct far-field/Poynting calculation. Inputs and outputs are host arrays;
// the implementation stages inputs as needed and computes on the GPU.
extern "C" void ifdda_direct_scattering_cuda_(const COMPLEX* field,
                                              const REAL* xs,
                                              const REAL* ys,
                                              const REAL* zs,
                                              const int* dipole_count,
                                              const int* ntheta,
                                              const int* nphi,
                                              const int* nfft,
                                              const REAL* k0,
                                              const REAL* theta_degrees,
                                              const REAL* phi_degrees,
                                              const REAL* intensity,
                                              const REAL* const_poynting,
                                              REAL* angular_poynting,
                                              REAL* theta_values,
                                              REAL* phi_values,
                                              REAL* positive_spectrum,
                                              REAL* negative_spectrum,
                                              REAL* kx_values,
                                              REAL* asymmetry,
                                              REAL* scattering,
                                              int* status_out);
extern "C" bool register_device_result_vector_alias(const void* host_key,
                                                    const CUCOMPLEX* device_data,
                                                    size_t count);
extern "C" const CUCOMPLEX* get_device_result_vector_alias(const void* host_key, size_t count);
extern "C" void ifdda_quick_diffraction_cuda_(const int* nx,
                                              const int* ny,
                                              const int* nz,
                                              const int* nfft,
                                              int* fft_map,
                                              const REAL* k0,
                                              const REAL* xs,
                                              const REAL* ys,
                                              const REAL* zs,
                                              const REAL* spacing,
                                              const COMPLEX* field,
                                              const int* ntheta,
                                              const int* nphi,
                                              const REAL* intensity,
                                              const REAL* theta,
                                              const REAL* phi,
                                              const REAL* const_poynting,
                                              int* imax,
                                              REAL* deltakx,
                                              REAL* deltaky,
                                              REAL* positive_spectrum,
                                              REAL* negative_spectrum,
                                              REAL* kx_values,
                                              REAL* angular_poynting,
                                              REAL* theta_values,
                                              REAL* phi_values,
                                              REAL* asymmetry,
                                              REAL* scattering,
                                              int* status_out);
extern "C" void clear_device_result_vector_alias(const void* host_key);
// Marks a host ABI result as immediately consumed by a CUDA-only downstream
// stage. The legacy solver bridge transfers this request to its typed staging
// address; the solver then omits the otherwise redundant D2H publication.
extern "C" bool request_device_only_result_vector(const void* host_key);
extern "C" bool take_device_only_result_vector_request(const void* host_key);
extern "C" void clear_device_only_result_vector_request(const void* host_key);

// Six separately allocated GPU arrays, each of tensor_count complex values.
// Their addresses and capacities cannot change until CUDA reset.
namespace cuda_data {
GreenTensorDevicePointers get_green(size_t tensor_count);
// Called only after successful GPU generation of all six Green components.
// Legacy product/solver interfaces may present different host mirrors.
void mark_green_generated();
// Six more independent allocations for widefield's possibly different grid.
GreenTensorDevicePointers get_widefield_green(size_t tensor_count);
}
// An existing generated tensor is reused directly. CPU upload is needed only
// for a first call without a GPU producer; explicit refresh remains available.
GreenTensorDevicePointers get_device_green(const COMPLEX* const cpu_tensors[6], size_t tensor_count);
// Non-allocating, non-uploading resolution for product-FFT legacy ABI adapters.
// Only inverse drivers / Green producers may acquire or upload tensor storage.
GreenTensorDevicePointers find_ready_device_green(
    const COMPLEX* const cpu_tensors[6], size_t tensor_count);
extern "C" int ifdda_refresh_green_tensors_cuda(const COMPLEX* tensor_xx,
                                                const COMPLEX* tensor_xy,
                                                const COMPLEX* tensor_xz,
                                                const COMPLEX* tensor_yy,
                                                const COMPLEX* tensor_yz,
                                                const COMPLEX* tensor_zz,
                                                size_t tensor_count);
// Reserves six separate persistent arrays for a GPU-side producer.
GreenTensorDevicePointers get_device_green_output(const COMPLEX* const cpu_tensor_keys[6],
                                              size_t tensor_count);
// Reserves the same six persistent outputs while accepting host keys whose
// element type is std::complex rather than CUCOMPLEX. Keys are identity-only.
GreenTensorDevicePointers get_device_green_output_for_host_keys(const void* const host_tensor_keys[6],
                                                            size_t tensor_count);
// Consumer-only lookup: validates all six host keys and component strides,
// without allocating or copying any Green tensor values.
GreenTensorDevicePointers find_device_green_for_host_keys(
    const void* const host_tensor_keys[6], size_t tensor_count);
extern "C" CUCOMPLEX* get_device_multilayer_green_output(const COMPLEX* const cpu_tensor_keys[6],
                                                         size_t tensor_count);
// Scalar Green tensor: upload once for an unchanged host tensor and reuse it.
extern "C" CUCOMPLEX* get_device_scalar_green(const COMPLEX* cpu_tensor, size_t tensor_count);
// Reserves the scalar Green buffer for a CUDA-side producer and associates it
// with the corresponding host array without uploading stale host contents.
extern "C" CUCOMPLEX* get_device_scalar_green_output(const COMPLEX* cpu_tensor_key,
                                                     size_t tensor_count);

// Uploads the CPU map into the single persistent GPU tabdip array.
// Repeated calls refresh the values because legacy callers can modify the host map.
extern "C" CU_INT* get_device_tabdip(const int* cpu_tabdip, size_t count);

// Reuses a process-owned device buffer and refreshes it from the host array.
extern "C" CUCOMPLEX* copy_polarizability_to_device(const COMPLEX* cpu_polarizability,
                                                    size_t count);

// Creates and keeps one cuFFT plan for the requested dimensions.
extern "C" cufftHandle get_fft_plan(int nx, int ny, int nz);
extern "C" cufftHandle get_speckle_fft_plan(int nfft);

// Keeps GPU buffers associated with CPU FFX/FFB addresses. Copies both input
// arrays CPU->GPU because FFB is read-modify-written by the product kernel.
extern "C" bool get_ffx_ffb(const COMPLEX* cpu_ffx,
                            COMPLEX* cpu_ffb,
                            size_t count,
                            CUCOMPLEX** gpu_ffx,
                            CUCOMPLEX** gpu_ffb);

// Copies the calculated FFB buffer back to its matching CPU array.
extern "C" bool copy_ffb_to_cpu(COMPLEX* cpu_ffb, const CUCOMPLEX* gpu_ffb, size_t count);

// Disabled legacy host-refresh path: unused by active CUDA/Fortran solvers.
// The resident variants below are the supported entry points.
/*
extern "C" bool get_gpbicg_device_arrays(COMPLEX* cpu_xi,
                                         const COMPLEX* cpu_xr,
                                         const COMPLEX* cpu_b,
                                         COMPLEX* cpu_work,
                                         size_t vector_count,
                                         size_t work_count,
                                         CUCOMPLEX** gpu_xi,
                                         CUCOMPLEX** gpu_xr,
                                         CUCOMPLEX** gpu_b,
                                         CUCOMPLEX** gpu_work);
*/
// Resident path used by the inverse driver. initial_guess_mode 0, 1, and 2
// reserve Xi/XR for device-side null, Born, and Rytov initialization;
// subsequent callbacks refresh XR only, as the product routine writes it.
extern "C" bool get_gpbicg_device_arrays_resident(COMPLEX* cpu_xi,
                                                  const COMPLEX* cpu_xr,
                                                  const COMPLEX* cpu_b,
                                                  COMPLEX* cpu_work,
                                                  size_t vector_count,
                                                  size_t work_count,
                                                  bool begin_solve,
                                                  CUCOMPLEX** gpu_xi,
                                                  CUCOMPLEX** gpu_xr,
                                                  CUCOMPLEX** gpu_b,
                                                  CUCOMPLEX** gpu_work);
// Direct driver path can initialize Xi on-device as zero or from the uploaded
// incident RHS, avoiding a redundant CPU initialization/upload.
extern "C" bool get_gpbicg_device_arrays_resident_initialized(COMPLEX* cpu_xi,
                                                              const COMPLEX* cpu_xr,
                                                              const COMPLEX* cpu_b,
                                                              COMPLEX* cpu_work,
                                                              size_t vector_count,
                                                              size_t work_count,
                                                              bool begin_solve,
                                                              int initial_guess_mode,
                                                              CUCOMPLEX** gpu_xi,
                                                              CUCOMPLEX** gpu_xr,
                                                              CUCOMPLEX** gpu_b,
                                                              CUCOMPLEX** gpu_work);
// Lets the CUDA Chan setup pass its already-computed right-hand side directly
// to the resident solver without downloading it to the Fortran buffer first.
extern "C" bool
register_gpbicg_device_b_alias(const void* cpu_b_key, CUCOMPLEX* gpu_b, size_t count);
extern "C" bool
get_gpbicg_device_b_alias(const void* cpu_b_key, size_t required_count, CUCOMPLEX** gpu_b);
extern "C" void release_gpbicg_device_b_alias(const void* cpu_b_key);
extern "C" CUCOMPLEX* get_gpbicg_device_xi_buffer(const COMPLEX* cpu_xi_key, size_t count);
extern "C" void mark_gpbicg_device_xi_current(const COMPLEX* cpu_xi_key, bool current);
extern "C" void set_gpbicg_device_resident(const COMPLEX* cpu_xi, bool enabled);
extern "C" void set_gpbicg_device_b_resident(const COMPLEX* cpu_b, bool enabled);
extern "C" void mark_gpbicg_device_xr_current(const COMPLEX* cpu_xr, bool current);
extern "C" bool
copy_device_complex_to_host(COMPLEX* cpu_destination, const CUCOMPLEX* gpu_source, size_t count);

extern "C" void pimzbicgstab_(COMPLEX* cpu_x,
                              COMPLEX* cpu_xi,
                              COMPLEX* cpu_xr,
                              const COMPLEX* cpu_b,
                              const int* lda,
                              const int* nlar,
                              const int* ndim,
                              int* nou,
                              COMPLEX* cpu_work,
                              COMPLEX* alpha,
                              COMPLEX* omega,
                              COMPLEX* rho,
                              REAL* norm,
                              const REAL* tolerance,
                              int* iteration,
                              const int* max_iteration,
                              int* status,
                              int* step_error);
extern "C" bool copy_gpbicg_results_to_cpu(COMPLEX* cpu_xi,
                                           COMPLEX* cpu_work,
                                           const CUCOMPLEX* gpu_xi,
                                           const CUCOMPLEX* gpu_work,
                                           size_t vector_count,
                                           size_t work_count,
                                           bool xi_modified,
                                           bool work_modified);

extern "C" bool get_gpbicor_device_arrays(COMPLEX* cpu_xi,
                                          const COMPLEX* cpu_xr,
                                          const COMPLEX* cpu_b,
                                          COMPLEX* cpu_ffloc,
                                          COMPLEX* cpu_work,
                                          size_t vector_count,
                                          size_t work_count,
                                          bool initialize_ffloc,
                                          CUCOMPLEX** gpu_xi,
                                          CUCOMPLEX** gpu_xr,
                                          CUCOMPLEX** gpu_b,
                                          CUCOMPLEX** gpu_ffloc,
                                          CUCOMPLEX** gpu_work);
extern "C" bool copy_gpbicor_results_to_cpu(COMPLEX* cpu_xi,
                                            COMPLEX* cpu_ffloc,
                                            const CUCOMPLEX* gpu_xi,
                                            const CUCOMPLEX* gpu_ffloc,
                                            size_t vector_count,
                                            bool copy_ffloc);
extern "C" CUCOMPLEX*
get_tfqmr_device_x(const COMPLEX* cpu_x, const COMPLEX* initial_x, size_t count, bool initialize);
extern "C" CUCOMPLEX* get_tfqmr_device_x_from_device(const COMPLEX* cpu_x,
                                                     const CUCOMPLEX* initial_device_x,
                                                     size_t count,
                                                     bool initialize);
extern "C" bool copy_tfqmr_x_to_cpu(COMPLEX* cpu_x, const CUCOMPLEX* gpu_x, size_t count);

#endif // PRODUITMATVECTFFT_CUDA_DATA_H
