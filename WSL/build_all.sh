#!/usr/bin/env bash
# Configure et compile toutes les cibles IFDDA CUDA/WSL.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUDA_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${CUDA_ROOT}/src"
BUILD_DIR="${IFDDA_BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ "$(uname -s)" != Linux ]]; then
    echo "Erreur : executer ce script sous Ubuntu WSL (pas sous Windows)." >&2
    exit 1
fi

for required_file in \
    "${SCRIPT_DIR}/CMakeLists.txt" \
    "${SOURCE_DIR}/main0.cpp" \
    "${SOURCE_DIR}/cdmlib.cpp" \
    "${SOURCE_DIR}/kernel.h" \
    "${SOURCE_DIR}/cdmlib_cpp_abi.hpp" \
    "${SOURCE_DIR}/cdmlib_geometry_abi.hpp" \
    "${SOURCE_DIR}/cdmlib_cpp_fftw_precision.hpp" \
    "${SOURCE_DIR}/cdmlib_abi.h" \
    "${SOURCE_DIR}/cdmlib_config.h" \
    "${SOURCE_DIR}/cdmlib_fftphase.h" \
    "${SOURCE_DIR}/cdmlib_incidentphase.h" \
    "${SOURCE_DIR}/cdmlib_initphase.h" \
    "${SOURCE_DIR}/cdmlib_precision.h" \
    "${SOURCE_DIR}/hdf5_helpers.h" \
    "${SOURCE_DIR}/plan_int64_abi.h" \
    "${SOURCE_DIR}/force_green_derivative_kernels.cuh" \
    "${SOURCE_DIR}/gaussian_incident_kernel.cuh" \
    "${SOURCE_DIR}/gaussian_gkp_tables.h" \
    "${SOURCE_DIR}/kernel.cu"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Erreur : fichier requis introuvable : ${required_file}" >&2
        exit 1
    fi
done

if [[ "${BUILD_DIR}" != /* ]]; then
    BUILD_DIR="${SCRIPT_DIR}/${BUILD_DIR}"
fi
mkdir -p -- "${BUILD_DIR}"
BUILD_DIR="$(cd -- "${BUILD_DIR}" && pwd)"

# Remove obsolete executable names left by earlier WSL configurations.
for obsolete in \
    test_cuda_float test_cuda_double \
    test_hybrid_fortran_cdmlib_cuda_float test_hybrid_fortran_cdmlib_cuda_double \
    test_main0_f90_cdmlib_f90_cuda_float test_main0_f90_cdmlib_f90_cuda_double \
    test_main0_fortran_cdmlib_cpp_cuda_float test_main0_fortran_cdmlib_cpp_cuda_double; do
    rm -f -- "${BUILD_DIR}/${obsolete}"
done

BUILD_TYPE="${IFDDA_BUILD_TYPE:-Release}"
CUDA_ARCH="${IFDDA_CUDA_ARCH:-70}"
JOBS="${IFDDA_BUILD_JOBS:-4}"

# CUDA 13 removed offline compilation for Volta (sm_70) and earlier.
# Fail with a meaningful diagnostic rather than an obscure nvcc error.
NVCC_BIN="${CUDACXX:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}"
if [[ ! -x "${NVCC_BIN}" ]]; then
    NVCC_BIN="$(command -v nvcc || true)"
fi
if [[ -n "${NVCC_BIN}" ]]; then
    NVCC_MAJOR="$("${NVCC_BIN}" --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p' | tail -n 1)"
    if [[ "${NVCC_MAJOR}" =~ ^[0-9]+$ ]] && (( NVCC_MAJOR >= 13 )); then
        case ";${CUDA_ARCH};" in
            *';50;'*|*';52;'*|*';53;'*|*';60;'*|*';61;'*|*';62;'*|*';70;'*|*';72;'*)
                echo "[IFDDA] CUDA ${NVCC_MAJOR} ne peut pas compiler SM=${CUDA_ARCH}." >&2
                echo "[IFDDA] GPU Volta/SM70 : utiliser CUDA 12.9 ou plus ancien." >&2
                echo "[IFDDA] GPU recent : IFDDA_CUDA_ARCH=<sm reel> ./build_all.sh" >&2
                exit 2
                ;;
        esac
    fi
fi

echo "[IFDDA] Projet CMake : ${SCRIPT_DIR}"
echo "[IFDDA] Sources CUDA : ${SOURCE_DIR}"
echo "[IFDDA] Build         : ${BUILD_DIR}"
echo "[IFDDA] Mode=${BUILD_TYPE}, CUDA SM=${CUDA_ARCH}, jobs=${JOBS}"

# Les trois familles demandees : main0.cpp + cdmlib.cpp,
# main0.f90 + cdmlib.cpp, puis main0.f90 + cdmlib.f90, en deux precisions.
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DIFDDA_CUDA_CHAN_INITIAL_RESIDENT=ON

cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo "[IFDDA] Compilation terminee. Executables dans ${BUILD_DIR} :"
find "${BUILD_DIR}" -maxdepth 1 -type f -name 'test_*' -executable -printf '  %f\n' | sort
