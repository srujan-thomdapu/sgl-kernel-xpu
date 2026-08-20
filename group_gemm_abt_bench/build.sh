#!/usr/bin/env bash
# Build the standalone A @ B^T grouped-GEMM example with the same toolchain and
# flags the cutlass-sycl examples use (extracted from build-examples/build.ninja).
#
# It is a single translation unit over header-only CUTLASS, so a direct icpx
# invocation is the most robust build. A CMakeLists.txt is also provided for an
# out-of-tree CMake build if preferred.
#
# Usage:
#   ./build.sh                 # source oneAPI setvars, then build
#   CUTLASS_DIR=/path ./build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Root of the fetched cutlass-sycl (sycl-tla) source tree. Override with CUTLASS_DIR.
CUTLASS_DIR="${CUTLASS_DIR:-${REPO_ROOT}/build/_deps/repo-cutlass-sycl-src}"
# Directory holding the generated version_extended.h (built alongside the examples).
CUTLASS_GEN_INCLUDE="${CUTLASS_GEN_INCLUDE:-${CUTLASS_DIR}/build-examples/include}"

# oneAPI environment (icpx + MKL). Source setvars unless icpx is already on PATH.
if ! command -v icpx >/dev/null 2>&1; then
  ONEAPI_SETVARS="${ONEAPI_SETVARS:-/home/gta/intel/oneapi/setvars.sh}"
  if [[ -f "${ONEAPI_SETVARS}" ]]; then
    # setvars.sh can return non-zero / call `set`; don't let it trip errexit.
    set +e
    # shellcheck disable=SC1090
    source "${ONEAPI_SETVARS}" >/dev/null 2>&1
    set -e
  fi
fi
command -v icpx >/dev/null 2>&1 || { echo "icpx not found; source your oneAPI setvars.sh first." >&2; exit 1; }

MKLROOT="${MKLROOT:?MKLROOT not set (source oneAPI setvars.sh)}"
CMPLR_ROOT="${CMPLR_ROOT:?CMPLR_ROOT not set (source oneAPI setvars.sh)}"

OUT="${SCRIPT_DIR}/bmg_grouped_gemm_abt"
SRC="${SCRIPT_DIR}/bmg_grouped_gemm_abt.cpp"

echo "CUTLASS_DIR=${CUTLASS_DIR}"
echo "Building ${OUT} ..."

icpx "${SRC}" -o "${OUT}" \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED -DMKL_ILP64 \
  -O3 -DNDEBUG -std=c++17 -fPIE -Wall \
  -Wno-unused-variable -Wno-unused-local-typedef -Wno-unused-but-set-variable \
  -Wno-uninitialized -Wno-reorder-ctor -Wno-logical-op-parentheses \
  -Wno-unused-function -Wno-unknown-pragmas \
  -fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen \
  -I"${REPO_ROOT}/src" \
  -I"${CUTLASS_DIR}/include" \
  -I"${CUTLASS_DIR}/examples/common" \
  -I"${CUTLASS_GEN_INCLUDE}" \
  -I"${CUTLASS_DIR}/tools/util/include" \
  -isystem "${MKLROOT}/include" \
  -Xsycl-target-backend=spir64_gen "-device bmg-g21,bmg-g31" \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -Xs "-options -cl-poison-unsupported-fp64-kernels \
       -options -cl-intel-enable-auto-large-GRF-mode \
       -options -cl-fp32-correctly-rounded-divide-sqrt \
       -options -cl-intel-greater-than-4GB-buffer-required \
       -options \"-igc_opts 'VectorAliasBBThreshold=10000'\"" \
  -Wl,-rpath="${MKLROOT}/lib" \
  "${MKLROOT}/lib/libmkl_intel_ilp64.so" \
  "${MKLROOT}/lib/libmkl_intel_thread.so" \
  "${MKLROOT}/lib/libmkl_core.so" \
  "${CMPLR_ROOT}/lib/libiomp5.so" \
  -lm -ldl -lpthread

echo "Built: ${OUT}"
