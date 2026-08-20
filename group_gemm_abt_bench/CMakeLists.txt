# Optional out-of-tree CMake build for the standalone A @ B^T grouped-GEMM
# example. The canonical build is ./build.sh (a single icpx invocation over
# header-only CUTLASS); this file mirrors it for those who prefer CMake.
#
# CRITICAL: the offline-compiler options below must match the sgl-kernel-xpu
# build (cmake/BuildFlags.cmake) -- in particular
# -cl-intel-enable-auto-large-GRF-mode. Without large-GRF mode this kernel
# spills ~170 registers and runs ~3x slower, which makes the standalone a
# non-faithful proxy for the in-kernel group GEMM and produces bogus (negative)
# metadata-overhead numbers in the Python drivers.
#
# Usage:
#   source /home/gta/intel/oneapi/setvars.sh
#   cmake -S . -B build -DCUTLASS_DIR=/path/to/cutlass-sycl
#   cmake --build build
cmake_minimum_required(VERSION 3.20)
project(bmg_grouped_gemm_abt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Repo layout: this dir is <repo>/group_gemm_abt_bench.
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

set(CUTLASS_DIR "${REPO_ROOT}/build/_deps/repo-cutlass-sycl-src"
    CACHE PATH "Root of the fetched cutlass-sycl (sycl-tla) source tree")
set(CUTLASS_GEN_INCLUDE "${CUTLASS_DIR}/build-examples/include"
    CACHE PATH "Directory holding the generated version_extended.h")

if(NOT DEFINED ENV{MKLROOT})
  message(FATAL_ERROR "MKLROOT not set -- source your oneAPI setvars.sh first.")
endif()
if(NOT DEFINED ENV{CMPLR_ROOT})
  message(FATAL_ERROR "CMPLR_ROOT not set -- source your oneAPI setvars.sh first.")
endif()
set(MKLROOT "$ENV{MKLROOT}")
set(CMPLR_ROOT "$ENV{CMPLR_ROOT}")

# Require the Intel LLVM compiler (icpx).
if(NOT CMAKE_CXX_COMPILER MATCHES "icpx|icx")
  message(WARNING "Set -DCMAKE_CXX_COMPILER=icpx to build this SYCL target.")
endif()

add_executable(bmg_grouped_gemm_abt bmg_grouped_gemm_abt.cpp)

target_compile_definitions(bmg_grouped_gemm_abt PRIVATE
  CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET CUTLASS_VERSIONS_GENERATED MKL_ILP64)

target_include_directories(bmg_grouped_gemm_abt PRIVATE
  "${REPO_ROOT}/src"
  "${CUTLASS_DIR}/include"
  "${CUTLASS_DIR}/examples/common"
  "${CUTLASS_GEN_INCLUDE}"
  "${CUTLASS_DIR}/tools/util/include")
target_include_directories(bmg_grouped_gemm_abt SYSTEM PRIVATE "${MKLROOT}/include")

set(_common_flags
  -O3 -DNDEBUG -fPIE -Wall
  -Wno-unused-variable -Wno-unused-local-typedef -Wno-unused-but-set-variable
  -Wno-uninitialized -Wno-reorder-ctor -Wno-logical-op-parentheses
  -Wno-unused-function -Wno-unknown-pragmas
  -fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen)

target_compile_options(bmg_grouped_gemm_abt PRIVATE ${_common_flags})

# Device backend + offline-compiler options. Keep in lockstep with build.sh and
# cmake/BuildFlags.cmake (SYCL_OFFLINE_COMPILER_CG_OPTIONS).
target_link_options(bmg_grouped_gemm_abt PRIVATE
  ${_common_flags}
  -Xsycl-target-backend=spir64_gen "-device bmg-g21,bmg-g31"
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
  "SHELL:-Xs \"-options -cl-poison-unsupported-fp64-kernels \
    -options -cl-intel-enable-auto-large-GRF-mode \
    -options -cl-fp32-correctly-rounded-divide-sqrt \
    -options -cl-intel-greater-than-4GB-buffer-required \
    -options \\\"-igc_opts 'VectorAliasBBThreshold=10000'\\\"\""
  -Wl,-rpath=${MKLROOT}/lib)

target_link_libraries(bmg_grouped_gemm_abt PRIVATE
  ${MKLROOT}/lib/libmkl_intel_ilp64.so
  ${MKLROOT}/lib/libmkl_intel_thread.so
  ${MKLROOT}/lib/libmkl_core.so
  ${CMPLR_ROOT}/lib/libiomp5.so
  m dl pthread)
