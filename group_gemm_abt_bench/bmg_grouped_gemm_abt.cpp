/***************************************************************************************************
 * Copyright (C) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel BMG Grouped GEMM computing A @ B^T per group.

    This is a standalone twin of the upstream 04_bmg_grouped_gemm example, adapted
    to the exact contract the sgl-kernel-xpu LoRA grouped-GEMM kernels use:

      * B is ColumnMajor -- the weight is physically stored [N, K] row-major, so
        viewed as the CUTLASS B operand it is ColumnMajor (K, N) and the auto-
        selected 2D copy atom free-transposes it. The kernel therefore computes
        D = alpha * (A @ B^T) + beta * C for every group (where "B^T" is the stored
        [N, K] weight). This is the single "B alignment adjustment" versus the
        upstream A @ B example (which uses RowMajor B).
      * ElementOutput matches the input dtype (bf16/fp16), like the LoRA kernels,
        so the output-write memory traffic is faithful (matters for the K-thin,
        bandwidth-bound B-fwd / QKV-B-fwd shapes).
      * The tile config is selectable at runtime (--tile) so it can match each
        LoRA kernel exactly:
          large : TileShape 256 x 256 x 32, ThreadLayout 8 x 4 x 1  (A-fwd / B-fwd)
          tall  : TileShape  32 x 512 x 32, ThreadLayout 2 x 16 x 1 (QKV-B-fwd)

    Unlike the upstream example, per-group problem sizes are NOT uniform: they are
    read from a problem file (--problem_file, one "M N K" triple per line). This
    lets the Python benchmark drivers replay the exact per-segment (and, for QKV,
    per-projection) shapes the LoRA kernels see, so the measured pure group-GEMM
    time is directly comparable to the full LoRA kernel time (which additionally
    pays for the on-device metadata build in grouped_gemm_meta.hpp).

    All grouped-GEMM metadata (per-group problem sizes, strides, base pointers) is
    computed once during setup and the timed loop calls only gemm_op.run(), so the
    reported number is the pure group-GEMM API floor with zero per-launch metadata
    overhead.

    Build (see build.sh / CMakeLists.txt in this folder), then e.g.:
      $ ./bmg_grouped_gemm_abt --tile=large --dtype=bf16 --problem_file=probs.txt
      $ ./bmg_grouped_gemm_abt --m=4096 --n=4096 --k=64 --groups=8 --tile=tall
*/

#include <cfloat>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "helper.h"
#include "sycl_common.hpp"

// The exact per-group-alpha grouped epilogue the sgl-kernel-xpu LoRA kernels use
// (collective/xe_lora_epilogue.hpp): a drop-in subclass of the stock
// CollectiveEpilogue<IntelXeGenericGroup, ...> that pre-offsets the alpha/beta
// pointer arrays per group. Using it here (instead of the stock epilogue) makes
// the standalone benchmark run the identical kernel the LoRA launchers build, so
// the measured group-GEMM time is a faithful floor for the LoRA comparison.
#include "sycl/kernels/lora/collective/xe_lora_epilogue.hpp"

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

using ElementAccumulator = float;      // <- data type of accumulator
using ElementComputeEpilogue = float;  // <- data type of epilogue operations

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {
  bool error = false;
  bool help = false;

  float alpha, beta;
  int m, n, k, groups, iterations, verify;
  int per_group_alpha;      // 1 => distinct alpha per group via the alpha_ptr_array path
  std::string tile;         // "large" | "tall"
  std::string dtype;        // "bf16"  | "fp16"
  std::string problem_file; // one "M N K" per line; overrides m/n/k/groups when set

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  Options()
      : error(false),
        help(false),
        alpha(1.f),
        beta(0.f),
        m(4096),
        n(4096),
        k(64),
        groups(8),
        iterations(100),
        verify(0),
        per_group_alpha(0),
        tile("large"),
        dtype("bf16"),
        problem_file("") {}

  // Load per-group problem sizes from a text file: one "M N K" triple per line,
  // blank lines and lines beginning with '#' are ignored.
  bool load_problem_file() {
    std::ifstream in(problem_file);
    if (!in) {
      std::cerr << "Could not open problem_file: " << problem_file << std::endl;
      return false;
    }
    problem_sizes_host.clear();
    std::string line;
    while (std::getline(in, line)) {
      // Trim leading whitespace to detect comments/blank lines.
      size_t first = line.find_first_not_of(" \t\r\n");
      if (first == std::string::npos || line[first] == '#') {
        continue;
      }
      std::istringstream ss(line);
      int mm = 0, nn = 0, kk = 0;
      if (!(ss >> mm >> nn >> kk)) {
        std::cerr << "Malformed problem line: '" << line << "'" << std::endl;
        return false;
      }
      problem_sizes_host.push_back({mm, nn, kk});
    }
    groups = static_cast<int>(problem_sizes_host.size());
    if (groups == 0) {
      std::cerr << "problem_file contained no problems." << std::endl;
      return false;
    }
    return true;
  }

  // Parses the command line
  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 64);
    cmd.get_cmd_line_argument("groups", groups, 8);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 0);
    cmd.get_cmd_line_argument("per_group_alpha", per_group_alpha, 0);
    cmd.get_cmd_line_argument("tile", tile, std::string("large"));
    cmd.get_cmd_line_argument("dtype", dtype, std::string("bf16"));
    cmd.get_cmd_line_argument("problem_file", problem_file, std::string(""));

    if (!problem_file.empty()) {
      if (!load_problem_file()) {
        error = true;
      }
      return;
    }

    assert(groups > 0);
    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for (int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }
  }

  /// Prints the usage statement.
  std::ostream& print_usage(std::ostream& out) const {
    out << "BMG Grouped GEMM (A @ B^T per group)\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --m=<int>                   M extent for all groups (uniform mode)\n"
        << "  --n=<int>                   N extent for all groups (uniform mode)\n"
        << "  --k=<int>                   K extent for all groups (uniform mode)\n"
        << "  --groups=<int>              Number of GEMM problems (uniform mode)\n"
        << "  --problem_file=<path>       Text file with one 'M N K' triple per line;\n"
        << "                              overrides --m/--n/--k/--groups (variable shapes)\n"
        << "  --tile=<large|tall>         Tile config: large=256x256x32 (A-/B-fwd),\n"
        << "                              tall=32x512x32 (QKV-B-fwd)\n"
        << "  --dtype=<bf16|fp16>         Input/output element type\n"
        << "  --alpha=<f32>               Epilogue scalar alpha (broadcast to all groups)\n"
        << "  --beta=<f32>                Epilogue scalar beta; 1 => fuse residual C read\n"
        << "  --per_group_alpha=<int>     1 => distinct alpha per group via the\n"
        << "                              alpha_ptr_array path (exercises the LoRA\n"
        << "                              GroupedEpiloguePerGroupScalar per-group fix)\n"
        << "  --iterations=<int>          Number of profiling iterations to perform\n"
        << "  --verify=<int>              1 => verify against a reference GEMM per group\n\n";
    out << "\nExamples:\n\n"
        << "$ bmg_grouped_gemm_abt --tile=large --dtype=bf16 --problem_file=probs.txt\n"
        << "$ bmg_grouped_gemm_abt --m=4096 --n=4096 --k=64 --groups=8 --tile=tall\n\n";
    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {
    uint64_t fmas = uint64_t();
    for (auto const& problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) * static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Gemm>
struct ExampleRunner {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementAccumulator = float;  // reference GEMM / epilogue arithmetic in fp32

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  // Host-side allocations
  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;

  // Device-side allocations
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  cutlass::DeviceAllocation<const ElementA*> ptr_A;
  cutlass::DeviceAllocation<const ElementB*> ptr_B;
  cutlass::DeviceAllocation<const ElementC*> ptr_C;
  cutlass::DeviceAllocation<ElementOutput*> ptr_D;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;

  // Per-group alpha (only populated when options.per_group_alpha): a contiguous
  // fp32 value buffer (one alpha per group) plus an array-of-pointers (one fp32*
  // per group into that buffer). This is exactly the shape the LoRA launcher
  // builds (meta.alpha + make_device_ptrs) and is what exercises the
  // GroupedEpiloguePerGroupScalar per-group fix.
  std::vector<float> alpha_host;
  cutlass::DeviceAllocation<float> block_alpha;
  cutlass::DeviceAllocation<const float*> ptr_alpha;

  uint64_t seed = 0;

  //
  // Methods
  //

  bool verify(const Options& options) {
    bool passed = true;
    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
      // B is ColumnMajor with logical shape (K, N): its packed layout has leading
      // dim K, so the block physically holds the [N, K] weight -- ref computes
      // A @ B where B(k,n) = weight(n,k), i.e. A @ weight^T.
      cutlass::TensorRef ref_A(block_A.get() + offset_A.at(i), LayoutA::packed({M, K}));
      cutlass::TensorRef ref_B(block_B.get() + offset_B.at(i), LayoutB::packed({K, N}));
      cutlass::TensorRef ref_C(block_C.get() + offset_C.at(i), LayoutC::packed({M, N}));
      cutlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i), LayoutD::packed({M, N}));

      // Each group's own alpha when per-group alpha is enabled -- so a regression
      // to the alpha[0]-for-all-groups bug would make the reference disagree.
      float alpha_i = options.per_group_alpha ? alpha_host.at(i) : options.alpha;

      cutlass::reference::device::GemmComplex(
          {M, N, K},
          ElementAccumulator(alpha_i),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementAccumulator(options.beta),
          ref_C,
          ref_D,
          ElementAccumulator(0),
          1,      // batch_count
          M * K,  // batch_stride_A
          K * N,  // batch_stride_B
          M * N,  // batch_stride_C
          M * N   // batch_stride_D
      );

      compat::wait();

      passed &= cutlass::reference::device::BlockCompareRelativelyEqual(
          block_ref_D.get() + offset_D.at(i),
          block_D.get() + offset_D.at(i),
          M * N,
          ElementOutput(0.05),
          ElementOutput(0.05));
      if (!passed) {
        break;
      }
    }
    return passed;
  }

  /// Allocates device-side data
  void allocate(const Options& options) {
    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;
    int64_t total_elements_D = 0;

    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);

      offset_A.push_back(total_elements_A);
      offset_B.push_back(total_elements_B);
      offset_C.push_back(total_elements_C);
      offset_D.push_back(total_elements_D);

      int64_t elements_A = M * K;
      int64_t elements_B = K * N;
      int64_t elements_C = M * N;
      int64_t elements_D = M * N;

      total_elements_A += elements_A;
      total_elements_B += elements_B;
      total_elements_C += elements_C;
      total_elements_D += elements_D;

      // A: RowMajor (M, K).  B: ColumnMajor -- its CUTLASS stride is
      // Stride<int64_t, _1, int64_t> (leading dim in slot 0) because B's canonical
      // shape is (N, K, L); make_cute_packed_stride sets slot 0 from shape[1], so
      // the shape must be {N, K, 1} to get leading dim = K (the [N, K] weight,
      // giving A @ B^T = A @ weight^T). C/D: RowMajor (M, N).
      stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
      stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
      stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
      stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
    }

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);
    block_ref_D.reset(total_elements_D);
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const Options& options) {
    uint64_t seed = 2020;

    problem_sizes.reset(options.groups);
    problem_sizes.copy_from_host(options.problem_sizes_host.data());

    std::vector<ElementA*> ptr_A_host(options.groups);
    std::vector<ElementB*> ptr_B_host(options.groups);
    std::vector<ElementC*> ptr_C_host(options.groups);
    std::vector<ElementOutput*> ptr_D_host(options.groups);

    for (int32_t i = 0; i < options.groups; ++i) {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    }

    ptr_A.reset(options.groups);
    ptr_A.copy_from_host(ptr_A_host.data());
    ptr_B.reset(options.groups);
    ptr_B.copy_from_host(ptr_B_host.data());
    ptr_C.reset(options.groups);
    ptr_C.copy_from_host(ptr_C_host.data());
    ptr_D.reset(options.groups);
    ptr_D.copy_from_host(ptr_D_host.data());

    stride_A.reset(options.groups);
    stride_A.copy_from_host(stride_A_host.data());
    stride_B.reset(options.groups);
    stride_B.copy_from_host(stride_B_host.data());
    stride_C.reset(options.groups);
    stride_C.copy_from_host(stride_C_host.data());
    stride_D.reset(options.groups);
    stride_D.copy_from_host(stride_D_host.data());

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);

    // Per-group alpha buffer + pointer array (mirrors meta.alpha +
    // make_device_ptrs in the LoRA launcher). Distinct values per group so that
    // the (buggy) "alpha_ptr_array[0] for every group" behaviour would fail
    // verification -- the GroupedEpiloguePerGroupScalar fix must be in effect.
    if (options.per_group_alpha) {
      alpha_host.resize(options.groups);
      for (int32_t i = 0; i < options.groups; ++i) {
        alpha_host[i] = options.alpha * (0.5f + 0.25f * static_cast<float>(i % 5));
      }
      block_alpha.reset(options.groups);
      block_alpha.copy_from_host(alpha_host.data());

      std::vector<const float*> ptr_alpha_host(options.groups);
      for (int32_t i = 0; i < options.groups; ++i) {
        ptr_alpha_host.at(i) = block_alpha.get() + i;
      }
      ptr_alpha.reset(options.groups);
      ptr_alpha.copy_from_host(ptr_alpha_host.data());
    }
  }

  /// Populates a Gemm::Arguments structure from the given commandline options
  typename Gemm::Arguments args_from_options(
      const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;

    // beta stays a broadcast scalar for both LoRA forward kernels.
    fusion_args.beta = options.beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // The L (group) stride is a runtime int64 in Stride<_0,_0,int64_t>, so the
    // literal must be int64_t -- an int literal deduces the wrong tuple type.
    fusion_args.dBeta = cute::make_stride(cute::_0{}, cute::_0{}, int64_t{0});

    if (options.per_group_alpha) {
      // Per-group alpha via the array-of-pointers path: the fusion reads
      // *(alpha_ptr_array[l_coord]). The array kernel hard-codes the tile L coord
      // to 0, so the stock epilogue would read alpha_ptr_array[0] for EVERY group;
      // GroupedEpiloguePerGroupScalar advances the array by the group index in
      // to_base_arguments() so l_coord==0 resolves to *this* group's alpha. The
      // dAlpha L-stride is 1 (one fp32* per group).
      fusion_args.alpha = 0.f;
      fusion_args.alpha_ptr_array = reinterpret_cast<float const* const*>(ptr_alpha.get());
      fusion_args.dAlpha = cute::make_stride(cute::_0{}, cute::_0{}, int64_t{1});
    } else {
      // Single alpha broadcast to all groups (dAlpha L-stride 0).
      fusion_args.alpha = options.alpha;
      fusion_args.alpha_ptr_array = nullptr;
      fusion_args.dAlpha = cute::make_stride(cute::_0{}, cute::_0{}, int64_t{0});
    }

    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        // Host problem-sizes pointer is intentionally nullptr to match the LoRA
        // kernel's args_from_options (group_gemm_types.hpp) exactly: it passes
        // only the device problem_sizes and leaves the host copy null, so the Xe
        // group scheduler derives the tile count on device. Passing the host copy
        // here instead changes scheduler setup / occupancy and makes this
        // standalone a non-faithful proxy for the in-kernel group GEMM.
        {options.groups, problem_sizes.get(), nullptr},
        {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
        {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
        hw_info,
        {1, RasterOrderOptions::AlongN}};

    return arguments;
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    allocate(options);
    initialize(options);

    Gemm gemm_op;

    auto arguments = args_from_options(options, hw_info);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM once (warm-up + optional verification).
    CUTLASS_CHECK(gemm_op.run());
    compat::wait();

    if (options.verify != 0) {
      bool passed = verify(options);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
      if (!passed) {
        return cutlass::Status::kErrorInternal;
      }
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int iter = 0; iter < options.iterations; ++iter) {
        CUTLASS_CHECK(gemm_op.run());
      }
      compat::wait();

      float cute_time = timer.seconds() * 1000;
      double cute_average_time = double(cute_time) / double(options.iterations);
      double gflops = options.gflops(cute_average_time / 1000.0);

      std::cout << "  Groups      : " << options.groups << std::endl;
      std::cout << "  Tile        : " << options.tile << std::endl;
      std::cout << "  Dtype       : " << options.dtype << std::endl;
      std::cout << "  Beta        : " << options.beta << std::endl;
      std::cout << "  Avg runtime : " << cute_average_time << " ms" << std::endl;
      std::cout << "  GFLOPS      : " << gflops << std::endl;
      // Machine-parseable line for the Python benchmark drivers.
      std::cout << "RESULT avg_runtime_ms=" << cute_average_time << " gflops=" << gflops
                << " groups=" << options.groups << " tile=" << options.tile << " dtype=" << options.dtype
                << " beta=" << options.beta << std::endl;
    }

    return cutlass::Status::kSuccess;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Assemble the full CUTLASS grouped-GEMM type set for a given input element type
// and tile / thread layout, mirroring the sgl-kernel-xpu LoRA GroupGemmTypes<>
// bundle (ColumnMajor B => A @ B^T, XE_DPAS_TT<8,float,T> atom, 2 pipeline
// stages, per-group scalar epilogue reduced here to a scalar broadcast).
template <class ElementInputT, class TileShapeT, class ThreadLayoutT>
cutlass::Status run_typed(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
  using ElementA = ElementInputT;
  using ElementB = ElementInputT;
  using ElementOutput = ElementInputT;  // narrow output, like the LoRA kernels

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;  // <- the A @ B^T "B alignment"
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  using TileShape = TileShapeT;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>, Layout<TileShape>, ThreadLayoutT>::
          TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1StagedGroup<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

  // Epilogue op (D = alpha * acc + beta * C), template parameters ordered exactly
  // as the LoRA GroupGemmTypes<>: the fusion computes in the fp32 accumulator
  // type and the residual source C is the narrow output dtype (bf16/fp16).
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementAccumulator,     // ElementOutput_ of the fusion node (compute in fp32)
      ElementComputeEpilogue,
      ElementOutput,          // ElementSource_ (C) matches the narrow residual storage
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  // Per-group-alpha grouped epilogue (the LoRA "fix"): the dispatch policy
  // (IntelXeGenericGroup) is fixed inside the subclass, so its template list is
  // the stock CollectiveEpilogue list minus the leading policy parameter.
  using CollectiveEpilogue = cutlass::lora::kernel::GroupedEpiloguePerGroupScalar<
      TileShape,
      void,           // Epilogue tile (void = automatic)
      ElementOutput,  // ElementC -- residual C matches the narrow output dtype
      cutlass::gemm::TagToStrideC_t<LayoutC*>,
      ElementOutput,  // ElementD -- narrow store (bf16/fp16)
      cutlass::gemm::TagToStrideC_t<LayoutD*>,
      FusionCallBacks,
      void,   // copy atom to load C  (void = automatic)
      void>;  // copy atom to store D (void = automatic)

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA*>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB*>,
      TiledMma,
      GmemTiledCopyA,
      void,
      void,
      cute::identity,  // A
      GmemTiledCopyB,
      void,
      void,
      cute::identity>;  // B

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  ExampleRunner<Gemm> runner;
  return runner.run(options, hw_info);
}

// Tile configs matching the LoRA kernels.
//   large : 256 x 256 x 32, ThreadLayout 8 x 4 x 1  (sgemm_lora_a_fwd / _b_fwd)
//   tall  :  32 x 512 x 32, ThreadLayout 2 x 16 x 1 (qkv_lora_b_fwd)
using TileLarge = Shape<_256, _256, _32>;
using ThreadLarge = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
using TileTall = Shape<_32, _512, _32>;
using ThreadTall = Layout<Shape<_2, _16, _1>, Stride<_16, _1, _0>>;

template <class ElementInputT>
cutlass::Status dispatch_tile(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
  if (options.tile == "large") {
    return run_typed<ElementInputT, TileLarge, ThreadLarge>(options, hw_info);
  } else if (options.tile == "tall") {
    return run_typed<ElementInputT, TileTall, ThreadTall>(options, hw_info);
  }
  std::cerr << "Unknown --tile='" << options.tile << "' (expected 'large' or 'tall')." << std::endl;
  return cutlass::Status::kErrorInvalidProblem;
}

int main(int argc, const char** argv) {
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  cutlass::Status status = cutlass::Status::kSuccess;
  if (options.dtype == "bf16") {
    status = dispatch_tile<cutlass::bfloat16_t>(options, hw_info);
  } else if (options.dtype == "fp16") {
    status = dispatch_tile<cutlass::half_t>(options, hw_info);
  } else {
    std::cerr << "Unknown --dtype='" << options.dtype << "' (expected 'bf16' or 'fp16')." << std::endl;
    return -1;
  }

  CUTLASS_CHECK(status);
  return 0;
}
