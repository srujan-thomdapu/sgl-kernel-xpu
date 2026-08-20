"""Group-GEMM-only benchmark mirroring bench_sgemm_lora_b_fwd.py.

Isolates the pure A @ B^T pointer-array grouped-GEMM cost from the metadata-build
overhead the sgemm_lora_b_fwd kernel pays on every launch. For each LoRA-B case
the per-group problem is (M=seg_len, N=output_dim, K=max_rank); LoRA-B applies a
per-adapter scaling, so per-group alpha is exercised, and use_base_output cases
add a residual C source (beta=1). Tile is `large` (256x256x32) to match
sgemm_lora_b_fwd_types.hpp.

Run:  source /home/gta/intel/oneapi/setvars.sh && ./build.sh   # build the binary
      python bench_group_gemm_sgemm_lora_b.py
"""

import common

common.add_benchmark_to_path()
import bench_sgemm_lora_b_fwd as ref  # noqa: E402

TILE = "large"


def _build_problems(inputs, case):
    n = case["output_dim"]       # N = output_dim
    k = case["max_rank"]         # K = max_rank (LoRA rank)
    return [(int(sl), n, k) for sl in inputs["seg_lens"].tolist()]


def _flops(inputs, case):
    return ref._compute_flops_by_segment(
        inputs["seg_lens"],
        inputs["weight_indices"],
        inputs["lora_ranks"],
        case["output_dim"],
    )


def _bytes(inputs, case, elem_size):
    return ref._estimate_bytes(
        inputs["seg_lens"],
        inputs["weight_indices"],
        inputs["lora_ranks"],
        case["output_dim"],
        elem_size,
    )


def main():
    args = common.common_argparser(
        "Group-GEMM-only overhead analysis for sgemm_lora_b_fwd"
    ).parse_args()

    rows = common.drive(
        title="lora-b",
        module=ref,
        build_problems=_build_problems,
        flops_fn=_flops,
        bytes_fn=_bytes,
        tile=TILE,
        per_group_alpha=True,  # LoRA-B: per-adapter scaling (scalings) -> per-group alpha
        beta_of_case=lambda case: 1.0 if case.get("use_base_output", False) else 0.0,
        dtype_names=args.dtypes,
        iterations=args.iterations,
        seed=args.seed,
        case_ids=common.resolve_case_ids(args.cases, ref),
        verify=args.verify,
    )
    common.print_report("Group-GEMM vs sgemm_lora_b_fwd (metadata overhead)", rows)


if __name__ == "__main__":
    main()
