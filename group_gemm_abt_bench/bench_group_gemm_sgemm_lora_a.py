"""Group-GEMM-only benchmark mirroring bench_sgemm_lora_a_fwd.py.

Isolates the pure A @ B^T pointer-array grouped-GEMM cost from the metadata-build
overhead the sgemm_lora_a_fwd kernel pays on every launch. For each LoRA-A case
the per-group problem is (M=seg_len, N=stack_num*max_rank, K=input_dim); the
epilogue is pure projection (beta=0) with a single broadcast alpha=1 (LoRA-A has
no per-adapter scaling), and the tile is `large` (256x256x32) to match
sgemm_lora_a_fwd_types.hpp.

Run:  source /home/gta/intel/oneapi/setvars.sh && ./build.sh   # build the binary
      python bench_group_gemm_sgemm_lora_a.py
"""

import common

common.add_benchmark_to_path()
import bench_sgemm_lora_a_fwd as ref  # noqa: E402

TILE = "large"


def _build_problems(inputs, case):
    total_n = case["stack_num"] * case["max_rank"]  # N = stack_num * max_rank
    k = case["input_dim"]                            # K = input_dim (hidden size)
    return [(int(sl), total_n, k) for sl in inputs["seg_lens"].tolist()]


def _flops(inputs, case):
    return ref._compute_flops_by_segment(
        inputs["seg_lens"],
        inputs["weight_indices"],
        inputs["lora_ranks"],
        case["stack_num"],
        case["input_dim"],
    )


def _bytes(inputs, case, elem_size):
    return ref._estimate_bytes(
        inputs["seg_lens"],
        inputs["weight_indices"],
        inputs["lora_ranks"],
        case["stack_num"],
        case["input_dim"],
        elem_size,
    )


def main():
    args = common.common_argparser(
        "Group-GEMM-only overhead analysis for sgemm_lora_a_fwd"
    ).parse_args()

    rows = common.drive(
        title="lora-a",
        module=ref,
        build_problems=_build_problems,
        flops_fn=_flops,
        bytes_fn=_bytes,
        tile=TILE,
        per_group_alpha=False,  # LoRA-A: no per-adapter scaling (alpha=1 broadcast)
        beta_of_case=lambda case: 0.0,  # pure projection, no residual
        dtype_names=args.dtypes,
        iterations=args.iterations,
        seed=args.seed,
        case_ids=common.resolve_case_ids(args.cases, ref),
        verify=args.verify,
    )
    common.print_report("Group-GEMM vs sgemm_lora_a_fwd (metadata overhead)", rows)


if __name__ == "__main__":
    main()
