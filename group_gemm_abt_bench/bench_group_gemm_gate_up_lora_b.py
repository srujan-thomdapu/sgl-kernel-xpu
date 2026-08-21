"""Group-GEMM-only benchmark mirroring bench_gate_up_lora_b_fwd.py.

Isolates the pure A @ B^T pointer-array grouped-GEMM cost from the metadata-build
overhead the gate_up_lora_b_fwd kernel pays on every launch. The fused gate/up LoRA-B
emits TWO grouped-GEMM groups per segment (the gate and up projections, both of width
output_dim), so per segment the problems are
    (seg_len, output_dim, max_rank), (seg_len, output_dim, max_rank)
-- 2 * num_segments groups total. Per-adapter scaling makes this a per-group alpha
kernel, and use_base_output cases add a residual (beta=1). Tile is `tall` (32x512x32)
to match gate_up_lora_b_fwd_types.hpp (identical config to qkv_lora_b_fwd).

Run:  source /home/gta/intel/oneapi/setvars.sh && ./build.sh   # build the binary
      python bench_group_gemm_gate_up_lora_b.py
"""

import common

common.add_benchmark_to_path()
import bench_gate_up_lora_b_fwd as ref  # noqa: E402

TILE = "tall"


def _build_problems(inputs, case):
    band_dims = inputs["band_dims"]  # [output_dim, output_dim] (gate, up)
    k = case["max_rank"]             # K = max_rank (LoRA rank)
    problems = []
    for sl in inputs["seg_lens"].tolist():
        for n_p in band_dims:  # one group per gate/up band, matching device metadata
            problems.append((int(sl), int(n_p), k))
    return problems


def _flops(inputs, case):
    return ref._compute_flops_by_segment(
        inputs["seg_lens"],
        inputs["weight_indices"],
        inputs["lora_ranks"],
        inputs["band_dims"],
    )


def _bytes(inputs, case, elem_size):
    return ref._estimate_bytes(
        inputs["seg_lens"],
        inputs["weight_indices"],
        inputs["lora_ranks"],
        inputs["band_dims"],
        elem_size,
    )


def main():
    args = common.common_argparser(
        "Group-GEMM-only overhead analysis for gate_up_lora_b_fwd"
    ).parse_args()

    rows = common.drive(
        title="gate-up-b",
        module=ref,
        build_problems=_build_problems,
        flops_fn=_flops,
        bytes_fn=_bytes,
        tile=TILE,
        per_group_alpha=True,  # per-adapter scaling -> per-group alpha
        beta_of_case=lambda case: 1.0 if case.get("use_base_output", False) else 0.0,
        dtype_names=args.dtypes,
        iterations=args.iterations,
        seed=args.seed,
        case_ids=common.resolve_case_ids(args.cases, ref),
        verify=args.verify,
    )
    common.print_report("Group-GEMM vs gate_up_lora_b_fwd (metadata overhead)", rows)


if __name__ == "__main__":
    main()
