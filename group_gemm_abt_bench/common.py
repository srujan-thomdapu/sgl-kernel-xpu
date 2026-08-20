"""Shared helpers for the group-GEMM-only benchmarks.

These drivers isolate the *pure* CUTLASS pointer-array grouped-GEMM cost from the
per-launch overhead the LoRA kernels pay. The LoRA kernels build all grouped-GEMM
metadata on device every call -- per-group problem sizes, strides, byte offsets,
absolute pointer arrays and the per-group alpha buffer (build_grouped_gemm_meta /
make_device_ptrs in grouped_gemm_meta.hpp) -- and only then hand it to the group
GEMM. The standalone bmg_grouped_gemm_abt binary computes that same metadata ONCE
during setup (here it is even precomputed host-side) and times only
gemm_op.run(), so it measures the pure group-GEMM API floor.

For each LoRA benchmark case the matching driver:
  1. rebuilds the exact same inputs (same shapes / seed) as the LoRA benchmark,
  2. derives the per-group (M, N, K) list HOST-SIDE -- the identical numbers the
     device metadata kernel would emit -- and runs the standalone binary on them
     with the matching tile / dtype / beta / alpha mode, and
  3. times the real LoRA kernel (which rebuilds the metadata each call) on the
     same inputs.

overhead = lora_ms - gemm_ms is the metadata-build + argument-setup + output
allocation cost the LoRA wrapper adds on top of the raw group-GEMM API.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent / "benchmark"
DEFAULT_BIN = _HERE / "bmg_grouped_gemm_abt"

Problem = Tuple[int, int, int]  # (M, N, K)


def add_benchmark_to_path() -> None:
    """Make the LoRA benchmark modules (../benchmark) importable."""
    p = str(_BENCH_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def find_binary() -> Path:
    env = os.environ.get("GROUP_GEMM_ABT_BIN")
    binary = Path(env) if env else DEFAULT_BIN
    if not binary.exists():
        raise FileNotFoundError(
            f"group-GEMM binary not found at {binary}. Build it first with "
            f"`./build.sh` in {_HERE} (or set GROUP_GEMM_ABT_BIN)."
        )
    return binary


_RESULT_RE = re.compile(r"avg_runtime_ms=([0-9.eE+-]+)\s+gflops=([0-9.eE+-]+)")


def _parse_result(stdout: str) -> Dict[str, float]:
    match = None
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            match = _RESULT_RE.search(line)
    if match is None:
        raise RuntimeError(f"No RESULT line in binary output:\n{stdout}")
    return {"avg_runtime_ms": float(match.group(1)), "gflops": float(match.group(2))}


def run_group_gemm(
    problems: Sequence[Problem],
    *,
    tile: str,
    dtype: str,
    beta: float,
    per_group_alpha: bool,
    alpha: float = 1.0,
    iterations: int = 200,
    verify: bool = False,
) -> Dict[str, float]:
    """Run the standalone binary on an explicit per-group problem list.

    The per-group (M, N, K) triples are written to a problem file so the binary
    replays the exact segmented shapes the LoRA kernel sees; only gemm_op.run()
    is timed, so the returned avg_runtime_ms is the pure group-GEMM cost.
    """
    binary = find_binary()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("# M N K per group (host-computed grouped-GEMM metadata)\n")
        for (m, n, k) in problems:
            f.write(f"{int(m)} {int(n)} {int(k)}\n")
        path = f.name
    try:
        cmd = [
            str(binary),
            f"--problem_file={path}",
            f"--tile={tile}",
            f"--dtype={dtype}",
            f"--beta={beta}",
            f"--alpha={alpha}",
            f"--per_group_alpha={1 if per_group_alpha else 0}",
            f"--iterations={iterations}",
            f"--verify={1 if verify else 0}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"binary failed (rc={proc.returncode}):\n"
                f"cmd: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return _parse_result(proc.stdout)
    finally:
        os.unlink(path)


def bench_lora_ms(
    run_once: Callable[[], object], *, iterations: int, warmup: int = 10
) -> float:
    """Per-call wall time (ms) of the full LoRA kernel, timed the SAME way the
    standalone binary times gemm_op.run().

    The binary uses a host wall-clock (cutlass GPU_Clock == std::chrono): one
    warm-up, then `iterations` back-to-back submits followed by a single device
    wait, reporting wall_time / iterations. We mirror that exactly here (perf_counter
    around the submit loop + one xpu.synchronize) so overhead = lora_ms - gemm_ms
    is a like-for-like difference -- the extra metadata build, argument setup and
    output allocation the LoRA wrapper does per call. Using triton.do_bench instead
    would be apples-to-oranges (it flushes L2 and subtracts its own overhead), which
    can even make the pure GEMM look slower than the full kernel.
    """
    import time

    import torch

    for _ in range(warmup):
        run_once()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iterations):
        run_once()
    torch.xpu.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1e3 / iterations


def make_row(
    *,
    case_id: int,
    case_label: str,
    dtype_name: str,
    group_gemm: Dict[str, float],
    lora_ms: float,
    flops: float,
    total_bytes: float,
    groups: int,
) -> Dict[str, float]:
    """Assemble one comparison row: pure group-GEMM vs full LoRA kernel.

    flops / total_bytes are the SAME effective-work and memory-traffic estimates
    the LoRA benchmark computes for this case (from the module's own
    _compute_flops_by_segment / _estimate_bytes), so gemm_* and lora_* metrics are
    directly comparable to the numbers those scripts report.
    """
    gemm_ms = group_gemm["avg_runtime_ms"]
    overhead_ms = lora_ms - gemm_ms

    def tflops(ms: float) -> float:
        return flops * 1e-12 / (ms * 1e-3) if ms > 0 else float("nan")

    def gbs(ms: float) -> float:
        return total_bytes * 1e-9 / (ms * 1e-3) if ms > 0 else float("nan")

    return {
        "case_id": case_id,
        "case_label": case_label,
        "dtype": dtype_name,
        "groups": groups,
        "gemm_ms": gemm_ms,
        "lora_ms": lora_ms,
        "overhead_ms": overhead_ms,
        "overhead_pct": 100.0 * overhead_ms / lora_ms if lora_ms > 0 else float("nan"),
        "overhead_us_per_group": 1e3 * overhead_ms / groups if groups else float("nan"),
        "gemm_tflops": tflops(gemm_ms),
        "lora_tflops": tflops(lora_ms),
        "total_bytes_mb": total_bytes / 1e6,
        "gemm_gbs": gbs(gemm_ms),
        "lora_gbs": gbs(lora_ms),
    }


def dtype_from_name(name: str):
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def drive(
    *,
    title: str,
    module,
    build_problems: Callable[[dict, dict], List[Problem]],
    flops_fn: Callable[[dict, dict], float],
    bytes_fn: Callable[[dict, dict, int], float],
    tile: str,
    per_group_alpha: bool,
    beta_of_case: Callable[[dict], float],
    dtype_names: Sequence[str],
    iterations: int,
    seed: int,
    case_ids: Sequence[int],
    verify: bool = False,
) -> List[Dict[str, float]]:
    """Run every (case, dtype) pair: pure group-GEMM floor vs full LoRA kernel.

    module supplies DEFAULT_CASES, _make_inputs, _run_cutlass_once, _case_label.
    build_problems derives the host-side per-group (M, N, K) list -- the same
    metadata the device kernel computes -- and flops_fn returns the effective
    flops for that case (from the module's own _compute_flops_by_segment).
    """
    import torch

    device = torch.device("xpu")
    cases = module.DEFAULT_CASES
    rows: List[Dict[str, float]] = []

    for case_id in case_ids:
        case = cases[case_id]
        beta = beta_of_case(case)
        for dtype_name in dtype_names:
            torch.manual_seed(seed)  # identical inputs for group-GEMM and LoRA timing
            dtype = dtype_from_name(dtype_name)
            elem_size = torch.tensor([], dtype=dtype).element_size()
            inputs = module._make_inputs(case, dtype, device)

            problems = build_problems(inputs, case)
            group_gemm = run_group_gemm(
                problems,
                tile=tile,
                dtype=dtype_name,
                beta=beta,
                per_group_alpha=per_group_alpha,
                alpha=1.0,
                iterations=iterations,
                verify=verify,
            )

            lora_ms = bench_lora_ms(
                lambda: module._run_cutlass_once(inputs), iterations=iterations
            )
            flops = flops_fn(inputs, case)
            total_bytes = bytes_fn(inputs, case, elem_size)

            rows.append(
                make_row(
                    case_id=case_id,
                    case_label=module._case_label(case),
                    dtype_name=dtype_name,
                    group_gemm=group_gemm,
                    lora_ms=lora_ms,
                    flops=flops,
                    total_bytes=total_bytes,
                    groups=len(problems),
                )
            )
            last = rows[-1]
            print(
                f"[{title}] case {case_id} {dtype_name}: "
                f"groups={last['groups']} gemm={last['gemm_ms']:.4f}ms "
                f"lora={last['lora_ms']:.4f}ms overhead={last['overhead_ms']:.4f}ms "
                f"({last['overhead_pct']:.1f}%) | "
                f"bw gemm={last['gemm_gbs']:.1f} lora={last['lora_gbs']:.1f} GB/s "
                f"({last['total_bytes_mb']:.1f} MB)"
            )

    return rows


def common_argparser(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["bf16", "fp16"],
        choices=["bf16", "fp16"],
        help="Element dtypes to benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Timed group-GEMM iterations in the standalone binary (after 1 warm-up).",
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=None,
        help="Subset of case ids to run (default: all).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Input-generation seed.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Also run the standalone binary's correctness check per case.",
    )
    return parser


def resolve_case_ids(cli_cases, module) -> List[int]:
    if cli_cases is not None:
        return list(cli_cases)
    return list(range(len(module.DEFAULT_CASES)))


def print_report(title: str, rows: List[Dict[str, float]]) -> None:
    import pandas as pd

    print("\n" + "=" * 128)
    print(title)
    print("=" * 128)
    if not rows:
        print("No results collected.")
        return

    df = pd.DataFrame(rows)
    round_cols = {
        "gemm_ms": 4,
        "lora_ms": 4,
        "overhead_ms": 4,
        "overhead_pct": 1,
        "overhead_us_per_group": 2,
        "gemm_tflops": 1,
        "lora_tflops": 1,
        "total_bytes_mb": 1,
        "gemm_gbs": 1,
        "lora_gbs": 1,
    }
    for col, nd in round_cols.items():
        if col in df.columns:
            df[col] = df[col].round(nd)

    display_cols = [
        "case_id",
        "case_label",
        "dtype",
        "groups",
        "gemm_ms",
        "lora_ms",
        "overhead_ms",
        "overhead_pct",
        "overhead_us_per_group",
        "gemm_tflops",
        "lora_tflops",
        "total_bytes_mb",
        "gemm_gbs",
        "lora_gbs",
    ]
    print("\nPer-case (gemm_ms = pure group-GEMM API; lora_ms = full LoRA kernel):")
    print(df[display_cols].to_string(index=False))

    print("\n" + "-" * 128)
    print("Overhead summary by dtype (overhead = metadata build + arg setup + output alloc)")
    print("-" * 128)
    summary = df.groupby("dtype")[
        ["gemm_ms", "lora_ms", "overhead_ms", "overhead_pct", "overhead_us_per_group"]
    ].agg(["mean", "min", "max"])
    print(summary.to_string())

    print("\n" + "-" * 128)
    print("Bandwidth summary by dtype (total_bytes = read x + read weight + write output, per group)")
    print("-" * 128)
    bw_summary = df.groupby("dtype")[["gemm_gbs", "lora_gbs", "total_bytes_mb"]].agg(
        ["mean", "min", "max"]
    )
    print(bw_summary.to_string())
