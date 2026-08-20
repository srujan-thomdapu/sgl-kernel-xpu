"""Max-capacity benchmark for the CUTLASS pointer-array grouped GEMM on BMG.

The other three drivers (bench_group_gemm_sgemm_lora_{a,b}.py, bench_group_gemm_qkv_lora_b.py)
replay the *LoRA* shapes, which are deliberately skinny (K = lora_rank is tiny for
B-fwd/qkv, M = seg_len is small for the launch-bound cases). Those measure the group
GEMM in the regime the LoRA kernels actually use it -- and it is memory-bound there.

This script does the opposite: it feeds the SAME kernel large, tile-aligned,
compute-bound shapes so we can see the ceiling -- the best TFLOP/s and GB/s the group
GEMM API can reach on this device. That ceiling is the yardstick the LoRA numbers should
be read against ("how far below peak are we, and is that the API's fault or the shape's?").

Two shape families:
  * compute-bound  -- large square-ish M=N=K, high arithmetic intensity, XMX-bound.
                      This is where TFLOP/s peaks.
  * bandwidth-bound -- large M*N but small K, low arithmetic intensity, memory-bound.
                      This is where GB/s peaks (and is the regime the LoRA kernels live in).

Nothing here touches torch or the LoRA modules -- it is the pure group-GEMM binary only.

Run:  source /home/gta/intel/oneapi/setvars.sh && ./build.sh   # build the binary
      python bench_group_gemm_peak.py
      python bench_group_gemm_peak.py --dtypes bf16 --iterations 200
"""

import argparse
from typing import Dict, List, Sequence

import common

# Intel Arc B580 (BMG-g21) device ceilings.
#   - Memory: 12 GB GDDR6, 192-bit bus @ 19 Gbps  -> 456 GB/s theoretical.
#   - Compute: 20 Xe2 cores, XMX bf16/fp16 peak ~= 110-120 TFLOP/s (vendor-class
#     estimate; used only to express "% of peak", not as a hard spec).
MEM_PEAK_GBS = 456.0
COMPUTE_PEAK_TFLOPS = 120.0

# `large` tile = 256x256x32 (A-/B-fwd), `tall` = 32x512x32 (qkv-b-fwd). We benchmark
# the `large` tile for peak because it is the general-purpose config; the LoRA kernels'
# tiles are picked for their skinny shapes, not for peak throughput.
TILE = "large"

# (label, [(M, N, K)] * groups) -- all dims tile-aligned (M,N % 256, K % 32).
# Compute-bound: square, growing K raises arithmetic intensity toward the XMX ceiling.
# One big group already fills 20 Xe cores (a 2048x2048 output is 8x8=64 tiles), so we
# also sweep group counts to confirm the API scales without per-group launch penalty.
_COMPUTE_SHAPES = [
    ("1x  2048^3", [(2048, 2048, 2048)] * 1),
    ("1x  4096^3", [(4096, 4096, 4096)] * 1),
    ("1x  8192^3", [(8192, 8192, 8192)] * 1),
    ("4x  4096^3", [(4096, 4096, 4096)] * 4),
    ("8x  4096^3", [(4096, 4096, 4096)] * 8),
    ("16x 2048^3", [(2048, 2048, 2048)] * 16),
    ("8x  8192^3", [(8192, 8192, 8192)] * 8),
]

# Bandwidth-bound: skinny-K, so traffic dominates rather than compute. Two sub-shapes:
#   * write-heavy   (large M*N, tiny K)  -- the output write dominates. GDDR6 write-only
#                     tops out lower (~190 GB/s here).
#   * stream-heavy  (large M and K, moderate N) -- the A read (M*K) is streamed and
#                     dominates; this mirrors the LoRA A-fwd large cases and is where
#                     achieved bandwidth actually peaks (~370 GB/s).
# This is the regime the LoRA kernels live in (skinny K = lora_rank / small hidden dims),
# so its ceiling -- not the compute ceiling -- is the fair yardstick for them.
_BANDWIDTH_SHAPES = [
    ("wr 1x 16384x16384x128", [(16384, 16384, 128)] * 1),
    ("wr 8x  8192x 8192x 64", [(8192, 8192, 64)] * 8),
    ("wr 16x 8192x 8192x 32", [(8192, 8192, 32)] * 16),
    ("st 8x  8192x 1024x4096", [(8192, 1024, 4096)] * 8),
    ("st 16x 8192x  512x4096", [(8192, 512, 4096)] * 16),
    ("st 32x 4096x  256x4096", [(4096, 256, 4096)] * 32),
]


def _flops(problems: Sequence[common.Problem]) -> float:
    return float(sum(2 * m * n * k for (m, n, k) in problems))


def _bytes(problems: Sequence[common.Problem], elem_size: int) -> float:
    # Same estimate the LoRA benchmarks use: read A (M*K) + read B (N*K) + write C (M*N).
    return float(sum((m * k + n * k + m * n) * elem_size for (m, n, k) in problems))


def _run_family(
    name: str,
    shapes,
    *,
    dtype_names: Sequence[str],
    iterations: int,
    verify: bool,
) -> List[Dict[str, float]]:
    import torch

    rows: List[Dict[str, float]] = []
    print(f"\n### {name} shapes (tile={TILE})")
    for dtype_name in dtype_names:
        elem_size = torch.tensor([], dtype=common.dtype_from_name(dtype_name)).element_size()
        for label, problems in shapes:
            res = common.run_group_gemm(
                problems,
                tile=TILE,
                dtype=dtype_name,
                beta=0.0,
                per_group_alpha=False,
                alpha=1.0,
                iterations=iterations,
                verify=verify,
            )
            ms = res["avg_runtime_ms"]
            flops = _flops(problems)
            total_bytes = _bytes(problems, elem_size)
            tflops = flops * 1e-12 / (ms * 1e-3)
            gbs = total_bytes * 1e-9 / (ms * 1e-3)
            rows.append(
                {
                    "family": name,
                    "shape": label,
                    "dtype": dtype_name,
                    "groups": len(problems),
                    "gemm_ms": ms,
                    "tflops": tflops,
                    "pct_compute_peak": 100.0 * tflops / COMPUTE_PEAK_TFLOPS,
                    "gbs": gbs,
                    "pct_mem_peak": 100.0 * gbs / MEM_PEAK_GBS,
                    "total_bytes_mb": total_bytes / 1e6,
                }
            )
            last = rows[-1]
            print(
                f"  [{dtype_name}] {label:>20}  groups={last['groups']:>2}  "
                f"{last['gemm_ms']:8.4f} ms  "
                f"{last['tflops']:7.1f} TFLOP/s ({last['pct_compute_peak']:4.0f}% peak)  "
                f"{last['gbs']:7.1f} GB/s ({last['pct_mem_peak']:4.0f}% peak)"
            )
    return rows


def _print_report(rows: List[Dict[str, float]]) -> None:
    import pandas as pd

    print("\n" + "=" * 120)
    print("Group-GEMM max capacity (Intel Arc B580 / BMG-g21)")
    print(f"peaks: compute ~{COMPUTE_PEAK_TFLOPS:.0f} TFLOP/s (bf16/fp16 XMX), memory {MEM_PEAK_GBS:.0f} GB/s")
    print("=" * 120)
    df = pd.DataFrame(rows)
    for col, nd in {
        "gemm_ms": 4, "tflops": 1, "pct_compute_peak": 1,
        "gbs": 1, "pct_mem_peak": 1, "total_bytes_mb": 1,
    }.items():
        df[col] = df[col].round(nd)
    print(df.to_string(index=False))

    print("\n" + "-" * 120)
    for dtype_name, sub in df.groupby("dtype"):
        comp = sub[sub["family"] == "compute-bound"]
        band = sub[sub["family"] == "bandwidth-bound"]
        print(
            f"[{dtype_name}] peak compute = {comp['tflops'].max():.1f} TFLOP/s "
            f"({comp['pct_compute_peak'].max():.0f}% of ~{COMPUTE_PEAK_TFLOPS:.0f})   |   "
            f"peak bandwidth = {band['gbs'].max():.1f} GB/s "
            f"({band['pct_mem_peak'].max():.0f}% of {MEM_PEAK_GBS:.0f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Max-capacity (peak TFLOP/s and GB/s) benchmark for the group GEMM."
    )
    parser.add_argument("--dtypes", nargs="+", default=["bf16", "fp16"], choices=["bf16", "fp16"])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    rows: List[Dict[str, float]] = []
    rows += _run_family(
        "compute-bound", _COMPUTE_SHAPES,
        dtype_names=args.dtypes, iterations=args.iterations, verify=args.verify,
    )
    rows += _run_family(
        "bandwidth-bound", _BANDWIDTH_SHAPES,
        dtype_names=args.dtypes, iterations=args.iterations, verify=args.verify,
    )
    _print_report(rows)


if __name__ == "__main__":
    main()
