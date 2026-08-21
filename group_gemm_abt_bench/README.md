# Group-GEMM (A @ Bᵀ) standalone example + LoRA metadata-overhead analysis

This directory isolates the **pure CUTLASS pointer-array grouped-GEMM cost** on
Intel BMG (Battlemage) XPU from the per-launch overhead the LoRA forward kernels
(`sgemm_lora_a_fwd`, `sgemm_lora_b_fwd`, `qkv_lora_b_fwd`) pay to build their
grouped-GEMM metadata on device every call.

## What's here

| File | Purpose |
|------|---------|
| `bmg_grouped_gemm_abt.cpp` | Standalone A @ Bᵀ grouped-GEMM, adapted from cutlass-sycl `examples/04_bmg_grouped_gemm`. Reproduces the LoRA `GroupGemmTypes<>` bundle (ColumnMajor B ⇒ A @ Bᵀ, `XE_DPAS_TT<8,float,T>`, 2 pipeline stages, per-group scalar epilogue). Reads per-group `(M N K)` triples from `--problem_file` and times only `gemm_op.run()`. |
| `build.sh` | Canonical build (single `icpx` invocation). |
| `CMakeLists.txt` | Optional out-of-tree CMake build (mirrors `build.sh`). |
| `common.py` | Shared driver: runs the binary on a host-computed problem list, times the real LoRA kernel with the **same** wall-clock methodology, computes overhead. |
| `bench_group_gemm_sgemm_lora_a.py` | Driver mirroring `benchmark/bench_sgemm_lora_a_fwd.py`. |
| `bench_group_gemm_sgemm_lora_b.py` | Driver mirroring `benchmark/bench_sgemm_lora_b_fwd.py`. |
| `bench_group_gemm_qkv_lora_b.py` | Driver mirroring `benchmark/bench_qkv_lora_b_fwd.py`. |
| `bench_group_gemm_gate_up_lora_b.py` | Driver mirroring `benchmark/bench_gate_up_lora_b_fwd.py`. |
| `bench_group_gemm_peak.py` | Max-capacity benchmark: same kernel on large tile-aligned shapes to find the peak TFLOP/s and GB/s (the ceiling the LoRA numbers are read against). |
| `EFFICIENCY_REPORT.md` | **Team-facing writeup**: does each LoRA kernel use the group-GEMM API efficiently? Verdict + per-kernel analysis + peak ceilings. Start here. |
| `results/` | Saved run logs (`lora_a.txt`, `lora_b.txt`, `qkv_b.txt`, `gate_up_b.txt`, `peak.txt`). |

## The A @ Bᵀ adjustment

The base example computes `A @ B`. The LoRA kernels need `A @ Bᵀ` for every
group (weight `W` is stored `[N, K]` and we want `x @ Wᵀ`). This is achieved with
`LayoutB = cutlass::layout::ColumnMajor`; the auto-selected 2D copy atom
free-transposes it. The one subtlety is the B stride: for ColumnMajor B,
CUTLASS's `InternalStrideB` places the leading dim in slot 0 (B's canonical shape
is `(N, K, L)`), so `make_cute_packed_stride(StrideB{}, {N, K, 1})` is required to
get leading dim = K. See the comments in `allocate()`.

## Per-group problem shapes (must match the LoRA benchmarks)

For each benchmark case the driver rebuilds the **exact same inputs** (same
shapes, same seed) as the corresponding LoRA benchmark, derives the segment
lengths, and emits the per-group `(M, N, K)` the device metadata kernel would
compute:

| Kernel | Groups per segment | (M, N, K) | Epilogue |
|--------|--------------------|-----------|----------|
| `sgemm_lora_a_fwd` | 1 | `(seg_len, stack_num·max_rank, input_dim)` | β=0, single α=1 (no per-adapter scaling) |
| `sgemm_lora_b_fwd` | 1 | `(seg_len, output_dim, max_rank)` | per-group α (scalings); β=1 iff `use_base_output` |
| `qkv_lora_b_fwd` | 3 (q, k, v) | `(seg_len, {n_q|n_kv|n_kv}, max_rank)` | per-group α; β=1 iff `use_base_output` |
| `gate_up_lora_b_fwd` | 2 (gate, up) | `(seg_len, output_dim, max_rank)` | per-group α; β=1 iff `use_base_output` |

Tile config matches the per-kernel `*_types.hpp`: `large` (256×256×32) for A-/B-fwd,
`tall` (32×512×32) for qkv- and gate/up-b-fwd.

## Building

```bash
source /home/gta/intel/oneapi/setvars.sh
./build.sh
```

> **Critical build flag.** The offline compiler must run in
> `-cl-intel-enable-auto-large-GRF-mode` (as the sgl-kernel-xpu build does — see
> `cmake/BuildFlags.cmake`). Without it this kernel spills ~170 registers and
> runs **~3× slower**, which makes the standalone a non-faithful proxy for the
> in-kernel group GEMM and yields bogus (negative) overhead. `build.sh` and
> `CMakeLists.txt` already pass it; keep them in lockstep with the project.

## Running the analysis

```bash
python bench_group_gemm_sgemm_lora_a.py      # all cases, bf16 + fp16
python bench_group_gemm_sgemm_lora_b.py
python bench_group_gemm_qkv_lora_b.py
python bench_group_gemm_gate_up_lora_b.py
python bench_group_gemm_peak.py              # max-capacity ceiling (no LoRA module needed)
# options: --cases 0 2 5   --dtypes bf16   --iterations 300   --verify
```

## Methodology

Both sides are timed **identically** — one warm-up, then N back-to-back submits
followed by a single device wait, wall-time ÷ N (the binary uses cutlass
`GPU_Clock`, which is `std::chrono`; `common.bench_lora_ms` mirrors it with
`time.perf_counter` + `torch.xpu.synchronize`). This is deliberate: `triton.do_bench`
flushes the L2 and subtracts its own overhead, which is *not* what the binary
does, and mixing the two can make the pure GEMM look slower than the full kernel.

```
overhead_ms = lora_ms − gemm_ms
```

- `gemm_ms` — pure group-GEMM API floor (metadata precomputed host-side, only
  `gemm_op.run()` timed).
- `lora_ms` — full LoRA kernel: on-device metadata build (`grouped_gemm_meta.hpp`
  `build_grouped_gemm_meta` / `make_device_ptrs`) + argument setup + output
  allocation + the same group GEMM.

So `overhead` is what the LoRA wrapper adds on top of the raw group-GEMM API.

### Bandwidth (estimated bytes)

Alongside TFLOP/s, each driver reports achieved bandwidth using the **same**
memory-traffic estimate the LoRA benchmarks use (`_estimate_bytes` +
`calc_metrics`): per group, `read x + read weight + write output`, summed over
segments (and over q/k/v bands for qkv). For bf16/fp16 `elem_size = 2`.

```
total_bytes = Σ_group (M·K + N·K + M·N) · elem_size
gemm_gbs = total_bytes / gemm_time      lora_gbs = total_bytes / lora_time
```

`gemm_gbs` is the bandwidth of the pure group-GEMM floor; `lora_gbs` is the
effective bandwidth of the full LoRA kernel (lower, since it also does the
metadata build). These are directly comparable to the `bandwidth_gbs` column the
LoRA benchmark scripts print. Reported per case and summarised per dtype
(`total_bytes_mb`, `gemm_gbs`, `lora_gbs`).

## Max capacity (the ceiling the LoRA numbers are read against)

`bench_group_gemm_peak.py` runs the **same** group-GEMM kernel on large, tile-aligned
shapes to find how fast the API can go on this device (Intel Arc B580: ~120 TFLOP/s
bf16/fp16 XMX, 456 GB/s bus). Results (`results/peak.txt`):

| Regime | Best shape | TFLOP/s | GB/s |
|--------|-----------|---------|------|
| Compute-bound (large square) | `16× 2048³`, `4× 4096³` | **~92 (77 % of peak)** | ~65–135 |
| Bandwidth, stream-heavy (the LoRA A-fwd pattern) | `32× 4096×256×4096` | ~79 | **~350 (77 % of bus)** |
| Bandwidth, write-heavy (large M·N, tiny K) | `1× 16384²×128` | ~22 | ~178 |

Throughput scales cleanly from 1→16 groups (all hold ~89–92 TFLOP/s), so **the
pointer-array API adds no per-group scheduling penalty**. The LoRA kernels are
memory-bound (skinny `K` = LoRA rank / small hidden dims), so the fair yardstick is the
**~350 GB/s realistic bandwidth ceiling**, not the compute peak — and their pure-GEMM
floor reaches 50–99 % of it.

## Findings (BMG, 300 iterations)

> Full analysis and the per-kernel efficiency verdict are in
> [`EFFICIENCY_REPORT.md`](EFFICIENCY_REPORT.md). Summary below.


The overhead is essentially a **fixed per-launch cost** (device metadata kernel +
output `torch.empty` + argument marshalling) that grows only weakly with the
group count. It therefore dominates small problems and fades on large ones:

- **`sgemm_lora_a_fwd`:** overhead ≈ 0.05–0.27 ms/call. As a fraction of total:
  ~20–30 % for the small cases, dropping to ~3–6 % for the large ones.
- **`sgemm_lora_b_fwd`:** overhead ≈ 0.18–0.6 ms/call, ≈ 1–11 % of total
  (these cases start at 65 536 tokens, so the GEMM is larger and the fixed cost
  is a smaller share).
- **`qkv_lora_b_fwd`:** overhead ≈ 0.4–5.2 ms/call, ≈ 5–43 % of total —
  markedly larger and more shape-dependent than A/B-fwd. Each segment emits
  **three** groups (q, k, v), so a launch carries 3× the metadata entries
  (24–768 groups here), and the device build also handles the `output_offset`
  band partitioning. The worst cases (case 0 ≈ 43 %, case 2 ≈ 36 %) share
  `n_kv=1024` with few tokens per group, i.e. a launch-bound regime where the
  fixed build cost is a large share; the compute-heavier cases settle to ~5–13 %.
- **`gate_up_lora_b_fwd`:** overhead ≈ 0.4–4.4 ms/call, ≈ 5–36 % of total — a
  milder, 2-band version of qkv (each segment emits **two** groups, gate + up).
  The few-group case (case 0, 16 groups) spikes to ~36 %; the many-group cases
  (64–512 groups) amortize to ~5–10 % and the largest reaches the floor.

**Bandwidth.** These kernels are memory-bound, so bandwidth tracks the timing
directly. The pure group-GEMM floor sustains ~185–195 GB/s (B-fwd), ~200–207
GB/s (qkv), ~193–206 GB/s (gate/up), and up to ~345–373 GB/s on the largest A-fwd cases; the full LoRA
kernel runs ~10–35 % lower `lora_gbs` on the launch-bound cases (the metadata
build eats into effective bandwidth) and converges to within a few percent of
the floor on the large, compute-heavier cases — the same pattern as the overhead
percentages above. Full per-case bandwidth is in `results/*.txt`.

Takeaway: `sgemm_lora_a_fwd` / `sgemm_lora_b_fwd` are efficient consumers of the
group-GEMM API — the on-device metadata build is a small, roughly constant tax
that is only material in the launch-bound regime (few tokens / small hidden
dims); their large, compute-heavier cases run within a few percent of the raw
group-GEMM floor. The multi-band B kernels `qkv_lora_b_fwd` (3 bands, up to ~43 %)
and `gate_up_lora_b_fwd` (2 bands, up to ~36 %) pay a distinctly higher and more
variable tax because of their per-band metadata + `output_offset` partitioning;
both amortize to single-digit % once there are enough groups. `qkv_lora_b_fwd` is
the strongest candidate for overhead reduction (most bands, largest spike), with
the same fix — cheaper/amortized on-device metadata across bands — applying to
`gate_up_lora_b_fwd`. Numbers on the sub-percent margin (largest A-/B-fwd kernels)
are within run-to-run wall-clock noise and occasionally read slightly negative.
