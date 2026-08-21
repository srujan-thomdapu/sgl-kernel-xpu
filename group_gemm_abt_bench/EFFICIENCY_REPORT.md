# Are the LoRA forward kernels using the group-GEMM API efficiently?

**Device:** Intel Arc B580 (BMG-g21) — 20 Xe2 cores, 12 GB GDDR6, 192-bit bus.
**Kernels under test:** `sgemm_lora_a_fwd`, `sgemm_lora_b_fwd`, `qkv_lora_b_fwd`,
`gate_up_lora_b_fwd`.
**Question:** each kernel calls the CUTLASS pointer-array grouped-GEMM API, but first
builds all of the grouped-GEMM metadata *on device, on every launch*
(`grouped_gemm_meta.hpp`: per-group problem sizes, strides, byte offsets, absolute
pointer arrays, the per-group alpha buffer). How much does that cost us, and are we
leaving performance on the table versus what the group-GEMM API can actually do?

---

## TL;DR verdict

| Kernel | Wrapper overhead (metadata build + arg setup + alloc) | Efficient use of the API? |
|--------|-------------------------------------------------------|---------------------------|
| `sgemm_lora_a_fwd` | ~0.08–0.27 ms/call, near-constant. 3–6 % on the large cases, 20–31 % only where the GEMM itself is sub-0.5 ms. | **Yes.** Large cases run within a few % of the raw group-GEMM floor. |
| `sgemm_lora_b_fwd` | ~0.18–0.9 ms/call, 3–11 %. | **Yes.** These cases are all ≥64k tokens, so the fixed cost is a small share throughout. |
| `gate_up_lora_b_fwd` | ~0.4–4.4 ms/call, **5–36 %**, shape-dependent. | **Mostly.** 2 bands (gate, up), so a milder version of qkv: 5–10 % on many-group launches, but the same launch-bound spike (36 %) on the few-group case. |
| `qkv_lora_b_fwd` | ~0.4–5.2 ms/call, **5–43 %**, highly shape-dependent. | **Partially.** Efficient when there are many groups to amortize over; the fixed metadata/band cost is disproportionate on few-group launches. **This is the optimization target.** |

Two distinct efficiency questions, both answered "the shapes, not the API":

1. **Does the group-GEMM API run the LoRA shapes well?** Yes. The pure group-GEMM
   floor (metadata precomputed, only `gemm_op.run()` timed) reaches **50–99 % of the
   realistic bandwidth ceiling** for these shapes. The LoRA problems are memory-bound
   (skinny `K` = LoRA rank / small hidden dims), so bandwidth — not TFLOP/s — is the
   correct yardstick, and we are close to it.
2. **How much does the LoRA wrapper add on top of the API?** Small and roughly constant
   for A-/B-fwd; large and shape-sensitive for qkv.

---

## How this was measured

For every case in the three LoRA benchmarks (`benchmark/bench_*_fwd.py`) we:

1. Rebuild the **exact same inputs** (same shapes, same seed) as the LoRA benchmark.
2. Derive, host-side, the **identical per-group `(M, N, K)` list** the device metadata
   kernel would emit, and run a standalone CUTLASS grouped-GEMM (`bmg_grouped_gemm_abt`,
   same `GroupGemmTypes<>` bundle: ColumnMajor-B ⇒ `A @ Bᵀ`, `XE_DPAS_TT<8,float,T>`,
   2 pipeline stages, matching tile) on just those problems — timing **only**
   `gemm_op.run()`. This is `gemm_ms`, the pure group-GEMM API floor.
3. Time the real LoRA kernel on the same inputs with the **same wall-clock method** (one
   warm-up, N back-to-back submits, one device wait, wall ÷ N). This is `lora_ms`.

```
overhead_ms = lora_ms − gemm_ms      # = on-device metadata build + arg setup + output alloc
```

Both sides use the same offline-compiler flags (critically
`-cl-intel-enable-auto-large-GRF-mode`; without it the standalone spills ~170 registers
and runs ~3× slower, which would make the comparison meaningless). Bandwidth uses the
LoRA benchmarks' own `_estimate_bytes`: `Σ_group (M·K + N·K + M·N)·elem_size`.

Full logs: `results/lora_a.txt`, `results/lora_b.txt`, `results/qkv_b.txt`.

---

## The yardstick: what the group GEMM can actually do (peak capacity)

`bench_group_gemm_peak.py` runs the **same kernel** on large, tile-aligned shapes to
establish the ceiling. Peaks are expressed against the device limits: ~120 TFLOP/s
(bf16/fp16 XMX, vendor-class estimate) and 456 GB/s (192-bit GDDR6). Full log:
`results/peak.txt`.

| Regime | Best shape | Time | TFLOP/s | GB/s |
|--------|-----------|------|---------|------|
| **Compute-bound** (large square, high intensity) | `16× 2048³`, `4× 4096³` | 3.0 / 6.1 ms | **~92 (77 % of ~120)** | ~65–135 |
| **Bandwidth, stream-heavy** (large `M`,`K`; small `N` — the LoRA A-fwd pattern) | `32× 4096×256×4096` | 3.4 ms | ~79 | **~350 (76–79 % of 456)** |
| Bandwidth, write-heavy (large `M·N`, tiny `K`) | `1× 16384²×128` | 3.1 ms | ~22 | ~178 |

**Key findings from the ceiling run:**

- **Compute ceiling ≈ 92 TFLOP/s** (~77 % of theoretical). Reached only by
  compute-bound square shapes with `K` in the thousands. It scales cleanly with group
  count (1→16 groups all hold ~89–92 TFLOP/s), so **the pointer-array API adds no
  per-group scheduling penalty** — a launch of N groups is as efficient as one big GEMM.
- **Realistic bandwidth ceiling ≈ 350 GB/s** (~77 % of the 456 GB/s bus), reached by
  the stream-heavy shape — which is exactly the LoRA A-fwd access pattern (large token
  count, large hidden `K`, narrow projection `N`). Pure write-dominated shapes plateau
  lower (~180 GB/s), so 350 GB/s is the honest ceiling to hold the memory-bound LoRA
  kernels against, **not** 456.
- The very large `8192³` cases dip to ~63–69 TFLOP/s (working set spills cache); the
  sweet spot for this kernel is a few groups of ~4096-scale dims.

**Why this matters for the verdict:** the LoRA kernels never approach 92 TFLOP/s
(their `lora_tflops` are mostly 2–15, up to ~47) — **not because the API is slow, but
because their shapes are memory-bound**. `K` is the LoRA rank (16–64) for B-fwd/qkv, so
there is very little compute per byte moved. The fair question is therefore "how close
to the ~350 GB/s bandwidth ceiling do they get?", and the answer is: close.

---

## Per-kernel findings

### `sgemm_lora_a_fwd` — efficient ✅

Per group `(M=seg_len, N=stack_num·rank, K=input_dim)`, β=0, single α. `K` is the model
hidden size (2k–8k), so these are the least memory-bound of the three.

- Overhead is a near-constant **~0.15–0.27 ms** (bf16), essentially independent of group
  count: 0.077 ms at 4 groups → 0.26 ms at 128 groups. Per-group cost *falls* from ~19 µs
  to ~2 µs as groups grow — i.e. it is a fixed launch cost, not a per-group tax.
- As a fraction: **20–31 %** only on the tiny cases (0–2, where the GEMM is 0.1–0.4 ms);
  drops to **3–6 %** on the large cases (6, 9, 10, 11).
- Bandwidth: pure floor `gemm_gbs` reaches **345–373 GB/s** on the largest case (11) —
  at the ceiling — and the full kernel `lora_gbs` tracks it to within a few percent
  (346 vs 345 for bf16). Smaller cases sit lower purely because they are shorter and
  more launch-bound.

**Verdict:** the wrapper is a thin, constant tax; on realistically large shapes the kernel
runs at the group-GEMM floor, which is itself at the hardware bandwidth ceiling.

### `sgemm_lora_b_fwd` — efficient ✅

Per group `(M=seg_len, N=output_dim, K=rank)`, per-group α, β=1 for `use_base_output`.
Every case is ≥64k tokens, so the GEMM is always several ms.

- Overhead **0.18–0.9 ms**, a steady **3–11 %** across all 15 cases. Per-group cost is
  small (2–45 µs) and shrinks with group count.
- The `+base` cases (12–14) show a *lower* `gemm_gbs` (~125–138 vs ~185 GB/s) — this is
  **not** overhead: β=1 makes the epilogue additionally read and write the base output,
  which the byte estimate doesn't fully credit and which is inherent to the fused residual.
- Bandwidth floor is a steady **~185–195 GB/s**; `lora_gbs` runs ~5–10 % below it.
  (Lower than A-fwd's peak because `K`=rank is tiny → lower arithmetic intensity and a
  write-heavier profile, closer to the ~180 GB/s write-bound plateau than the 350 GB/s
  stream ceiling. Shape-limited, not wrapper-limited.)

**Verdict:** efficient; the fixed cost is a single-digit-percent share on every case.

### `gate_up_lora_b_fwd` — mostly efficient, same launch-bound caveat as qkv ⚠️

Each segment emits **two** groups (the gate and up projections, both width `output_dim`):
`(seg_len, output_dim, rank)` ×2, per-group α, β=1 for `use_base_output`. Same `tall`
(32×512×32) tile and shared sliced metadata build as qkv (`n_slices = 2 + output_offset`),
so it is a lighter, 2-band version of the qkv story.

- Overhead **0.4–4.4 ms**, **5–36 %**; per-group 4–142 µs. Mean ~11–12 % (vs qkv's ~16 %).
- Split by group count, exactly like qkv:
  - **Few-group launch is punished:** case 0 (16 groups, seg=8) pays 2.27 ms = **36 %**
    (142 µs/group); `lora_gbs` drops to ~131 GB/s while the pure floor is ~206 GB/s.
  - **Many-group launches amortize:** cases 2–6 (64–512 groups) settle to **5–10 %**
    (4–8 µs/group); case 6 (512 groups) is within noise of the floor (~0 %), with
    `lora_gbs` ≈ `gemm_gbs` ≈ 199 GB/s.
  - `+base` cases (7–9) add 1.1–4.4 ms and show the same β=1 traffic drop (`gemm_gbs`
    ~130–145 vs ~200) — inherent to the fused residual, not overhead.
- Pure floor bandwidth is a steady **~193–206 GB/s** (like qkv; the wide gate/up bands
  make it a touch higher than B-fwd), so the API handles these shapes well; the gap to
  `lora_gbs` is the 2-band metadata + band-partitioning wrapper.

**Verdict:** efficient on realistic many-group launches; carries the same
few-group launch-bound spike as qkv but one band lighter (2 vs 3 → worst case ~36 % vs
~43 %). It benefits from the **same** fix as qkv — cheaper/amortized on-device metadata
across the gate/up bands.

### `qkv_lora_b_fwd` — the optimization target ⚠️

Each segment emits **three** groups (q, k, v bands): `(seg_len, {n_q|n_kv|n_kv}, rank)`,
per-group α, plus device-side `output_offset` band partitioning into the fused QKV buffer.

- Overhead ranges **0.46–5.2 ms**, i.e. **5–43 %** — far higher and far more variable
  than A-/B-fwd. Per-group overhead spans **1.4 µs to 124 µs**.
- The split is by group count / amortization:
  - **Few-group launches are punished:** case 0 (24 groups) pays 2.98 ms overhead = **43 %**
    (124 µs/group); case 2 (96 groups) pays 4.58 ms = **36 %**. Here `lora_gbs` collapses to
    ~118–129 GB/s while the pure floor is ~200–207 GB/s — the metadata/band build is
    eating nearly half the launch.
  - **Many-group launches amortize well:** cases 3–6 (192–768 groups) settle to
    **5–8 %** (1.4–2.8 µs/group), and `lora_gbs` (~187–191) comes within ~7 % of the
    ~200–207 GB/s floor.
  - `+base` cases (7–9) add another 0.4–1.5 ms of fixed cost on top.
- Pure floor bandwidth is a consistent **~200–207 GB/s** (higher than B-fwd because the
  q band is wide), so the *API* is doing fine on these shapes; the gap to `lora_gbs` is
  the 3-band metadata + partitioning wrapper.

**Verdict:** the group-GEMM API itself runs qkv shapes at ~200 GB/s, but the wrapper's
per-launch cost is 3× the metadata of a single-band kernel and does extra band
partitioning, so it dominates whenever the group count is small. **This is where to
invest:** cheaper on-device metadata construction (or precompute/cache it across the
q/k/v bands), and/or fusing the three bands so a launch carries one metadata set instead
of three.

---

## Bottom line for the team

- **The group-GEMM API is not the bottleneck.** On its own it hits ~92 TFLOP/s
  (77 % of peak) when compute-bound and ~350 GB/s (77 % of bus) when bandwidth-bound,
  and it scales across group counts with no per-group scheduling penalty.
- **The LoRA kernels are memory-bound by shape** (skinny `K`), so their low TFLOP/s
  numbers are expected and correct — the relevant ceiling is bandwidth, and they sit at
  50–99 % of the realistic ~350 GB/s ceiling on their pure-GEMM floor.
- **`sgemm_lora_a_fwd` and `sgemm_lora_b_fwd` are efficient consumers of the API:** the
  on-device metadata build is a small, roughly constant tax, only material when the GEMM
  itself is sub-millisecond (few tokens / small dims).
- **The multi-band B kernels (`qkv`, `gate_up`) share one weakness:** their per-band
  metadata + output-offset partitioning make the fixed per-launch cost disproportionate
  on few-group launches — up to ~43 % (qkv, 3 bands) / ~36 % (gate_up, 2 bands). Both
  amortize to single-digit % once there are enough groups. **`qkv_lora_b_fwd` is the
  highest-value target** (most bands, largest spike); the same fix — cheaper or amortized
  on-device metadata across the bands — applies to `gate_up_lora_b_fwd` too.

---

## Reproduce

```bash
source /home/gta/intel/oneapi/setvars.sh
cd group_gemm_abt_bench && ./build.sh          # builds bmg_grouped_gemm_abt (needs large-GRF flag)

python bench_group_gemm_sgemm_lora_a.py        # per-case overhead + bandwidth vs LoRA-A
python bench_group_gemm_sgemm_lora_b.py        # ... LoRA-B
python bench_group_gemm_qkv_lora_b.py          # ... qkv-B
python bench_group_gemm_gate_up_lora_b.py      # ... gate/up-B
python bench_group_gemm_peak.py                # group-GEMM max capacity (the ceilings above)
```

All numbers in this report are from `results/*.txt` at 200–300 iterations, bf16 + fp16.
```
