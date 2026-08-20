# Sliced Grouped-GEMM Metadata: One Build for Every Fused LoRA Kernel

**Component:** `build_grouped_gemm_meta()` /
`BuildGroupedGemmMetaKernel` in
`src/sycl/kernels/lora/common/grouped_gemm_meta.hpp`, on Intel BMG (Battlemage)
via CUTLASS-SYCL / sycl-tla.

**Status:** Landed. This note is the recipe for adding the next fused LoRA
forward kernel (gate-up-B, gate-up-A, qkv-A, …) without touching the metadata
build.

**TL;DR:** Every LoRA forward GEMM we run — plain SGEMM A/B-fwd, fused QKV-B,
and the gate-up / A-fwd fusions still to come — is the *same* pointer-array
grouped GEMM. They differ only in how each input segment is sliced into
per-projection groups, and that difference is fully captured by **four
arguments** to one shared build: `output_offset`, `n_slices`, `a_row_stride`,
`a_slice_stride`. There is one metadata kernel and one set of offset formulas,
so a new fused kernel needs **zero** new metadata code and **zero** new compiled
GEMM kernels — just a runner that passes the right four values.

---

## 0. Why one kernel suffices

A CUTLASS pointer-array grouped GEMM runs many independent `D = alpha*(A@B^T) +
beta*C` problems ("groups") of different shapes in one launch. Its *type* is
fixed by `(element_dtype, tile_config)` only; the group count and every
per-group problem size / stride / base pointer are **runtime** data read from
device arrays. So `n_slices` (2 for gate-up, 3 for qkv, 1 for plain) never
changes the compiled kernel — it only changes the metadata we build on the host
side. That is the whole reason this unification is free.

We build that metadata on device with one SYCL thread per group. `build_grouped_gemm_meta()`
sizes the arrays to `num_groups = n_slices * num_segments` and launches
`BuildGroupedGemmMetaKernel`.

## 1. The group model

Each input segment `s` (a run of tokens sharing one LoRA adapter) emits
`n_slices` groups, one per projection `p`:

```
g          = n_slices * s + p          // group index
s          = g / n_slices              // segment  (decoded in the kernel)
p          = g % n_slices              // projection: q/k/v, or gate/up, …
row_start  = seg_indptr[s]             // first token row of the segment
M_g        = seg_indptr[s+1] - row_start
lora_id    = weight_indices[s]
```

The projection's output column band comes from `output_offset` (an
`int32[n_slices + 1]` prefix-sum of per-projection widths), or spans the whole
output when `output_offset` is absent (the plain single-slice case):

```
col_start  = output_offset ? output_offset[p]                 : 0
N_p        = output_offset ? output_offset[p+1] - col_start    : N_total
```

## 2. The four offset formulas (identical for every kernel)

```
problem_sizes[g] = (M_g, N_p, K)
stride_A[g]      = a_row_stride
stride_B[g]      = K
stride_D[g]      = N_total

a_off[g] = (row_start * a_row_stride + p * a_slice_stride) * elem_bytes
b_off[g] = (lora_id  * N_total * K   + col_start * K)      * elem_bytes
d_off[g] = (row_start * N_total      + col_start)          * elem_bytes

alpha[g] = scalings[lora_id]     // only when scalings supplied (B-fwd); else unset
```

`b_off` / `d_off` / `stride_B` / `stride_D` are already fully general — they need
nothing per-kernel beyond `output_offset` and `N_total`. The **only** operand
that varies structurally is A, and it is described by exactly two scalars:

- `a_row_stride`  — leading dim of A (elements per token row of the A input).
- `a_slice_stride` — how far projection `p`'s A columns advance (0 when A is a
  *shared* input read by every projection).

## 3. The recipe per kernel

`K` below is always the GEMM reduction dim (= the A/B row width). `rank` is the
LoRA adapter rank.

| Kernel | `n_slices` | `output_offset` | `a_row_stride` | `a_slice_stride` | `scalings` | Notes |
|---|---|---|---|---|---|---|
| **Plain SGEMM A-fwd** | 1 | *none* | `K` (=in_dim) | 0 | *none* | `K` = in_dim; N_total = stack_num·rank |
| **Plain SGEMM B-fwd** | 1 | *none* | `K` (=rank) | 0 | yes | `K` = rank; N_total = output_dim |
| **Fused QKV-B** | 3 | `[0, N_q, N_q+N_kv, N_q+2·N_kv]` | `3·K` | `K` | yes | A packed `[num_tokens, 3·K]`; `K` = rank |
| **Fused gate-up-B** | 2 | `[0, N_gate, N_gate+N_up]` | `2·K` | `K` | yes | A packed `[num_tokens, 2·K]`; `K` = rank |
| **Fused QKV-A / gate-up-A** | 3 / 2 | `[0, rank, 2·rank, …]` (uniform) | `K` (=in_dim) | 0 | *none* | A is the **shared** hidden state; each projection writes a rank-wide band |

Reading the table:

- **B-fwd fusions** slice the A *input* (each projection reads its own K-wide
  band of the packed LoRA-A output) → `a_row_stride = n_slices·K`,
  `a_slice_stride = K`. Output N per projection is variable → real `output_offset`.
- **A-fwd fusions** share one A input (the hidden state) across all projections
  → `a_slice_stride = 0`, `a_row_stride = K = in_dim`. Output N per projection is
  the uniform rank → `output_offset = [0, rank, 2·rank, …]`.
- **Plain** is the degenerate `n_slices = 1`, no `output_offset` case — bit-for-bit
  the pre-unification behavior (verified by the SGEMM A/B test suites).

## 4. Defaults keep plain callers untouched

The four sliced parameters are trailing and defaulted
(`output_offset = nullopt`, `n_slices = 1`, `a_row_stride = 0` → resolved to `K`,
`a_slice_stride = 0`), so the plain SGEMM A-/B-fwd runners call
`build_grouped_gemm_meta()` exactly as before. Only fused runners pass the extra
four.

## 5. What is *not* covered here

This build assumes **contiguous segments**: a group's physical token rows are
`row_start .. row_start + M_g`. A permuted / gathered layout (tokens sorted by
adapter, the decode `SORTED_BY_ADAPTER` path) cannot be expressed as one strided
tile per group, so it does **not** belong here. When it lands it will live in a
separate `permute_grouped_gemm_meta.hpp` (most likely a gather-into-scratch →
same grouped GEMM → scatter-back wrapper, leaving these formulas untouched).

## 6. Checklist for adding a fused kernel

1. Pick `n_slices` and build the `int32[n_slices+1]` `output_offset` on device.
2. Decide the A operand: shared input → `a_slice_stride = 0`, `a_row_stride =
   in_dim`; packed input → `a_slice_stride = K`, `a_row_stride = n_slices·K`.
3. Call `build_grouped_gemm_meta(...)` with those four values (+ `scalings` for
   B-fwd).
4. Everything downstream — `make_device_ptrs`, `args_from_options`, the
   `GroupGemmLoraFwd` launcher, the per-group alpha epilogue — is reused verbatim.
   No new `.cpp.in` / cmake / dispatch tile axis is needed for the metadata; you
   only add the kernel's own runner + dispatch + types, as the existing kernels do.
