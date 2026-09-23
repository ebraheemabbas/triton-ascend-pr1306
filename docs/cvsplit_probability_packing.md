# CVSplit probability packing

The selector changes only how the materialized VECTOR softmax expresses its FP16 probability output.

## Usage

```python
dict(enable_cv_split_scheduling=True,
     enable_dynamic_cv_pipeline=False,
     cv_split_schedule_candidate_id=2,
     cv_split_post_split_schedule_mode="materialize",
     cv_split_probability_packing="direct_nz")  # default
```

Select `cv_split_probability_packing="nd_transpose"` for ordinary P followed by
reshape/transpose. The native pass option is `probability-packing=direct_nz` or
`probability-packing=nd_transpose`. Invalid values fail. The option does not enable
CVSplit, select a candidate, or alter generic CVSplit when materialization is disabled.

## Semantics

- `direct_nz`: FP16 chunks are inserted directly into `[N/16, rows, 16]` storage.
- `nd_transpose`: FP16 chunks are inserted into ordinary `[rows, N]` storage;
  a subsequent loop in that SIMD region reads 64 elements per chunk, reshapes
  `[1, 64]` to `[1, 4, 16]`, transposes `[1, 0, 2]`, and inserts into the NZ output.
  This is the same full ND-to-NZ permutation, tiled to register capacity.
  September7 Bisheng rejected the original full-tile transpose inside explicit
  SIMD with `Exceeds vector capacity`; that failed capture is retained.
- Both SIMD regions return the same `[N/16, rows, 16]` NZ tensor. Existing final
  reshape, vector-core L1 subview, UB-to-L1 copy, and release/publication events
  stay unchanged. Keeping the transpose inside the producer region lets Bisheng
  attempt store fusion without inserting an extra explicit SIMD-region boundary.
- The CUBE schedule, shared UB/L1 pools, explicit L0C pools, row arithmetic and
  denominator reduction tree are not selected differently by this option.
- Scratch allocation and final backend code may differ. Neither fusion nor a
  speedup is guaranteed. Compare actual IR/binaries and same-device timings.
- The option participates in the existing NPUOptions cache hash. Default and
  explicit direct_nz hash alike; nd_transpose has a distinct cache identity.

## Qualification

The regression driver `Inputs/check_probability_packing.py` checks HD64/HD128,
U2/U4 and BM64/BM128 fixtures, default/direct identity, actual materialization,
unchanged CUBE/shared storage/events/copies and floating-operation counts,
bounded chunk transposes inside each ND SIMD region, generic-mode invariance, unsupported
geometry/no-candidate rollback, and invalid-option failure. Backend Python tests
cover option parsing, opt-in independence, validation and cache separation.

These compiler gates do not establish hardware correctness, emitted physical
lifetimes or performance. New-mode hardware qualification must be reported separately.
Source is isolated on `cvsplit/impl-amros/probability-packing-v1`, based on `662a3a4cd`.
