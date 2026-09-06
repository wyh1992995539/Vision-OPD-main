# Memory A/B comparison

Status: `REVIEW_WORKLOAD_DIFFERENCE`

Formal training authorized: `false`

Checks:

- comparison_config_match: True
- train_sha256_match: True
- sample_ids_match: True
- hardware_match: True
- cpu_capacity_bytes_match: True
- gpu_abort_ratio_match: True
- cpu_abort_ratio_match: True
- source_hashes_match_or_verified_audit_revision: True
- variant_order: True
- sequential_run_windows: True
- matching_observed_workload: False
- candidate_below_memory_abort_lines: True
- no_gpu_regression: True
- meaningful_observed_reduction: True

| GPU | Baseline peak GiB | Deferred peak GiB | Reduction GiB |
| --- | ---: | ---: | ---: |
| 0 | 94.387 | 78.395 | 15.992 |
| 1 | 93.965 | 77.293 | 16.672 |

CPU peak increase: -0.238 GiB; wall time increase: 0.35 seconds.

- A single pair is not a causal or statistical proof; matching length summaries is not matching generated tokens.

- NVML samples may miss spikes; actor phase allocator peaks and synchronized device markers are separate measurements.

- Cold reload, post-warmup and long-response training gates remain required outside this diagnostic comparison.

- Timing includes synchronized profiling and must not replace the formal budget estimate.
