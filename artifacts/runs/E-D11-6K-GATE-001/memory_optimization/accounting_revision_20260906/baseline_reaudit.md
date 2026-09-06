# Memory A/B audit

Status: `PASS_MEMORY_AB_RUN`

Formal training authorized: `false`

Training passed: `True`; memory evidence passed: `True`.

Cold reload is not required for this diagnostic run; optimization is decided by the two-run comparison.

Allocator reserved > device capacity: 189 markers. These are bookkeeping counters, not physical occupancy. Raw values are retained.

Offline re-audit with explicitly archived audit-source revisions; original launch evidence is unchanged.

| GPU | Observed peak GiB | Used ratio |
| --- | ---: | ---: |
| 0 | 94.387 | 0.9874 |
| 1 | 93.965 | 0.9830 |
