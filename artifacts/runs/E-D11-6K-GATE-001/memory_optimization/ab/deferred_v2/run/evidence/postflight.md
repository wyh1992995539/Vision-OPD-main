# Memory A/B audit

Status: `PASS_MEMORY_AB_RUN`

Formal training authorized: `false`

Training passed: `True`; memory evidence passed: `True`.

Cold reload is not required for this diagnostic run; optimization is decided by the two-run comparison.

Allocator reserved > device capacity: 192 markers. These are bookkeeping counters, not physical occupancy. Raw values are retained.

| GPU | Observed peak GiB | Used ratio |
| --- | ---: | ---: |
| 0 | 78.395 | 0.8201 |
| 1 | 77.293 | 0.8086 |
