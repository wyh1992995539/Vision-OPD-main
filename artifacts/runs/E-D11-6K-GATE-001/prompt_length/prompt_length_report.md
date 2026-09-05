# Vision-OPD Prompt Length Audit

- Status: **PASS**
- Rows: **6241**
- Candidate limit: **8192**
- Processing errors: **0**
- GPU used: **No**

| View | Metric | P50 | P95 | P99 | Max | Overlength |
|---|---|---:|---:|---:|---:|---:|
| student | text_tokens | 76 | 88 | 100 | 186 | 0 |
| student | image_tokens | 3290 | 3901 | 4371 | 7802 | 0 |
| student | total_tokens | 3366 | 3974 | 4445 | 7880 | 0 |
| teacher | text_tokens | 76 | 88 | 100 | 186 | 0 |
| teacher | image_tokens | 210 | 1464 | 1820 | 2730 | 0 |
| teacher | total_tokens | 289 | 1545 | 1899 | 2809 | 0 |

The processor was called without truncation or max-length arguments.
