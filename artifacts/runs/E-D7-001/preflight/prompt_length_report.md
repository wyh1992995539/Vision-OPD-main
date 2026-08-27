# Day 7 Prompt Length Audit

- Status: **PASS**
- Rows: **1024**
- Candidate limit: **8192**
- Processing errors: **0**
- GPU used: **No**

| View | Metric | P50 | P95 | P99 | Max | Overlength |
|---|---|---:|---:|---:|---:|---:|
| student | text_tokens | 76 | 87 | 100 | 129 | 0 |
| student | image_tokens | 3290 | 3901 | 4559 | 7802 | 0 |
| student | total_tokens | 3366 | 3974 | 4638 | 7880 | 0 |
| teacher | text_tokens | 76 | 87 | 100 | 129 | 0 |
| teacher | image_tokens | 209 | 1444 | 1800 | 2134 | 0 |
| teacher | total_tokens | 287 | 1527 | 1881 | 2213 | 0 |

The processor was called without truncation or max-length arguments.
