# Day 7 Prompt Length Audit

- Status: **SMOKE_PASS**
- Rows: **1024**
- Candidate limit: **8192**
- Processing errors: **0**
- GPU used: **No**

| View | Metric | P50 | P95 | P99 | Max | Overlength |
|---|---|---:|---:|---:|---:|---:|
| student | text_tokens | 72 | 72 | 72 | 72 | 0 |
| student | image_tokens | 2914 | 2914 | 2914 | 2914 | 0 |
| student | total_tokens | 2986 | 2986 | 2986 | 2986 | 0 |
| teacher | text_tokens | 72 | 72 | 72 | 72 | 0 |
| teacher | image_tokens | 65 | 65 | 65 | 65 | 0 |
| teacher | total_tokens | 137 | 137 | 137 | 137 | 0 |

The processor was called without truncation or max-length arguments.
