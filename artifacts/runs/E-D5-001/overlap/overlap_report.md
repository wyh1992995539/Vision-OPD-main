# Vision-OPD Benchmark Overlap Audit

**Decision status:** `confirmed_overlap`

## Technical summary

- Project samples: 1216; benchmark samples: 2536.
- Candidate sample pairs: 5; confirmed: 4; dismissed: 1; unresolved: 0.
- Fingerprint/data errors: 0.

## Results by benchmark

| Benchmark | Samples | Candidates | Confirmed | Dismissed | Unresolved | Confirmed impacted rate |
|---|---:|---:|---:|---:|---:|---:|
| mmstar | 1500 | 0 | 0 | 0 | 0 | 0.000% |
| vstar | 191 | 4 | 4 | 0 | 0 | 2.094% |
| zoombench | 845 | 1 | 0 | 1 | 0 | 0.000% |

## Method

- Exact images: SHA256 equality across project and benchmark image references.
- Exact questions: NFKC normalization, Unicode casefold, and whitespace collapse.
- Perceptual images: 64-bit DCT pHash after EXIF transpose; distance <= 5 is unresolved until manual review.
- Exact image matches are automatically confirmed; question-only and pHash-only matches remain unresolved.

## Reporting consequence

Official full-set scores must always be preserved. If confirmed or unresolved candidates exist, any deduplicated score is diagnostic only and the benchmark must not be described as fully independent.

## Files

- Machine-readable summary: `overlap_report.json`
- Candidate evidence: `overlap_candidates.jsonl`
- Manual review sheet: `manual_review.csv`
- Applied review decisions: `manual_review_decisions.json`
- Reusable image fingerprint cache: `image_fingerprint_cache.json`
