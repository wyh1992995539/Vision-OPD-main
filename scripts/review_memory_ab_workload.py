#!/usr/bin/env python3
"""CPU-only workload review companion. Does not change the source-bound A/B audit."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def log_lengths(path):
    """Independently read scalar log fields, not the audit's parse_steps helper."""
    records = {}
    for step, line in re.findall(r'\bstep:(\d+) - ([^\r\n]+)', path.read_text(errors='replace')):
        step = int(step)
        require(step not in records, 'Duplicate metric step')
        values = []
        for field in ('prompt_length/max', 'response_length/mean', 'response_length/max'):
            match = re.search(r'(?:^| - )' + re.escape(field) + r':([^\s]+)', line)
            require(match is not None, f'Missing metric: {field}')
            values.append(float(match[1]))
        records[step] = [step, *values]
    return [records[k] for k in sorted(records)]


def pair_rollouts(a, b, step):
    require(len(a) == len(b) == 8, 'Expected eight rollouts per step and variant')
    require(all(r['step'] == step for r in a + b), 'Rollout step mismatch')
    indexed = [{r['input']: (i, r) for i, r in enumerate(rows)} for rows in (a, b)]
    require(all(len(rows) == 8 for rows in indexed), 'Duplicate decoded input; cannot join safely')
    require(indexed[0].keys() == indexed[1].keys(), 'Unmatched decoded input sets')
    pairs = []
    for prompt, (ai, ar) in indexed[0].items():
        bi, br = indexed[1][prompt]
        require(ar['gts'] == br['gts'], 'Ground-truth mismatch after input-key join')
        pairs.append(dict(step=step, decoded_input_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                          baseline_line=ai + 1, deferred_line=bi + 1,
                          output_text_equal=ar['output'] == br['output'],
                          baseline_output_chars=len(ar['output']), deferred_output_chars=len(br['output'])))
    return pairs


def review(comparison_path):
    comparison_path = Path(comparison_path).resolve()
    comparison = json.loads(comparison_path.read_text())
    require(comparison['status'] == 'REVIEW_WORKLOAD_DIFFERENCE', 'Expected unresolved workload review')
    runs = comparison['runs']
    inputs = {str(comparison_path): digest(comparison_path)}
    for run in runs.values():
        require(run['status'] == 'PASS_MEMORY_AB_RUN', 'Both runs must pass evidence audit')
        for entry in run['inputs']:
            path = Path(entry['path'])
            require(digest(path) == entry['sha256'], f'Changed audit input: {path}')
            inputs[str(path)] = entry['sha256']
        config = run['comparison_config']
        require(config['data']['train_batch_size'] == 8 and config['rollout']['n'] == 1,
                'Token totals require batch=8 and rollout n=1')
        require(config['training']['padding_rows'] == 0, 'Padded sample weighting is unsupported')
        require([row[0] for row in run['workload']] == list(range(1, 9)), 'Expected complete eight-step workload')
        require(len(run['sample_ids']) == len(set(run['sample_ids'])) == 64, 'Invalid sample coverage')
        require(log_lengths(Path(run['output_dir']) / 'logs/train.log') == run['workload'],
                'Independent log metrics disagree with comparison')
    require(runs['baseline']['sample_ids'] == runs['deferred']['sample_ids'], 'Selection IDs/order differ')
    step_rows, joined = [], []
    for a, b in zip(runs['baseline']['workload'], runs['deferred']['workload']):
        step = a[0]
        rows = []
        for variant in ('baseline', 'deferred'):
            path = Path(runs[variant]['output_dir']) / 'rollouts' / f'{step}.jsonl'
            inputs[str(path)] = digest(path)
            rows.append(jsonl(path))
        matched = pair_rollouts(*rows, step)
        joined.extend(matched)
        row = dict(step=step, samples=8, baseline_prompt_max=a[1], deferred_prompt_max=b[1],
                   baseline_response_mean=a[2], deferred_response_mean=b[2],
                   baseline_response_max=a[3], deferred_response_max=b[3],
                   baseline_tokens=a[2]*8, deferred_tokens=b[2]*8,
                   identical_output_texts=sum(r['output_text_equal'] for r in matched))
        require(all(row[k].is_integer() for k in ('baseline_tokens', 'deferred_tokens')), 'Nonintegral token subtotal')
        for variant in ('baseline', 'deferred'):
            shapes = runs[variant]['memory_stages']['forward_shapes']
            row[variant + '_student_microbatches'] = sum(s[1] == step and s[2] == 'student_forward/before' for s in shapes)
        step_rows.append(row)
    totals = {v: sum(r[v + '_tokens'] for r in step_rows) for v in ('baseline', 'deferred')}
    shapes = {}
    for variant, run in runs.items():
        records = run['memory_stages']['forward_shapes']
        shapes[variant] = dict(forward_records=len(records),
            phase_records=dict(Counter(r[2] for r in records)),
            response_widths=sorted({r[6] for r in records}),
            student_sequence_widths=sorted({r[4] for r in records if r[2] == 'student_forward/before'}),
            max_unpadded_tokens=max(r[5] for r in records))
        for phase in ('student_forward/before', 'teacher_forward/before'):
            require(sum(r[3] for r in records if r[2] == phase) == 64, 'Forward sample coverage mismatch')
    for name in ('scripts/review_memory_ab_workload.py', 'verl/trainer/ppo/metric_utils.py',
                 'verl/trainer/ppo/ray_trainer.py'):
        inputs[str(ROOT / name)] = digest(ROOT / name)
    return dict(status='REVIEW_WORKLOAD_DIFFERENCE', validation='Share with caveats',
        formal_training_authorized=False, optimization_validated=False,
        comparison_path=str(comparison_path), comparison_generated_at_utc=comparison['generated_at_utc'],
        scope='Two sequential 64-sample / eight-step diagnostic runs; all rows included',
        checks=dict(original_audit_inputs_unchanged=True, independent_log_metric_match=True,
                    unique_sample_ids_64=True, selected_ids_order_match=True,
                    per_step_decoded_input_set_match=True, joined_ground_truth_match=True,
                    forward_sample_coverage_64=True),
        total_response_tokens=totals, mean_response_tokens={v:n/64 for v,n in totals.items()},
        deferred_total_change_percent=100*(totals['deferred']/totals['baseline']-1),
        joined_output_text_equal=sum(r['output_text_equal'] for r in joined),
        changed_row_positions=sum(r['baseline_line'] != r['deferred_line'] for r in joined),
        step_rows=step_rows, decoded_input_pairs=joined, shape_summary=shapes,
        limitations=[
            'Totals reconstructed from logged response_mask mean * 8, independently reconciled to training logs; not retokenized decoded text.',
            'Rollouts omit sample IDs, raw generated token IDs and per-sample mask lengths. Join is step plus unique decoded input, not asserted raw multimodal/token identity.',
            'Decoded output character counts are not token counts. Per-sample exact token lengths and quantiles are unavailable from these exports.',
            'Shape counts describe microbatch calls, not a sample-by-sample paired compute trace. Dynamic batching differs.',
            'Same seed and temperature=1 do not establish identical generated tokens; the cause of divergence is not proven by these records.',
            'Allocator bookkeeping is not physical VRAM residency; no linear token-to-VRAM normalization or causal attribution is valid here.',
            'No post-warmup steps and no near-1024-token response safety validation. Formal training remains blocked.'],
        inputs=[dict(path=p, sha256=h) for p,h in sorted(inputs.items())])


def write_artifacts(report, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'workload_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    for name, rows in [('step_lengths.csv', report['step_rows']), ('decoded_input_pairs.csv', report['decoded_input_pairs'])]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    totals, means = report['total_response_tokens'], report['mean_response_tokens']
    lines = ['# A/B 生成长度复核', '', '**结论：需复核负载差异，不授权正式训练。**', '',
        '两组各 64 条、8 步均完整。以步骤＋唯一解码输入配对，不按导出行号配对。', '',
        f"原比较生成时间（UTC）：{report['comparison_generated_at_utc']}", '',
        f"总回复 token：{totals['baseline']:.0f} → {totals['deferred']:.0f}（{report['deferred_total_change_percent']:.2f}%）；"
        f"均值 {means['baseline']:.4f} → {means['deferred']:.4f}。", '',
        f"64/64 条输入配对且标签一致；{report['changed_row_positions']}/64 条的行位置不同；"
        f"仅 {report['joined_output_text_equal']}/64 条解码输出文本相同。", '',
        '| step | B mean | D mean | B max | D max | B tokens | D tokens | B/D Student 微批次 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in report['step_rows']:
        lines.append(f"| {r['step']} | {r['baseline_response_mean']} | {r['deferred_response_mean']} | {r['baseline_response_max']:.0f} | {r['deferred_response_max']:.0f} | {r['baseline_tokens']:.0f} | {r['deferred_tokens']:.0f} | {r['baseline_student_microbatches']}/{r['deferred_student_microbatches']} |")
    lines += ['', '## 质量判断与影响', '',
        '- 高影响、高置信：8/8 步的平均/最大回复长度统计不同，前向微批次划分也不同；不满足既定同负载自动准入。',
        '- 中影响、高置信：导出行顺序不同。已按唯一输入重新连接，避免把不同题目当成生成差异。',
        '- 中影响、已知缺口：未导出逐样本原始 token IDs/response mask，不能重建精确逐样本 token 分布。',
        '- 两组配置的 response 上限和记录的 response_width 均为 1024；实际有效 token 更少。固定张量宽度不等于相同计算量，remove-padding 和动态微批次仍会改变负载。',
        '- 总 token 接近，且 deferred 并非每步更短；但较短的长尾与不同微批次仍是混杂因素，不能把显存降幅全部归因于延迟加载。',
        '- temperature=1 的采样、执行数值差异或后续参数轨迹可能参与输出分歧；尚未通过受控重放定位原因。', '',
        '## 下一步（尚未执行）', '',
        '若需因果归因，准备同 token/同计算负载重放或明确的配对压力测试；若目标是资源安全，单独做近 1024-token 与 warmup 后更新验证。不要仅为得到 PASS 放宽现有负载判据。', '',
        '## 口径与局限', '']
    lines += ['- '+s for s in report['limitations']]
    lines += ['', '## 证据', '',
              '- 主比较：`../comparison_after_run.json` / `.md` / `.sha256`。',
              '- 本目录 JSON 列出所有输入路径和 SHA256，CSV 保存逐步和逐输入配对记录。',
              '- `workload_review.ipynb` 为可执行复核记录，仅使用 CPU。',
              '- 未修改训练配置、原日志、checkpoint、历史 FAIL 或比较阈值。']
    (output / 'workload_review.md').write_text('\n'.join(lines)+'\n')


def notebook(report, output):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.v4.new_notebook()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb.cells = [md('## tl;dr\n\n总回复 token 5939 → 5823；长尾和微批次不同，维持 REVIEW_WORKLOAD_DIFFERENCE。'),
        md('## Context & Methods\n\n两组各 64 条、8 步。以步骤＋唯一解码输入连接。\n\n### Key Assumptions\n\n日志 response_mask 均值×8 重建每步总 token；解码文本不能替代原始 token IDs。'),
        code(f'import sys, json\nfrom pathlib import Path\nsys.path.insert(0, {str(ROOT)!r})\nfrom scripts.review_memory_ab_workload import review\ncomparison_path = Path({report["comparison_path"]!r})'),
        md('## Data\n\n校验原审计输入哈希、8 步日志指标、64 条输入配对及完整覆盖。'),
        code('r = review(comparison_path)\nassert all(r["checks"].values())\nprint(json.dumps(r["checks"], indent=2))'),
        md('## Results\n\n逐步长度与总体复算：'),
        code('for row in r["step_rows"]:\n    print(row)\nfor variant in ("baseline", "deferred"):\n    total = sum(x[variant+"_response_mean"] * x["samples"] for x in r["step_rows"])\n    assert total == r["total_response_tokens"][variant]\n    print(variant, "tokens", total, "mean", total/64)\nprint("forward shapes:", r["shape_summary"])\nprint("identical decoded outputs:", r["joined_output_text_equal"])'),
        md('## Takeaways\n\n同一批输入，但生成和计算负载未受控。不能按 token 比例线性归一化显存，也不能解除长回复和 warmup 后验证。逐样本 token 分布缺失是明确限制，不是已完成验证。')]
    nb.metadata.kernelspec = dict(name='python3', display_name='Python 3', language='python')
    nbformat.validate(nb)
    NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.validate(nb)
    nbformat.write(nb, output / 'workload_review.ipynb')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    report = review(args.comparison)
    write_artifacts(report, args.output_dir)
    notebook(report, args.output_dir)
    paths = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    (args.output_dir / 'sha256.txt').write_text(''.join(f'{digest(p)}  {p.resolve()}\n' for p in paths))
    print(json.dumps({k: report[k] for k in ['status', 'total_response_tokens', 'mean_response_tokens',
          'deferred_total_change_percent', 'joined_output_text_equal', 'changed_row_positions', 'shape_summary']}, indent=2))
    return 0  # Successful review execution does not mean optimization passed.


if __name__ == '__main__':
    raise SystemExit(main())
