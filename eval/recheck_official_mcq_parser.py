import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = subprocess.run(
    ['git', 'show', 'official/main:eval/judge_qwenlm.py'],
    cwd=root, check=True, text=True, capture_output=True,
).stdout
wanted = {'extract_first_option', 'extract_mcq_option', 'first_letter_match', 'extract_answer'}
tree = ast.parse(source)
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
assert {node.name for node in functions} == wanted
namespace = {'re': re}
exec(compile(ast.Module(body=functions, type_ignores=[]), '<official/main:eval/judge_qwenlm.py>', 'exec'), namespace)

extract_answer = namespace['extract_answer']
extract_first_option = namespace['extract_first_option']
first_letter_match = namespace['first_letter_match']

crafted = []
for raw in ['Answer: D', 'Final Answer: **D**', 'Answer: D.', '<answer>D</answer>', 'D']:
    extracted = extract_answer(raw)
    crafted.append({
        'raw': raw,
        'extracted': extracted,
        'official_option': extract_first_option(extracted),
        'matches_reference_A': first_letter_match('A', extracted),
        'matches_reference_D': first_letter_match('D', extracted),
    })

def load_jsonl(path, benchmark='mmstar'):
    out = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get('benchmark') == benchmark:
                out[item['sample_uid']] = item
    return out

preds = load_jsonl(root / 'artifacts/runs/E-PAPER-BASEJUDGE-001/base/predictions.jsonl')
r3 = load_jsonl(root / 'artifacts/runs/E-PAPER-BASEJUDGE-001/base/scores.jsonl')
r4 = load_jsonl(root / 'artifacts/runs/E-PAPER-BASEJUDGE-R4-001/base/scores.jsonl')
assert len(preds) == len(r3) == len(r4) == 1500

counts = Counter()
tuples = Counter()
examples = []
direct_keys = set()
for key, pred in preds.items():
    extracted = extract_answer(pred['raw_model_answer'])
    option = extract_first_option(extracted)
    direct = first_letter_match(pred['reference_answer'], extracted)
    old_direct = r3[key]['rule_source'] == 'first_letter'
    if direct:
        direct_keys.add(key)
        counts['official_direct_match'] += 1
    if old_direct:
        counts['r3_first_letter'] += 1
    if direct != old_direct:
        counts['official_vs_r3_disagreement'] += 1
    status = r4[key].get('mcq_parse_status')
    counts[f'official_direct__r4_{status}'] += int(direct)
    if direct and status == 'mismatch':
        counts['confirmed_explicit_wrong_but_official_direct_correct'] += 1
        tup = (str(pred['reference_answer']), option, str(r4[key].get('mcq_predicted_option')))
        tuples[tup] += 1
        if len(examples) < 12:
            examples.append({
                'sample_uid': key,
                'reference': pred['reference_answer'],
                'official_option': option,
                'r4_explicit_option': r4[key].get('mcq_predicted_option'),
                'official_extracted_answer_tail': extracted[-220:],
            })

r3_keys = {k for k, v in r3.items() if v['rule_source'] == 'first_letter'}
result = {
    'official_ref': subprocess.run(['git', 'rev-parse', 'official/main'], cwd=root, check=True, text=True, capture_output=True).stdout.strip(),
    'official_function_names_loaded_via_ast': sorted(wanted),
    'crafted_cases': crafted,
    'counts': dict(sorted(counts.items())),
    'official_direct_keys_equal_r3_first_letter_keys': direct_keys == r3_keys,
    'explicit_wrong_tuple_counts_reference_official_r4': [
        {'reference': k[0], 'official_option': k[1], 'r4_explicit_option': k[2], 'count': v}
        for k, v in sorted(tuples.items())
    ],
    'examples': examples,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
