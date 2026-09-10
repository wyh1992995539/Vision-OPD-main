from pathlib import Path
import json,sqlite3
O=Path(__file__).resolve().parent;R=O.parents[2]
a=json.loads((O/'artifact.json').read_text());d=json.loads((O/'audit_data.json').read_text())
c=sqlite3.connect(':memory:');c.row_factory=sqlite3.Row
c.execute('CREATE TABLE r4_samples (benchmark TEXT, model TEXT, correct INTEGER, finish_reason TEXT, completion_tokens INTEGER, inference_error INTEGER, judge_required INTEGER)')
for role in ['base','vision_opd','cached_prefix']:
 p=R/'artifacts/runs/E-PAPER-BASEJUDGE-R4-001'/role
 scores={x['sample_uid']:x for x in map(json.loads,(p/'scores.jsonl').read_text().splitlines())}
 for x in map(json.loads,(p/'predictions.jsonl').read_text().splitlines()):
  s=scores[x['sample_uid']];c.execute('INSERT INTO r4_samples VALUES (?,?,?,?,?,?,?)',(x['benchmark'],role,int(s['final_is_correct']),x['finish_reason'],x['completion_tokens'],int(bool(x['error'])),int(s['judge_required'])))
query=(O/'chart_query.sql').read_text();rows=[dict(x) for x in c.execute(query)]
assert sorted(rows,key=lambda x:(x['benchmark'],x['model']))==sorted(d['results'],key=lambda x:(x['benchmark'],x['model']))
a['snapshot']['datasets']['chart']=rows
for ss in [a['sources'],a['manifest']['sources']]:
 for s in ss:
  if s['id']=='audit':
   s['query']['sql']=query;s['query']['language']='sql';s['query']['engine']='SQLite in memory';s['query']['tables_used']=['r4_samples'];s['query']['description']='r4_samples 由 E-PAPER-BASEJUDGE-R4-001 三模型的 scores.jsonl 和 predictions.jsonl 按 sample_uid 对齐加载。实际 SQL 聚合结果与 analyze.py 的 Python 逐题复算完全一致；完整配对与训练指标来自 audit_data.json。'
(O/'artifact.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
print('SQL and Python aggregates match, nine benchmark/model rows.')
