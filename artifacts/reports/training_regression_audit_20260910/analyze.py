from pathlib import Path
from collections import Counter
import json, math, statistics, hashlib
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
roles=['base','vision_opd','cached_prefix']
S={};P={};hashes={}
for role in roles:
 d=ROOT/'artifacts/runs/E-PAPER-BASEJUDGE-R4-001'/role
 for name,target in [('scores.jsonl',S),('predictions.jsonl',P)]:
  p=d/name;rows=[json.loads(t) for t in p.read_text().splitlines()];target[role]={x['sample_uid']:x for x in rows}
  assert len(rows)==len(target[role])==2536
  hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
assert all(set(S[r])==set(S['base'])==set(P[r]) for r in roles)
assert all(len({S[r][u]['reference_answer'] for r in roles})==1 for u in S['base'])
assert all(len({P[r][u]['prompt'] for r in roles})==1 for u in S['base'])
benchmarks=sorted({x['benchmark'] for x in S['base'].values()})
def paired(us,role):
 gain=sum(not S['base'][u]['final_is_correct'] and S[role][u]['final_is_correct'] for u in us)
 loss=sum(S['base'][u]['final_is_correct'] and not S[role][u]['final_is_correct'] for u in us)
 n=len(us);d=(gain-loss)/n
 se=math.sqrt(max(0,(gain+loss)/n-d*d)/n)
 m=gain+loss
 p=min(1.,2*sum(math.comb(m,k) for k in range(min(gain,loss)+1))/2**m) if m else 1.
 return {'n':n,'gain':gain,'loss':loss,'delta_pp':100*d,'approx_paired_95ci_pp':[100*(d-1.96*se),100*(d+1.96*se)],'mcnemar_exact_p':p}
results=[];pairs=[];subsets=[];cats=[]
for b in benchmarks:
 us=[u for u,x in S['base'].items() if x['benchmark']==b]
 for r in roles:
  pred=[P[r][u] for u in us]
  results.append({'benchmark':b,'model':r,'n':len(us),'correct':sum(S[r][u]['final_is_correct'] for u in us),'accuracy':sum(S[r][u]['final_is_correct'] for u in us)/len(us),'length':sum(x['finish_reason']=='length' for x in pred),'mean_tokens':statistics.mean(x['completion_tokens'] for x in pred),'errors':sum(bool(x['error']) for x in pred),'judge_required':sum(S[r][u]['judge_required'] for u in us)})
 for r in roles[1:]:pairs.append({'benchmark':b,'model':r,**paired(us,r)})
 for kind,sel in [('all_stop',[u for u in us if all(P[r][u]['finish_reason']=='stop' for r in roles)]),('all_stop_explicit',[u for u in us if all(P[r][u]['finish_reason']=='stop' and S[r][u].get('mcq_parse_status') in ['match','mismatch'] for r in roles)])]:
  if sel:
   subsets.append({'benchmark':b,'subset':kind,'n':len(sel),'accuracy_percent':{r:100*sum(S[r][u]['final_is_correct'] for u in sel)/len(sel) for r in roles},'pairs':{r:paired(sel,r) for r in roles[1:]}})
 for c in sorted({S['base'][u].get('official_category') or 'unknown' for u in us}):
  sub=[u for u in us if (S['base'][u].get('official_category') or 'unknown')==c]
  cats.append({'benchmark':b,'category':c,'n':len(sub),**{r:100*sum(S[r][u]['final_is_correct'] for u in sub)/len(sub) for r in roles}})
metrics={}
for r,run in [('vision_opd','E-D12-6K-VOPD-001'),('cached_prefix','E-D14-6K-CACHED-001')]:
 p=ROOT/'artifacts/runs'/run/'evidence/runtime_metrics.jsonl';a=[json.loads(t) for t in p.read_text().splitlines()];assert [x['step'] for x in a]==list(range(1,781))
 metrics[r]={'rows':len(a),'all_finite':all(math.isfinite(x['loss']) for x in a),'positive_lr_student_updates':sum(x['learning_rate']>0 and x['student_optimizer_delta']>0 for x in a),'teacher_direct_gradient_steps':sum(x['teacher_grad_non_none_count']>0 for x in a),'teacher_optimizer_change_steps':sum(x['teacher_optimizer_delta']>0 for x in a),'ema_events':sum(x['ema_update_applied'] for x in a),'loss_first':a[0]['loss'],'loss_last':a[-1]['loss'],'loss_mean_first50':statistics.mean(x['loss'] for x in a[:50]),'loss_mean_last50':statistics.mean(x['loss'] for x in a[-50:]),'sum_lr':sum(x['learning_rate'] for x in a)}
res={'scope':'R4 fixed-1024 three-model paired snapshot; subgroup analyses are diagnostic, not new benchmark scores','checks':{'unique_rows_per_role':2536,'identical_ids_prompts_labels':True},'results':results,'pairs':pairs,'subsets':subsets,'categories':cats,'metrics':metrics,'ema':{'half_life_updates':math.log(.5)/math.log(.95),'half_life_prompts_author':96*math.log(.5)/math.log(.95),'half_life_prompts_local':8*math.log(.5)/math.log(.95),'initial_coefficient_author':.95**65,'initial_coefficient_local':.95**780,'sample_clock_matched_alpha':1-.95**(1/12)},'source_sha256':hashes}
(OUT/'audit_data.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in res.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
