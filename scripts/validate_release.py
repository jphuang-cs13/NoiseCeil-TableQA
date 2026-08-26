#!/usr/bin/env python3
"""Offline scientific, provenance, privacy, and scope validation."""
import csv, gzip, hashlib, json, os, re
from decimal import Decimal as D, ROUND_HALF_UP
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GRID=(1,5,10,20,30,40,50)
RATES={'gpt-oss-20b':(D('0.02'),D('0.10')),'qwen3-32b':(D('0.29'),D('0.59')),'claude-haiku-4-5':(D('1.00'),D('5.00')),'gpt-4o':(D('2.50'),D('10.00'))}
def rows(path):
 with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def rounded(x,n):return str(D(str(x)).quantize(D('1.'+'0'*n),rounding=ROUND_HALF_UP))
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def gzip_content_sha(path):
 digest=hashlib.sha256()
 with gzip.open(ROOT/path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
 return digest.hexdigest()

trial=rows('artifacts/camera_ready/controlled_trials_author_resolved.csv'); assert len(trial)==84
score=rows('artifacts/camera_ready/score_nrr_frozen.csv'); assert len(score)==84
display=rows('artifacts/camera_ready/score_nrr_std_camera_ready.csv'); assert len(display)==252
t4=rows('artifacts/camera_ready/table4_author_resolved.csv'); assert len(t4)==84
trialmap={(r['Dataset'],r['Model'],int(r['K'])):r for r in trial}; scoremap={(r['Dataset'],r['Model'],int(r['K'])):r for r in score}; t4map={(r['Dataset'],r['Model'],int(r['K'])):r for r in t4}

# Complete controlled Score/SD/NRR display validation and cross-CSV consistency.
for r in display:
 if r['condition']=='bge_m3':continue
 key=(r['dataset'],r['reader'],int(r['K'])); title=r['condition'].title(); tr=trialmap[key]; sr=scoremap[key]; qr=t4map[key]
 mean=tr[f'Score ({title} Avg Mean)']; sd=tr[f'Standard deviation ({title})']; base=trialmap[(key[0],key[1],1)][f'Score ({title} Avg Mean)']; nrr=D(mean)/D(base)
 assert mean==sr[f'Score ({title} mean)']==qr[f'Score ({title} mean)']
 assert r['score_display']==rounded(mean,3) and r['std_display']==rounded(sd,3) and r['nrr_display']==rounded(nrr,3)

# Controlled Average Cost and CpS from preserved tokens and nominal rates.
for key,r in t4map.items():
 rin,rout=RATES[key[1]]
 for title in ('Soft','Hard'):
  cost=(D(r[f'Input Tokens ({title} Rotation Avg)'])*rin+D(r[f'Output Tokens ({title} Rotation Avg)'])*rout)/D(1000000)
  cps=cost/D(r[f'Score ({title} mean)'])
  assert abs(cost-D(r[f'{title} Avg_Cost($)']))<D('1e-15')
  assert abs(cps-D(r[f'CpS({title})']))<D('1e-15')

# All 96 controlled CpS cells actually displayed in the appendix.
cps_display=rows('artifacts/camera_ready/cps_displayed_camera_ready.csv'); assert len(cps_display)==96
for r in cps_display:
 key=(r['dataset'],r['reader'],int(r['K'])); title=r['condition'].title(); q=t4map[key]
 assert r['input_tokens_display'].replace(',','')==rounded(q[f'Input Tokens ({title} Rotation Avg)'],0)
 assert r['output_tokens_display'].replace(',','')==rounded(q[f'Output Tokens ({title} Rotation Avg)'],0)
 assert r['cps_display']==rounded(q[f'CpS({title})'],4)

cps=rows('artifacts/camera_ready/cps_recomputed_release_time.csv'); assert len(cps)==252 and all(r['matches_camera_ready_4dp']=='true' for r in cps)

# NRR/Kc and manuscript matrix, with no monotonicity assumption.
matrix=rows('artifacts/camera_ready/retrieval_spec_matrix.csv')
expected={('E2E-WTQ','qwen3-32b'):('10','20'),('E2E-WTQ','gpt-oss-20b'):('>=50','>=50'),('E2E-WTQ','claude-haiku-4-5'):('30','>=50'),('E2E-WTQ','gpt-4o'):('>=50','>=50'),('OTTQA','qwen3-32b'):('>=50','20'),('OTTQA','gpt-oss-20b'):('>=50','>=50'),('OTTQA','claude-haiku-4-5'):('>=50','>=50'),('OTTQA','gpt-4o'):('>=50','40'),('FeTaQA','qwen3-32b'):('1','20'),('FeTaQA','gpt-oss-20b'):('1','>=50'),('FeTaQA','claude-haiku-4-5'):('1','5'),('FeTaQA','gpt-4o'):('5','>=50')}
assert {(r['dataset'],r['reader']):(r['Kc_hard'],r['Kc_soft']) for r in matrix}==expected
for (dataset,reader),(eh,es) in expected.items():
 for title,want in [('Hard',eh),('Soft',es)]:
  base=D(scoremap[(dataset,reader,1)][f'Score ({title} mean)']); passing=[k for k in GRID if D(scoremap[(dataset,reader,k)][f'Score ({title} mean)'])/base>=D('.9')]; got='>=50' if max(passing)==50 else str(max(passing)); assert got==want

# Frozen BGE-M3 publication-facing values must remain unchanged.
trial_cols=['Dataset','Model','K','Score (BGE-m3) 1st','Score (BGE-m3) 2nd','Score (BGE-m3) 3rd','Score (BGE-m3 Avg mean)','Standard deviation (BGE-m3)','SEM (BGE-m3)']
t4_cols=['Dataset','Model','K','Score (BGE-m3)','Input Tokens (BGE-m3)','Output Tokens (BGE-m3)','BGE-m3 Avg_Cost($)','CpS(BGE-m3)']
digest=lambda data,cols:hashlib.sha256(json.dumps([[r[c] for c in cols] for r in data],separators=(',',':')).encode()).hexdigest()
assert digest(trial,trial_cols)=='77461b713b754acab7dfd99f4e20491b7b333f01f292fd260907cc878d3c3622'
assert digest(t4,t4_cols)=='ccaed38d42d768fa922b575e52142621de6e8ec8e84cd6bcdfe86a9d798964b8'

# Final figure hashes, Figure 5 aggregate chain, and human-validation invariants.
assert sha('artifacts/figures/table1_new_impact_gap_faceted.pdf')=='cd0327ccd688f923df234d49fe0eafd50d253574be64672d7311849a1c1c55b8'
assert sha('artifacts/figures/table4_new_hard_negative_tax.pdf')=='3d9fbff67c4f944ccf60bc328beffa37cd3c0182f7606cd2b4ce54e0ef14838e'
fig5=rows('artifacts/camera_ready/figure5_transition_camera_ready.csv'); assert len(fig5)==72
for r in fig5:assert sum(int(r[k]) for k in ['both_success_count','both_fail_count','perfect_to_fail_count','perfect_from_fail_count'])==int(r['total'])
for d,n in {'e2ewtq':717,'feta':6000,'ottqa':6642}.items():assert {int(r['total']) for r in fig5 if r['dataset']==d and r['target_k']=='50' and r['negative']=='hard'}=={n}
human=rows('human_validation/results/final_human_judge_metrics_v2.csv'); overall=next(r for r in human if r['dataset']=='overall')
assert (overall['binary_eligible_n'],overall['final_uncertain_excluded_n'])==('598','2')
assert abs(float(overall['raw_human_judge_agreement'])-0.862876254180602)<1e-15 and abs(float(overall['cohens_kappa'])-0.7229761030450257)<1e-15

# Six frozen dataset snapshot identifiers.
datasets=rows('docs/DATASET_VERSION_MANIFEST.csv'); assert len(datasets)==6 and all(int(r['row_count'])>0 and re.fullmatch(r'[0-9a-f]{64}',r['sha256']) for r in datasets)

# Public scope, provenance wording, and privacy/security scans.
for forbidden in [
 'artifacts/semantic_judge/verdict_manifest.csv.gz',
 'artifacts/semantic_judge/source_files.csv',
 'artifacts/legacy/recovered_candidates/table1-1_recovered_candidate_NOT_FINAL.csv',
 'outputs',
 'llm_logs',
 'human_validation/annotation_items.csv',
 'human_validation/annotation_template.csv',
 'human_validation/sampling_result.json',
]:
 assert not (ROOT/forbidden).exists()
assert not (ROOT/'artifacts/distractors/injection_manifest.csv').exists()
assert gzip_content_sha('artifacts/distractors/injection_manifest.csv.gz')=='b53e2221d0ce61ca90810a92b3151a75d9f7ce50a8a33624e4f5fb7d40c1f4cf'
prov=(ROOT/'PROVENANCE_LIMITATIONS.md').read_text(); assert len(re.findall(r'^\d+\. \*\*',prov,flags=re.M))==2
for stale in ['116 Score','116 controlled','wrong-version','Figure 5 verdict','stale Qwen','historical LaTeX']:
 assert stale.lower() not in prov.lower()
for figure_script in ['scripts/figures/plot_table1_new_impact_gap.py','scripts/figures/plot_table4_new_hard_negative_tax.py']:
 text=(ROOT/figure_script).read_text().lower()
 assert 'simulate' not in text and 'simulated' not in text
root_git=ROOT/'.git'
assert not root_git.exists() or root_git.is_dir(),f'unexpected root Git metadata file: {root_git}'
credential_pattern=re.compile(
 rb'(?i)(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|'
 rb'xox[baprs]-[A-Za-z0-9-]{10,}|BEGIN (?:RSA |OPENSSH |EC |PGP )?PRIVATE KEY)'
)
email_pattern=re.compile(rb'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b')
absolute_path_pattern=re.compile(
 rb'(?:/' + b'Users/|/' + b'home/|[A-Za-z]:[\\\\/]' + b'Users[\\\\/])'
)
def scan_sensitive(data,path):
 assert not absolute_path_pattern.search(data),f'absolute user path: {path}'
 assert not credential_pattern.search(data),f'credential-like token: {path}'
 assert not email_pattern.search(data),f'email address: {path}'
for dirpath,dirnames,filenames in os.walk(ROOT):
 current=Path(dirpath)
 for generated in ['.pytest_cache','.venv','__pycache__']:
  if generated in dirnames:dirnames.remove(generated)
 if current==ROOT:
  # Root .git metadata is expected after repository initialization;
  # nested Git metadata remains forbidden.
  if '.git' in dirnames:dirnames.remove('.git')
 else:
  assert '.git' not in dirnames and '.git' not in filenames,f'nested Git metadata: {current.relative_to(ROOT)}'
 for name in filenames:
  p=current/name; rel=p.relative_to(ROOT); parts=set(rel.parts)
  assert p.stat().st_size<100*1024*1024,f'file exceeds ordinary GitHub blob limit: {rel}'
  assert p.name not in {'.DS_Store','.env'} and p.suffix.lower() not in {'.xls','.xlsx','.pyc'}
  if p.suffix.lower() not in {'.pdf','.png','.gz'}:
   scan_sensitive(p.read_bytes(),rel)

with gzip.open(ROOT/'artifacts/distractors/injection_manifest.csv.gz','rb') as f:
 for line_number,line in enumerate(f,1):
  scan_sensitive(line,f'artifacts/distractors/injection_manifest.csv.gz:{line_number}')

print('PASS: 0 controlled Score/SD/NRR mismatches; 0/96 displayed CpS mismatches; 24/24 Kc; final Figure 2/4 hashes; frozen Figure 5, human, dataset, BGE, scope, path, and secret checks.')
