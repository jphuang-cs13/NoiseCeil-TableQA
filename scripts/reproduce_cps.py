#!/usr/bin/env python3
"""Recompute camera-ready CpS from the author-maintained pricing record."""
import argparse, csv
from decimal import Decimal as D, ROUND_HALF_UP
from pathlib import Path

MODEL_KEYS={'gpt-oss-20b':'GPT-OSS-20b','qwen3-32b':'Qwen-3-32b','claude-haiku-4-5':'Claude-Haiku-4.5','gpt-4o':'GPT-4o'}
def rounded4(x): return x.quantize(D('0.0001'),rounding=ROUND_HALF_UP)
def display_interval(text):
 x=D(text); places=len(text.split('.')[1]) if '.' in text else 0; half=D(5)*(D(10)**D(-(places+1))); return x-half,x+half
def load_pricing(path):
 """Parse the release's deliberately small, fixed YAML pricing schema."""
 lines=path.read_text().splitlines(); cfg={'rates':{}}; current=None
 for raw in lines:
  line=raw.strip()
  if not line or line.startswith('#') or line=='rates:': continue
  key,value=(x.strip() for x in line.split(':',1))
  if raw.startswith('  ') and not raw.startswith('    ') and not value:
   current=key; cfg['rates'][current]={}
  elif raw.startswith('    ') and current:
   cfg['rates'][current][key]=D(value) if key in {'input','output'} else value
  else:
   cfg[key]=value
 return cfg
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--pricing',type=Path,default=Path('configs/camera_ready_pricing.yaml')); ap.add_argument('--published',type=Path,default=Path('artifacts/camera_ready/cps_published_camera_ready.csv')); ap.add_argument('--output',type=Path,default=Path('artifacts/camera_ready/cps_recomputed_release_time.csv')); a=ap.parse_args()
 cfg=load_pricing(a.pricing); assert cfg['currency']=='USD' and cfg['unit']=='per_1m_tokens'
 rows=list(csv.DictReader(a.published.open())); out=[]
 for r in rows:
  rate=cfg['rates'][MODEL_KEYS[r['reader']]]; inp=D(str(rate['input'])); outp=D(str(rate['output']))
  il,ih=display_interval(r['input_tokens_display']); ol,oh=display_interval(r['output_tokens_display']); sl,sh=display_interval(r['score_display'])
  average_cost=(D(r['input_tokens_display'])*inp+D(r['output_tokens_display'])*outp)/D(1000000); cps=average_cost/D(r['score_display']); cost_lo=(il*inp+ol*outp)/D(1000000); cost_hi=(ih*inp+oh*outp)/D(1000000); cps_lo=cost_lo/sh; cps_hi=cost_hi/sl; published=D(r['published_cps']); target=rounded4(published); target_lo,target_hi=target-D('0.00005'),target+D('0.00005'); ok=max(cps_lo,target_lo)<min(cps_hi,target_hi)
  out.append({**{k:r[k] for k in ['dataset','reader','K','condition']},'input_rate_usd_per_1m':str(inp),'output_rate_usd_per_1m':str(outp),'recomputed_average_cost_usd':f'{average_cost:.12f}','recomputed_cps_point_usd':f'{cps:.12f}','recomputed_cps_low_from_display_rounding':f'{cps_lo:.12f}','recomputed_cps_high_from_display_rounding':f'{cps_hi:.12f}','published_cps':r['published_cps'],'published_cps_4dp':f'{target:.4f}','matches_camera_ready_4dp':str(ok).lower()})
 if len(out)!=252 or not all(r['matches_camera_ready_4dp']=='true' for r in out): raise SystemExit(f"FAIL: {sum(r['matches_camera_ready_4dp']=='true' for r in out)}/{len(out)} CpS cells match at four decimals")
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=out[0]); w.writeheader(); w.writerows(out)
 print('PASS: 252/252 camera-ready CpS cells reproduce at four decimal places.')
if __name__=='__main__': main()
