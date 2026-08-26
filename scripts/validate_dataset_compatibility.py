#!/usr/bin/env python3
"""Release-time utility: validate locally acquired normalized snapshots and IDs."""
import argparse, csv, hashlib, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True,help='Root containing dataset/query.jsonl and dataset/table.jsonl'); ap.add_argument('--manifest',type=Path,default=Path('docs/DATASET_VERSION_MANIFEST.csv')); a=ap.parse_args()
    checked=0
    for r in csv.DictReader(a.manifest.open(encoding='utf-8-sig')):
        dataset=r['dataset']; role=r['normalized_file_role']; name='query.jsonl' if role=='queries' else 'table.jsonl'; p=a.data_root/dataset/name
        raw=p.read_bytes(); assert hashlib.sha256(raw).hexdigest()==r['sha256'],f'hash mismatch: {dataset}/{name}'
        lines=raw.splitlines(); assert len(lines)==int(r['row_count']),f'row-count mismatch: {dataset}/{name}'
        ids=set()
        for line in lines:
            x=json.loads(line); candidates=('query_id','id') if role=='queries' else ('table_id','id')
            key=next((k for k in candidates if k in x),None); assert key,f'missing stable ID: {dataset}/{name}'; assert str(x[key]) not in ids,f'duplicate ID: {dataset}/{name}'; ids.add(str(x[key]))
        checked+=1
    print(f'PASS: {checked} normalized snapshot files match frozen hashes, row counts, and stable-ID uniqueness.')
if __name__=='__main__': main()
