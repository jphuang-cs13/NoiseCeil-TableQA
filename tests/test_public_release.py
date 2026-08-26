import csv, gzip, hashlib, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).parents[1]
def rows(path):
 with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def test_final_camera_ready_row_counts():
 assert len(rows('artifacts/camera_ready/controlled_trials_author_resolved.csv'))==84
 assert len(rows('artifacts/camera_ready/score_nrr_std_camera_ready.csv'))==252
 assert len(rows('artifacts/camera_ready/cps_displayed_camera_ready.csv'))==96
 assert len(rows('artifacts/camera_ready/table4_author_resolved.csv'))==84

def test_matrix_all_24_cells():
 x=rows('artifacts/camera_ready/retrieval_spec_matrix.csv')
 assert len(x)==12 and sum(bool(r['Kc_hard'])+bool(r['Kc_soft']) for r in x)==24

def test_cps_reconciliation():
 x=rows('artifacts/camera_ready/cps_recomputed_release_time.csv')
 assert len(x)==252 and all(r['matches_camera_ready_4dp']=='true' for r in x)

def test_final_figure_hashes():
 expected={'table1_new_impact_gap_faceted.pdf':'cd0327ccd688f923df234d49fe0eafd50d253574be64672d7311849a1c1c55b8','table4_new_hard_negative_tax.pdf':'3d9fbff67c4f944ccf60bc328beffa37cd3c0182f7606cd2b4ce54e0ef14838e'}
 for name,want in expected.items():assert hashlib.sha256((ROOT/'artifacts/figures'/name).read_bytes()).hexdigest()==want

def test_human_results_unchanged():
 x=rows('human_validation/results/final_human_judge_metrics_v2.csv')
 assert any(r.get('dataset')=='overall' and r.get('binary_eligible_n')=='598' and r.get('final_uncertain_excluded_n')=='2' for r in x)

def test_per_query_verdict_archive_excluded():
 assert not (ROOT/'artifacts/semantic_judge/verdict_manifest.csv.gz').exists()
 assert not (ROOT/'artifacts/semantic_judge/source_files.csv').exists()

def test_six_dataset_snapshots_retained():
 assert len(rows('docs/DATASET_VERSION_MANIFEST.csv'))==6

def test_compressed_injection_manifest_preserves_schema_and_count():
 path=ROOT/'artifacts/distractors/injection_manifest.csv.gz'
 assert path.exists() and not (ROOT/'artifacts/distractors/injection_manifest.csv').exists()
 with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as f:
  manifest=csv.reader(f); header=next(manifest); count=sum(1 for _ in manifest)
 assert header==['dataset','source_row_index','query_id','K','rotation','negative_type','gold_table_ids','distractor_table_ids','ordered_table_ids','gold_zero_based_positions','source_condition_identifier','source_dataset_version']
 assert count==195932

def test_release_validator_allows_root_git_metadata():
 result=subprocess.run([sys.executable,ROOT/'scripts/validate_release.py'],capture_output=True,text=True)
 assert result.returncode==0,result.stderr

def test_release_validator_rejects_nested_git_metadata(tmp_path):
 nested=ROOT/'some/subdirectory/.git'
 nested.mkdir(parents=True)
 try:
  result=subprocess.run([sys.executable,ROOT/'scripts/validate_release.py'],capture_output=True,text=True)
  assert result.returncode!=0
  assert 'nested Git metadata' in result.stderr
 finally:
  nested.rmdir()
  nested.parent.rmdir()
  nested.parent.parent.rmdir()
