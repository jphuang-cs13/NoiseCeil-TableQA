"""Create and freeze blinded human-validation samples; performs no annotation."""
from __future__ import annotations
import csv,hashlib,json,random
from collections import Counter,defaultdict
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"outputs";HV=ROOT/"human_validation"
INDEX=OUT/"record_index.csv";RAW=OUT/"official_metric_scores.csv";SENS=OUT/"extraction_sensitivity_scores.csv"
PRIMARY_SEED=42;DIAGNOSTIC_SEED=42;ORDER_SEED=314159
DATASETS=("e2ewtq","feta","ottqa");Q={"e2ewtq":239,"feta":2000,"ottqa":2214}
PRIMARY_FIELDS=["sample_id","sample_membership","dataset","query_id","question","reference_answer","candidate_answer","run","reader_model","condition","K","rotation","negative_type","existing_judge_score","raw_official_denotation_correct","extracted_official_denotation_correct","raw_official_em","extracted_official_em","raw_official_f1","extracted_official_f1"]
DIAG_FIELDS=PRIMARY_FIELDS+["diagnostic_stratum"]
ANNOTATION_FIELDS=["sample_id","question","reference_answer","candidate_answer"]

@dataclass(frozen=True)
class SelectionRecord:
 dataset:str;query_id:str;question:str;reference_answer:str;candidate_answer:str;run:str;reader_model:str;condition:str;K:str;rotation:str;negative_type:str
 @property
 def question_identity(self):return (self.dataset,self.query_id,self.question,self.reference_answer)
 @property
 def record_identity(self):return (*self.question_identity,self.run,self.reader_model,self.condition)

def project_selection_row(row):
 """Expose no judge/metric/extraction outcome to primary selection."""
 return SelectionRecord(**{k:row[k] for k in SelectionRecord.__dataclass_fields__})
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
 return h.hexdigest()

def select_primary():
 identities={d:[] for d in DATASETS};seen={d:set() for d in DATASETS}
 with INDEX.open(newline="",encoding="utf8") as f:
  for raw in csv.DictReader(f):
   row=project_selection_row(raw);identity=row.question_identity
   if identity not in seen[row.dataset]:seen[row.dataset].add(identity);identities[row.dataset].append(identity)
 for d in DATASETS:
  if len(identities[d])!=Q[d]:raise RuntimeError(f"question population mismatch {d}: {len(identities[d])}")
 rng=random.Random(PRIMARY_SEED);targets={};order=[]
 for d in DATASETS:
  chosen=rng.sample(identities[d],200)
  for identity in chosen:targets[identity]=rng.randrange(444);order.append(identity)
 occurrence=Counter();selected={}
 with INDEX.open(newline="",encoding="utf8") as f:
  for raw in csv.DictReader(f):
   row=project_selection_row(raw);identity=row.question_identity
   if identity in targets:
    if occurrence[identity]==targets[identity]:selected[identity]=row
    occurrence[identity]+=1
 if len(selected)!=600 or any(occurrence[x]!=444 for x in targets):raise RuntimeError("primary record selection incomplete")
 return [selected[x] for x in order]

def enrich_primary(selected):
 wanted={x.record_identity:x for x in selected};rows={}
 with INDEX.open(newline="",encoding="utf8") as fi,RAW.open(newline="",encoding="utf8") as fr:
  for canonical,raw in zip(csv.DictReader(fi),csv.DictReader(fr),strict=True):
   key=(canonical["dataset"],canonical["query_id"],canonical["question"],canonical["reference_answer"],canonical["run"],canonical["reader_model"],canonical["condition"])
   if key in wanted:
    s=wanted[key];row=asdict(s);row.update(existing_judge_score=canonical["existing_judge_score"],raw_official_denotation_correct=raw["official_denotation_correct"],extracted_official_denotation_correct="",raw_official_em=raw["official_em"],extracted_official_em="",raw_official_f1=raw["official_f1"],extracted_official_f1="");rows[key]=row
 sens_wanted={k for k in wanted if k[0] in ("e2ewtq","ottqa")}
 with SENS.open(newline="",encoding="utf8") as f:
  for r in csv.DictReader(f):
   key=(r["dataset"],r["query_id"],r["question"],r["reference_answer"],r["run"],r["reader_model"],r["condition"])
   if key in sens_wanted:
    rows[key]["extracted_official_denotation_correct"]=r["extracted_official_denotation_correct"];rows[key]["extracted_official_em"]=r["extracted_official_em"];rows[key]["extracted_official_f1"]=r["extracted_official_f1"]
 if len(rows)!=600:raise RuntimeError(f"primary enrichment incomplete: {len(rows)}")
 return [rows[x.record_identity] for x in selected]

def diagnostic_cell(row):
 if row["existing_judge_score"]!="1":return None
 if row["dataset"]=="e2ewtq" and row["raw_official_denotation_correct"]=="0":return "E2E_EXTRACTION_RESCUED" if row["extracted_official_denotation_correct"]=="1" else "E2E_PERSISTENT_DISAGREEMENT"
 if row["dataset"]=="ottqa" and row["raw_official_em"]=="0":return "OTT_EXTRACTION_RESCUED" if row["extracted_official_em"]=="1" else "OTT_PERSISTENT_DISAGREEMENT"
 return None
def diag_qid(r):return (r["dataset"],r["query_id"],r["question"],r["reference_answer"])
def diag_record_id(r):return (*diag_qid(r),r["run"],r["reader_model"],r["condition"])
def select_diagnostic(primary):
 primary_records={x.record_identity for x in primary};eligible=defaultdict(list);seen=defaultdict(set)
 with SENS.open(newline="",encoding="utf8") as f:
  for r in csv.DictReader(f):
   cell=diagnostic_cell(r);qid=diag_qid(r)
   if cell and diag_record_id(r) not in primary_records and qid not in seen[cell]:seen[cell].add(qid);eligible[cell].append(qid)
 cells=("E2E_EXTRACTION_RESCUED","E2E_PERSISTENT_DISAGREEMENT","OTT_EXTRACTION_RESCUED","OTT_PERSISTENT_DISAGREEMENT")
 availability={c:len(eligible[c]) for c in cells}
 if any(availability[c]<50 for c in cells):raise RuntimeError(f"insufficient diagnostic distinct-question eligibility: {availability}")
 rng=random.Random(DIAGNOSTIC_SEED);chosen={c:set(rng.sample(eligible[c],50)) for c in cells};reservoir={};counts=Counter()
 with SENS.open(newline="",encoding="utf8") as f:
  for r in csv.DictReader(f):
   cell=diagnostic_cell(r)
   if cell and diag_record_id(r) not in primary_records and diag_qid(r) in chosen[cell]:
    key=(cell,diag_qid(r));counts[key]+=1
    if counts[key]==1 or rng.randrange(counts[key])==0:reservoir[key]=dict(r)
 if len(reservoir)!=200:raise RuntimeError(f"diagnostic selection incomplete: {len(reservoir)}")
 rows=[]
 for cell in cells:
  for qid in sorted(chosen[cell]):
   r=reservoir[(cell,qid)];rows.append({"dataset":r["dataset"],"query_id":r["query_id"],"question":r["question"],"reference_answer":r["reference_answer"],"candidate_answer":r["original_candidate"],"run":r["run"],"reader_model":r["reader_model"],"condition":r["condition"],"K":r["K"],"rotation":r["rotation"],"negative_type":r["negative_type"],"existing_judge_score":r["existing_judge_score"],"raw_official_denotation_correct":r["raw_official_denotation_correct"],"extracted_official_denotation_correct":r["extracted_official_denotation_correct"],"raw_official_em":r["raw_official_em"],"extracted_official_em":r["extracted_official_em"],"raw_official_f1":r["raw_official_f1"],"extracted_official_f1":r["extracted_official_f1"],"diagnostic_stratum":cell})
 return rows,availability

def write_csv(path,fields,rows):
 with path.open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 HV.mkdir(exist_ok=True);primary_surface=select_primary();primary=enrich_primary(primary_surface);diagnostic,availability=select_diagnostic(primary_surface)
 combined=[("PRIMARY_REPRESENTATIVE",r) for r in primary]+[("DIAGNOSTIC_ONLY_NONREPRESENTATIVE",r) for r in diagnostic]
 random.Random(ORDER_SEED).shuffle(combined);id_by_record={}
 for i,(membership,row) in enumerate(combined,1):id_by_record[(membership,row["dataset"],row["query_id"],row["run"],row["reader_model"],row["condition"])]=f"HV{i:06d}"
 primary_rows=[]
 for r in primary:
  x={"sample_id":id_by_record[("PRIMARY_REPRESENTATIVE",r["dataset"],r["query_id"],r["run"],r["reader_model"],r["condition"])],"sample_membership":"PRIMARY_REPRESENTATIVE",**r};primary_rows.append(x)
 diag_rows=[]
 for r in diagnostic:
  x={"sample_id":id_by_record[("DIAGNOSTIC_ONLY_NONREPRESENTATIVE",r["dataset"],r["query_id"],r["run"],r["reader_model"],r["condition"])],"sample_membership":"DIAGNOSTIC_ONLY_NONREPRESENTATIVE",**r};diag_rows.append(x)
 write_csv(HV/"primary_sample_manifest.csv",PRIMARY_FIELDS,primary_rows);write_csv(HV/"diagnostic_sample_manifest.csv",DIAG_FIELDS,diag_rows)
 annotation=[]
 for membership,r in combined:
  sid=id_by_record[(membership,r["dataset"],r["query_id"],r["run"],r["reader_model"],r["condition"])]
  annotation.append({"sample_id":sid,"question":r["question"],"reference_answer":r["reference_answer"],"candidate_answer":r["candidate_answer"]})
 write_csv(HV/"annotation_items.csv",ANNOTATION_FIELDS,annotation);write_csv(HV/"annotation_template.csv",["sample_id","annotator_label","optional_note"],[{"sample_id":x["sample_id"],"annotator_label":"","optional_note":""} for x in annotation])
 overlap=len({x.record_identity for x in primary_surface}&{(r["dataset"],r["query_id"],r["question"],r["reference_answer"],r["run"],r["reader_model"],r["condition"]) for r in diagnostic})
 result={"primary_counts":dict(Counter(r["dataset"] for r in primary_rows)),"diagnostic_counts":dict(Counter(r["diagnostic_stratum"] for r in diag_rows)),"diagnostic_distinct_question_availability_excluding_exact_primary_items":availability,"overlap_count":overlap,"unique_annotation_workload":len(annotation)}
 (HV/"sampling_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
