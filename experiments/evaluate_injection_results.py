import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Ensure project root is on sys.path for `app` imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.llm_metrics import LLMBasedMetrics



def parse_injection_filename(name: str) -> Tuple[int, str, str]:
    """
    Parse injection filename to extract recall@K (K), rotation, and negative type.

    Examples:
    - real_result_1.jsonl -> (1, "real_result", "real_result")
    - real_result_50.jsonl -> (50, "real_result", "real_result")
    - 1_perfect.jsonl -> (1, "perfect", "perfect")
    - 2_end_hard.jsonl -> (2, "end", "hard")
    - 4_middle_soft.jsonl -> (4, "middle", "soft")
    """
    base = name.rsplit(".", 1)[0]
    
    # Handle new real_result format
    if base.startswith("real_result_"):
        try:
            k = int(base.replace("real_result_", ""))
            return k, "real_result", "real_result"
        except ValueError:
            pass
    
    # Handle old formats
    parts = base.split("_")
    if len(parts) == 1 and parts[0] == "1":
        # unlikely format; prefer explicit perfect
        return 1, "", ""
    if base == "1_perfect":
        return 1, "perfect", "perfect"
    try:
        k = int(parts[0])
    except Exception:
        k = 0
    rotation = parts[1] if len(parts) > 1 else ""
    negative = parts[2] if len(parts) > 2 else ""
    return k, rotation, negative


def load_reasoning_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                continue
    return records


def load_existing_eval(eval_path: Path) -> Optional[Dict]:
    if not eval_path.exists():
        return None
    try:
        with eval_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_eval(eval_path: Path, payload: Dict) -> None:
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_summary_csv(csv_path: Path, model: str, rotation: str, negative: str, k: int, accuracy_rate: float):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Models", "Rotation", "Negatives", "Recall@K", "Score"]
    exists = csv_path.exists()
    rows = []
    if exists:
        try:
            with csv_path.open("r", encoding="utf-8") as rf:
                reader = csv.reader(rf)
                rows = list(reader)
        except Exception:
            rows = []
    # Ensure header present
    if not rows or rows[0] != header:
        rows = [header]
    
    # Check if row with same combination already exists
    new_row = [model, rotation, negative, str(k), f"{accuracy_rate:.4f}"]
    row_found = False
    for i in range(1, len(rows)):  # Skip header at index 0
        if (rows[i][0] == model and 
            rows[i][1] == rotation and 
            rows[i][2] == negative and 
            rows[i][3] == str(k)):
            # Update existing row with new score
            rows[i] = new_row
            row_found = True
            break
    
    # If not found, append new row
    if not row_found:
        rows.append(new_row)
    
    with csv_path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerows(rows)


def switch_provider_env(env_file: Optional[str]) -> None:
    if not env_file:
        return
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)


def evaluate_injection_file(
    dataset: str,
    model_dir: Path,
    injection_file: Path,
    eval_root: Path,
    alt_env_files: List[str],
    eval_provider: Optional[str] = None,
    eval_model: Optional[str] = None,
):
    model_name = model_dir.name
    k, rotation, negative = parse_injection_filename(injection_file.name)

    # Paths
    eval_dir = eval_root / model_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / injection_file.name
    summary_csv = eval_root / ("summary_real_result.csv" if model_name.endswith("_real_result") else "summary.csv")

    # Load reasoning inputs
    inputs = load_reasoning_records(injection_file)
    total_questions = len(inputs)
    processed_results: List[Dict] = []
    processed_ids = set()

    existing = load_existing_eval(out_path)
    if existing and isinstance(existing.get("records"), list):
        for r in existing["records"]:
            processed_results.append(r)
            qid = r.get("query_id")
            if qid is not None:
                processed_ids.add(qid)
    print(f"[START] {model_name} :: {injection_file.name} | total={total_questions} processed={len(processed_results)}")

    # Initialize metrics; will be re-created when switching env
    metrics = LLMBasedMetrics(provider=eval_provider, model=eval_model, temperature=0.0, max_retries=2)

    # Iterate records in original order
    for idx, rec in enumerate(inputs, start=1):
        qid = rec.get("query_id")
        if qid in processed_ids:
            continue

        print(f"  [RUN] {model_name} :: {injection_file.name} | {idx}/{total_questions} | qid={qid}")

        question = str(rec.get("question", ""))
        gold = str(rec.get("gold_answer", ""))
        user = str(rec.get("llm_response", ""))
        req_toks = rec.get("request_tokens")
        resp_toks = rec.get("response_tokens")
        exec_time = rec.get("execution_time")

        result_score = None
        try:
            
            result_score = metrics.llm_exact_match(gold, user, question=question)
        except Exception as e:
            print(f"    [Error] Metric evaluation failed for qid={qid}: {e}")
            result_score = None

        if result_score not in (0, 1):

            for env_file in alt_env_files:
                try:
                    print(f"    [RETRY] switching to {env_file} for qid={qid}")
                    switch_provider_env(env_file)
                    metrics = LLMBasedMetrics(provider=eval_provider, model=eval_model, temperature=0.0, max_retries=2)
                    result_score = metrics.llm_exact_match(gold, user, question=question)
                    if result_score in (0, 1):
                        print(f"    [RECOVERED] provider={env_file} qid={qid} score={result_score}")
                        break
                except Exception:
                    result_score = None

            # Restore default .env
            switch_provider_env(".env")
            metrics = LLMBasedMetrics(provider=eval_provider, model=eval_model, temperature=0.0, max_retries=2)

        if result_score not in (0, 1):
            # As a last resort, treat as incorrect (conservative)
            result_score = 0

        # Build result record
        result_record = {
            "query_id": qid,
            "score": int(result_score),
            "request_tokens": req_toks,
            "response_tokens": resp_toks,
            "execution_time": exec_time,
        }
        
        processed_results.append(result_record)
        processed_ids.add(qid)

        # Incremental save for resume
        partial_accuracy = (
            sum(1 for r in processed_results if r.get("score") == 1) / max(1, len(processed_results))
        )
        payload = {
            "model_name": model_name,
            "rotation": rotation,
            "negative": negative,
            "total_questions": total_questions,
            "recall_at_k": k,
            "accuracy_rate": round(partial_accuracy, 6),
            "records": processed_results,
        }
        save_eval(out_path, payload)
    print(f"[DONE] {model_name} :: {injection_file.name} | total={total_questions} processed={len(processed_results)}")

    # Final accuracy across all questions
    final_accuracy = (
        sum(1 for r in processed_results if r.get("score") == 1) / max(1, len(processed_results))
    )
    final_payload = {
        "model_name": model_name,
        "rotation": rotation,
        "negative": negative,
        "total_questions": total_questions,
        "recall_at_k": k,
        "accuracy_rate": round(final_accuracy, 6),
        "records": processed_results,
    }
    save_eval(out_path, final_payload)

    # Update summary CSV
    update_summary_csv(summary_csv, model_name, rotation, negative, k, final_accuracy)


def main():

    parser = argparse.ArgumentParser(description="Evaluate injection experiment results using LLM-based semantic EM.")
    parser.add_argument("dataset", help="Dataset name (e.g., e2ewtq, bird, ottqa)")
    parser.add_argument("--injection-file", dest="injection_file", default=None, help="Optional single injection filename to evaluate (e.g., 1_perfect.jsonl, 2_end_hard.jsonl)")
    parser.add_argument("--model-folder", dest="model_folder", default=None, help="Optional model folder name to evaluate (e.g., my_model)")
    parser.add_argument("--eval-provider", default=None, help="LLM provider for evaluation")
    parser.add_argument("--eval-model", default=None, help="LLM model for evaluation")
    args = parser.parse_args()

    dataset = args.dataset
    injection_file_filter = args.injection_file
    model_folder_filter = args.model_folder

    # Base paths
    root = Path(__file__).resolve().parents[1]
    reasoning_root = root / "reasoning_results" / dataset
    eval_root = root / "evaluations" / f"{dataset}_test" / "injection"

    if not reasoning_root.exists():
        print(f"Reasoning results folder not found: {reasoning_root}")
        return 1

    # Alternate env providers for retries
    alt_env_files = [".env.groq", ".env.openrouter", ".env.fireworks"]

    # Iterate models
    if model_folder_filter:
        model_dirs = [reasoning_root / model_folder_filter]
        if not model_dirs[0].exists() or not model_dirs[0].is_dir():
            print(f"Model folder not found: {model_dirs[0]}")
            return 1
    else:
        model_dirs = [p for p in reasoning_root.iterdir() if p.is_dir()]
        if not model_dirs:
            print(f"No model folders found under {reasoning_root}")
            return 1

    for model_dir in model_dirs:
        # Injection files
        files = sorted([p for p in model_dir.glob("*.jsonl")])
        if injection_file_filter:
            files = [p for p in files if p.name == injection_file_filter]
            if not files:
                print(f"[Skip] {model_dir.name}: no file {injection_file_filter}")
                continue

        for inj in files:
            try:
                evaluate_injection_file(dataset, model_dir, inj, eval_root, alt_env_files, args.eval_provider, args.eval_model)
            except KeyboardInterrupt:
                print("Interrupted. Progress saved.")
                return 130
            except Exception as e:
                print(f"Error evaluating {inj}: {e}")
                # Continue to next file
                continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
