#!/usr/bin/env python3
"""
Error Analysis Stage 3: LLM-assisted diagnosis + deterministic final labeling.

Final error labels are assigned by fixed deterministic rules in this order:
1) Distractor Extraction (exact normalized substring match to distractor cell)
2) Premature Refusal (explicit refusal/insufficient-information phrase)
3) Reasoning Hallucination (residual category)

The LLM pass remains auxiliary diagnostics only.
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()


DE_LABEL = "Distractor Extraction"
PR_LABEL = "Premature Refusal"
RH_LABEL = "Reasoning Hallucination"

CLASSIFICATION_POLICY = "rule_order_v2"

LABEL_SOURCE_DISTRACTOR = "distractor_specific_cell_rule"
LABEL_SOURCE_REFUSAL = "explicit_refusal_rule"
LABEL_SOURCE_RESIDUAL = "residual_reasoning_rule"
# Legacy migration compatibility; not part of the official rule_order_v2
# classification pipeline or any released paper-facing result.


REFUSAL_PATTERNS = [
    "information is not available",
    "not available in the context",
    "not available in the provided",
    "not available in the tables",
    "isn't available",
    "is not available",
    "cannot determine",
    "cannot be determined",
    "unable to determine",
    "insufficient information",
    "not enough information",
    "cannot answer",
    "unable to answer",
    "cannot help with that",
    "can't help with that",
    "provided information does not",
    "cannot be answered from the given",
    "cannot be answered based on the given",
    "cannot be answered with the provided",
    "the provided tables do not contain",
    "none of the provided tables",
    "not possible to determine",
    "cannot find",
    "does not include information",
    "do not include information",
    "not specified in the provided",
    "additional external sources would be required",
]


ERROR_FILE_RE = re.compile(
    r"^(?P<model>.+)_K(?P<k>\d+)_(?P<negative>hard|soft)_(?P<category>both_success|both_fail|perfect_to_fail|perfect_from_fail)_errors\.json$"
)

DATASET_ALIASES = {
    "e2e-wtq": "e2ewtq",
    "e2e_wtq": "e2ewtq",
    "e2ewtq": "e2ewtq",
    "feta": "feta",
    "fetaqa": "feta",
    "feta-qa": "feta",
    "ott-qa": "ottqa",
    "ott_qa": "ottqa",
    "ottqa": "ottqa",
}


@dataclass
class SamplingInfo:
    sample_size: Optional[int]
    total_available_cases: int
    seed: int
    sampling_method: str


def normalize_dataset_name(dataset: str) -> str:
    normalized = normalize_for_exact_match(dataset).replace(" ", "")
    return DATASET_ALIASES.get(normalized, normalized)


def resolve_model_aliases(model: str) -> List[str]:
    """Return deterministic directory-name candidates without conflating readers."""
    candidates = [model]
    if model.endswith("_real_result"):
        candidates.append(model[: -len("_real_result")])
    else:
        candidates.append(f"{model}_real_result")
    return list(dict.fromkeys(candidates))


def load_prompt_template(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with prompt_file.open("r", encoding="utf-8") as f:
        return f.read()


def normalize_for_exact_match(text: str) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower().strip()
    return re.sub(r"\s+", " ", normalized)


def normalize_label(label: Optional[str]) -> str:
    raw = normalize_for_exact_match(label or "")
    if "distractor" in raw:
        return DE_LABEL
    if "premature" in raw or "refusal" in raw or "truncation" in raw:
        return PR_LABEL
    if "aggregation" in raw or "hallucination" in raw:
        return RH_LABEL
    return RH_LABEL


def detect_explicit_refusal(model_answer: str) -> Optional[str]:
    norm_answer = normalize_for_exact_match(model_answer)
    for phrase in REFUSAL_PATTERNS:
        norm_phrase = normalize_for_exact_match(phrase)
        if norm_phrase and norm_phrase in norm_answer:
            return phrase
    return None


def _parse_csv_row(raw: str) -> List[str]:
    try:
        return next(csv.reader([raw]))
    except Exception:
        return [raw]


def _flatten_table_cells(table_record: Dict) -> List[str]:
    """Return actual row data cells only; headers are not Rule 1 evidence."""
    cells: List[str] = []
    instances = table_record.get("instances", [])
    if isinstance(instances, list):
        for row in instances:
            if isinstance(row, str):
                cells.extend(_parse_csv_row(row))
            elif isinstance(row, list):
                for v in row:
                    if v is not None:
                        cells.append(str(v))
            elif row is not None:
                cells.append(str(row))

    return [c for c in cells if c is not None and str(c).strip()]


def load_table_cell_map(dataset: str) -> Dict[str, List[str]]:
    table_file = (
        PROJECT_ROOT / "data" / "single-table-retrieval" / "test" / dataset / "table.jsonl"
    )
    if not table_file.exists():
        return {}

    table_cells: Dict[str, List[str]] = {}
    with table_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            table_id = str(record.get("id", "")).strip()
            if not table_id:
                continue
            table_cells[table_id] = _flatten_table_cells(record)

    return table_cells


def load_gold_table_map(dataset: str) -> Dict[str, List[str]]:
    query_file = (
        PROJECT_ROOT / "data" / "single-table-retrieval" / "test" / dataset / "query.jsonl"
    )
    if not query_file.exists():
        return {}

    result: Dict[str, List[str]] = {}
    with query_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = record.get("id", record.get("query_id"))
            if qid is None:
                continue
            ground_truth = record.get("ground_truth_list", [])
            gold_ids: List[str] = []
            if isinstance(ground_truth, list):
                for gt in ground_truth:
                    if isinstance(gt, dict) and gt.get("id") is not None:
                        gold_ids.append(str(gt.get("id")))
                    elif gt is not None:
                        gold_ids.append(str(gt))
            result[str(qid)] = gold_ids

    return result


def _iter_json_or_jsonl(path: Path) -> Iterable[Dict]:
    if not path or not path.exists():
        return
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("records", payload.get("results", payload.get("data", [])))
        if not isinstance(rows, list):
            rows = [payload]
    else:
        rows = []
    for row in rows:
        if isinstance(row, dict):
            yield row


def _load_jsonl_records_by_query_id(path: Path) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    if not path or not path.exists():
        return records
    for row in _iter_json_or_jsonl(path):
        qid = row.get("query_id", row.get("id"))
        if qid is not None:
            records[str(qid)] = row
    return records


def find_injection_file(
    inject_root: Path,
    dataset: str,
    target_k: int,
    rotation: str,
    negative: str,
) -> Optional[Path]:
    candidates: List[Path] = []
    dataset_dir = inject_root / dataset
    if not dataset_dir.exists():
        return None

    base_name = f"{target_k}_{rotation}_{negative}"
    candidates.extend(sorted(dataset_dir.glob(f"*/{base_name}.jsonl")))
    candidates.extend(sorted(dataset_dir.glob(f"*/{base_name}.json")))
    candidates.extend(sorted(dataset_dir.glob(f"{base_name}.jsonl")))
    candidates.extend(sorted(dataset_dir.glob(f"{base_name}.json")))

    return candidates[0] if candidates else None


def find_reasoning_file(
    reasoning_root: Path,
    dataset: str,
    model: str,
    target_k: int,
    rotation: str,
    negative: str,
) -> Optional[Path]:
    candidates: List[Path] = []
    for dataset_alias in [normalize_dataset_name(dataset), dataset]:
        for model_alias in resolve_model_aliases(model):
            model_dir = reasoning_root / dataset_alias / model_alias
            candidates.extend([
                model_dir / f"{target_k}_{rotation}_{negative}.jsonl",
                model_dir / f"{target_k}_{rotation}_{negative}.json",
                model_dir / f"real_result_{target_k}.jsonl",
                model_dir / f"real_result_{target_k}.json",
            ])
        candidates.extend([
            reasoning_root / dataset_alias / f"{target_k}_{rotation}_{negative}" / "records.jsonl",
            reasoning_root / dataset_alias / f"{target_k}_{rotation}_{negative}" / "records.json",
        ])
    for c in candidates:
        if c.exists():
            return c
    return None


def extract_distractor_cells(
    injection_table_ids: List[str],
    gold_table_ids: List[str],
    table_cell_map: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    gold_set = {str(g) for g in gold_table_ids}
    distractor_cells: List[Tuple[str, str]] = []
    for table_id in injection_table_ids:
        tid = str(table_id)
        if tid in gold_set:
            continue
        for cell in table_cell_map.get(tid, []):
            distractor_cells.append((tid, cell))
    return distractor_cells


def extract_gold_cells(
    gold_table_ids: List[str], table_cell_map: Dict[str, List[str]]
) -> List[str]:
    return [cell for table_id in gold_table_ids for cell in table_cell_map.get(str(table_id), [])]


def find_distractor_match(
    model_answer: str,
    distractor_cells: List[Tuple[str, str]],
) -> Optional[Dict[str, str]]:
    norm_answer = normalize_for_exact_match(model_answer)
    if not norm_answer:
        return None

    best: Optional[Dict[str, str]] = None
    for table_id, cell in distractor_cells:
        norm_cell = normalize_for_exact_match(cell)
        if len(norm_cell) < 2:
            continue
        if not any(ch.isalnum() for ch in norm_cell):
            continue
        if norm_cell in norm_answer:
            candidate = {
                "matched_distractor_table": table_id,
                "matched_distractor_cell": cell,
                "matched_distractor_cell_normalized": norm_cell,
            }
            if best is None or len(norm_cell) > len(best["matched_distractor_cell_normalized"]):
                best = candidate

    return best


def assign_final_error_label(
    model_answer: str,
    distractor_cells: List[Tuple[str, str]],
    gold_cells: Optional[List[str]] = None,
) -> Dict[str, Optional[str]]:
    # Provenance disambiguation: an exact value shared by gold and distractor
    # data cells cannot establish that the answer came from the distractor.
    gold_values = {normalize_for_exact_match(cell) for cell in (gold_cells or [])}
    qualifying_cells = [
        (table_id, cell) for table_id, cell in distractor_cells
        if normalize_for_exact_match(cell) not in gold_values
    ]
    # find_distractor_match deterministically selects the longest qualifying
    # normalized substring; ties retain source-table/cell iteration order.
    distractor_match = find_distractor_match(model_answer, qualifying_cells)
    if distractor_match:
        return {
            "final_error_type": DE_LABEL,
            "label_source": LABEL_SOURCE_DISTRACTOR,
            "matched_distractor_table": distractor_match["matched_distractor_table"],
            "matched_distractor_cell": distractor_match["matched_distractor_cell"],
            "matched_refusal_phrase": None,
        }

    refusal_phrase = detect_explicit_refusal(model_answer)
    if refusal_phrase:
        return {
            "final_error_type": PR_LABEL,
            "label_source": LABEL_SOURCE_REFUSAL,
            "matched_distractor_table": None,
            "matched_distractor_cell": None,
            "matched_refusal_phrase": refusal_phrase,
        }

    return {
        "final_error_type": RH_LABEL,
        "label_source": LABEL_SOURCE_RESIDUAL,
        "matched_distractor_table": None,
        "matched_distractor_cell": None,
        "matched_refusal_phrase": None,
    }


def parse_llm_error_response(classification_text: str) -> Tuple[str, float]:
    error_type = "Unknown"
    confidence = 0.0
    for line in classification_text.splitlines():
        if "ERROR_TYPE:" in line:
            parts = line.split(":", 1)
            if len(parts) > 1:
                error_type = parts[1].strip()
        elif "CONFIDENCE:" in line:
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return error_type, confidence


def parse_llm_question_response(classification_text: str) -> Tuple[str, float]:
    question_type = "Unknown"
    confidence = 0.0
    for line in classification_text.splitlines():
        if "QUESTION_TYPE:" in line:
            parts = line.split(":", 1)
            if len(parts) > 1:
                question_type = parts[1].strip()
        elif "CONFIDENCE:" in line:
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return question_type, confidence


def _openrouter_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    generic_key = os.getenv("API_KEY")
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name == "OPENROUTER_API_KEY":
                return value
            if name == "API_KEY" and not generic_key:
                generic_key = value
    if generic_key:
        return generic_key
    raise RuntimeError("OPENROUTER_API_KEY is not configured")


def _openrouter_chat_completion(model: str, prompt: str, max_retries: int = 2) -> Dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body, method="POST",
        headers={"Authorization": f"Bearer {_openrouter_api_key()}", "Content-Type": "application/json"},
    )
    last_error = None
    retries = 0
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
            usage = payload.get("usage", {}) or {}
            choices = payload.get("choices", [])
            text = choices[0].get("message", {}).get("content", "") if choices else ""
            return {"text": text, "usage": usage, "retry_count": retries}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:500]}")
            if exc.code != 429 and exc.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
        if attempt < max_retries:
            retries += 1
            time.sleep(2 ** attempt)
    raise RuntimeError(str(last_error or "OpenRouter request failed"))


def _chat_completion(provider: Optional[str], model: Optional[str], prompt: str) -> Dict:
    if normalize_for_exact_match(provider or "") == "openrouter":
        return _openrouter_chat_completion(model or "openai/gpt-5.6-luna", prompt)
    from llm.llm_client import LLMClient
    client = LLMClient(provider=provider, max_retries=2, enable_logging=False)
    return client.chat_completion(
        model=model or client.model, messages=[{"role": "user", "content": prompt}],
        temperature=0.0, return_full_response=True,
    )


def classify_error_with_llm(
    question: str,
    reasoning_record: Dict,
    gold_answer: str,
    model_answer: str,
    prompt_template: str,
    llm_provider: Optional[str],
    llm_model: Optional[str],
) -> Dict:
    retrieved_tables_info = reasoning_record.get("retrieved_tables_info", "")
    context_summary = reasoning_record.get("context", "")[:1000]

    prompt = prompt_template.format(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer,
        retrieved_tables=retrieved_tables_info[:2000],
        context=context_summary,
    )

    try:
        response = _chat_completion(llm_provider, llm_model, prompt)
        text = str(response.get("text", "")).strip() if isinstance(response, dict) else str(response).strip()
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(str(response["error"]))

        error_type, confidence = parse_llm_error_response(text)
        return {
            "llm_diagnostic_type": error_type,
            "llm_confidence": confidence,
            "llm_raw_response": text,
            "llm_input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "llm_output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "llm_retry_count": int(response.get("retry_count", 0) or 0),
        }
    except Exception as exc:
        return {
            "llm_diagnostic_type": "Unknown",
            "llm_confidence": 0.0,
            "llm_raw_response": "",
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "llm_retry_count": 0,
            "llm_error": str(exc),
        }


def classify_question_type_with_llm(
    question: str,
    question_prompt_template: str,
    llm_provider: Optional[str],
    llm_model: Optional[str],
) -> Dict:
    prompt = question_prompt_template.format(question=question)
    try:
        response = _chat_completion(llm_provider, llm_model, prompt)
        text = str(response.get("text", "")).strip() if isinstance(response, dict) else str(response).strip()
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(str(response["error"]))

        question_type, confidence = parse_llm_question_response(text)
        return {
            "question_type": question_type,
            "question_confidence": confidence,
            "question_raw_response": text,
            "question_input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "question_output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "question_retry_count": int(response.get("retry_count", 0) or 0),
        }
    except Exception as exc:
        return {
            "question_type": "Unknown",
            "question_confidence": 0.0,
            "question_raw_response": "",
            "question_input_tokens": 0,
            "question_output_tokens": 0,
            "question_retry_count": 0,
            "question_error": str(exc),
        }


def deterministic_sort_key(record: Dict) -> Tuple:
    qid = record.get("query_id")
    rotation = str(record.get("rotation", ""))
    filename = str(record.get("file", ""))
    try:
        qid_key = (0, int(qid))
    except (TypeError, ValueError):
        qid_key = (1, str(qid))
    return qid_key + (rotation, filename)


def sample_records(records: List[Dict], sample_size: Optional[int], seed: int) -> Tuple[List[Dict], SamplingInfo]:
    ordered = sorted(records, key=deterministic_sort_key)
    total = len(ordered)

    if sample_size is None:
        return ordered, SamplingInfo(
            sample_size=None,
            total_available_cases=total,
            seed=seed,
            sampling_method="all",
        )

    n = min(sample_size, total)
    rng = random.Random(seed)
    sampled = rng.sample(ordered, n)
    sampled = sorted(sampled, key=deterministic_sort_key)
    return sampled, SamplingInfo(
        sample_size=sample_size,
        total_available_cases=total,
        seed=seed,
        sampling_method="random_without_replacement",
    )


def load_distribution_records(
    output_dataset_dir: Path,
    model: str,
    target_k: int,
    negative: str,
    category: str,
) -> List[Dict]:
    dist_file = output_dataset_dir / f"{model}_K{target_k}_distribution.json"
    if not dist_file.exists():
        raise FileNotFoundError(f"Distribution file not found. Run Stage 1 first: {dist_file}")

    with dist_file.open("r", encoding="utf-8") as f:
        dist_data = json.load(f)

    records = dist_data.get("distribution", {}).get(negative, {}).get(category, [])
    if not isinstance(records, list):
        return []
    return records


def extract_injection_table_ids(record: Dict) -> List[str]:
    for field in ("injection", "table_ids", "retrieved_table_ids", "injected_table_ids"):
        value = record.get(field)
        if isinstance(value, list):
            return [str(item.get("id", item)) if isinstance(item, dict) else str(item) for item in value]
    return []


def extract_reasoning_fields(record: Dict) -> Tuple[str, str, str]:
    question = record.get("question", record.get("query", ""))
    answer = record.get("llm_response", record.get("model_answer", record.get("response", "")))
    gold = record.get("gold_answer", record.get("answer", record.get("reference_answer", "")))
    return str(question or ""), str(answer or ""), str(gold or "")


def audit_resolvability(
    *, dataset: str, records: List[Dict], model: str, target_k: int, negative: str,
    inject_root: Path, reasoning_root: Path, table_cell_map: Dict[str, List[str]],
    gold_table_map: Dict[str, List[str]],
) -> Tuple[List[Dict], Dict[str, Any]]:
    report: Dict[str, Any] = {
        "total_eligible": len(records), "fully_resolvable": 0, "missing_model_answer": 0,
        "missing_injection": 0, "missing_gold_table_mapping": 0, "missing_table_data": 0,
        "other_failures": 0, "unresolved_cases": [],
    }
    resolvable: List[Dict] = []
    reasoning_cache: Dict[Tuple[str, int, str, str], Dict[str, Dict]] = {}
    injection_cache: Dict[Tuple[int, str, str], Dict[str, Dict]] = {}
    for record in records:
        qid, rotation = str(record.get("query_id")), str(record.get("rotation", ""))
        reasons: List[str] = []
        rkey = (model, target_k, rotation, negative)
        if rkey not in reasoning_cache:
            path = find_reasoning_file(reasoning_root, dataset, model, target_k, rotation, negative)
            reasoning_cache[rkey] = _load_jsonl_records_by_query_id(path) if path else {}
        rr = reasoning_cache[rkey].get(qid, {})
        _, model_answer, _ = extract_reasoning_fields(rr)
        if not model_answer.strip():
            reasons.append("missing_model_answer")
            report["missing_model_answer"] += 1
        ikey = (target_k, rotation, negative)
        if ikey not in injection_cache:
            path = find_injection_file(inject_root, dataset, target_k, rotation, negative)
            injection_cache[ikey] = _load_injection_index(path) if path else {}
        ir = injection_cache[ikey].get(qid, {})
        table_ids = extract_injection_table_ids(ir)
        if not ir or not table_ids:
            reasons.append("missing_injection")
            report["missing_injection"] += 1
        gold_ids = gold_table_map.get(qid, [])
        if not gold_ids:
            reasons.append("missing_gold_table_mapping")
            report["missing_gold_table_mapping"] += 1
        missing_tables = [tid for tid in table_ids if tid not in table_cell_map]
        if missing_tables:
            reasons.append("missing_table_data")
            report["missing_table_data"] += 1
        if reasons:
            report["unresolved_cases"].append({"query_id": qid, "rotation": rotation, "reasons": reasons})
        else:
            report["fully_resolvable"] += 1
            resolvable.append(record)
    return resolvable, report


def _load_injection_index(path: Path) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    if not path or not path.exists():
        return index
    for row in _iter_json_or_jsonl(path):
        qid = row.get("query_id", row.get("id"))
        if qid is not None:
            index[str(qid)] = row
    return index


def classify_records(
    *,
    dataset: str,
    model: str,
    target_k: int,
    category: str,
    negative: str,
    records: List[Dict],
    inject_root: Path,
    reasoning_root: Path,
    table_cell_map: Dict[str, List[str]],
    gold_table_map: Dict[str, List[str]],
    prompt_template: Optional[str],
    question_prompt_template: Optional[str],
    llm_provider: Optional[str],
    llm_model: Optional[str],
    rules_only: bool,
    output_file: Path,
) -> List[Dict]:
    existing_classifications: List[Dict] = []
    processed_keys: set = set()

    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_classifications = existing.get("classifications", []) if isinstance(existing, dict) else []
            for cls in existing_classifications:
                key = f"{cls.get('query_id')}::{cls.get('rotation', '')}"
                processed_keys.add(key)
            print(f"  Found existing output with {len(processed_keys)} processed case keys; will skip them.")
        except Exception as exc:
            print(f"  Warning: failed to load existing output; recomputing from scratch: {exc}")
            existing_classifications = []
            processed_keys = set()

    results = list(existing_classifications)

    injection_cache: Dict[Tuple[int, str, str], Dict[str, Dict]] = {}
    reasoning_cache: Dict[Tuple[str, int, str, str], Dict[str, Dict]] = {}

    for idx, record in enumerate(records, start=1):
        qid = str(record.get("query_id"))
        rotation = str(record.get("rotation", ""))
        case_key = f"{qid}::{rotation}"
        if case_key in processed_keys:
            print(f"  [{idx}/{len(records)}] {case_key} already processed, skipping.")
            continue

        print(f"  [{idx}/{len(records)}] Processing qid={qid}, rotation={rotation} ...", end=" ")

        injection_key = (target_k, rotation, negative)
        if injection_key not in injection_cache:
            inj_file = find_injection_file(inject_root, dataset, target_k, rotation, negative)
            injection_cache[injection_key] = _load_injection_index(inj_file) if inj_file else {}
        injection_index = injection_cache[injection_key]
        injection_record = injection_index.get(qid, {})
        injection_tables = extract_injection_table_ids(injection_record)

        reasoning_key = (model, target_k, rotation, negative)
        if reasoning_key not in reasoning_cache:
            reasoning_file = find_reasoning_file(reasoning_root, dataset, model, target_k, rotation, negative)
            reasoning_cache[reasoning_key] = _load_jsonl_records_by_query_id(reasoning_file) if reasoning_file else {}
        reasoning_record = reasoning_cache[reasoning_key].get(qid, {})

        question, model_answer, gold_answer = extract_reasoning_fields(reasoning_record)
        if not model_answer.strip():
            raise RuntimeError(f"Unresolved source for qid={qid}, rotation={rotation}: missing model answer")

        gold_table_ids = gold_table_map.get(qid, [])
        distractor_cells = extract_distractor_cells(injection_tables, gold_table_ids, table_cell_map)
        gold_cells = extract_gold_cells(gold_table_ids, table_cell_map)
        rule_result = assign_final_error_label(model_answer, distractor_cells, gold_cells)

        llm_diag = {
            "llm_diagnostic_type": "Unknown",
            "llm_confidence": 0.0,
            "llm_raw_response": "",
        }
        qtype_diag = {
            "question_type": "Unknown",
            "question_confidence": 0.0,
            "question_raw_response": "",
        }

        if not rules_only and prompt_template and question_prompt_template and reasoning_record:
            llm_diag = classify_error_with_llm(
                question=question,
                reasoning_record=reasoning_record,
                gold_answer=gold_answer,
                model_answer=model_answer,
                prompt_template=prompt_template,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            qtype_diag = classify_question_type_with_llm(
                question=question,
                question_prompt_template=question_prompt_template,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )

        out = {
            "query_id": qid,
            "rotation": rotation,
            "error_type": rule_result["final_error_type"],
            "final_error_type": rule_result["final_error_type"],
            "label_source": rule_result["label_source"],
            "matched_distractor_cell": rule_result["matched_distractor_cell"],
            "matched_distractor_table": rule_result["matched_distractor_table"],
            "matched_refusal_phrase": rule_result["matched_refusal_phrase"],
            "llm_diagnostic_type": llm_diag.get("llm_diagnostic_type", "Unknown"),
            "llm_confidence": llm_diag.get("llm_confidence", 0.0),
            "llm_raw_response": llm_diag.get("llm_raw_response", ""),
            "llm_input_tokens": llm_diag.get("llm_input_tokens", 0),
            "llm_output_tokens": llm_diag.get("llm_output_tokens", 0),
            "question_type": qtype_diag.get("question_type", "Unknown"),
            "question_confidence": qtype_diag.get("question_confidence", 0.0),
            "question_raw_response": qtype_diag.get("question_raw_response", ""),
            "question_input_tokens": qtype_diag.get("question_input_tokens", 0),
            "question_output_tokens": qtype_diag.get("question_output_tokens", 0),
            "model_answer": model_answer,
            "gold_answer": gold_answer,
        }
        if llm_diag.get("llm_error"):
            out["llm_error"] = llm_diag["llm_error"]
        if qtype_diag.get("question_error"):
            out["question_error"] = qtype_diag["question_error"]
        results.append(out)
        print("ok")

    return results


def _parse_model_k_negative_category(error_file_name: str) -> Optional[Tuple[str, int, str, str]]:
    m = ERROR_FILE_RE.match(error_file_name)
    if not m:
        return None
    return (
        m.group("model"),
        int(m.group("k")),
        m.group("negative"),
        m.group("category"),
    )


def relabel_existing_files(
    *,
    dataset: str,
    inject_root: Path,
    reasoning_root: Path,
    output_dataset_dir: Path,
    relabel_pattern: str,
    backup_suffix: str,
) -> Dict:
    table_cell_map = load_table_cell_map(dataset)
    gold_table_map = load_gold_table_map(dataset)

    error_files = sorted(output_dataset_dir.glob(relabel_pattern))
    error_files = [p for p in error_files if _parse_model_k_negative_category(p.name)]

    if not error_files:
        raise FileNotFoundError(f"No Stage 3 files matched: {output_dataset_dir / relabel_pattern}")

    report = {
        "classification_policy": CLASSIFICATION_POLICY,
        "files": [],
        "total_cases": 0,
        "agreement_count": 0,
        "changed_labels": {},
        "final_distribution": {DE_LABEL: 0, PR_LABEL: 0, RH_LABEL: 0},
        "distractor_with_audit_cell": 0,
    }

    injection_cache: Dict[Tuple[int, str, str], Dict[str, Dict]] = {}
    reasoning_cache: Dict[Tuple[str, int, str, str], Dict[str, Dict]] = {}

    for error_file in error_files:
        parsed = _parse_model_k_negative_category(error_file.name)
        if not parsed:
            continue
        model, target_k, negative, category = parsed

        backup_path = error_file.with_name(f"{error_file.stem}{backup_suffix}.json")
        if not backup_path.exists():
            shutil.copy2(error_file, backup_path)

        with error_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        dist_records = load_distribution_records(
            output_dataset_dir=output_dataset_dir,
            model=model,
            target_k=target_k,
            negative=negative,
            category=category,
        )
        rotation_by_qid: Dict[str, str] = {}
        for r in dist_records:
            qid = str(r.get("query_id"))
            if qid not in rotation_by_qid:
                rotation_by_qid[qid] = str(r.get("rotation", ""))

        updated: List[Dict] = []
        file_changed = 0
        file_total = 0

        for item in payload.get("classifications", []):
            if not isinstance(item, dict):
                continue
            file_total += 1
            qid = str(item.get("query_id"))
            rotation = str(item.get("rotation") or rotation_by_qid.get(qid, ""))

            llm_diagnostic_type = item.get("llm_diagnostic_type") or item.get("error_type") or "Unknown"
            llm_confidence = item.get("llm_confidence", item.get("error_confidence", 0.0))
            llm_raw = item.get("llm_raw_response", item.get("error_raw_response", ""))

            model_answer = item.get("model_answer", "")
            gold_answer = item.get("gold_answer", "")

            if not model_answer and rotation:
                reasoning_key = (model, target_k, rotation, negative)
                if reasoning_key not in reasoning_cache:
                    reasoning_file = find_reasoning_file(
                        reasoning_root, dataset, model, target_k, rotation, negative
                    )
                    reasoning_cache[reasoning_key] = _load_jsonl_records_by_query_id(reasoning_file) if reasoning_file else {}
                rr = reasoning_cache[reasoning_key].get(qid, {})
                model_answer = rr.get("llm_response", "")
                gold_answer = gold_answer or rr.get("gold_answer", "")

            final_label: str
            label_source: str
            matched_distractor_cell = None
            matched_distractor_table = None
            matched_refusal_phrase = None

            if model_answer and rotation:
                injection_key = (target_k, rotation, negative)
                if injection_key not in injection_cache:
                    inj_file = find_injection_file(inject_root, dataset, target_k, rotation, negative)
                    injection_cache[injection_key] = _load_injection_index(inj_file) if inj_file else {}

                injection_record = injection_cache[injection_key].get(qid, {})
                injection_tables = [str(t) for t in injection_record.get("injection", [])]
                gold_ids = gold_table_map.get(qid, [])
                distractor_cells = extract_distractor_cells(injection_tables, gold_ids, table_cell_map)
                gold_cells = extract_gold_cells(gold_ids, table_cell_map)
                rule_result = assign_final_error_label(model_answer, distractor_cells, gold_cells)
                final_label = rule_result["final_error_type"]
                label_source = rule_result["label_source"]
                matched_distractor_cell = rule_result["matched_distractor_cell"]
                matched_distractor_table = rule_result["matched_distractor_table"]
                matched_refusal_phrase = rule_result["matched_refusal_phrase"]
            else:
                raise ValueError(
                    f"Missing reader answer or rotation for publication-facing classification: {qid}"
                )

            old_label = normalize_label(item.get("error_type"))
            if old_label != final_label:
                file_changed += 1
                change_key = f"{old_label} -> {final_label}"
                report["changed_labels"][change_key] = report["changed_labels"].get(change_key, 0) + 1

            if normalize_label(llm_diagnostic_type) == final_label:
                report["agreement_count"] += 1

            report["final_distribution"][final_label] = report["final_distribution"].get(final_label, 0) + 1
            if final_label == DE_LABEL and matched_distractor_cell:
                report["distractor_with_audit_cell"] += 1

            updated.append(
                {
                    **item,
                    "query_id": qid,
                    "rotation": rotation,
                    "error_type": final_label,
                    "final_error_type": final_label,
                    "label_source": label_source,
                    "matched_distractor_cell": matched_distractor_cell,
                    "matched_distractor_table": matched_distractor_table,
                    "matched_refusal_phrase": matched_refusal_phrase,
                    "llm_diagnostic_type": llm_diagnostic_type,
                    "llm_confidence": llm_confidence,
                    "llm_raw_response": llm_raw,
                    "model_answer": model_answer,
                    "gold_answer": gold_answer,
                }
            )

        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        metadata.update(
            {
                "classification_policy": CLASSIFICATION_POLICY,
                "relabel_mode": "rules_only_no_llm",
                "backup_file": backup_path.name,
            }
        )

        out_payload = {
            "metadata": metadata,
            "classifications": updated,
        }

        with error_file.open("w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)

        report["files"].append(
            {
                "file": error_file.name,
                "backup": backup_path.name,
                "total_cases": file_total,
                "changed_cases": file_changed,
            }
        )
        report["total_cases"] += file_total

    report["agreement_rate"] = (
        report["agreement_count"] / report["total_cases"] if report["total_cases"] else 0.0
    )

    report_file = output_dataset_dir / "relabel_report.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_file"] = str(report_file)

    return report


def run_stage3(args: argparse.Namespace) -> None:
    args.dataset = normalize_dataset_name(args.dataset)
    output_dataset_dir = Path(args.output_dir) / args.dataset
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.relabel_existing:
        report = relabel_existing_files(
            dataset=args.dataset,
            inject_root=Path(args.inject_root),
            reasoning_root=Path(args.reasoning_root),
            output_dataset_dir=output_dataset_dir,
            relabel_pattern=args.relabel_pattern,
            backup_suffix=args.backup_suffix,
        )
        print("[Stage 3] Relabel Existing Outputs (rules-only, no LLM)")
        print(f"  Files migrated: {len(report['files'])}")
        print(f"  Total cases: {report['total_cases']}")
        print(f"  LLM label == deterministic final label: {report['agreement_count']} ({report['agreement_rate'] * 100:.2f}%)")
        print("  Final distribution:")
        for label in [DE_LABEL, PR_LABEL, RH_LABEL]:
            print(f"    {label}: {report['final_distribution'].get(label, 0)}")
        print(f"  Distractor labels with auditable matched cell: {report['distractor_with_audit_cell']}")
        print(f"  Report saved to: {report['report_file']}")
        return

    prompt_template = None
    question_prompt_template = None
    if not args.rules_only:
        prompt_template = load_prompt_template(Path(args.prompt_file))
        question_prompt_template = load_prompt_template(Path(args.question_prompt_file))

    distribution_dataset_dir = Path(args.distribution_root) / args.dataset
    records = load_distribution_records(
        output_dataset_dir=distribution_dataset_dir,
        model=args.model,
        target_k=args.target_k,
        negative=args.negative,
        category=args.category,
    )

    table_cell_map = load_table_cell_map(args.dataset)
    gold_table_map = load_gold_table_map(args.dataset)
    resolvable_records, resolver_report = audit_resolvability(
        dataset=args.dataset, records=records, model=args.model, target_k=args.target_k,
        negative=args.negative, inject_root=Path(args.inject_root),
        reasoning_root=Path(args.reasoning_root), table_cell_map=table_cell_map,
        gold_table_map=gold_table_map,
    )
    resolver_report["percent_fully_resolvable"] = round(
        resolver_report["fully_resolvable"] / resolver_report["total_eligible"] * 100, 4
    ) if resolver_report["total_eligible"] else 0.0
    print("[Resolver audit] " + json.dumps({k: v for k, v in resolver_report.items() if k != "unresolved_cases"}))
    if args.scan_resolvability:
        audit_file = output_dataset_dir / f"{args.model}_K{args.target_k}_{args.negative}_{args.category}_resolver_audit.json"
        with audit_file.open("w", encoding="utf-8") as f:
            json.dump(resolver_report, f, ensure_ascii=False, indent=2)
        print(f"  Resolver audit saved to: {audit_file}")
        return
    sampled_records, sampling_info = sample_records(resolvable_records, args.sample_size, args.seed)
    print("[Stage 3] Deterministic Error Labeling")
    print(f"  Dataset: {args.dataset}, Model: {args.model}, K: {args.target_k}")
    print(f"  Category: {args.category}, Negative: {args.negative}")
    print(f"  Rules only: {args.rules_only}")
    print(
        f"  Sampling: {sampling_info.sampling_method}, selected={len(sampled_records)}, total={sampling_info.total_available_cases}, seed={sampling_info.seed}"
    )

    output_file = (
        output_dataset_dir
        / f"{args.model}_K{args.target_k}_{args.negative}_{args.category}_errors.json"
    )

    classifications = classify_records(
        dataset=args.dataset,
        model=args.model,
        target_k=args.target_k,
        category=args.category,
        negative=args.negative,
        records=sampled_records,
        inject_root=Path(args.inject_root),
        reasoning_root=Path(args.reasoning_root),
        table_cell_map=table_cell_map,
        gold_table_map=gold_table_map,
        prompt_template=prompt_template,
        question_prompt_template=question_prompt_template,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        rules_only=args.rules_only,
        output_file=output_file,
    )

    payload = {
        "metadata": {
            "dataset": args.dataset,
            "model": args.model,
            "target_k": args.target_k,
            "category": args.category,
            "negative": args.negative,
            "sample_size": sampling_info.sample_size,
            "total_available_cases": sampling_info.total_available_cases,
            "seed": sampling_info.seed,
            "sampling_method": sampling_info.sampling_method,
            "classification_policy": CLASSIFICATION_POLICY,
            "rules_only": args.rules_only,
            "resolver_report": {k: v for k, v in resolver_report.items() if k != "unresolved_cases"},
        },
        "classifications": classifications,
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    counts: Dict[str, int] = {}
    agree = 0
    for cls in classifications:
        et = normalize_label(cls.get("final_error_type") or cls.get("error_type"))
        counts[et] = counts.get(et, 0) + 1
        llm_et = normalize_label(cls.get("llm_diagnostic_type"))
        if llm_et == et:
            agree += 1

    total = len(classifications)
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    api_failures = 0
    for cls in classifications:
        for prefix in ("llm", "question"):
            input_tokens = int(cls.get(f"{prefix}_input_tokens", 0) or 0)
            output_tokens = int(cls.get(f"{prefix}_output_tokens", 0) or 0)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            if input_tokens or output_tokens:
                total_calls += 1
        if cls.get("llm_error") or cls.get("question_error"):
            api_failures += 1
    input_cost = total_input_tokens / 1_000_000 * args.input_price_per_million
    output_cost = total_output_tokens / 1_000_000 * args.output_price_per_million
    print(f"\nSaved: {output_file}")
    print("\n[Final Error Type Summary]")
    for label in [DE_LABEL, PR_LABEL, RH_LABEL]:
        c = counts.get(label, 0)
        pct = (c / total * 100.0) if total else 0.0
        print(f"  {label:25s}: {c:4d} ({pct:6.2f}%)")
    agree_pct = (agree / total * 100.0) if total else 0.0
    print(f"  LLM diagnostic agreement: {agree}/{total} ({agree_pct:.2f}%)")
    print("\n[LLM Usage Summary]")
    print(f"  total_llm_calls: {total_calls}")
    print(f"  total_input_tokens: {total_input_tokens}")
    print(f"  total_output_tokens: {total_output_tokens}")
    print(f"  avg_input_tokens_per_case: {total_input_tokens / total if total else 0:.2f}")
    print(f"  avg_output_tokens_per_case: {total_output_tokens / total if total else 0:.2f}")
    print(f"  estimated_cost_usd: {input_cost + output_cost:.6f}")
    print(f"  api_failure_cases: {api_failures}")
    payload["metadata"]["llm_usage"] = {
        "total_llm_calls": total_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_input_tokens_per_case": total_input_tokens / total if total else 0.0,
        "avg_output_tokens_per_case": total_output_tokens / total if total else 0.0,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
        "api_failure_cases": api_failures,
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Error Analysis Stage 3: LLM-assisted diagnosis + deterministic final labeling"
    )

    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--target-k", type=int, default=30, help="Target K")
    parser.add_argument(
        "--category",
        default="perfect_to_fail",
        choices=["both_success", "both_fail", "perfect_to_fail", "perfect_from_fail"],
        help="Category to analyze",
    )
    parser.add_argument(
        "--negative",
        default="hard",
        choices=["hard", "soft"],
        help="Negative type to analyze",
    )

    parser.add_argument("--llm-provider", default=None, help="LLM provider (optional)")
    parser.add_argument("--llm-model", default=None, help="LLM model name (optional)")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="If omitted, classify all matching cases; if set, use reproducible random sample",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--rules-only", action="store_true", help="Skip LLM calls and run deterministic rules only")

    parser.add_argument("--eval-root", default="evaluation_results/", help="Reserved for compatibility")
    parser.add_argument("--inject-root", default="injections/", help="Injections root")
    parser.add_argument("--reasoning-root", default="reasoning_results/", help="Reasoning results root")
    parser.add_argument("--output-dir", default="error_analysis/", help="Output directory")
    parser.add_argument("--distribution-root", default="error_analysis/", help="Stage 1 distribution root")
    parser.add_argument("--scan-resolvability", action="store_true", help="Audit sources only; make no LLM calls")
    parser.add_argument("--input-price-per-million", type=float, default=0.10)
    parser.add_argument("--output-price-per-million", type=float, default=0.60)
    parser.add_argument(
        "--prompt-file",
        default="prompts/error_classification_prompt.txt",
        help="Error classification prompt template",
    )
    parser.add_argument(
        "--question-prompt-file",
        default="prompts/question_classification_prompt.txt",
        help="Question classification prompt template",
    )

    parser.add_argument(
        "--relabel-existing",
        action="store_true",
        help="Relabel existing Stage 3 outputs with deterministic rules only (no LLM calls)",
    )
    parser.add_argument(
        "--relabel-pattern",
        default="*_errors.json",
        help="Glob pattern under error_analysis/<dataset> for relabel-existing mode",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".legacy_llm_only",
        help="Backup suffix for relabeled files (written as <stem><suffix>.json)",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_stage3(args)


if __name__ == "__main__":
    main()
