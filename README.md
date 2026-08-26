# NoiseCeil-TableQA

Public artifact for the EMNLP 2026 Findings paper **“How Much Retrieval Depth Does Your Reader Need? Noise Ceilings and Cost-Effectiveness in Table QA.”** The controlled stress test guarantees the gold table and varies retrieval depth and distractor hardness to measure reader-side noise tolerance.

## Release scope and provenance classes

- `artifacts/camera_ready/` contains the final paper-facing Score/SD/NRR, Noise Ceiling, token, cost, and CpS sources.
- `artifacts/figures/` and `artifacts/error_analysis/` contain the frozen Figure 2–6 chains.
- `artifacts/semantic_judge/` contains exact judge prompt/configuration provenance and aggregate token usage; **per-query semantic-judge verdict records are not part of this public release**.
- `human_validation/` contains the approved sanitized final-v2 methods and aggregate results.
- `docs/` records official dataset acquisition information and six hash-identified normalized experiment snapshots; benchmark text is not redistributed.

`CAMERA_READY_PAPER_FACING` and `FROZEN_CAMERA_READY_AGGREGATE` identify frozen publication-facing records. `AUTHOR_RESOLVED_AGGREGATE` identifies the final three-run controlled source. `RELEASE_TIME_RECONSTRUCTION_TOOLING` identifies scripts that reproduce reported analyses from released artifacts; these scripts are not represented as the exact historical execution environment.

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Offline reproduction needs no API key. New inference is outside the reproduction path and may incur provider cost. The release does not claim complete end-to-end byte-identical re-execution of the original paid API experiments.

## Datasets

Acquire E2E-WTQ/WikiTableQuestions, FeTaQA, and OTT-QA from their official distributions and follow their licenses. See [docs/DATASETS.md](docs/DATASETS.md) and `docs/DATASET_VERSION_MANIFEST.csv`.

The six normalized query/table snapshots used in the experiments are identified by exact row counts and SHA-256 hashes as `FROZEN_EXPERIMENT_INPUT`. The release supports compatibility verification for these inputs but does not provide a complete upstream-to-normalized conversion pipeline for the FeTaQA and OTT-QA snapshots. Benchmark question/table text is not bundled.

## Controlled Score, SD, NRR, and Noise Ceilings

- `controlled_trials_author_resolved.csv`: final three-run Hard/Soft trial values, means, sample SD (`ddof=1`), and SEM; BGE-M3 columns remain the frozen publication-facing record.
- `score_nrr_frozen.csv`: final Figure 2 Score/NRR source.
- `score_nrr_std_camera_ready.csv`: exact three-decimal manuscript displays.

```bash
python scripts/reproduce_noise_ceilings.py
python scripts/reproduce_retrieval_spec_matrix.py
```

NRR is `Score@K / Score@1`. Kc is the maximum tested K in `{1,5,10,20,30,40,50}` satisfying `NRR >= 0.9`, without assuming monotonicity. All **24/24** Hard/Soft Noise Ceiling cells reproduce.

## Cost per Success

- `table4_author_resolved.csv`: complete final Score/token/Average Cost/CpS grid.
- `cps_camera_ready.csv`: Figure 4 plotting source.
- `cps_displayed_camera_ready.csv`: the 96 controlled CpS cells displayed in the appendix.
- `cps_published_camera_ready.csv` and `cps_recomputed_release_time.csv`: complete publication-facing and rate-recomputed grids.

```bash
python scripts/reproduce_cps.py
python scripts/figures/plot_table4_new_hard_negative_tax.py
```

CpS uses `Average Cost / Score`, where Average Cost is `(input tokens × input rate + output tokens × output rate) / 1,000,000`. `configs/camera_ready_pricing.yaml` preserves the author-maintained nominal provider-hosted pricing record dating approximately January–May 2026; its exact calendar snapshot date was not preserved. The released rates, tokens, and final Scores reproduce all controlled camera-ready CpS displays at four decimals.

The release provides the frozen BGE-M3 trial and aggregate values used in the camera-ready paper. They are the publication-facing reference for the uncontrolled BGE-M3 condition. Figure 4 requires these values and exits if they are absent.

## Figures 2–6

- Figures 2 and 4 use the final corrected authoritative CSVs and validated PDFs.
- Figure 3 uses the frozen positional aggregate.
- Figure 5 is reproduced from `figure5_transition_camera_ready.csv`, classified as a frozen camera-ready aggregate.
- Figure 6 uses the final deterministic `rule_order_v2` aggregate chain.

The commands and authoritative inputs for each figure are listed in `RESULT_SOURCE_MAP.csv`; frozen reference PDFs are in `artifacts/figures/`.

## Semantic judge and human validation

Exact semantic-judge prompt/configuration provenance and aggregate judge-validation results are released. Semantic-judge per-query verdict/evaluation files, raw reader outputs, and reasoning are intentionally excluded. See `artifacts/semantic_judge/README.md`.

Final public human-validation results retain N=598 binary-eligible judgments and two UNCERTAIN exclusions. See `human_validation/README.md`.

## Validation

```bash
python scripts/validate_release.py
pytest -q
shasum -a 256 -c CHECKSUMS.sha256
```

`RESULT_SOURCE_MAP.csv` links paper-facing results to released sources, commands, hashes, and provenance classes. `RELEASE_MANIFEST.csv` inventories the release. `PROVENANCE_LIMITATIONS.md` documents the remaining pricing-date and dataset-preprocessing limitations.

## Security, privacy, and licenses

This release contains publication-facing aggregate results and text-free identifier manifests. It excludes benchmark text not approved for redistribution, per-query model and judge outputs, annotation identities and free-text notes, credentials, provider/account metadata, and machine-local execution data. Dataset and model licensing information is in `THIRD_PARTY_LICENSES.md`; project-authored material is covered by `LICENSE`.
