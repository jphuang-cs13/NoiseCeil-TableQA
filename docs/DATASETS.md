# Dataset Acquisition and Provenance

The experiments use E2E-WTQ/WikiTableQuestions, FeTaQA, and OTT-QA. Obtain each dataset from its official project distribution and comply with its license; benchmark question/table text is not redistributed here. See `THIRD_PARTY_LICENSES.md` and `DATASET_VERSION_MANIFEST.csv` for project/license details, row counts, and the six exact normalized snapshot SHA-256 hashes.

Stable IDs are the `query_id` and table identifier fields in the normalized schema. Public Hard/Soft and injection/order manifests contain identifiers only. Release-time compatibility utilities validate IDs against locally acquired data.

Exact normalized experiment snapshots and hashes are frozen. The release supports compatibility verification for these inputs; it does not provide a complete upstream-to-normalized conversion pipeline for the FeTaQA and OTT-QA snapshots.
