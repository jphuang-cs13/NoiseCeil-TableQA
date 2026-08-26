# Constructed Distractor Manifests

`hard_soft_manifest.csv` has one hard and one soft row for each of 4,456 experiment source rows. Ordered JSON arrays preserve gold IDs, distractor IDs, and hard-negative retrieval ranks. `source_row_index` disambiguates repeated query IDs in the FeTaQA-derived source (query ID 1832 occurs four times with distinct soft-negative selections).

`injection_manifest.csv.gz` is a deterministic, lossless gzip-compressed copy of the complete generated condition/query manifest. Decompress it before workflows requiring a plain CSV. `ordered_table_ids` preserves exact context order, while `gold_zero_based_positions` exposes gold placement without text. Reconstruct a context by loading legally obtained normalized tables by ID and serializing them in `ordered_table_ids` order. The generation logic is in `scripts/generate_hard_soft_negatives.py` and `scripts/generate_injections.py`.

No question, answer, table, or serialized context text is included. Upstream revisions were not recorded and are explicitly marked in the manifests.
