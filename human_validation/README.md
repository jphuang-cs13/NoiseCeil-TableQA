# Human Validation

This directory contains the approved sanitized subset of the blinded human validation.

- `instructions/`: protocol, annotation guide, and agreement methodology.
- `prompts/`: exact semantic-judge prompt provenance.
- `scripts/`: sampling and agreement methodology.
- `manifests/`: 600-row opaque, text-free primary manifests and final labels.
- `results/`: final v2 aggregate metrics, confusion matrices, and summary.
- `provenance/`: public sampling provenance.

Final primary validation uses the frozen canonical stimulus. Twenty-four PRIMARY items were independently re-annotated before adjudication. The separate set of 19 DIAGNOSTIC items is outside the finalized validation scope and is not reported.

Two final UNCERTAIN items are excluded from binary judge-human metrics. The release contains no annotator identities, responses, optional notes, raw stimuli, private mappings, raw mismatch rows, or spreadsheet metadata.
