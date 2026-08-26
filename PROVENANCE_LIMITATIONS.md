# Provenance Limitations

1. **CpS pricing date:** CpS uses the nominal provider-hosted API rates in the experiment pricing record from approximately January–May 2026. A day-specific price date is unavailable. The released rates, token counts, and Score values reproduce all camera-ready CpS values at the reported precision.

2. **Dataset preprocessing:** The normalized experiment snapshots are identified by frozen row counts and SHA-256 hashes for all three datasets. The release does not provide a complete upstream-to-normalized conversion pipeline for the FeTaQA and OTT-QA snapshots and therefore supports compatibility verification, rather than rebuilding those snapshots from the original benchmark distributions.
