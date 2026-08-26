# Final Error Analysis (`rule_order_v2`)

Only the final paper-facing path is released. Labels are mutually exclusive and assigned in this fixed order:

1. **Distractor Extraction:** exact normalized match to a distractor data-cell value absent from the gold table.
2. **Premature Refusal:** explicit refusal or insufficient-information phrase among remaining cases.
3. **Reasoning Hallucination:** all remaining cases.

LLM evidence is auxiliary only. `Error Analysis - Error Type_v2.csv`, sanitized classifications, crosstabs, and Figure 6 scripts are the released path. Raw text and provider data are excluded.
