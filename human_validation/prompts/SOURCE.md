# Semantic-Judge Prompt Provenance

Status: **RESOLVED**

The two files in this directory are the exact original semantic-judge prompt files used in the completed paper experiment. They are preserved verbatim and must not be modified, reformatted, normalized, or reconstructed.

| Role | Verbatim prompt file | SHA-256 |
| --- | --- | --- |
| System | `llm_exact_match_system.txt` | `a18410e4c3cfd80464eb8975ace96f1d0dc819024bfe5620a13fbd32022b42c5` |
| User | `llm_exact_match_user.txt` | `81d812fc77e321d1c6fc55e6e81ec9dde4d746062d3edfae11ebf93e70acf019` |

The completed experiment used this judge configuration:

| Setting | Value |
| --- | --- |
| Judge model | GPT-OSS-20b |
| Temperature | `0.0` |
| Top-p | `1.0` |
| Maximum new tokens | `10` |

These files are retained for provenance only. The semantic judge has already been run and will not be rerun.
