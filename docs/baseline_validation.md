# Baseline Validation Policy

This document defines which external/custom attack implementations are eligible for the AA_2 paper-facing main benchmark.

## Policy

A baseline is eligible for the **main benchmark** only when at least one of the following is true:

1. the benchmark uses a verified official-author implementation through an adapter;
2. the method is a canonical/simple reference algorithm (for example FGSM/BIM/PGD) whose implementation contract is straightforward and independently testable; or
3. a custom reimplementation has reproduced the paper protocol/results within a documented tolerance.

A custom implementation is **not** promoted to the main benchmark merely because it runs or satisfies the L0 budget contract. Paper identity, threat model, optimization protocol, stopping rule, query/gradient accounting, and reported behavior must also be validated.

Status meanings:

- `main`: eligible for paper-facing benchmark tables.
- `appendix`: useful as a secondary/exploratory reference, but not yet strong enough to support a main-table claim.
- `exclude`: do not use for claims against the proposed method until the stated validation blocker is resolved.

## Custom-only audit

| Method | Paper identity | Current implementation | Decision | Required action before promotion |
|---|---|---|---|---|
| FGSM | Verified canonical method | Custom | **main** (dense reference only) | Keep contract tests for epsilon/input scaling. |
| BIM | Verified canonical method | Custom | **main** (dense reference only) | Keep contract tests for epsilon/step schedule. |
| PGD | Verified canonical method | Custom | **main** (dense reference only) | Keep contract tests for epsilon/restarts/step schedule. |
| SFA | Previous registry citation could not be tied confidently to the current FFT top-k implementation | Custom | **exclude** | Identify a primary-source paper and reproduce its algorithm, or relabel this implementation as an internal exploratory attack. |
| JSMA | Verified published method | Custom | **appendix** | Reproduce a published/reference implementation under the same model and threat model; document parameter mapping. |
| OnePixel | Verified published method | Custom | **appendix** | Validate differential-evolution/query protocol against a trusted implementation or published setting. |
| SAIF | Paper identity corrected to **SAIF: Sparse Adversarial and Imperceptible Attack Framework (TMLR 2025)** | Custom; `third_party/saif` is only a placeholder README | **exclude** | Replace/validate the current implementation against the actual SAIF algorithm and reproduce a paper result before use. |
| BruSLe | Published method identity is credible | Custom | **appendix** | Prefer an author/released implementation if available; otherwise reproduce a reported result and document query-budget parity. |
| IPFSA | Previous registry title/venue did not map reliably to the current white-box Laplacian/gradient implementation | Custom | **exclude** | Establish the exact primary source and algorithm identity; otherwise treat as an internal method, not a literature baseline. |
| GradientGuidance | Previous registry paper citation could not be verified as the source of the current implementation | Custom | **exclude** | Establish a primary source and reproduce the method; otherwise remove literature-baseline claims. |
| Pixle | Verified published black-box method family | Custom | **appendix** | Validate swap/query semantics and reproduce a trusted implementation/paper setting. |

## Consequences for reporting

- `main_benchmark: false` in `src/attacks/attack_registry.yaml` means the method must not be used to support a main-table SOTA/baseline claim.
- Excluded implementations may remain in the repository for debugging and exploratory experiments, but their output must be labeled as non-canonical.
- Moving an `appendix` or `exclude` method to `main` requires an evidence note containing: exact paper, exact implementation source, parameter mapping, model/dataset setting, and a reproduction comparison.
- Official adapters remain preferred whenever trustworthy author code is available. Custom duplicates are validation aids, not canonical baselines.

## Validation record template

For each future promotion, record:

```yaml
method: METHOD_NAME
paper: PAPER_TITLE
paper_url: URL
implementation_source: official | trusted-independent | custom-reimplementation
upstream_repo: URL_OR_NULL
upstream_commit: SHA_OR_NULL
parameter_mapping:
  benchmark_parameter: upstream_parameter
reproduction:
  dataset: DATASET
  model: MODEL
  metric: METRIC
  paper_or_reference_value: VALUE
  reproduced_value: VALUE
  tolerance: VALUE
reviewed_at: YYYY-MM-DD
```
