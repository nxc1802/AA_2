# Third-Party Code Provenance & Vendor Documentation

This file records the provenance of code under `third_party/` that is used, or intended to be used, by AA_2 official-author adapters.

## Provenance rules

- `Exact tree` means the vendored directory tree SHA matches an upstream commit tree SHA.
- `Core file verified` means the file actually imported by the AA_2 adapter has the same Git blob SHA as the file at the stated upstream commit. The vendored directory may intentionally contain only a subset of upstream files.
- `Modified snapshot` means the folder is derived from an upstream repository but the current vendored files do not identify a single exact upstream commit. Such code must not be described as an exact copy.
- `Unverified` means provenance is insufficient for a paper-facing official-code claim.
- A license is recorded only when it is present/detectable in the upstream repository. Absence of a detected license is **not** treated as MIT by default.

## Vendored repositories

| Method(s) | Vendored path | Upstream repository | Pinned upstream commit | Verification | Upstream license | Adapter |
|---|---|---|---|---|---|---|
| Sparse-RS | `third_party/sparse_rs` | `https://github.com/fra31/sparse-rs` | `21d875969a1455e4d5b26dcf32c843e6262d1f9c` | **Exact tree** (`7534582b9cf676541c4abed8d468437d761e7268`) | MIT | `src/attacks/adapters/sparse_rs_adapter.py` |
| CornerSearch, PGD0 | `third_party/sparse_imperceivable_attacks` | `https://github.com/fra31/sparse-imperceivable-attacks` | `57c23ead05803a631c332be93f824ebd8020385d` | **Core files verified**; e.g. `pgd_attacks_pt.py` blob `9ebd92927a60a33af401cf43efed0e610bab5350` matches upstream | **No license detected upstream** | `cornersearch_adapter.py`, `pgd0_adapter.py` |
| Sparse-PGD | `third_party/spgd` | `https://github.com/CityU-MLO/sPGD` | `37564941d11a9a72c4c4fa2b07299f30dae26154` | **Core files verified**; `adversarial_training/spgd.py` blob `a9ac38b85292fb59e570ed21af4259e1b243b2b0` matches upstream | **No license detected upstream** | `src/attacks/adapters/spgd_adapter.py` |
| SparseFool | `third_party/sparsefool` | `https://github.com/LTS4/SparseFool` | `958e2bf663ea4b22ecc7a24dcf226a9fab77124f` | **Core file verified**; `sparsefool.py` blob `26ddb964c28fad93bcd0973a5b8d17ebadca4d94` matches upstream | Apache-2.0 | `src/attacks/adapters/sparsefool_adapter.py` |
| Sigma-Zero | `third_party/sigma_zero` | `https://github.com/sigma0-advx/sigma-zero` | `f59494ca5fdb8f041e618381eb925e0bec00b01e` | **Core file verified**; `sigma_zero_attack.py` blob `1a310f07ce4eb292ade383859d096263141803f6` matches upstream | **No license detected upstream** | `src/attacks/adapters/sigma_zero_adapter.py` |
| Homotopy | `third_party/sparseadv_homotopy` | `https://github.com/VITA-Group/SparseADV_Homotopy` | **Unpinned** | **Modified snapshot**; current `demo_attack.py` blob differs from current upstream, so no exact-commit claim is made | MIT upstream | `src/attacks/adapters/homotopy_adapter.py` |
| GSE | `third_party/gse` | `https://github.com/wagnermoritz/GSE` | `0123cdf6b88f9313f40886aa13631dc82ff43874` | **Exact tree** (`1d40bfb42f13f90938ba5b11346ff73bc3db3271`) | **No license detected upstream** | `src/attacks/adapters/gse_adapter.py` |
| SAIF placeholder | `third_party/saif` | **Unverified** | **Unpinned** | **Not an official-code snapshot**: the directory currently contains only a short README placeholder | **Unverified** | none |

## Corrections from the previous provenance table

The previous version contained placeholder repository URLs and assigned `MIT` to every vendored dependency. Both practices are unsafe for a reproducibility artifact.

This revision therefore:

1. replaces placeholder URLs with verified upstream repositories where identified;
2. records concrete commits only when supported by repository/file identity evidence;
3. distinguishes exact-tree copies from partial/core-file matches and modified snapshots;
4. removes unsupported license claims;
5. explicitly marks the SAIF folder as a placeholder rather than official code.

## Adapter modifications

AA_2 adapters are glue code and are not claimed to be upstream files. Their allowed responsibilities are:

- convert AA_2 PyTorch tensors/model interfaces to the upstream API;
- move tensors between devices/formats when required;
- normalize return values to the AA_2 attack interface;
- expose benchmark statistics when this does not change the attack algorithm;
- provide compatibility shims for missing package imports when algorithm behavior is preserved.

Any change to the upstream optimization objective, update rule, projection, stopping rule, restart policy, or query logic must be documented separately and invalidates an `official-author-code` claim until revalidated.

## Paper-facing policy

For the main paper benchmark:

- prefer verified official-author adapters whenever trustworthy upstream code exists;
- do not silently substitute a custom reimplementation for an official baseline;
- do not use a vendored dependency with `Unverified` provenance to support a main-table SOTA claim;
- record the exact AA_2 commit and the upstream commit(s) used for every released result table.

See `docs/baseline_validation.md` for custom-only baseline eligibility and promotion requirements.

## Runtime fairness

Wall-clock runtime for official adapters may include CPU/NumPy conversion or device-transfer overhead inherited from upstream code. Paper reporting should therefore include wall-clock time **and** algorithm-appropriate model-query / gradient-evaluation counts rather than treating runtime alone as a fair cross-method efficiency metric.
