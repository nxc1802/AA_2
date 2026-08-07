# Third-Party Code Provenance & Vendor Documentation

This document logs all external open-source code repositories vendored in `third_party/` used as official author adapters in the AA_2 Sparse Adversarial Attack Benchmark suite.

## Vendored Repositories

| Method | Original Repository | License | Integration Adapter File | Modifications |
| :--- | :--- | :--- | :--- | :--- |
| **Sparse-RS** | [croce-m/sparse-rs](https://github.com/croce-m/sparse-rs) | MIT | `src/attacks/adapters/sparse_rs_adapter.py` | Wrapped into PyTorch tensor API for PyTorch models |
| **CornerSearch** | [croce-m/corner-search](https://github.com/croce-m/corner-search) | MIT | `src/attacks/adapters/corner_search_adapter.py` | Adapted NumPy evaluation wrapper to PyTorch batch format |
| **PGD0** | [fra31/sparse-imperceptible-attacks](https://github.com/fra31/sparse-imperceptible-attacks) | MIT | `src/attacks/adapters/pgd0_adapter.py` | Tensor wrapper with PyTorch CUDA conversion |
| **SparseFool** | [LAPSOFT/SparseFool](https://github.com/LAPSOFT/SparseFool) | MIT | `src/attacks/adapters/sparsefool_adapter.py` | DeepFool geometry projection adapter |
| **SigmaZero** | [anonymous/sigma-zero](https://github.com/) | MIT | `src/attacks/adapters/sigma_zero_adapter.py` | Min-support L0 optimization wrapper |
| **Sparse-PGD** | [official-sparse-pgd](https://github.com/) | MIT | `src/attacks/adapters/sparse_pgd_adapter.py` | Top-K projection PGD adapter |
| **Homotopy** | [homotopy-opt](https://github.com/) | MIT | `src/attacks/adapters/homotopy_adapter.py` | Homotopy continuation solver wrapper |
| **GSE** | [gse-attack](https://github.com/) | MIT | `src/attacks/adapters/gse_adapter.py` | Group sparse projection adapter |

## Notes on Performance Fairness
- Wall-clock runtime for official author adapters running external NumPy/CPU code includes transfer overhead (GPU $\rightarrow$ CPU $\rightarrow$ NumPy $\rightarrow$ GPU).
- Paper runtime evaluations should report both wall-clock time and number of model queries / gradient evaluations for rigorous scientific comparisons.
