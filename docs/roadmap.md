# Development Roadmap

This document tracks completed and remaining tasks for AA_2.

---

## Refactor Phases & Status

- [x] **Phase 1 — Freeze & Tag Current Repo** (`git tag pre-minimal-refactor`)
- [x] **Phase 2 — Package Setup** (`pyproject.toml`, `pip install -e .`)
- [x] **Phase 3 — Core Package** (`src/aa/metrics.py`, `data.py`, `models.py`, `defenses.py`, `utils.py`)
- [x] **Phase 4 — Attack Interface & Dense Baselines** (`src/aa/attacks/base.py`, `dense.py`)
- [x] **Phase 5 — External Baselines Adapters** (`src/aa/attacks/external/`)
- [x] **Phase 6 — Proposed Method & Registry** (`src/aa/attacks/proposed.py`, `registry.py`)
- [x] **Phase 7 — Generic Benchmark Engine** (`src/aa/benchmark.py`, `scripts/attack_benchmark.py`, `scripts/defense_benchmark.py`)
- [x] **Phase 8 — Unit Tests Suite** (`tests/test_core.py`, `tests/test_attacks.py`, `tests/test_benchmark.py`)
- [x] **Phase 9 — Documentation Consolidation** (`docs/protocol.md`, `docs/proposed_method.md`, `docs/roadmap.md`)
