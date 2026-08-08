import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src" / "attacks" / "attack_registry.yaml"
THIRD_PARTY_DOC = ROOT / "THIRD_PARTY.md"


def _load_attacks():
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict) and isinstance(data.get("attacks"), dict)
    return data["attacks"]


def test_unverified_baselines_are_not_main_benchmark():
    attacks = _load_attacks()
    for name, meta in attacks.items():
        status = str(meta.get("validation_status", ""))
        if status.startswith("unverified") or status.startswith("appendix"):
            assert meta.get("main_benchmark") is False, (
                f"{name} is {status!r} but is enabled for the main benchmark"
            )


def test_null_paper_identity_cannot_enter_main_benchmark():
    attacks = _load_attacks()
    for name, meta in attacks.items():
        if meta.get("paper") is None:
            assert meta.get("main_benchmark") is False, (
                f"{name} has no verified paper identity but is marked main_benchmark=true"
            )


def test_official_adapters_have_real_upstream_metadata():
    attacks = _load_attacks()
    sha40 = re.compile(r"^[0-9a-f]{40}$")

    for name, meta in attacks.items():
        if meta.get("implementation") != "official-adapter":
            continue

        upstream = meta.get("upstream_repo")
        assert isinstance(upstream, str) and upstream.startswith("https://github.com/")
        assert upstream != "https://github.com/"

        commit = meta.get("upstream_commit")
        status = str(meta.get("validation_status", ""))
        if commit is None:
            assert "modified" in status or "unpinned" in status, (
                f"{name} has no pinned upstream commit without an explicit modified/unpinned status"
            )
        else:
            assert sha40.fullmatch(commit), f"{name} has invalid upstream commit: {commit!r}"


def test_third_party_document_has_no_generic_github_placeholders():
    text = THIRD_PARTY_DOC.read_text(encoding="utf-8")
    # The old provenance table used bare https://github.com/ placeholders.
    assert "`https://github.com/`" not in text
    assert "(https://github.com/)" not in text


def test_modified_upstream_adapter_commit_must_be_null_or_unpinned():
    attacks = _load_attacks()
    for name, meta in attacks.items():
        if meta.get("implementation") == "modified-upstream-adapter":
            assert meta.get("upstream_commit") is None, (
                f"{name} is marked modified-upstream-adapter but claims an exact upstream_commit SHA"
            )


def test_paper_protocol_main_benchmark_gate():
    from src.benchmark.run_attack_benchmark import is_paper_eligible
    attacks = _load_attacks()
    for name, meta in attacks.items():
        main_bm = meta.get("main_benchmark", False)
        eligible = is_paper_eligible(name)
        assert eligible == main_bm, (
            f"Mismatch for {name}: registry main_benchmark={main_bm} but is_paper_eligible={eligible}"
        )
