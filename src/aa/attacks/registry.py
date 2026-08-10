import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Any
import torch.nn as nn

from aa.attacks.base import Attack
from aa.attacks.dense import FGSM, BIM, PGD
from aa.attacks.proposed import SparseFeatureAttack
from aa.attacks.casa import CoalitionSparseAttack
from aa.attacks.external.cornersearch import CornerSearch
from aa.attacks.external.pgd0 import PGD0
from aa.attacks.external.spgd import SparsePGD
from aa.attacks.external.sparse_rs import SparseRS
from aa.attacks.external.sparsefool import SparseFool
from aa.attacks.external.sigma_zero import SigmaZero
from aa.attacks.external.gse import GSE


@dataclass
class AttackSpec:
    name: str
    factory: Callable[..., Attack]
    mode: str  # "dense" | "budget" | "minimal" | "progressive"


ATTACK_REGISTRY: Dict[str, AttackSpec] = {
    "fgsm": AttackSpec(name="FGSM", factory=FGSM, mode="dense"),
    "bim": AttackSpec(name="BIM", factory=BIM, mode="dense"),
    "pgd": AttackSpec(name="PGD", factory=PGD, mode="dense"),
    "cornersearch": AttackSpec(name="CornerSearch", factory=CornerSearch, mode="progressive"),
    "pgd0": AttackSpec(name="PGD0", factory=PGD0, mode="budget"),
    "spgd": AttackSpec(name="Sparse-PGD", factory=SparsePGD, mode="budget"),
    "sparse_rs": AttackSpec(name="Sparse-RS", factory=SparseRS, mode="budget"),
    "sparsefool": AttackSpec(name="SparseFool", factory=SparseFool, mode="minimal"),
    "sigma_zero": AttackSpec(name="Sigma-Zero", factory=SigmaZero, mode="minimal"),
    "gse": AttackSpec(name="GSE", factory=GSE, mode="minimal"),
    "ours": AttackSpec(name="CASA", factory=CoalitionSparseAttack, mode="budget"),
    "casa": AttackSpec(name="CASA", factory=CoalitionSparseAttack, mode="budget"),
    "ours_v2": AttackSpec(name="CASA", factory=CoalitionSparseAttack, mode="budget"),
    "ours_v1": AttackSpec(name="Ours-v1", factory=SparseFeatureAttack, mode="budget"),
}


def get_attack_spec(name: str) -> AttackSpec:
    key = name.lower()
    if key not in ATTACK_REGISTRY:
        raise ValueError(f"Attack '{name}' not found in registry. Options: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[key]


def create_attack(name: str, model: nn.Module, strict: bool = True, **kwargs) -> Attack:
    """Instantiates an attack by name from the registry.

    Args:
        name: Attack name (key in ATTACK_REGISTRY).
        model: PyTorch model to attack.
        strict: If True (default), raise ValueError for unknown kwargs instead of
                silently dropping them. Set False only for exploratory/debug use.
        **kwargs: Attack hyperparameters.

    Raises:
        ValueError: If ``strict=True`` and unknown kwargs are passed.
    """
    spec = get_attack_spec(name)

    sig = inspect.signature(spec.factory.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "model"}

    unknown = {k for k in kwargs if k not in valid_params}
    if unknown and strict:
        raise ValueError(
            f"Unknown kwargs for attack '{name}': {unknown}. "
            f"Valid parameters: {sorted(valid_params)}"
        )

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return spec.factory(model=model, **filtered_kwargs)

