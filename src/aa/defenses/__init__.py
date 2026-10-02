from aa.defenses.base import BaseDefense
from aa.defenses.pipeline import DefensePipeline
from aa.defenses.bpda import BPDAFunction
from aa.defenses.adapters import DefendedModelAdapter
from aa.defenses.preprocessing import (
    GaussianBlurDefense,
    gaussian_blur,
    MedianFilterDefense,
    median_filter,
    JPEGDefense,
    jpeg_compression,
    TVMDefense,
    total_variation_minimization,
)

DEFENSES_MAP = {
    "gaussian": GaussianBlurDefense(),
    "blur": GaussianBlurDefense(),
    "median": MedianFilterDefense(),
    "jpeg": JPEGDefense(),
    "tvm": TVMDefense(),
}


def get_defense(name: str) -> BaseDefense:
    key = name.lower()
    if key not in DEFENSES_MAP:
        raise ValueError(f"Unknown defense '{name}'. Supported: {list(DEFENSES_MAP.keys())}")
    return DEFENSES_MAP[key]


__all__ = [
    "BaseDefense",
    "DefensePipeline",
    "BPDAFunction",
    "DefendedModelAdapter",
    "GaussianBlurDefense",
    "gaussian_blur",
    "MedianFilterDefense",
    "median_filter",
    "JPEGDefense",
    "jpeg_compression",
    "TVMDefense",
    "total_variation_minimization",
    "DEFENSES_MAP",
    "get_defense",
]
