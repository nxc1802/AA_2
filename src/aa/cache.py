import os
import json
import hashlib
import torch
from typing import Optional, Dict, Any
from aa.attacks.base import AttackOutput


class AttackArtifactCache:
    """
    Artifact Cache for adversarial attack outputs.
    
    Caches x_adv tensors, query/eval counters, and attack metadata on disk based on
    a deterministic content hash of attack configuration, dataset hash, model ID, and seed.
    """
    def __init__(self, cache_dir: str = "result/attack_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def compute_cache_key(
        dataset_hash: str,
        model_identifier: str,
        attack_name: str,
        attack_kwargs: Dict[str, Any],
        seed: int,
        k: Optional[int] = None
    ) -> str:
        """Computes deterministic SHA256 cache key from attack execution parameters."""
        serialized_kwargs = json.dumps(attack_kwargs, sort_keys=True, default=str)
        payload = f"{dataset_hash}|{model_identifier}|{attack_name}|{serialized_kwargs}|{seed}|{k}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_filepath(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.pt")

    def has(self, cache_key: str) -> bool:
        """Returns True if cached artifact exists for the given key."""
        return os.path.exists(self._get_filepath(cache_key))

    def get(self, cache_key: str, device: Optional[torch.device] = None) -> Optional[AttackOutput]:
        """Loads cached AttackOutput from disk if available."""
        filepath = self._get_filepath(cache_key)
        if not os.path.exists(filepath):
            return None
        try:
            data = torch.load(filepath, map_location=device or "cpu", weights_only=False)
            return AttackOutput(
                x_adv=data["x_adv"],
                queries=data.get("queries", 0),
                forward_evals=data.get("forward_evals", 0),
                backward_evals=data.get("backward_evals", 0),
                metadata=data.get("metadata", {})
            )
        except Exception as e:
            print(f"⚠️ Failed to read attack artifact cache '{filepath}': {e}", flush=True)
            return None

    def put(self, cache_key: str, output: AttackOutput) -> None:
        """Saves AttackOutput artifact to disk."""
        filepath = self._get_filepath(cache_key)
        data = {
            "x_adv": output.x_adv.detach().cpu(),
            "queries": getattr(output, "queries", 0),
            "forward_evals": getattr(output, "forward_evals", 0),
            "backward_evals": getattr(output, "backward_evals", 0),
            "metadata": getattr(output, "metadata", {})
        }
        tmp_filepath = filepath + ".tmp"
        torch.save(data, tmp_filepath)
        os.replace(tmp_filepath, filepath)
