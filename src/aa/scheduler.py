import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import List, Dict, Any, Optional, Callable
import torch.multiprocessing as mp
from aa.utils import set_seed
from aa.models import get_model
from aa.attacks import create_attack
from aa.attacks.base import AttackOutput


def _worker_attack_shard(
    gpu_id: int,
    model_name: str,
    checkpoint_path: Optional[str],
    expected_sha256: Optional[str],
    attack_name: str,
    attack_kwargs: dict,
    seed: int,
    dataset_name: str,
    sample_indices: List[int],
    batch_size: int,
    return_dict: dict,
    shard_rank: int,
    defense_spec: Optional[dict] = None
):
    """Worker process targeting specific GPU to evaluate an attack shard with primitive serializable args."""
    try:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        # Deterministic seed per worker
        set_seed(seed + gpu_id)

        model = get_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            expected_sha256=expected_sha256,
            device=device
        )
        model.eval()

        if defense_spec is not None:
            from aa.defenses import DEFENSES_MAP, DefendedModelAdapter
            def_name = defense_spec.get("name")
            def_mode = defense_spec.get("mode", "adaptive")
            def_obj = DEFENSES_MAP.get(def_name) if def_name else None
            attack_model = DefendedModelAdapter(model, defense=def_obj, mode=def_mode)
        else:
            attack_model = model

        attack = create_attack(attack_name, model=attack_model, **attack_kwargs)

        # Load dataset subset for this shard
        from aa.data import HFDatasetWrapper, get_dataset_transforms, HF_REPO_ID, HF_TOKEN
        from datasets import load_dataset
        ds_name = dataset_name.lower()
        hf_test = load_dataset(HF_REPO_ID, name=ds_name, split="test", token=HF_TOKEN)
        pt_dataset = HFDatasetWrapper(hf_test, transform=get_dataset_transforms(is_train=False))
        subset = Subset(pt_dataset, sample_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_adv = []
        total_queries = 0
        total_fwd = 0
        total_bwd = 0

        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out: AttackOutput = attack.attack(x, y) if hasattr(attack, "attack") else attack(x, y)
            all_adv.append(out.x_adv.detach().cpu())
            total_queries += getattr(out, "queries", 0)
            total_fwd += getattr(out, "forward_evals", 0)
            total_bwd += getattr(out, "backward_evals", 0)

        elapsed = time.time() - t0
        x_adv_cat = torch.cat(all_adv, dim=0) if len(all_adv) > 0 else torch.empty(0)
        return_dict[shard_rank] = {
            "x_adv": x_adv_cat,
            "queries": total_queries,
            "forward_evals": total_fwd,
            "backward_evals": total_bwd,
            "runtime": elapsed,
            "error": None
        }
    except Exception as e:
        return_dict[shard_rank] = {"error": str(e)}


class MultiGPUScheduler:
    """
    Parallel attack execution scheduler across multiple GPUs.
    Shards sample evaluation across available CUDA devices when CUDA GPU count > 1.
    """
    def __init__(self, gpus: Optional[List[int]] = None):
        if torch.cuda.is_available():
            available = list(range(torch.cuda.device_count()))
            if gpus is None:
                self.gpus = available
            else:
                self.gpus = [g for g in gpus if g in available]
        else:
            self.gpus = []

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)

    def is_multi_gpu(self) -> bool:
        return self.num_gpus > 1

    def run_sharded_attack(
        self,
        model_name: str,
        checkpoint_path: Optional[str],
        expected_sha256: Optional[str],
        attack_name: str,
        attack_kwargs: dict,
        seed: int,
        dataset_name: str,
        selected_sample_indices: List[int],
        batch_size: int = 512,
        defense_spec: Optional[dict] = None
    ) -> AttackOutput:
        """Runs sharded attack across multiple GPUs and aggregates AttackOutput in exact sample order."""
        if not self.is_multi_gpu():
            raise RuntimeError("MultiGPUScheduler called in non-multi-GPU environment.")

        total_samples = len(selected_sample_indices)
        num_workers = self.num_gpus

        # Split sample indices evenly among GPUs
        shard_size = (total_samples + num_workers - 1) // num_workers
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()
        processes = []

        t0_start = time.time()

        for rank, gpu_id in enumerate(self.gpus):
            shard_idx = selected_sample_indices[rank * shard_size : (rank + 1) * shard_size]
            if not shard_idx:
                continue
            p = ctx.Process(
                target=_worker_attack_shard,
                args=(
                    gpu_id,
                    model_name,
                    checkpoint_path,
                    expected_sha256,
                    attack_name,
                    attack_kwargs,
                    seed,
                    dataset_name,
                    shard_idx,
                    batch_size,
                    return_dict,
                    rank,
                    defense_spec
                )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        t_total_gen = time.time() - t0_start

        # Collect and combine results in strict rank order (0..len(processes)-1)
        all_x_adv = []
        total_queries = 0
        total_fwd = 0
        total_bwd = 0

        for rank in range(len(processes)):
            res = return_dict.get(rank)
            if res is None or res.get("error") is not None:
                err = res.get("error") if res else "Worker process failed"
                raise RuntimeError(f"Multi-GPU worker rank {rank} failed: {err}")
            all_x_adv.append(res["x_adv"])
            total_queries += res["queries"]
            total_fwd += res["forward_evals"]
            total_bwd += res["backward_evals"]

        x_adv_combined = torch.cat(all_x_adv, dim=0)
        return AttackOutput(
            x_adv=x_adv_combined,
            queries=total_queries,
            forward_evals=total_fwd,
            backward_evals=total_bwd,
            metadata={"attack_generation_runtime": t_total_gen}
        )
