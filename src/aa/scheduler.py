import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import List, Dict, Any, Optional, Callable
import torch.multiprocessing as mp
from aa.attacks.base import AttackOutput


def _worker_attack_shard(
    gpu_id: int,
    model_fn: Callable[[], nn.Module],
    attack_factory: Callable[[nn.Module], Any],
    dataset: Any,
    shard_indices: List[int],
    batch_size: int,
    return_dict: dict,
    shard_rank: int
):
    """Worker process targeting specific GPU to evaluate an attack shard."""
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        model = model_fn().to(device)
        model.eval()

        attack = attack_factory(model)
        subset = Subset(dataset, shard_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_adv = []
        total_queries = 0
        total_fwd = 0
        total_bwd = 0

        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out: AttackOutput = attack.attack(x, y) if hasattr(attack, "attack") else attack(x, y)
            all_adv.append(out.x_adv.cpu())
            total_queries += getattr(out, "queries", 0)
            total_fwd += getattr(out, "forward_evals", 0)
            total_bwd += getattr(out, "backward_evals", 0)

        x_adv_cat = torch.cat(all_adv, dim=0) if len(all_adv) > 0 else torch.empty(0)
        return_dict[shard_rank] = {
            "x_adv": x_adv_cat,
            "queries": total_queries,
            "forward_evals": total_fwd,
            "backward_evals": total_bwd,
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
        model_fn: Callable[[], nn.Module],
        attack_factory: Callable[[nn.Module], Any],
        dataset: Any,
        batch_size: int = 512
    ) -> AttackOutput:
        """Runs sharded attack across multiple GPUs and aggregates AttackOutput."""
        if not self.is_multi_gpu():
            raise RuntimeError("MultiGPUScheduler called in non-multi-GPU environment.")

        total_samples = len(dataset)
        num_workers = self.num_gpus
        indices = list(range(total_samples))

        # Split indices evenly among GPUs
        shard_size = (total_samples + num_workers - 1) // num_workers
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()
        processes = []

        for rank, gpu_id in enumerate(self.gpus):
            shard_idx = indices[rank * shard_size : (rank + 1) * shard_size]
            if not shard_idx:
                continue
            p = ctx.Process(
                target=_worker_attack_shard,
                args=(
                    gpu_id,
                    model_fn,
                    attack_factory,
                    dataset,
                    shard_idx,
                    batch_size,
                    return_dict,
                    rank
                )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Collect and combine results in order
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
            backward_evals=total_bwd
        )
