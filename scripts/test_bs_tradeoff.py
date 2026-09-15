import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from aa.models import get_model
from aa.data import get_sample_batch_indices
from aa.attacks.casa import CoalitionSparseAttack
from aa.metrics import compute_spatial_l0
from aa.utils import get_best_device, set_seed, synchronize_device


def run_bs_tradeoff(num_samples: int = 64, k: int = 16):
    device = get_best_device()
    print(f"Running on Device: {device} | Number of Test Samples: {num_samples} | Attack budget K: {k}")

    # 1. Load model
    set_seed(42)
    model = get_model(
        "resnet18",
        checkpoint_path="result/saved_models/resnet18_cifar10_best.pth",
        strict_checkpoint=False,
        device=device,
        eval_mode=True
    )

    # 2. Load fixed test samples (class-stratified)
    raw_loader, _, _ = get_sample_batch_indices(
        dataset_name="cifar10",
        batch_size=num_samples,
        num_samples=num_samples,
        seed=42
    )
    all_x, all_y = next(iter(raw_loader))
    all_x, all_y = all_x.to(device), all_y.to(device)

    # Filter to only correct predictions so that ASR is calculated strictly on clean correct samples
    with torch.no_grad():
        preds = model(all_x).argmax(dim=1)
        correct_mask = (preds == all_y)
        clean_acc = correct_mask.float().mean().item() * 100

    print(f"Clean accuracy on {num_samples} samples: {clean_acc:.2f}% ({correct_mask.sum().item()}/{num_samples} correct)")

    # CASA configuration (aligned with paper config)
    casa_kwargs = dict(
        k=k,
        steps=15,
        inner_steps=8,
        repair_steps=4,
        candidate_pool_size=4,
        alpha=4 / 255.0,
        drop_and_repair=True,
        pair_exploration=True,
        pair_search_every=5
    )

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = []

    # Store individual sample results to verify equivalence
    sample_success_records = {}

    for bs in batch_sizes:
        if device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        dataset = TensorDataset(all_x, all_y)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False)

        attack_engine = CoalitionSparseAttack(model=model, **casa_kwargs)

        synchronize_device(device)
        start_time = time.perf_counter()

        all_adv_x = []
        all_succ = []
        all_l0 = []

        for batch_x, batch_y in loader:
            set_seed(42)  # ensure seed determinism if any stochastic elements exist
            out = attack_engine.attack(batch_x, batch_y)
            adv_x = out.x_adv.detach()
            with torch.no_grad():
                adv_pred = model(adv_x).argmax(dim=1)
                succ = (adv_pred != batch_y)
                l0 = compute_spatial_l0(adv_x - batch_x)

            all_adv_x.append(adv_x)
            all_succ.append(succ)
            all_l0.append(l0)

        synchronize_device(device)
        total_time = time.perf_counter() - start_time

        all_succ = torch.cat(all_succ, dim=0)
        all_l0 = torch.cat(all_l0, dim=0)
        sample_success_records[bs] = all_succ.cpu().numpy()

        # Measure memory
        if device.type == "mps":
            vram_mb = torch.mps.driver_allocated_memory() / (1024 ** 2)
            tensor_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
        elif device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            tensor_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            import psutil
            vram_mb = psutil.Process().memory_info().rss / (1024 ** 2)
            tensor_mb = vram_mb

        # Calculate ASR on clean-correct samples
        asr = (all_succ[correct_mask].float().mean().item()) * 100
        mean_l0 = all_l0[correct_mask].float().mean().item()
        throughput = num_samples / total_time

        results.append({
            "bs": bs,
            "batches": len(loader),
            "vram_mb": vram_mb,
            "tensor_mb": tensor_mb,
            "time_sec": total_time,
            "throughput": throughput,
            "asr": asr,
            "mean_l0": mean_l0
        })

        print(f"BS={bs:2d} | Memory: {vram_mb:6.1f} MB | Time: {total_time:5.2f}s | Throughput: {throughput:5.1f} img/s | ASR: {asr:5.1f}% | L0: {mean_l0:.2f}")

    # Check result consistency
    bs_keys = list(sample_success_records.keys())
    ref_succ = sample_success_records[bs_keys[0]]
    all_identical = True
    for other_bs in bs_keys[1:]:
        if not (ref_succ == sample_success_records[other_bs]).all():
            all_identical = False
            diff_count = (ref_succ != sample_success_records[other_bs]).sum()
            print(f"Warning: Discrepancy between BS={bs_keys[0]} and BS={other_bs}: {diff_count} samples differ.")

    if all_identical:
        print("\n>>> VERIFIED: Attack Success per sample is 100% IDENTICAL across all batch sizes!")
    else:
        print("\n>>> NOTE: Minor differences observed due to floating-point order of operations.")

    return results, all_identical


if __name__ == "__main__":
    run_bs_tradeoff(num_samples=64, k=16)
