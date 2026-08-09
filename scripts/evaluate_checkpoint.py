import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from aa.utils import compute_file_sha256, get_best_device
from aa.data import get_dataloaders
from aa.models import get_model, evaluate_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Checkpoint & Certify Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name override (e.g. cifar10)")
    parser.add_argument("--architecture", type=str, default=None, help="Architecture override (e.g. resnet18)")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--json", action="store_true", help="Print result in JSON format")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    device = torch.device(args.device) if args.device else get_best_device()
    sha256 = compute_file_sha256(ckpt_path)

    checkpoint_payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    dataset_name = args.dataset or (
        checkpoint_payload.get("dataset") if isinstance(checkpoint_payload, dict) else "cifar10"
    )
    architecture = args.architecture or (
        checkpoint_payload.get("architecture") if isinstance(checkpoint_payload, dict) else "resnet18"
    )
    git_commit = checkpoint_payload.get("git_commit", "unknown") if isinstance(checkpoint_payload, dict) else "unknown"
    seed = checkpoint_payload.get("seed", 42) if isinstance(checkpoint_payload, dict) else 42

    _, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        eval_batch_size=512,
        num_workers=0,
        seed=seed
    )

    model = get_model(
        model_name=architecture,
        dataset_name=dataset_name,
        checkpoint_path=ckpt_path,
        strict_checkpoint=True,
        device=device
    )

    val_acc = evaluate_accuracy(model, val_loader, device=device)
    test_acc = evaluate_accuracy(model, test_loader, device=device)

    result = {
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": sha256,
        "architecture": architecture,
        "dataset": dataset_name,
        "seed": seed,
        "git_commit": git_commit,
        "validation_accuracy": round(val_acc, 2),
        "test_accuracy": round(test_acc, 2),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=== Checkpoint Evaluation & Certification ===")
        print(f"File: {ckpt_path}")
        print(f"SHA256: {sha256}")
        print(f"Architecture: {architecture} | Dataset: {dataset_name} | Seed: {seed}")
        print(f"Git Commit: {git_commit}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Test Accuracy:       {test_acc:.2f}%")


if __name__ == "__main__":
    main()
