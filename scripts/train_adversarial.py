import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from aa.utils import set_seed, get_best_device
from aa.data import get_dataloaders
from aa.models import get_model
from aa.training import (
    AdversarialTrainer,
    CheckpointManager,
    TrainingHistory,
    create_optimizer,
    create_scheduler,
)
from aa.training.adversarial import PGDTrainAttack, SparsePGDTrainAttack


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Model Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_cfg = config.get("experiment", {})
    ds_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    opt_cfg = config.get("optimizer", {})
    sched_cfg = config.get("scheduler", {})
    ckpt_cfg = config.get("checkpoint", {})
    adv_cfg = config.get("adversarial_training", {})

    seed = args.seed if args.seed is not None else exp_cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device(args.device) if args.device else get_best_device()

    dataset_name = ds_cfg.get("name", "cifar10")
    train_batch_size = train_cfg.get("train_batch_size", 128)
    eval_batch_size = train_cfg.get("eval_batch_size", 256)
    num_workers = ds_cfg.get("num_workers", 0)

    attack_type = adv_cfg.get("attack", "pgd").lower()
    attack_kwargs = adv_cfg.get("attack_kwargs", {})

    print(f"--- Launching Adversarial Training ---")
    print(f"Experiment: {exp_cfg.get('name', 'unnamed')}")
    print(f"Attack Type: {attack_type} | Dataset: {dataset_name} | Model: {model_cfg.get('name', 'resnet18')}")

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed
    )

    model = get_model(
        model_name=model_cfg.get("name", "resnet18"),
        dataset_name=dataset_name,
        pretrained=model_cfg.get("pretrained", False),
        strict_checkpoint=False,
        eval_mode=False,
        device=device
    )

    if attack_type == "pgd":
        attack_generator = PGDTrainAttack(**attack_kwargs)
    elif attack_type in ["spgd", "sparse_pgd", "pgd0"]:
        attack_generator = SparsePGDTrainAttack(**attack_kwargs)
    else:
        raise ValueError(f"Unsupported training attack type: {attack_type}")

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 200)
    optimizer = create_optimizer(model, opt_cfg)
    scheduler = create_scheduler(optimizer, sched_cfg, total_epochs=epochs)

    ckpt_dir = ckpt_cfg.get("directory", "result/checkpoints")
    exp_name = exp_cfg.get("name", "adv_experiment")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        experiment_name=exp_name,
        dataset_name=dataset_name,
        architecture=model_cfg.get("name", "resnet18"),
        seed=seed,
        config=config,
        monitor=ckpt_cfg.get("monitor", "val_accuracy"),
        mode=ckpt_cfg.get("mode", "max")
    )

    start_epoch = 0
    if args.resume:
        print(f"Resuming adversarial training from checkpoint: {args.resume}")
        start_epoch = checkpoint_manager.load_resume(args.resume, model, optimizer, scheduler, device=device)

    trainer = AdversarialTrainer(
        model=model,
        optimizer=optimizer,
        attack_generator=attack_generator,
        clean_weight=adv_cfg.get("clean_weight", 0.5),
        adversarial_weight=adv_cfg.get("adversarial_weight", 0.5),
        scheduler=scheduler,
        device=device,
        checkpoint_manager=checkpoint_manager,
        amp=train_cfg.get("amp", False),
        grad_accum_steps=train_cfg.get("grad_accum_steps", 1)
    )

    history = TrainingHistory()
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        start_epoch=start_epoch,
        history=history
    )

    # Evaluate best checkpoint on test set
    best_path = os.path.join(checkpoint_manager.experiment_dir, "best.pth")
    if os.path.isfile(best_path):
        print(f"Loading best checkpoint for test set evaluation: {best_path}")
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = trainer.evaluate(test_loader)
    print(f"Final Clean Test Accuracy: {test_metrics['accuracy']:.2f}% | Test Loss: {test_metrics['loss']:.4f}")

    # Export history and summary
    history.save_json(os.path.join(checkpoint_manager.experiment_dir, "history.json"))
    history.save_csv(os.path.join(checkpoint_manager.experiment_dir, "history.csv"))
    history.save_summary(
        summary_path=os.path.join(checkpoint_manager.experiment_dir, "summary.json"),
        best_epoch=checkpoint_manager.best_epoch,
        best_val_accuracy=checkpoint_manager.best_metric,
        test_accuracy=test_metrics["accuracy"],
        extra_metadata={
            "experiment_name": exp_name,
            "attack_type": attack_type,
            "dataset": dataset_name,
            "architecture": model_cfg.get("name", "resnet18"),
            "seed": seed
        }
    )
    print(f"Adversarial training completed. Results saved in: {checkpoint_manager.experiment_dir}")


if __name__ == "__main__":
    main()
