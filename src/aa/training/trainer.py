import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from aa.training.checkpoint import CheckpointManager
from aa.training.history import TrainingHistory
from aa.utils import get_best_device


class Trainer:
    """
    Decoupled clean training engine for PyTorch image classification models.
    Supports mixed precision (AMP), gradient accumulation, evaluation, and checkpoint management.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        amp: bool = False,
        grad_accum_steps: int = 1,
    ):
        self.device = device or get_best_device()
        self.model = model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.checkpoint_manager = checkpoint_manager
        self.amp = amp
        self.grad_accum_steps = max(1, grad_accum_steps)

        from aa.utils import enable_gpu_optimizations
        enable_gpu_optimizations()

        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and torch.cuda.is_available())
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp and torch.cuda.is_available())

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, Any]:
        """Trains for a single epoch and returns detailed metrics dict."""
        self.model.train()
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.amp and torch.cuda.is_available():
                if hasattr(torch.amp, "autocast"):
                    autocast_ctx = torch.amp.autocast("cuda")
                else:
                    autocast_ctx = torch.cuda.amp.autocast()
                with autocast_ctx:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    scaled_loss = loss / self.grad_accum_steps
                self.scaler.scale(scaled_loss).backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                scaled_loss = loss / self.grad_accum_steps
                scaled_loss.backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += batch_size

        runtime = time.time() - start_time
        avg_loss = running_loss / max(1, total)
        accuracy = 100.0 * correct / max(1, total)

        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": total,
            "learning_rate": current_lr,
            "runtime_seconds": runtime,
        }

    def evaluate(self, eval_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Evaluates model performance on validation or test dataset."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += batch_size

        avg_loss = running_loss / max(1, total)
        accuracy = 100.0 * correct / max(1, total)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": total,
        }

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 200,
        start_epoch: int = 0,
        history: Optional[TrainingHistory] = None
    ) -> TrainingHistory:
        """Executes full multi-epoch training loop."""
        if history is None:
            history = TrainingHistory()

        for epoch in range(start_epoch + 1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
            else:
                val_metrics = {"loss": 0.0, "accuracy": 0.0}

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "learning_rate": train_metrics["learning_rate"],
                "runtime_seconds": train_metrics["runtime_seconds"],
            }
            history.add_epoch(epoch_record)

            is_best = False
            monitored_val = val_metrics["accuracy"] if val_loader is not None else train_metrics["accuracy"]

            if self.checkpoint_manager is not None:
                if self.checkpoint_manager.is_better(monitored_val):
                    is_best = True
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    current_metric=monitored_val,
                    is_best=is_best
                )

            val_str = f" | Val Acc: {val_metrics['accuracy']:.2f}%" if val_loader is not None else ""
            best_str = " (Best)" if is_best else ""
            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%"
                f"{val_str}{best_str} | LR: {train_metrics['learning_rate']:.6f} | "
                f"Time: {train_metrics['runtime_seconds']:.1f}s"
            )

        return history
