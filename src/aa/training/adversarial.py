import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from aa.training.trainer import Trainer
from aa.training.checkpoint import CheckpointManager
from aa.training.history import TrainingHistory
from aa.utils import get_best_device


class PGDTrainAttack:
    """Fast inline PGD-L_inf attack generator optimized for training loops."""
    def __init__(
        self,
        eps: float = 8.0 / 255.0,
        alpha: float = 2.0 / 255.0,
        steps: int = 10,
        random_start: bool = True
    ):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        model.eval()
        x_adv = x.clone().detach()

        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.steps):
            x_adv.requires_grad = True
            outputs = model(x_adv)
            loss = criterion(outputs, y)

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            eta = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + eta, 0.0, 1.0)

        return x_adv.detach()


class SparsePGDTrainAttack:
    """Fast inline Sparse PGD (L_0 bounded) attack generator for training loops."""
    def __init__(
        self,
        k: int = 16,
        step_size: float = 0.2,
        steps: int = 10
    ):
        self.k = k
        self.step_size = step_size
        self.steps = steps

    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        model.eval()
        x_adv = x.clone().detach()
        batch_size, c, h, w = x.shape
        num_pixels = h * w

        # Calculate initial gradient on clean image to determine the top-k pixel mask
        x_temp = x.clone().detach()
        x_temp.requires_grad = True
        outputs = model(x_temp)
        loss = criterion(outputs, y)
        grad = torch.autograd.grad(loss, x_temp, retain_graph=False, create_graph=False)[0]
        grad_flat = grad.detach().abs().view(batch_size, c, -1).mean(dim=1)  # (batch_size, H*W)

        topk_vals, topk_indices = torch.topk(grad_flat, min(self.k, num_pixels), dim=1)
        mask = torch.zeros_like(grad_flat)
        mask.scatter_(1, topk_indices, 1.0)
        mask = mask.view(batch_size, 1, h, w)

        for _ in range(self.steps):
            x_adv.requires_grad = True
            outputs = model(x_adv)
            loss = criterion(outputs, y)

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            x_adv = x_adv.detach()
            update = self.step_size * grad.sign() * mask
            x_adv = torch.clamp(x_adv + update, 0.0, 1.0)

        return x_adv.detach()


class AdversarialTrainer(Trainer):
    """
    Adversarial Training Engine supporting PGD-L_inf and Sparse-PGD objectives.
    Supports pure adversarial loss and mixed loss Objectives:
      loss = (1 - lambda) * CE(f(x), y) + lambda * CE(f(x_adv), y)
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        attack_generator: Any,
        clean_weight: float = 0.5,
        adversarial_weight: float = 0.5,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        amp: bool = False,
        grad_accum_steps: int = 1,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            checkpoint_manager=checkpoint_manager,
            amp=amp,
            grad_accum_steps=grad_accum_steps
        )
        self.attack_generator = attack_generator
        self.clean_weight = clean_weight
        self.adversarial_weight = adversarial_weight

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, Any]:
        """Runs single adversarial training epoch."""
        start_time = time.time()
        running_loss = 0.0
        correct_clean = 0
        correct_adv = 0
        total = 0

        self.optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Generate adversarial examples (with model in eval mode internally)
            x_adv = self.attack_generator.generate(self.model, images, labels, self.criterion)

            # Train model forward pass
            self.model.train()

            if self.clean_weight > 0.0 and self.adversarial_weight > 0.0:
                outputs_clean = self.model(images)
                outputs_adv = self.model(x_adv)
                loss_clean = self.criterion(outputs_clean, labels)
                loss_adv = self.criterion(outputs_adv, labels)
                loss = self.clean_weight * loss_clean + self.adversarial_weight * loss_adv
            elif self.adversarial_weight > 0.0:
                outputs_adv = self.model(x_adv)
                loss = self.criterion(outputs_adv, labels)
                outputs_clean = outputs_adv
            else:
                outputs_clean = self.model(images)
                loss = self.criterion(outputs_clean, labels)
                outputs_adv = outputs_clean

            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            _, pred_clean = torch.max(outputs_clean, 1)
            _, pred_adv = torch.max(outputs_adv, 1)
            correct_clean += (pred_clean == labels).sum().item()
            correct_adv += (pred_adv == labels).sum().item()
            total += batch_size

        runtime = time.time() - start_time
        avg_loss = running_loss / max(1, total)
        clean_acc = 100.0 * correct_clean / max(1, total)
        adv_acc = 100.0 * correct_adv / max(1, total)

        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "loss": avg_loss,
            "accuracy": clean_acc,
            "adv_accuracy": adv_acc,
            "num_samples": total,
            "learning_rate": current_lr,
            "runtime_seconds": runtime,
        }
