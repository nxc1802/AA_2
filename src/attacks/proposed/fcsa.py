import torch
import torch.nn as nn
from src.core.projections import project_l0, exact_spatial_topk_mask
from src.core.utils import prepare_model_for_eval, get_best_device

class FunctionalCoalitionSparseAttack:
    """
    Functional Coalition Sparse Attack (FCSA).
    Evaluates joint functional coalition contribution score: Score(S) = grad_mean * grad_max.
    Strictly projects perturbation onto exact K-sparse spatial L0 ball.
    """
    def __init__(self, model: nn.Module, max_coalition_size: int = 15, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.max_coalition_size = max_coalition_size
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else get_best_device()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape

        delta = torch.zeros_like(orig_images)
        best_adv = orig_images.clone()

        with torch.no_grad():
            out_init = self.model(orig_images)
            best_loss = self.criterion(out_init, labels)

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = (out_init.argmax(dim=1) != labels)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            adv_images = (orig_images + delta).clamp(0.0, 1.0).requires_grad_(True)
            outputs = self.model(adv_images)
            loss_vec = self.criterion(outputs, labels)
            loss = loss_vec.sum()

            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            grad_mean = grad.abs().mean(dim=1, keepdim=True)
            grad_max = grad.abs().max(dim=1, keepdim=True)[0]
            coalition_score = grad_mean * grad_max

            coalition_mask = exact_spatial_topk_mask(coalition_score, self.max_coalition_size).float()
            candidate_delta = delta + self.alpha * grad.sign() * coalition_mask
            
            delta = project_l0(candidate_delta, self.max_coalition_size)
            adv_images_proj = torch.clamp(orig_images + delta, 0.0, 1.0)

            with torch.no_grad():
                out_step = self.model(adv_images_proj)
                curr_loss = self.criterion(out_step, labels)
                preds = out_step.argmax(dim=1)

                improved = curr_loss > best_loss
                best_loss[improved] = curr_loss[improved]
                best_adv[improved] = adv_images_proj[improved]

                current_fooled = (preds != labels)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return best_adv
