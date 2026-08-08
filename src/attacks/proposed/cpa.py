import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.projections import project_l0, exact_spatial_topk_mask
from src.core.utils import prepare_model_for_eval, get_best_device

class CooperativePixelsAttack:
    """
    Cooperative Pixels Attack (CPA).
    Exploits spatial pixel interaction and cooperative saliency scores.
    Strictly projects perturbation onto exact K-sparse spatial L0 ball.
    """
    def __init__(self, model: nn.Module, coalition_size: int = 15, steps: int = 25, alpha: float = 4/255.0, coop_weight: float = 0.5, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.coalition_size = coalition_size
        self.steps = steps
        self.alpha = alpha
        self.coop_weight = coop_weight
        self.device = device if device is not None else get_best_device()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape

        kernel = torch.ones(1, 1, 3, 3, device=self.device) / 9.0
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
            grad_mag = grad.abs().sum(dim=1, keepdim=True) # (B, 1, H, W)
            local_coop = F.conv2d(grad_mag, kernel, padding=1)
            coop_score = grad_mag + self.coop_weight * local_coop

            coalition_mask = exact_spatial_topk_mask(coop_score, self.coalition_size).float()
            candidate_delta = delta + self.alpha * grad.sign() * coalition_mask
            
            # Project onto hard L0 ball of radius coalition_size
            delta = project_l0(candidate_delta, self.coalition_size)
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

        steps_list = steps_to_fool.cpu().numpy().tolist()
        self.last_steps = steps_list
        self.last_queries = [int(s * 2 + 1) for s in steps_list]
        self.last_grad_evals = [int(s) for s in steps_list]
        return best_adv
