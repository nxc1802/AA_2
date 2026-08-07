import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.projections import project_l0
from src.core.utils import prepare_model_for_eval

class GSEAttack:
    """Group Sparse Attack (GSE - 2x2 Spatial Blocks) with exact L0 projection."""
    def __init__(self, model: nn.Module, k: int = 15, group_size: int = 4, max_groups: int = None, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.group_size = group_size
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        if max_groups is not None:
            self.num_groups = max_groups
        else:
            self.num_groups = max(1, k // group_size)

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape

        delta = torch.zeros_like(x)
        best_adv = x.clone()

        with torch.no_grad():
            out_init = self.model(x)
            best_loss = self.criterion(out_init, y)

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = (out_init.argmax(dim=1) != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            x_adv = (x + delta).clamp(0.0, 1.0).requires_grad_(True)
            out = self.model(x_adv)
            loss_vec = self.criterion(out, y)
            loss = loss_vec.sum()

            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data
            grad_mag = grad.abs().sum(dim=1, keepdim=True)
            group_pool = F.avg_pool2d(grad_mag, kernel_size=2, stride=2)

            flat_group = group_pool.view(B, -1)
            topk_vals, _ = torch.topk(flat_group, k=min(self.num_groups, flat_group.size(1)), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1, 1)

            group_mask = (group_pool >= thresh).float()
            spatial_mask = F.interpolate(group_mask, scale_factor=2, mode='nearest')
            
            candidate_delta = delta + self.alpha * grad.sign() * spatial_mask
            delta = project_l0(candidate_delta, self.num_groups * self.group_size)
            adv_images_proj = torch.clamp(x + delta, 0.0, 1.0)

            with torch.no_grad():
                out_step = self.model(adv_images_proj)
                curr_loss = self.criterion(out_step, y)
                preds = out_step.argmax(dim=1)

                improved = curr_loss > best_loss
                best_loss[improved] = curr_loss[improved]
                best_adv[improved] = adv_images_proj[improved]

                current_fooled = (preds != y)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return best_adv

GroupSparseAttack = GSEAttack
