import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.projections import project_l0, exact_spatial_topk_mask, compute_spatial_l0
from src.core.utils import prepare_model_for_eval, get_best_device

class CooperativePixelsAttack:
    """
    Cooperative Pixels Attack (CPA).
    Exploits directional spatial gradient alignment cooperation I(i, j) = cos(g_i, g_j) * ||g_j||_1.
    Strictly projects perturbation onto exact K-sparse spatial L0 ball with Success-First Selection.
    """
    def __init__(self, model: nn.Module, coalition_size: int = 15, steps: int = 25, alpha: float = 4/255.0, coop_weight: float = 0.5, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.coalition_size = coalition_size
        self.steps = steps
        self.alpha = alpha
        self.coop_weight = coop_weight
        self.device = device if device is not None else get_best_device()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _compute_directional_cooperation(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Computes spatial directional gradient alignment cooperation score:
        I(i) = ||g_i||_1 + lambda * sum_{j in N(i)} cos(g_i, g_j) * ||g_j||_1
        """
        B, C, H, W = grad.shape
        grad_norm = grad.norm(p=2, dim=1, keepdim=True) + 1e-8
        grad_unit = grad / grad_norm
        grad_abs_sum = grad.abs().sum(dim=1, keepdim=True) # (B, 1, H, W)

        coop_score = grad_abs_sum.clone()
        shifts = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

        for dh, dw in shifts:
            unit_shift = torch.roll(grad_unit, shifts=(dh, dw), dims=(2, 3))
            mag_shift = torch.roll(grad_abs_sum, shifts=(dh, dw), dims=(2, 3))
            alignment = (grad_unit * unit_shift).sum(dim=1, keepdim=True) # Cosine similarity
            coop_score += self.coop_weight * F.relu(alignment) * mag_shift

        return coop_score

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape

        delta = torch.zeros_like(orig_images)
        best_adv = orig_images.clone()

        with torch.no_grad():
            out_init = self.model(orig_images)
            best_loss = self.criterion(out_init, labels)
            best_succ = (out_init.argmax(dim=1) != labels)
            best_l0 = torch.where(best_succ, torch.zeros(B, device=self.device), torch.full((B,), float('inf'), device=self.device))

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = best_succ.clone()
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            adv_images = (orig_images + delta).clamp(0.0, 1.0).requires_grad_(True)
            outputs = self.model(adv_images)
            loss_vec = self.criterion(outputs, labels)
            loss = loss_vec.sum()

            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            coop_score = self._compute_directional_cooperation(grad)

            # Mask out already fooled samples from coalition selection & delta updates
            coop_score_masked = coop_score.clone()
            coop_score_masked[fooled_mask] = -float('inf')

            coalition_mask = exact_spatial_topk_mask(coop_score_masked, self.coalition_size).float()
            candidate_delta = delta + self.alpha * grad.sign() * coalition_mask * (~fooled_mask).view(B, 1, 1, 1).float()
            
            # Project onto hard L0 ball of radius coalition_size
            delta_proj = project_l0(candidate_delta, self.coalition_size)
            
            # Freeze delta for already fooled samples
            delta = torch.where(fooled_mask.view(B, 1, 1, 1), delta, delta_proj)
            adv_images_proj = torch.clamp(orig_images + delta, 0.0, 1.0)

            with torch.no_grad():
                out_step = self.model(adv_images_proj)
                curr_loss = self.criterion(out_step, labels)
                preds = out_step.argmax(dim=1)
                cand_succ = (preds != labels)
                cand_l0 = compute_spatial_l0(adv_images_proj - orig_images).float()

                # Success-First Selection logic (Bug #20 fix)
                replace = (
                    (cand_succ & ~best_succ) |
                    (cand_succ & best_succ & ((cand_l0 < best_l0) | ((cand_l0 == best_l0) & (curr_loss > best_loss)))) |
                    (~cand_succ & ~best_succ & (curr_loss > best_loss))
                )

                best_adv[replace] = adv_images_proj[replace]
                best_succ[replace] = cand_succ[replace]
                best_l0[replace] = cand_l0[replace]
                best_loss[replace] = curr_loss[replace]

                newly_fooled = cand_succ & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        steps_list = steps_to_fool.cpu().numpy().tolist()
        self.last_steps = steps_list
        self.last_queries = [int(s * 2 + 1) for s in steps_list]
        self.last_grad_evals = [int(s) for s in steps_list]
        return best_adv
