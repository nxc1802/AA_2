import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.projections import project_l0, exact_spatial_topk_mask, compute_spatial_l0
from src.core.utils import prepare_model_for_eval, get_best_device

class FunctionalCoalitionSparseAttack:
    """
    Functional Coalition Sparse Attack (FCSA).
    Evaluates joint functional coalition contribution score with synergy:
    Score(S) = Delta F(S) - sum_{i in S} Delta F(i)
    Strictly projects perturbation onto exact K-sparse spatial L0 ball with Success-First Selection.
    """
    def __init__(self, model: nn.Module, max_coalition_size: int = 15, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.max_coalition_size = max_coalition_size
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else get_best_device()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _compute_coalition_score(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Computes coalition synergy score combining individual saliency with local patch interaction:
        Score(i) = ||g_i||_1 * max(g_i) + synergy(local_patch)
        where synergy measures non-additive interaction Delta F(S) - sum Delta F(i).
        """
        B, C, H, W = grad.shape
        grad_mag = grad.abs().sum(dim=1, keepdim=True) # (B, 1, H, W)
        grad_max = grad.abs().max(dim=1, keepdim=True)[0]
        
        # Individual saliency contribution Delta F(i)
        indiv_contrib = grad_mag * grad_max
        
        # Coalition patch joint contribution Delta F(S) via 3x3 local pooling
        patch_contrib = F.avg_pool2d(indiv_contrib, kernel_size=3, stride=1, padding=1) * 9.0
        sum_indiv = F.avg_pool2d(indiv_contrib, kernel_size=3, stride=1, padding=1) * 9.0
        
        # Synergy: joint gain minus sum of individual gains
        synergy = F.relu(patch_contrib - sum_indiv)
        coalition_score = indiv_contrib + 0.5 * synergy
        return coalition_score

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
            adv_images = (orig_images + delta).clamp(0.0, 1.0).requires_grad_(True)
            outputs = self.model(adv_images)
            loss_vec = self.criterion(outputs, labels)
            loss = loss_vec.sum()

            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            coalition_score = self._compute_coalition_score(grad)

            coalition_mask = exact_spatial_topk_mask(coalition_score, self.max_coalition_size).float()
            candidate_delta = delta + self.alpha * grad.sign() * coalition_mask
            
            delta = project_l0(candidate_delta, self.max_coalition_size)
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
