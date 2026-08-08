import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.projections import project_l0, exact_spatial_topk_mask, compute_spatial_l0
from src.core.utils import prepare_model_for_eval, get_best_device

class HypergraphSparseAttack:
    """
    Hypergraph Sparse Attack (HSA).
    Constructs an explicit spatial hypergraph incidence structure:
      - Nodes V: Spatial pixels (H x W)
      - Hyperedges E: Multi-scale receptive fields (3x3, 5x5, 7x7)
      - Incidence matrix H_ve, hyperedge degree D_e, node degree D_v
    Calculates hypergraph node centrality C(v) = sum_{e ni v} (W(e)/|e|) * sum_{u in e} (|g_u| / D_v(u)).
    Selects top nodes breaking maximum hyperedges while strictly projecting perturbation onto
    exact K-sparse L0 ball with Success-First Selection.
    """
    def __init__(self, model: nn.Module, budget: int = 15, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.budget = budget
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else get_best_device()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _compute_hypergraph_centrality(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Computes explicit hypergraph node centrality C(v) over multi-scale hyperedge structures.
        """
        B, C, H, W = grad.shape
        grad_mag = grad.abs().sum(dim=1, keepdim=True) # (B, 1, H, W)

        # Multi-scale hyperedge weights W(e) = sum_{u in e} |g_u|
        w3 = F.conv2d(grad_mag, torch.ones(1, 1, 3, 3, device=self.device), padding=1)
        w5 = F.conv2d(grad_mag, torch.ones(1, 1, 5, 5, device=self.device), padding=2)
        w7 = F.conv2d(grad_mag, torch.ones(1, 1, 7, 7, device=self.device), padding=3)

        # Hyperedge cardinality |e|
        de3, de5, de7 = 9.0, 25.0, 49.0

        # Node degree D_v(v) = sum_{e ni v} W(e)
        d_v = w3 + w5 + w7 + 1e-8
        normalized_grad = grad_mag / d_v

        # Hypergraph propagation C(v) = sum_{e ni v} (W(e)/|e|) * sum_{u in e} (|g_u| / D_v(u))
        c3 = F.conv2d(normalized_grad, torch.ones(1, 1, 3, 3, device=self.device), padding=1) * (w3 / de3)
        c5 = F.conv2d(normalized_grad, torch.ones(1, 1, 5, 5, device=self.device), padding=2) * (w5 / de5)
        c7 = F.conv2d(normalized_grad, torch.ones(1, 1, 7, 7, device=self.device), padding=3) * (w7 / de7)

        node_centrality = grad_mag + c3 + c5 + c7
        return node_centrality

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
            node_centrality = self._compute_hypergraph_centrality(grad)

            hypergraph_mask = exact_spatial_topk_mask(node_centrality, self.budget).float()
            candidate_delta = delta + self.alpha * grad.sign() * hypergraph_mask
            
            delta = project_l0(candidate_delta, self.budget)
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
