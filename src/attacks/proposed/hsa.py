import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.projections import project_l0, exact_spatial_topk_mask
from src.core.utils import prepare_model_for_eval, get_best_device

class HypergraphSparseAttack:
    """
    Hypergraph Sparse Attack (HSA).
    Constructs a spatial hypergraph structure:
      - Nodes: Spatial pixels (H x W)
      - Hyperedges: Multi-scale receptive fields (3x3, 5x5, 7x7)
    Selects top node centrality scores to break maximum hyperedges while strictly
    projecting perturbation onto exact K-sparse L0 ball.
    """
    def __init__(self, model: nn.Module, budget: int = 15, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.budget = budget
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else get_best_device()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _construct_hypergraph_degree(self, images: torch.Tensor, labels: torch.Tensor):
        images.requires_grad = True
        outputs = self.model(images)
        loss_vec = self.criterion(outputs, labels)
        loss = loss_vec.sum()
        
        self.model.zero_grad()
        loss.backward()

        grad = images.grad.data
        grad_mag = grad.abs().sum(dim=1, keepdim=True) # (B, 1, H, W)

        # Multi-scale spatial receptive field hyperedge pooling (3x3, 5x5, 7x7)
        h_pool3 = F.avg_pool2d(grad_mag, kernel_size=3, stride=1, padding=1)
        h_pool5 = F.avg_pool2d(grad_mag, kernel_size=5, stride=1, padding=2)
        h_pool7 = F.avg_pool2d(grad_mag, kernel_size=7, stride=1, padding=3)

        node_centrality = grad_mag + 0.5 * h_pool3 + 0.3 * h_pool5 + 0.2 * h_pool7
        return node_centrality, grad

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
            adv_images = (orig_images + delta).clamp(0.0, 1.0)
            node_centrality, grad = self._construct_hypergraph_degree(adv_images, labels)

            hypergraph_mask = exact_spatial_topk_mask(node_centrality, self.budget).float()
            candidate_delta = delta + self.alpha * grad.sign() * hypergraph_mask
            
            delta = project_l0(candidate_delta, self.budget)
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
