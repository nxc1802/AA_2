import os
import torch
import torch.nn as nn
from src.attacks.adapters.utils import scoped_sys_path
from src.core.utils import prepare_model_for_eval

THIRD_PARTY_SIGMA_ZERO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../third_party/sigma_zero"))

class SigmaZeroOfficialAdapter:
    """
    Adapter wrapping official author implementation of SigmaZero (sigma0-advx/sigma-zero, ICLR 2025).
    Features differentiable L0 approximation and dynamic thresholding.
    """
    def __init__(self, model: nn.Module, k: int = 15, steps: int = 50, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = k
        self.steps = steps
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with scoped_sys_path(THIRD_PARTY_SIGMA_ZERO):
            try:
                from sigma_zero_attack import sigma_zero
            except ImportError:
                import torch.optim.lr_scheduler as lr_scheduler
                
                def difference_of_logits(logits, labels):
                    batch_size = logits.shape[0]
                    labels_logits = logits[torch.arange(batch_size), labels]
                    other_logits = logits.clone()
                    other_logits[torch.arange(batch_size), labels] = -float('inf')
                    max_other_logits = other_logits.max(dim=1).values
                    return labels_logits - max_other_logits

                def sigma_zero(model, inputs, labels, steps=100, lr=1.0, sigma=1e-3, threshold=0.3, verbose=False, epsilon_budget=None, grad_norm=torch.inf, t=0.01):
                    clamp = lambda tensor: tensor.data.add_(inputs.data).clamp_(min=0, max=1).sub_(inputs.data)
                    l0_approximation = lambda tensor, sig: tensor.square().div(tensor.square().add(sig)).sum(dim=1)
                    batch_view = lambda tensor: tensor.view(tensor.shape[0], *[1] * (inputs.ndim - 1))
                    normalize = lambda tensor: (tensor.flatten(1) / tensor.flatten(1).norm(p=grad_norm, dim=1, keepdim=True).clamp_(min=1e-12)).view(tensor.shape)

                    dev = next(model.parameters()).device
                    batch_size, max_size = inputs.shape[0], torch.prod(torch.tensor(inputs.shape[1:]))

                    delta = torch.zeros_like(inputs, requires_grad=True, device=dev)
                    optimizer = torch.optim.Adam([delta], lr=lr)
                    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 10)
                    best_delta = delta.clone()
                    query_mask = torch.full((batch_size,), True, device=dev)
                    best_l0 = torch.full((batch_size,), max_size, device=dev)
                    is_adv_below_eps = torch.full((batch_size,), False, device=dev)
                    th = torch.ones(size=inputs.shape, device=dev) * threshold
                    
                    for i in range(steps):
                        optimizer.zero_grad()
                        active_delta = delta[query_mask].clone().detach().requires_grad_(True)
                        active_inputs = inputs[query_mask]
                        active_labels = labels[query_mask]  
                        adv_inputs = active_inputs + active_delta

                        logits = model(adv_inputs)
                        dl_loss = difference_of_logits(logits, active_labels).clip(0)
                        l0_approx = l0_approximation(active_delta.flatten(1), sigma)
                        l0_approx_normalized = l0_approx / active_delta.data.flatten(1).shape[1]

                        predicted_classes = logits.argmax(1)
                        true_l0 = active_delta.data.flatten(1).ne(0).sum(dim=1)
                        is_not_adv = predicted_classes == active_labels
                        is_smaller = true_l0 < best_l0[query_mask]
                        is_both = ~is_not_adv & is_smaller
                        best_l0[query_mask] = torch.where(is_both, true_l0.detach(), best_l0[query_mask])
                        best_delta[query_mask] = torch.where(batch_view(is_both), active_delta.data.clone().detach(), best_delta[query_mask]) 
                        is_adv_below_eps = best_l0 <= epsilon_budget if epsilon_budget is not None else is_adv_below_eps 
                    
                        adv_loss = (is_not_adv + dl_loss + l0_approx_normalized).mean()
                        adv_loss.backward()

                        if delta.grad is None:
                            delta.grad = torch.zeros_like(delta, device=dev)
                        delta.grad[query_mask] += active_delta.grad
                        delta.grad.data = normalize(delta.grad.data)
                        optimizer.step()
                        scheduler.step()

                        with torch.no_grad():
                            clamp(delta.data)
                            th_active = th[query_mask]
                            th_active[is_not_adv, :, :, :] -= t * scheduler.get_last_lr()[0]
                            th_active[~is_not_adv, :, :, :] += t * scheduler.get_last_lr()[0]
                            th[query_mask] = th_active
                            th.clamp_(0, 1)
                            delta.data[delta.data.abs() < th] = 0
                            query_mask[is_adv_below_eps] = False
                            if not any(query_mask):
                                break

                    return (inputs + best_delta)

            x = x.to(self.device)
            y = y.to(self.device)
            adv_x = sigma_zero(self.model, x, y, steps=self.steps, epsilon_budget=self.k)
            return adv_x
