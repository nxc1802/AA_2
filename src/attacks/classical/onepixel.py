import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.utils import prepare_model_for_eval

class OnePixelAttack:
    """
    Differential Evolution (DE) OnePixel / Multi-Pixel Attack.
    Uses standard (F, CR) Differential Evolution algorithm (Su et al. 2019).
    """
    def __init__(self, model: nn.Module, k: int = 1, max_iter: int = 20, pop_size: int = 20, F_param: float = 0.5, CR: float = 0.7, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = k
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.F_param = F_param
        self.CR = CR
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        with torch.no_grad():
            init_preds = self.model(x).argmax(dim=1)
        
        active_b = (init_preds == y)
        steps_to_fool = torch.full((B,), self.max_iter, dtype=torch.float, device=self.device)
        steps_to_fool[~active_b] = 0.0

        for b in range(B):
            if not active_b[b]:
                continue

            img_orig = x[b:b+1]
            target_y = y[b:b+1]

            # Vector representation per pixel tuple: [x_coord, y_coord, r, g, b] -> total dim 5 * k
            # Normalize coords to [0, 1] continuous space for DE, map to integer coords on eval
            pop = torch.rand((self.pop_size, self.k, 5), device=self.device)
            
            def decode_population(p_tensor):
                # (pop_size, k, 5) -> images tensor (pop_size, C, H, W)
                cand_imgs = img_orig.repeat(len(p_tensor), 1, 1, 1)
                for p_idx in range(len(p_tensor)):
                    for i in range(self.k):
                        px = int(p_tensor[p_idx, i, 0].item() * (W - 1))
                        py = int(p_tensor[p_idx, i, 1].item() * (H - 1))
                        r, g, b_val = p_tensor[p_idx, i, 2:].tolist()
                        cand_imgs[p_idx, 0, py, px] = r
                        cand_imgs[p_idx, 1, py, px] = g
                        cand_imgs[p_idx, 2, py, px] = b_val
                return cand_imgs

            with torch.no_grad():
                cand_imgs = decode_population(pop)
                outs = self.model(cand_imgs)
                pop_losses = F.cross_entropy(outs, target_y.repeat(self.pop_size), reduction='none')

            best_idx = pop_losses.argmax().item()
            best_img = cand_imgs[best_idx:best_idx+1]
            best_loss = pop_losses[best_idx].item()

            for it in range(self.max_iter):
                # Check for success in population
                preds = outs.argmax(dim=1)
                succ_mask = (preds != target_y.item())
                if succ_mask.any():
                    first_succ = succ_mask.nonzero(as_tuple=True)[0][0]
                    best_img = cand_imgs[first_succ:first_succ+1]
                    steps_to_fool[b] = it + 1
                    break

                # Differential Evolution update per individual
                mutants = torch.zeros_like(pop)
                for i in range(self.pop_size):
                    idxs = [j for j in range(self.pop_size) if j != i]
                    r1, r2, r3 = [idxs[idx] for idx in torch.randperm(len(idxs))[:3]]
                    mutant = pop[r1] + self.F_param * (pop[r2] - pop[r3])
                    mutants[i] = mutant.clamp(0.0, 1.0)

                # Binomial Crossover
                crossover_mask = torch.rand_like(pop) < self.CR
                trials = torch.where(crossover_mask, mutants, pop)

                # Selection step
                with torch.no_grad():
                    trial_imgs = decode_population(trials)
                    trial_outs = self.model(trial_imgs)
                    trial_losses = F.cross_entropy(trial_outs, target_y.repeat(self.pop_size), reduction='none')

                improved = trial_losses > pop_losses
                pop[improved] = trials[improved]
                pop_losses[improved] = trial_losses[improved]
                outs[improved] = trial_outs[improved]

                curr_best = pop_losses.argmax().item()
                if pop_losses[curr_best].item() > best_loss:
                    best_loss = pop_losses[curr_best].item()
                    cand_imgs = decode_population(pop)
                    best_img = cand_imgs[curr_best:curr_best+1]

            x_adv[b] = best_img[0].detach()

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
