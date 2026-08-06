import torch
import torch.nn as nn
import torch.nn.functional as F

class OnePixelAttack:
    """Vectorized OnePixel / Multi-Pixel Differential Evolution Attack."""
    def __init__(self, model, k=1, max_iter=20, pop_size=20, device=None):
        self.model = model
        self.k = k
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
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
            img_orig = x[b:b+1].to(self.device)
            target_y = y[b:b+1].to(self.device)

            best_img = img_orig.clone()
            best_loss = -1.0

            for it in range(self.max_iter):
                coords_x = torch.randint(0, W, (self.pop_size, self.k), device=self.device)
                coords_y = torch.randint(0, H, (self.pop_size, self.k), device=self.device)
                pert_vals = torch.rand((self.pop_size, self.k, C), device=self.device)

                cand_batch = img_orig.repeat(self.pop_size, 1, 1, 1)
                p_idx = torch.arange(self.pop_size, device=self.device).unsqueeze(1).expand(-1, self.k)
                cand_batch[p_idx, :, coords_y, coords_x] = pert_vals

                with torch.no_grad():
                    outs = self.model(cand_batch)
                    preds = outs.argmax(dim=1)
                    losses = F.cross_entropy(outs, target_y.repeat(self.pop_size), reduction='none')

                succ_mask = (preds != target_y.item())
                if succ_mask.any():
                    succ_idx = succ_mask.nonzero(as_tuple=True)[0][0]
                    best_img = cand_batch[succ_idx:succ_idx+1]
                    steps_to_fool[b] = it + 1
                    break

                max_idx = losses.argmax().item()
                if losses[max_idx].item() > best_loss:
                    best_loss = losses[max_idx].item()
                    best_img = cand_batch[max_idx:max_idx+1]

            x_adv[b] = best_img[0].detach()

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv
