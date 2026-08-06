import torch
import torch.nn as nn

class SFAAttack:
    """Authentic Spectral Frequency Attack (SFA - FFT Frequency Domain)."""
    def __init__(self, model, k=15, freq_k=None, steps=25, alpha=4/255.0, device=None):
        self.model = model
        self.k = freq_k if freq_k is not None else k
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            x_adv.requires_grad = True
            out = self.model(x_adv)
            loss = self.criterion(out, y)
            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data
            fft_grad = torch.fft.fft2(grad)
            fft_mag = fft_grad.abs().sum(dim=1)

            flat_fft = fft_mag.view(B, -1)
            topk_vals, _ = torch.topk(flat_fft, k=min(self.k, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            freq_mask = (fft_mag >= thresh).unsqueeze(1).float()
            filtered_fft = fft_grad * freq_mask
            spatial_update = torch.fft.ifft2(filtered_fft).real
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * spatial_update.sign() * active_mask
            x_adv = torch.clamp(x_adv + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv

SpectralFrequencyAttack = SFAAttack
