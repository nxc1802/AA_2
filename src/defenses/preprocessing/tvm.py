import torch

class TotalVariationMinimizationDefense:
    """Total Variation Minimization (TVM) smoothing defense."""
    def __init__(self, steps: int = 10, weight: float = 0.05, lr: float = 0.01, device: torch.device = None):
        self.steps = steps
        self.weight = weight
        self.lr = lr
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def defend(self, images: torch.Tensor) -> torch.Tensor:
        orig = images.clone().detach().to(self.device)
        x = orig.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([x], lr=self.lr)
        
        for _ in range(self.steps):
            # Spatial gradients
            dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
            dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
            tv_loss = dx + dy
            rec_loss = torch.mean((x - orig) ** 2)
            
            loss = rec_loss + self.weight * tv_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
                
        return x.detach()
