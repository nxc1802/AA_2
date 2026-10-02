import torch
import torch.nn.functional as F


class CoalitionObjective:
    """Computes attack loss (DLR / Margin / CE) and decision margin."""
    def __init__(self, loss_fn: str = "dlr"):
        self.loss_fn = loss_fn.lower()

    @staticmethod
    def compute_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes attack margin: J(x, y) = max_{c != y} z_c(x) - z_y(x).
        J > 0 implies misclassification.
        """
        B, C = logits.shape
        one_hot = F.one_hot(y, num_classes=C).bool()
        z_y = logits[one_hot]

        logits_other = logits.clone()
        logits_other[one_hot] = float("-inf")
        z_other = logits_other.max(dim=1)[0]
        return z_other - z_y

    @staticmethod
    def compute_dlr_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Difference of Logits Ratio (DLR) loss (Croce & Hein 2020):
        DLR(x, y) = (z_other - z_y) / (z_{pi_1} - z_{pi_3} + eps)
        """
        B, C = logits.shape
        sorted_logits, _ = logits.sort(dim=1, descending=True)
        z_p1 = sorted_logits[:, 0]
        z_p3 = sorted_logits[:, 2]
        denom = z_p1 - z_p3 + 1e-6

        one_hot = F.one_hot(y, num_classes=C).bool()
        z_y = logits[one_hot]

        logits_other = logits.clone()
        logits_other[one_hot] = float("-inf")
        z_other = logits_other.max(dim=1)[0]
        return (z_other - z_y) / denom

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.loss_fn == "dlr":
            return self.compute_dlr_loss(logits, y)
        elif self.loss_fn == "margin":
            return self.compute_margin(logits, y)
        elif self.loss_fn == "ce":
            return F.cross_entropy(logits, y, reduction="none")
        else:
            raise ValueError(f"Unknown loss_fn '{self.loss_fn}'. Options: ['dlr', 'margin', 'ce']")
