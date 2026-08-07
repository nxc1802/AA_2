import os
import torch
import torch.nn as nn
from src.attacks.adapters.utils import scoped_sys_path
from src.core.utils import prepare_model_for_eval

THIRD_PARTY_HOMOTOPY = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../third_party/sparseadv_homotopy"))

class HomotopyOfficialAdapter:
    """
    Adapter wrapping official author implementation of Homotopy Attack (VITA-Group/SparseADV_Homotopy, ICML 2021).
    Uses non-monotone Accelerated Proximal Gradient (nmAPG) with homotopy continuation schedule.
    """
    def __init__(self, model: nn.Module, k: int = 15, target_sparsity: int = None, steps: int = 25, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = target_sparsity if target_sparsity is not None else k
        self.steps = steps
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with scoped_sys_path(THIRD_PARTY_HOMOTOPY):
            from demo_attack import homotopy

            x = x.to(self.device)
            y = y.to(self.device)
            B = x.size(0)
            x_adv = x.clone()

            steps_list = []
            for b in range(B):
                img_b = x[b:b+1]
                lbl_b = y[b].item()
                try:
                    pert = homotopy(
                        loss_type='ce',
                        net=self.model,
                        original_img=img_b,
                        target_class=lbl_b,
                        original_class=lbl_b,
                        tar=0,
                        max_epsilon=1.0,
                        dec_factor=0.8,
                        val_c=1.0,
                        val_w1=1e-4,
                        val_w2=1e-3,
                        max_update=self.k,
                        maxiter=self.steps,
                        val_gamma=1.0
                    )
                    if isinstance(pert, tuple):
                        pert = pert[0]
                    x_adv[b] = (img_b + pert).clamp(0.0, 1.0)[0]
                    steps_list.append(self.steps)
                except Exception:
                    x_adv[b] = img_b[0]
                    steps_list.append(self.steps)

            self.last_steps = steps_list
            return x_adv
