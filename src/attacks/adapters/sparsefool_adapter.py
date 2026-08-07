import os
import torch
import torch.nn as nn
from src.attacks.adapters.utils import scoped_sys_path
from src.core.utils import prepare_model_for_eval

THIRD_PARTY_SPARSEFOOL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../third_party/sparsefool"))

class SparseFoolOfficialAdapter:
    """
    Adapter wrapping official author implementation of SparseFool (LTS4/SparseFool, CVPR 2019).
    Uses DeepFool decision boundary approximation and sparse linear solver.
    """
    def __init__(self, model: nn.Module, k: int = 250, steps: int = 20, lambda_val: float = 3.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = k
        self.steps = steps
        self.lambda_val = lambda_val
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with scoped_sys_path(THIRD_PARTY_SPARSEFOOL):
            from sparsefool import sparsefool
            
            x = x.to(self.device)
            y = y.to(self.device)
            B = x.size(0)
            x_adv = x.clone()

            steps_list = []
            for b in range(B):
                x_single = x[b:b+1]
                fool_im, r, p_label, f_label, loops = sparsefool(
                    x_single, self.model, lb=0.0, ub=1.0, lambda_=self.lambda_val, max_iter=self.steps, device=str(self.device)
                )
                x_adv[b] = fool_im[0].detach()
                steps_list.append(loops)

            self.last_steps = steps_list
            return x_adv
