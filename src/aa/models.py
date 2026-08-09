import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from huggingface_hub import hf_hub_download
from typing import Optional

from aa.utils import get_best_device, prepare_model_for_eval

HF_REPO_ID = "Cuong2004/AA"
HF_TOKEN = os.getenv("HF_TOKEN", None)


def adapt_resnet_for_cifar(model: nn.Module, num_classes: int = 10) -> nn.Module:
    """Adapts standard PyTorch ResNet for CIFAR (32x32 spatial input)."""
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


class BasicBlockWRN(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.equal_in_out) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.conv1(self.relu1(self.bn1(x)))
        if self.drop_rate > 0:
            out = nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.add(self.conv_shortcut(x) if self.conv_shortcut is not None else x, out)


class NetworkBlockWRN(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride=1, drop_rate=0.0):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet28_10(nn.Module):
    """WideResNet-28-10 for CIFAR (Zagoruyko & Komodakis, 2016)."""
    def __init__(self, depth=28, widen_factor=10, drop_rate=0.0, num_classes=10):
        super().__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlockWRN
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlockWRN(n, n_channels[0], n_channels[1], block, 1, drop_rate)
        self.block2 = NetworkBlockWRN(n, n_channels[1], n_channels[2], block, 2, drop_rate)
        self.block3 = NetworkBlockWRN(n, n_channels[2], n_channels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]
        self.architecture_name = "wideresnet28_10"

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = nn.functional.avg_pool2d(out, 8)
        out = out.view(-1, self.n_channels)
        return self.fc(out)


def find_existing_checkpoint(checkpoint_path_or_name: str = "resnet18_cifar10_best.pth") -> Optional[str]:
    """Finds checkpoint file locally or downloads from HF repository."""
    if os.path.isfile(checkpoint_path_or_name):
        return os.path.abspath(checkpoint_path_or_name)

    filename = os.path.basename(checkpoint_path_or_name)

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    local_root_p = os.path.join(workspace_root, filename)
    if os.path.isfile(local_root_p):
        return local_root_p

    saved_p = os.path.join(workspace_root, "result", "saved_models", filename)
    if os.path.isfile(saved_p):
        return saved_p

    # Download from Hugging Face
    try:
        hf_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"models/{filename}",
            repo_type="dataset",
            token=HF_TOKEN
        )
        if hf_path and os.path.isfile(hf_path):
            return hf_path
    except Exception:
        pass

    return None


def get_model(
    model_name: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    strict_checkpoint: bool = True,
    expected_sha256: Optional[str] = None,
    min_clean_acc: Optional[float] = None,
    validation_loader = None
) -> nn.Module:
    """Instantiates neural network backbones and loads checkpoints strictly."""
    from aa.utils import compute_file_sha256
    if device is None:
        device = get_best_device()

    name_clean = model_name.lower()
    if name_clean == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model = adapt_resnet_for_cifar(model, num_classes=num_classes)
        model.architecture_name = "resnet18"
    elif name_clean == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model = adapt_resnet_for_cifar(model, num_classes=num_classes)
        model.architecture_name = "resnet50"
    elif name_clean in ["wideresnet28_10", "wideresnet", "wrn28_10"]:
        model = WideResNet28_10(depth=28, widen_factor=10, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    if checkpoint_path is None:
        checkpoint_path = f"{name_clean}_cifar10_best.pth"

    resolved_ckpt = find_existing_checkpoint(checkpoint_path)

    if resolved_ckpt is None or not os.path.isfile(resolved_ckpt):
        if strict_checkpoint and not pretrained:
            raise FileNotFoundError(
                f"Model checkpoint for '{name_clean}' not found at '{checkpoint_path}' (resolved: {resolved_ckpt}). "
                f"Benchmark execution aborted to prevent uninitialized/random model evaluations (P0.2)."
            )
    else:
        checkpoint = torch.load(resolved_ckpt, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.checkpoint_path = resolved_ckpt
        model.checkpoint_sha256 = compute_file_sha256(resolved_ckpt)

        if expected_sha256 and model.checkpoint_sha256 != expected_sha256:
            raise ValueError(
                f"Checkpoint SHA256 mismatch for {resolved_ckpt}. Expected: {expected_sha256}, Actual: {model.checkpoint_sha256}"
            )

    model = prepare_model_for_eval(model, device=device)

    if min_clean_acc is not None and validation_loader is not None:
        acc = evaluate_accuracy(model, validation_loader, device=device)
        if acc < min_clean_acc:
            raise RuntimeError(
                f"Model clean accuracy ({acc:.2f}%) is below minimum required threshold ({min_clean_acc:.2f}%). Aborting."
            )

    return model


def evaluate_accuracy(model: nn.Module, data_loader, device: Optional[torch.device] = None) -> float:
    """Evaluates clean top-1 accuracy."""
    if device is None:
        device = get_best_device()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if total == 0:
        return 0.0
    return 100.0 * correct / total
