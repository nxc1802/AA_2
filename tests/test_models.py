import torch
import pytest
from aa.models import get_model, adapt_resnet_for_cifar, adapt_mobilenet_for_cifar, WideResNet28_10, ViTCIFAR


@pytest.mark.parametrize("model_name", [
    "resnet18",
    "resnet50",
    "wideresnet28_10",
    "mobilenet_v2",
    "vit_cifar"
])
def test_model_instantiation_and_forward(model_name):
    # Test un-initialized forward shape with strict_checkpoint=False
    model = get_model(
        model_name=model_name,
        num_classes=10,
        strict_checkpoint=False,
        checkpoint_path=None,
        eval_mode=True
    )
    assert model is not None
    device = next(model.parameters()).device
    x = torch.randn(2, 3, 32, 32, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape} for {model_name}"


def test_vit_cifar_custom_classes():
    model = ViTCIFAR(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 100)
