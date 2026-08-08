# ==============================================================================
# IMPORTS & DEPENDENCIES
# ==============================================================================
import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from huggingface_hub import HfApi, hf_hub_download
from src.datasets.dataset_loader import HFDatasetWrapper

from src.core.utils import get_best_device

HF_REPO_ID = "Cuong2004/AA"
HF_TOKEN = os.getenv("HF_TOKEN", None)

# ==============================================================================
# CONFIGURABLE PARAMETERS & PATHS (GPU OPTIMIZED)
# ==============================================================================
MODEL_NAME = "resnet18"
NUM_CLASSES = 10
DEVICE = get_best_device()
RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../result"))
SAVED_MODELS_DIR = os.path.join(RESULT_DIR, "saved_models")
LOG_DIR = os.path.join(RESULT_DIR, "logs")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logger = logging.getLogger("ModelFactory")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "model_factory.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================================
# HUGGING FACE CHECKPOINT MANAGER
# ==============================================================================
def upload_checkpoint_to_hf(local_path, path_in_repo="models/resnet18_cifar10_best.pth", repo_id=HF_REPO_ID, token=HF_TOKEN):
    """Uploads model checkpoint directly to Hugging Face repository."""
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        logger.info(f"✅ Successfully uploaded checkpoint '{local_path}' to HF '{repo_id}/{path_in_repo}'!")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Failed to upload checkpoint to HF: {e}")
        return False

def download_checkpoint_from_hf(path_in_repo="models/resnet18_cifar10_best.pth", repo_id=HF_REPO_ID, token=HF_TOKEN):
    """Downloads model checkpoint directly from Hugging Face repository."""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=token
        )
        logger.info(f"✅ Successfully downloaded checkpoint from HF '{repo_id}/{path_in_repo}' to '{local_path}'!")
        return local_path
    except Exception as e:
        logger.warning(f"⚠️ Checkpoint not found on HF '{repo_id}/{path_in_repo}': {e}")
        return None

def find_existing_checkpoint(filename="resnet18_cifar10_best.pth"):
    """Downloads model checkpoint from Hugging Face 'Cuong2004/AA' with local fallback."""
    # Always prioritize downloading official checkpoint from Hugging Face
    hf_path = download_checkpoint_from_hf(path_in_repo=f"models/{filename}")
    if hf_path and os.path.isfile(hf_path):
        return hf_path

    k_path = "/kaggle/input/models/cuongnguyen1802/resnet18-aa/tensorflow2/default/1"
    if os.path.exists(k_path):
        exact_p = os.path.join(k_path, filename)
        if os.path.isfile(exact_p):
            return exact_p
        for root, dirs, files in os.walk(k_path):
            for f in files:
                if f == filename or f.endswith(".pth") or f.endswith(".pt"):
                    return os.path.join(root, f)

    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f == filename or (f.endswith(".pth") and "resnet18" in f.lower()):
                    return os.path.join(root, f)

    p1 = os.path.join(SAVED_MODELS_DIR, filename)
    if os.path.isfile(p1):
        return p1

    return None

# ==============================================================================
# MODEL FACTORY & TRAINING LOOP
# ==============================================================================
def adapt_resnet_for_cifar(model, num_classes=10):
    """Adapts standard PyTorch ResNet for CIFAR-10 (32x32 spatial input)."""
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model

class BasicBlockWRN(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlockWRN, self).__init__()
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
            out = torch.nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.add(self.conv_shortcut(x) if self.conv_shortcut is not None else x, out)

class NetworkBlockWRN(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride=1, drop_rate=0.0):
        super(NetworkBlockWRN, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet28_10(nn.Module):
    """WideResNet-28-10 for CIFAR-10 (Zagoruyko & Komodakis, 2016)."""
    def __init__(self, depth=28, widen_factor=10, drop_rate=0.0, num_classes=10):
        super(WideResNet28_10, self).__init__()
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
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = out.view(-1, self.n_channels)
        return self.fc(out)

def get_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False, checkpoint_path=None, device=DEVICE):
    """Instantiates neural network backbones."""
    logger.info(f"Instantiating backbone: '{model_name}' (Device={device})")

    if model_name.lower() == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model = adapt_resnet_for_cifar(model, num_classes=num_classes)
        model.architecture_name = "resnet18"
    elif model_name.lower() == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model = adapt_resnet_for_cifar(model, num_classes=num_classes)
        model.architecture_name = "resnet50"
    elif model_name.lower() in ["wideresnet28_10", "wideresnet", "wrn28_10"]:
        model = WideResNet28_10(depth=28, widen_factor=10, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    model = model.to(device)

    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Specified model checkpoint file not found: '{checkpoint_path}'")
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    return model

def evaluate_accuracy(model, data_loader, device=DEVICE):
    """Evaluates top-1 accuracy with GPU AMP autocast."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        raise RuntimeError("DataLoader returned 0 samples for evaluation.")

    return 100.0 * correct / total

def train_clean_resnet18(train_loader, val_loader, test_loader, epochs=200, batch_size=1024, device=DEVICE, seed=42):
    """Fast ResNet-18 training loop with VRAM caching & HF auto-upload/download."""
    from src.core.utils import set_seed
    set_seed(seed)

    existing_ckpt = find_existing_checkpoint("resnet18_cifar10_best.pth")
    best_checkpoint_path = os.path.join(SAVED_MODELS_DIR, "resnet18_cifar10_best.pth")

    if existing_ckpt:
        logger.info(f"Found trained model checkpoint at '{existing_ckpt}'! Skipping training...")
        model = get_model(checkpoint_path=existing_ckpt, device=device)
        test_acc = evaluate_accuracy(model, test_loader, device=device)
        logger.info(f"Clean Test Accuracy: {test_acc:.2f}%")
        return model

    logger.info("--- Caching Training Set into VRAM for Ultra-Fast Training ---")
    # Wrap underlying dataset without random augmentation for initial VRAM caching
    if hasattr(train_loader.dataset, "dataset") and hasattr(train_loader.dataset, "indices"):
        orig_ds = train_loader.dataset.dataset.hf_ds
        raw_wrapper = HFDatasetWrapper(orig_ds, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
        raw_subset = Subset(raw_wrapper, train_loader.dataset.indices)
        cache_loader = DataLoader(raw_subset, batch_size=4096, shuffle=False, num_workers=0)
    else:
        cache_loader = DataLoader(train_loader.dataset, batch_size=4096, shuffle=False, num_workers=0)
    
    cached_x, cached_y = [], []
    for bx, by in cache_loader:
        cached_x.append(bx)
        cached_y.append(by)

    X_train = torch.cat(cached_x).to(device)
    Y_train = torch.cat(cached_y).to(device)

    train_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ])

    model = get_model(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    num_samples = len(X_train)
    t0_train = time.time()

    logger.info(f"Starting Training for {epochs} Epochs with Batch Size {batch_size}...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        perm = torch.randperm(num_samples, device=device)
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]

        for i in range(0, num_samples, batch_size):
            bx = train_aug(X_shuffled[i:i+batch_size])
            by = Y_shuffled[i:i+batch_size]
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(bx)
                    loss = criterion(outputs, by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(bx)
                loss = criterion(outputs, by)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * bx.size(0)
            _, predicted = torch.max(outputs, 1)
            total += by.size(0)
            correct += (predicted == by).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = running_loss / total

        if epoch % 20 == 0 or epoch == epochs:
            val_acc = evaluate_accuracy(model, val_loader, device=device)
            logger.info(f"Epoch [{epoch:03d}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.5f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, best_checkpoint_path)
                logger.info(f"  >>> Best Checkpoint Saved! Val Acc: {best_val_acc:.2f}%")

    logger.info(f"Training completed in {time.time()-t0_train:.2f}s!")
    upload_checkpoint_to_hf(best_checkpoint_path, path_in_repo="models/resnet18_cifar10_best.pth")
    model = get_model(checkpoint_path=best_checkpoint_path, device=device)
    clean_test_acc = evaluate_accuracy(model, test_loader, device=device)
    logger.info(f"=== FINAL CLEAN TEST ACCURACY (10,000 Test Images): {clean_test_acc:.2f}% ===")
    return model

if __name__ == "__main__":
    logger.info("=== Running Standalone Test for src/models/model_factory.py ===")
    model = get_model("resnet18", pretrained=False)
    dummy_input = torch.randn(16, 3, 32, 32).to(DEVICE)
    out = model(dummy_input)
    logger.info(f"Dummy output shape: {out.shape}")
