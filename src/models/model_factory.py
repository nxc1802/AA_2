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
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from huggingface_hub import HfApi, hf_hub_download

HF_REPO_ID = "Cuong2004/AA"
HF_TOKEN = os.getenv("HF_TOKEN", None)

# ==============================================================================
# CONFIGURABLE PARAMETERS & PATHS (GPU OPTIMIZED)
# ==============================================================================
MODEL_NAME = "resnet18"
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def get_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False, checkpoint_path=None, device=DEVICE):
    """Instantiates neural network backbones."""
    logger.info(f"Instantiating backbone: '{model_name}' (Device={device})")

    if model_name.lower() == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model = adapt_resnet_for_cifar(model, num_classes=num_classes)
    elif model_name.lower() == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model = adapt_resnet_for_cifar(model, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    model = model.to(device)

    if checkpoint_path and os.path.isfile(checkpoint_path):
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

def train_clean_resnet18(train_loader, val_loader, test_loader, epochs=200, batch_size=1024, device=DEVICE):
    """Fast ResNet-18 training loop with VRAM caching & HF auto-upload/download."""
    existing_ckpt = find_existing_checkpoint("resnet18_cifar10_best.pth")
    best_checkpoint_path = os.path.join(SAVED_MODELS_DIR, "resnet18_cifar10_best.pth")

    if existing_ckpt:
        logger.info(f"Found trained model checkpoint at '{existing_ckpt}'! Skipping training...")
        model = get_model(checkpoint_path=existing_ckpt, device=device)
        test_acc = evaluate_accuracy(model, test_loader, device=device)
        logger.info(f"Clean Test Accuracy: {test_acc:.2f}%")
        return model

    logger.info("--- Caching Training Set into VRAM for Ultra-Fast Training ---")
    train_subset = train_loader.dataset
    cache_loader = DataLoader(train_subset, batch_size=4096, shuffle=False, num_workers=0)
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
