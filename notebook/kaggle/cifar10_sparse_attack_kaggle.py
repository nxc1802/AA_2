"""
CIFAR-10 Sparse Adversarial Attack Benchmark & Proposed Framework (Single-File Kaggle Script)

Pipeline:
1. Load CIFAR-10 from HF (Cuong2004/AA) with stratified 40k train / 10k val / 10k test split.
2. Train CIFAR-adapted ResNet-18 with CUDA VRAM Tensor Caching & AMP (200 epochs).
3. Evaluate 3 Dense Baselines + 16 SOTA Sparse Attacks + 4 Proposed Methods (CPA, FCSA, FMSA, HSA) across K in {1, 2, 4, 8, 16, 32, 64, 128}.
   - High-Performance Vectorized PyTorch implementations for JSMA, OnePixel, SparseFool, CornerSearch, PGD0, SigmaZero, Homotopy, SAIF, IPFSA, GradientGuidance, GSE, SFA, Sparse-RS, BruSLe, Pixle.
4. Evaluate Unconstrained Minimum L0 Perturbation Suite for JSMA, SparseFool, PGD0, CornerSearch.
5. Export pivot tables & metrics to result/metrics/ (ASR-K, Robust Acc-K, PSNR, SSIM, LPIPS, L0, L2, L_inf, Min L0).
"""

import os
import sys
import time
import math
import json
import base64
import subprocess
from pathlib import Path

# Auto-install lpips if not present
try:
    import lpips
except ImportError:
    print("Installing lpips package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lpips", "--quiet", "--break-system-packages"])
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "lpips", "--quiet"])
        except Exception as e:
            print(f"Warning: Failed to install lpips ({e}). LPIPS metric will be omitted.")
    try:
        import lpips
    except ImportError:
        lpips = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision.transforms as transforms
from torchvision.models import resnet18

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Set working directory to /kaggle/working if on Kaggle, else current dir
if os.path.exists("/kaggle/working"):
    BASE_DIR = "/kaggle/working"
else:
    BASE_DIR = os.path.abspath(".")

RESULT_DIR = os.path.join(BASE_DIR, "result")
SAVED_MODELS_DIR = os.path.join(RESULT_DIR, "saved_models")
METRICS_DIR = os.path.join(RESULT_DIR, "metrics")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Dataset & Evaluation Config
HF_REPO_ID = "Cuong2004/AA"
HF_TOKEN = os.getenv("HF_TOKEN", None)
RANDOM_SEED = 42

TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
NUM_BENCHMARK_TEST_SAMPLES = 1000

if torch.cuda.is_available():
    try:
        major_cap = torch.cuda.get_device_capability(0)[0]
        if major_cap < 7:
            print(f"Warning: GPU '{torch.cuda.get_device_name(0)}' (sm_{major_cap}0) is not supported by PyTorch CUDA build (sm_70+ required). Falling back to CPU.")
            DEVICE = torch.device("cpu")
        else:
            DEVICE = torch.device("cuda")
    except Exception:
        DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"=== KAGGLE SPARSE ATTACK PIPELINE (Device: {DEVICE}) ===")

# ==============================================================================
# 1. DATASET LOADER WITH STRATIFIED 40K TRAIN / 10K VAL / 10K TEST SPLIT
# ==============================================================================
class HFDatasetWrapper(Dataset):
    def __init__(self, hf_ds, transform=None):
        self.hf_ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        raw_img = item.get("img", item.get("image"))
        img = raw_img.convert("RGB")
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataset_transforms(is_train=False):
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

def get_dataloaders(batch_size=128, seed=RANDOM_SEED):
    print(f"Loading CIFAR-10 from Hugging Face '{HF_REPO_ID}'...")
    hf_train_full = load_dataset(HF_REPO_ID, name="cifar10", split="train", token=HF_TOKEN)
    hf_test = load_dataset(HF_REPO_ID, name="cifar10", split="test", token=HF_TOKEN)

    all_indices = list(range(len(hf_train_full)))
    all_labels = hf_train_full["label"]
    train_idx, val_idx = train_test_split(
        all_indices, test_size=10000, random_state=seed, stratify=all_labels
    )

    pt_train_full = HFDatasetWrapper(hf_train_full, transform=get_dataset_transforms(is_train=True))
    pt_val_full = HFDatasetWrapper(hf_train_full, transform=get_dataset_transforms(is_train=False))
    pt_test = HFDatasetWrapper(hf_test, transform=get_dataset_transforms(is_train=False))

    train_subset = Subset(pt_train_full, train_idx)
    val_subset = Subset(pt_val_full, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(pt_test, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Loaded CIFAR-10: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(pt_test)}")
    return train_loader, val_loader, test_loader

# ==============================================================================
# 2. MODEL FACTORY & TRAINING
# ==============================================================================
def adapt_resnet_for_cifar(model, num_classes=10):
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model

def get_model(num_classes=10, checkpoint_path=None, device=DEVICE):
    model = resnet18(weights=None)
    model = adapt_resnet_for_cifar(model, num_classes=num_classes)
    model = model.to(device)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    return model

def evaluate_accuracy(model, data_loader, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / max(1, total)

def download_checkpoint_from_hf(path_in_repo="models/resnet18_cifar10_best.pth", repo_id=HF_REPO_ID, token=HF_TOKEN):
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=token
        )
        print(f"✅ Successfully downloaded checkpoint from HF '{repo_id}/{path_in_repo}' to '{local_path}'!")
        return local_path
    except Exception as e:
        print(f"⚠️ Checkpoint not found on HF '{repo_id}/{path_in_repo}': {e}")
        return None

def find_existing_checkpoint(filename="resnet18_cifar10_best.pth"):
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

    p1 = os.path.join(SAVED_MODELS_DIR, filename)
    if os.path.isfile(p1):
        return p1

    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f == filename or (f.endswith(".pth") and "resnet18" in f.lower()):
                    return os.path.join(root, f)

    return None

def train_clean_resnet18(train_loader, val_loader, test_loader, epochs=200, device=DEVICE):
    existing_ckpt = find_existing_checkpoint("resnet18_cifar10_best.pth")
    best_checkpoint_path = os.path.join(SAVED_MODELS_DIR, "resnet18_cifar10_best.pth")

    if existing_ckpt:
        print(f"Found trained model checkpoint at {existing_ckpt}! Skipping training...")
        model = get_model(checkpoint_path=existing_ckpt, device=device)
        test_acc = evaluate_accuracy(model, test_loader, device=device)
        print(f"Clean Test Accuracy: {test_acc:.2f}%")
        return model

    print("--- Caching Training Set into VRAM for Ultra-Fast Training ---")
    train_dataset = train_loader.dataset
    cached_x = [train_dataset[i][0] for i in range(len(train_dataset))]
    cached_y = [train_dataset[i][1] for i in range(len(train_dataset))]

    X_train = torch.stack(cached_x).to(device)
    Y_train = torch.tensor(cached_y, dtype=torch.long, device=device)

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
    batch_size = 128
    t0_train = time.time()

    print(f"Starting Training for {epochs} Epochs...")
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

        if epoch % 10 == 0 or epoch == epochs:
            val_acc = evaluate_accuracy(model, val_loader, device=device)
            print(f"Epoch [{epoch:03d}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.5f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, best_checkpoint_path)
                print(f"  >>> Best Checkpoint Saved! Val Acc: {best_val_acc:.2f}%")

    print(f"Training completed in {time.time()-t0_train:.2f}s!")
    model = get_model(checkpoint_path=best_checkpoint_path, device=device)
    clean_test_acc = evaluate_accuracy(model, test_loader, device=device)
    print(f"=== FINAL CLEAN TEST ACCURACY (10,000 Test Images): {clean_test_acc:.2f}% ===")
    return model

# ==============================================================================
# 3. HIGH-PERFORMANCE VECTORIZED SOTA SPARSE ATTACKS
# ==============================================================================

class JSMAAttack:
    """Batched High-Performance Jacobian-based Saliency Map Attack (JSMA)."""
    def __init__(self, model, k=15, theta=1.0, max_iter=25, device=DEVICE):
        self.model = model
        self.k = k
        self.theta = theta
        self.max_iter = max_iter
        self.device = device

    def attack(self, x, y):
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        with torch.no_grad():
            init_out = self.model(x)
            top2 = init_out.argsort(dim=1, descending=True)[:, :2]
            target_cls = torch.where(top2[:, 0] == y, top2[:, 1], top2[:, 0])

        perturbed_count = torch.zeros(B, dtype=torch.int32, device=self.device)
        mask_perturbed = torch.zeros((B, H, W), dtype=torch.bool, device=self.device)
        
        steps_to_fool = torch.full((B,), self.max_iter, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.max_iter):
            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step
            fooled_mask = fooled_mask | newly_fooled

            active = (preds == y) & (perturbed_count < self.k)
            if not active.any():
                break

            x_adv.requires_grad = True
            out = self.model(x_adv)
            loss = (out[torch.arange(B), target_cls] - out[torch.arange(B), y]).sum()
            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data.abs().sum(dim=1)  # (B, H, W)
            grad[mask_perturbed] = -1.0
            grad[~active] = -1.0

            flat_grad = grad.view(B, -1)
            best_idx = flat_grad.argmax(dim=1)

            row = best_idx // W
            col = best_idx % W

            x_adv = x_adv.detach()
            for b in range(B):
                if active[b]:
                    r, c = row[b].item(), col[b].item()
                    mask_perturbed[b, r, c] = True
                    perturbed_count[b] += 1
                    x_adv[b, :, r, c] = torch.clamp(x_adv[b, :, r, c] + self.theta, 0.0, 1.0)

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class OnePixelAttack:
    """Vectorized OnePixel / Multi-Pixel Differential Evolution Attack."""
    def __init__(self, model, k=1, max_iter=20, pop_size=20, device=DEVICE):
        self.model = model
        self.k = k
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
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


class SparseFoolAttack:
    """Vectorized SparseFool Attack ($L_0$ boundary projection)."""
    def __init__(self, model, k=15, steps=20, lambda_val=3.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.lambda_val = lambda_val
        self.device = device

    def attack(self, x, y):
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
            
            top2 = out.argsort(dim=1, descending=True)[:, :2]
            target_cls = torch.where(top2[:, 0] == y, top2[:, 1], top2[:, 0])

            loss = (out[torch.arange(B), target_cls] - out[torch.arange(B), y]).sum()
            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data
            grad_mag = grad.abs().sum(dim=1)  # (B, H, W)
            flat_grad = grad_mag.view(B, -1)

            topk_vals, _ = torch.topk(flat_grad, k=min(self.k, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            sparse_mask = (grad_mag >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_direction = grad.sign() * sparse_mask * active_mask

            with torch.no_grad():
                x_adv = torch.clamp(x_adv + (1.0 / 255.0) * self.lambda_val * step_direction, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class CornerSearchAttack:
    """Vectorized CornerSearch Attack."""
    def __init__(self, model, k=15, max_iter=20, device=DEVICE):
        self.model = model
        self.k = k
        self.max_iter = max_iter
        self.device = device

    def attack(self, x, y):
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        max_steps = min(self.k, self.max_iter)
        steps_to_fool = torch.full((B,), max_steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(max_steps):
            if fooled_mask.all():
                break

            x_adv.requires_grad = True
            loss = nn.CrossEntropyLoss()(self.model(x_adv), y)
            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data.abs().sum(dim=1)  # (B, H, W)
            top_idx = grad.view(B, -1).argmax(dim=1)  # (B,)

            row = top_idx // W
            col = top_idx % W

            with torch.no_grad():
                cand0 = x_adv.clone()
                cand1 = x_adv.clone()
                active = (~fooled_mask)
                for b in range(B):
                    if active[b]:
                        r, c = row[b].item(), col[b].item()
                        cand0[b, :, r, c] = 0.0
                        cand1[b, :, r, c] = 1.0

                l0 = F.cross_entropy(self.model(cand0), y, reduction='none')
                l1 = F.cross_entropy(self.model(cand1), y, reduction='none')

                use_c1 = (l1 > l0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                new_adv = torch.where(use_c1, cand1, cand0)
                x_adv = torch.where(active.view(B, 1, 1, 1), new_adv, x_adv).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class PGD0Attack:
    """Authentic PGD-0 Attack (PGD with hard $L_0$-ball projection)."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
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
            diff = (x_adv + self.alpha * grad.sign()) - x
            diff_mag = diff.abs().sum(dim=1)  # (B, H, W)

            flat_diff = diff_mag.view(B, -1)
            topk_vals, _ = torch.topk(flat_diff, k=min(self.k, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            mask = (diff_mag >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            new_adv = torch.clamp(x + diff * mask, 0.0, 1.0)
            x_adv = torch.where(active_mask.bool(), new_adv, x_adv).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class SigmaZeroAttack:
    """Authentic SigmaZero Attack (Adaptive Hard-Thresholding L0-PGD)."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        B, C, H, W = x.shape
        x_adv = x.clone().detach()
        accum_grad = torch.zeros_like(x)

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
            accum_grad = 0.9 * accum_grad + grad
            score = accum_grad.abs().sum(dim=1)  # (B, H, W)

            flat_score = score.view(B, -1)
            topk_vals, _ = torch.topk(flat_score, k=min(self.k, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            mask = (score >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * accum_grad.sign() * mask * active_mask
            x_adv = torch.clamp(x_adv + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class HomotopyAttack:
    """Authentic Homotopy Continuation Sparse Attack."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
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

            gamma = (step + 1) / float(self.steps)
            x_adv.requires_grad = True
            out = self.model(x_adv)
            loss_cls = self.criterion(out, y)

            diff = x_adv - x
            loss_l0_proxy = torch.sum(diff.abs() / (diff.abs() + 1e-3))
            loss = loss_cls - gamma * 0.01 * loss_l0_proxy

            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data
            score = grad.abs().sum(dim=1)
            flat_score = score.view(B, -1)
            topk_vals, _ = torch.topk(flat_score, k=min(self.k, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            mask = (score >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * mask * active_mask
            x_adv = torch.clamp(x_adv + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class SAIFAttack:
    """Authentic Sparsity-Aware Iterative Fast Attack (SAIF)."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        B, C, H, W = x.shape
        x_adv = x.clone().detach()
        cum_grad_mag = torch.zeros((B, H, W), device=self.device)

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
            cum_grad_mag += grad.abs().sum(dim=1)

            flat_cum = cum_grad_mag.view(B, -1)
            topk_vals, _ = torch.topk(flat_cum, k=min(self.k, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            mask = (cum_grad_mag >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * mask * active_mask
            x_adv = torch.clamp(x_adv + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class IPFSAttack:
    """Authentic Iterative Pixel Filtered Sparse Attack (IPFSA)."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

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
            grad_mag = grad.abs().sum(dim=1, keepdim=True)
            spatial_var = F.conv2d(grad_mag, laplacian_kernel, padding=1).abs().squeeze(1)

            filtered_score = grad_mag.squeeze(1) * (1.0 + 0.5 * spatial_var)
            flat_score = filtered_score.view(B, -1)
            topk_vals, _ = torch.topk(flat_score, k=min(self.k, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            mask = (filtered_score >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * mask * active_mask
            x_adv = torch.clamp(x_adv + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class GradientGuidanceAttack:
    """Authentic Gradient Guidance Sparse Attack."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device

    def attack(self, x, y):
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

            top2 = out.argsort(dim=1, descending=True)[:, :2]
            target_cls = torch.where(top2[:, 0] == y, top2[:, 1], top2[:, 0])

            margin_loss = (out[torch.arange(B), target_cls] - out[torch.arange(B), y]).sum()
            self.model.zero_grad()
            margin_loss.backward()

            grad = x_adv.grad.data
            grad_score = grad.abs().sum(dim=1)

            flat_score = grad_score.view(B, -1)
            topk_vals, _ = torch.topk(flat_score, k=min(self.k, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            mask = (grad_score >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * mask * active_mask
            x_adv = torch.clamp(x_adv + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class GroupSparseAttack:
    """Authentic Group Sparse Attack (GSE - 2x2 Spatial Blocks)."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
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
            grad_mag = grad.abs().sum(dim=1, keepdim=True)
            group_pool = F.avg_pool2d(grad_mag, kernel_size=2, stride=2)

            flat_group = group_pool.view(B, -1)
            num_groups = max(1, self.k // 4)
            topk_vals, _ = torch.topk(flat_group, k=min(num_groups, flat_group.size(1)), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1, 1)

            group_mask = (group_pool >= thresh).float()
            spatial_mask = F.interpolate(group_mask, scale_factor=2, mode='nearest')

            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * spatial_mask * active_mask
            x_adv = torch.clamp(x_adv + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class SpectralFrequencyAttack:
    """Authentic Spectral Frequency Attack (SFA - FFT Frequency Domain)."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
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


class SparseRSAttack:
    """Vectorized Sparse Random Search Attack (Sparse-RS)."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            best_losses = F.cross_entropy(self.model(x_adv), y, reduction='none')
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            cand = x_adv.clone()
            coords = torch.randint(0, H * W, (B, self.k), device=self.device)
            signs = (torch.randint(0, 2, (B, self.k, C), device=self.device).float() * 2.0 - 1.0) * self.alpha

            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, self.k)
            c_y = coords // W
            c_x = coords % W

            for c_ch in range(C):
                cand[b_idx, c_ch, c_y, c_x] = torch.clamp(cand[b_idx, c_ch, c_y, c_x] + signs[:, :, c_ch], 0.0, 1.0)

            active_mask = (~fooled_mask).view(B, 1, 1, 1)
            cand = torch.where(active_mask, cand, x_adv)

            with torch.no_grad():
                l_cand = F.cross_entropy(self.model(cand), y, reduction='none')
                improved = (l_cand > best_losses) & (~fooled_mask)
                x_adv = torch.where(improved.unsqueeze(1).unsqueeze(2).unsqueeze(3), cand, x_adv)
                best_losses = torch.where(improved, l_cand, best_losses)

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class BruSLeAttack:
    """Vectorized Patch-based Random Search Sparse Attack (BruSLe)."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        B, C, H, W = x.shape
        x_adv = x.clone().detach()
        patch_dim = max(1, int(math.sqrt(self.k)))

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            best_losses = F.cross_entropy(self.model(x_adv), y, reduction='none')
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            cand = x_adv.clone()
            top_y = torch.randint(0, H - patch_dim + 1, (B,), device=self.device)
            top_x = torch.randint(0, W - patch_dim + 1, (B,), device=self.device)

            noise = (torch.randint(0, 2, (B, C, patch_dim, patch_dim), device=self.device).float() * 2.0 - 1.0) * self.alpha

            active = (~fooled_mask)
            for b in range(B):
                if active[b]:
                    ty, tx = top_y[b].item(), top_x[b].item()
                    cand[b, :, ty:ty+patch_dim, tx:tx+patch_dim] = torch.clamp(
                        cand[b, :, ty:ty+patch_dim, tx:tx+patch_dim] + noise[b], 0.0, 1.0
                    )

            cand = torch.where(active.view(B, 1, 1, 1), cand, x_adv)

            with torch.no_grad():
                l_cand = F.cross_entropy(self.model(cand), y, reduction='none')
                improved = (l_cand > best_losses) & (~fooled_mask)
                x_adv = torch.where(improved.unsqueeze(1).unsqueeze(2).unsqueeze(3), cand, x_adv)
                best_losses = torch.where(improved, l_cand, best_losses)

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


class PixleAttack:
    """Vectorized Pixle Attack (Pixel Patch Rearrangement / Perturbation)."""
    def __init__(self, model, k=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, x, y):
        B, C, H, W = x.shape
        x_adv = x.clone().detach()

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            best_losses = F.cross_entropy(self.model(x_adv), y, reduction='none')
            preds = self.model(x_adv).argmax(dim=1)
        fooled_mask = (preds != y)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            cand = x_adv.clone()
            coords = torch.randint(0, H * W, (B, self.k), device=self.device)
            rand_vals = torch.rand((B, self.k, C), device=self.device)

            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, self.k)
            c_y = coords // W
            c_x = coords % W

            for c_ch in range(C):
                cand[b_idx, c_ch, c_y, c_x] = rand_vals[:, :, c_ch]

            active_mask = (~fooled_mask).view(B, 1, 1, 1)
            cand = torch.where(active_mask, cand, x_adv)

            with torch.no_grad():
                l_cand = F.cross_entropy(self.model(cand), y, reduction='none')
                improved = (l_cand > best_losses) & (~fooled_mask)
                x_adv = torch.where(improved.unsqueeze(1).unsqueeze(2).unsqueeze(3), cand, x_adv)
                best_losses = torch.where(improved, l_cand, best_losses)

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv


# ==============================================================================
# 4. PROPOSED SPARSE ATTACK METHODS (CPA, FCSA, FMSA, HSA)
# ==============================================================================
class CooperativePixelsAttack:
    def __init__(self, model, coalition_size=15, steps=25, alpha=4/255.0, coop_weight=0.5, device=DEVICE):
        self.model, self.coalition_size, self.steps, self.alpha = model, coalition_size, steps, alpha
        self.coop_weight, self.device = coop_weight, device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, images, labels):
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape
        adv_images = orig_images.clone().detach()

        kernel = torch.ones(1, 1, 3, 3, device=self.device) / 9.0

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(adv_images).argmax(dim=1)
        fooled_mask = (preds != labels)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            grad_mag = grad.abs().sum(dim=1)
            local_coop = F.conv2d(grad_mag.unsqueeze(1), kernel, padding=1).squeeze(1)
            coop_score = grad_mag + self.coop_weight * local_coop

            flat_score = coop_score.view(B, -1)
            topk_vals, _ = torch.topk(flat_score, k=min(self.coalition_size, H*W), dim=1)
            k_th_thresh = topk_vals[:, -1].view(B, 1, 1)

            coalition_mask = (coop_score >= k_th_thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * coalition_mask * active_mask
            adv_images = torch.clamp(adv_images + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(adv_images).argmax(dim=1)
            current_fooled = (preds != labels)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return adv_images

class FunctionalCoalitionSparseAttack:
    def __init__(self, model, max_coalition_size=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model, self.max_coalition_size, self.steps, self.alpha, self.device = model, max_coalition_size, steps, alpha, device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, images, labels):
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape
        adv_images = orig_images.clone().detach()

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(adv_images).argmax(dim=1)
        fooled_mask = (preds != labels)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            grad_mean = grad.abs().mean(dim=1)
            grad_max = grad.abs().max(dim=1)[0]
            coalition_score = grad_mean * grad_max

            flat_grad = coalition_score.view(B, -1)
            topk_vals, _ = torch.topk(flat_grad, k=min(self.max_coalition_size, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            coalition_mask = (coalition_score >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * coalition_mask * active_mask
            adv_images = torch.clamp(adv_images + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(adv_images).argmax(dim=1)
            current_fooled = (preds != labels)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return adv_images

class FeatureToMinimalSupportAttack:
    def __init__(self, model, support_budget=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model, self.support_budget, self.steps, self.alpha, self.device = model, support_budget, steps, alpha, device
        self.extracted_features = None
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            self.extracted_features = output
        if hasattr(self.model, "layer4"):
            self.model.layer4.register_forward_hook(hook)

    def attack(self, images, labels):
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape
        adv_images = orig_images.clone().detach()

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(adv_images).argmax(dim=1)
        fooled_mask = (preds != labels)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            if self.extracted_features is not None:
                loss = -torch.mean(self.extracted_features.abs())
            else:
                loss = -nn.CrossEntropyLoss()(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            grad_mag = grad.abs().sum(dim=1)
            flat_grad = grad_mag.view(B, -1)
            topk_vals, _ = torch.topk(flat_grad, k=min(self.support_budget, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            support_mask = (grad_mag >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * support_mask * active_mask
            adv_images = torch.clamp(adv_images + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(adv_images).argmax(dim=1)
            current_fooled = (preds != labels)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return adv_images

class HypergraphSparseAttack:
    def __init__(self, model, budget=15, steps=25, alpha=4/255.0, device=DEVICE):
        self.model, self.budget, self.steps, self.alpha, self.device = model, budget, steps, alpha, device
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, images, labels):
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape
        adv_images = orig_images.clone().detach()

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(adv_images).argmax(dim=1)
        fooled_mask = (preds != labels)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            grad_mag = grad.abs().sum(dim=1)

            p1 = F.avg_pool2d(grad_mag, kernel_size=3, stride=1, padding=1)
            p2 = F.avg_pool2d(grad_mag, kernel_size=5, stride=1, padding=2)
            node_centrality = grad_mag + p1 + p2

            flat_centrality = node_centrality.view(B, -1)
            topk_vals, _ = torch.topk(flat_centrality, k=min(self.budget, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            hypergraph_mask = (node_centrality >= thresh).unsqueeze(1).float()
            
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * hypergraph_mask * active_mask
            adv_images = torch.clamp(adv_images + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(adv_images).argmax(dim=1)
            current_fooled = (preds != labels)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return adv_images

# ==============================================================================
# 5. DENSE BASELINES
# ==============================================================================
class FGSMAttack:
    def __init__(self, model, eps=8/255.0, device=DEVICE):
        self.model, self.eps, self.device = model, eps, device
        self.criterion = nn.CrossEntropyLoss()
    def attack(self, x, y):
        x_adv = x.clone().detach().requires_grad_(True)
        out = self.model(x_adv)
        loss = self.criterion(out, y)
        self.model.zero_grad()
        loss.backward()
        self.last_steps = [1] * x.size(0)
        return torch.clamp(x + self.eps * x_adv.grad.sign(), 0, 1).detach()

class BIMAttack:
    def __init__(self, model, eps=8/255.0, alpha=2/255.0, steps=10, device=DEVICE):
        self.model, self.eps, self.alpha, self.steps, self.device = model, eps, alpha, steps, device
        self.criterion = nn.CrossEntropyLoss()
    def attack(self, x, y):
        B = x.size(0)
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

            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_adv = x_adv + self.alpha * x_adv.grad.sign() * active_mask
            eta = torch.clamp(step_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + eta, 0, 1).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv

class PGDAttack:
    def __init__(self, model, eps=8/255.0, alpha=2/255.0, steps=20, device=DEVICE):
        self.model, self.eps, self.alpha, self.steps, self.device = model, eps, alpha, steps, device
        self.criterion = nn.CrossEntropyLoss()
    def attack(self, x, y):
        B = x.size(0)
        x_adv = x.clone().detach() + torch.FloatTensor(*x.shape).uniform_(-self.eps, self.eps).to(self.device)
        x_adv = torch.clamp(x_adv, 0, 1).detach()

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

            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_adv = x_adv + self.alpha * x_adv.grad.sign() * active_mask
            eta = torch.clamp(step_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + eta, 0, 1).detach()

            with torch.no_grad():
                preds = self.model(x_adv).argmax(dim=1)
            current_fooled = (preds != y)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return x_adv

# ==============================================================================
# 6. METRIC COMPUTATION & BENCHMARK EVALUATION ENGINE
# ==============================================================================
def compute_psnr(orig, adv):
    mse = torch.mean((orig - adv) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.mean().item()

def compute_ssim(orig, adv):
    kernel = torch.ones((3, 1, 3, 3), device=orig.device) / 9.0
    mu_x = F.conv2d(orig, kernel, padding=1, groups=3)
    mu_y = F.conv2d(adv, kernel, padding=1, groups=3)

    sigma_x_sq = F.conv2d(orig * orig, kernel, padding=1, groups=3) - mu_x * mu_x
    sigma_y_sq = F.conv2d(adv * adv, kernel, padding=1, groups=3) - mu_y * mu_y
    sigma_xy = F.conv2d(orig * adv, kernel, padding=1, groups=3) - mu_x * mu_y

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean().item()

def run_unconstrained_minimum_l0_benchmark(model, test_loader, device=DEVICE):
    print("\n======================================================================")
    print("--- RUNNING UNCONSTRAINED MINIMUM L0 PERTURBATION SUITE ---")
    print("======================================================================")
    test_ds = test_loader.dataset
    if NUM_BENCHMARK_TEST_SAMPLES and NUM_BENCHMARK_TEST_SAMPLES < len(test_ds):
        test_ds = Subset(test_ds, range(NUM_BENCHMARK_TEST_SAMPLES))
    eval_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    unconstrained_attacks = {
        "JSMA (Unconstrained)": JSMAAttack(model, k=250, max_iter=50, device=device),
        "SparseFool (Unconstrained)": SparseFoolAttack(model, k=250, steps=50, device=device),
        "PGD0 (Unconstrained)": PGD0Attack(model, k=250, steps=50, device=device),
        "CornerSearch (Unconstrained)": CornerSearchAttack(model, k=250, max_iter=30, device=device)
    }

    min_l0_results = []
    for name, attacker in unconstrained_attacks.items():
        t0 = time.time()
        success_count, total_count = 0, 0
        l0_list, l2_list = [], []

        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            with torch.no_grad():
                clean_preds = model(x).argmax(dim=1)
            c_mask = (clean_preds == y)

            x_adv = attacker.attack(x, y)
            with torch.no_grad():
                adv_preds = model(x_adv).argmax(dim=1)

            succ_mask = c_mask & (adv_preds != y)
            diff = (x_adv - x).abs()
            l0_per = torch.sum(diff.max(dim=1)[0] > 1e-4, dim=(1, 2)).float()
            l2_per = torch.norm(diff.view(B, -1), p=2, dim=1)

            success_count += succ_mask.sum().item()
            total_count += c_mask.sum().item()

            if succ_mask.any():
                l0_list.extend(l0_per[succ_mask].cpu().numpy().tolist())
                l2_list.extend(l2_per[succ_mask].cpu().numpy().tolist())

        dt = time.time() - t0
        asr = 100.0 * success_count / max(1, total_count)
        mean_l0 = pd.Series(l0_list).mean() if l0_list else float('nan')
        median_l0 = pd.Series(l0_list).median() if l0_list else float('nan')
        mean_l2 = pd.Series(l2_list).mean() if l2_list else float('nan')

        res = {
            "Attack Method": name,
            "ASR (%)": round(asr, 2),
            "Mean L0 Pixels": round(mean_l0, 2) if not math.isnan(mean_l0) else "N/A",
            "Median L0 Pixels": round(median_l0, 2) if not math.isnan(median_l0) else "N/A",
            "Mean L2": round(mean_l2, 4) if not math.isnan(mean_l2) else "N/A",
            "Avg Time/Img (s)": round(dt / max(1, total_count), 4)
        }
        print(f"[{name}] ASR={res['ASR (%)']}%, Mean L0={res['Mean L0 Pixels']}, Median L0={res['Median L0 Pixels']}")
        min_l0_results.append(res)

    df_min_l0 = pd.DataFrame(min_l0_results)
    df_min_l0.to_csv(os.path.join(METRICS_DIR, "min_l0_unconstrained_metrics.csv"), index=False)
    with open(os.path.join(METRICS_DIR, "min_l0_unconstrained_metrics.json"), "w") as f:
        json.dump(df_min_l0.to_dict(orient="records"), f, indent=4)

    return df_min_l0


def run_attack_benchmark_suite(model, test_loader, eval_batch_size=EVAL_BATCH_SIZE, num_samples=NUM_BENCHMARK_TEST_SAMPLES, device=DEVICE):
    test_ds = test_loader.dataset
    if num_samples and num_samples < len(test_ds):
        print(f"Subsetting test set to {num_samples} samples for benchmark evaluation...")
        test_ds = Subset(test_ds, range(num_samples))

    eval_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    print(f"Evaluation DataLoader created with batch_size={eval_batch_size}, total_samples={len(test_ds)}")

    lpips_fn = None
    if lpips is not None:
        try:
            lpips_fn = lpips.LPIPS(net='alex').to(device)
        except Exception:
            lpips_fn = None

    K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
    results_list = []
    full_csv_path = os.path.join(METRICS_DIR, "full_attack_metrics.csv")

    # ==========================================================================
    # GROUP C: Non-pixel-K Attacks (Dense + Spectral Frequency Domain)
    # ==========================================================================
    print("=== GROUP C: Non-pixel-K Attacks (FGSM, BIM, PGD, SFA) ===")
    group_c_attacks = {
        "FGSM": FGSMAttack(model, device=device),
        "BIM": BIMAttack(model, device=device),
        "PGD": PGDAttack(model, device=device),
        "SFA": SpectralFrequencyAttack(model, freq_k=15, device=device)
    }

    for name, attacker in group_c_attacks.items():
        t0 = time.time()
        clean_correct, total_count, robust_correct, adv_succ = 0, 0, 0, 0
        total_l0, total_l2, total_linf = 0.0, 0.0, 0.0
        total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
        total_steps = 0.0

        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            with torch.no_grad():
                clean_preds = torch.argmax(model(x), dim=1)
            c_mask = (clean_preds == y)

            x_adv = attacker.attack(x, y)
            
            if hasattr(attacker, "last_steps"):
                total_steps += sum(attacker.last_steps)
            else:
                steps = getattr(attacker, "steps", getattr(attacker, "max_iter", 1))
                total_steps += steps * B

            with torch.no_grad():
                adv_preds = torch.argmax(model(x_adv), dim=1)

            r_mask = (adv_preds == y)
            diff = (x_adv - x).abs()
            l0_per = torch.sum(diff.max(dim=1)[0] > 1e-4, dim=(1, 2)).float()
            l2_per = torch.norm(diff.view(B, -1), p=2, dim=1)
            linf_per = torch.norm(diff.view(B, -1), p=float('inf'), dim=1)

            clean_correct += c_mask.sum().item()
            robust_correct += r_mask.sum().item()
            total_count += B
            adv_succ += (c_mask & (~r_mask)).sum().item()

            total_l0 += l0_per.sum().item()
            total_l2 += l2_per.sum().item()
            total_linf += linf_per.sum().item()

            total_psnr += compute_psnr(x, x_adv) * B
            total_ssim += compute_ssim(x, x_adv) * B
            if lpips_fn:
                with torch.no_grad():
                    total_lpips += lpips_fn(x * 2 - 1, x_adv * 2 - 1).mean().item() * B

        dt = time.time() - t0
        clean_acc = 100.0 * clean_correct / total_count
        rob_acc = 100.0 * robust_correct / total_count
        asr = 100.0 * adv_succ / max(1, clean_correct)

        res = {
            "Group": "Group C", "Attack Method": name, "K": "N/A",
            "Clean Acc (%)": round(clean_acc, 2), "Robust Acc (%)": round(rob_acc, 2),
            "ASR (%)": round(asr, 2), "Accuracy Drop (%)": round(clean_acc - rob_acc, 2),
            "Avg L0": round(total_l0 / total_count, 2), "Avg L0 Ratio": round((total_l0 / total_count) / 1024.0, 4),
            "Avg L2": round(total_l2 / total_count, 4), "Avg L_inf": round(total_linf / total_count, 4),
            "PSNR (dB)": round(total_psnr / total_count, 2), "SSIM": round(total_ssim / total_count, 4),
            "LPIPS": round(total_lpips / total_count, 4) if lpips_fn else float('nan'),
            "Avg Iterations": round(total_steps / total_count, 2),
            "Time/Img (s)": round(dt / total_count, 4)
        }
        print(f"[Group C] {name}: ASR={res['ASR (%)']}%, Robust Acc={res['Robust Acc (%)']}%, Avg L0={res['Avg L0']}")
        results_list.append(res)
        pd.DataFrame(results_list).to_csv(full_csv_path, index=False)

    # ==========================================================================
    # GROUP A: Direct K-Sweep Attacks (Budget Constrained)
    # ==========================================================================
    print("=== GROUP A: Direct K-Sweep Attacks (Explicit K-budget) ===")
    group_a_factories = {
        "JSMA": lambda m, k, d: JSMAAttack(m, k=k, device=d),
        "OnePixel": lambda m, k, d: OnePixelAttack(m, k=k, device=d),
        "CornerSearch": lambda m, k, d: CornerSearchAttack(m, k=k, device=d),
        "SAIF": lambda m, k, d: SAIFAttack(m, k=k, device=d),
        "PGD0": lambda m, k, d: PGD0Attack(m, k=k, device=d),
        "Sparse-PGD": lambda m, k, d: SparsePGDAttack(m, sparsity_budget=k, device=d),
        "Sparse-RS": lambda m, k, d: SparseRSAttack(m, n_pixels=k, device=d),
        "BruSLe": lambda m, k, d: BruSLeAttack(m, block_size=max(1, int(k**0.5)), device=d),
        "IPFSA": lambda m, k, d: IPFSAttack(m, k_pixels=k, device=d),
        "GradientGuidance": lambda m, k, d: GradientGuidanceAttack(m, sparsity_budget=k, device=d),
        "CPA": lambda m, k, d: CooperativePixelsAttack(m, coalition_size=k, device=d),
        "FCSA": lambda m, k, d: FunctionalCoalitionSparseAttack(m, max_coalition_size=k, device=d),
        "FMSA-budgeted": lambda m, k, d: FeatureToMinimalSupportAttack(m, support_budget=k, device=d),
        "HSA-budgeted": lambda m, k, d: HypergraphSparseAttack(m, budget=k, device=d)
    }

    for name, factory in group_a_factories.items():
        for K in K_VALUES:
            attacker = factory(model, K, device)
            t0 = time.time()
            clean_correct, total_count, robust_correct, adv_succ = 0, 0, 0, 0
            total_l0, total_l2, total_linf = 0.0, 0.0, 0.0
            total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
            total_steps = 0.0

            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                B = x.size(0)
                with torch.no_grad():
                    clean_preds = torch.argmax(model(x), dim=1)
                c_mask = (clean_preds == y)

                x_adv = attacker.attack(x, y)
                
                if hasattr(attacker, "last_steps"):
                    total_steps += sum(attacker.last_steps)
                else:
                    steps = getattr(attacker, "steps", getattr(attacker, "max_iter", 1))
                    total_steps += steps * B

                with torch.no_grad():
                    adv_preds = torch.argmax(model(x_adv), dim=1)

                r_mask = (adv_preds == y)
                diff = (x_adv - x).abs()
                l0_per = torch.sum(diff.max(dim=1)[0] > 1e-4, dim=(1, 2)).float()
                l2_per = torch.norm(diff.view(B, -1), p=2, dim=1)
                linf_per = torch.norm(diff.view(B, -1), p=float('inf'), dim=1)

                clean_correct += c_mask.sum().item()
                robust_correct += r_mask.sum().item()
                total_count += B
                adv_succ += (c_mask & (~r_mask)).sum().item()

                total_l0 += l0_per.sum().item()
                total_l2 += l2_per.sum().item()
                total_linf += linf_per.sum().item()

                total_psnr += compute_psnr(x, x_adv) * B
                total_ssim += compute_ssim(x, x_adv) * B
                if lpips_fn:
                    with torch.no_grad():
                        total_lpips += lpips_fn(x * 2 - 1, x_adv * 2 - 1).mean().item() * B

            dt = time.time() - t0
            clean_acc = 100.0 * clean_correct / total_count
            rob_acc = 100.0 * robust_correct / total_count
            asr = 100.0 * adv_succ / max(1, clean_correct)
            avg_l0_val = round(total_l0 / total_count, 2)

            res = {
                "Group": "Group A", "Attack Method": name, "K": K,
                "Clean Acc (%)": round(clean_acc, 2), "Robust Acc (%)": round(rob_acc, 2),
                "ASR (%)": round(asr, 2), "Accuracy Drop (%)": round(clean_acc - rob_acc, 2),
                "Avg L0": avg_l0_val, "Avg L0 Ratio": round(avg_l0_val / 1024.0, 4),
                "Avg L2": round(total_l2 / total_count, 4), "Avg L_inf": round(total_linf / total_count, 4),
                "PSNR (dB)": round(total_psnr / total_count, 2), "SSIM": round(total_ssim / total_count, 4),
                "LPIPS": round(total_lpips / total_count, 4) if lpips_fn else float('nan'),
                "Avg Iterations": round(total_steps / total_count, 2),
                "Time/Img (s)": round(dt / total_count, 4)
            }
            print(f"[Group A] {name} (K={K}): ASR={res['ASR (%)']}%, Robust Acc={res['Robust Acc (%)']}%, Avg L0={res['Avg L0']}")
            results_list.append(res)
            pd.DataFrame(results_list).to_csv(full_csv_path, index=False)

    # ==========================================================================
    # GROUP B: Unconstrained Minimum Support Optimization -> Calculate ASR@K
    # ==========================================================================
    print("=== GROUP B: Minimal Support Optimization (Post-hoc ASR@K Evaluation) ===")
    group_b_attacks = {
        "SparseFool": SparseFoolAttack(model, k=250, steps=50, device=device),
        "SigmaZero": SigmaZeroAttack(model, steps=50, device=device),
        "Homotopy": HomotopyAttack(model, target_sparsity=250, steps=50, device=device),
        "GSE": GroupSparseAttack(model, group_size=4, max_groups=64, steps=50, device=device),
        "Pixle": PixleAttack(model, n_swaps=20, max_trials=50, device=device),
        "FMSA-minimal-support": FeatureToMinimalSupportAttack(model, support_budget=250, device=device)
    }

    for name, attacker in group_b_attacks.items():
        t0 = time.time()
        clean_correct, total_count = 0, 0
        sample_l0s, sample_l2s, sample_linfs = [], [], []
        sample_fooled = []
        total_steps = 0.0

        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            with torch.no_grad():
                clean_preds = torch.argmax(model(x), dim=1)
            c_mask = (clean_preds == y)

            x_adv = attacker.attack(x, y)
            
            if hasattr(attacker, "last_steps"):
                total_steps += sum(attacker.last_steps)
            else:
                steps = getattr(attacker, "steps", getattr(attacker, "max_iter", 1))
                total_steps += steps * B

            with torch.no_grad():
                adv_preds = torch.argmax(model(x_adv), dim=1)

            fooled_mask = c_mask & (adv_preds != y)
            diff = (x_adv - x).abs()
            l0_per = torch.sum(diff.max(dim=1)[0] > 1e-4, dim=(1, 2)).float()
            l2_per = torch.norm(diff.view(B, -1), p=2, dim=1)
            linf_per = torch.norm(diff.view(B, -1), p=float('inf'), dim=1)

            clean_correct += c_mask.sum().item()
            total_count += B

            sample_fooled.extend(fooled_mask.cpu().numpy().tolist())
            sample_l0s.extend(l0_per.cpu().numpy().tolist())
            sample_l2s.extend(l2_per.cpu().numpy().tolist())
            sample_linfs.extend(linf_per.cpu().numpy().tolist())

        dt = time.time() - t0
        clean_acc = 100.0 * clean_correct / total_count

        fooled_arr = pd.Series(sample_fooled)
        l0_arr = pd.Series(sample_l0s)
        l2_arr = pd.Series(sample_l2s)
        linf_arr = pd.Series(sample_linfs)

        for K in K_VALUES:
            succ_k = fooled_arr & (l0_arr <= K)
            asr_k = 100.0 * succ_k.sum() / max(1, clean_correct)
            rob_acc_k = clean_acc - asr_k

            res = {
                "Group": "Group B", "Attack Method": name, "K": K,
                "Clean Acc (%)": round(clean_acc, 2), "Robust Acc (%)": round(rob_acc_k, 2),
                "ASR (%)": round(asr_k, 2), "Accuracy Drop (%)": round(clean_acc - rob_acc_k, 2),
                "Avg L0": round(l0_arr.mean(), 2), "Avg L0 Ratio": round(l0_arr.mean() / 1024.0, 4),
                "Avg L2": round(l2_arr.mean(), 4), "Avg L_inf": round(linf_arr.mean(), 4),
                "PSNR (dB)": float('nan'), "SSIM": float('nan'), "LPIPS": float('nan'),
                "Avg Iterations": round(total_steps / total_count, 2),
                "Time/Img (s)": round(dt / total_count, 4)
            }
            print(f"[Group B] {name} (K={K}): ASR@K={res['ASR (%)']}%, Mean L0={res['Avg L0']}")
            results_list.append(res)
            pd.DataFrame(results_list).to_csv(full_csv_path, index=False)

    df_all = pd.DataFrame(results_list)
    df_all.to_csv(full_csv_path, index=False)
    with open(os.path.join(METRICS_DIR, "full_attack_metrics.json"), "w") as f:
        json.dump(df_all.to_dict(orient="records"), f, indent=4)

    return df_all

# ==============================================================================
# 7. PIVOT TABLES & REPORT GENERATION
# ==============================================================================
def generate_reports(df):
    df_sparse = df[pd.to_numeric(df["K"], errors="coerce").notna()].copy()
    if len(df_sparse) > 0:
        df_sparse["K"] = pd.to_numeric(df_sparse["K"])
        idx_cols = ["Group", "Attack Method"] if "Group" in df_sparse.columns else ["Attack Method"]

        asr_pivot = df_sparse.pivot(index=idx_cols, columns="K", values="ASR (%)").reset_index()
        asr_pivot.to_csv(os.path.join(METRICS_DIR, "asr_k_pivot.csv"), index=False)
        with open(os.path.join(METRICS_DIR, "asr_k_pivot.json"), "w") as f:
            json.dump(asr_pivot.to_dict(orient="records"), f, indent=4)

        rob_pivot = df_sparse.pivot(index=idx_cols, columns="K", values="Robust Acc (%)").reset_index()
        rob_pivot.to_csv(os.path.join(METRICS_DIR, "robust_accuracy_k_pivot.csv"), index=False)
        with open(os.path.join(METRICS_DIR, "robust_accuracy_k_pivot.json"), "w") as f:
            json.dump(rob_pivot.to_dict(orient="records"), f, indent=4)

        iter_pivot = df_sparse.pivot(index=idx_cols, columns="K", values="Avg Iterations").reset_index()
        iter_pivot.to_csv(os.path.join(METRICS_DIR, "iterations_k_pivot.csv"), index=False)
        with open(os.path.join(METRICS_DIR, "iterations_k_pivot.json"), "w") as f:
            json.dump(iter_pivot.to_dict(orient="records"), f, indent=4)

    cols = ["Group", "Attack Method", "K", "PSNR (dB)", "SSIM", "LPIPS", "Avg L0", "Avg L2", "Avg L_inf", "Avg Iterations"]
    cols = [c for c in cols if c in df.columns]
    img_quality = df[cols].copy()
    img_quality.to_csv(os.path.join(METRICS_DIR, "image_quality_metrics.csv"), index=False)
    with open(os.path.join(METRICS_DIR, "image_quality_metrics.json"), "w") as f:
        json.dump(img_quality.to_dict(orient="records"), f, indent=4)

    print("=== SUMMARY PIVOT TABLES GENERATED ===")
    print("ASR - K Pivot Table Sample:")
    print(asr_pivot.head(15).to_string())

# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=TRAIN_BATCH_SIZE)
    model = train_clean_resnet18(train_loader, val_loader, test_loader, epochs=200, device=DEVICE)
    df_metrics = run_attack_benchmark_suite(model, test_loader, device=DEVICE)
    generate_reports(df_metrics)
    print("=== ALL KAGGLE PIPELINE STAGES COMPLETED SUCCESSFULLY! ===")
