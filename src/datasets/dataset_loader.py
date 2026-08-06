# ==============================================================================
# IMPORTS & DEPENDENCIES
# ==============================================================================
import os
import sys
import logging
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURABLE PARAMETERS & PATHS (GPU 96GB VRAM OPTIMIZED)
# ==============================================================================
DATASET_NAME = "cifar10"  # Supported HF Configs: ["cifar10", "cifar100"]
HF_REPO_ID = "Cuong2004/AA"
HF_TOKEN = os.getenv("HF_TOKEN", None)
RANDOM_SEED = 42

RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../result"))
LOG_DIR = os.path.join(RESULT_DIR, "logs")

BATCH_SIZE = 256  # High VRAM training batch size
NUM_WORKERS = 0
PIN_MEMORY = True

os.makedirs(LOG_DIR, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logger = logging.getLogger("DatasetLoader")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "dataset_loader.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================================
# HUGGING FACE PYTORCH DATASET WRAPPER
# ==============================================================================
class HFDatasetWrapper(Dataset):
    """Wraps Hugging Face Dataset into a PyTorch Dataset with torchvision transforms."""
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

def get_dataset_transforms(dataset_name="cifar10", is_train=False):
    """
    Returns dataset-specific PyTorch image transformations.
    Train split uses RandomCrop(32, padding=4) and RandomHorizontalFlip.
    Val/Test/Attack split uses ToTensor() without random augmentation.
    """
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

# ==============================================================================
# HUGGING FACE ONLY DATASET LOADER WITH STRATIFIED 40K/10K TRAIN/VAL SPLIT
# ==============================================================================
def get_dataloaders(dataset_name=DATASET_NAME, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, hf_token=HF_TOKEN, seed=RANDOM_SEED):
    """
    Downloads and loads full dataset splits directly from Hugging Face 'Cuong2004/AA'.
    Train split (50,000) is deterministically stratified into 40,000 train and 10,000 val.
    Test set (10,000) is held out strictly for evaluation & attack reporting.
    """
    ds_name = dataset_name.lower()
    logger.info(f"Loading full dataset '{ds_name}' from Hugging Face '{HF_REPO_ID}'...")

    try:
        hf_train_full = load_dataset(HF_REPO_ID, name=ds_name, split="train", token=hf_token)
        hf_test = load_dataset(HF_REPO_ID, name=ds_name, split="test", token=hf_token)
    except Exception as e:
        logger.error(f"Failed to load dataset '{ds_name}' from HF: {str(e)}")
        raise RuntimeError(f"Hugging Face dataset load failed: {str(e)}") from e

    # Stratified 40k train / 10k val split using fixed random seed
    all_indices = list(range(len(hf_train_full)))
    all_labels = hf_train_full["label"]
    train_idx, val_idx = train_test_split(
        all_indices, test_size=10000, random_state=seed, stratify=all_labels
    )

    pt_train_full = HFDatasetWrapper(hf_train_full, transform=get_dataset_transforms(ds_name, is_train=True))
    pt_val_full = HFDatasetWrapper(hf_train_full, transform=get_dataset_transforms(ds_name, is_train=False))
    pt_test = HFDatasetWrapper(hf_test, transform=get_dataset_transforms(ds_name, is_train=False))

    train_subset = Subset(pt_train_full, train_idx)
    val_subset = Subset(pt_val_full, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(pt_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    logger.info(f"Loaded '{ds_name}': Train={len(train_subset)}, Val={len(val_subset)}, Test={len(pt_test)}")
    return train_loader, val_loader, test_loader

def get_sample_batch(dataset_name=DATASET_NAME, batch_size=BATCH_SIZE, num_samples=1000, hf_token=HF_TOKEN):
    """Downloads sample batch directly from Hugging Face 'Cuong2004/AA' test set."""
    ds_name = dataset_name.lower()
    logger.info(f"Downloading sample batch for '{ds_name}' ({num_samples} samples) from Hugging Face '{HF_REPO_ID}'...")

    try:
        hf_ds = load_dataset(HF_REPO_ID, name=ds_name, split=f"test[:{num_samples}]", token=hf_token)
    except Exception as e:
        logger.error(f"Failed to download sample batch '{ds_name}' from Hugging Face: {str(e)}")
        raise RuntimeError(f"Hugging Face sample download failed: {str(e)}") from e

    pt_dataset = HFDatasetWrapper(hf_ds, transform=get_dataset_transforms(ds_name, is_train=False))
    loader = DataLoader(pt_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)

    logger.info(f"Successfully loaded '{ds_name}' DataLoader with {len(pt_dataset)} samples from Hugging Face!")
    return loader

if __name__ == "__main__":
    logger.info("=== Running Standalone Hugging Face Dataset Loader Test ===")
    tr_loader, val_loader, te_loader = get_dataloaders("cifar10", batch_size=128)
    imgs, lbls = next(iter(te_loader))
    logger.info(f"CIFAR-10 Test Batch Loaded - Shape: {imgs.shape}, Labels: {lbls.shape}")
