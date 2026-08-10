import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

HF_REPO_ID = "Cuong2004/AA"
HF_TOKEN = os.getenv("HF_TOKEN", None)


DATASET_SPECS = {
    "cifar10": {
        "num_classes": 10,
        "image_size": 32,
        "channels": 3,
    },
    "cifar100": {
        "num_classes": 100,
        "image_size": 32,
        "channels": 3,
    },
}


def get_dataset_spec(dataset_name: str = "cifar10") -> dict:
    """Returns dataset specifications dictionary given a dataset name."""
    ds_name = dataset_name.lower()
    if ds_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'. Supported: {list(DATASET_SPECS.keys())}")
    return DATASET_SPECS[ds_name]


class HFDatasetWrapper(Dataset):
    """Wraps Hugging Face Dataset into a PyTorch Dataset."""
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


def get_dataset_transforms(is_train: bool = False):
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor()
        ])


def seed_worker(worker_id):
    """Deterministic worker initialization function for PyTorch DataLoader."""
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(
    dataset_name: str = "cifar10",
    batch_size: Optional[int] = None,
    train_batch_size: int = 256,
    eval_batch_size: int = 512,
    num_workers: Optional[int] = None,
    seed: int = 42,
    hf_token: Optional[str] = HF_TOKEN,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads full dataset from Hugging Face 'Cuong2004/AA'.
    Train split (50,000) is deterministically stratified into 40,000 train and 10,000 val.
    Test set (10,000) is held out strictly for evaluation.
    """
    if batch_size is not None:
        train_batch_size = batch_size
        eval_batch_size = batch_size

    if num_workers is None:
        num_workers = min(os.cpu_count() or 2, 4)

    spec = get_dataset_spec(dataset_name)
    ds_name = dataset_name.lower()
    hf_train_full = load_dataset(HF_REPO_ID, name=ds_name, split="train", token=hf_token)
    hf_test = load_dataset(HF_REPO_ID, name=ds_name, split="test", token=hf_token)

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

    g = torch.Generator()
    g.manual_seed(seed)

    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_subset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g,
        pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
    )
    val_loader = DataLoader(
        val_subset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
    )
    test_loader = DataLoader(
        pt_test, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
    )

    return train_loader, val_loader, test_loader


def get_sample_batch_indices(
    dataset_name: str = "cifar10",
    batch_size: int = 512,
    num_samples: int = 1000,
    seed: int = 42,
    num_workers: Optional[int] = None,
    hf_token: Optional[str] = HF_TOKEN,
    pin_memory: Optional[bool] = None
) -> Tuple[DataLoader, list, str]:
    """Loads deterministic, class-stratified subset from Hugging Face test set for benchmark evaluation."""
    import hashlib
    ds_name = dataset_name.lower()
    hf_test = load_dataset(HF_REPO_ID, name=ds_name, split="test", token=hf_token)
    total_test = len(hf_test)

    if num_samples >= total_test:
        selected_indices = list(range(total_test))
    else:
        all_indices = list(range(total_test))
        all_labels = hf_test["label"]
        _, selected_indices = train_test_split(
            all_indices,
            test_size=num_samples,
            random_state=seed,
            stratify=all_labels
        )
        selected_indices = sorted(selected_indices)

    if num_workers is None:
        num_workers = min(os.cpu_count() or 2, 4) if torch.cuda.is_available() else 0

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers > 0 else None

    pt_dataset = HFDatasetWrapper(hf_test, transform=get_dataset_transforms(is_train=False))
    subset = Subset(pt_dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    indices_str = ",".join(map(str, selected_indices))
    indices_hash = hashlib.sha256(indices_str.encode("utf-8")).hexdigest()
    return loader, selected_indices, indices_hash


def get_sample_batch(
    dataset_name: str = "cifar10",
    batch_size: int = 512,
    num_samples: int = 1000,
    seed: int = 42,
    num_workers: Optional[int] = None,
    hf_token: Optional[str] = HF_TOKEN
) -> DataLoader:
    """Loads deterministic class-stratified subset from Hugging Face test set for benchmark evaluation."""
    loader, _, _ = get_sample_batch_indices(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_samples=num_samples,
        seed=seed,
        num_workers=num_workers,
        hf_token=hf_token
    )
    return loader


