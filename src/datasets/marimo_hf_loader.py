# ==============================================================================
# MARIMO CODE SNIPPET: FAST DATASET DOWNLOAD FROM HUGGING FACE (Cuong2004/AA)
# ==============================================================================
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class HFDatasetWrapper(Dataset):
    """Wraps Hugging Face Dataset into a PyTorch Dataset with torchvision transforms."""
    def __init__(self, hf_ds, transform=None):
        self.hf_ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_marimo_dataset(config_name="cifar10", batch_size=256, num_samples=1500):
    """
    Downloads dataset subset (1000-2000 samples) directly from Hugging Face 'Cuong2004/AA'.
    Supported config_name: ['cifar10', 'cifar100', 'mnist', 'fashion_mnist']
    """
    print(f"Loading '{config_name}' subset ({num_samples} samples) from Hugging Face 'Cuong2004/AA'...")
    ds = load_dataset("Cuong2004/AA", name=config_name, split=f"test[:{num_samples}]")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    pt_dataset = HFDatasetWrapper(ds, transform=transform)
    loader = DataLoader(pt_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Successfully loaded '{config_name}' DataLoader with {len(pt_dataset)} samples!")
    return loader

# ==============================================================================
# EXAMPLE USAGE FOR MARIMO NOTEBOOK CELL
# ==============================================================================
if __name__ == "__main__":
    # Test loading CIFAR-10 subset from Hugging Face
    loader = get_marimo_dataset(config_name="cifar10", batch_size=256, num_samples=1500)
    images, labels = next(iter(loader))
    print(f"Loaded batch tensor shape: {images.shape}, Labels shape: {labels.shape}")
