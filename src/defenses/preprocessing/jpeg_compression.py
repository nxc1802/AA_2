import io
import torch
import torchvision.transforms as transforms
from PIL import Image

class JPEGCompressionDefense:
    """Applies JPEG compression simulation on batch tensors."""
    def __init__(self, quality: int = 75):
        self.quality = quality
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def defend(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        defended_list = []
        for i in range(images.size(0)):
            img_pil = self.to_pil(images[i].cpu())
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            img_compressed = Image.open(buffer)
            defended_list.append(self.to_tensor(img_compressed))
        
        defended = torch.stack(defended_list, dim=0).to(device)
        return torch.clamp(defended, 0.0, 1.0)
