import os
import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class StegoDataset(Dataset):
    """
    Dataset, отдающий пару (cover, secret), оба – [3×256×256] в [-1,1].
    """

    def __init__(self, root: str):
        """
        :param root: папка, где лежат изображения DIV2K (или любой другой датасет)
        """
        self.paths = sorted(
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        )
        assert len(self.paths) >= 2, "Нужно минимум 2 картинки в папке"

        self.transform = transforms.Compose([
            transforms.Resize(256),            # приведём к 256×256
            transforms.CenterCrop(256),        # на всякий случай
            transforms.ToTensor(),             # [0,1]
            transforms.Normalize((0.5,)*3, (0.5,)*3),  # →[-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # cover
        c = Image.open(self.paths[idx]).convert('RGB')
        cover = self.transform(c)

        # secret — случайная другая картинка
        j = random.randrange(0, len(self.paths) - 1)
        if j >= idx:
            j += 1
        s = Image.open(self.paths[j]).convert('RGB')
        secret = self.transform(s)

        return cover, secret