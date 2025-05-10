import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from utils.chaos_permutation import ChaosPermutation
from utils.helps import prepare_inputs
import os
import random

class SteganographyDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.image_pairs = self._load_pairs()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.chaos = ChaosPermutation()

    def _load_pairs(self):
        pairs = []
        for fname in os.listdir(os.path.join(self.root, "cover")):
            if fname.endswith(".jpg"):
                cover_path = os.path.join(self.root, "cover", fname)
                secret_path = os.path.join(self.root, "secret", fname)
                if os.path.exists(secret_path):
                    pairs.append((cover_path, secret_path))
        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        cover_path, secret_path = self.image_pairs[idx]
        
        cover = Image.open(cover_path).convert("RGB")
        secret = Image.open(secret_path).convert("RGB")
        
        cover_tensor = self.transform(cover)
        secret_tensor = self.transform(secret)
        
        cover_y, secret_scrambled = prepare_inputs(
            cover_tensor, 
            secret_tensor,
            self.chaos
        )
        
        return cover_y.squeeze(0), secret_scrambled.squeeze(0)
    
class DIV2KDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        """
        root_dir: Путь к папке dataset
        mode: "train" или "valid" для выбора папки
        """
        self.root = root_dir
        self.mode = mode
        self.image_dir = os.path.join(self.root, f"DIV2K_{mode}_HR")
        self.image_list = [os.path.join(self.image_dir, fname) 
                           for fname in os.listdir(self.image_dir) 
                           if fname.endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor()       
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Случайное изображение как контейнер
        cover_path = self.image_list[idx]
        # Случайное изображение как секрет
        secret_path = random.choice(self.image_list)

        cover = Image.open(cover_path).convert("RGB")
        secret = Image.open(secret_path).convert("RGB")

        cover_tensor = self.transform(cover)
        secret_tensor = self.transform(secret)

        return cover_tensor, secret_tensor