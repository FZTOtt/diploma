import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from attention.models.generator import Generator
from attention.models.extractor import Extractor

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    G = Generator().to(device)
    E = Extractor().to(device)
    G.load_state_dict(ckpt['G'])
    E.load_state_dict(ckpt['E'])
    G.eval()
    E.eval()
    return G, E

def preprocess(img_path, device):
    """Загружает и нормализует в [-1,1]"""
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    img = Image.open(img_path).convert('RGB')
    return tf(img).unsqueeze(0).to(device)  # [1,3,256,256]

def deprocess(tensor):
    """Из [-1,1] в [0,1] и в numpy"""
    t = tensor.detach().cpu().clamp(-1,1)
    t = (t + 1) / 2
    return t.squeeze(0).permute(1,2,0).numpy()

def visualize_pair(G, E, cover_path, secret_path, device):
    cover = preprocess(cover_path, device)
    secret = preprocess(secret_path, device)

    with torch.no_grad():
        stego = G(cover, secret)
        extracted = E(stego)

    images = [
        ("Cover",    deprocess(cover)),
        ("Stego",    deprocess(stego)),
        ("Extracted",deprocess(extracted)),
        ("Secret",   deprocess(secret)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    for ax, (title, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="путь к файлу epoch_XXX.pth", default='../checkpoints/full_epoch_001.pth')
    parser.add_argument("--cover", type=str, required=True,
                        help="путь к изображению cover", default='../../dataset/DIV2K_train_HR/0001.png')
    parser.add_argument("--secret", type=str, required=True,
                        help="путь к изображению secret", default='../../dataset/DIV2K_train_HR/0002.png')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, E = load_checkpoint(args.checkpoint, device)
    visualize_pair(G, E, args.cover, args.secret, device)

# python .\visualisation.py --checkpoint ../checkpoints/full_epoch_001.pth --cover ../../dataset/DIV2K_train_HR/0001.png --secret ../../dataset/DIV2K_train_HR/0002.png