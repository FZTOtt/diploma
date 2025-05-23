import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import torch.optim as optim
from attention.models.generator import Generator
from attention.models.extractor import Extractor
from attention.models.discriminator import Discriminator
from losses import total_generator_loss, discriminator_adversarial_loss, concealment_loss
from torch.utils.data import DataLoader
from attention.dataset import StegoDataset

def save_checkpoint(models, optimizers, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'G':   models['G'].state_dict(),
        'E':   models['E'].state_dict(),
        'D':   models['D'].state_dict(),
        'optG': optimizers['G'].state_dict(),
        'optD': optimizers['D'].state_dict(),
    }
    torch.save(state, path)

def train(
    data_root: str,
    batch_size: int = 8,
    pretrain_epochs=10, 
    full_epochs: int = 100,
    lr: float = 1e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
    device: str = 'cuda'
):
    
    dataset = StegoDataset(root=data_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    G = Generator().to(device)
    E = Extractor().to(device)
    D = Discriminator().to(device)

    optG = optim.Adam(
        list(G.parameters()) + list(E.parameters()),
        lr=lr, betas=(beta1, beta2)
    )
    optD = optim.Adam(
        D.parameters(),
        lr=lr, betas=(beta1, beta2)
    )
    optE = optim.Adam(
        E.parameters(), 
        lr=lr, betas=(beta1, beta2)
    )

    print(f"=== Stage 1: Pre-training Generator for {pretrain_epochs} epochs ===")
    G.train()
    for epoch in range(1, pretrain_epochs+1):
        running_lc = 0.0
        for i, (cover, _) in enumerate(loader, 1):
            cover = cover.to(device)
            secret = torch.randn_like(cover)
            stego = G(cover, secret)  
            lc = concealment_loss(cover, stego)

            optG.zero_grad()
            lc.backward()
            optG.step()

            running_lc += lc.item()
        print(f"[Epoch {epoch}/{pretrain_epochs}]  LC = {running_lc/len(loader):.4f}")

    print(f"\n=== Stage 2: Full training for {full_epochs} epochs ===")
    for epoch in range(1, full_epochs+1):
        running_g_loss = 0.0
        running_d_loss = 0.0

        for i, (cover, secret) in enumerate(loader, 1):
            cover, secret = cover.to(device), secret.to(device)

            # ———— 1) Дискриминатор ————
            # генерируем, но detach(), чтобы не обновлять G/E
            stego_det = G(cover, secret).detach()
            d_logits_real = D(cover, cover)
            d_logits_fake = D(cover, stego_det)
            d_loss = discriminator_adversarial_loss(d_logits_real, d_logits_fake)

            optD.zero_grad()
            d_loss.backward()
            optD.step()

            # ———— 2) Генератор + Extractor ————
            stego = G(cover, secret)
            extracted = E(stego)
            d_logits_fake_for_g = D(cover, stego)

            g_loss = total_generator_loss(
                cover, stego, secret, extracted, d_logits_fake_for_g
            )

            optG.zero_grad()
            optE.zero_grad()
            g_loss.backward()
            optG.step()
            optE.step()

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()

            if i % 50 == 0:
                print(f"[Epoch {epoch}/{full_epochs} | Batch {i}/{len(loader)}] "
                      f"G_loss = {running_g_loss/i:.4f}, "
                      f"D_loss = {running_d_loss/i:.4f}")

        ckpt = {
            'G': G.state_dict(),
            'E': E.state_dict(),
            'D': D.state_dict()
        }
        torch.save(ckpt, f"./checkpoints1/full_epoch_{epoch:03d}.pth")

    print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train AttentionGAN")
    parser.add_argument("--data_root", type=str, required=True, default='../dataset/DIV2K_train_HR',
                        help="путь к папке с изображеняими")
    parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        data_root=args.data_root,
        batch_size=args.batch_size,
        # epochs=args.epochs,
        lr=args.lr,
        device=device
    )