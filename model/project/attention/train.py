import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import torch.nn as nn

import torch.optim as optim
from attention.models.generator import Generator, ImprovedGenerator
from attention.models.extractor import Extractor, ImprovedExtractor
from attention.models.discriminator import Discriminator, MultiScaleDiscriminator
from losses import total_generator_loss, discriminator_adversarial_loss, concealment_loss
from torch.utils.data import DataLoader, random_split
from attention.dataset import StegoDataset
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm

def ssim(img1, img2, window_size=11, size_average=True, data_range=1.0):
    # Реализация SSIM на PyTorch
    from torchvision.ops import gaussian_blur
    
    channels = img1.size(1)
    window = torch.ones((channels, 1, window_size, window_size)) / (window_size ** 2)
    window = window.to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

def compute_gradient_penalty(D, real_cover, real_stego, fake_stego):
    """Улучшенный расчет градиентного штрафа с учетом парных входов"""
    alpha = torch.rand(real_cover.size(0), 1, 1, 1, device=real_cover.device)
    
    # Интерполяция между реальными и фейковыми стего
    interpolates_stego = (alpha * real_stego + (1 - alpha) * fake_stego).requires_grad_(True)
    
    # Интерполяция для cover изображений (если требуется)
    interpolates_cover = real_cover  # Можно добавить шум при необходимости
    
    # Forward pass через дискриминатор
    d_interpolates = D(interpolates_cover, interpolates_stego)
    
    # Расчет градиентов
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates_stego,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Градиентный штраф с учетом пространственной структуры
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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
    # Инициализация AMP
    scaler = GradScaler()
    
    # Загрузка данных
    full_dataset = StegoDataset(root=data_root)
    train_size = int(0.9 * len(full_dataset))
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, len(full_dataset) - train_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)

    # Инициализация моделей с spectral normalization
    class DiscriminatorWithSN(ImprovedGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Применяем SN к каждому слою
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.utils.spectral_norm(layer)
    
    D = DiscriminatorWithSN().to(device)

    # G = Generator().to(device)
    # E = Extractor().to(device)
    G = ImprovedGenerator().to(device)
    E = ImprovedExtractor().to(device)
    # D = Discriminator().to(device)

    def compute_metrics(cover, stego, secret, extracted):
        with torch.no_grad():
            mse = F.mse_loss(cover, stego)
            psnr = 10 * torch.log10(1.0 / mse)
            ssim_val = ssim(cover, stego, data_range=1.0)
            secret_acc = 1 - F.l1_loss(secret, extracted)
            return psnr, ssim_val, secret_acc
        
    def total_generator_loss(cover, stego, secret, extracted, d_fake_g, 
                            lambda_conceal=0.4, lambda_adv=0.4, lambda_ssim=0.2):
        # Потеря незаметности
        l_conceal = F.l1_loss(cover, stego)
        
        # Adversarial loss
        l_adv = F.mse_loss(d_fake_g, torch.ones_like(d_fake_g))
        
        # Потеря восстановления секрета
        l_secret = F.mse_loss(secret, extracted)
        
        # SSIM loss
        l_ssim = 1 - ssim(cover, stego, data_range=1.0)
        
        # Multi-scale perceptual loss
        # (Добавьте VGG16 если возможно)
        
        return (lambda_conceal * l_conceal +
                lambda_adv * l_adv +
                lambda_ssim * l_ssim +
                0.2 * l_secret)
    
    # Инициализация весов
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    G.apply(weights_init)
    E.apply(weights_init)
    D.apply(weights_init)

    # Оптимизаторы с разными LR
    optG = optim.AdamW(
        list(G.parameters()) + list(E.parameters()),
        lr=lr, 
        betas=(beta1, beta2),
        weight_decay=1e-4
    )
    optD = optim.AdamW(
        D.parameters(),
        lr=lr*0.5,  # Меньший LR для D
        betas=(beta1, beta2),
        weight_decay=1e-4
    )
    
    # Schedulers
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=full_epochs)
    schedulerD = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=full_epochs)
    
    print("=== Stage 1: Generator Pre-training ===")
    for epoch in range(pretrain_epochs):
        G.train()
        # total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{pretrain_epochs}")
        
        for cover, _ in train_loader:
            cover = cover.to(device)
            secret = torch.randn_like(cover)  # Генерация случайного секрета
            
            # Контекст autocast с новым API
            with autocast(device_type='cuda'):
                stego = G(cover, secret)
                # loss = concealment_loss(cover, stego) * 0.7
                loss = F.l1_loss(cover, stego) * 0.7 + (1 - ssim(cover, stego)) * 0.3

            optG.zero_grad()
            scaler.scale(loss).backward()
            clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(optG)
            scaler.update()
            
            # total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        # print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # Основное обучение
    best_metric = float('inf')
    best_psnr = 0.0
    for epoch in range(full_epochs):
        G.train()
        E.train()
        D.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{full_epochs}")

        for i, (cover, secret) in enumerate(train_loader):
            cover = cover.to(device, non_blocking=True)
            secret = secret.to(device, non_blocking=True)

            # Обновление D
            if i % 2 == 0:
                with autocast(device_type='cuda'):
                    stego = G(cover, secret)
                    
                    # Real data (cover + slightly noisy cover)
                    real_stego = cover + torch.randn_like(cover) * 0.01
                    d_real = D(cover, real_stego)
                    d_fake = D(cover, stego.detach())
                    
                    gp = compute_gradient_penalty(D, cover, real_stego, stego.detach())
                    d_loss = discriminator_adversarial_loss(d_real, d_fake) + 10.0 * gp

            # Backward D
            optD.zero_grad()
            scaler.scale(d_loss).backward()
            clip_grad_norm_(D.parameters(), 1.0)
            scaler.step(optD)
            scaler.update()

            # Обновление G дважды реже
            if i % 2 == 0 or True:
                with autocast(device_type='cuda'):
                    stego = G(cover, secret)
                    extracted = E(stego)
                    d_fake_g = D(cover, stego)
                    
                    # Сбалансированные коэффициенты
                    g_loss = total_generator_loss(
                        cover, stego, secret, extracted, d_fake_g,
                        lambda_conceal=0.3,  # Уменьшенный вес
                        lambda_adv=0.5       # Увеличенный вес
                    )

                # Backward G
                optG.zero_grad()
                scaler.scale(g_loss).backward()
                clip_grad_norm_(G.parameters(), 1.0)
                clip_grad_norm_(E.parameters(), 1.0)
                scaler.step(optG)
                scaler.update()

                train_g_loss += g_loss.item()

            # Логирование
            train_d_loss += d_loss.item()
            progress.set_postfix(g_loss=train_g_loss/(i+1), d_loss=train_d_loss/(i+1))


        # Валидация
        G.eval()
        E.eval()
        D.eval()
        val_metrics = {'psnr': 0.0, 'ssim': 0.0, 'secret_acc': 0.0}
        
        val_g_loss = 0.0
        val_d_loss = 0.0
        
        with torch.no_grad():
            for cover, secret in val_loader:
                cover, secret = cover.to(device), secret.to(device)
                
                stego = G(cover, secret)
                extracted = E(stego)
                
                # Calculate metrics
                mse = F.mse_loss(cover, stego)
                val_metrics['psnr'] += 10 * torch.log10(1.0 / mse).item()
                val_metrics['ssim'] += ssim(cover, stego).item()
                val_metrics['secret_acc'] += (1 - F.l1_loss(secret, extracted)).item()

        # Average metrics
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'G': G.state_dict(),
                'E': E.state_dict(),
                'D': D.state_dict(),
                'optG': optG.state_dict(),
                'optD': optD.state_dict(),
                'epoch': epoch,
                'metrics': val_metrics
            }, "best_model.pth")

        # Update schedulers
        schedulerG.step()
        schedulerD.step()

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train G Loss: {train_g_loss/len(train_loader):.4f} | D Loss: {train_d_loss/len(train_loader):.4f}")
        print(f"Val PSNR: {val_metrics['psnr']:.2f} | SSIM: {val_metrics['ssim']:.3f} | Secret ACC: {val_metrics['secret_acc']:.3f}\n")

if __name__ == "__main__":
    train(
        data_root="../dataset/DIV2K_train_HR",
        batch_size=8,
        pretrain_epochs=10,
        full_epochs=20,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )