import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import torch.nn as nn

import torch.optim as optim
from attention.models.generator import Generator
from attention.models.extractor import Extractor
from attention.models.discriminator import Discriminator
from losses import total_generator_loss, discriminator_adversarial_loss, concealment_loss
from torch.utils.data import DataLoader, random_split
from attention.dataset import StegoDataset
from torch.nn.utils import clip_grad_norm_
# import torch.cuda.amp as amp
from torch.amp import GradScaler, autocast

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
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
    G = Generator().to(device)
    E = Extractor().to(device)
    D = Discriminator().to(device)
    
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

    # Gradient Penalty
    def compute_gradient_penalty(D, real_cover, real_stego, fake_stego):
        """
        real_cover: исходное изображение (cover) [B, C, H, W]
        real_stego: реальное стего (например, исходное изображение) [B, C, H, W]
        fake_stego: сгенерированное стего [B, C, H, W]
        """
        # Создаем интерполированные стего
        alpha = torch.rand(real_cover.size(0), 1, 1, 1, device=real_cover.device)
        interpolates_stego = (alpha * real_stego + (1 - alpha) * fake_stego).requires_grad_(True)
        
        # Передаем в дискриминатор пару (cover, интерполированное стего)
        d_interpolates = D(real_cover, interpolates_stego)
        
        # Вычисляем градиенты
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates_stego,  # Градиенты относительно интерполированного стего
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Вычисляем градиентный штраф
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    print("=== Stage 1: Generator Pre-training ===")
    for epoch in range(pretrain_epochs):
        G.train()
        total_loss = 0.0
        
        for cover, _ in train_loader:
            cover = cover.to(device)
            secret = torch.randn_like(cover)  # Генерация случайного секрета
            
            # Контекст autocast с новым API
            with autocast(device_type='cuda'):
                stego = G(cover, secret)
                loss = concealment_loss(cover, stego) * 0.7

            optG.zero_grad()
            scaler.scale(loss).backward()
            clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(optG)
            scaler.update()
            
            total_loss += loss.item()

        print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # Основное обучение
    best_metric = float('inf')
    for epoch in range(full_epochs):
        G.train()
        E.train()
        D.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0

        for i, (cover, secret) in enumerate(train_loader):
            cover = cover.to(device, non_blocking=True)
            secret = secret.to(device, non_blocking=True)

            # Обновление D
            with autocast(device_type='cuda'):
                stego = G(cover, secret)
    
                # Для реальных данных: cover + cover (если real_stego == cover)
                d_real = D(cover, cover)
                
                # Для фейковых данных: cover + сгенерированное стего
                d_fake = D(cover, stego.detach())
                
                # Вычисляем градиентный штраф
                gp = compute_gradient_penalty(D, cover, cover, stego.detach())
                
                d_loss = discriminator_adversarial_loss(d_real, d_fake) + 10.0 * gp

            # Backward D
            optD.zero_grad()
            scaler.scale(d_loss).backward()
            clip_grad_norm_(D.parameters(), 1.0)
            scaler.step(optD)
            scaler.update()

            # Обновление G дважды реже
            if i % 2 == 0:
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

            # Логирование
            total_g_loss += g_loss.item() if i % 2 == 0 else 0
            total_d_loss += d_loss.item()

        # Обновление schedulers
        schedulerG.step()
        schedulerD.step()

        # Валидация
        G.eval()
        E.eval()
        D.eval()
        
        val_g_loss = 0.0
        val_d_loss = 0.0
        
        with torch.no_grad():
            for cover, secret in val_loader:
                cover, secret = cover.to(device), secret.to(device)
                
                # Для дискриминатора
                stego = G(cover, secret)
                d_real = D(cover, cover)
                d_fake = D(cover, stego)
                val_d_loss += discriminator_adversarial_loss(d_real, d_fake).item()
                
                # Для генератора
                extracted = E(stego)
                d_fake_g = D(cover, stego)
                val_g_loss += total_generator_loss(
                    cover, stego, secret, extracted, d_fake_g
                ).item()

        # Сохранение лучшей модели
        if val_g_loss < best_metric:
            best_metric = val_g_loss
            torch.save({
                'G': G.state_dict(),
                'E': E.state_dict(),
                'D': D.state_dict(),
                'optG': optG.state_dict(),
                'optD': optD.state_dict(),
            }, f"checkpoints/best_model.pth")

        print(f"Epoch {epoch+1}/{full_epochs} | "
              f"G Loss: {val_g_loss/len(val_loader):.4f} | "
              f"D Loss: {val_d_loss/len(val_loader):.4f}")

    print("Training completed successfully!")

if __name__ == "__main__":
    train(
        data_root="../dataset/DIV2K_train_HR",
        batch_size=8,
        pretrain_epochs=50,
        full_epochs=100,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )