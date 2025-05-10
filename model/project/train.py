from data.dataset import DIV2KDataset
from models.chase import CHASE
from utils.helps import rgb_to_ycrcb
from utils.losses import hiding_loss, reconstruction_loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import config

# Загрузка данных
train_dataset = DIV2KDataset(config["data_root"], mode="train")
valid_dataset = DIV2KDataset(config["data_root"], mode="valid")

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model = CHASE().to(device)
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Обучение
for epoch in range(config["num_epochs"]):
    model.train()
    total_loss = 0

    for batch_idx, (cover, secret) in enumerate(train_loader):
        cover, secret = cover.to(device), secret.to(device)

        assert not torch.isnan(cover).any(), "NaN in cover image"
        assert not torch.isnan(secret).any(), "NaN in secret image"
        # Прямой проход
        stego, r, idx = model.forward_hide(cover, secret)

        # Обратный проход
        cover_recon, secret_recon = model.forward_reveal(stego, r, idx)

        # assert not torch.isnan(cover_recon).any(), "NaN in cover cover_recon"
        assert not torch.isnan(secret_recon).any(), "NaN in secret secret_recon"
        # Расчет потерь
        loss_hid = hiding_loss(stego, rgb_to_ycrcb(cover)[:, :1])
        loss_rec = reconstruction_loss(secret, secret_recon)
        print(loss_hid, loss_rec)
        loss = 8 * loss_hid + 1 * loss_rec

        # Оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Логирование
        if batch_idx % config["log_interval"] == 0:
            print(f"Epoch: {epoch+1}/{config['num_epochs']} | "
                  f"Batch: {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed | Avg Loss: {total_loss / len(train_loader):.4f}")

    # Сохранение модели
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"{config['checkpoint_dir']}/model_epoch_{epoch+1}.pth")

# Валидация
model.eval()
val_loss = 0
with torch.no_grad():
    for cover, secret in valid_loader:
        cover, secret = cover.to(device), secret.to(device)

        stego, r = model.forward_hide(cover, secret)
        z = torch.randn_like(r)
        cover_recon, secret_recon = model.forward_reveal(stego, z)

        loss_hid = hiding_loss(cover, stego)
        loss_rec = reconstruction_loss(secret, secret_recon)
        val_loss += (8 * loss_hid + 1 * loss_rec).item()

print(f"Validation Loss: {val_loss / len(valid_loader):.4f}")