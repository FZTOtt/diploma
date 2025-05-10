import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_imagenet_loaders(data_root, batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Изменение размера
        transforms.RandomHorizontalFlip(),  # Аугментация
        transforms.RandomRotation(10),  # Аугментация
        transforms.ToTensor()  # Преобразование в тензор
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader