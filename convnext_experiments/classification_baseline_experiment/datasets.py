import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import kagglehub
from transforms import get_train_transforms, get_val_transforms

def download_and_get_paths():
    print("Проверка/загрузка Mini-ImageNet...")
    mini_imagenet_path = kagglehub.dataset_download("deeptrial/miniimagenet")
    
    print("Проверка/загрузка Flowers Dataset...")
    flowers_path = kagglehub.dataset_download("imsparsh/flowers-dataset")
    
    return mini_imagenet_path, flowers_path

def create_dataloaders(dataset_name, batch_size, input_size=224, num_workers=4):
    mini_imagenet_path, flowers_path = download_and_get_paths()
    
    if dataset_name == "mini-imagenet":
        data_dir = os.path.join(mini_imagenet_path, "ImageNet-Mini", "images")
        
        full_dataset = datasets.ImageFolder(root=data_dir)
        num_classes = len(full_dataset.classes)
        
    elif dataset_name == "flowers":
        data_dir = os.path.join(flowers_path, "flowers")
        if not os.path.exists(data_dir):
            data_dir = flowers_path
        full_dataset = datasets.ImageFolder(root=data_dir)
        num_classes = len(full_dataset.classes)
        
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_name}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = get_train_transforms(input_size)
    val_dataset.dataset.transform = get_val_transforms(input_size)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Датасет {dataset_name} загружен. Классов: {num_classes}. Train: {train_size}, Val: {val_size}")
    return train_loader, val_loader, num_classes

