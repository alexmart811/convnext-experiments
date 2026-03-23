import os
import kagglehub
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from transforms import get_train_transforms, get_val_transforms

def download_and_get_paths():
    """
    Скачивает датасеты с Kaggle (если они еще не скачаны) 
    и возвращает пути к корневым папкам.
    """
    print("Проверка/загрузка Mini-ImageNet...")
    mini_imagenet_path = kagglehub.dataset_download("deeptrial/miniimagenet")
    
    print("Проверка/загрузка Flowers Dataset...")
    flowers_path = kagglehub.dataset_download("imsparsh/flowers-dataset")
    
    return mini_imagenet_path, flowers_path

def create_dataloaders(dataset_name, batch_size, input_size=224, num_workers=4):
    """
    Создает DataLoader'ы (train/val) для выбранного датасета.
    """
    mini_imagenet_path, flowers_path = download_and_get_paths()
    
    if dataset_name == "mini-imagenet":
        data_dir = os.path.join(mini_imagenet_path, "mini-imagenet") 
        if not os.path.exists(data_dir):
            data_dir = mini_imagenet_path
            
        full_dataset = datasets.ImageFolder(root=data_dir)
        
    elif dataset_name == "flowers":
        data_dir = os.path.join(flowers_path, "flowers")
        if not os.path.exists(data_dir):
            data_dir = flowers_path
            
        full_dataset = datasets.ImageFolder(root=data_dir)
        
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_name}. Выберите 'mini-imagenet' или 'flowers'.")

    # Разбиваем датасет на train (80%) и val (20%)
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

    num_classes = len(full_dataset.classes)
    
    print(f"Датасет {dataset_name} загружен. Классов: {num_classes}. Train: {train_size}, Val: {val_size}")
    
    return train_loader, val_loader, num_classes

if __name__ == "__main__":
    print("Тестирование загрузки Flowers Dataset...")
    train_dl, val_dl, classes = create_dataloaders("flowers", batch_size=32)
    print("Успешно!")
