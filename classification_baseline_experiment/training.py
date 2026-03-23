import torch
import torch.nn as nn
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm

def get_mixup_fn(num_classes):
    """
    Инициализация Mixup и Cutmix согласно рецепту обучения из статьи ConvNeXt
    """
    return Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=0.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.0,
        num_classes=num_classes
    )

def train_one_epoch(model, dataloader, optimizer, device, epoch, mixup_fn=None):
    model.train()
    
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate(model, dataloader, device, epoch):
    """
    Добавлен 4-й аргумент 'epoch' для отображения в прогресс-баре
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}", leave=False)
    
    with torch.no_grad():
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    epoch_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return epoch_loss, accuracy
