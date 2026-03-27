from timm.data import create_transform
from torchvision import transforms

def get_train_transforms(input_size=224):
    """
    Классические легкие аугментации, идеальные для обучения ResNet с нуля на малых данных.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(),                         
        transforms.ColorJitter(brightness=0.2, contrast=0.2),       
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(input_size=224):
    """
    Трансформации для валидации (без аугментаций).
    """
    crop_pct = 224 / 256
    scale_size = int(input_size / crop_pct)
    
    return transforms.Compose([
        transforms.Resize(scale_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])   
    ])
