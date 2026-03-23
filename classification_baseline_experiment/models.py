import timm

def get_model(model_name: str, num_classes: int, pretrained: bool = False):
    """
    Возвращает модель по ее названию: 'convnext_tiny' или 'resnet50'
    """
    if model_name == 'convnext_tiny':
        model = timm.create_model(
            'convnext_tiny', 
            pretrained=pretrained, 
            num_classes=num_classes,
            drop_path_rate=0.1
        )
    elif model_name == 'resnet50':
        model = timm.create_model(
            'resnet50', 
            pretrained=pretrained, 
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
        
    return model
