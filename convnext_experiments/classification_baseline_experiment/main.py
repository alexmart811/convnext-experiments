import torch
import time
import datetime
import os  
from config import ConvNeXtConfig
from datasets import create_dataloaders
from models import get_model
from training import train_one_epoch, validate, get_mixup_fn

def main():
    config = ConvNeXtConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(current_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    dataset_name = "flowers" 
    print(f"Загрузка датасета {dataset_name}...")
    train_loader, val_loader, num_classes = create_dataloaders(
        dataset_name=dataset_name, 
        batch_size=config.batch_size, 
        input_size=config.input_size
    )

    mixup_fn = get_mixup_fn(num_classes=num_classes)
    
    models_to_test = ['resnet50', 'convnext_tiny']
    results = {}

    for model_name in models_to_test:
        print(f"\n" + "="*40)
        print(f" ЗАПУСК ЭКСПЕРИМЕНТА: {model_name.upper()}")
        print("="*40)
        
        model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

        best_acc = 0.0
        total_time_elapsed = 0.0
        
        for epoch in range(1, config.epochs + 1):
            start_time = time.time()
            
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, mixup_fn)
            scheduler.step()
            val_loss, val_acc = validate(model, val_loader, device, epoch)
            
            epoch_time = time.time() - start_time
            total_time_elapsed += epoch_time
            
            print(f"Эпоха [{epoch}/{config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Время: {epoch_time:.0f}с")
            
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(runs_dir, f"best_{model_name}.pth")
                torch.save(model.state_dict(), save_path)

        results[model_name] = best_acc
        print(f"Обучение {model_name} завершено. Лучшая точность: {best_acc:.2f}%")

    print("\n" + "="*40)
    print(" ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ (10 ЭПОХ)")
    print("="*40)
    for model_name, acc in results.items():
        print(f"{model_name.ljust(15)} : {acc:.2f}% Accuracy")

if __name__ == "__main__":
    main()
