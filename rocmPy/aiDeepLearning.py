"""
Plik: aiDeepLearning.py
Opis: Zawiera funkcję treningową do trenowania modeli sieci neuronowych (np. ResNet18) 
      w celu sterowania samochodem w środowisku FH4. 
      Obsługuje trening na GPU, użycie mieszanej precyzji (mixed precision), 
      scheduler dla tempa uczenia oraz zapisywanie checkpointów modelu.
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import subprocess
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

import LossFunc

def train_model(
    model_name: str,
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    criterion: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epochs: int,
    device: torch.device,
    save_dir: str = "checkpoints"
) -> dict:

    model.to(device)
    # Wagi do funkcji kosztu
    weights = torch.tensor([0.5, 0.25, 0.25]).to(device=device)
    history = {'train_loss': [], 'train_acc': []}
    
    # Tworzenie folderu na zapis (wykorzystanie pathlib)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    log_path = save_path / "training_log.txt"
    
    with open(log_path, "w") as log_file:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0
            stopwatch = time.perf_counter()
            
            for i, (batch_inputs, targets) in enumerate(dataloader):
                images_raw, speeds_batch = batch_inputs

                # Optymalizacja: Przenoszenie i normalizacja w jednym kroku (oszczędność VRAM i czasu)
                # Używamy float16 dla mieszanej precyzji
                images_batch = images_raw.to(device, dtype=torch.float16, non_blocking=True).permute(0,3,1,2).div_(255.0) 
                targets = targets.to(device, dtype=torch.float16, non_blocking=True)

                # Automatyczne mieszanie precyzji (AMP)
                with torch.autocast(device.type, enabled=True, dtype=torch.bfloat16):
                    outputs = model(images_batch, speeds_batch)
                    # Uwaga: Używamy wag w cross_entropy
                    loss = F.cross_entropy(outputs, targets, weight=weights) / 4
                
                loss.backward()
                
                # Optymalizacja: Krok optymalizatora co kilka kroków lub na końcu batcha (opcjonalne)
                if i % 3 == 0 or i == len(dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True) # Szybsze niż zero_grad()

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                
                if i % 50 == 0:
                    elapsed = time.perf_counter() - stopwatch
                    msg = f"Epoka [{epoch+1}/{epochs}] | Batch [{i+1}/{len(dataloader)}] | Loss: {loss.item():.4f} | Time: {elapsed:.4f}s"
                    print(msg)
                    log_file.write(msg + "\n")
                    log_file.flush()

            epoch_train_loss = running_loss / total_samples
            epoch_time = time.perf_counter() - stopwatch
            log_str = f"Epoka [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Time: {epoch_time:.2f}s"
            print(log_str)
            log_file.write(log_str + "\n")
            log_file.flush()

            # Mechanizm bezpieczeństwa: wyłącz komputer, jeśli loss przestanie spadać (stagnacja)
            if epoch > 3 and len(history["train_loss"]) >= 2:
                if abs(history["train_loss"][-1] - epoch_train_loss) < 1e-6:
                    print("Loss stagnates. Shutting down system to save energy/prevent damage...")
                    # subprocess.run(["shutdown", "/s", "/t", "600"]) # Odkomentuj przy realnym treningu
                    break
            
            scheduler.step()

            # Zapis modelu po każdej epoce w dedykowanym folderze
            model_filename = f"{model_name}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path / model_filename)
            print(f"Zapisano checkpoint: {model_filename}")

        return history

if __name__ == "__main__":
    import argparse
    
    # CLI Argument Parser - to sprawia, że Twój projekt jest profesjonalnym narzędziem
    parser = argparse.ArgumentParser(description="Professional Training Script for FH4 AI Driver")
    parser.add_argument("--model_name", type=str, default="OneImgModel_Base", help="Base name for the model file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--data_dir", type=str, default="D:/screenshots", help="Path to training images")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models and logs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    
    args = parser.parse_args()
    
    # Konfiguracja sprzętowa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multiprocessing.freeze_support()

    # Importy modeli (używane wewnątrz main)
    from AiModels.OneImg import OneImgModel
    from DataLoaders.Resnet18DL import Resnet18DL

    # 1. Inicjalizacja komponentów
    model = OneImgModel(num_speeds=15, rest_input_dim=48).to(device)
    criterion = LossFunc.CustomDriveLoss(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 2. Przygotowanie danych
    train_loader = DataLoader(
            Resnet18DL(args.data_dir, "AUDI_TT", 5),
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
    )

    # 3. Start treningu
    print(f"Rozpoczynanie treningu na urządzeniu: {device}")
    history = train_model(
        model_name=args.model_name,
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        save_dir=args.save_dir
    )

