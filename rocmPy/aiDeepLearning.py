import os
import time
import torch
import torch.nn as nn
import multiprocessing
import subprocess
from torch.utils.data import DataLoader


import LossFunc
# os.environ["MIOPEN_CONV_PREFER_EXEC_TIME"] = "0"
# os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "1"
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
def train_model(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    criterion: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epochs: int,
    device: torch.device
) -> dict:

    model.to(device)
    history = {'train_loss': [], 'train_acc': []}
    model_number = 1
    file = open("log.txt","w")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        stopwatch = time.perf_counter()
        for i, (batch_inputs, targets) in enumerate(dataloader):
            images_raw, speeds_batch = batch_inputs

            # Przenosimy na GPU i normalizujemy w jednym kroku (oszczędność VRAM)
            images_batch = images_raw.to(device, dtype=torch.float16, non_blocking=True).permute(0, 4, 1, 2, 3).div_(255.0)
            speeds_batch = speeds_batch.to(device, dtype=torch.float16, non_blocking=True)
            targets = targets.to(device, dtype=torch.float16, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # Zwalnia pamięć po gradientach zamiast zerować
            with torch.autocast(device.type, enabled=True, dtype=torch.float16):  # Włączamy automatyczne mieszanie precyzji
                outputs = model(images_batch, speeds_batch)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            if i % 10 == 0:
                file.write(f"Epoka [{epoch+1}/{epochs}] | Batch [{i+1}/{len(dataloader)}] | Loss: {loss.item():.4f}\n")
                print(f"Epoka [{epoch+1}/{epochs}] | Batch [{i+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / total_samples
        history['train_loss'].append(epoch_train_loss)
        epoch_time = time.perf_counter() - stopwatch
        log_str = f"Epoka [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Time: {epoch_time:.2f}s"
        print(log_str)
        file.write(log_str + "\n")
        file.flush()
        if epoch > 3:
            if history["train_loss"][epoch - 1] == history["train_loss"][epoch] and history["train_loss"][epoch - 2] == history["train_loss"][epoch]:
                subprocess.run(["shutdown", "/s", "/t", "600"])
                break
        scheduler.step()
        if (epoch + 1) % 2 == 0:
            model_path = f"model_{epochs}E{dataloader.batch_size}N{model_number}AUDI_TT_With_Goliath.pth"
            torch.save(model.state_dict(), model_path )
            print(f"Zapisano model do {model_path}")
            model_number += 1
            

        # # Czyszczenie bufora alokatora CUDA po każdej epoce
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return history
if __name__ == "__main__":
    from AiModels.OnlyTanhModel import OnlyTanhModel
    from AiModels.FirstModel import ForzaH4Model
    from AiModels.SecondModel import SecondFH4Model

    from DataLoaders.DataLoader import SimpleDataset
    from DataLoaders.OnlyTahnOutputsDataSet import OnlyTanhLabelsDataSet
    from DataLoaders.TestDataLoader import TestDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multiprocessing.freeze_support()
    # 1. Inicjalizacja komponentów
    model = SecondFH4Model().to(device)
    # checkpoint = torch.load(r"C:\FH4_AI_Driver\FirstModels\model_200E32BLR0_0000111.pth", map_location=device)
        
    #     # Wczytujemy stan modelup
    # model.load_state_dict(checkpoint)
    criterion = LossFunc.CustomDriveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_loader = DataLoader(
        TestDataset("D:\\screenshots", "AUDI_Front_TT"),
          batch_size=16, 
          shuffle=True,
          num_workers=12,
          pin_memory=True,
          persistent_workers=False,
          prefetch_factor=2)
    # 2. Uruchomienie pętli na N epok
    history = train_model(
        model=model,
        dataloader=train_loader,      # DataLoader zwracający ((x1, x2), y) lub ({'x1': x1, 'x2': x2}, y)
        criterion=criterion,
        optimizer=optimizer,
        epochs=200,
        device=device
    )
