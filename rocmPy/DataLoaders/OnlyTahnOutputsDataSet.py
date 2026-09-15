
from torch.utils.data import Dataset, DataLoader
import torch
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gc
import MenagePhotos.Screenshot as Screenshot
# class FolderCsv:
#     def __init__(self, folder):
class OnlyTanhLabelsDataSet(Dataset):
    def __init__(self, main_path=None, main_name_of_folder=None):
        self.main_path = main_path
        self.main_name_of_folder = main_name_of_folder
        
        self.data_folders = [f for f in os.listdir(main_path) if f.startswith(self.main_name_of_folder)]
        self.folders_csv_map = {}
        
        # Wstępne przetworzenie danych do czystych tablic NumPy (redukcja narzutu RAM)
        min_samples = float('inf')
        for folder in self.data_folders:
            csv_path = os.path.join(main_path, folder, "telemetry.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                # Wyciągamy czyste tablice NumPy
                names = df['screnshot_name'].values
                speeds = (df['Speed_kmh'].values.astype(np.float16) / 360.0).astype(np.float16)
                steering = df['Steering'].values.astype(np.float16)
                lt = df['LT'].values.astype(np.float16)
                rt = df['RT'].values.astype(np.float16)

                length = len(names)
                if length < min_samples:
                    min_samples = length
                    
                self.folders_csv_map[folder] = {
                    'names': names,
                    'speeds': speeds,
                    'targets': np.stack([steering,  np.where(lt > 0, -np.round(lt,4), np.round(rt,4)) ], axis=1) # [N, 3]
                }
        
        self.samples_per_folder = min_samples - 10
        self.total_samples = self.samples_per_folder * len(self.data_folders)
        self.get_item_index = 0  # Indeks do śledzenia, który element jest pobierany w __getitem__

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        folder_idx = idx // self.samples_per_folder
        image_idx = idx % self.samples_per_folder + 5

        current_folder = self.data_folders[folder_idx]
        folder_path = os.path.join(self.main_path, current_folder)
        data = self.folders_csv_map[current_folder]

        # Prealokacja bezpośrednio w formacie PyTorch (5, C, H, W) jako uint8
        list_of_images = np.zeros((5, 720, 1280, 3), dtype=np.uint8)
        
        for i in range(5):
            image_name = data['names'][image_idx + i]
            img = Screenshot.read_screenshot(folder_path, image_name)
            list_of_images[i] =  np.asarray(img, dtype=np.uint8)

        # Utworzenie tensorów (zero-copy z NumPy dla obrazów)
        torch_images = torch.from_numpy(list_of_images) # uint8, shape: [5, 3, 720, 1280]
        speeds = torch.from_numpy(data['speeds'][image_idx:image_idx + 5]) # float16
        label = torch.from_numpy(data['targets'][image_idx + 4]) # float16

        return (torch_images, speeds), label


if __name__ == "__main__":
    simple_dataset = OnlyTanhLabelsDataSet(r"C:\screenshots", "ASTON_MARTIN_FHEDITION")
    # pin_memory=True przyspiesza transfer z RAM do VRAM
    simple_dataloader = DataLoader(simple_dataset, batch_size=8, shuffle=True, pin_memory=True)

    for images, labels in simple_dataloader:
        print(f"Images shape: {images[0].shape}")  # [8, 5, 3, H, W]
        print(f"Speeds shape: {images[1].shape}")  # [8, 5]
        print(f"Labels shape: {labels.shape}")     # [8, 3]
        break