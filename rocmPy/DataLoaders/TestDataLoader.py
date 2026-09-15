
from torch.utils.data import Dataset, DataLoader
import torch
import os
import re
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gc
import MenagePhotos.Screenshot as Screenshot
# class FolderCsv:
#     def __init__(self, folder):
class TestDataset(Dataset):
    def extract_number(text):
        match = re.search(r'\d+', str(text))
        return int(match.group()) if match else 0
    def __init__(self, main_path=None, main_name_of_folder=None):
        self.main_path = main_path
        self.main_name_of_folder = main_name_of_folder
        
        self.data_folders = [f for f in os.listdir(main_path) if f.startswith(self.main_name_of_folder)]
        self.folders_csv_map = {}
        self.folders_lenght_map = {}
        # Wstępne przetworzenie danych do czystych tablic NumPy (redukcja narzutu RAM)
        min_samples = 0
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
                numbers = np.array([TestDataset.extract_number(name) for name in names])
                sort_indices = np.argsort(numbers)
                min_samples += length
                self.folders_lenght_map[folder] = {"lenght":length }   
                self.folders_csv_map[folder] = {
                    'names': names[sort_indices],
                    'speeds': speeds[sort_indices],
                    'targets': np.stack([steering, rt ,lt ], axis=1)[sort_indices] # [N, 3]
                }
                
         
        self.total_samples = min_samples - min_samples // 100
        self.get_item_index = 0  # Indeks do śledzenia, który element jest pobierany w __getitem__

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        previous_folders_lenght = 0
        data,folder_path,current_folder = (0,0,0)
        for folder in self.data_folders:
            folder_path = os.path.join(self.main_path, folder)
            if previous_folders_lenght + self.folders_lenght_map[folder]['lenght']> idx:
                current_folder = folder
                data = self.folders_csv_map[current_folder]
                image_idx = idx - previous_folders_lenght 
                break
            previous_folders_lenght += self.folders_lenght_map[folder]['lenght']
        
        
        # Prealokacja bezpośrednio w formacie PyTorch (5, C, H, W) jako uint8
        list_of_images = np.zeros((5, 720, 1280, 3), dtype=np.uint8)
        if image_idx + 10 >= self.folders_lenght_map[folder]['lenght']:
            image_idx -= 10
        
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
    simple_dataset = TestDataset(r"C:\screenshots", "ASTON_MARTIN_FHEDITION")
    # pin_memory=True przyspiesza transfer z RAM do VRAM
    simple_dataloader = DataLoader(simple_dataset, batch_size=8, shuffle=True, pin_memory=True)

    for images, labels in simple_dataloader:
        print(f"Images shape: {images[0].shape}")  # [8, 5, 3, H, W]
        print(f"Speeds shape: {images[1].shape}")  # [8, 5]
        print(f"Labels shape: {labels.shape}")     # [8, 3]
        break
