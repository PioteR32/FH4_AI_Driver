import bettercam
import torch
import torch.nn.functional as F
import time

device = torch.device('cuda')

# 1. Wskazujemy output_color="BGR" (3 kanały zamiast 4)
camera = bettercam.create(output_idx=0, output_color="BGR")
camera.start(target_fps=60, video_mode=True)

# Twój docelowy rozmiar dla modelu (np. 360x640 lub 720x1280)
target_size = (360, 640)  # (Height, Width)

# Bufor na 5 klatek 3-kanałowych [5, Height_2K, Width_2K, 3]
raw_frames_gpu = torch.empty((5, camera.height, camera.width, 3), dtype=torch.uint8, device=device)

# Główny tenzor [Batch=1, Channels=3, Depth=5, Height=360, Width=640]
input_tensor = torch.empty((1, 3, 5, target_size[0], target_size[1]), dtype=torch.float32, device=device)

print("Pętla gotowa na 3 kanały (BGR)...")

while True:
    loop_start = time.perf_counter()

    # KROK 1: Grab 3-kanałowej klatki BGR (uint8)
    img_np = camera.get_latest_frame() 

    if img_np is None: 
        continue

    # KROK 2: Przesunięcie i wgranie do VRAM (o 25% mniej bajtów po PCIe)
    raw_frames_gpu = torch.roll(raw_frames_gpu, shifts=-1, dims=0)
    raw_frames_gpu[-1] = torch.from_numpy(img_np)

    # KROK 3: Operacje na GPU
    # [5, H, W, 3] -> konwersja do float i div_(255.0)
    gpu_processed = raw_frames_gpu.float().div_(255.0)
    
    # Przestawienie pod Conv3D: [5, H, W, 3] -> [1, 3, 5, H, W]
    gpu_processed = gpu_processed.permute(3, 0, 1, 2).unsqueeze(0)

    # Skalowanie w VRAM
    resized = F.interpolate(
        gpu_processed, 
        size=(5, target_size[0], target_size[1]), 
        mode='nearest', 
    )
    
    input_tensor.copy_(resized)

    iteration_time_ms = (time.perf_counter() - loop_start) * 1000
    print(f"Czas iteracji (3 channels): {iteration_time_ms:.2f} ms")
    