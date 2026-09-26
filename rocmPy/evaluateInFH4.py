"""
Plik: evaluateInFH4.py
Opis: Skrypt do ewaluacji modelu w czasie rzeczywistym podczas rozgrywki w Forza Horizon 4.
      Pobiera dane z kamery i telemetrii, przetwarza je i wysyła sterowania (control signals)
      do emulatora kontrolera Xbox w celu przetestowania modelu w środowisku gry.
"""
import bettercam
import time
import torch
import torch.nn.functional as F
import keyboard
import math
from InputHandlers import XboxControllerEmulator
from FH4Connect import fh4Statistic
from AiModels import FirstModel
from AiModels import SecondModel
def evaluate_model_in_FH4(
    model: torch.nn.Module, 
    device: torch.device,
) -> dict:

    model.eval()  # Set model to evaluation mode
    listner = fh4Statistic.FH4TelemetryListener()
    controller = XboxControllerEmulator()
    device = torch.device('cuda')
# 1. Wskazujemy output_color="BGR" (3 kanały zamiast 4)
    camera = bettercam.create(output_idx=0, output_color="BGR")
    camera.start(target_fps=60, video_mode=True)

# Twój docelowy rozmiar dla modelu (np. 360x640 lub 720x1280)
    target_size = (720, 1280)  # (Height, Width)

# Bufor na 5 klatek 3-kanałowych [5, Height_2K, Width_2K, 3]
    raw_frames_gpu = torch.empty((5, camera.height, camera.width, 3), dtype=torch.uint8, device=device)

# Główny tenzor [Batch=1, Channels=3, Depth=5, Height=360, Width=640]
    input_tensor = torch.empty((1, 3, 5, target_size[0], target_size[1]), dtype=torch.float32, device=device)
    speeds = []
    while camera.get_latest_frame() is None:
        pass
    for i in range(5):
            # KROK 1: Grab 3-kanałowej klatki BGR (uint8)
            img_np = camera.get_latest_frame() 
        
            if img_np is None: 
                continue
            
            # KROK 2: Przesunięcie i wgranie do VRAM (o 25% mniej bajtów po PCIe)
            raw_frames_gpu = torch.roll(raw_frames_gpu, shifts=-1, dims=0)
            raw_frames_gpu[-1] = (
    torch.from_numpy(img_np)
    .float()
    .div(255.0)
)
           
        
            # KROK 3: Operacje na GPU
            # [5, H, W, 3] -> konwersja do float i div_(255.0)
            gpu_processed = raw_frames_gpu
            
            # Przestawienie pod Conv3D: [5, H, W, 3] -> [1, 3, 5, H, W]
            gpu_processed = gpu_processed.permute(3,0, 1, 2).unsqueeze(0)
        
            # Skalowanie w VRAM
            gpu_processed =  F.interpolate(
                                        gpu_processed,
                                        size=(5,target_size[0], target_size[1]), 
                                        mode='nearest', 
                                    )
            
            input_tensor.copy_(gpu_processed)
            speeds.append(listner.get_speed_from_telemetry())
    print("Pętla gotowa na 3 kanały (BGR)...")
    with torch.no_grad():
        while not  keyboard.is_pressed("p"):
            pass
        while not keyboard.is_pressed("q"):
            img_np = camera.get_latest_frame() 
                    
            if img_np is None: 
                continue
            stopwatch = time.perf_counter()        
                        # KROK 2: Przesunięcie i wgranie do VRAM (o 25% mniej bajtów po PCIe)
            raw_frames_gpu = torch.roll(raw_frames_gpu, shifts=-1, dims=0)
            raw_frames_gpu[-1] = torch.from_numpy(img_np)
            core_speeds = [speeds[0]/360,speeds[1]/360,speeds[2]/360,speeds[3]/360,listner.get_speed_from_telemetry()/360]
            speeds = core_speeds
               # KROK 3: Operacje na GPU
               # [5, H, W, 3] -> konwersja do float i div_(255.0)
            gpu_processed = raw_frames_gpu.float().div_(255.0)
               
                        
                        # Przestawienie pod Conv3D: [5, H, W, 3] -> [1, 3, 5, H, W]
            gpu_processed = gpu_processed.permute(3, 0, 1, 2).unsqueeze(0)
                    
                        # Skalowanie w VRAM
                        
            resized = F.interpolate(
                           gpu_processed, 
                           size=(5,  target_size[0],target_size[1]), 
                           mode='nearest', 
                       )  
            input_tensor.copy_(resized)
 
            images_batch = input_tensor
            speeds_batch = speeds_batch = torch.Tensor(speeds).to(device, non_blocking=True).unsqueeze(0)
            
            with torch.autocast(device.type, enabled=True, dtype=torch.float16):
                outputs = model(images_batch, speeds_batch)
            # lt_rt = round(outputs[0][1].item(),2)
            # if lt_rt < 0:
            #     controller.set_triggers(-lt_rt,0)
            # else:
            #     controller.set_triggers(0,lt_rt)
            controller.set_left_stick(round(outputs[0][0].item(),3),0)
            controller.set_triggers(outputs[0][2],outputs[0][1])
            print(f"Time {(time.perf_counter() - stopwatch) * 1000}")
            print(outputs)
if __name__ == "__main__":
    from AiModels.OnlyTanhModel import OnlyTanhModel
    device = torch.device("cuda")
    model = FirstModel.ForzaH4Model().to(device)
            
    # # Wczytujemy checkpoint
    checkpoint = torch.load(r"C:\FH4_AI_Driver\model_200E16N4AUDI_TT_With_Goliath.pth", map_location=device)
    
    # Wczytujemy stan modelup
    model.load_state_dict(checkpoint)

    evaluate_model_in_FH4(model,device)