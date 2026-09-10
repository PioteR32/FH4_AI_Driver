import bettercam
import torch
import torch.nn.functional as F
import time
import threading


class CameraProcessor:
    def __init__(self, target_size=(360, 640), fps=60):
        self.target_size = target_size
        self.fps = fps

        # Inicjalizacja kamery
        self.camera = bettercam.create(output_idx=0, output_color="BGR")
        self.camera.start(target_fps=fps, video_mode=True)

        # Parametry
        self.height = self.camera.height
        self.width = self.camera.width

        # Bufor GPU [5, H, W, 3]
        self.raw_frames_gpu = torch.empty(
            (5, self.height, self.width, 3),
            dtype=torch.uint8,
            device=torch.device('cuda')
        )

        # Tensor wejściowy [1, 3, 5, H, W]
        self.input_tensor = torch.empty(
            (1, 3, 5, self.target_size[0], self.target_size[1]),
            dtype=torch.float32,
            device=torch.device('cuda')
        )

        # Wątek
        self.running = False
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True

        # Zmienna do przechowywania ostatniego batcha
        self.last_batch = None
        self.batch_lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.camera.stop()

    def get_last_batch(self):
        """Funkcja zwracająca ostatni batch"""
        with self.batch_lock:
            return self.last_batch

    def _loop(self):
        print("Rozpoczynam odbiór kadrów...")
        while self.running:
            loop_start = time.perf_counter()

            # KROK 1: Pobranie klatki BGR (uint8)
            img_np = self.camera.get_latest_frame()
            if img_np is None:
                continue

            # KROK 2: Przesunięcie i wgranie do VRAM
            self.raw_frames_gpu = torch.roll(self.raw_frames_gpu, shifts=-1, dims=0)
            self.raw_frames_gpu[-1] = torch.from_numpy(img_np)

            # KROK 3: Operacje na GPU
            gpu_processed = self.raw_frames_gpu.float().div_(255.0)
            gpu_processed = gpu_processed.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, 5, H, W]

            # Skalowanie do docelowego rozmiaru
            resized = F.interpolate(
                gpu_processed,
                size=(5, self.target_size[0], self.target_size[1]),
                mode='nearest'
            )

            self.input_tensor.copy_(resized)

            # Zapisz ostatni batch
            with self.batch_lock:
                self.last_batch = self.input_tensor.clone()

            iteration_time_ms = (time.perf_counter() - loop_start) * 1000
            print(f"Czas iteracji: {iteration_time_ms:.2f} ms")


def main():
    print("Uruchamiam test kamery...")
    
    # Stwórz procesor
    processor = CameraProcessor(target_size=(360, 640), fps=60)
    processor.start()

    try:
        # Test przez 5 sekund
        for i in range(10):  # 10 iteracji po 0.5s
            # time.sleep(0.5)
            # Pobierz ostatni batch
            batch = processor.get_last_batch()
            while batch is None:
                pass
            if batch is not None:
                print(f"Otrzymano batch: {batch.shape}")
            else:
                print("Brak danych")
        
        print("Test zakończony.")
    except KeyboardInterrupt:
        print("Przerwano przez użytkownika.")
    finally:
        processor.stop()


if __name__ == "__main__":
    main()
