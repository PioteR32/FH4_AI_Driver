import bettercam
import torch
import torch.nn.functional as F
import time
import keyboard
import multiprocessing
import numpy as np
class RealTimeData:
    def __init__(self, device, camera_fps: int = 90):
        self._device = device
        self._camera_fps = camera_fps
        self._isListning = False
        self._process = None
        self.shared_queue = multiprocessing.Queue(maxsize=1)
        self.lock = multiprocessing.Lock()
        
    def _start_listening(self, target_size=(720, 1280), stop_key="q",start_key="p"):
        """
        Worker function that runs in separate process
        """
        try:
            # Create camera instance inside this process only
            camera = bettercam.create(output_idx=0, output_color="BGR")
            camera.start(target_fps=self._camera_fps, video_mode=True)
            size = (5, target_size[0], target_size[1])
            print(f"Camera started in worker process: {camera}")
            
            # Create GPU tensor in worker process
            raw_frames_gpu = torch.empty((5, camera.height, camera.width, 3), 
                               dtype=torch.uint8, device=self._device)
            
            print("Processing frames...")
            # while not keyboard.is_pressed(start_key):
            #     pass
            while not keyboard.is_pressed(stop_key):
                loop_start = time.perf_counter()
                
                # Get frame (this is safe in worker process)
                img_np = camera.get_latest_frame() 
                if img_np is None: 
                    continue

                # Process in worker process
                raw_frames_gpu = torch.roll(raw_frames_gpu, shifts=-1, dims=0)
                raw_frames_gpu[-1].copy_(torch.from_numpy(img_np), non_blocking=True)

                gpu_processed = raw_frames_gpu.float().div_(255.0)
                gpu_processed = gpu_processed.permute(3, 0, 1, 2).unsqueeze(0)
                
                self.shared_queue.put(F.interpolate(
                        gpu_processed, 
                        size=size, 
                        mode='nearest'
                    ).to('cpu'))
                # This is where you'd want to share the result
                # For now just print timing info
                iteration_time_ms = (time.perf_counter() - loop_start) * 1000
                # print(f"Czas iteracji: {iteration_time_ms:.2f} ms")

                # time.sleep(0.2)
                
            camera.stop()
            
        except Exception as e:
            print(f"Worker error in process: {e}")
            raise

    def start_listening(self, target_size=(720, 1280), stop_key="q"):
        """
        Start the camera processing in a separate process
        """
        if self._isListning:
            print("Already listening")
            return
            
        self._isListning = True
        
        # Create and start worker process - only serializable arguments
        self._process = multiprocessing.Process(
            target=self._start_listening,
            args=(target_size, stop_key)
        )
        self._process.daemon = True
        self._process.start()
        
        print(f"Started listening in separate process: PID={self._process.pid}")
        return self._process

    def stop_listening(self):
        """
        Stop the worker process
        """
        if self._process and self._process.is_alive():
            self._isListning = False
            self._process.terminate()
            self._process.join(timeout=2)
            if self._process.is_alive():
                print("Process did not terminate properly")
        self._isListning = False
    def get_frames(self)->torch.Tensor:
        
        return self.shared_queue.get_nowait().clone()

# Usage:
if __name__ == "__main__":
    real = RealTimeData(device=torch.device("cuda"))
    process = real.start_listening(target_size=(720, 1280), stop_key="q")
    
    try: 
        i = 0
        
        while True:
           
            stopwatch = time.perf_counter()
            real.get_frames()
            print((time.perf_counter() - stopwatch) * 1000)
            
    except KeyboardInterrupt:
        print("Stopping...")
        real.stop_listening()