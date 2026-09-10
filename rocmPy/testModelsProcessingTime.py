
import AiModels.FirstModel as FirstModel
import MenagePhotos.Screenshot as Screenshot
import numpy as np
import torch.nn as nn
import torch
import time
region_crop = 0,0,(int)(1280),(int)(720)
def get_one_batch_screenshots(num_of_screenshots: int = 1):
    list_of_screenshots = []
    for _ in range(num_of_screenshots):
        screenshot = Screenshot.capture_screenshot(region=region_crop)
        list_of_screenshots.append(np.asarray(screenshot))
    return list_of_screenshots

if __name__ == "__main__":
    model = FirstModel.ForzaH4Model(input_channels=5).to('cuda')
    model.eval()  # Set the model to evaluation mode

    screenshots = get_one_batch_screenshots(5)
    tensor_start = time.perf_counter()
    screenshots_np = np.stack(screenshots, axis=0)
    torch_screenshots = torch.from_numpy(screenshots_np).float()
    torch_screenshots = torch.tensor(screenshots, dtype=torch.float32,device='cuda').unsqueeze(0)
    torch_screenshots = torch_screenshots.permute(0, 1, 4, 2, 3) # Add batch dimension
    tensor_end = time.perf_counter()
    print(f"Time taken to convert screenshots to tensor: {tensor_end - tensor_start} seconds")
    print(f"Shape of screenshots: {torch_screenshots.shape}")
    for i in range(10):
        time_start = time.perf_counter()
        with torch.no_grad():
            output = model(torch_screenshots)
            torch.cuda.synchronize()  # Ensure all CUDA operations are finished
        time_end = time.perf_counter()
        print(f"Processing time: {time_end - time_start} seconds")
        print(f"Output shape: {output.shape}")
    