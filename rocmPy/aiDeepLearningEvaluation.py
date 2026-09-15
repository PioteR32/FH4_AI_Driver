import os
import time
import torch
import torch.nn as nn
import multiprocessing
from torch.utils.data import DataLoader
from AiModels.FirstModel import ForzaH4Model
from AiModels.SecondModel import SecondFH4Model
from DataLoaders.DataLoader import SimpleDataset

def evaluate_model(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    criterion: torch.nn.Module, 
    device: torch.device,
    model_name: str = "model"
) -> dict:
    """
    Evaluate a trained model on test data
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    
    # For accuracy calculation, assuming we're using CrossEntropyLoss for classification
    with torch.no_grad():
        for i, (batch_inputs, targets) in enumerate(dataloader):
            images_raw, speeds_batch = batch_inputs

            # Przenosimy na GPU i normalizujemy w jednym kroku (oszczędność VRAM)
            images_batch = images_raw.to(device, dtype=torch.float16, non_blocking=True).permute(0, 4, 1, 2, 3).div_(255.0)
            speeds_batch = speeds_batch.to(device, dtype=torch.float16, non_blocking=True)
            targets = targets.to(device, dtype=torch.float16, non_blocking=True)

            with torch.autocast(device.type, enabled=True, dtype=torch.float16):
                outputs = model(images_batch, speeds_batch)
                loss = criterion(outputs, targets)
            
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calculate accuracy (if applicable for the task)
            # For multi-class classification, we'll compute accuracy based on max values
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == targets).sum().item()
    
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    print(f"Evaluation results for {model_name}:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'model_name': model_name
    }

def find_model_files(directory: str, pattern: str = "model_200E32BLR0_0001") -> list:
    """
    Find all model files with a specific pattern in the given directory
    """
    model_files = []
    try:
        for filename in os.listdir(directory):
            if filename.startswith(pattern) and filename.endswith('.pth'):
                model_files.append(os.path.join(directory, filename))
    except Exception as e:
        print(f"Error finding model files: {e}")
    
    # Sort files numerically to get them in order
    model_files.sort()
    return model_files

def load_and_evaluate_models(model_files: list, device: torch.device, test_loader: DataLoader) -> dict:
    """
    Load multiple models from file paths and evaluate them on test data
    
    Args:
        model_files (list): List of model file paths to load and evaluate
        device (torch.device): Device to run evaluation on (cuda or cpu)
        test_loader (DataLoader): DataLoader for test data
        
    Returns:
        dict: Dictionary with results for each model
    """
    # Define the loss function (using MSE since it's a regression task based on model structure)
    criterion = nn.SmoothL1Loss()
    
    results = {}
    
    for model_file in model_files:
        print(f"Loading and evaluating model from: {model_file}")
        
        # Create a new instance of the model
        model = ForzaH4Model()
        
        # Load the model state dict
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
            print(f"Model loaded successfully from {model_file}")
        except Exception as e:
            print(f"Error loading model from {model_file}: {e}")
            results[model_file] = {'error': str(e)}
            continue
            
        # Move model to device
        model = model.to(device)
        
        # Evaluate the model
        stopwatch = time.perf_counter()
        model_name = os.path.basename(model_file)
        eval_results = evaluate_model(model, test_loader, criterion, device, model_name)
        print(f"czas: {time.perf_counter() - stopwatch } s" )
        results[model_file] = eval_results
        
        # Clear GPU cache to keep only one model in memory at a time
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    from DataLoaders.TestDataLoader import TestDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multiprocessing.freeze_support()
    
    # Example usage for model evaluation
    # In practice, you would need to set up your test dataset path
    # and potentially create a separate test loader
    
    print("Example evaluation setup:")
    print("To use this evaluation loop:")
    print("1. Create a test dataset with your test data")
    print("2. Set up a DataLoader for the test data")
    print("3. Specify model paths you want to evaluate")
    print("4. Run the evaluation with:")
    print("   model_paths = ['model1.pth', 'model2.pth']")
    print("   results = load_and_evaluate_models(model_paths, device, test_loader)")
    
    # NEW: Auto-discover model files in current directory
    current_directory = "C:\\FH4_AI_Driver"
    print(f"Looking for model files in: {current_directory}")
    
    # Find all model files with the pattern used during training
    model_files = find_model_files(current_directory, "model_")
    
    if model_files:
        print(f"Found {len(model_files)} model files:")
        for model_file in model_files:
            print(f"  - {os.path.basename(model_file)}")
        
        # Setup test loader (example with same configuration as training)
        # You may need to modify the path and folder name based on your actual test data
        try:
            test_loader = DataLoader(
                TestDataset("c:\\screenshots", "AUDI_TT_EVAL"),
                  batch_size=1, 
                  shuffle=False,  # Usually we don't shuffle for evaluation
                  num_workers=8,
                  pin_memory=True,
                  persistent_workers=True,
                  prefetch_factor=2)
            
            print("Test DataLoader created successfully")
            print("Running evaluation...")
            results = load_and_evaluate_models(model_files,device,test_loader)
            with open("__modelsTests.txt","a") as file:
                for result in results:
                    file.write("\n" +str(results[result]))
                    file.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print("error: " + str(e.args))