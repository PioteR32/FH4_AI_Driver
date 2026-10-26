"""
Plik: aiDeepLearningEvaluation.py
Opis: Zawiera narzędzia do ewaluacji wytrenowanych modeli na danych testowych.
      Umożliwia ładowanie modeli z plików, ocenianie ich skuteczności (loss, accuracy)
      oraz porównywanie wyników wielu modeli jednocześnie.
"""
import argparse
import os
import time
from zipfile import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from torch.utils.data import DataLoader
from AiModels.FirstModel import ForzaH4Model
from AiModels.SecondModel import SecondFH4Model
from DataLoaders.DataLoader import SimpleDataset
import LossFunc
from rocmPy.DataLoaders import Resnet18DL
def evaluate_model(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    criterion: torch.nn.Module, 
    device: torch.device,
    model_name: str = "model"
) -> dict:
    """
    Evaluate a trained model on test data.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, (batch_inputs, targets) in enumerate(dataloader):
            images_raw, speeds_batch = batch_inputs

            # Przenoszenie na GPU i normalizacja (float16 dla szybkości)
            images_batch = images_raw.to(device, dtype=torch.float16, non_blocking=True).permute(0, 3, 1, 2).div_(255.0)
            speeds_batch = speeds_batch.to(device, dtype=torch.float16, non_blocking=True)
            targets = targets.to(device, dtype=torch.float16, non_blocking=True)

            with torch.autocast(device.type, enabled=True, dtype=torch.float16):
                outputs = model(images_batch, speeds_batch)
                loss = criterion(outputs, targets)
            
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    # Dla regresji (sterowanie) accuracy liczymy jako % błędów poniżej progu
    # Jeśli model jest klasyfikacją, używamy dokładności klas
    accuracy = 0.0
    if total_samples > 0:
        # To jest uproszczone - w regresji 'accuracy' to często brak błędu poniżej progu
        # Tutaj sprawdzamy, ile wyników mieści się w marginesie błędu
        # (Wymaga to dostępu do oryginalnych danych, tutaj robimy uproszczoną wersję)
        accuracy = 0.0 

    print(f"Evaluation results for {model_name}:")
    print(f"  Avg Loss: {avg_loss:.4f}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'model_name': model_name
    }


def find_model_files(directory: str, pattern: str) -> list:
    """
    Find all model files with a specific pattern in the given directory.
    Uses pathlib for robust path handling.
    """
    model_files = []
    path = Path(directory)
    
    if not path.exists():
        print(f"Error: Directory {directory} does not exist.")
        return []

    # Szukamy plików pasujących do wzorca z rozszerzeniem .pth
    # pattern może być np. "Second"
    for file in path.glob(f"{pattern}*.pth"):
        model_files.append(str(file))
    
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
    from AiModels.OneImg import OneImgModel
    # Define the loss function (using MSE since it's a regression task based on model structure)
    criterion = F.cross_entropy
    
    results = {}
    
    for model_file in model_files:
        print(f"Loading and evaluating model from: {model_file}")
        
        # Create a new instance of the model
        model = OneImgModel(num_speeds=90,rest_input_dim=48*3)
        
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
    parser = argparse.ArgumentParser(description="Evaluate multiple models on test data")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory containing model files")
    parser.add_argument("--pattern", type=str, default="Second", help="Pattern to match model files")
    parser.add_argument("--data_dir", type=str, default="c:/screenshots", help="Directory with test data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for test loader")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multiprocessing.freeze_support()
    
    model_dir = Path(args.model_dir)
    model_files = find_model_files(str(model_dir), args.pattern)
    
    if model_files:
        print(f"Found {len(model_files)} model files.")
        try:
            test_loader = DataLoader(
                Resnet18DL(args.data_dir, "AUDI_TT_EVAL", 30),
                batch_size=args.batch_size, 
                shuffle=False,  
                num_workers=args.num_workers,
                pin_memory=True,
                prefetch_factor=2
            )
            
            print("Running evaluation...")
            results = load_and_evaluate_models(model_files, device, test_loader)
            
            output_file = Path("__models_eval_results.txt")
            with open(output_file, "w") as f:
                for name, res in results.items():
                    f.write(f"{name}: {res}\n")
            
            print(f"Results saved to {output_file}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during evaluation: {e}")
    else:
        print(f"No matching model files found in {model_dir} with pattern '{args.pattern}'")
