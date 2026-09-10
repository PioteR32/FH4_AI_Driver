import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
import gc

# Sprawdź, czy CUDA jest dostępne
if not torch.cuda.is_available():
    print("CUDA nie jest dostępne. Skrypt zakończony.")
    exit()

device = torch.device("cuda")
print(f"Używane urządzenie: {device}")

# Mocny model sieci neuronowej
class HeavyModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(HeavyModel, self).__init__()
        layers = []
        prev_size = input_size
        
        # Tworzymy wiele warstw
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden
            
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# Parametry dla mocnego obciążenia
input_size = 4096
hidden_sizes = [2048,1524, 1024, 512, 256, 128,64, 32, 16, 8, 4]  # Duża liczba warstw
num_classes = 1200000
batch_size = 400  # Duży batch
num_epochs = 1000

# Tworzenie modelu i danych
model = HeavyModel(input_size, hidden_sizes, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Dane treningowe (duże)
train_data = torch.randn(batch_size, input_size, device=device)
train_labels = torch.randint(0, num_classes, (batch_size,), device=device)

print("Rozpoczynanie mocnego obciążenia GPU...")
print(f"Symulacja trwa: {num_epochs} epok")
print(f"Rozpoczęto: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

start_time = time.time()
epoch_count = 0

try:
    while True:
        # Mocne obciążenie - wiele operacji na GPU
        model.train()
        
        # Wykonujemy wiele iteracji operacji na GPU
        for batch_idx in range(50):  # Dużo batchy
            optimizer.zero_grad()
            
            # Forward pass z wieloma operacjami
            outputs = model(train_data)
            
            # Dodatkowe obliczenia
            with torch.no_grad():
                temp1 = torch.relu(outputs)
                temp2 = torch.softmax(temp1, dim=1)
                temp3 = torch.tanh(temp2)
                temp4 = torch.sigmoid(temp3)
                
            loss = criterion(outputs, train_labels)
            
            # Backward pass (realne obliczenia)
            loss.backward()
            optimizer.step()
            
            # Wymuszamy synchronizację
            torch.cuda.synchronize()
        
        epoch_count += 1
        
        # Czyścimy pamięć co jakiś czas
        if epoch_count % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
        # Wyświetlanie postępu
        if epoch_count % 50 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            memory_used = torch.cuda.memory_allocated(device) / (1024**3)
            print(f"Epoka {epoch_count}: {elapsed:.2f}s | GPU: {memory_used:.2f} GB")
        
        # # Sprawdzenie czasu trwania (opcjonalnie zakończ po 6 godzinach)
        # if epoch_count >= num_epochs:
        #     break
            
except KeyboardInterrupt:
    print("\nSymulacja przerwana przez użytkownika")

end_time = time.time()
print(f"\nZakończono symulację trenowania.")
print(f"Całkowity czas: {(end_time - start_time):.2f} sekund")
print(f"Ukończonych epok: {epoch_count}")
print(f"Zakończono: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Wyświetlenie statystyk
print(f"\nStatystyki:")
print(f"GPU Memory Used: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB")
