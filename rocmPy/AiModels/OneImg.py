import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class OneImgModel(nn.Module):
    def __init__(self, input_channels=3, num_speeds=3, rest_input_dim=48):
        super().__init__()
        
        # Warstwy splotowe - każda ma SWOJĄ WŁASNĄ warstwę normalizacji!
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.GroupNorm(num_groups=16, num_channels=128)
        
        # Unikamy filtra 5x5 - używamy zoptymalizowanego 3x3 z dedykowanym GroupNorm
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5_conv = nn.GroupNorm(num_groups=16, num_channels=128) 
        
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.GroupNorm(num_groups=16, num_channels=256)
        
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.GroupNorm(num_groups=32, num_channels=512)
        
        # Adaptive pool 4x4 dla lepszego zachowania cech przestrzennych
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Przetwarzanie prędkości
        self.restInputs = nn.Sequential(
            nn.Linear(num_speeds, rest_input_dim),
            nn.ReLU()
        )
        
        # 512 kanałów * 4 * 4 piksele = 8192
        img_feature_dim = 512 * 4 * 4
        
        # Odchudzone i znacznie szybsze warstwy FC
        self.fc1 = nn.Linear(img_feature_dim + rest_input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.output_layer = nn.Linear(128, 3)
        
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x, speeds):
        """
        Input:
            x: (batch_size, 3, 224, 224)
            speeds: (batch_size, num_speeds)
        Output:
            combined_output: (batch_size, 3) -> [steering (-1 do 1), throttle (0 do 1), brake (0 do 1)]
        """
        # Każda warstwa używa swojego dedykowanego bloku BN/GN
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn5_conv(self.conv5(x))) # Unikalny bn5_conv
        x = F.relu(self.bn5_conv(self.conv5(x))) # Unikalny bn5_conv
        x = F.relu(self.bn4(self.conv6(x)))
        x = F.relu(self.bn5(self.conv8(x)))
        
        # Pooling i spłaszczenie (batch_size, 8192)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        # Dodanie cech prędkości
        extra_features = self.restInputs(speeds)
        x = torch.cat((x, extra_features), dim=1)
        
        # Warstwy Fully Connected z umiarkowanym Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        
        # Warstwa wyjściowa
        output = self.output_layer(x)
        
        # Aktywacje dopasowane do wartości wyjściowych
        steering = torch.tanh(output[:, 0])
        throttle = torch.sigmoid(output[:, 1])
        brake = torch.sigmoid(output[:, 2])
        
        # Złożenie wyjścia w jeden tensor (batch_size, 3)
        combined_output = torch.stack([steering, throttle, brake], dim=1)
        
        return combined_output
    
    def predict(self, image_batch, speeds):
        """
        Make prediction on a batch of images
        Args:
            image_batch: Tensor of shape (batch_size, 3, 224, 224)
            speeds: Tensor of shape (batch_size, num_speeds)
        Returns:
            steering, throttle, brake values in range [-1.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            return self(image_batch, speeds)

def demo_usage():
    """
    Demonstrate how to use the model
    """
    # Create a sample model instance
    num_speeds = 1
    model = OneImgModel(input_channels=3, num_speeds=num_speeds)
    
    # Create sample input (batch_size=1, 3 channels, 224x224 each)
    sample_input = torch.randn(1, 3, 224, 224)
    sample_speeds = torch.randn(1, num_speeds)
    
    # Make prediction
    with torch.no_grad():
        output = model(sample_input, sample_speeds)
        
    print(f"Input shape: {sample_input.shape}")
    print(f"Speeds shape: {sample_speeds.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample outputs (steering, throttle, brake): {output[0]}")
    
    return model

if __name__ == "__main__":
    demo_usage()
