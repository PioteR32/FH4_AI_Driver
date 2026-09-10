# Create the Forza Horizon 4 AI model file
import torch
import torch.nn as nn
import torch.nn.functional as F

class ForzaH4Model(nn.Module):
    """
    AI Model for Forza Horizon 4 that takes 5 time-ordered images and outputs
    steering (-1.0 to 1.0), throttle (0.0 to 1.0), brake (0.0 to 1.0)
    
    Input: 5 grayscale images of shape (batch, 5, 224, 224)
    Output: (batch, 3) where [steering, throttle, brake] in range [-1.0, 1.0], [0.0, 1.0], [0.0, 1.0]
    """
    
    def __init__(self, input_channels=3,num_of_img=5):
        super(ForzaH4Model, self).__init__()
        
        # Image processing backbone using a modified ResNet-like architecture
        
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Batch normalization layers for stable training
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.bn3 = nn.GroupNorm(num_groups=8, num_channels=128)
        self.bn4 = nn.GroupNorm(num_groups=8, num_channels=256)
        
        # Pooling layer to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool3d((num_of_img, 4, 4))
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(20480, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Output layer - 3 outputs for steering, throttle, brake
        self.output_layer = nn.Linear(128 + num_of_img, 3)  # 5 for the speeds input
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x,speeds):
        """
        Forward pass for the model
        Args:
            x: Input tensor of shape (batch_size, 5, 224, 224) - 5 time-ordered grayscale images
            speeds: Input tensor of shape (batch_size, 5) - corresponding speeds
        Returns:
            output: Tensor of shape (batch_size, 3) with [steering, throttle, brake] values
        """
        # Input shape: (batch_size, 5, 224, 224)
        
        # First convolutional block
       
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool3d(x,kernel_size=(1, 2, 2)) # Reduce spatial dimensions
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool3d(x,kernel_size=(1, 2, 2))
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool3d(x,kernel_size=(1, 2, 2))
        
        # Fourth convolutional block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool3d(x,kernel_size=(1, 2, 2))
        
        # Adaptive pooling to ensure fixed size
        x = self.pool(x)  # Shape: (batch_size, 256, 4, 4, 4)
        
        # Flatten for fully connected layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Shape: (batch_size, 256*4*4*4)
        
        # Fully connected layers with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.cat((x, speeds), dim=1)  # Concatenate speeds to the features
        # Output layer
        output = self.output_layer(x)
        
        # Process outputs with appropriate activation functions
        # Steering: tanh to map to [-1, 1]
        steering = torch.tanh(output[:, 0])
        
        # Throttle and brake: sigmoid to map to [0, 1] 
        throttle = torch.sigmoid(output[:, 1])
        brake = torch.sigmoid(output[:, 2])
        
        # Combine outputs
        combined_output = torch.stack([steering, throttle, brake], dim=1)
        
        return combined_output
    
    def predict(self, image_batch):
        """
        Make prediction on a batch of images
        Args:
            image_batch: Tensor of shape (batch_size, 5, 224, 224) - 5 time-ordered grayscale images
        Returns:
            steering, throttle, brake values in range [-1.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            output = self(image_batch)
            return output

# Example usage function
def demo_usage():
    """
    Demonstrate how to use the model
    """
    # Create a sample model instance
    model = ForzaH4Model()
    
    # Create sample input (batch_size=1, 5 images, 224x224 each)
    sample_input = torch.randn(1, 5, 224, 224)
    
    # Make prediction
    with torch.no_grad():
        output = model(sample_input)
        
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample outputs (steering, throttle, brake): {output[0]}")
    
    return model

# If this file is run directly, demonstrate usage
if __name__ == "__main__":
    demo_usage()