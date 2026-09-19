import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18,ResNet18_Weights
class Resnet18Model(nn.Module):
    """
    AI Model for Forza Horizon 4 that takes 5 time-ordered images and outputs
    steering (-1.0 to 1.0), throttle (0.0 to 1.0), brake (0.0 to 1.0)
    
    Input: 5 grayscale images of shape (batch, 5, 224, 224)
    Output: (batch, 3) where [steering, throttle, brake] in range [-1.0, 1.0], [0.0, 1.0], [0.0, 1.0]
    """
    
    def __init__(self, input_channels=3,):
        super(Resnet18Model, self).__init__()
        self.restInputs = nn.Linear(3,48)
        self.backbone = resnet18(weights= ResNet18_Weights)
        self.backbone.fc = nn.Identity()

        self.l1 = nn.Sequential(nn.Linear(560,256),nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(256,128),nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(128,3))
        
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
        x = self.backbone(x)
        x = self.l1(torch.cat([x,self.restInputs(speeds)],dim=1))
        x = self.l2(x)
        # Output layer
        output = self.l3(x)
        
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
    model = Resnet18Model()
    
    # Create sample input (batch_size=1, 5 channels, 224x224 each)
    sample_input = torch.randn(1, 3, 224, 224)
    rest = torch.randn(1, 3)
    
    # Make prediction
    with torch.no_grad():
        output = model(sample_input,rest)
        
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample outputs (steering, throttle, brake): {output[0]}")
    
    return model

# If this file is run directly, demonstrate usage
if __name__ == "__main__":
    demo_usage()