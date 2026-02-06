import torch
import torch.nn as nn
import os

# 1. Path Logic
# This ensures the model file is found regardless of where you run the script from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # IMPORTANT: This assumes your input image is 128x128. 
        # After three MaxPool layers (2x2), the size becomes 16x16.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), 
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model():
    """
    Initializes the model architecture and loads the saved weights.
    """
    model = AlzheimerCNN(num_classes=4)
    
    if os.path.exists(MODEL_PATH):
        # map_location="cpu" ensures it works even if you don't have a GPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"✅ Success: Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Model file not found at {MODEL_PATH}. Using random weights.")
    
    model.eval() # Set to evaluation mode
    return model