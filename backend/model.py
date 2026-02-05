<<<<<<< HEAD
import torch
import torch.nn as nn
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")

class CNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # ✅ 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
=======
import torch.nn as nn

class AlzheimerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
>>>>>>> 8d8b0dccf054482428d4f687cbccbc95945b0d3a
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
<<<<<<< HEAD
            nn.Linear(128 * 16 * 16, 256),  # ✅ matches training
            nn.ReLU(),
            nn.Linear(256, num_classes)
=======
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
>>>>>>> 8d8b0dccf054482428d4f687cbccbc95945b0d3a
        )

    def forward(self, x):
        x = self.features(x)
<<<<<<< HEAD
        x = self.classifier(x)
        return x


def load_model():
    model = CNNModel(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model
=======
        return self.classifier(x)
>>>>>>> 8d8b0dccf054482428d4f687cbccbc95945b0d3a
