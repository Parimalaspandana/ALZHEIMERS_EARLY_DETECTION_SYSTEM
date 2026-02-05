<<<<<<< HEAD
from PIL import Image
import torchvision.transforms as transforms
import torch

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ✅ grayscale
    transforms.Resize((128, 128)),                 # ✅ correct size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)  # batch dimension
    return image
=======
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class AlzheimerDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []

        self.classes = [
            "Non_Demented",
            "Very_Mild_Demented",
            "Mild_Demented",
            "Moderate_Demented"
        ]

        for label, cls in enumerate(self.classes):
            folder = os.path.join(root_dir, cls)
            for img in os.listdir(folder):
                self.images.append(os.path.join(folder, img))
                self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
>>>>>>> 8d8b0dccf054482428d4f687cbccbc95945b0d3a
