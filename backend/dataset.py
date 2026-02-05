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
