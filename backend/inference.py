from PIL import Image
import torch
from fastapi import UploadFile

from backend.model import load_model
from backend.dataset import preprocess_image


model = load_model()

CLASSES = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

async def predict_image(file: UploadFile):
    image = Image.open(file.file).convert("RGB")

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return {
        "prediction": CLASSES[predicted_class]
    }
