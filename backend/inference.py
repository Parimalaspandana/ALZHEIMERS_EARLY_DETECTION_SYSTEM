<<<<<<< HEAD
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
=======
import torch
from PIL import Image
from torchvision import transforms
from model import AlzheimerCNN   # ✅ SAME-FOLDER IMPORT
from pathlib import Path
import io

# Class labels (MUST match training order)
CLASSES = [
    "Non_Demented",
    "Very_Mild_Demented",
    "Mild_Demented",
    "Moderate_Demented"
]

# Load model relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AlzheimerCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Image preprocessing (MATCH TRAINING)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

async def predict_image(file):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probs = torch.softmax(output, dim=1)[0]

        idx = torch.argmax(probs).item()

        return {
            "prediction": CLASSES[idx],
            "confidence": round(float(probs[idx]) * 100, 2),
            "probabilities": {
                CLASSES[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(CLASSES))
            }
        }

    except Exception as e:
        return {"error": str(e)}
>>>>>>> 8d8b0dccf054482428d4f687cbccbc95945b0d3a
