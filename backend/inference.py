import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import io
from fastapi import UploadFile

# Import the class from your model.py
from .model import AlzheimerCNN 

# 1. Configuration & Pathing
CLASSES = [
    "Non_Demented",
    "Very_Mild_Demented",
    "Mild_Demented",
    "Moderate_Demented"
]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Model Once
def init_model():
    model = AlzheimerCNN()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Model weights loaded onto {DEVICE}")
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found!")
    model.to(DEVICE)
    model.eval()
    return model

model = init_model()

# 3. Preprocessing (Must match your training input: 128x128, Grayscale)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 4. Prediction Logic
async def predict_image(file: UploadFile):
    try:
        # Read file contents
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess and add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image_tensor)
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