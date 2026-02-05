from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


from backend.inference import predict_image

from inference import predict_image   # ✅ SAME-FOLDER IMPORT


app = FastAPI(title="Alzheimer MRI Image Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an MRI image and get Alzheimer stage prediction
    """
    return await predict_image(file)
