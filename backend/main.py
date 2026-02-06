from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_image 
# OR, if 'backend' is treated as a package:
from backend.inference import predict_image

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
    return await predict_image(file)

if __name__ == "__main__":
    import uvicorn
    # Change host to "127.0.0.1" to get the clickable local link in terminal
    uvicorn.run(app, host="127.0.0.1", port=8000)