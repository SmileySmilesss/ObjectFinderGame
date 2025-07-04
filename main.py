from fastapi import FastAPI, File, UploadFile, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import os

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download model if missing
model_path = "models/yolov8n.pt"
if not os.path.exists(model_path):
    from urllib.request import urlretrieve
    urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt", model_path)

# Load model
model = YOLO(model_path)

@app.options("/detect")
async def options_handler():
    # This explicitly handles preflight OPTIONS requests
    return Response(status_code=204)

@app.post("/detect")
async def detect_object(file: UploadFile = File(...), target: str = Form(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    results = model.predict(image, conf=0.25, verbose=False)[0]
    labels = [model.names[int(cls)] for cls in results.boxes.cls]
    match = target.lower() in [label.lower() for label in labels]

    return JSONResponse(content={"match": match, "labels": labels})
