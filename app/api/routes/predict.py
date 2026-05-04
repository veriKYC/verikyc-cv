from io import BytesIO

from PIL import Image
from fastapi import APIRouter, UploadFile, File

from app.pipeline.classifier import classify

router = APIRouter()

@router.post("/predict")
def predict(file: UploadFile = File(...)):
    # Placeholder for actual classification logic
    contents = file.file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    result = classify(image)
    return result
