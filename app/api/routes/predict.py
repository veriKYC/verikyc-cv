from io import BytesIO
from app.pipeline.ocr import ocr

from PIL import Image
from fastapi import APIRouter, UploadFile, File

from app.pipeline.classifier import classify

router = APIRouter()

@router.post("/predict")
def predict(file: UploadFile = File(...)):

    contents = file.file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    classification = classify(image)
    document_type = classification["document_type"]

    ocr_result = ocr(image, document_type)


    return {
        "document_type": document_type,
        "confidence": classification["confidence"],
        "extracted_fields": ocr_result["extracted_fields"],
        "confidence_scores": ocr_result["confidence_scores"]
    }
