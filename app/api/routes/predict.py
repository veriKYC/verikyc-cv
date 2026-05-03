from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/predict")
def classify(file: UploadFile = File(...)):
    # Placeholder for actual classification logic
    return {
        "document_type": "pan", 
        "confidence": 0.95
    }
