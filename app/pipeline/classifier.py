import numpy as np                                                                                                                                                                 
import onnxruntime as ort                                                                                                                                                          
from PIL import Image                                                                                                                                                              
from app.config.settings import settings

CLASS_NAMES = {0: "CHEQUE", 1: "PAN"}

# Load model (do this once at module level, not per request):
session = ort.InferenceSession(settings.model_path)

# Preprocess function:
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = image.transpose(2, 0, 1)  # HWC → CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict function:
def classify(image: Image.Image) -> dict:
    input_tensor = preprocess(image)
    outputs = session.run(None, {"image": input_tensor})
    probabilities = softmax(outputs[0][0])
    predicted_idx = np.argmax(probabilities)
    return {
        "document_type": CLASS_NAMES[predicted_idx],
        "confidence": float(probabilities[predicted_idx])
    }

# Softmax helper:
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()