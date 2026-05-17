import re
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


print("INIT: Creating PaddleOCR engine...") 
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
print("INIT: PaddleOCR engine created successfully")

BOILERPLATE = {
    # PAN
    "INCOME TAX DEPARTMENT", "GOVT OF INDIA", "PERMANENT ACCOUNT NUMBER",
    "SIGNATURE", "GOVERNMENT OF INDIA", "PERMANENT ACCOUNT NUMBER CARD", "INCOME TAXDEPARTMENT", "INCOMETAX DEPARTMENT", "PERMANENT ACCOUNT NUM",
    # AADHAAR
    "UNIQUE IDENTIFICATION AUTHORITY OF INDIA", "AADHAAR",
    "MERA AADHAAR MERI PEHCHAAN",
    # PASSPORT
    "REPUBLIC OF INDIA", "PASSPORT", "TYPE", "COUNTRY CODE",
    "NATIONALITY", "INDIAN", "IND",
    # DL
    "DRIVING LICENCE", "TRANSPORT DEPARTMENT",
    "UNION OF INDIA",
    # CHEQUE (no name fields, so less of an issue)
}


FIELD_PATTERNS = {                                                                                                                                                             
    "PAN": {                                                                                                                                                                   
        "pan_number":   r"[A-Z]{5}[0-9]{4}[A-Z]",                                                                                                                              
        "dob":          r"\d{2}/\d{2}/\d{4}",                                                                                                                                  
        "name":         None,                                                                                                                                              
        "father_name":  None,
    },
    "CHEQUE": {
        "ifsc_code":        r"[A-Z]{4}0[A-Z0-9]{6}",
        "micr_code":        r"\d{9}",
        "account_number":   r"\d{9,18}",
    },
    "AADHAAR": {
        "aadhaar_number":   r"\d{4}\s\d{4}\s\d{4}",
        "dob":              r"\d{2}/\d{2}/\d{4}",
        "gender":           r"\b(MALE|FEMALE|OTHER)\b",
        "name":             None,
    },
    # dob and expiry_date share the same date pattern
    # first match goes to dob, second to expiry_date — order depends on OCR spatial reading
    "PASSPORT": {
        "passport_number":  r"[A-Z][0-9]{7}",
        "dob":              r"\d{2}/\d{2}/\d{4}",
        "expiry_date":      r"\d{2}/\d{2}/\d{4}",
        "name":             None,
    },
    "DL": {
        "dl_number":    r"[A-Z]{2}[0-9]{13}",
        "dob":          r"\d{2}/\d{2}/\d{4}",
        "expiry_date":  r"\d{2}/\d{2}/\d{4}",
        "name":         None,
    }
}

def ocr(image: Image.Image, document_type: str) -> dict:
    extracted_fields = {}
    confidence_scores = {}

    try:
        image_array = np.array(image)
        result = ocr_engine.ocr(image_array)                                                                                            
        print("OCR RAW:", result)

        patterns = FIELD_PATTERNS.get(document_type, {})
        lines = result[0] if result and result[0] else []
        texts = [line[1][0] for line in lines]
        scores = [line[1][1] for line in lines]

        for text, score in zip(texts, scores):
            text = text.strip().upper()
            for field, pattern in patterns.items():
                if field in extracted_fields:
                    continue
                if pattern and re.search(pattern, text):
                    extracted_fields[field] = text
                    confidence_scores[field] = round(float(score), 4)

        used_texts = set()                                                                                                                                                     
        for text, score in zip(texts, scores):                                                                                                                                 
            text = text.strip().upper()                                                                                                                                        
            if text in BOILERPLATE or text in used_texts:                                                                                                                  
                continue
            for field, pattern in patterns.items():
                if field in extracted_fields:
                    continue
                if pattern is None and score >= 0.7 and re.match(r"^[A-Z\s]+$", text) and len(text) > 2:
                    extracted_fields[field] = text
                    confidence_scores[field] = round(float(score), 4)
                    used_texts.add(text)
                    break


    except Exception as e:
        print("OCR ERROR:", e) 
        return {"extracted_fields": {}, "confidence_scores": {}}
    
    for field in FIELD_PATTERNS.get(document_type, {}):                                                                                                                        
        if field not in extracted_fields:                                                                                                                                      
            extracted_fields[field] = None                                                                                                                                 
            confidence_scores[field] = 0.0

    return {
        "extracted_fields": extracted_fields,
        "confidence_scores": confidence_scores
    }
    