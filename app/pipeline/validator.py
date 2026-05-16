from datetime import datetime
import re
# Verhoeff algorithm tables for Aadhaar checksum
D = [
[0,1,2,3,4,5,6,7,8,9],
[1,2,3,4,0,6,7,8,9,5],
[2,3,4,0,1,7,8,9,5,6],
[3,4,0,1,2,8,9,5,6,7],
[4,0,1,2,3,9,5,6,7,8],
[5,9,8,7,6,0,4,3,2,1],
[6,5,9,8,7,1,0,4,3,2],
[7,6,5,9,8,2,1,0,4,3],
[8,7,6,5,9,3,2,1,0,4],
[9,8,7,6,5,4,3,2,1,0],
]
P = [
[0,1,2,3,4,5,6,7,8,9],
[1,5,7,6,2,8,3,0,9,4],
[5,8,0,3,7,9,6,1,4,2],
[8,9,1,6,0,4,3,5,2,7],
[9,4,5,3,1,2,6,8,7,0],
[4,2,8,6,5,7,3,9,0,1],
[2,7,9,3,8,0,6,4,1,5],
[7,0,4,6,9,1,3,2,5,8],
]
INV = [0,4,3,2,1,5,6,7,8,9]

def _verhoeff_check(number: str) -> bool:
    c = 0
    for i, digit in enumerate(reversed(number)):
        c = D[c][P[i % 8][int(digit)]]
    return c == 0

def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%d/%m/%Y")
        return True
    except (ValueError, TypeError):
        return False

def _future_date(value: str) -> bool:
    try:
        return datetime.strptime(value, "%d/%m/%Y") > datetime.now()
    except (ValueError, TypeError):
        return False

def _ok(reason=None):
    return {"valid": True, "reason": reason}

def _fail(reason):
    return {"valid": False, "reason": reason}


VALIDATORS = {
    "PAN": {
        "pan_number": lambda v: _ok() if v and re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", v) else _fail("Invalid PAN format"),
        "dob":        lambda v: _ok() if _valid_date(v) else _fail("Invalid or missing date"),
        "name":       lambda v: _ok() if v and len(v.strip()) > 1 else _fail("Missing name"),
        "father_name":lambda v: _ok() if v and len(v.strip()) > 1 else _fail("Missing father name"),
    },
    "AADHAAR": {
        "aadhaar_number": lambda v: _ok() if v and _verhoeff_check(re.sub(r"\s", "", v)) else _fail("Invalid Aadhaar checksum"),
        "dob":            lambda v: _ok() if _valid_date(v) else _fail("Invalid or missing date"),
        "gender":         lambda v: _ok() if v in ("MALE", "FEMALE", "OTHER") else _fail("Invalid gender"),
        "name":           lambda v: _ok() if v and len(v.strip()) > 1 else _fail("Missing name"),
    },
    "PASSPORT": {
        "passport_number": lambda v: _ok() if v and re.fullmatch(r"[A-Z][0-9]{7}", v) else _fail("Invalid passport number"),
        "dob":             lambda v: _ok() if _valid_date(v) else _fail("Invalid or missing date"),
        "expiry_date":     lambda v: _ok() if _future_date(v) else _fail("Expired or missing expiry date"),
        "name":            lambda v: _ok() if v and len(v.strip()) > 1 else _fail("Missing name"),
    },
    "DL": {
        "dl_number":   lambda v: _ok() if v and re.fullmatch(r"[A-Z]{2}[0-9]{13}", v) else _fail("Invalid DL number format"),
        "dob":         lambda v: _ok() if _valid_date(v) else _fail("Invalid or missing date"),
        "expiry_date": lambda v: _ok() if _future_date(v) else _fail("Expired or missing expiry date"),
        "name":        lambda v: _ok() if v and len(v.strip()) > 1 else _fail("Missing name"),
    },
    "CHEQUE": {
        "ifsc_code":      lambda v: _ok() if v and re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", v) else _fail("Invalid IFSC code"),
        "micr_code":      lambda v: _ok() if v and re.fullmatch(r"\d{9}", v) else _fail("Invalid MICR code"),
        "account_number": lambda v: _ok() if v and re.fullmatch(r"\d{9,18}", v) else _fail("Invalid account number"),
    },
}
def validate(extracted_fields: dict, document_type: str) -> dict:
    validators = VALIDATORS.get(document_type, {})
    field_validations = {}

    for field, rule in validators.items():
        value = extracted_fields.get(field)
        field_validations[field] = rule(value)

    is_valid = all(r["valid"] for r in field_validations.values())

    return {
        "is_valid": is_valid,
        "field_validations": field_validations,
    }
