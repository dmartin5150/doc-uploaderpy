from io import BytesIO
from PIL import Image, ImageEnhance
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract
import easyocr
import requests
import json
import re
from numpy import array as np_array

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

easyocr_reader = easyocr.Reader(["en"])

def enhance_contrast(pil_image, factor=2.0):
    img_gray = pil_image.convert("L")
    enhancer = ImageEnhance.Contrast(img_gray)
    img_contrast = enhancer.enhance(factor)
    return img_contrast

def build_prompt(text: str, prompt_type: str) -> str:
    if prompt_type == "pdf":
        prompt = f"""
You are extracting structured data from a PDF medical referral document.

IMPORTANT INSTRUCTIONS:
- Ignore dates in headers and footers (e.g., fax timestamps).
- Use ONLY the date that appears next to "Date:" under "Practice Name".
- DO NOT reformat or reinterpret dates—return them exactly as seen.
- Format phone numbers as (###) ###-####.
- Format DOB as MM/DD/YYYY.
- If fields are missing, return empty strings.
- DO NOT include any comments or explanations before or after the JSON.
- In the header, there is a Patient: followed by a number and the patient's name. Use the number as the MRN.

Return ONLY valid JSON:

{{
  "practice_name": "",
  "date": "",
  "patient_name": "",
  "mrn": "",
  "dob": "",
  "sex": "",
  "address": "",
  "city_state_zip": "",
  "home_phone": "",
  "work_phone": "",
  "mobile_phone": ""
}}

Text to analyze:
{text}
"""
    else:
        prompt = f"""
You are extracting structured data from a scanned TIFF or image medical referral document.

IMPORTANT INSTRUCTIONS:
- Be extra cautious of OCR errors—validate dates and numbers carefully.
- If you are unsure about a field, leave it blank.
- Ignore dates in headers and footers (e.g., fax timestamps).
- The referral_date is the date under the Practice Name.
- Format phone numbers as (###) ###-####.
- Format DOB and referral_date as MM/DD/YYYY.
- If fields are missing, return empty strings.
- DO NOT include any comments or explanations before or after the JSON.
- In the header, there is a Patient: followed by a number and the patient's name. Use the number as the MRN.

Return ONLY valid JSON:

{{
  "practice_name": "",
  "referral_date": "",
  "patient_name": "",
  "mrn": "",
  "dob": "",
  "home_phone": "",
  "mobile_phone": "",
  "diagnosis": "",
  "code": "",
  "test_ordered": "",
  "referring_physician": ""
}}

Text to analyze:
{text}
"""
    return prompt

def extract_json_from_string(raw_output: str) -> dict:
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return {"error": "Could not locate JSON in output.", "raw_output": raw_output}
    json_str = match.group(0)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "json_snippet": json_str}
    return parsed

def normalize_date(date_str: str) -> str:
    """
    Normalizes dates to MM/DD/YYYY.
    """
    date_str = date_str.strip().replace("-", "/")
    if re.match(r"\d{2}/\d{2}/\d{4}", date_str):
        return date_str

    m = re.match(r"(\d{1,2})(\d{2})(\d{4})", date_str)
    if m:
        month, day, year = m.groups()
        month = month.zfill(2)
        return f"{month}/{day}/{year}"

    return date_str

def normalize_phone(phone_str: str) -> str:
    """
    Normalizes phone numbers to (###) ###-####
    """
    digits = re.sub(r"\D", "", phone_str)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone_str.strip()

def call_ollama(text: str, prompt_type: str) -> dict:
    prompt = build_prompt(text, prompt_type)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    data = response.json()
    print("Raw Ollama Response:", data)

    content = data.get("response", "")
    if not content:
        return {"error": "No content returned from Ollama.", "raw_data": data}

    parsed = extract_json_from_string(content)
    return parsed

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()

    text_tesseract = ""
    text_easyocr = ""
    prompt_type = "tiff"

    if filename.endswith(".pdf"):
        prompt_type = "pdf"
        try:
            doc = fitz.open(stream=contents, filetype="pdf")
            page_index = 1 if doc.page_count > 1 else 0
            text_tesseract = doc[page_index].get_text()
            if len(text_tesseract.strip()) < 10:
                raise ValueError("PDF text empty")
        except Exception as e:
            print("PyMuPDF failed, falling back to OCR:", e)
            images = convert_from_bytes(contents, dpi=300)
            img = images[0]
            img = enhance_contrast(img)
            text_tesseract = pytesseract.image_to_string(img, config="--psm 6")
            text_easyocr = "\n".join([line[1] for line in easyocr_reader.readtext(np_array(img))])
    else:
        img = Image.open(BytesIO(contents))
        img = enhance_contrast(img)
        text_tesseract = pytesseract.image_to_string(img, config="--psm 6")
        text_easyocr = "\n".join([line[1] for line in easyocr_reader.readtext(np_array(img))])

    combined_text = text_tesseract + "\n" + text_easyocr

    if not combined_text.strip():
        return JSONResponse(content={"error": "No text found in document."})

    print("--- OCR Output Snippet ---")
    print(combined_text[:500])

    extracted = call_ollama(combined_text, prompt_type)
    print("--- Ollama Extraction Result ---")
    print(extracted)

    if "error" in extracted:
        return JSONResponse(content={"error": extracted["error"], "details": extracted})

    # Clean up and normalize fields
    if prompt_type == "pdf":
        expected_keys = [
            "practice_name", "date", "patient_name", "mrn",
            "dob", "sex", "address", "city_state_zip",
            "home_phone", "work_phone", "mobile_phone"
        ]
    else:
        expected_keys = [
            "practice_name", "referral_date", "patient_name", "mrn",
            "dob", "home_phone", "mobile_phone",
            "diagnosis", "code", "test_ordered", "referring_physician"
        ]

    clean_data = {key: extracted.get(key, "") for key in expected_keys}

    # Normalize date fields
    for date_field in ["date", "dob", "referral_date"]:
        if date_field in clean_data and clean_data[date_field]:
            clean_data[date_field] = normalize_date(clean_data[date_field])

    # Normalize phone fields
    for phone_field in ["home_phone", "work_phone", "mobile_phone"]:
        if phone_field in clean_data and clean_data[phone_field]:
            clean_data[phone_field] = normalize_phone(clean_data[phone_field])

    return JSONResponse(content={
        "file_type": prompt_type,
        "type": "referral",
        "data": clean_data
    })
