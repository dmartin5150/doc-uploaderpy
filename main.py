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
from numpy import array as np_array

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader
easyocr_reader = easyocr.Reader(["en"])

def enhance_contrast(pil_image, factor=2.0):
    """
    Convert to grayscale and increase contrast.
    """
    img_gray = pil_image.convert("L")
    enhancer = ImageEnhance.Contrast(img_gray)
    img_contrast = enhancer.enhance(factor)
    return img_contrast

def call_ollama(text: str) -> dict:
    prompt = f"""
You are extracting structured data from a medical referral document.

IMPORTANT INSTRUCTIONS:
- Ignore dates in headers and footers (e.g., fax timestamps).
- Use ONLY the date that appears next to "Date:" under "Practice Name".
- DO NOT reformat or reinterpret dates—return them *exactly* as seen in the text. For example, if the date reads "11/07/2024", return it exactly as "11/07/2024".
- Format phone numbers as ###-###-####.
- Format DOB as MM/DD/YYYY in all cases.  If it isn't in this format do your best to force it into this format.
- If fields are missing, return empty strings.
- DO NOT include any comments, explanations, or any other text before or after the JSON.
- Respond ONLY with valid JSON that can be parsed directly.
-In the header, there is a Patient: followed by a number and the patent's name. Use the number as the MRN

Return ONLY valid JSON in this format:

{{
  "practice_name": "",
  "referral_date": "",
  "patient_name": "",
  "mrn": "",
  "dob": "",
  "sex": "",
  "address": "",
  "city_state_zip": "",
  "home_phone": "",
  "work_phone": "",
  "mobile_phone": "",
  "diagnosis": "",
  "code:": "",
  "test_ordered": "",
  "referring_physician": ""
}}

Text to analyze:
{text}
"""

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

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned.", "raw_output": content}

    return parsed


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()

    text_tesseract = ""
    text_easyocr = ""

    if filename.endswith(".pdf"):
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

    # Combine both OCR outputs
    combined_text = text_tesseract + "\n" + text_easyocr

    if not combined_text.strip():
        return JSONResponse(content={"error": "No text found in document."})

    print("--- OCR Output Snippet ---")
    print(combined_text[:500])

    extracted = call_ollama(combined_text)
    print("--- Ollama Extraction Result ---")
    print(extracted)

    return JSONResponse(content={
        "type": "referral",
        "data": extracted
    })
