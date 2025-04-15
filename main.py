from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Live OMR Scanner is ready!"}

@app.post("/scan")
async def scan_omr(file: UploadFile = File(...)):
    # Read image bytes
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # Detect bubbles (contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_count = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # Assume bubble shape
        if 20 < w < 60 and 20 < h < 60:
            bubble_count += 1

    return JSONResponse({
        "status": "success",
        "bubbles_detected": bubble_count
    })

