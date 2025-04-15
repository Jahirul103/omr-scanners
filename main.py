from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ OMR Scanner is running"

@app.route('/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if 20 < w < 50 and 20 < h < 50 and 0.8 < aspect_ratio < 1.2:
            center = (int(x + w / 2), int(y + h / 2))
            radius = int((w + h) / 4)
            cv2.circle(image, center, radius, (255, 0, 0), 2)

    # Convert back to PIL image to send as response
    _, img_encoded = cv2.imencode('.png', image)
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)


