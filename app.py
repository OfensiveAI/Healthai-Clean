from flask import Flask, request, jsonify, render_template
import os
from google.cloud import vision
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)

# Path to your Service Account JSON key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"  # Update if you kept the original name

# Initialize Google Vision API Client
client = vision.ImageAnnotatorClient()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    photo = request.files['photo']
    img = Image.open(photo)

    # Calculate brightness
    img_array = np.array(img)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    brightness = np.mean(img_cv2)

    # Prepare image for Vision API
    photo.seek(0)
    content = photo.read()
    image = vision.Image(content=content)

    # Use Vision API to detect labels
    response = client.label_detection(image=image)
    labels = [label.description for label in response.label_annotations]

    health_tip = f"The brightness level of the uploaded photo is {brightness:.2f}. The image contains: {', '.join(labels)}."

    return jsonify({"health_tip": health_tip})

if __name__ == '__main__':
    app.run(debug=True)
