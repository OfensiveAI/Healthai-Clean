from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Renders the upload form

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    photo = request.files['photo']
    img = Image.open(photo)

    # Convert image to numpy array for OpenCV processing
    img_array = np.array(img)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Placeholder: Basic Image Analysis (e.g., checking image brightness)
    brightness = np.mean(img_cv2)

    # Simple health tip based on brightness (just an example!)
    if brightness > 150:
        health_tip = "Looks bright! Make sure to wear sunscreen if you're outside."
    else:
        health_tip = "Image looks dark. Consider increasing your vitamin D intake!"

    return jsonify({"health_tip": health_tip}), 200

if __name__ == '__main__':
    app.run(debug=True)
