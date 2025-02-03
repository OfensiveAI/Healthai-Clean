from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image
import os
from openai import OpenAI

# Flask app instance
app = Flask(__name__)

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_photo():
    # Check if 'photo' is in the request
    if 'photo' not in request.files:
        print("No photo found in the request.")  # Debug line
        return jsonify({"error": "No photo uploaded"}), 400

    photo = request.files['photo']

    # Check if the file is empty
    if photo.filename == '':
        print("No selected file.")  # Debug line
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(photo)
    except Exception as e:
        print(f"Error opening image: {e}")  # Debug line
        return jsonify({"error": "Invalid image file"}), 400

    img_array = np.array(img)

    try:
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error converting image: {e}")  # Debug line
        return jsonify({"error": "Error processing image"}), 400

    brightness = np.mean(img_cv2)
    prompt = f"The brightness level of the uploaded photo is {brightness}. Provide a health tip based on this."

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        health_tip = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")  # Debug line
        return jsonify({"error": "Error getting health tip from AI"}), 500

    return jsonify({"health_tip": health_tip}), 200

# Ensure Flask runs on Render's port
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
