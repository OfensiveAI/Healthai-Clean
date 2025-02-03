import openai
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    photo = request.files['photo']
    img = Image.open(photo)

    img_array = np.array(img)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Example: Simple brightness check
    brightness = np.mean(img_cv2)

    # Use OpenAI to generate a health tip
    prompt = f"The brightness level of the uploaded photo is {brightness}. Provide a health tip based on this."

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )

    health_tip = response.choices[0].text.strip()

    return jsonify({"health_tip": health_tip}), 200

if __name__ == '__main__':
    app.run(debug=True)
