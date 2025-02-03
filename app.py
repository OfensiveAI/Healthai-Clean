from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image
import os
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    # Example: Simple brightness check (can be replaced with more complex analysis)
    brightness = np.mean(img_cv2)

    # Create a prompt for OpenAI to generate health tips
    prompt = f"The uploaded photo has an average brightness level of {brightness}. Provide a relevant health tip based on the brightness and possible health considerations."

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    health_tip = response.choices[0].message.content.strip()

    return jsonify({"health_tip": health_tip}), 200

if __name__ == '__main__':
    app.run(debug=True)
