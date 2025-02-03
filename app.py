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
    try:
        if 'photo' not in request.files:
            return jsonify({"error": "No photo uploaded"}), 400

        photo = request.files['photo']
        img = Image.open(photo)

        img_array = np.array(img)
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        brightness = np.mean(img_cv2)
        prompt = f"The brightness level of the uploaded photo is {brightness}. Provide a health tip based on this."

        # Debug print
        print(f"Prompt sent to OpenAI: {prompt}")

        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",  # Using specific model snapshot to avoid access issues
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        health_tip = response.choices[0].message.content.strip()
        return jsonify({"health_tip": health_tip}), 200

    except Exception as e:
        print(f"Error: {e}")  # Detailed error logging
        return jsonify({"error": f"Error getting health tip from AI: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
