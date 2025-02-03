from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return "AI Health App Backend is Running!"

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    photo = request.files['photo']
    img = Image.open(photo)
    
    # Convert image to numpy array for OpenCV processing
    img_array = np.array(img)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Placeholder for image analysis
    health_tip = analyze_photo(img_cv2)

    return jsonify({"message": "Photo uploaded successfully!", "health_tip": health_tip})

def analyze_photo(image):
    height, width, _ = image.shape
    if height > width:
        return "Your photo looks great! Stay hydrated and keep smiling!"
    else:
        return "Consider adding more veggies to your meals for better health!"

if __name__ == '__main__':
    app.run(debug=True)
