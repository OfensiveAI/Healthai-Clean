from flask import Flask, request, jsonify
from google.cloud import vision
import os

app = Flask(__name__)

# Set the environment variable for the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ai-health-app-key.json"

# Initialize the Google Vision client
client = vision.ImageAnnotatorClient()

@app.route('/')
def home():
    return "Welcome to the AI Health App!"

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    content = image_file.read()

    # Use Google Vision API to analyze the image
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = [label.description for label in response.label_annotations]

    return jsonify({'labels': labels})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
