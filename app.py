import os
from google.cloud import vision
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Write the service account JSON from the environment variable to a temporary file
# New Code (Fix for Render Secret File)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/SERVICE_ACCOUNT_JSON"

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

@app.route('/')
def home():
    return "AI Health App is running!"

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)

    labels = [label.description for label in response.label_annotations]
    return jsonify({'labels': labels})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
