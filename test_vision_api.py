from google.cloud import vision

# Initialize the Vision client
client = vision.ImageAnnotatorClient()

# Provide a sample image URL
image = vision.Image(source={"image_uri": "https://cloud.google.com/images/products/ai/ai-products-icon.svg"})

# Perform label detection
response = client.label_detection(image=image)

# Print the labels
for label in response.label_annotations:
    print(label.description)
