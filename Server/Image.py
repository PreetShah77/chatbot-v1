
import os
from flask import Flask, request, jsonify
from PIL import Image
import io
import google.generativeai as genai

from flask_cors import CORS
# ... (other imports)

app = Flask(__name__)
CORS(app)  # Enable CORS

# Importing the model
genai.configure(api_key="AIzaSyBcGPKbHKNpayc-yKWm1whEYCn3amT7O98")
model = genai.GenerativeModel('gemini-pro-vision')

# Function to get the response from the model
def get_data(input_prompt, image_data):
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

# Function to process the uploaded image
def image_process(file):
    if file:
        bytes_data = file.read()
        image_parts = [
            {
                "mime_type": file.content_type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("Check the file is uploaded properly")

# Server endpoint to handle the image request
@app.route('/image_description', methods=['POST'])
def image_description():
    input_prompt = """You are an expert in analysing the image. User will upload any kind of image, and you need to answer questions about the image from the image alone. You can also find faces in the image and tell their names if you have the information."""
    input_text = "Write information about the uploaded image and write in detail about its contents."

    file = request.files.get('image')
    image_data = image_process(file)

    response = get_data(input_prompt, image_data)

    return jsonify({'description': response})

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=6942)
