from flask import Flask, request, jsonify
import requests
import numpy as np
import cv2
from pymongo import MongoClient
import gridfs
from recognize import load_known_faces

app = Flask(__name__)

# MongoDB configuration
MONGO_URI = 'mongodb+srv://alimalik:ALIMALIKALIMALIK1234@cluster0.xsf9mmu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
client = MongoClient(MONGO_URI)
db = client['FaceDB']
fs = gridfs.GridFS(db)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        image_url = request.form['image_url']
        name = request.form['name']

        if image_url and name:
            response = requests.get(image_url)
            if response.status_code != 200:
                return jsonify({'error': 'Image could not be fetched.'}), 400

            # Convert the image data to a numpy array
            np_img = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({'error': 'Invalid image data'}), 400

            # Save the image to MongoDB
            filename = f"{name}.jpg"
            file_id = fs.put(response.content, filename=filename)

            # Reload known faces
            load_known_faces()

            return jsonify({'message': 'Image uploaded successfully.', 'id': str(file_id)}), 200
        else:
            return jsonify({'error': 'Invalid input'}), 400

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
