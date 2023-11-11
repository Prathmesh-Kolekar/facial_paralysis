from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the model (you need to adjust the path to your model)
model = load_model('my_model.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        image_file = request.files['image']

        # Read and preprocess the image
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized / 255.0  # Normalize the image
 
        # Make the prediction
        predicted_prob = model.predict(np.array([image_normalized]))[0][0]
        threshold = 0.5
        predicted_class = 1 if predicted_prob >= threshold else 0

        # Close the image file
        image_file.close()

        # Define the response
        response = {
            'predicted_class': predicted_class,
            'predicted_prob': float(predicted_prob)
        }
        
        return jsonify(response), 200
    except Exception as e:
        return str(e), 500
    
if __name__ == '__main__':
    app.run(debug=True)
