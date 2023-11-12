import traceback
from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
import cv2
import numpy as np
import tflite

app = Flask(__name__)

with open('api/my_model.tflite', 'rb') as f:
    model_content = f.read()

interpreter = tflite.Interpreter(model_content=model_content)

# Use the interpreter to perform inference

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
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], image_normalized.reshape((1, 224, 224, 3)).astype(np.float32))
        interpreter.invoke()

        predicted_prob = interpreter.get_tensor(output_details[0]['index'])
        threshold = 0.5
        predicted_class = 1 if predicted_prob >= threshold else 0

        print(predicted_class,predicted_prob)

        # Close the image file
        image_file.close()

        # Define the response
        response = {
            'predicted_class': predicted_class,
            'predicted_prob': float(predicted_prob)
        }

        return jsonify(response), 200
    except Exception as e:

        traceback.print_exc()

        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
