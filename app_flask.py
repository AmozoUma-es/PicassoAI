# coding=utf-8

from flask import Flask, request, jsonify
from flask_cors import CORS
from process import process_image, process_text, process_image_text
from PIL import Image
from io import BytesIO
import numpy as np

app = Flask(__name__)

CORS(app)

@app.route('/process-image', methods=['POST'])
def request_image():
    if 'image' not in request.files:
        return jsonify({"message": "Imagen no enviada", "resultado": ""})
    image_file = request.files['image']
    
    image_data = image_file.read()

    result = process_image(np.array(Image.open(BytesIO(image_data))))
    return jsonify({"message": "Imagen procesada", "items": result})

@app.route('/process-text', methods=['POST'])
def request_text():
    input_text = request.json.get('q')
    result = process_text(input_text)
    print(result)
    return jsonify({"message": "texto procesado", "items": result})

@app.route('/process-text-image', methods=['POST'])
def request_text_image():
    if 'image' not in request.files:
        return jsonify({"message": "Imagen no enviada", "resultado": ""})
    input_text = request.form.get('q')

    image_file = request.files['image']
    image_data = image_file.read()

    result = process_image_text(input_text, np.array(Image.open(BytesIO(image_data))))
    sorted_value = sorted(result, key=lambda x:x[1], reverse=True)
    sorted_res = dict(sorted_value)
    tops = list(sorted_res.items())
    return jsonify({"message": "texto vs imagen procesado", "items": tops})

if __name__ == '__main__':
    # para producción
    # app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)
