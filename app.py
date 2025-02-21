from flask import Flask, request, send_file, jsonify
import os
import cv2
import numpy as np
from detector import detectar_objetos

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "El archivo no tiene nombre"}), 400
    
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    # Procesar la imagen con SIFT dentro de la ROI del XML
    detectar_objetos(filename)  # Se procesa pero NO se retorna la imagen

    # Retornar la imagen match_result.jpg generada
    result_path = os.path.join(RESULTS_FOLDER, 'match_result.jpg')

    if not os.path.exists(result_path):
        return jsonify({"error": "No se generó la imagen match_result.jpg"}), 500

    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)