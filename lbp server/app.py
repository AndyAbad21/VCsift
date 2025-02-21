from flask import Flask, request, jsonify, send_file, Response
import cv2
import numpy as np
import os

app = Flask(__name__)

# Cargar el clasificador SVM previamente entrenado
SVM_PATH = "svm_limit.yml"
if not os.path.exists(SVM_PATH):
    raise FileNotFoundError(f"No se encontró el archivo {SVM_PATH}")

svm = cv2.ml.SVM_load(SVM_PATH)
if svm.empty():
    raise ValueError("El modelo SVM no se pudo cargar correctamente.")

# Función para calcular la imagen LBP
def computeLBPImage(src):
    lbp = np.zeros((src.shape[0] - 2, src.shape[1] - 2), dtype=np.uint8)
    for i in range(1, src.shape[0] - 1):
        for j in range(1, src.shape[1] - 1):
            center = src[i, j]
            code = (
                (src[i-1, j-1] > center) << 7 |
                (src[i-1, j] > center) << 6 |
                (src[i-1, j+1] > center) << 5 |
                (src[i, j+1] > center) << 4 |
                (src[i+1, j+1] > center) << 3 |
                (src[i+1, j] > center) << 2 |
                (src[i+1, j-1] > center) << 1 |
                (src[i, j-1] > center)
            )
            lbp[i-1, j-1] = code
    return lbp

# Función para calcular el histograma LBP
def computeLBPHistogram(lbpImg):
    hist = np.histogram(lbpImg.ravel(), bins=256, range=(0, 256))[0].astype(np.float32)
    hist /= (hist.sum() + 1e-6)  # Normalización
    return hist

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    temp_path = "temp.jpg"
    image_file.save(temp_path)

    # Cargar imagen
    image = cv2.imread(temp_path)
    if image is None:
        os.remove(temp_path)
        return jsonify({'error': 'Invalid image format'}), 400

    # Convertir a HSV y segmentar el color rojo
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 70, 70), (180, 255, 255))
    mask = mask1 | mask2

    # Procesamiento morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 300:  # Filtrar áreas pequeñas
            continue

        roi_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(roi_gray, (64, 64))
        lbp = computeLBPImage(roi_gray)
        hist = computeLBPHistogram(lbp).reshape(1, -1)

        # Predicción con SVM
        _, response = svm.predict(hist)
        label = int(response[0, 0])

        # Dibujar detección
        text = "30 km/h" if label == 1 else "50 km/h"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Guardar la imagen procesada
    processed_path = "processed.jpg"
    cv2.imwrite(processed_path, image)

    # Eliminar imagen temporal
    os.remove(temp_path)

    # Leer la imagen procesada y devolverla como respuesta
    with open(processed_path, "rb") as img_file:
        img_bytes = img_file.read()
    
    os.remove(processed_path)  # Eliminar después de enviar

    response = Response(img_bytes, mimetype="image/jpeg")
    response.headers["Content-Disposition"] = "attachment; filename=processed.jpg"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    
    return response

@app.route('/')
def index():
    return """
    <h1>Detección de Señales</h1>
    <p>Envía una imagen a <code>/detect</code> para recibir la imagen procesada.</p>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
