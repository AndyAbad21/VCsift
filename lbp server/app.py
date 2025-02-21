from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

# Endpoint para la detección
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    temp_path = 'temp.jpg'
    image_file.save(temp_path)

    try:
        # Llamamos al ejecutable detect_sign pasando la imagen temporal como argumento
        result = subprocess.run(['./detect_sign', temp_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=10)
        if result.returncode != 0:
            return jsonify({'error': result.stderr.strip()}), 500
        output = result.stdout.strip()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Borrar la imagen temporal
    os.remove(temp_path)

    return jsonify({'result': output})

# Opción simple para una página de prueba (sin usar index.html)
@app.route('/')
def index():
    return """
    <h1>Detección de Señales</h1>
    <p>Utiliza el endpoint <code>/detect</code> para enviar una imagen (campo 'image').</p>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
