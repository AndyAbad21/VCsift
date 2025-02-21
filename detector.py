import cv2
import numpy as np
import os
import pytinyxml2 as tx

# Carpeta con imágenes de referencia
REFERENCE_FOLDER = 'letreros_images'
reference_images = []
reference_names = []
ref_descriptors = []
ref_keypoints = []

# Inicializar SIFT
sift = cv2.SIFT_create(nfeatures=500)

# FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
search_params = dict(checks=30)
flann = cv2.FlannBasedMatcher(index_params, search_params)


def leer_roi_desde_xml(image_path):
    """Extrae la ROI desde el XML asociado a la imagen de referencia."""
    xml_path = os.path.splitext(image_path)[0] + ".xml"

    if not os.path.exists(xml_path):
        print(f"[ERROR] No se encontró el archivo XML para {image_path}")
        return None

    doc = tx.XMLDocument()
    if doc.LoadFile(xml_path) != tx.XML_SUCCESS:
        print(f"[ERROR] No se pudo cargar el XML: {xml_path}")
        return None

    annotation = doc.FirstChildElement("annotation")
    if not annotation:
        return None
    object_ = annotation.FirstChildElement("object")
    if not object_:
        return None
    bndbox = object_.FirstChildElement("bndbox")
    if not bndbox:
        return None

    # Extraer coordenadas de la ROI
    xmin = bndbox.FirstChildElement("xmin").IntText()
    ymin = bndbox.FirstChildElement("ymin").IntText()
    xmax = bndbox.FirstChildElement("xmax").IntText()
    ymax = bndbox.FirstChildElement("ymax").IntText()

    return (xmin, ymin, xmax, ymax)


# Cargar las imágenes de referencia y extraer sus descriptores
for filename in os.listdir(REFERENCE_FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(REFERENCE_FOLDER, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Obtener la ROI desde el XML
        roi_coords = leer_roi_desde_xml(image_path)
        if roi_coords is None:
            continue

        xmin, ymin, xmax, ymax = roi_coords
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(xmax, img.shape[1]), min(ymax, img.shape[0])

        if (xmax - xmin) < 10 or (ymax - ymin) < 10:
            print(f"[ERROR] ROI muy pequeña en {image_path}")
            continue

        # Recortar la ROI
        roi_img = img[ymin:ymax, xmin:xmax]

        # Extraer descriptores y keypoints de la ROI
        kp, des = sift.detectAndCompute(roi_img, None)

        if des is not None:
            reference_images.append(img)
            reference_names.append(filename)
            ref_keypoints.append(kp)
            ref_descriptors.append(des)

print(f"[INFO] Se cargaron {len(ref_descriptors)} imágenes de referencia con sus descriptores.")


def detectar_contorno_rojo(img):
    """Convierte la imagen a HSV y detecta el contorno rojo del letrero."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definir rangos de color para rojo en HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Crear máscara para el color rojo
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    # Encontrar contornos en la máscara roja
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_red:
        # Encontrar el contorno más grande (asumiendo que es el del letrero)
        largest_contour = max(contours_red, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Dibujar un rectángulo alrededor del contorno rojo
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Letrero", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"[INFO] Contorno rojo detectado en ({x}, {y}), tamaño ({w}, {h})")

    return img

def detectar_objetos(image_path):
    """Detecta objetos en una imagen de prueba y muestra los keypoints coincidentes con la imagen de referencia."""
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[ERROR] No se pudo cargar la imagen de prueba: {image_path}")
        return None

    # Detectar características en la imagen de prueba
    kp, des = sift.detectAndCompute(img, None)
    if des is None:
        print(f"[ERROR] No se encontraron descriptores en la imagen de prueba: {image_path}")
        return img

    best_match = None
    max_matches = 0
    best_kp = []
    best_matches = []

    # Comparar con imágenes de referencia
    for i, ref_des in enumerate(ref_descriptors):
        if ref_des is None or des is None:
            continue

        matches = flann.knnMatch(ref_des, des, k=2)

        # Filtrar buenos matches usando la razón de Lowe
        good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

        if len(good_matches) > max_matches:
            max_matches = len(good_matches)
            best_match = i
            best_kp = ref_keypoints[i]
            best_matches = good_matches

    # Mostrar keypoints y matches en una sola imagen
    if best_match is not None and len(best_matches) > 10:
        img_reference = reference_images[best_match]
        kp_reference = ref_keypoints[best_match]

        # Dibujar los keypoints en la imagen de referencia
        ref_with_kp = cv2.drawKeypoints(img_reference, kp_reference, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Aplicar detección de contorno rojo en la imagen original
        img_original = cv2.imread(image_path)
        img_con_contorno = detectar_contorno_rojo(img_original)

        # Redimensionar ambas imágenes para que tengan la misma altura
        height = max(ref_with_kp.shape[0], img_con_contorno.shape[0])
        width_ref = ref_with_kp.shape[1]
        width_test = img_con_contorno.shape[1]

        ref_with_kp = cv2.resize(ref_with_kp, (width_ref, height))
        img_con_contorno = cv2.resize(img_con_contorno, (width_test, height))

        # Concatenar ambas imágenes horizontalmente (lado a lado)
        resultado_final = np.hstack((ref_with_kp, img_con_contorno))

        # Guardar la imagen final con detección
        result_path = "static/results/match_result.jpg"
        cv2.imwrite(result_path, resultado_final)
        print(f"[INFO] Imagen guardada con detección en: {result_path}")

        return resultado_final

    return img
