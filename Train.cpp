#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <tinyxml2.h>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace tinyxml2;
namespace fs = std::filesystem;

// Variables globales
vector<string> imagePaths;
vector<Rect> boundingBoxes;
Ptr<SIFT> sift = SIFT::create();

// 🔹 Función para leer bounding boxes desde archivos XML (VOC)
Rect parseVOCBoundingBox(const string &xml_path)
{
    XMLDocument doc;
    if (doc.LoadFile(xml_path.c_str()) != XML_SUCCESS)
    {
        cerr << "❌ Error al cargar XML: " << xml_path << endl;
        return Rect();
    }

    XMLElement *annotation = doc.FirstChildElement("annotation");
    if (!annotation)
        return Rect();

    XMLElement *object = annotation->FirstChildElement("object");
    if (!object)
        return Rect();

    XMLElement *bndbox = object->FirstChildElement("bndbox");
    if (!bndbox)
        return Rect();

    int xmin, ymin, xmax, ymax;
    bndbox->FirstChildElement("xmin")->QueryIntText(&xmin);
    bndbox->FirstChildElement("ymin")->QueryIntText(&ymin);
    bndbox->FirstChildElement("xmax")->QueryIntText(&xmax);
    bndbox->FirstChildElement("ymax")->QueryIntText(&ymax);

    return Rect(xmin, ymin, xmax - xmin, ymax - ymin);
}

// 🔹 Función para cargar imágenes y bounding boxes desde el dataset
void loadDataset(const string &dataset_path)
{
    for (const auto &entry : fs::directory_iterator(dataset_path))
    {
        if (entry.path().extension() == ".jpg")
        {
            string image_path = entry.path().string();
            fs::path xmlFilePath = entry.path();
            xmlFilePath.replace_extension(".xml");
            string xml_path = xmlFilePath.string();

            if (fs::exists(xml_path))
            {
                Rect bbox = parseVOCBoundingBox(xml_path);
                if (bbox.width > 0 && bbox.height > 0)
                {
                    imagePaths.push_back(image_path);
                    boundingBoxes.push_back(bbox);
                }
            }
        }
    }

    cout << "✅ Total de imágenes cargadas: " << imagePaths.size() << endl;
}

// 🔹 Función para extraer SIFT y guardar bounding boxes en el archivo YML
void extractAndSaveSIFT(const string &output_file)
{
    FileStorage fs(output_file, FileStorage::WRITE);

    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        Mat img = imread(imagePaths[i], IMREAD_GRAYSCALE);
        if (img.empty())
        {
            cerr << "⚠️ Error al cargar la imagen: " << imagePaths[i] << endl;
            continue;
        }

        // Validar que la imagen tenga un bounding box asignado
        if (i >= boundingBoxes.size())
        {
            cerr << "⚠️ La imagen " << imagePaths[i] << " no tiene bounding box asociado. Se omite." << endl;
            continue;
        }

        Rect bbox = boundingBoxes[i];

        // Ajustar bounding box si está fuera de la imagen
        bbox.x = max(0, min(bbox.x, img.cols - 1));
        bbox.y = max(0, min(bbox.y, img.rows - 1));
        bbox.width = min(bbox.width, img.cols - bbox.x);
        bbox.height = min(bbox.height, img.rows - bbox.y);

        // 🔹 Validar que el bounding box tenga un tamaño mínimo
        if (bbox.width < 20 || bbox.height < 20) // 🔹 Evitar bounding boxes demasiado pequeños
        {
            cerr << "⚠️ Bounding box muy pequeño en la imagen " << imagePaths[i] << ". Se omite." << endl;
            continue;
        }

        // 🔹 Extraer el ROI
        Mat roi = img(bbox);

        // 🔹 Aplicar preprocesamiento para mejorar detección de SIFT
        equalizeHist(roi, roi); // 🔹 Aumentar el contraste

        vector<KeyPoint> keypoints;
        Mat descriptors;
        sift->detectAndCompute(roi, noArray(), keypoints, descriptors);

        if (!descriptors.empty())
        {
            fs << "image_" + to_string(i) << descriptors;
            fs << "bbox_" + to_string(i) << bbox;  // 🔹 Guardamos el bounding box en el YAML
        }
        else
        {
            cerr << "⚠️ No se detectaron descriptores en la imagen " << imagePaths[i] << ". Se omite." << endl;
        }
    }

    fs.release();
    cout << "✅ Descriptores de SIFT y bounding boxes guardados en " << output_file << endl;
}

// 🔹 Función para revisar que los bounding boxes fueron guardados correctamente
void checkBoundingBoxes(const string &output_file)
{
    FileStorage fs(output_file, FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "❌ Error al abrir el archivo de descriptores SIFT." << endl;
        return;
    }

    int i = 0;
    while (true)
    {
        Rect bbox;
        string key_bbox = "bbox_" + to_string(i);

        fs[key_bbox] >> bbox;

        if (bbox.width == 0 || bbox.height == 0)
            break;

        cout << "📏 Bounding Box " << i << ": x=" << bbox.x << ", y=" << bbox.y
             << ", width=" << bbox.width << ", height=" << bbox.height << endl;

        i++;
    }

    fs.release();
}

// 🔹 Main
int main()
{
    string dataset_path = "train/";
    string sift_output_file = "sift_descriptors.yml";

    loadDataset(dataset_path);
    extractAndSaveSIFT(sift_output_file);

    // 🔹 Revisar que los bounding boxes fueron correctamente guardados
    checkBoundingBoxes(sift_output_file);

    return 0;
}
