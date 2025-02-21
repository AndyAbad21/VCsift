#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Variables globales
vector<Mat> dataset_descriptors;
vector<Rect> dataset_bboxes;
Ptr<SIFT> sift = SIFT::create();

// 🔹 Cargar los descriptores SIFT y bounding boxes desde YAML
void loadSIFTDescriptors(const string &filename)
{
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "Error al abrir el archivo de descriptores SIFT." << endl;
        return;
    }

    int i = 0;
    while (true)
    {
        Mat descriptors;
        Rect bbox;
        string key = "image_" + to_string(i);
        fs[key] >> descriptors;
        fs["bbox_" + to_string(i)] >> bbox;

        if (descriptors.empty())
            break;

        dataset_descriptors.push_back(descriptors);
        dataset_bboxes.push_back(bbox);
        i++;
    }

    fs.release();
    cout << "✅ Descriptores de SIFT y bounding boxes cargados desde " << filename << " (" << dataset_descriptors.size() << " imágenes, " << dataset_bboxes.size() << " bounding boxes)" << endl;
}

// 🔹 Detección del ROI basado en la densidad de matches SIFT
Rect detectROIFromMatches(const vector<DMatch> &matches, const vector<KeyPoint> &keypoints)
{
    if (matches.empty())
        return Rect();

    vector<Point2f> matched_points;
    for (const auto &match : matches)
    {
        matched_points.push_back(keypoints[match.trainIdx].pt);
    }

    // 🔹 Encontrar un rectángulo que encierre la mayoría de los keypoints coincidentes
    return boundingRect(matched_points);
}

// 🔹 Proceso de prueba con detección de SIFT
void processTestImage(const string &image_path)
{
    Mat img = imread(image_path, IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cerr << "Error al cargar la imagen de test: " << image_path << endl;
        return;
    }

    cout << "🔍 Procesando imagen de test: " << image_path << " Tamaño: " << img.cols << "x" << img.rows << endl;

    // 🔹 Extraer descriptores SIFT de la imagen de test
    vector<KeyPoint> keypoints_test;
    Mat descriptors_test;
    sift->detectAndCompute(img, noArray(), keypoints_test, descriptors_test);

    if (descriptors_test.empty())
    {
        cerr << "⚠️ No se detectaron descriptores en la imagen de test." << endl;
        return;
    }

    cout << "✅ Descriptores detectados en la imagen de test: " << descriptors_test.rows << endl;

    // 🔹 Comparación con el dataset
    FlannBasedMatcher matcher;
    int best_match_idx = -1;
    vector<DMatch> best_matches;
    int max_matches = 0;

    for (size_t i = 0; i < dataset_descriptors.size(); ++i)
    {
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(dataset_descriptors[i], descriptors_test, knn_matches, 2);

        vector<DMatch> good_matches;
        for (const auto &m : knn_matches)
        {
            if (m[0].distance < 0.75 * m[1].distance)
                good_matches.push_back(m[0]);
        }

        if (good_matches.size() > max_matches)
        {
            max_matches = good_matches.size();
            best_match_idx = i;
            best_matches = good_matches;
        }
    }

    if (best_match_idx == -1)
    {
        cerr << "⚠️ No se encontró un match significativo con el dataset." << endl;
        return;
    }

    cout << "🎯 Mejor match con imagen " << best_match_idx << " con " << max_matches << " coincidencias." << endl;

    // 🔹 Obtener bounding box basado en los keypoints coincidentes
    Rect roi = detectROIFromMatches(best_matches, keypoints_test);

    if (roi.area() == 0)
    {
        cerr << "⚠️ No se encontró un ROI confiable con SIFT." << endl;
        return;
    }

    // 🔹 Dibujar el ROI en la imagen original
    Mat img_result;
    cvtColor(img, img_result, COLOR_GRAY2BGR);
    rectangle(img_result, roi, Scalar(0, 255, 0), 2); // Verde para ROI basado en matches

    imshow("1. Keypoints Detectados en la Imagen", img_result);
    waitKey(0);

    destroyAllWindows();
}

// Main
int main()
{
    string sift_file = "sift_descriptors.yml"; // Archivo con los descriptores guardados
    string test_folder = "test/"; // Carpeta donde están las imágenes de prueba

    // 🔹 Cargar los descriptores SIFT del dataset
    loadSIFTDescriptors(sift_file);

    // 🔹 Iterar sobre todas las imágenes en la carpeta de test
    for (const auto &entry : fs::directory_iterator(test_folder))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
        {
            string test_image = entry.path().string();
            processTestImage(test_image); // Probar imagen
        }
    }

    return 0;
}
