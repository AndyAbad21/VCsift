#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
namespace fs = std::filesystem;

// Carpeta con imágenes de referencia
string REFERENCE_FOLDER = "reference_images";
vector<Mat> reference_images;
vector<string> reference_names;
vector<vector<KeyPoint>> ref_keypoints;
vector<Mat> ref_descriptors;

// Inicializar SIFT y FLANN Matcher
Ptr<SIFT> sift = SIFT::create();
Ptr<FlannBasedMatcher> flannMatcher = FlannBasedMatcher::create();

void cargarReferencias() {
    for (const auto& entry : fs::directory_iterator(REFERENCE_FOLDER)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            string imagePath = entry.path().string();
            Mat img = imread(imagePath, IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            vector<KeyPoint> kp;
            Mat des;
            sift->detectAndCompute(img, noArray(), kp, des);

            if (!des.empty()) {
                reference_images.push_back(img);
                reference_names.push_back(imagePath);
                ref_keypoints.push_back(kp);
                ref_descriptors.push_back(des);
            }
        }
    }
    cout << "[INFO] Se cargaron " << ref_descriptors.size() << " imágenes de referencia con descriptores." << endl;
}

void detectarObjetos(Mat& frame) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    vector<KeyPoint> kp;
    Mat des;
    sift->detectAndCompute(gray, noArray(), kp, des);

    if (des.empty()) {
        cout << "[ERROR] No se encontraron descriptores en el frame." << endl;
        return;
    }

    int best_match = -1;
    vector<DMatch> best_matches;
    int max_matches = 0;

    for (size_t i = 0; i < ref_descriptors.size(); i++) {
        vector<vector<DMatch>> matches;
        flannMatcher->knnMatch(ref_descriptors[i], des, matches, 2);


        vector<DMatch> good_matches;
        for (const auto& m : matches) {
            if (m[0].distance < 0.6 * m[1].distance) { // Filtro de Lowe (ajustado a 0.6)
                good_matches.push_back(m[0]);
            }
        }

        if (good_matches.size() > max_matches) {
            max_matches = good_matches.size();
            best_match = i;
            best_matches = good_matches;
        }
    }

    if (best_match != -1 && best_matches.size() > 30) { // Se requieren al menos 30 matches
        Mat match_img;
        drawMatches(reference_images[best_match], ref_keypoints[best_match], frame, kp, best_matches, match_img,
                    Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        imshow("Matches", match_img);
    }
}

int main() {
    cargarReferencias();

    VideoCapture cap(0);  // Abre la cámara (0 para cámara predeterminada)
    if (!cap.isOpened()) {
        cerr << "[ERROR] No se pudo abrir la cámara." << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        detectarObjetos(frame);
        imshow("Cámara", frame);

        if (waitKey(1) == 27) break; // Presionar 'ESC' para salir
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
