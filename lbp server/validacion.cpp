#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::ml;

//----------------------------------------------------------
// Estructura para almacenar una detección: rectángulo + clase
//----------------------------------------------------------
struct Detection {
    Rect box;
    int label; // 1: 30 km/h, 2: 50 km/h
};

//----------------------------------------------------------
// Función para calcular la imagen LBP a partir de una imagen en escala de grises
//----------------------------------------------------------
Mat computeLBPImage(const Mat &src) {
    if(src.rows < 3 || src.cols < 3) {
        cerr << "La imagen es demasiado pequeña para calcular LBP." << endl;
        return Mat();
    }
    Mat lbp = Mat::zeros(src.rows - 2, src.cols - 2, CV_8UC1);
    for(int i = 1; i < src.rows - 1; i++) {
        for(int j = 1; j < src.cols - 1; j++) {
            uchar center = src.at<uchar>(i, j);
            uchar code = 0;
            code |= (src.at<uchar>(i - 1, j - 1) > center) << 7;
            code |= (src.at<uchar>(i - 1, j    ) > center) << 6;
            code |= (src.at<uchar>(i - 1, j + 1) > center) << 5;
            code |= (src.at<uchar>(i,     j + 1) > center) << 4;
            code |= (src.at<uchar>(i + 1, j + 1) > center) << 3;
            code |= (src.at<uchar>(i + 1, j    ) > center) << 2;
            code |= (src.at<uchar>(i + 1, j - 1) > center) << 1;
            code |= (src.at<uchar>(i,     j - 1) > center) << 0;
            lbp.at<uchar>(i - 1, j - 1) = code;
        }
    }
    return lbp;
}

//----------------------------------------------------------
// Función para calcular el histograma LBP (256 bins)
//----------------------------------------------------------
vector<float> computeLBPHistogram(const Mat &lbpImg) {
    vector<float> hist(256, 0.0f);
    for(int i = 0; i < lbpImg.rows; i++) {
        for(int j = 0; j < lbpImg.cols; j++) {
            int bin = lbpImg.at<uchar>(i, j);
            hist[bin]++;
        }
    }
    // Normalizar el histograma
    float sum = 0.0f;
    for(int i = 0; i < 256; i++) {
        sum += hist[i];
    }
    if(sum > 0.0f) {
        for(int i = 0; i < 256; i++) {
            hist[i] /= sum;
        }
    }
    return hist;
}

//----------------------------------------------------------
// MAIN
//----------------------------------------------------------
int main() {
    // Cargar el clasificador SVM previamente entrenado (3 clases: 0, 1, 2)
    Ptr<SVM> svm = SVM::load("svm_limit.yml");
    if(svm.empty()) {
        cerr << "No se pudo cargar el clasificador SVM desde 'svm_limit.yml'." << endl;
        return -1;
    }

    // Abrir la cámara (índice 0)
    VideoCapture cap(0);
    if(!cap.isOpened()) {
        cerr << "No se pudo abrir la cámara." << endl;
        return -1;
    }
    namedWindow("Detection", WINDOW_AUTOSIZE);

    while(true) {
        Mat frame;
        cap >> frame;
        if(frame.empty()) break;

        // ---------------------------------------------------
        // 1. Convertir a HSV y segmentar el color rojo (aprox.)
        // ---------------------------------------------------
        Mat frameHSV;
        cvtColor(frame, frameHSV, COLOR_BGR2HSV);

        // Rango aproximado para el rojo (dos rangos para cubrir [0..10] y [170..180])
        Mat mask1, mask2;
        inRange(frameHSV, Scalar(0, 70, 70), Scalar(10, 255, 255), mask1);
        inRange(frameHSV, Scalar(170, 70, 70), Scalar(180, 255, 255), mask2);
        Mat maskRed = mask1 | mask2;

        // Operaciones morfológicas para limpiar ruido
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
        morphologyEx(maskRed, maskRed, MORPH_CLOSE, kernel);
        morphologyEx(maskRed, maskRed, MORPH_OPEN, kernel);

        // ---------------------------------------------------
        // 2. Encontrar contornos en la máscara
        // ---------------------------------------------------
        vector<vector<Point>> contours;
        findContours(maskRed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Vector para almacenar detecciones
        vector<Detection> detections;

        for(const auto &contour : contours) {
            Rect candidateRect = boundingRect(contour);
            // Filtrar contornos muy pequeños
            if(candidateRect.area() < 300) continue;

            // Extraer la ROI en escala de grises
            Mat roiGray;
            cvtColor(frame(candidateRect), roiGray, COLOR_BGR2GRAY);
            // Redimensionar a 64x64
            Mat resized;
            resize(roiGray, resized, Size(64,64));

            // Calcular LBP y su histograma
            Mat lbpImg = computeLBPImage(resized);
            if(lbpImg.empty()) continue;
            vector<float> hist = computeLBPHistogram(lbpImg);

            // Convertir el histograma a Mat para el SVM
            Mat featureMat(1, 256, CV_32F);
            for(int i = 0; i < 256; i++) {
                featureMat.at<float>(0, i) = hist[i];
            }

            // Predecir con el SVM (0 -> no señal, 1 -> 30 km/h, 2 -> 50 km/h)
            int response = (int)svm->predict(featureMat);

            if(response == 1 || response == 2) {
                // Guardar la detección
                Detection det;
                det.box = candidateRect;
                det.label = response;
                detections.push_back(det);
            }
        }

        // ---------------------------------------------------
        // 3. Dibujar detecciones
        // ---------------------------------------------------
        for(const auto &d : detections) {
            string text = (d.label == 1) ? "Speed Limit 30 km/h" : "Speed Limit 50 km/h";
            rectangle(frame, d.box, Scalar(0, 255, 0), 2);
            putText(frame, text, Point(d.box.x, d.box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
        }

        imshow("Detection", frame);
        int key = waitKey(30);
        if(key == 27) break; // ESC para salir
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
