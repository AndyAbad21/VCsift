
// Librería base para flujos de entrada y salida 
#include <iostream> 
// Librería para el manejo de memoria, búsqueda, ordenamiento, etc.
#include <cstdlib>
// Librería para el manejo de cadenas de texto
#include <cstring>
// Librería para el manejo de arreglos dinámicos (lista ligada)
#include <vector>
// Librería para manejo de flujos de caracteres
#include <sstream>
// Librería para manejo de carpetas y ficheros
#include <filesystem>

#include <cmath>

#include <random>

// Librería para leer y escribir archivos de texto
#include <fstream>

// Librerías de OpenCV
#include <opencv2/core/core.hpp> // Funciones base (representación de matrices, operaciones, etc.)
#include <opencv2/highgui/highgui.hpp> // Funciones de interfaz gráfica
#include <opencv2/imgcodecs/imgcodecs.hpp> // Cargar y manipular imágenes en distintos formatos gráficos
#include <opencv2/imgproc/imgproc.hpp> // Operaciones de procesamiento sobre imágenes
#include <opencv2/video/video.hpp> // Manejo de vídeo
#include <opencv2/videoio/videoio.hpp> // Lectura y escritura de vídeo
#include <opencv2/features2d/features2d.hpp>  // Librería que incluye el método SIFT

#include <opencv2/xfeatures2d/nonfree.hpp>

//#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv; // Espacio de nombres de OpenCV

int main(int argc, char *argv[]){

    VideoCapture video("/dev/video0");
                           //VideoCapture video("/home/video.mp4");
    if(video.isOpened()){
        namedWindow("Video", WINDOW_AUTOSIZE);
        namedWindow("KeyPoints", WINDOW_AUTOSIZE);

        Mat frame;
        Mat frameKeyPoints;
        Mat logoKeyPoints;
        Mat frameResultado;
        Mat logo = imread("logoCatedra2025.jpg");

        

        Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
        
        // Key Points del vídeo y del logo
        vector<KeyPoint> keyPoints;
        vector<KeyPoint> keyPointsLogo;

        // Descriptores del vídeo y del logo
        Mat descriptorVideo, descriptorLogo;

        while(3==3){
            video >> frame;
            //flip(frame, frame, 1);
            resize(frame, frame, Size(), 0.7, 0.7);

            // Detección de los KeyPoints
            detector->detect(frame, keyPoints);
            detector->detect(logo, keyPointsLogo);

            // Cálculo del descriptor
            detector->compute(frame, keyPoints,descriptorVideo);
            detector->compute(logo, keyPointsLogo, descriptorLogo);

            // Búsqueda del logo en el vídeo usando el BFMatcher
            BFMatcher matcher;
            Mat img_matches;

            // Vector de coincidencias (matches)
            vector<vector<DMatch> > matches;
            matcher.knnMatch(descriptorLogo, descriptorVideo, matches, 2);

            // Matches o coincidencias que cumplen con el valor del umbral propuesto por el 
            // Prof. David Lowe
            vector<DMatch> matchesFiltrados;
            float ratio = 0.67;
            for(int i=0;i<matches.size();i++){
                if(matches[i][0].distance < ratio*matches[i][1].distance){
                    matchesFiltrados.push_back(matches[i][0]);
                }
            }
            cout << "Matches => Sin Filtrar = " << matches.size() << " Filtrados = " << matchesFiltrados.size() << endl;



            frameKeyPoints = frame.clone();

            drawKeypoints(frame, keyPoints, frameKeyPoints);
            drawKeypoints(logo, keyPointsLogo, logoKeyPoints);

            if(matchesFiltrados.size()>50){
                drawMatches(logo, keyPointsLogo, frame, keyPoints, matchesFiltrados, img_matches);
                imshow("Matches", img_matches);
            }

            imshow("Video", frame);            
            imshow("KeyPoints", frameKeyPoints);
            imshow("KeyPointsLogo", logoKeyPoints);
            

            if(waitKey(23)==27)
                break;
        }

        video.release();
        destroyAllWindows();
    }

    return 0;
}