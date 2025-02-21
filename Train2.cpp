#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <tinyxml2.h>
#include <filesystem>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
namespace fs = std::filesystem;
using namespace tinyxml2;

int main() {
    string datasetPath = "train";  
    string outputYml = "train_sift_descriptors.yml";

    Ptr<SIFT> sift = SIFT::create();
    FileStorage fsOut(outputYml, FileStorage::WRITE);
    if (!fsOut.isOpened()) {
        cerr << "[ERROR] No se pudo abrir " << outputYml << " para escritura." << endl;
        return -1;
    }

    int descriptorCount = 0;

    for (const auto &entry : fs::directory_iterator(datasetPath)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            string imagePath = entry.path().string();
            fs::path xmlPath = entry.path();
            xmlPath.replace_extension(".xml");

            Mat img = imread(imagePath, IMREAD_COLOR);
            if (img.empty()) {
                cerr << "[ERROR] No se pudo cargar la imagen: " << imagePath << endl;
                continue;
            }

            if (!fs::exists(xmlPath)) {
                cerr << "[ERROR] No existe el XML para la imagen: " << imagePath << endl;
                continue;
            }

            XMLDocument doc;
            if (doc.LoadFile(xmlPath.string().c_str()) != XML_SUCCESS) {
                cerr << "[ERROR] No se pudo cargar el XML: " << xmlPath << endl;
                continue;
            }

            XMLElement* annotation = doc.FirstChildElement("annotation");
            if (!annotation) continue;
            XMLElement* object = annotation->FirstChildElement("object");
            if (!object) continue;
            XMLElement* bndbox = object->FirstChildElement("bndbox");
            if (!bndbox) continue;

            int xmin, ymin, xmax, ymax;
            bndbox->FirstChildElement("xmin")->QueryIntText(&xmin);
            bndbox->FirstChildElement("ymin")->QueryIntText(&ymin);
            bndbox->FirstChildElement("xmax")->QueryIntText(&xmax);
            bndbox->FirstChildElement("ymax")->QueryIntText(&ymax);

            xmin = max(0, min(xmin, img.cols - 1));
            ymin = max(0, min(ymin, img.rows - 1));
            xmax = max(0, min(xmax, img.cols - 1));
            ymax = max(0, min(ymax, img.rows - 1));

            if ((xmax - xmin) < 10 || (ymax - ymin) < 10) {
                cerr << "[ERROR] ROI muy pequeña en " << imagePath << endl;
                continue;
            }

            Rect roi(xmin, ymin, xmax - xmin, ymax - ymin);
            Mat roiImg = img(roi).clone();

            vector<KeyPoint> kp;
            Mat des;
            sift->detectAndCompute(roiImg, noArray(), kp, des);

            if (des.empty()) {
                cerr << "[ERROR] No se detectaron descriptores en " << imagePath << endl;
                continue;
            }

            // Guardar descriptores y keypoints en el archivo
            fsOut << ("descriptor_" + to_string(descriptorCount)) << des;
            fsOut << ("bbox_" + to_string(descriptorCount)) << "[:" << xmin << ymin << xmax << ymax << "]";
            fsOut << ("imagePath_" + to_string(descriptorCount)) << imagePath;

            fsOut << ("keypoints_" + to_string(descriptorCount)) << "[";
            for (const auto& k : kp) {
                fsOut << k.pt.x << k.pt.y;
            }
            fsOut << "]";

            cout << "[DEBUG] Descriptor " << descriptorCount << " guardado con " << des.rows << " x " << des.cols << " y " << kp.size() << " keypoints." << endl;
            descriptorCount++;
        }
    }

    fsOut.release();
    cout << "[INFO] Se guardaron " << descriptorCount << " descriptores en " << outputYml << endl;
    return 0;
}
