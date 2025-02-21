#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <filesystem>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
namespace fs = std::filesystem;

// Estructura para almacenar los ROIs entrenados
struct TrainROI {
    Mat descriptors;
    vector<Point2f> kpCoords;
    Rect bbox;
};

// Función para cargar los descriptores desde el archivo YML
vector<TrainROI> loadTrainDescriptors(const string &trainYml) {
    FileStorage fsIn(trainYml, FileStorage::READ);
    vector<TrainROI> trainROIs;

    if (!fsIn.isOpened()) {
        cerr << "[ERROR] No se pudo abrir " << trainYml << endl;
        return trainROIs;
    }

    int i = 0;
    while (true) {
        string descriptorKey = "descriptor_" + to_string(i);
        FileNode descNode = fsIn[descriptorKey];
        if (descNode.empty()) break;

        Mat des;
        descNode >> des;

        string bboxKey = "bbox_" + to_string(i);
        FileNode bboxNode = fsIn[bboxKey];
        Rect roiRect(0, 0, 0, 0);
        if (!bboxNode.empty() && bboxNode.size() == 4) {
            roiRect = Rect((int)bboxNode[0], (int)bboxNode[1], 
                           (int)bboxNode[2] - (int)bboxNode[0], 
                           (int)bboxNode[3] - (int)bboxNode[1]);
        }

        string kpKey = "keypoints_" + to_string(i);
        FileNode kpNode = fsIn[kpKey];
        vector<Point2f> roiKpCoords;
        for (FileNodeIterator it = kpNode.begin(); it != kpNode.end(); ++it) {
            float x = (float)(*it); ++it;
            float y = (float)(*it);
            roiKpCoords.push_back(Point2f(x, y));
        }

        trainROIs.push_back({des, roiKpCoords, roiRect});
        i++;
    }
    fsIn.release();
    cout << "[INFO] Se cargaron " << trainROIs.size() << " ROIs del dataset." << endl;
    return trainROIs;
}

int main() {
    string trainYml = "train_sift_descriptors.yml";
    vector<TrainROI> trainROIs = loadTrainDescriptors(trainYml);
    if (trainROIs.empty()) {
        cerr << "[ERROR] No se encontraron descriptores de entrenamiento." << endl;
        return -1;
    }

    Ptr<SIFT> sift = SIFT::create();
    BFMatcher matcher(NORM_L2);
    float ratioThresh = 0.75f;
    int minGoodMatches = 10;
    double ransacReprojThresh = 5.0;

    string testFolder = "test";

    for (const auto &entry : fs::directory_iterator(testFolder)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            string testImagePath = entry.path().string();
            cout << "[DEBUG] Procesando imagen de test: " << testImagePath << endl;

            Mat testImg = imread(testImagePath, IMREAD_COLOR);
            if (testImg.empty()) {
                cerr << "[ERROR] No se pudo cargar la imagen de test." << endl;
                continue;
            }

            Mat testGray;
            cvtColor(testImg, testGray, COLOR_BGR2GRAY);
            vector<KeyPoint> testKp;
            Mat testDes;
            sift->detectAndCompute(testGray, noArray(), testKp, testDes);

            if (testDes.empty()) {
                cerr << "[ERROR] No se detectaron descriptores en la imagen de test." << endl;
                continue;
            }
            cout << "[INFO] Se detectaron " << testKp.size() << " keypoints en la imagen de prueba." << endl;

            Mat frameMatches = testImg.clone();

            for (size_t idx = 0; idx < trainROIs.size(); idx++) {
                const TrainROI &troi = trainROIs[idx];
                if (troi.descriptors.empty() || troi.kpCoords.empty()) {
                    cout << "[DEBUG] ROI " << idx << " - Sin descriptores ni keypoints." << endl;
                    continue;
                }

                vector<vector<DMatch>> knnMatches;
                matcher.knnMatch(troi.descriptors, testDes, knnMatches, 2);
                cout << "[DEBUG] Número de coincidencias encontradas: " << knnMatches.size() << endl;

                vector<DMatch> goodMatches;
                for (auto &km : knnMatches) {
                    if (km.size() < 2) continue;
                    if (km[0].distance < ratioThresh * km[1].distance) {
                        goodMatches.push_back(km[0]);
                    }
                }

                cout << "[DEBUG] ROI " << idx << " - Good matches: " << goodMatches.size() << endl;

                if ((int)goodMatches.size() >= minGoodMatches) {
                    vector<Point2f> roiPoints, testPoints;
                    for (auto &gm : goodMatches) {
                        roiPoints.push_back(troi.kpCoords[gm.queryIdx]);
                        testPoints.push_back(testKp[gm.trainIdx].pt);
                    }

                    Mat maskInliers;
                    Mat H = findHomography(roiPoints, testPoints, RANSAC, ransacReprojThresh, maskInliers);
                    if (!H.empty() && !maskInliers.empty()) {
                        int inliersCount = countNonZero(maskInliers);
                        cout << "[DEBUG] ROI " << idx << " - Inliers: " << inliersCount << endl;

                        if (inliersCount >= 8) {
                            vector<Point2f> corners = {
                                Point2f(0, 0), Point2f((float)troi.bbox.width, 0),
                                Point2f((float)troi.bbox.width, (float)troi.bbox.height),
                                Point2f(0, (float)troi.bbox.height)
                            };
                            vector<Point2f> cornersTransformed(4);
                            perspectiveTransform(corners, cornersTransformed, H);

                            bool valid = true;
                            for (const auto &pt : cornersTransformed) {
                                if (pt.x < 0 || pt.y < 0 || pt.x >= testImg.cols || pt.y >= testImg.rows || isnan(pt.x) || isnan(pt.y)) {
                                    valid = false;
                                    break;
                                }
                            }

                            if (valid) {
                                vector<vector<Point>> polygon(1);
                                for (const auto &pt : cornersTransformed) {
                                    polygon[0].push_back(Point((int)pt.x, (int)pt.y));
                                }
                                polylines(frameMatches, polygon, true, Scalar(0, 255, 0), 2, LINE_AA);
                            } else {
                                cout << "[WARNING] ROI " << idx << " - Transformación no válida, no se dibujará el rectángulo." << endl;
                            }
                        }
                    } else {
                        cout << "[DEBUG] ROI " << idx << " - Homografía no encontrada." << endl;
                    }
                }
            }

            imshow("Matches", frameMatches);
            waitKey(0);
        }
    }

    return 0;
}
