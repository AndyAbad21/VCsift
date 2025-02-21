all:
	g++ Test.cpp -std=c++17 \
-I/home/andy/aplicaciones/librerias/opencv/opencvi/include/opencv4/ \
-I/home/andy/aplicaciones/librerias/tinyxml2/ \
-L/home/andy/aplicaciones/librerias/opencv/opencvi/lib \
-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
-lopencv_video -lopencv_videoio -lopencv_features2d -lopencv_xfeatures2d \
-lopencv_flann -lopencv_calib3d -ltinyxml2 -o vision.bin -lstdc++fs

run:
	./vision.bin

clean:
	rm -f vision.bin
