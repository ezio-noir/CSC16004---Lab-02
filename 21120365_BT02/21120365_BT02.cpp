#include "image_displayer.h"
#include "edge_detection.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <unordered_map>

using namespace std;
using namespace cv;

// Global object that stores images to display
ImageDisplayer displayer;

// Print program usage
void printHelp() {
	cout << R"(
Usage: <program> <image_path> <algorithm> [algorithm_parameters]
Algorithm:
	--sobel  , -s  : Sobel
	--prewitt, -p  : Prewitt
	--laplace, -l  : Laplace
	--canny  , -c  : Canny
Parameters: --params:param1=value1,param2=value2,...
	blur_ksize     : (default: none) size of Gaussian fiter size (if not specified or set to 1, blurring is not applied)
	blur_sigma     : (default: 1.0) filter sigma
	laplace_kernel : (Laplace, default: 0) choose Laplace discrete kernel
					 0: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
					 1: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
	low            : (Canny, default: 80 or high / 3) low threshold
	high           : (Canny, default: 255 or low * 3) high threshold
Example:
	.\21120365_BT02.exe ".\lena.png" --canny --params:blur_ksize=3,low=80,high=255
	)" << endl;
}

// Parse the string of format: --params:param1=val1;params2=val2;... into algorithm parameters
void parseParams(string str, unordered_map<string, double> &params) {
	if (str.substr(0, 9).compare("--params:") == 0) {
		params.clear();
		string strToParse = str.substr(9);
		vector<string> tokens;

		// Loop through the string to parse pairs of param=val
		int offset = 0, semicolonPos;
		do {
			semicolonPos = strToParse.find(",", offset);
			tokens.push_back(strToParse.substr(offset, semicolonPos - offset));
			offset = semicolonPos + 1;
		} while (semicolonPos != strToParse.npos);

		// Parse each pair of param=val and store it into the unordered map
		string param;
		double value;
		for (const auto& token : tokens) {
			param = token.substr(0, token.find("="));
			value = stod(token.substr(param.length() + 1));
			params[param] = value;
		}
	}
}

int main(int argc, char* argv[]) {
	if (argc < 3 || argc > 4) {
		printHelp();
		return -1;
	}

	// Read original image
	Mat originalSrc = imread(argv[1], IMREAD_ANYCOLOR);
	if (originalSrc.empty()) {
		cerr << "Unable to load image " << argv[1] << endl;
	}
	// Read original image into grayscale
	Mat src = imread(argv[1], IMREAD_GRAYSCALE);
	Mat dst;
	displayer.addImage("Original image", originalSrc);
	displayer.addImage("Original (grayscale)", src);

	// An unordered map to store parameter value for algorithm
	unordered_map<string, double> params;

	// Parse the `--params:param1=val1,...` if it is specified by user
	if (argc == 4) parseParams(argv[3], params);

	// Match command to algorithm
	if (strcmp(argv[2], "--sobel") == 0 || strcmp(argv[2], "-s") == 0) {
		detectBySobel(src, dst, params);
	}
	else if (strcmp(argv[2], "--prewitt") == 0 || strcmp(argv[2], "-p") == 0) {
		detectByPrewitt(src, dst, params);
	}
	else if (strcmp(argv[2], "--laplace") == 0 || strcmp(argv[2], "-l") == 0) {
		detectByLaplace(src, dst, params);
	}
	else if (strcmp(argv[2], "--canny") == 0 || strcmp(argv[2], "-c") == 0) {
		detectByCanny(src, dst, params);
	}

	// Display all images, include original, grayscale original, result, and images in some steps of algorithm
	displayer.displayAllImages();

	return 0;
}