#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <cmath>
#include <stack>
#include "image_displayer.h"

using namespace cv;
using namespace std;

const double PI = acos(-1.0);

// Defines a kernel as an alias of 2D-vector
template<typename T>
using Kernel = vector<vector<T>>;

// Converts some numerical type to corresponding OpenCV type
template<typename T>
inline int getOpenCVType();

template<>
inline int getOpenCVType<uchar>() {
	return CV_8UC1;
}

template<>
inline int getOpenCVType<int>() {
	return CV_32SC1;
}

template<>
inline int getOpenCVType<double>() {
	return CV_64FC1;
}

// Global object displayer to display intermediate and output images
extern ImageDisplayer displayer;

template<typename T_in, typename T_out>
Mat convertMat(Mat input);

template<>
Mat convertMat<double, uchar>(Mat input) {
	Mat output = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < input.rows; ++i) {
		for (int j = 0; j < input.cols; ++j) {
			output.at<uchar>(i, j) =
				(input.at<double>(i, j) >= 0.0) ?
				((input.at<double>(i, j) < 255.0) ? input.at<double>(i, j) : 255.0) :
				0.0;
		}
	}
	return output;
}

template<>
Mat convertMat<int, uchar>(Mat input) {
	Mat output = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < input.rows; ++i) {
		for (int j = 0; j < input.cols; ++j) {
			output.at<uchar>(i, j) =
				(input.at<int>(i, j) >= 0) ?
				((input.at<int>(i, j) <= 255) ? input.at<int>(i, j) : 255) :
				0;
		}
	}
	return output;
}

template<>
Mat convertMat<uchar, double>(Mat input) {
	Mat output = Mat::zeros(input.rows, input.cols, CV_64FC1);
	for (int i = 0; i < input.rows; ++i) {
		for (int j = 0; j < input.cols; ++j) {
			output.at<double>(i, j) = (double)input.at<uchar>(i, j);
		}
	}
	return output;
}

// General, naive implementation for 2D-convolution operator
// Returns output of type double
template<typename T>
Mat convolve2D(Mat input, Kernel<T> &kernel) {
	Mat output = Mat::zeros(input.rows, input.cols, getOpenCVType<T>());
	int rows = input.rows, cols = input.cols, kCenter = kernel.size() / 2;
	int i, j, offset_i, offset_j, ii, jj;	// (i, j): coor. or current pixel
											// (offset_i, offset_j): 2D offset on kernel
											// (ii, jj): coor. of pixel that corresponds to the offset on kernel
	T accum; // Variable to store accumulate result of convoluton operation at each pixel
	// Loop through output image
	for (i = 0; i < rows; ++i) {
		for (j = 0; j < cols; ++j) {
			// Compute convolution at each pixel
			accum = 0;
			for (offset_i = -kCenter; offset_i <= kCenter; ++offset_i) {
				for (offset_j = -kCenter; offset_j <= kCenter; ++offset_j) {
					ii = i + offset_i;
					jj = j + offset_j;
					if (ii >= 0 && ii < rows && jj >= 0 && jj < cols) {
						accum += input.at<T>(ii, jj) * kernel[kCenter + offset_i][kCenter + offset_j];
					}
				}
			}
			output.at<T>(i, j) = accum;
		}
	}
	return output;
}

// Slight modification for 2D-convolution on integer that takes input of type uchar
template<>
Mat convolve2D<int>(Mat input, Kernel<int>& kernel) {
	Mat output = Mat::zeros(input.rows, input.cols, CV_32SC1);
	int rows = input.rows, cols = input.cols, kCenter = kernel.size() / 2;
	int i, j, offset_i, offset_j, ii, jj;	// (i, j): coor. or current pixel
											// (offset_i, offset_j): 2D offset on kernel
											// (ii, jj): coor. of pixel that corresponds to the offset on kernel
	int accum; // Variable to store accumulate result of convoluton operation at each pixel
	// Variable to store accumulate result of convoluton operation at each pixel
	for (i = 0; i < rows; ++i) {
		for (j = 0; j < cols; ++j) {
			// Compute convolution at each pixel
			accum = 0;
			for (offset_i = -kCenter; offset_i <= kCenter; ++offset_i) {
				for (offset_j = -kCenter; offset_j <= kCenter; ++offset_j) {
					ii = i + offset_i;
					jj = j + offset_j;
					if (ii >= 0 && ii < rows && jj >= 0 && jj < cols) {
						// Modification here: the input pixel is get as uchar instead of int
						accum += input.at<uchar>(ii, jj) * kernel[kCenter + offset_i][kCenter + offset_j];
					}
				}
			}
			output.at<int>(i, j) = accum;
		}
	}
	return output;
}

// Generates a Gaussian filter with given size and sigma
Kernel<double> generateGaussianFilter(int size, double sigma) {
	double s = sigma * sigma;
	Kernel<double> kernel(size, vector<double>(size));
	double sum = 0.0; // Stores sum of all elements for normalization
	int halfSize = size / 2;

	// Compute Gaussian at each pixel
	for (int x = -halfSize; x <= halfSize; ++x) {
		for (int y = -halfSize; y <= halfSize; ++y) {
			kernel[x + halfSize][y + halfSize] = (-exp((x * x + y * y) / 2 * s)) / (2 * PI * s);
			sum += kernel[x + halfSize][y + halfSize];
		}
	}

	// Normalize result
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			kernel[i][j] /= sum;
		}
	}

	return kernel;
}

// Generates Gaussian filter with given size and sigma
// then applies the filter onto input
Mat blurWithGaussian(Mat input, int kernelSize, double sigma) {
	Kernel<double> kernel = generateGaussianFilter(kernelSize, sigma);
	return convolve2D<double>(input, kernel);
}

// Computes gradient magnitude at each pixel with given horizontal and vertical gradients.
template<typename T_in>
Mat computeGradientMagnitude(Mat grad_x, Mat grad_y) {
	Mat g = Mat::zeros(grad_x.rows, grad_x.cols, CV_64FC1);
	int rows = grad_x.rows, cols = grad_x.cols;
	// Compute gradient intensity at each pixel
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			// ||g|| = ||gx + gy|| = sqrt(||gx|| + ||gy||)
			g.at<double>(i, j) = sqrt(grad_x.at<T_in>(i, j) * grad_x.at<T_in>(i, j) + grad_y.at<T_in>(i, j) * grad_y.at<T_in>(i, j));
		}
	}
	return g;
}

// Computes both gradient magnitude and gradient direction at each pixel with given horizontal and vertical gradients.
// Note: the direction is compute in atan2 function for consistency with the axis of cv::Mat,
// so the direction of angle is somehow "flipped" compare to the common unit circle.
template<typename T_in>
Mat computeGradient(Mat grad_x, Mat grad_y, Mat theta) {
	Mat output = Mat::zeros(grad_x.rows, grad_y.cols, CV_64FC1);
	int rows = grad_x.rows, cols = grad_x.cols;

	// Compute gradient magnitude and gradient direction at each pixel
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			// ||g|| = ||gx + gy|| = sqrt(||gx|| + ||gy||)
			output.at<double>(i, j) = sqrt(grad_x.at<T_in>(i, j) * grad_x.at<T_in>(i, j) + grad_y.at<T_in>(i, j) * grad_y.at<T_in>(i, j));
			// theta = arctan(gy/gx)
			theta.at<double>(i, j) = atan2((double)grad_y.at<T_in>(i, j), (double)grad_x.at<T_in>(i, j)) * 180 / PI;
		}
	}
	
	return output;
}

// Suppress pixels are not the local maxima along the gradient direction
// Coor. of local maxima is stored in `maxima`
Mat suppressNonMax(Mat g, Mat theta, vector<pair<int, int>> &maxima) {

	int rows = g.rows, cols = g.cols;
	Mat output = Mat::zeros(rows, cols, CV_64FC1);
	int prev_i, prev_j, next_i, next_j; // coordinates of neighbours along the gradient direction
	double gradDirection;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			gradDirection = theta.at<double>(i, j);
			// Main direction is 0 or 180 degree
			if (gradDirection < 22.5 || gradDirection >= 157.5) {
				prev_i = i; prev_j = j - 1;
				next_i = i; next_j = j + 1;
			}
			// Main direction is 45 degree
			else if (gradDirection >= 22.5 && gradDirection < 67.5) {
				prev_i = i - 1; prev_j = j - 1;
				next_i = i + 1; next_j = j + 1;
			}
			// Main direction is 90 degree
			else if (gradDirection >= 67.5 && gradDirection < 112.5) {
				prev_i = i - 1; prev_j = j;
				next_i = i + 1; next_j = j;
			}
			// Main direction is 135 degree
			else {
				prev_i = i + 1; prev_j = j - 1;
				next_i = i - 1; next_j = j + 1;
			}
			// If prev is a valid coor.
			if (prev_i >= 0 && prev_i < rows && prev_j >= 0 && prev_j < cols) {
				// and has greater intensity
				if (g.at<double>(i, j) < g.at<double>(prev_i, prev_j)) {
					// then suppress current pixel
					output.at<double>(i, j) = 0;
					continue;
				}
			}
			// Else if next is a valid coor.
			if (next_i >= 0 && next_i < rows && next_j >= 0 && next_j < cols) {
				// and has greater intensity
				if (g.at<double>(i, j) < g.at<double>(next_i, next_j)) {
					// then suppress current pixel
					output.at<double>(i, j) = 0;
					continue;
				}
			}
			// Else current pixel is the local maximum
			maxima.push_back(make_pair(i, j));
			output.at<double>(i, j) = g.at<double>(i, j);
		}
	}
	return output;
}

// Detect edges by Sobel
int detectBySobel(Mat src, Mat dst, unordered_map<string, double> &params) {
	int rows = src.rows, cols = src.cols;
	Mat mat, grad_x, grad_y;
	
	// No blur
	if (params.find("blur_ksize") == params.end() || params["blur_ksize"] == 1) {
		Kernel<int> kernel_x = {
			{ -1, 0, 1 },
			{ -2, 0, 2 },
			{ -1, 0, 1 }
		};
		Kernel<int> kernel_y = {
			{ -1, -2, -1 },
			{ 0, 0, 0 },
			{ 1, 2, 1 }
		};
		// Gradient along x
		grad_x = convolve2D<int>(src, kernel_x);
		// Gradient along y
		grad_y = convolve2D<int>(src, kernel_y);
		// Gradient intensity
		mat = computeGradientMagnitude<int>(grad_x, grad_y);

		displayer.addImage("Sobel: grad x", convertMat<int, uchar>(grad_x));
		displayer.addImage("Sobel: grad y", convertMat<int, uchar>(grad_y));
	}
	// Blur
	else {
		// Blur image with Gaussian filter
		double sigma = (params.find("blur_sigma") == params.end()) ? 1.0 : params["blur_sigma"];
		double kernelSize = (int)params["blur_ksize"];
		mat = blurWithGaussian(convertMat<uchar, double>(src), kernelSize, sigma);

		Kernel<double> kernel_x = {
			{ -1, 0, 1 },
			{ -2, 0, 2 },
			{ -1, 0, 1 }
		};
		Kernel<double> kernel_y = {
			{ -1, -2, -1 },
			{ 0, 0, 0 },
			{ 1, 2, 1 }
		};
		// Gradient along x
		grad_x = convolve2D<double>(mat, kernel_x);
		// Gradient along y
		grad_y = convolve2D<double>(mat, kernel_y);
		// Gradient intensity
		mat = computeGradientMagnitude<double>(grad_x, grad_y);

		displayer.addImage("Sobel: grad x", convertMat<double, uchar>(grad_x));
		displayer.addImage("Sobel: grad y", convertMat<double, uchar>(grad_y));
	}

	dst = convertMat<double, uchar>(mat);
	displayer.addImage("Sobel: result", dst);

	return 0;
}

// Detect edges by Prewitt
int detectByPrewitt(Mat src, Mat dst, unordered_map<string, double> &params) {
	int rows = src.rows, cols = src.cols;
	Mat mat, grad_x, grad_y;

	// No blur
	if (params.find("blur_ksize") == params.end() || params["blur_ksize"] == 1) {
		Kernel<int> kernel_x = {
			{ -1, 0, 1 },
			{ -1, 0, 1 },
			{ -1, 0, 1 },
		};
		Kernel<int> kernel_y = {
			{ -1, -1, -1 },
			{ 0, 0, 0 },
			{ 1, 1, 1 }
		};
		// Gradient along x
		grad_x = convolve2D<int>(src, kernel_x);
		// Gradient along y
		grad_y = convolve2D<int>(src, kernel_y);
		// Gradient intensity
		mat = computeGradientMagnitude<int>(grad_x, grad_y);

		displayer.addImage("Prewitt: grad x", convertMat<int, uchar>(grad_x));
		displayer.addImage("Prewitt: grad y", convertMat<int, uchar>(grad_y));
	}
	// Blur
	else {
		// Blur image with Gaussian filter
		double sigma = (params.find("blur_sigma") == params.end()) ? 1.0 : params["blur_sigma"];
		double kernelSize = (int)params["blur_ksize"];
		mat = blurWithGaussian(convertMat<uchar, double>(src), kernelSize, sigma);

		Kernel<double> kernel_x = {
			{ -1, 0, 1 },
			{ -1, 0, 1 },
			{ -1, 0, 1 },
		};
		Kernel<double> kernel_y = {
			{ -1, -1, -1 },
			{ 0, 0, 0 },
			{ 1, 1, 1 }
		};
		// Gradient along x
		grad_x = convolve2D<double>(mat, kernel_x);
		// Gradient along y
		grad_y = convolve2D<double>(mat, kernel_y);
		// Gradient intensity
		mat = computeGradientMagnitude<double>(grad_x, grad_y);

		displayer.addImage("Prewitt: grad x", convertMat<double, uchar>(grad_x));
		displayer.addImage("Prewitt: grad y", convertMat<double, uchar>(grad_y));
	}

	dst = convertMat<double, uchar>(mat);
	displayer.addImage("Prewitt: result", dst);

	return 0;
}

// Detects edges with Laplace
int detectByLaplace(Mat src, Mat dst, unordered_map<string, double> &params) {
	Mat mat;
	int kernelId = (params.find("laplace_kernel") == params.end()) ? 0 : (int)(params["laplace_kernel"]);
	// If blurring is not applied, then perform integer convolution for faster computation
	if (params.find("blur_ksize") == params.end() || params["blur_ksize"] == 1) {
		// Set kernel by user's specified
		Kernel<int> kernel =
			(kernelId == 0) ?
			vector<vector<int>>({ {0, -1, 0}, {-1, 4, -1}, {0, - 1, 0} }) :
			vector<vector<int>>({ {-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1} });
		mat = convolve2D<int>(src, kernel);
		dst = convertMat<int, uchar>(mat);
	}
	// Else, perform convolution on real numbers
	else {
		// Blurs image with Gaussian filter
		double sigma = (params.find("blur_sigma") == params.end()) ? 1.0 : params["blur_sigma"];
		double kernelSize = (int)params["blur_ksize"];
		mat = blurWithGaussian(convertMat<uchar, double>(src), kernelSize, sigma);

		// Set kernel by user's specified
		Kernel<double> kernel =
			(kernelId == 0) ?
			vector<vector<double>>({ {0, -1, 0}, {-1, 4, -1}, {0, - 1, 0} }) :
			vector<vector<double>>({ {-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1} });
		mat = convolve2D<double>(mat, kernel);
		dst = convertMat<double, uchar>(mat);
	}
	displayer.addImage("Laplacian", dst);
	return 0;
}

// Detects edges with Canny
int detectByCanny(Mat src, Mat dst, unordered_map<string, double> &params) {
	int rows = src.rows, cols = src.cols;
	Mat mat;

	// Blur
	if (params.find("blur_ksize") == params.end() || params["blur_ksize"] == 1) {
		mat = convertMat<uchar, double>(src);
	}
	else {
		double sigma = (params.find("blur_sigma") == params.end()) ? 1.0 : params["blur_sigma"];
		double kernelSize = (int)params["blur_ksize"];
		mat = blurWithGaussian(convertMat<uchar, double>(src), kernelSize, sigma);
	}

	// Sobel
	Kernel<double> kernel_x = {
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1}
	};
	Kernel<double> kernel_y = {
		{ -1, -2, -1 },
		{ 0, 0, 0 },
		{ 1, 2, 1 }
	};

	Mat grad_x = convolve2D<double>(mat, kernel_x);
	Mat grad_y = convolve2D<double>(mat, kernel_y);
	Mat theta = Mat::zeros(rows, cols, CV_64FC1);
	mat = computeGradient<double>(grad_x, grad_y, theta);

	// Non-maximum suppression
	vector<pair<int, int>> maxima;
	mat = suppressNonMax(mat, theta, maxima);

	// Double-thresholding and Hysteresis tracking
	dst = Mat::zeros(rows, cols, CV_8UC1);

	// DFS-simulated tracking
	stack<pair<int, int>> strongs;	// Maintains a set of "strong" pixels that are added to result
									// but not yes expanded (i.e. their neighbour are not examined)
	double low, high; // Low and high thresholds
	if (params.find("high") == params.end()) {
		if (params.find("low") == params.end()) {
			low = 85;
			high = 255;
		}
		else {
			low = params["low"];
			high = low * 3;
		}
	}
	else {
		high = params["high"];
		low = (params.find("low") == params.end()) ? (high / 3) : params["low"];
	}

	// First iterate through local maxima and filter out "strong" pixels
	for (int i = 0; i < maxima.size(); ++i)
		if (mat.at<double>(maxima[i].first, maxima[i].second) > high) {
			strongs.push(maxima[i]);
			dst.at<uchar>(maxima[i].first, maxima[i].second) = 255;
		}
	
	// While their are strong pixels not examined
	while (!strongs.empty()) {
		// Get one strong pixel and remove it
		pair<int, int> strong = strongs.top();
		strongs.pop();
		int i = strong.first, j = strong.second;
		// Examine it neighbours
		for (int offset_i = -1; offset_i <= 1; ++offset_i) {
			for (int offset_j = -1; offset_j <= 1; ++offset_j) {
				// Exclude the strong pixel itself
				if (offset_i == 0 && offset_j == 0)
					continue;
				int ii = i + offset_i, jj = j + offset_j;
				// If the neighbour has valid coor.
				if (ii >= 0 && ii < rows && jj >= 0 && jj < cols)
					// and it's intensity falls between the two thresholds, and it's not examined (not added into result)
					if (mat.at<double>(ii, jj) >= low && mat.at<double>(ii, jj) < high && dst.at<uchar>(ii, jj) == 0) {
						// then add it into the result
						dst.at<uchar>(ii, jj) = 255;
						// and push it into the stack for examining
						strongs.push(make_pair(ii, jj));
					}
			}
		}
	}

	displayer.addImage("Canny", dst);

	return 0;
}