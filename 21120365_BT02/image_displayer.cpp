#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_map>
#include "image_displayer.h"

using namespace cv;
using namespace std;

ImageDisplayer::ImageDisplayer() {
	this->images = {};
}

// Add an image with given name to display
bool ImageDisplayer::addImage(string name, Mat images) {
	if (this->images.find(name) == this->images.end()) {
		this->images[name] = images;
		return true;
	}
	return false;
}

// Display all images
void ImageDisplayer::displayAllImages() {
	for (const auto& entry : this->images) {
		namedWindow(entry.first, WINDOW_AUTOSIZE);
		imshow(entry.first, entry.second);
	}
	waitKey();
	destroyAllWindows();
}