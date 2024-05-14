#pragma once

#include <unordered_map>
#include <opencv2/core.hpp>

class ImageDisplayer {
private:
	std::unordered_map<std::string, cv::Mat> images;

public:
	ImageDisplayer();

public:
	bool addImage(std::string name, cv::Mat images);
	void displayAllImages();
};

