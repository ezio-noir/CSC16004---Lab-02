#pragma once

#include "opencv2/core.hpp"
#include <unordered_map>

int detectBySobel(cv::Mat src, cv::Mat dst, std::unordered_map<std::string, double>& params);
int detectByPrewitt(cv::Mat src, cv::Mat dst, std::unordered_map<std::string, double>& params);
int detectByLaplace(cv::Mat src, cv::Mat dst, std::unordered_map<std::string, double>& params);
int detectByCanny(cv::Mat src, cv::Mat dst, std::unordered_map<std::string, double>& params);