#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

class data_loader {
 private:
  std::string folderPath;
  int batchSize;
  int currentImageIndex;

  using _pairvec = std::vector<std::pair<std::string, std::string>>;
  using _matpair = std::vector<std::pair<cv::Mat, int>>;
  std::unordered_map<int, std::pair<std::string, int>> imageFileNames;

  void loadImages();
  int regf(std::string my_str);

 public:
  data_loader(const std::string& folderPath, const int& batchSize = 3);
  bool end();
  _matpair next();
};
