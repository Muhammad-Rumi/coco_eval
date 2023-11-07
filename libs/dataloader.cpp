#include "data_loader.hpp"

void data_loader::loadImages() {
  std::vector<std::string> temp;
  cv::glob(folderPath, temp, false);

  for (int i = 0; i < temp.size(); i++) {
    int img_id = regf(temp[i]);
    imageFileNames.insert({i, {temp[i], img_id}});
  }
}

int data_loader::regf(std::string my_str) {
  std::regex rex("/(\\d+)");

  std::smatch m;
  std::regex_search(my_str, m, rex);
  int img_id = std::stoi(m[1]);
  // std::cout << img_id << std::endl;
  return img_id;
}

data_loader::data_loader(const std::string& folderPath, const int& batchSize)
    : folderPath(folderPath), batchSize(batchSize), currentImageIndex(0) {
  loadImages();
}

bool data_loader::end() { return currentImageIndex >= imageFileNames.size(); }

data_loader::_matpair data_loader::next() {
  data_loader::_matpair batch;

  // Preload the next batch of images.
  // std::cout << currentImageIndex << std::endl;
  // for (int i = currentImageIndex + batchSize; i < imageFileNames.size();
  //      i++) {
  //   cv::Mat image = cv::imread(imageFileNames[i].first);
  //   if (!image.empty()) {
  //     batch.push_back({image, imageFileNames[i].second});
  //      std::cout << "here in next batch"<< std::endl;
  //   }
  // }

  // Process the current batch of images.
  for (int i = 0; i < batchSize && currentImageIndex < imageFileNames.size();
       i++, currentImageIndex++) {
    cv::Mat image = cv::imread(imageFileNames[currentImageIndex].first);
    if (!image.empty()) {
      batch.push_back({image, imageFileNames[currentImageIndex].second});
      // std::cout << "here in current batch"<< std::endl;
    }
  }
  // std::cout << "current loaded images: " << batch.size() << std::endl;
  return batch;
}
