#pragma once  // NOLINT

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>  // NOLINT
#include <vector>  // NOLINT

#define EXTRACT(x, k, J) x = J[#k].get<decltype(x)>()
#define PRINT(message, variable) \
  std::cout << message << ": " << variable << std::endl
#define SEPARATOR std::cout << "-------------------------------" << std::endl

struct point {
  float x, y;
};
struct label {
  int imgid;
  std::vector<std::vector<float>> bbox;
  std::vector<int> catids;
  std::vector<float> scores;
};

class coco {
 private:
  using json = nlohmann::json;
  using _map_label = std::map<int, label>;
  using _shared_map = std::shared_ptr<_map_label>;
  using _vecjson = std::vector<json>;
  json dataset;
  std::map<int, json> anns, cats, imgs, imgToAnns;
  std::map<int, std::vector<int>> catToImgs;  //  must remove later
  _map_label gt, dt;
  void create_index();
  float iou(const std::vector<float>& gt_bbox,
            const std::vector<float>& dt_bbox);
  void filter(const std::shared_ptr<_map_label> original,
              const float thres = 0.5);
  void precision_recall(const std::vector<float>& thres);
  float computemAP(float thres);

 public:
  explicit coco(const std::string& annotation_file);
  ~coco();
  void evaluation(const float* IOU_range);
  _vecjson get_annotations(int image_id);
  // void loadRes(const std::string resFile);
  void loadRes(const std::string resFile, std::string flag);
};
void zero_loc(
    const std::vector<int>& myVector) {  // works for only sorted vectors, sad!!
  std::vector<int> tempVector = myVector;
  std::vector<int>::iterator lower =
      std::lower_bound(tempVector.begin(), tempVector.end(), 5);
  std::vector<int>::iterator upper =
      std::upper_bound(tempVector.begin(), tempVector.end(), 5);

  for (std::vector<int>::iterator it = lower; it != upper; ++it) {
    std::cout << "Zero found at index: " << it - tempVector.begin()
              << std::endl;
  }
}
void zero_count(const std::vector<int>& myVector) {
  int zeroCount = std::count_if(myVector.begin(), myVector.end(),
                                [](int x) { return x == 0; });
  std::cout << "Number of zeros: " << zeroCount << std::endl;
  for (auto&& i : myVector) {
    std::cout << i << std::endl;
  }
}
