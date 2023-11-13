#pragma once  // NOLINT

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>  // NOLINT
#include <vector>  // NOLINT

#define EXTRACT(x, J) x = J[#x].get<decltype(x)>()
#define PRINT(message, variable) \
  std::cout << message << ": " << variable << std::endl
struct label {
  int imgid;
  std::vector<float> bbox;
  std::vector<float> catid;
  std::vector<float> score;
};

class coco {
 private:
  using json = nlohmann::json;
  using _vecjson = std::vector<json>;
  json dataset;
  std::map<int, json> anns, cats, imgs, imgToAnns, catToImgs;
  std::map<int, label> gt;
  std::shared_ptr<json> detections;
  void create_index();
  float iou(const std::vector<float>& gt_bbox,
            const std::vector<float>& dt_bbox);
std::shared_ptr<json> filter(std::shared_ptr<json> original,
                                   float thres = 0.5);
 public:
  explicit coco(const std::string& annotation_file);
  ~coco();
  void evalutaion(float score_thres, float IOU_thres);
  _vecjson get_annotations(int image_id);
  void loadRes(const std::string resFile);
};
