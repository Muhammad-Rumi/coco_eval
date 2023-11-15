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
struct label {
  int imgid;
  std::vector<std::vector<float>> bbox;
  std::vector<float> catids;
  std::vector<float> scores;
};

class coco {
 private:
  using json = nlohmann::json;
  using _map_label = std::map<int, label>;
  using _shared_map = std::shared_ptr<_map_label>;
  using _vecjson = std::vector<json>;
  json dataset;
  std::map<int, json> anns, cats, imgs, imgToAnns,
      catToImgs;  //  must remove later
  _map_label gt, dt;
  // dt-> bbox must be sorted in decending order for scores values.
  // _shared_map dt;
  void create_index();
  float iou(const std::vector<float>& gt_bbox,
            const std::vector<float>& dt_bbox);
  void filter(const std::shared_ptr<_map_label> original,
              const float thres = 0.5);
    void computemAP(float thres);

 public:
  explicit coco(const std::string& annotation_file);
  ~coco();
  void evaluation(float score_thres, float IOU_thres);
  _vecjson get_annotations(int image_id);
  // void loadRes(const std::string resFile);
  void loadRes(const std::string resFile, std::string flag);
};
