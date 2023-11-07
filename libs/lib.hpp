#pragma once  // NOLINT

#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>  // NOLINT
#include <vector>  // NOLINT

class coco {
 private:
  using json = nlohmann::json;
  using _vecjson = std::vector<json>;
  std::map<int, json> anns, cats, imgs, imgToAnns, catToImgs;

  json dataset;
  void create_index();

 public:
  coco(const std::string& annotation_file);
  ~coco();

  std::vector<json> get_annotations(int image_id);
};
