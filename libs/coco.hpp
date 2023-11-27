// Copyright [2023] <Mhammad Rumi>

#pragma once
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>  // NOLINT
#include <vector>  // NOLINT

#include "./lib.hpp"
class coco {
 protected:
  using json = nlohmann::json;
  using _map_label = std::map<Key<int>, std::vector<label>>;
  using _shared_map = std::shared_ptr<_map_label>;
  using _vecjson = std::vector<json>;
  using _img_cat = std::map<Key<int>, std::vector<std::vector<float>>>;
  using _curve = std::vector<point<float>>;
  using _match = std::vector<match>;

 private:
  std::map<int, std::vector<int>> catToImgs;
  _map_label gt, dt;
  _img_cat ious;
  _match mapped;

 public:
  std::string validation_file;
  std::string detection_file;
  std::vector<int> imgIds;
  std::vector<int> catIds;
  params val_params;

 private:
  void create_index();
  match evalImg(const int&, const int&, const point<float>&, const int&);
  float iou(const std::vector<float>&, const std::vector<float>&, const int&);

  void get_scores();
  std::map<int, _curve> precision_recall(const std::vector<float>& thres);
  float computemAP(float thres);

 public:
  explicit coco(const std::string&);
  ~coco();
  void evaluation(const float* IOU_range);
  _vecjson get_annotations(int image_id);
  void loadRes(const std::string resFile);
};
