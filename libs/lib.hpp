// Copyright [2023] <Mhammad Rumi>
#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>  // NOLINT
#include <vector>  // NOLINT

#define EP0 1.E-5
#define EXTRACT(x, J) x = J[#x].get<decltype(x)>()
#define PRINT(message, variable) std::cout << message << variable << std::endl
#define SEPARATOR std::cout << "-------------------------------" << std::endl
template <typename _T>
struct point {
  _T x;
  _T y;
  bool operator()(_T value) const { return (value >= x && value <= y); }
  friend std::ostream& operator<<(std::ostream& os, const point& p) {
    return os << p.x << ", " << p.y;
  }
};
template <typename _T>
struct Key {
  _T k1;
  _T k2;
  friend std::ostream& operator<<(std::ostream& os, const Key& k) {
    return os << k.k1 << ", " << k.k2;
  }

  bool operator==(const Key& other) const {
    return k1 == other.k1 && k2 == other.k2;
  }
  bool operator<(const Key& other) const {
    return (k1 < other.k1) || (k1 == other.k1 && k2 < other.k2);
  }
  bool operator>(const Key& other) const {
    return (k1 > other.k1) || (k1 == other.k1 && k2 > other.k2);
  }
  size_t hash() const {
    std::hash<_T> hasher;
    return hasher(k1) ^ hasher(k2);
  }
};

struct label {
  int id;
  int is_crowd;
  int imgid, catids;
  float scores, area;
  std::vector<float> bbox;
};
struct params {
  int x = 10;
  std::vector<float> iou_Thrs;
  std::vector<float> recThrs;
  std::vector<int> maxDet;
  std::vector<point<float>> aRng;
  params() : x(10), iou_Thrs(10), recThrs(101), maxDet(3), aRng(4) {
    aRng = {{0, 1E5 * 1E5},
            {0, 32 * 32},
            {32 * 32},
            {96 * 96},
            {96 * 96, 1E5 * 1E5}};
    std::generate_n(iou_Thrs.begin(), iou_Thrs.size(), []() {
      static float iouThreshold = 0.45;
      iouThreshold += 0.05;
      return iouThreshold;
    });
    std::generate_n(recThrs.begin(), recThrs.size(), []() {
      static float iouThreshold = 0.;
      iouThreshold += 0.01;
      return iouThreshold;
    });
  }
};

struct match {
  using _2dvec = std::vector<std::vector<float>>;
  int imgId, catId, maxDet;
  point<float> aRng;
  _2dvec dtm, gtm, dtIg;
  std::vector<float> score;
  std::vector<bool> gtIg;
  match(const int& Id, const int& id, const int& Det,
        const point<float>& area_rng, _2dvec d_match, _2dvec g_match,
        _2dvec d_ig, std::vector<float> scores, std::vector<bool> g_ig) {
    imgId = Id;
    catId = id;
    maxDet = Det;
    aRng = area_rng;
    dtm = d_match;
    gtm = g_match;
    dtIg = d_ig;
    score = scores;
    gtIg = g_ig;
  }
};

class coco {
 protected:
  using json = nlohmann::json;
  using _map_label = std::map<Key, std::vector<label>>;
  using _shared_map = std::shared_ptr<_map_label>;
  using _vecjson = std::vector<json>;
  using _img_cat = std::map<Key, std::vector<std::vector<float>>>;
  using _curve = std::vector<point<float, float>>;
  // using _eval_img = std::vector<

 private:
  std::map<int, std::vector<int>> catToImgs;
  _map_label gt, dt;
  _img_cat ious;

 public:
  std::string validation_file;
  std::string detection_file;
  std::vector<int> imgIds;
  std::vector<int> catIds;

 private:
  void create_index();
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
int count_x(const std::vector<int>& myVector, int x) {
  int Count = std::count_if(myVector.begin(), myVector.end(),
                            [x](int y) { return y == x; });
  return Count;
}
int count_x(const std::vector<label>& myVector, int x) {
  int Count = std::count_if(myVector.begin(), myVector.end(),
                            [x](label y) { return y.catids == x; });
  return Count;
}
