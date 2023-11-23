// Copyright [2023] <Mhammad Rumi>
#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>  // NOLINT
#include <vector>  // NOLINT

#define EXTRACT(x, J) x = J[#x].get<decltype(x)>()
#define PRINT(message, variable) std::cout << message << variable << std::endl
#define SEPARATOR std::cout << "-------------------------------" << std::endl
template <typename _T, typename _Y>
struct point {
  _T x;
  _Y y;
  friend std::ostream& operator<<(std::ostream& os, const point& p) {
    return os << p.x << ", " << p.y << std::endl;
  }
};
struct Key {
  int imgId;
  int catId;
  friend std::ostream& operator<<(std::ostream& os, const Key& k) {
    return os << k.imgId << ", " << k.catId << std::endl;
  }

  bool operator==(const Key& other) const {
    return imgId == other.imgId && catId == other.catId;
  }
  bool operator<(const Key& other) const {
    return (imgId < other.imgId) ||
           (imgId == other.imgId && catId < other.catId);
  }

  size_t hash() const {
    std::hash<int> hasher;
    return hasher(imgId) ^ hasher(catId);
  }
};

struct label {
  int imgid, catids;
  float scores;
  std::vector<float> bbox;
};

template <typename _t, typename _y, typename _r>
struct table {
  static int inst;
  int id, id1, id2;
  _t truePos;
  _y total_gt;
  _r total_dt;
  explicit table(const int& size) {
    // inst = size;
    id = -1, id1 = -1, id2 = -1;
    truePos.resize(size);
    total_gt.resize(size);
    total_dt.resize(size);
  }
  table(const _t& tp, const _y& tg, const _r& td) {
    inst++;
    id = -1, id1 = -1, id2 = -1;
    truePos = tp;
    total_gt = tg;
    total_dt = td;
  }
  table() {
    // inst++;
    id = -1, id1 = -1, id2 = -1;
  }
};

class coco {
 protected:
  using json = nlohmann::json;
  using _map_label = std::map<Key, std::vector<label>>;
  using _shared_map = std::shared_ptr<_map_label>;
  using _vecjson = std::vector<json>;
  using _img_cat = std::map<Key, std::vector<float>>;
  using _curve = std::vector<point<float, float>>;

 private:
  json dataset;
  std::map<int, json> anns, cats, imgs, imgToAnns;
  std::map<int, std::vector<int>> catToImgs;  //  must remove later
  _map_label gt, dt;
  _img_cat ious;

 public:
  std::string validation_file;
  std::string detection_file;

 private:
  void create_index();
  float iou(const std::vector<float>& gt_bbox,
            const std::vector<float>& dt_bbox);

  void get_scores();
  std::map<int, _curve> precision_recall(const std::vector<float>& thres);
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
int count_x(const std::vector<int>& myVector, int x) {
  int Count = std::count_if(myVector.begin(), myVector.end(),
                            [x](int y) { return y == x; });
  // std::cout << "Number of zeros: " << Count << std::endl;
  // for (auto&& i : myVector) {
  //   std::cout << i << std::endl;
  // }
  return Count;
}
