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
template <typename _T, typename _Y>
struct point {
  _T x;
  _Y y;
  friend std::ostream& operator<<(std::ostream& os, const point& p) {
    return os << "mAP: " << p.x << ", mRc: " << p.y;
  }
};
struct Key {
  int imgId;
  int catId;

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
  int imgid, len;
  std::vector<std::vector<float>> bbox;
  std::vector<int> catids;
  std::vector<float> scores;
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
  using _map_label = std::map<int, label>;
  using _processed_vla = point<float, std::vector<point<float, float>>>;
  using _table_scaler = table<float, float, float>;
  using _per_img_cat = std::map<Key, std::vector<coco::_table_scaler>>;
  using _vecjson = std::vector<json>;

 private:
  json dataset;
  std::map<int, json> anns, cats, imgs, imgToAnns;
  std::map<int, std::vector<int>> catToImgs;  //  must remove later
  _map_label gt, dt;

 public:
  int alo;

 private:
  void create_index();
  float iou(const std::vector<float>& gt_bbox,
            const std::vector<float>& dt_bbox);
  void filter(const std::shared_ptr<_map_label> original,
              const float thres = 0.5);
  _per_img_cat calculate_confusion(const float& thres);
  _processed_vla precision_recall(const std::vector<float>& thres);
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
  // for (auto&& i : myVector) {
  //   std::cout << i << std::endl;
  // }
}
