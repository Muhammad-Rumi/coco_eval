// Copyright [2023] <Mhammad Rumi>
#include "lib.hpp"  // NOLINT

coco::coco(const std::string& file) {
  validation_file = file;
  PRINT("Loading validation Truth form file: ", validation_file);
  create_index();
}
void coco::create_index() {
  std::cout << "Creating Index... " << std::endl;
  std::fstream file(validation_file);
  std::cout << "Loading annnotations to memory..." << std::endl;
  dataset = json::parse(file);

  std::vector<float> bbox;
  int category_id;
  int image_id;
  int iscrowd;
  if (dataset.contains("annotations")) {
    for (auto& ann : dataset["annotations"]) {
      EXTRACT(image_id, ann);
      EXTRACT(iscrowd, ann);
      EXTRACT(category_id, ann);
      EXTRACT(bbox, ann);
      gt[{image_id, category_id}].push_back(
          {iscrowd, image_id, category_id, 1., bbox});
      catToImgs[category_id].push_back(image_id);
    }
  }
  if (dataset.contains("images")) {
    for (auto&& img : dataset["images"]) {
      imgIds.push_back(img["id"]);
    }
  }
  if (dataset.contains("categories")) {
    for (auto&& cat : dataset["categories"]) {
      catIds.push_back(cat["id"]);
    }
  }
  // Print the size of the maps.
  PRINT("Number of images lable pair: ", gt.size());
  PRINT("Number of categories: ", catIds.size());
  PRINT("Total no. of images: ", imgIds.size());
  SEPARATOR;
}
float coco::iou(const std::vector<float>& gt_bbox,
                const std::vector<float>& dt_bbox) {
  /*
  input:
  gt_bbox = [xmin, ymin, w, h]
  dt_bbox = [xmin, ymin, w, h]
  ouput:
  float iou;
  */
  assert(gt_bbox.size() == 4 && dt_bbox.size() == 4);
  float x1, y1, x2, y2;
  x1 = std::max(gt_bbox[0], dt_bbox[0]);
  y1 = std::max(gt_bbox[1], dt_bbox[1]);
  x2 = std::min(gt_bbox[0] + gt_bbox[2], dt_bbox[0] + dt_bbox[2]);
  y2 = std::min(gt_bbox[1] + gt_bbox[3], dt_bbox[1] + dt_bbox[3]);
  if (x2 < x1 || y2 < y1) return 0.0;  // if there is no overlap between bboxes

  float intersection_area = (x2 - x1) * (y2 - y1);
  float area1 = (gt_bbox[2]) * (gt_bbox[3]);
  float area2 = (dt_bbox[2]) * (dt_bbox[3]);

  return intersection_area / (area1 + area2 - intersection_area);
}
void coco::get_scores() {
  PRINT("Calculating IOUs...", "");
  for (auto&& [key, dt_ann] : dt) {
    _map_label::iterator g_ann = gt.find(key);
    if (g_ann == gt.end()) {
      continue;
    }

    for (auto&& d : dt_ann) {
      std::vector<float> temp;
      for (auto&& g : g_ann->second) {
        int catId = g.catids;
        auto a = g.bbox;
        // if (g.is_crowd == true) continue;  // catId != d.catids &&
        float temps = coco::iou(g.bbox, d.bbox);
        temp.push_back(temps);
      }
      coco::ious[key].push_back(*std::max_element(temp.begin(), temp.end()));
    }
  }
  PRINT("IOUs calculated: ", ious.size());
  SEPARATOR;
}
std::map<int, coco::_curve> coco::precision_recall(
    const std::vector<float>& thres) {
  std::cout << "Calculating precision recall" << std::endl;

  std::map<int, _curve> pr;
  std::map<int, std::vector<point<int, int>>> per_sample_tp_dt;
  std::vector<int> truePos(91), total_gt(91), total_dt(91);

  for (const auto& [catId, imgIds] : catToImgs) {
    int TP = 0, gt_percat = 0, dt_percat = 0;
    for (const auto& imgId : imgIds) {
      Key keys = {imgId, catId};
      const auto& current_per_img_cat_id_iou = ious[keys];
      auto x = count_x(gt[keys], catId);
      // getting True positives upon IOU threshold.
      std::vector<int> matches;
      std::transform(current_per_img_cat_id_iou.begin(),
                     current_per_img_cat_id_iou.end(),
                     std::back_inserter(matches), [thres](float iou) {
                       return iou > 0.5;  // left to iterate over the thresholds
                     });
      auto tp = std::accumulate(matches.begin(), matches.end(), 0);
      if (tp > x) tp = x;
      TP += tp;
      assert(tp <= x);
      dt_percat += matches.size();
      per_sample_tp_dt[catId].push_back({TP, dt_percat});
      // total samples of a category in the entire dataset
      gt_percat += x;
    }
    // SEPARATOR;
    truePos[catId] = TP;
    total_dt[catId] = dt_percat;
    total_gt[catId] = gt_percat;
  }
  float P, R;
  for (auto&& [cats, tp_dt_per_cat] : per_sample_tp_dt) {
    for (int i = 0; i < tp_dt_per_cat.size(); ++i) {
      P = static_cast<float>(tp_dt_per_cat[i].x) /
          (static_cast<float>(tp_dt_per_cat[i].y) + EP0);
      R = static_cast<float>(tp_dt_per_cat[i].x) /
          (static_cast<float>(total_gt[cats]) + EP0);
      pr[cats].push_back({P, R});
    }
  }
  count_x(truePos, 0);
  PRINT("len of pr variable: ", pr.size());
  return pr;
}
void coco::evaluation(const float* IOU_range) {
  // assert(dt.size() == imgs.size());
  int rng = (IOU_range[1] - IOU_range[0]) / IOU_range[2];
  std::vector<float> iouThrs(rng);
  // compute IOUs.
  get_scores();
  PRINT("Size of dict<imgId><catId> of IOUs: ", ious.size());
  std::generate_n(iouThrs.begin(), iouThrs.size(), [IOU_range]() {
    static float iouThreshold = IOU_range[0] - IOU_range[2];
    iouThreshold += IOU_range[2];
    return iouThreshold;
  });
  auto preci_recall_vals = precision_recall(iouThrs);
  for (int i = 0; i < 91; i++) {
    std::ofstream file("logs/preci_recall_" + std::to_string(i) + ".txt");
    if (!file.is_open()) {
      std::cerr << "Error opening the file!"
                << std::endl;  // Return an error code
    }

    for (auto&& i : preci_recall_vals[i]) {
      file << i << std::endl;
    }
    file.close();
  }
}

void coco::loadRes(const std::string resFile, std::string flag) {
  std::fstream file(resFile);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file:" + resFile);
  }

  detection_file = resFile;
  PRINT("Loading results from: ", detection_file);
  json result = json::parse(file);

  std::vector<float> bbox;
  int category_id;
  float score;
  int image_id;
  std::cout << "Processing... " << std::endl;
  for (const auto& ann : result) {
    // imageId = ann["image_id"];
    EXTRACT(image_id, ann);
    EXTRACT(score, ann);
    EXTRACT(category_id, ann);
    EXTRACT(bbox, ann);
    dt[{image_id, category_id}].push_back(
        {0, image_id, category_id, score, bbox});
  }
  SEPARATOR;
}
coco::~coco() {}
