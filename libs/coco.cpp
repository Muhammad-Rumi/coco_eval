// Copyright [2023] <Mhammad Rumi>
#include "lib.hpp"  // NOLINT

coco::coco(const std::string& annotation_file) {
  std::fstream file(annotation_file);
  std::cout << "Loading annnotations to memory..." << std::endl;
  dataset = json::parse(file);
  create_index();
}
void coco::create_index() {
  std::cout << "Creating Index... " << std::endl;

  std::vector<float> bbox;
  int category_id;
  int image_id;
  if (dataset.contains("annotations")) {
    for (auto& ann : dataset["annotations"]) {
      //  = ann["image_id"];
      EXTRACT(image_id, ann);
      // int id = ann["id"];
      EXTRACT(category_id, ann);
      EXTRACT(bbox, ann);
      gt[{image_id, category_id}].push_back({image_id, category_id, 1., bbox});
    }
  }
  // Print the size of the maps.
  PRINT("Number of images lable pair: ", gt.size());
  PRINT("Number of categories: ", catToImgs.size());
  SEPARATOR;
  // for (const auto& [id, datas] : gt) {
  //   PRINT("Image id", id);
  //   PRINT("gor every img bbox size: ", datas.bbox.size());
  // }
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
  // PRINT("gt_bbox", gt_bbox.size());
  // PRINT("dt_bbox", dt_bbox.size());
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
  for (auto&& [imgId, dt_ann] : dt) {
    _map_label::iterator g = gt.find(imgId);
    if (g == gt.end()) {
      continue;
    }

    for (int i = 0; i < g->second.len; ++i) {
      std::vector<float> temp;
      int catId = g->second.catids[i];
      auto a = g->second.bbox[i];

      for (int det_idx = 0; det_idx < dt_ann.len; ++det_idx) {
        if (catId != dt_ann.catids[det_idx]) continue;
        float temps = coco::iou(a, dt_ann.bbox[i]);
        temp.push_back(temps);
        coco::ious[{imgId, catId}].push_back(temps);
      }

      // std::transform(
      //     dt_ann.bbox.begin(), dt_ann.bbox.end(), std::back_inserter(temp),
      //     [this, a](std::vector<float> b) { return coco::iou(a, b); });
      // coco::ious[{imgId, catId}].push_back(temp);
      // PRINT("temp size: ", temp.size());
      // std::for_each(temp.begin(), temp.end(),
      //               [](const float& i) { std::cout << i << ", "; });
      // std::cout << std::endl;
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
    PRINT("current CatID: ", catId);
    for (const auto& imgId : imgIds) {
      Key keys = {imgId, catId};
      const auto& current_per_img_cat_id_iou = ious[keys];
      // std::for_each(
      //     current_per_img_cat_id_iou.begin(),
      //     current_per_img_cat_id_iou.end(), [imgId, catId](const float& i) {
      //       std::cout << imgId << ", " << catId << ": " << i << std::endl;
      //     });
      // PRINT("current key: ", keys);
      // PRINT("Size of IOU: ", current_per_img_cat_id_iou.size());
      auto x = count_x(gt[imgId].catids, catId);

      // getting True positives upon IOU threshold.

      std::vector<int> matches;
      // PRINT("Size of ")
      std::transform(
          current_per_img_cat_id_iou.begin(), current_per_img_cat_id_iou.end(),
          std::back_inserter(matches), [thres](float iou) {
            return iou > thres[0];  // left to iterate over the thresholds
          });
      // PRINT("Matches size: ", matches.size());

      TP += std::accumulate(matches.begin(), matches.end(), 0);
      assert(std::accumulate(matches.begin(), matches.end(), 0) <= x);
      dt_percat += matches.size();

      per_sample_tp_dt[catId].push_back({TP, dt_percat});
      // total samples of a category in the entire dataset
      gt_percat += x;
    }
    SEPARATOR;
    truePos[catId] = TP;
    total_dt[catId] = dt_percat;
    total_gt[catId] = gt_percat;
    PRINT("Catid: ", catId);
    SEPARATOR;
    PRINT("TP: ", TP);
    PRINT("TP+FN: ", gt_percat);
    SEPARATOR;
    break;
  }
  float P, R;
  for (auto&& [cats, tp_dt_per_cat] : per_sample_tp_dt) {
    for (int i = 0; i < tp_dt_per_cat.size(); ++i) {
      P = static_cast<float>(tp_dt_per_cat[i].x) /
          (static_cast<float>(tp_dt_per_cat[i].y) + 0.00001);
      R = static_cast<float>(tp_dt_per_cat[i].x) /
          (static_cast<float>(total_gt[cats]) + 0.00001);
      pr[cats].push_back({P, R});
    }
    // break;
  }

  // for_each(pr.begin(), pr.end(),
  //          [](std::pair<const int, std::vector<point<float, float>>> a) {
  //            std::cout << a.first << ": ";
  //            for_each(a.second.begin(), a.second.end(),
  //                     [](point<float, float> j) { std::cout << j; });
  //            SEPARATOR;
  //          });
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
  // auto test = ious.find({139, 62});
  // std::for_each(
  //     test->second.begin(), test->second.end(), [](std::vector<float> a) {
  //       for_each(a.begin(), a.end(), [](float& j) { std::cout << j << ", ";
  //       }); std::cout << std::endl; SEPARATOR;
  //     });
  // PRINT("Size of dict<imgId><catId> of IOUs: ", ious.size());
  std::generate_n(iouThrs.begin(), iouThrs.size(), [IOU_range]() {
    static float iouThreshold = IOU_range[0] - IOU_range[2];
    iouThreshold += IOU_range[2];
    return iouThreshold;
  });

  auto preci_recall_vals = precision_recall(iouThrs);

  // std::vector<Key> keys;

  // std::transform(ious.begin(), ious.end(), std::back_inserter(keys),
  //                [](const auto& pair) { return pair.first; });
  // std::for_each(ious.begin(), ious.end(),
  //               [](const auto& pair) { std::cout << pair.first; });
  // auto it = std::max_element(keys.begin(), keys.end());
  // PRINT("Max catid", *it);
  // filter(dt, score_thres); // using the fact that the data is already sorted.

  // PRINT("size of dt", dt->size());
  // computemAP(IOU_thres);
}

void coco::loadRes(const std::string resFile, std::string flag) {
  std::fstream file(resFile);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file:" + resFile);
  }
  std::cout << "Loading results to memory..." << std::endl;
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
    dt[{image_id, category_id}].push_back({image_id, category_id, score, bbox});
    // dt[imgId].imgid = imgId;
    // dt[imgId].catids.push_back(category_id);
    // dt[imgId].scores.push_back(score);
  }
  /*
  would have to add a sort function if you want to draw bbox on the images.
  */
  // for (auto&& [key, value] : dt) {
  //   PRINT(std::to_string(key) + ": ", value.len);
  // }

  SEPARATOR;
}
coco::~coco() {}
