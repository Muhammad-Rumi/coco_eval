
#include "lib.hpp"  // NOLINT

template <typename _t, typename _y, typename _r>
int table<_t, _y, _r>::inst = 0;

coco::coco(const std::string& annotation_file) {
  std::fstream file(annotation_file);
  std::cout << "Loading annnotations to memory..." << std::endl;
  dataset = json::parse(file);
  alo = 0;
  create_index();
}
void coco::create_index() {
  std::cout << "Creating Index... " << std::endl;

  std::vector<float> bbox;
  int category_id;
  if (dataset.contains("annotations")) {
    for (auto& ann : dataset["annotations"]) {
      int image_id = ann["image_id"];
      int id = ann["id"];
      EXTRACT(category_id, category_id, ann);
      EXTRACT(bbox, bbox, ann);
      gt[image_id].len++;
      gt[image_id].bbox.push_back(bbox);
      gt[image_id].imgid = image_id;
      gt[image_id].catids.push_back(category_id);
      catToImgs[category_id].push_back(image_id);
    }
  }
  // Print the size of the maps.
  PRINT("Number of images lable pair: ", gt.size());
  PRINT("Number of categories: ", catToImgs.size());
  SEPARATOR;
}
float coco::iou(const std::vector<float>& gt_bbox,
                const std::vector<float>& dt_bbox) {
  /*
  input:
  gt_bbox = [xmin, ymin, w, h]
  dt_bbox = [ymin, xmin, ymax, xmax]
  ouput:
  float iou;
  */
  // PRINT("gt_bbox", gt_bbox.size());
  // PRINT("dt_bbox", dt_bbox.size());
  assert(gt_bbox.size() == 4 && dt_bbox.size() == 4);
  float x1, y1, x2, y2;
  x1 = std::max(gt_bbox[0], dt_bbox[1]);
  y1 = std::max(gt_bbox[1], dt_bbox[0]);
  x2 = std::min(gt_bbox[0] + gt_bbox[2], dt_bbox[2]);
  y2 = std::min(gt_bbox[1] + gt_bbox[3], dt_bbox[3]);

  float intersection_area = (x2 - x1) * (y2 - y1);
  float area1 = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1]);
  float area2 = (dt_bbox[2] - dt_bbox[0]) * (dt_bbox[3] - dt_bbox[1]);

  return intersection_area / (area1 + area2 - intersection_area);
}
coco::_per_img_cat coco::calculate_confusion(const float& thres) {
  _per_img_cat test;
  for (const auto& [imgId, d] : dt) {
    const auto& ground_ann = gt.find(imgId);
    if (ground_ann == gt.end()) {
      continue;
    }
    int TP = 0, gt_per_img_cat = 0, dt_per_img_cat = 0;
    for (int i = 0; i < ground_ann->second.len; ++i) {
      std::vector<float> io;
      auto a = ground_ann->second.bbox[i];  // some bug here
      auto catId = ground_ann->second.catids[i];
      std::transform(d.bbox.begin(), d.bbox.end(), std::back_inserter(io),
                     [this, a, thres](std::vector<float> b) {
                       return coco::iou(a, b) > thres;
                     });
      // per image per category
      TP = std::accumulate(io.begin(), io.end(), 0.0f);
      gt_per_img_cat = ground_ann->second.catids.size();
      dt_per_img_cat = io.size();
      //
      _table_scaler confusion_mat(0, 0, 0);  // per image tp fp fn;
      confusion_mat.truePos = TP;
      confusion_mat.total_dt = dt_per_img_cat;
      confusion_mat.total_gt = gt_per_img_cat;
      test[{imgId, catId}].push_back(confusion_mat);  // auto key_val =
      //     "{" + std::to_string(imgId) + " ," + std::to_string(catId) + "}";
      // PRINT(" At Current key the value is saved as" , key_val);
    }
  }

  return test;
}
coco::_processed_vla coco::precision_recall(const std::vector<float>& thres) {
  std::cout << "Calculating precision recall" << std::endl;

  _processed_vla process_values;
  float optimal_thres = 0, F1 = 0;

  for (auto& ith_thres : thres) {
    PRINT("current threshold", ith_thres);
    auto curr_confusion = calculate_confusion(ith_thres);
    // PRINT("size of map", curr_confusion.size());
    point<float, float> precision_recall;  // x will have recall
    // int inst;
    float avg_recall = 0, avg_precision = 0;
    // per image per category
    for (const auto& [key, value] : curr_confusion) {
      float recall = 0, precision = 0;
      float len = value.size();
      for (auto&& per_img_cat : value) {
        // if (per_img_cat.truePos == 0) continue;
        recall += per_img_cat.truePos / per_img_cat.total_gt;
        precision += per_img_cat.truePos / per_img_cat.total_dt;
        // inst = per_img_cat.inst;
      }
      avg_recall += recall / len;
      avg_precision += precision / len;
    }
   
    // recall
    precision_recall.x = avg_recall / static_cast<float>(dt.size());

    // precision
    precision_recall.y = avg_recall / static_cast<float>(dt.size());
    // F1 score per category
    auto temp = (2.0f * precision_recall.x * precision_recall.y) /
                (precision_recall.x + precision_recall.y);
    if (temp > F1) {
      F1 = temp;
      optimal_thres = ith_thres;
      process_values.x = ith_thres;
    }
    PRINT("Result", precision_recall);
    // add to graph points
    process_values.y.push_back(precision_recall);

    // zero_count(curr_confusion.total_dt);
  }
  PRINT("best IOU threshold", process_values.x);
  PRINT("size:", dt.size());
  SEPARATOR;
  return process_values;
}
void coco::evaluation(const float* IOU_range) {
  /*
    IOU_range = [start, stop, step]
  */
  int rng = (IOU_range[1] - IOU_range[0]) / IOU_range[2];
  std::vector<float> iouThrs(rng);

  std::generate_n(iouThrs.begin(), iouThrs.size(), [IOU_range]() {
    static float iouThreshold = IOU_range[0] - IOU_range[2];
    iouThreshold += IOU_range[2];
    return iouThreshold;
  });
  auto pre_rcal_thres = precision_recall(iouThrs);
  computemAP(pre_rcal_thres.x);
}
float coco::computemAP(float thres) {  // will have to change.
  std::cout << "Computing IOUs for every detection to ground truth"
            << std::endl;
  // std::map<std::pair<int, float>, std::vector<float>> ious;
  float mAP = 0;
  for (const auto& [d_imgId, d] : dt) {
    float AP = 0;
    auto g = gt.find(d_imgId);
    if (g == gt.end()) {
      continue;
    }
    for (int i = 0; i < g->second.bbox.size(); ++i) {
      std::vector<float> io;
      const auto& a = g->second.bbox[i];  // can add another filter by catid
      std::transform(d.bbox.begin(), d.bbox.end(), std::back_inserter(io),
                     [this, a, thres](std::vector<float> b) {
                       return coco::iou(a, b) > thres;
                     });
      AP += std::accumulate(io.begin(), io.end(), 0.0f) / io.size();
      // PRINT("Size of IOus", io.size());
      // ious[std::make_pair(d_imgId, g->second.catids[i])] = io;
    }

    AP = AP / g->second.catids.size();
    // AP/= 80;
    mAP += AP;
    // PRINT("catids sizes for every image", g.catids.size());
    // break;
  }
  mAP /= dt.size();
  PRINT("mAP", mAP * 100);
  // PRINT("IOus with unique labels", ious.size());
  SEPARATOR;
  return mAP;
}

void coco::loadRes(const std::string resFile,
                   std::string flag) {  // need to be completed.
  std::fstream file(resFile);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file:" + resFile);
  }
  std::cout << "Loading results to memory..." << std::endl;
  json result = json::parse(file);

  if (flag == "float") {
    label temp;
    std::map<int, label> detections;
    int id = 0;
    for (const auto& [imgid, ann] : result.items()) {
      temp.imgid = std::stoi(imgid);
      EXTRACT(temp.scores, scores, ann);
      EXTRACT(temp.bbox, bbox, ann);
      EXTRACT(temp.catids, catids, ann);
      temp.len++;
      dt[temp.imgid] = temp;
    }

  } else {
    std::vector<float> bbox;
    int category_id;
    float score;
    std::cout << "Processing... " << std::endl;
    for (const auto& ann : result) {
      int imgId = ann["image_id"];
      // auto scores = ann["score"];
      dt[imgId].bbox.push_back(EXTRACT(bbox, bbox, ann));
      dt[imgId].imgid = imgId;
      dt[imgId].catids.push_back(EXTRACT(category_id, category_id, ann));
      dt[imgId].scores.push_back(EXTRACT(score, score, ann));
      dt[imgId].len++;
    }
  }
  // std::sort(dt.begin(), dt.end(), [](a[imgId].scores) {}); SEPARATOR;
  // dt = detections;
}
coco::~coco() {}
