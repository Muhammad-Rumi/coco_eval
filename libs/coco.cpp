
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
  if (dataset.contains("annotations")) {
    for (auto& ann : dataset["annotations"]) {
      int image_id = ann["image_id"];
      int id = ann["id"];
      EXTRACT(category_id, category_id, ann);
      EXTRACT(bbox, bbox, ann);
      gt[image_id].bbox.push_back(bbox);
      gt[image_id].imgid = image_id;
      gt[image_id].catids.push_back(category_id);
      catToImgs[category_id].push_back(image_id);
      gt[image_id].len++;
    }
  }
  // Print the size of the maps.
  PRINT("Number of images lable pair: ", gt.size());
  SEPARATOR;
  PRINT("Number of categories: ", catToImgs.size());
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
      std::transform(
          dt_ann.bbox.begin(), dt_ann.bbox.end(), std::back_inserter(temp),
          [this, a](std::vector<float> b) { return coco::iou(a, b); });
      coco::ious[{imgId, catId}].push_back(temp);
      // PRINT("temp size", temp.size());
    }
  }
  PRINT("IOUs calculated", ious.size());
  SEPARATOR;
}
void coco::precision_recall(const std::vector<float>& thres) {
  std::cout << "Calculating precision recall" << std::endl;
  float mAP = 0;
  std::vector<int> truePos(91), total_gt(91), total_dt(91);
  for (const auto& [catId, imgIds] : catToImgs) {
    int TP = 0, gt_percat = 0, dt_percat = 0;
    for (const auto& imgId : imgIds) {
      const auto& detection_ann = dt.find(imgId);
      const auto& ground_ann = gt.find(imgId);
      if (ground_ann == gt.end()) {
        continue;
      }
      for (int i = 0; i < ground_ann->second.bbox.size(); ++i) {
        std::vector<float> io;
        auto a = ground_ann->second.bbox[i];  // can add another filter by catid
        std::transform(
            detection_ann->second.bbox.begin(),
            detection_ann->second.bbox.end(), std::back_inserter(io),
            [this, a, thres](std::vector<float> b) {
              return coco::iou(a, b) >
                     thres[0];  // left to iterate over the thresholds
            });
        TP += std::accumulate(io.begin(), io.end(), 0.0f);
        gt_percat += ground_ann->second.catids.size();
        dt_percat += io.size();
        // PRINT("Size of IOus", io.size());
        // ious[std::make_pair(d_imgId, g->second.catids[i])] = io;
        truePos[catId] = TP;
        total_dt[catId] = dt_percat;
        total_gt[catId] = gt_percat;
      }
    }
  }
  zero_count(truePos);
}
void coco::evaluation(const float* IOU_range) {
  // assert(dt.size() == imgs.size());
  int rng = (IOU_range[1] - IOU_range[0]) / IOU_range[2];
  std::vector<float> iouThrs(rng);
  get_scores();
  PRINT("Size of dict<imgId><catId> of IOUs: ", ious.size());
  std::generate_n(iouThrs.begin(), iouThrs.size(), [IOU_range]() {
    static float iouThreshold = IOU_range[0] - IOU_range[2];
    iouThreshold += IOU_range[2];
    return iouThreshold;
  });

  // precision_recall(iouThrs);
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
float coco::computemAP(float thres) {  // will have to change.
  std::cout << "Computing IOUs for every detection to ground truth"
            << std::endl;
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
    }

    AP = AP / g->second.catids.size();
    // AP/= 80;
    mAP += AP;
  }
  mAP /= dt.size();
  PRINT("mAP", mAP * 100);
  SEPARATOR;
  return mAP;
}

void coco::loadRes(const std::string resFile, std::string flag) {
  std::fstream file(resFile);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file:" + resFile);
  }
  std::cout << "Loading results to memory..." << std::endl;
  json result = json::parse(file);

  if (flag == "float") {
    label temp;
    std::map<int, label> detections;
    for (const auto& [imgid, ann] : result.items()) {
      temp.imgid = std::stoi(imgid);
      EXTRACT(temp.scores, scores, ann);
      EXTRACT(temp.bbox, bbox, ann);
      EXTRACT(temp.catids, catids, ann);
      dt[temp.imgid] = temp;
    }

  } else {
    std::vector<float> bbox;
    int category_id;
    float score;
    std::cout << "Processing... " << std::endl;
    for (const auto& ann : result) {
      int imgId = ann["image_id"];
      EXTRACT(score, score, ann);
      EXTRACT(category_id, category_id, ann);
      EXTRACT(bbox, bbox, ann);
      dt[imgId].bbox.push_back(bbox);
      dt[imgId].imgid = imgId;
      dt[imgId].catids.push_back(category_id);
      dt[imgId].scores.push_back(score);
      dt[imgId].len++;
    }
  }
  SEPARATOR;
}
coco::~coco() {}
