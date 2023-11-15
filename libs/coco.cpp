
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
      gt[image_id].bbox.push_back(EXTRACT(bbox, bbox, ann));
      gt[image_id].imgid = image_id;
      gt[image_id].catids.push_back(category_id);
      catToImgs[category_id].push_back(image_id);
    }
  }

  if (dataset.contains("images")) {
    for (auto& img : dataset["images"]) {
      int id = img["id"];
      imgs[id].push_back(img);
    }
  }

  if (dataset.contains("categories")) {
    for (auto& cat : dataset["categories"]) {
      int id = cat["id"];
      cats[id] = cat;
    }
  }

  // Print the size of the maps.
  // PRINT("Number of annotations: ", anns.size());
  PRINT("Number of images lable pair: ", gt.size());
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
// void coco::filter(const std::shared_ptr<_map_label> original,
//                   const float thres) {
//     auto clipped = std::make_shared<_map_label>();
//   // std::ofstream out("test.json");

//     for (const auto& [imgId, datas] : *original) {
//       auto boundingBoxes = datas.bbox;
//       auto scores = datas.scores;
//       auto catids = datas.catids;

//       // iterate over the bounding boxes and scores, adding them to the
//       clipped
//       // image detection
//       for (size_t i = 0; i < boundingBoxes.size(); i++) {
//         if (scores[i] >= thres) {
//           clipped->operator[](imgId).bbox.push_back(boundingBoxes[i]);
//           clipped->operator[](imgId).catids.push_back(catids[i]);
//           clipped->operator[](imgId).scores.push_back(scores[i]);
//         }
//       }
//     }
//     // out << clipped;
//     // PRINT("filtered array", clipped->size());
//     dt = clipped;
// }
void coco::precision_recall(const std::vector<float>& thres) {
  float mAP = 0;
  std::vector<int> truePos(90), total_gt(90), total_dt(90);
  for (const auto& [catId, imgIds] : catToImgs) {
    int TP = 0, gt_percat = 0, dt_percat = 0;
    for (const auto& imgId : imgIds) {
      auto detection_ann = dt.find(imgId);
      auto ground_ann = gt.find(imgId);
      if (ground_ann == gt.end()) {
        continue;
      }
      for (int i = 0; i < ground_ann->second.bbox.size(); ++i) {
        std::vector<float> io;
        auto a = ground_ann->second.bbox[i];  // can add another filter by catid
        std::transform(detection_ann->second.bbox.begin(),
                       detection_ann->second.bbox.end(), std::back_inserter(io),
                       [this, a, thres](std::vector<float> b) {
                         return coco::iou(a, b) > thres[0];
                       });
        TP += std::accumulate(io.begin(), io.end(), 0.0f);
        gt_percat += ground_ann->second.catids.size();
        dt_percat += io.size();
        // PRINT("Size of IOus", io.size());
        // ious[std::make_pair(d_imgId, g->second.catids[i])] = io;
      }
      truePos[catId] = TP;
      total_dt[catId] = dt_percat;
      total_gt[catId] = gt_percat;
    }
  }
}
void coco::evaluation(const float* IOU_range) {
  assert(dt.size() == imgs.size());
  int rng = (IOU_range[1] - IOU_range[0]) / IOU_range[2];
  std::vector<float> iouThrs(rng);

  std::generate_n(iouThrs.begin(), iouThrs.size(), [IOU_range]() {
    static float iouThreshold = IOU_range[0] - IOU_range[2];
    iouThreshold += IOU_range[2];
    return iouThreshold;
  });
  precision_recall(iouThrs);
  // std::vector<int> keys;

  // std::transform(catToImgs.begin(), catToImgs.end(),
  // std::back_inserter(keys),
  //                [](const auto& pair) { return pair.first; });
  // auto it = std::max_element(keys.begin(), keys.end());
  // PRINT("Max catid", *it);
  // filter(dt, score_thres); // using the fact that the data is already sorted.

  // PRINT("size of dt", dt->size());
  // computemAP(IOU_thres);
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
  PRINT("Completed IOU calculation", "");
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
  std::cout << "loaded to memory!! " << std::endl;

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
      // auto scores = ann["score"];
      dt[imgId].bbox.push_back(EXTRACT(bbox, bbox, ann));
      dt[imgId].imgid = imgId;
      dt[imgId].catids.push_back(EXTRACT(category_id, category_id, ann));
      dt[imgId].scores.push_back(EXTRACT(score, score, ann));
    }
  }

  // dt = detections;
}
coco::~coco() {}
