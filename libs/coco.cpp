
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

      gt[image_id].bbox.push_back(EXTRACT(bbox, bbox, ann));
      gt[image_id].imgid = image_id;
      gt[image_id].catids.push_back(
          static_cast<float>(EXTRACT(category_id, category_id, ann)));
    }
  }
  // Print the size of the maps.
  PRINT("Number of images lable pair: ", gt.size());
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

void coco::evaluation(float score_thres, float IOU_thres) {
  // assert(dt.size() == imgs.size());

  // filter(dt, score_thres); // using the fact that the data is already sorted.

  // PRINT("size of dt", dt->size());
  computemAP(IOU_thres);
}
void coco::computemAP(float thres) {
  std::cout << "Computing IOUs for every detection to ground truth"
            << std::endl;
  // std::map<std::pair<int, float>, std::vector<float>> ious;
  double mAP = 0;
  for (const auto& [d_imgId, d] : dt) {
    double AP = 0;
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
  SEPARATOR;
  // dt = detections;
}
coco::~coco() {}
