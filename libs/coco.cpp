
#include "lib.hpp"  // NOLINT

coco::coco(const std::string& annotation_file) {
  std::fstream file(annotation_file);
  std::cout << "Loading annnotations to memory..." << std::endl;
  dataset = json::parse(file);
  create_index();
}
void coco::create_index() {
  std::cout << "Creating Index... " << std::endl;
  //   _vecjson ;
  label temp;
  std::vector<float> bbox;
  if (dataset.contains("annotations")) {
    for (auto& ann : dataset["annotations"]) {
      int image_id = ann["image_id"];
      int id = ann["id"];
      int category_id = ann["category_id"];
      temp.bbox = EXTRACT(bbox, ann);
      temp.catid = {static_cast<float>(EXTRACT(category_id, ann))};
      temp.imgid = image_id;
      gt[image_id] = temp;
      imgToAnns[image_id].push_back(ann);
      anns[id].push_back(ann);
      catToImgs[category_id].push_back(image_id);  // change type
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
  PRINT("Number of annotations: ", anns.size());
  PRINT("Number of images lable pair: ", gt.size());
  PRINT("Number of categories: ", cats.size());
}
float coco::iou(const std::vector<float>& gt_bbox,
                const std::vector<float>& dt_bbox) {
  /*
  input:
  gt_bbox = [ymin, xmin, ymax, xmax]
  dt_bbox = [ymin, xmin, ymax, xmax]
  ouput:
  float iou;
  */
  float area_gt = (gt_bbox[2] - gt_bbox[0]) *
                  (gt_bbox[1] - gt_bbox[3]);  // ymax - ymin * xmax -xmin
  float area_dt = (dt_bbox[2] - dt_bbox[0]) * (dt_bbox[1] - dt_bbox[3]);
  std::vector<float> upper_left;
  std::transform(gt_bbox.begin(), gt_bbox.end() - 2, dt_bbox.begin(),
                 std::back_inserter(upper_left),
                 [](float a, float b) { return std::max(a, b); });

  std::vector<float> lower_right;
  std::transform(gt_bbox.begin() + 2, gt_bbox.end(), dt_bbox.begin() + 2,
                 std::back_inserter(upper_left),
                 [](float a, float b) { return std::min(a, b); });
  assert(upper_left.size() == 2 && lower_right.size() == 2);
  float inter_area =
      (lower_right[0] - upper_left[0]) * (lower_right[1] - upper_left[1]);
  float union_area = area_dt + area_gt - inter_area;
  float iou = inter_area / union_area;
  return iou;
}
std::shared_ptr<coco::json> coco::filter(std::shared_ptr<std::map<int, label>> original,
                                         float thres) {
  auto clipped = std::make_shared<coco::json>();
  // std::ofstream out("test.json");

  for (const auto& [imgId, datas] : original->items()) {
    /* do stuff */
    auto boundingBoxes = datas["bbox"];
    auto scores = datas["scores"];
    // create a shared pointer to a new json object
    auto clippedImageDetection = std::make_shared<json>();

    // iterate over the bounding boxes and scores, adding them to the clipped
    // image detection
    for (size_t i = 0; i < boundingBoxes.size(); i++) {
      if (scores[i] >= thres) {
        clippedImageDetection->push_back({"bbox", boundingBoxes[i]});
        clippedImageDetection->push_back({"scores", scores[i]});
      }
    }
    clipped->push_back({imgId, *clippedImageDetection});
  }
  // out<< *clipped;
  PRINT("filtered array", clipped->size());
  return clipped;
}

void coco::evalutaion(float score_thres, float IOU_thres) {
  std::cout << " Running pre image evaluation... " << std::endl;
  /* for (const auto& [imgid, ann] : imgToAnns) {
  //   std::cout << imgid << ": " << ann.size() << std::endl;
  // }
  for (const auto& [key, detec] : detections->items()) {
  //   std::cout << detections->size() << ' ' << std::stoi(key) << std::endl;
  //   break;
  // }*/
  assert(detections->size() == imgs.size());
  auto cliped = filter(detections, score_thres);

}
void coco::loadRes(const std::string resFile) {  // need to be completed.
  std::fstream file(resFile);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file:" + resFile);
  }
  std::cout << "Loading results to memory..." << std::endl;
  json result = json::parse(file);
  this->detections = std::make_shared<json>(result);
}

coco::~coco() {}
