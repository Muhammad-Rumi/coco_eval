
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

  if (dataset.contains("annotations")) {
    for (auto& ann : dataset["annotations"]) {
      int image_id = ann["image_id"];
      int id = ann["id"];
      int category_id = ann["category_id"];

      imgToAnns[image_id].push_back(ann);
      anns[id].push_back(ann);
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
  std::cout << "Number of annotations: " << anns.size() << std::endl;
  std::cout << "Number of images: " << imgs.size() << std::endl;
  std::cout << "Number of categories: " << cats.size() << std::endl;
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
void coco::loadRes(const std::string resFile) {
  std::fstream file(resFile);
  std::cout << "Loading results to memory..." << std::endl;
  json result = json::parse(file);
  std::vector<int> annsImgIds;
  for (const auto& ann : result) {
    annsImgIds.push_back(ann["image_id"]);
  }
}

coco::~coco() {}
