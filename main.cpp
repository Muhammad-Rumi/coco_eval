
#include "data_loader.hpp"  // NOLINT
#include "lib.hpp"          // NOLINT

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr << "Usage:" << argv[0]
              << " requires <IOU_thres start range> <IOU_thres end range> "
                 "<IOU_thres step> <inference mode "
                 "float/int> <dataset path>"
              << std::endl;
    return -1;
  }
  const std::string file = "dataset/instances_val2017.json";
  const std::string results = "dataset/detection_truth.json";
  const std::string model = "model/efficientdet-lite1.onnx";

  coco test(file);
  test.loadRes(argv[5], argv[4]);
  float iourange[] = {std::stof(argv[1]), std::stof(argv[2]),
                      std::stof(argv[3])};
  // std::vector<float> bbox1 = {10, 20, 40, 10};
  // std::vector<float> bbox2 = {20, 25, 50, 10};
  // auto x = test.iou(bbox1, bbox2);
  // PRINT("iou of dummy boxes: ", x);

  test.evaluation(iourange);
  // std::cout << "ImgIds" << std::endl;
  // std::for_each(test.imgIds.begin(), test.imgIds.end(),
  //               [](int a) { std::cout << a << ", " ;});
  // SEPARATOR;
  // std::cout << "CatIds" << std::endl;
  // std::for_each(test.catIds.begin(), test.catIds.end(),
  //               [](int a) { std::cout << a << ", " ;});
}
