
#include "data_loader.hpp"  // NOLINT
#include "lib.hpp"          // NOLINT

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "Usage:" << argv[0]
              << " requires <score_threshold> <IOU_threshold> <inference mode "
                 "float/int> <dataset path>"
              << std::endl;
    return -1;
  }
  const std::string file = "dataset/instances_val2017.json";
  const std::string results = "dataset/detection_truth.json";
  const std::string model = "model/efficientdet-lite1.onnx";

  coco test(file);
  test.loadRes(argv[4], argv[3]);
  test.evaluation(std::stof(argv[1]), std::stof(argv[2]));
}
