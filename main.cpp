
#include "data_loader.hpp"  // NOLINT
#include "lib.hpp"          // NOLINT

int main() {
  const std::string file = "dataset/instances_val2017.json";
  const std::string results = "dataset/data.json";
  const std::string model = "model/efficientdet-lite1.onnx";

  coco test(file);
  test.loadRes(results);
  test.evalutaion(0.5, 0.5);
}
