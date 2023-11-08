
// #include <onnxruntime/core/providers.h>
// #include <onnxruntime/core/session.h>
// #include <onnxruntime/core/tensor.h>
// #include <onnxruntime_cxx_api.h>

// #include <onnxruntime_cxx/onnxruntime_cxx.h>
#include "data_loader.hpp"  // NOLINT
#include "lib.hpp"          // NOLINT

std::vector<cv::Mat> infer(const cv::Mat& input_image, cv::dnn::Net& net);

int main() {
  const std::string file = "dataset/instances_val2017.json";
  const std::string img_data = "dataset/val2017";
  const std::string model = "model/efficientdet-lite1.onnx";

  coco test(file);
  std::cout << "Loading model" << std::endl;
  data_loader tests(img_data, 1);

  auto det_lite1 = cv::dnn::readNetFromONNX(model);

  for (int i = 0; i < 10; ++i) {
    auto data = tests.next();

    std::cout << "Image ID :" << data[0].second << std::endl;
    // auto detections = infer(data[0].first, det_lite1);
  }
}

std::vector<cv::Mat> infer(const cv::Mat& input_image, cv::dnn::Net& net) {
  cv::Mat blob;
  cv::dnn::blobFromImage(input_image, blob, 1., cv::Size(), cv::Scalar(), true,
                         false);
  net.setInput(blob);

  // Forward propagate.
  std::vector<cv::Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());
  std::cout << "outputs size:" << outputs[0].size << std::endl;
  return outputs;
}
