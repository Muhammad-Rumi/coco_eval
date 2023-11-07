
#include "data_loader.hpp"  // NOLINT
#include "lib.hpp"  // NOLINT
int main() {
  std::string file = "dataset/instances_val2017.json";
  std::string img_data = "dataset/val2017";
  coco test(file);
  data_loader tests(img_data, 1);
  for (int i = 0; i < 10; ++i) {
    auto data = tests.next();
    std::cout << data.size() << std::endl;
  }
}
