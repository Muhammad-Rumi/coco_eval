
#include "lib.hpp"

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
      imgs[id].push_back( img);
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
coco::~coco() {}
