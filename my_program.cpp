
#include <iostream>
#include <string>
#include <vector>

#include "loadAnns.h"
using namespace std;

int main() {
  // Load the COCO annotations file.
  std::string annFile = "instances_val2017.json";
  std::vector<coco_ann_t> anns;
  coco_ann_t* annsPtr = loadAnns(annFile.c_str());
  if (annsPtr != nullptr) {
    anns.push_back(*annsPtr);
    free(annsPtr);
  }
  // Iterate over the COCO annotations.
  for (int i = 0; i < anns.size(); i++) {
    // Print the annotation information.
    cout << "Annotation ID: " << anns[i].id << endl;
    cout << "Image ID: " << anns[i].image_id << endl;
    cout << "Category ID: " << anns[i].category_id << endl;
  }

  return 0;
}
