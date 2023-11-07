#ifndef LOADANNS_H
#define LOADANNS_H

// #include <Python.h>

typedef struct {
  int id;
  int image_id;
  int category_id;
} coco_ann_t;

coco_ann_t* loadAnns(const char* annFile);

#endif
