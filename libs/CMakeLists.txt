cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS}  )
add_library(libs SHARED  data_loader.hpp dataloader.cpp  coco.cpp)

