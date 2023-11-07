# cmake_minimum_required(VERSION 2.8.12)
# project(main) 
# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/libs )
# add_executable(  main main.cpp ${CMAKE_SOURCE_DIR}/libs/dataloader.cpp )
# target_link_libraries( main   ${OpenCV_LIBS} )
# # include_directories( main ${CMAKE_SOURCE_DIR}/lib)
cmake_minimum_required(VERSION 2.8.12)
project(main)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

include_directories(libs)

add_subdirectory(libs)

add_executable(main main.cpp)
target_link_libraries(main libs ${OpenCV_LIBS})

