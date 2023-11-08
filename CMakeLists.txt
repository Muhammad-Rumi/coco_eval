# cmake_minimum_required(VERSION 2.8.12)
# project(main) 
# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/libs )
# add_executable(  main main.cpp ${CMAKE_SOURCE_DIR}/libs/dataloader.cpp )
# target_link_libraries( main   ${OpenCV_LIBS} )
# # include_directories( main ${CMAKE_SOURCE_DIR}/lib)
cmake_minimum_required(VERSION 3.10.0)
project(main)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
# include_directories(/home/user98/onnxruntime/include/onnxruntime/core/session)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib/)


# find_package( onnxruntimeConfig REQUIRED)
# target_link_libraries(my_executable ONNXRuntime::onnxruntime)

include_directories(libs)

add_subdirectory(libs)

add_executable(main main.cpp)
target_link_libraries(main libs ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS})

