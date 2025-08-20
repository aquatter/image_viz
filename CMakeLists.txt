cmake_minimum_required(VERSION 3.8)
project(image_viz)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_interfaces REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED core imgproc imgcodecs highgui)

add_executable(ImageViz src/ImageViz.cpp)

target_include_directories(ImageViz PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)

target_compile_features(ImageViz PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(ImageViz ${OpenCV_LIBS} fmt::fmt)
target_include_directories(ImageViz PUBLIC ${OpenCV_INCLUDE_DIRS})

ament_target_dependencies(ImageViz rclcpp std_msgs sensor_msgs rosbag2_cpp rosbag2_interfaces)

install(TARGETS ImageViz
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
