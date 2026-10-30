cmake_minimum_required(VERSION 3.8)
project(sim2real_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "msg/TrajectoryStep.msg"
  "msg/RolloutEpisode.msg"
  "msg/FilterMetrics.msg"
  "msg/SafetyStatus.msg"
  "srv/ResetEnv.srv"
  "srv/GenerateRollouts.srv"
  DEPENDENCIES
  builtin_interfaces
  geometry_msgs
  std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    test_msg_roundtrip
    test/test_msg_roundtrip.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py
  )
endif()

ament_package()
