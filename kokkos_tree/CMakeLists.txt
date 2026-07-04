cmake_minimum_required(VERSION 3.21)
include(GoogleTest)
if(NOT PROJECT_NAME)
  project(test_dyn_rank CXX)
  set(CMAKE_CXX_STANDARD 20)
  find_package(Kokkos 5.1.1 REQUIRED)

  enable_testing()
endif()

add_executable(Kokkos_qun_test test_dyn_rank.cpp)
target_link_libraries(Kokkos_qun_test PRIVATE Kokkos::kokkos GTest::gtest)

add_test(
  NAME    Kokkos_qun_test
  COMMAND Kokkos_qun_test
)
