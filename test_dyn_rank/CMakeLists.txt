cmake_minimum_required(VERSION 3.21)

if(NOT PROJECT_NAME)
  project(test_dyn_rank CXX)
  set(CMAKE_CXX_STANDARD 20)
  find_package(Kokkos 5.1.1 REQUIRED)

  enable_testing()
endif()

add_executable(test_dyn_rank test_drank.cpp)
target_link_libraries(test_dyn_rank PRIVATE Kokkos::kokkos)

add_test(
  NAME    test_dyn_rank
  COMMAND test_dyn_rank
)
