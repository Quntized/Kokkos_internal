cmake_minimum_required(VERSION 3.20)
project(test_dual_rank CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Kokkos 5.1.0) # Find Kokkos version 5.0 or any newer version
add_executable(tests test_drank.cpp)
target_link_libraries(tests PRIVATE Kokkos::kokkos)
#set_target_properties(test_dual_rank PROPERTIES LINKER_LANGUAGE CXX)
