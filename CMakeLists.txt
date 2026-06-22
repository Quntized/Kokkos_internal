cmake_minimum_required(VERSION 3.21)
project(kokkos_internal CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(KOKKOS_FETCH_TAG  "5.1.1" CACHE STRING "Kokkos git tag to fetch")
find_package(Kokkos 5.1.1 QUIET)
if(NOT Kokkos_FOUND)
  message(STATUS "Kokkos not found locally - fetching tag ${KOKKOS_FETCH_TAG} via FetchContent")
  FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        ${KOKKOS_FETCH_TAG}
    GIT_SHALLOW    ON
  )
  FetchContent_MakeAvailable(kokkos)
else()
  message(STATUS "Using pre-installed Kokkos: ${Kokkos_DIR}")
endif()

enable_testing()

add_subdirectory(test_dyn_rank)
