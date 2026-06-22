cmake_minimum_required(VERSION 3.21)
project(kokkos_internal CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Fetch Kokkos via FetchContent so CI doesn't need a pre-installed Kokkos.
# Users who already have Kokkos installed can skip this by passing
#   -DKokkos_DIR=<path>
# on the cmake command line; find_package will be tried first.
# ---------------------------------------------------------------------------
include(FetchContent)

# Allow the user to override the Kokkos version fetched
set(KOKKOS_FETCH_TAG  "5.1.1" CACHE STRING "Kokkos git tag to fetch")

find_package(Kokkos 4.0 QUIET)
if(NOT Kokkos_FOUND)
  message(STATUS "Kokkos not found locally – fetching tag ${KOKKOS_FETCH_TAG} via FetchContent")
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

# ---------------------------------------------------------------------------
# Enable CTest so `ctest` picks up our tests
# ---------------------------------------------------------------------------
enable_testing()

# ---------------------------------------------------------------------------
# Subdirectories containing test targets
# ---------------------------------------------------------------------------
add_subdirectory(test_dyn_rank)
