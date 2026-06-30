#include "kokkos_pair_investigation.hpp"
#include "kokkos_raw_mdspan.hpp"
#include "test_drank.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}