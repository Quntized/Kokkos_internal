#include "desul/desul_test.hpp"
#include "kokkos_core/kokkos_pair_investigation.hpp"
#include "mdspan/kokkos_raw_mdspan.hpp"
#include "kokkos_core/test_drank.hpp"
#include "kokkos_core/View_data_analysis_kokkos_investigation.hpp"
#include "mdspan/kokkos_extent_inv.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
