#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>
#include <gtest/gtest.h>
#include <mdspan/mdspan.hpp>
#include <type_traits>
#include <utility>

namespace Test {

namespace mdspan_detail = MDSPAN_IMPL_STANDARD_NAMESPACE::detail;
using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;
TEST(mdspan_extent_extent_compatibility,
     include_experimental__p0009_bits_extentshpp) {
  static_assert(mdspan_detail::impl_compare_extent_compatible<3, 3>::value,
                "Equal static extent should be compatible");
  static_assert(!mdspan_detail::impl_compare_extent_compatible<3, 4>::value,
                "Unequal static extent should not be compatible");
  static_assert(
      mdspan_detail::impl_compare_extent_compatible<dynamic_extent, 4>::value,
      "Dynamic extent should be compatible with static extent");
  static_assert(
      mdspan_detail::impl_compare_extent_compatible<10, dynamic_extent>::value,
      "Static extent should be compatible with dynamic extent");
  static_assert(
      mdspan_detail::impl_compare_extent_compatible<dynamic_extent,
                                                    dynamic_extent>::value,
      "Dynamic should be compatible with each other");
  SUCCEED();
}
TEST(mdspan_static_array_impl, line_79_100) {
  using Stat_arr = mdspan_detail::static_array_impl<0, size_t, 10, 20, 30>;
  static_assert(Stat_arr::get(0) == 10, "Should be 10"); // It's for runtime
  static_assert(Stat_arr::get(1) == 20, "Should be 20"); // same for this;
  static_assert(Stat_arr::get<2>() == 30, "Should be 30");
  using Stat_arr2 = mdspan_detail::static_array_impl<2, size_t, 1, 2, 3>;
  static_assert(Stat_arr2::get<2>() == 1, "Should be 1");
  // static_assert(Stat_arr2::get<1>()==1,"Should be 1");
  auto val = Stat_arr2::get(2);
  std::cout << val << std::endl;
  auto val_2 = Stat_arr2::get(1);
  std::cout << val_2 << std::endl; // maybe there work can be done
}
} // namespace Test