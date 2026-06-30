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
  std::cout << val_2 << std::endl; // maybe some work can be done there
  using stat_arr = mdspan_detail::static_array<int, 10, 20, 30>;
  constexpr auto va = stat_arr::size();
  static_assert(va == 3, "Should match 3");
  EXPECT_EQ(stat_arr::get<2>(), 30);
  using ind_seq = mdspan_detail::index_sequence_scan_impl<0, 1, 2, 3>;
  auto constexpr val_in = ind_seq::get(2);
  EXPECT_EQ(val_in, 3);
  using RF = mdspan_detail::index_sequence_scan_impl<0, 42>;
  EXPECT_EQ(RF::get(2), 0);
  using ind_seq_test = mdspan_detail::index_sequence_scan_impl<42, 1, 2, 3, 4>;
  auto constexpr val_in_test = ind_seq_test::get(43);
  std::cout << "val_in_test" << val_in_test << std::endl;
  // EXPECT_EQ(val_in_test,3);
  using empty_arr = mdspan_detail::possibly_empty_array<int, 10>;
  empty_arr arr;
  std::cout << arr[2] << std::endl;
  EXPECT_EQ(arr[2], NULL);
}
TEST(maybe_static_array, line_206_start) {
  using HybridArray =
      mdspan_detail::maybe_static_array<size_t, size_t, dynamic_extent, 10,
                                        dynamic_extent, 30>;
  // HybridArray my_extents{50};
  // using static_val_type = HybridArray::static_vals_t;
  // using value_type = static_val_type::value_type;   inaccessible as private
  // static_assert(std::is_same_v<value_type,size_t>,"should Matched");
  using val_type = HybridArray::value_type;
  using stat_value_type = HybridArray::static_value_type;
  static_assert(std::is_same_v<val_type, size_t>, "should Matched");
  static_assert(std::is_same_v<stat_value_type, size_t>, "should Matched");
  auto constexpr dyn_tag = HybridArray::tag_value;
  static_assert(dyn_tag == dynamic_extent, "dynamic tag");
  EXPECT_EQ(MDSPAN_PP_CAT_IMPL(1, 2), 12);
  EXPECT_EQ(MDSPAN_PP_CAT(1, 2), 12);
  using maybe_static_test =
      mdspan_detail::maybe_static_array<size_t, size_t, dynamic_extent, 10, 10,
                                        dynamic_extent>;
  maybe_static_test mst(10);
  using maybe_static_test_2 =
      mdspan_detail::maybe_static_array<size_t, size_t, dynamic_extent, 10, 10,
                                        20>;
  maybe_static_test_2 mst_2(10, 20, 30);
  SUCCEED();
}
MDSPAN_TEMPLATE_REQUIRES(class T,
                         /*requires*/ (std::is_integral<T>::value))
int type_checker(T /*value*/) { return 1; }
MDSPAN_TEMPLATE_REQUIRES(class T,
                         /*requires*/ (std::is_floating_point<T>::value))
int type_checker(T /*value*/) { return 2; }
template <size_t... values> struct vartempfold {
  static constexpr size_t sum = MDSPAN_IMPL_FOLD_PLUS_RIGHT(values, 0);
};
TEST(macro_testing, for_line_204_218_in_macros_hpp) {
  EXPECT_EQ(type_checker(42), 1);
  EXPECT_EQ(type_checker(3.4), 2);
  auto constexpr sum_ = vartempfold<10, 20, 30>::sum;
  EXPECT_EQ(sum_, 60);
} // https://github.com/kokkos/mdspan/blob/3fdf85b01e10629ddb18a0a3ffd468d7f9cfa185/include/experimental/__p0009_bits/macros.hpp#L204-L218
TEST(cpp_version_checking, version_for_quntized_using) {
  long cxx_version = __cplusplus;
  std::cout << "\n __cplusplus macro: " << std::endl;
  if (cxx_version >= 202302L)
    std::cout << "\n C++23" << std::endl;
  else if (cxx_version >= 202002L)
    std::cout << "C++20" << std::endl;
  else if (cxx_version >= 201703L)
    std::cout << "C++17" << std::endl;
  else if (cxx_version >= 201402L)
    std::cout << "C++14" << std::endl;
  else
    std::cout << "Before C++14 (Kokkos may fail)" << std::endl;
  SUCCEED();
}
} // namespace Test