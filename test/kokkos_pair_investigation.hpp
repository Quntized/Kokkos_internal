#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>
#include <gtest/gtest.h>
#include <mdspan/mdspan.hpp>
#include <utility>
namespace Test {
TEST(kokkos_pair, host_execution) {
  Kokkos::pair<int, double> p_default;
  p_default.first = 10;
  p_default.second = 3.14;
  EXPECT_EQ(p_default.first, 10);
  EXPECT_DOUBLE_EQ(p_default.second, 3.14);

  Kokkos::pair<int, double> c_param(1, 2.0);
  EXPECT_EQ(c_param.first, 1);
  EXPECT_DOUBLE_EQ(c_param.second, 2.0);

  Kokkos::pair<int, double> c_copy(c_param);
  EXPECT_EQ(c_copy.first, 1);
  EXPECT_DOUBLE_EQ(c_copy.second, 2.0);

  std::pair<int, double> spair(12, 11.0);
  Kokkos::pair<int, double> k_from_spair(spair);
  EXPECT_EQ(k_from_spair.first, 12);
  EXPECT_DOUBLE_EQ(k_from_spair.second, 11.0);

  std::pair<int, double> st_f_k = c_param.to_std_pair();
  EXPECT_EQ(st_f_k.first, 1);
  EXPECT_DOUBLE_EQ(st_f_k.second, 2.0);

  Kokkos::pair deduction(c_param);
  EXPECT_EQ(deduction.first, 1);
  EXPECT_DOUBLE_EQ(deduction.second, 2.0);
}
void test_kokkos_pair_device() {
  Kokkos::View<Kokkos::pair<int, double> *> d_result("in device", 1);
  Kokkos::parallel_for(
      "TestKokkos", 1, KOKKOS_LAMBDA(const int i) {
        Kokkos::pair<int, double> p1(3, 4.0);
        Kokkos::pair<int, double> p2(p1);
        Kokkos::pair<int, double> p3;
        p3 = p2;
        d_result(i) = p3;
      });
  Kokkos::fence();
  auto h_res =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_result);
  EXPECT_EQ(h_res(0).first, 3);
  EXPECT_DOUBLE_EQ(h_res(0).second, 4.0);
}
TEST(kokkos_pair, device_execution) { test_kokkos_pair_device(); }
TEST(first_of_kokkos_detail, host_execution) {
  Kokkos::pair<int, double> p(11, 12.0);
  auto first_pair = Kokkos::detail::first_of(p);
  EXPECT_EQ(first_pair, 11);
}
TEST(second_of_kokkos_detail, host_execution) {
  Kokkos::pair<int, double> p(11, 12.0);
  struct Dummy {};
  Dummy dummy;
  std::integral_constant<size_t, 0> dummy_int_con;
  auto second = Kokkos::detail::last_of(dummy_int_con, dummy, p);
  EXPECT_DOUBLE_EQ(second, 12.0);
}
#ifdef KOKKOS_ENABLE_CUDA
static void test_device_pair_detail_test() {
  Kokkos::pair<int, double> p(1, 2.0);
  Kokkos::View<double *> v("initiate", 2);
  Kokkos::parallel_for(
      "test_detail_Kokkos_pair_in_view_in_mdspan_folder", 1,
      KOKKOS_LAMBDA(const int i) {
        auto first = Kokkos::detail::first_of(p);
        struct DUMMY {};
        DUMMY dummy;
        std::integral_constant<size_t, 0> dummy_int_const;
        auto second = Kokkos::detail::last_of(dummy_int_const, dummy, p);
        v(0) = first;
        v(1) = second;
      });
  Kokkos::fence();
  auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  EXPECT_EQ(host_view(0), 1);
  EXPECT_EQ(host_view(1), 2.0);
}
TEST(first_last_of_kokkos_detail, device_execution) {
  test_device_pair_detail_test();
}

#endif
TEST(kokkos_index_pair_like, host_execution) {
  auto val =
      Kokkos::detail::index_pair_like<Kokkos::pair<int, double>, int>::value;
  EXPECT_EQ(val, true);
}

TEST(mdspan_inside_kokkos, execute) {
  std::vector<double> raw_data = {1.0, 2.0, 3.0, 4.0};
  using MyExtents = MDSPAN_IMPL_STANDARD_NAMESPACE::extents<int, 2, 2>;
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<double, MyExtents> m(raw_data.data());
  EXPECT_EQ(m(0, 0), 1.0);
  EXPECT_EQ(m(0, 1), 2.0);
}

} // namespace Test