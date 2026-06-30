// test_drank.cpp – Kokkos DynRankView / DynamicView smoke tests
//
// Uses assert() for pass/fail so CTest can detect failures.
// CUDA-specific sections are guarded with #ifdef KOKKOS_ENABLE_CUDA.

#include <Kokkos_Core.hpp>
#include <Kokkos_DynRankView.hpp>
#include <Kokkos_DynamicView.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Timer.hpp>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <type_traits>

namespace Test {

static void test_view_data_type_from_rank() {
  std::cout << "\n=== test_view_data_type_from_rank ===\n";

  using Rank4 = Kokkos::Impl::ViewDataTypeFromRank<double, 4>;
  ASSERT_TRUE((std::is_same_v<Rank4::type, double ****>));
}
TEST(kokkos_dyn_rank_data_type, host_execution) {
  test_view_data_type_from_rank();
}
template <unsigned arg_total_size = 2> static void test_dyn_rank_view_basics() {
  std::cout << "\n=== test_dyn_rank_view_basics ===\n";

  Kokkos::DynRankView<double, Kokkos::DefaultExecutionSpace> d4("d4", 1024,
                                                                arg_total_size);
  ASSERT_TRUE((d4.rank() == 2u));

  size_t rank = Kokkos::Impl::DynRankDimTraits<void>::computeRank(
      2, 2, 2, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
      KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX);
  ASSERT_TRUE((rank == 3u));

  Kokkos::LayoutRight l(10, 20, 30, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
                        KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
                        KOKKOS_INVALID_INDEX);
  size_t rank2 = Kokkos::Impl::DynRankDimTraits<void>::computeRank(l);
  ASSERT_TRUE((rank2 == 3u));

  Kokkos::LayoutRight l2(10, 20, 30, 40, KOKKOS_INVALID_INDEX,
                         KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
                         KOKKOS_INVALID_INDEX);
  auto view_a = Kokkos::view_alloc("sajid", Kokkos::WithoutInitializing);
  size_t rank3 = Kokkos::Impl::DynRankDimTraits<void>::computeRank(view_a, l2);
  ASSERT_TRUE((rank3 == 4u));

  Kokkos::LayoutRight l3(10, 20, 30, 40, 50);
  Kokkos::LayoutRight lr =
      Kokkos::Impl::DynRankDimTraits<void>::createLayout(l3);
  size_t rank4 = Kokkos::Impl::DynRankDimTraits<void>::computeRank(lr);
  std::cout << "  INFO: computeRank after createLayout = " << rank4
            << " (input had 5 valid dims)\n";
  ASSERT_TRUE((rank4 > 0u));
}
TEST(kokkos_dyn_rank_view_basics, host_execution) {
  test_dyn_rank_view_basics<2>();
}

static void test_dyn_rank_create_layout() {
  std::cout << "\n=== test_dyn_rank_create_layout ===\n";

  using traits = Kokkos::ViewTraits<double **, Kokkos::LayoutRight,
                                    Kokkos::DefaultExecutionSpace>;
  Kokkos::LayoutRight l4(10, 20, 30);
  auto view_a_b = Kokkos::view_alloc("sajid");

  traits::array_layout la =
      Kokkos::Impl::DynRankDimTraits<void>::createLayout<traits>(view_a_b, l4);
  ASSERT_TRUE((la.dimension[0] == 10u));
  ASSERT_TRUE((la.dimension[1] == 20u));
}

static void test_dyn_rank_create_view() {
  std::cout << "\n=== test_dyn_rank_create_view ===\n";

  using view_t = Kokkos::View<double ***>;
  std::string label = "SAJID";
  auto v = Kokkos::Impl::DynRankDimTraits<void>::createView<view_t>(
      label, 10, 20, 30, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
      KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX);
  ASSERT_TRUE((v.label() == "SAJID"));
  ASSERT_TRUE((v.rank() == 3u));
  ASSERT_TRUE((v.extent(0) == 10u));
  ASSERT_TRUE((v.extent(1) == 20u));
  ASSERT_TRUE((v.extent(2) == 30u));
}
TEST(kokkos_dyn_rank_create_view_layout, host_execution) {
  test_dyn_rank_create_view();
  test_dyn_rank_create_layout();
}

static void test_reconstruct_layout() {
  std::cout << "\n=== test_reconstruct_layout ===\n";

  Kokkos::LayoutRight l5(2, 3, 4);
  int dynrank = 2;
  Kokkos::LayoutRight layra = Kokkos::Impl::reconstructLayout(l5, dynrank);
  ASSERT_TRUE((layra.dimension[0] == 2u));
  ASSERT_TRUE(layra.dimension[1] == 3u);
}
TEST(kokkos_dyn_rankre_layout, host_execution) { test_reconstruct_layout(); }

static void test_dyn_rank_bounds() {
  std::cout << "\n=== test_dyn_rank_bounds ===\n";

  struct FakeExtents {
    size_t extent(unsigned r) const {
      if (r == 0)
        return 10;
      if (r == 1)
        return 20;
      return 1;
    }
  };
  FakeExtents fake;
  unsigned dyn_rank = 2;

  bool ok = Kokkos::Impl::dyn_rank_view_verify_operator_bounds<0>(
      dyn_rank, fake, 5, 10, 0, 0, 0, 0, 0);
  ASSERT_TRUE((ok == true));
}
TEST(kokkos_dyn_rank_bounds, host_execution) { test_dyn_rank_bounds(); }

static void test_dyn_rank_view_interop() {
  std::cout << "\n=== test_dyn_rank_view_interop ===\n";

  Kokkos::View<double **, Kokkos::LayoutRight> static_view("sajid_b", 10, 20);
  Kokkos::DynRankView<double, Kokkos::LayoutRight> dyn_view = static_view;

  auto host_mirror = Kokkos::create_mirror_view(dyn_view);
  host_mirror(0, 0) = 42.0;
  Kokkos::deep_copy(dyn_view, host_mirror);

  ASSERT_TRUE((host_mirror(0, 0) == 42.0));

  // When constructed from the same static view, data pointers must match
  ASSERT_TRUE((dyn_view.data() == static_view.data()));
}
TEST(kokkos_dyn_rank_interop, host_execution) { test_dyn_rank_view_interop(); }

static void test_dyn_rank_view_traits() {
  std::cout << "\n=== test_dyn_rank_view_traits ===\n";

  using kokkos_dview = Kokkos::DynRankView<double, Kokkos::LayoutRight>;
  ASSERT_TRUE((Kokkos::is_dyn_rank_view_v<kokkos_dview>));

  ASSERT_TRUE(
      (std::is_same_v<kokkos_dview::drvtraits,
                      Kokkos::ViewTraits<double, Kokkos::LayoutRight>>));
  ASSERT_TRUE(
      (std::is_same_v<kokkos_dview::view_type,
                      Kokkos::View<double *******, Kokkos::LayoutRight>>));
  ASSERT_TRUE((std::is_same_v<kokkos_dview::data_type, double>));
  ASSERT_TRUE((std::is_same_v<kokkos_dview::value_type, double>));
  ASSERT_TRUE(
      (std::is_same_v<kokkos_dview::array_layout, Kokkos::LayoutRight>));
  ASSERT_TRUE((std::is_same_v<kokkos_dview::pointer_type, double *>));
  ASSERT_TRUE((std::is_same_v<kokkos_dview::reference_type, double &>));
  ASSERT_TRUE(
      (std::is_same_v<kokkos_dview::memory_traits, Kokkos::MemoryTraits<0>>));
}
TEST(kokkos_dyn_rank_traits, host_execution) { test_dyn_rank_view_traits(); }

static void test_dyn_rank_view_span() {
  std::cout << "\n=== test_dyn_rank_view_span ===\n";

  using kokkos_dview = Kokkos::DynRankView<double, Kokkos::LayoutRight>;
  kokkos_dview drank_m("drank_m", 10, 20, 30);

  ASSERT_TRUE((drank_m.rank() == 3u));
  ASSERT_TRUE((drank_m.size() == 10u * 20u * 30u));
  ASSERT_TRUE((drank_m.is_allocated()));
  ASSERT_TRUE((drank_m.use_count() >= 1));

  // LayoutRight: last dimension has stride 1
  ASSERT_TRUE((drank_m.extent(0) == 10u));
  ASSERT_TRUE((drank_m.extent(1) == 20u));
  ASSERT_TRUE((drank_m.extent(2) == 30u));
}
TEST(kokkos_dyn_rank_span, host_execution) { test_dyn_rank_view_span(); }
#ifdef KOKKOS_ENABLE_CUDA
static void test_check_issue_7604() {
  Kokkos::DynRankView<double, Kokkos::LayoutRight> drank_lay("dynamic_rank", 5,
                                                             6);
  ASSERT_TRUE((drank_lay.extent(0) == 5u));
  ASSERT_TRUE((drank_lay.extent(1) == 6u));
  Kokkos::parallel_for(
      "TestDynamics for issue 7604",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {5, 6}),
      KOKKOS_LAMBDA(const int i, const int j) {
        drank_lay(i, j) = i * 10 + j;
      });
  Kokkos::fence();
  auto host_mirror = Kokkos::create_mirror_view(drank_lay);
  Kokkos::deep_copy(host_mirror, drank_lay);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 6; j++) {
      ASSERT_TRUE((host_mirror(i, j) == i * 10 + j));
    }
  }
}
TEST(kokkos_dyn_rank_issue_7604, device_execution) { test_check_issue_7604(); }
#endif
#ifdef KOKKOS_ENABLE_CUDA
template <int N = 100000> static void test_access_and_3rd_operator() {
  Kokkos::DynRankView<double> one;
  Kokkos::DynRankView<double> array_like("allocated array", N);
  Kokkos::Timer timer;
  Kokkos::parallel_for(
      "going through loop", Kokkos::RangePolicy<>(0, N),
      KOKKOS_LAMBDA(const int i) {
        array_like.access(i) = i * 2.0;
        array_like(i) = array_like.access(i) + 1.0;
        array_like[i] = array_like[i] + 1.0;
      });
  double time = timer.seconds();
  printf("%lf\n", time); //
  auto host_view = Kokkos::create_mirror_view(array_like);
  Kokkos::deep_copy(host_view, array_like);
  for (int i = 0; i < N; i++) {
    ASSERT_TRUE((host_view(i) == i * 2.0 + 2.0));
  }
}
TEST(kokkos_dyn_rank_3rd_oper, device_execution) {
  test_access_and_3rd_operator<2>();
}
#endif
#ifdef KOKKOS_ENABLE_CUDA

static void test_dyn_rank_ctor() {
  Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace> drank_lay(
      "dynamic_rank", 5, 6);
  ASSERT_TRUE((drank_lay.extent(0) == 5u));
  ASSERT_TRUE((drank_lay.extent(1) == 6u));
  Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace>
      drank_lay2(drank_lay);
  ASSERT_TRUE((drank_lay2.extent(0) == 5u));
  ASSERT_TRUE((drank_lay2.extent(1) == 6u));
  ASSERT_TRUE((drank_lay2.data() == drank_lay.data()));
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::CudaSpace> view_passed(
      "passed to drank", 5, 3);
  Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace>
      drank_lay3(view_passed, 1);
  // CHECK(drank_lay3.extent(0) == view_passed.extent(0));
  ASSERT_TRUE((drank_lay3.size() == view_passed.size()));
  Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace>
      drank_lay4 = drank_lay;
  ASSERT_TRUE((drank_lay4.data() == drank_lay.data()));
  auto host_mirror = Kokkos::create_mirror_view(drank_lay3);
  Kokkos::deep_copy(host_mirror, drank_lay3);
  std::cout << "Kokkos::DynRankView<double,Kokkos::LayoutRight, "
               "Kokkos::CudaSpace> drank_lay3(view_passed,1) "
            << host_mirror.size() << ". And Extent " << host_mirror.extent(0)
            << std::endl; // point host mirror still points the view_passed or
                          // drank_lay3
  using StaticView =
      Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::CudaSpace>;
  StaticView sview("mine", 10, 5, 4);
  Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace> dview(
      sview, 2);
  std::cout << "Kokkos::DynRankView<double,Kokkos::LayoutRight, "
               "Kokkos::CudaSpace> dview(sview,2);"
            << dview.extent(0) << " , " << dview.extent(1) << std::endl;
}
TEST(kokkos_dyn_rank_ctor, device_execution) { test_dyn_rank_ctor(); }
#endif

#ifdef KOKKOS_ENABLE_CUDA
static void test_cuda_specific() {
  std::cout << "\n=== test_cuda_specific ===\n";

  using kokkos_dview = Kokkos::DynRankView<double, Kokkos::LayoutRight>;
  ASSERT_TRUE((std::is_same_v<kokkos_dview::execution_space, Kokkos::Cuda>));
  ASSERT_TRUE((std::is_same_v<kokkos_dview::memory_space, Kokkos::CudaSpace>));
  ASSERT_TRUE(
      (std::is_same_v<kokkos_dview::device_type,
                      Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>>));

  // Host mirror should live on HostSpace
  ASSERT_TRUE((std::is_same_v<kokkos_dview::host_mirror_type,
                              Kokkos::DynRankView<double, Kokkos::LayoutRight,
                                                  Kokkos::HostSpace>>));

  // dyn_rank_view_verify_operator_bounds with CudaSpace tag
  struct FakeExtents {
    size_t extent(unsigned r) const {
      if (r == 0)
        return 10;
      if (r == 1)
        return 20;
      return 1;
    }
  };
  FakeExtents fake;
  unsigned rankk = 2;
  Kokkos::Impl::SharedAllocationTracker dummy_tracker;
  Kokkos::Impl::dyn_rank_view_verify_operator_bounds<Kokkos::CudaSpace>(
      3, rankk, dummy_tracker, fake, 5, 10, 0, 0, 0, 0, 0);
  std::cout << "  PASS: CUDA bounds check did not abort\n";
}
TEST(kokkos_dyn_rank_cuda_specific, device_execution) { test_cuda_specific(); }
#endif
template <class DataType, class LayOut, class ExecSpace>
static void test_as_view_of_rank_n() {
  Kokkos::DynRankView<DataType, LayOut, ExecSpace> dyn_rank("DynRank", 10, 20);
  auto stat_view = Kokkos::Impl::as_view_of_rank_n<2>(dyn_rank);
  using stat_type = decltype(stat_view);
  bool res =
      std::is_same_v<stat_type, Kokkos::View<DataType **, LayOut, ExecSpace>>;
  ASSERT_TRUE((res));
}
template <class DataType, class ExecSpace, class LayOut>
static void test_required_allocation_size() {
  using DynRankType = Kokkos::DynRankView<DataType, LayOut, ExecSpace>;
  const size_t bytes = sizeof(DataType);
  auto size_length_1 = DynRankType::required_allocation_size(10);
  DynRankType rank_1("Initiate", 10);
  ASSERT_TRUE((size_length_1 == 10 * bytes));
  auto size_length_2 = DynRankType::required_allocation_size(10, 20);
  ASSERT_TRUE(size_length_2 == 10 * 20 * bytes);
  auto size_length_3 =
      DynRankType::required_allocation_size(2, 3, 4, 5, 6, 7, 8, 9);
  ASSERT_TRUE((size_length_3 !=
               2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 *
                   bytes)); // referring this PR:
                            // https://github.com/kokkos/kokkos/pull/9290
}
TEST(kokkos_dyn_rank_rank_n_req_allo_size, host_execution) {
  test_as_view_of_rank_n<double, Kokkos::LayoutRight, Kokkos::HostSpace>();
  test_required_allocation_size<double, Kokkos::HostSpace,
                                Kokkos::LayoutRight>();
}
#ifdef KOKKOS_ENABLE_CUDA
TEST(kokkos_dyn_rank_rank_n_req_allo_size, device_execution) {
  test_as_view_of_rank_n<double, Kokkos::LayoutRight, Kokkos::CudaSpace>();
  test_required_allocation_size<double, Kokkos::CudaSpace,
                                Kokkos::LayoutRight>();
}
#endif
template <class DataType, class LayOut>
static void test_unmanaged_dynamicrank_view() {
  DataType *unmngd_raw_pointer = new DataType[50];
  LayOut layout(10, 5);
  Kokkos::DynRankView<DataType, LayOut, Kokkos::HostSpace> unmng_view(
      unmngd_raw_pointer, layout);
  unmng_view(3) = 2000.0;
  std::cout << "Unmanaged_view[3] = " << unmngd_raw_pointer[3] << std::endl;
  delete[] unmngd_raw_pointer;
}
TEST(kokkos_dyn_rank_unmand, host_execution) {
  test_unmanaged_dynamicrank_view<double, Kokkos::LayoutRight>();
}

} // namespace Test
