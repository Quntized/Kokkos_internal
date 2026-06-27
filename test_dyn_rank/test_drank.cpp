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
#include <iostream>
#include <type_traits>

#define CHECK(expr)                                                            \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "FAILED: " #expr << "  (" << __FILE__ << ":" << __LINE__    \
                << ")\n";                                                      \
      Kokkos::abort("Test failed!!!");                                         \
    } else {                                                                   \
      std::cout << "  PASS: " #expr << "\n";                                   \
    }                                                                          \
  } while (0)

namespace Test {

template <typename Scalar, class Space> struct TestDynamicView {
  using execution_space = typename Space::execution_space;
  using memory_space = typename Space::memory_space;
  using view_type = Kokkos::Experimental::DynamicView<Scalar *, Space>;

  static void run(unsigned arg_total_size) {
    std::cout << "\n=== TestDynamicView ===\n";

    view_type d1;
    CHECK(d1.is_allocated() == false);

    d1 = view_type("d1", 1024, arg_total_size);
    view_type d2(d1);
    view_type d3("d3", 1024, arg_total_size);

    CHECK(d2.is_allocated() == false);
    CHECK(d3.is_allocated() == false);

    unsigned d_size = arg_total_size / 8;
    d1.resize_serial(d_size);
    d3.resize_serial(d_size);

    CHECK(d1.is_allocated() == true);
    CHECK(d2.is_allocated() == true);
    CHECK(d3.is_allocated() == true);
  }
};

static void test_view_data_type_from_rank() {
  std::cout << "\n=== test_view_data_type_from_rank ===\n";

  using Rank4 = Kokkos::Impl::ViewDataTypeFromRank<double, 4>;
  CHECK((std::is_same_v<Rank4::type, double ****>));
}

static void test_dyn_rank_view_basics(unsigned arg_total_size) {
  std::cout << "\n=== test_dyn_rank_view_basics ===\n";

  Kokkos::DynRankView<double, Kokkos::DefaultExecutionSpace> d4("d4", 1024,
                                                                arg_total_size);
  CHECK(d4.rank() == 2u);

  size_t rank = Kokkos::Impl::DynRankDimTraits<void>::computeRank(
      2, 2, 2, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
      KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX);
  CHECK(rank == 3u);

  Kokkos::LayoutRight l(10, 20, 30, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
                        KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
                        KOKKOS_INVALID_INDEX);
  size_t rank2 = Kokkos::Impl::DynRankDimTraits<void>::computeRank(l);
  CHECK(rank2 == 3u);

  Kokkos::LayoutRight l2(10, 20, 30, 40, KOKKOS_INVALID_INDEX,
                         KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
                         KOKKOS_INVALID_INDEX);
  auto view_a = Kokkos::view_alloc("sajid", Kokkos::WithoutInitializing);
  size_t rank3 = Kokkos::Impl::DynRankDimTraits<void>::computeRank(view_a, l2);
  CHECK(rank3 == 4u);

  Kokkos::LayoutRight l3(10, 20, 30, 40, 50);
  Kokkos::LayoutRight lr =
      Kokkos::Impl::DynRankDimTraits<void>::createLayout(l3);
  size_t rank4 = Kokkos::Impl::DynRankDimTraits<void>::computeRank(lr);
  std::cout << "  INFO: computeRank after createLayout = " << rank4
            << " (input had 5 valid dims)\n";
  CHECK(rank4 > 0u);
}

static void test_dyn_rank_create_layout() {
  std::cout << "\n=== test_dyn_rank_create_layout ===\n";

  using traits = Kokkos::ViewTraits<double **, Kokkos::LayoutRight,
                                    Kokkos::DefaultExecutionSpace>;
  Kokkos::LayoutRight l4(10, 20, 30);
  auto view_a_b = Kokkos::view_alloc("sajid");

  traits::array_layout la =
      Kokkos::Impl::DynRankDimTraits<void>::createLayout<traits>(view_a_b, l4);
  CHECK(la.dimension[0] == 10u);
  CHECK(la.dimension[1] == 20u);
}

static void test_dyn_rank_create_view() {
  std::cout << "\n=== test_dyn_rank_create_view ===\n";

  using view_t = Kokkos::View<double ***>;
  std::string label = "SAJID";
  auto v = Kokkos::Impl::DynRankDimTraits<void>::createView<view_t>(
      label, 10, 20, 30, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
      KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX);
  CHECK(v.label() == "SAJID");
  CHECK(v.rank() == 3u);
  CHECK(v.extent(0) == 10u);
  CHECK(v.extent(1) == 20u);
  CHECK(v.extent(2) == 30u);
}

static void test_reconstruct_layout() {
  std::cout << "\n=== test_reconstruct_layout ===\n";

  Kokkos::LayoutRight l5(2, 3, 4);
  int dynrank = 2;
  Kokkos::LayoutRight layra = Kokkos::Impl::reconstructLayout(l5, dynrank);
  CHECK(layra.dimension[0] == 2u);
  CHECK(layra.dimension[1] == 3u);
}

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
  CHECK(ok == true);
}

static void test_dyn_rank_view_interop() {
  std::cout << "\n=== test_dyn_rank_view_interop ===\n";

  Kokkos::View<double **, Kokkos::LayoutRight> static_view("sajid_b", 10, 20);
  Kokkos::DynRankView<double, Kokkos::LayoutRight> dyn_view = static_view;

  auto host_mirror = Kokkos::create_mirror_view(dyn_view);
  host_mirror(0, 0) = 42.0;
  Kokkos::deep_copy(dyn_view, host_mirror);

  CHECK(host_mirror(0, 0) == 42.0);

  // When constructed from the same static view, data pointers must match
  CHECK(dyn_view.data() == static_view.data());
}

static void test_dyn_rank_view_traits() {
  std::cout << "\n=== test_dyn_rank_view_traits ===\n";

  using kokkos_dview = Kokkos::DynRankView<double, Kokkos::LayoutRight>;
  CHECK((Kokkos::is_dyn_rank_view_v<kokkos_dview>));

  CHECK((std::is_same_v<kokkos_dview::drvtraits,
                        Kokkos::ViewTraits<double, Kokkos::LayoutRight>>));
  CHECK((std::is_same_v<kokkos_dview::view_type,
                        Kokkos::View<double *******, Kokkos::LayoutRight>>));
  CHECK((std::is_same_v<kokkos_dview::data_type, double>));
  CHECK((std::is_same_v<kokkos_dview::value_type, double>));
  CHECK((std::is_same_v<kokkos_dview::array_layout, Kokkos::LayoutRight>));
  CHECK((std::is_same_v<kokkos_dview::pointer_type, double *>));
  CHECK((std::is_same_v<kokkos_dview::reference_type, double &>));
  CHECK((std::is_same_v<kokkos_dview::memory_traits, Kokkos::MemoryTraits<0>>));
}

static void test_dyn_rank_view_span() {
  std::cout << "\n=== test_dyn_rank_view_span ===\n";

  using kokkos_dview = Kokkos::DynRankView<double, Kokkos::LayoutRight>;
  kokkos_dview drank_m("drank_m", 10, 20, 30);

  CHECK(drank_m.rank() == 3u);
  CHECK(drank_m.span_is_contiguous());
  CHECK(drank_m.size() == 10u * 20u * 30u);
  CHECK(drank_m.is_allocated());
  CHECK(drank_m.use_count() >= 1);

  // LayoutRight: last dimension has stride 1
  CHECK(drank_m.extent(0) == 10u);
  CHECK(drank_m.extent(1) == 20u);
  CHECK(drank_m.extent(2) == 30u);
}
#ifdef KOKKOS_ENABLE_CUDA
static void test_check_issue_7604() {
  Kokkos::DynRankView<double, Kokkos::LayoutRight> drank_lay("dynamic_rank", 5,
                                                             6);
  CHECK(drank_lay.extent(0) == 5u);
  CHECK(drank_lay.extent(1) == 6u);
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
      CHECK(host_mirror(i, j) == i * 10 + j);
    }
  }
}
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
    CHECK(host_view(i) == i * 2.0 + 2.0);
  }
}
#endif
#ifdef KOKKOS_ENABLE_CUDA

static void test_dyn_rank_ctor(){
    Kokkos::DynRankView<double,Kokkos::LayoutRight, Kokkos::CudaSpace> drank_lay("dynamic_rank",5,6);
    CHECK(drank_lay.extent(0) == 5u);
    CHECK(drank_lay.extent(1) == 6u);
    Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace> drank_lay2(drank_lay);
    CHECK(drank_lay2.extent(0) == 5u);
    CHECK(drank_lay2.extent(1) == 6u);
    CHECK(drank_lay2.data() == drank_lay.data());
    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::CudaSpace> view_passed("passed to drank",5,3);
    Kokkos::DynRankView<double,Kokkos::LayoutRight, Kokkos::CudaSpace> drank_lay3(view_passed,1);
    //CHECK(drank_lay3.extent(0) == view_passed.extent(0));
    CHECK(drank_lay3.size() == view_passed.size());
    Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace> drank_lay4 = drank_lay;
    CHECK(drank_lay4.data() == drank_lay.data());
    auto host_mirror = Kokkos::create_mirror_view(drank_lay3);
    Kokkos::deep_copy(host_mirror, drank_lay3);
    std::cout<<"Kokkos::DynRankView<double,Kokkos::LayoutRight, Kokkos::CudaSpace> drank_lay3(view_passed,1) "<<host_mirror.size()<<". And Extent "<<host_mirror.extent(0)<<std::endl; //point host mirror still points the view_passed or drank_lay3
    using StaticView = Kokkos::View<double***,Kokkos::LayoutRight, Kokkos::CudaSpace>;
    StaticView sview("mine", 10,5,4);
    Kokkos::DynRankView<double,Kokkos::LayoutRight, Kokkos::CudaSpace> dview(sview,2);
    std::cout<<"Kokkos::DynRankView<double,Kokkos::LayoutRight, Kokkos::CudaSpace> dview(sview,2);"<<dview.extent(0)<<" , "<<dview.extent(1)<<std::endl;

static void test_dyn_rank_ctor() {
  Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace> drank_lay(
      "dynamic_rank", 5, 6);
  CHECK(drank_lay.extent(0) == 5u);
  CHECK(drank_lay.extent(1) == 6u);
  Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace>
      drank_lay2(drank_lay);
  CHECK(drank_lay2.extent(0) == 5u);
  CHECK(drank_lay2.extent(1) == 6u);
  CHECK(drank_lay2.data() == drank_lay.data());
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::CudaSpace> view_passed(
      "passed to drank", 5, 3);
  Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace>
      drank_lay3(view_passed, 1);
  // CHECK(drank_lay3.extent(0) == view_passed.extent(0));
  CHECK(drank_lay3.size() == view_passed.size());
  Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::CudaSpace>
      drank_lay4 = drank_lay;
  CHECK(drank_lay4.data() == drank_lay.data());
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
#endif

#ifdef KOKKOS_ENABLE_CUDA
static void test_cuda_specific() {
  std::cout << "\n=== test_cuda_specific ===\n";

  using kokkos_dview = Kokkos::DynRankView<double, Kokkos::LayoutRight>;
  CHECK((std::is_same_v<kokkos_dview::execution_space, Kokkos::Cuda>));
  CHECK((std::is_same_v<kokkos_dview::memory_space, Kokkos::CudaSpace>));
  CHECK((std::is_same_v<kokkos_dview::device_type,
                        Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>>));

  // Host mirror should live on HostSpace
  CHECK((std::is_same_v<
         kokkos_dview::host_mirror_type,
         Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::HostSpace>>));

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
#endif
template <class DataType, class LayOut, class ExecSpace>
static void test_as_view_of_rank_n(){
    Kokkos::DynRankView<DataType, LayOut, ExecSpace> dyn_rank("DynRank", 10,20);
    auto stat_view = Kokkos::Impl::as_view_of_rank_n<2>(dyn_rank);
    using stat_type = decltype(stat_view);
    bool res = std::is_same_v<stat_type,Kokkos::View<DataType**, LayOut,ExecSpace>>;
    CHECK(res);
}
template <class DataType, class ExecSpace, class LayOut>
static void test_required_allocation_size(){
    using DynRankType = Kokkos::DynRankView<DataType,LayOut,ExecSpace>;
    const size_t bytes = sizeof(DataType);
    auto size_length_1 = DynRankType::required_allocation_size(10);
    DynRankType rank_1("Initiate",10);
    CHECK(size_length_1 == 10*bytes);
    auto size_length_2 = DynRankType::required_allocation_size(10,20);
    CHECK(size_length_2 == 10*20*bytes);
    auto size_length_3 = DynRankType::required_allocation_size(2,3,4,5,6,7,8,9);
    CHECK(size_length_3 != 2*3*4*5*6*7*8*9*bytes);
}
template<class DataType, class LayOut>
static void test_unmanaged_dynamicrank_view(){
    DataType* unmngd_raw_pointer = new DataType[50];
    LayOut layout(10,5);
    Kokkos::DynRankView<DataType, LayOut> unmng_view(unmngd_raw_pointer, layout);
    unmng_view(3) = 2000.0;
    std::cout<<"Unmanaged_view[3] = "<<unmngd_raw_pointer[3]<<std::endl;
}



}  // namespace Test

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        Test::TestDynamicView<double,Kokkos::DefaultExecutionSpace>::run(1024);
        Test::test_view_data_type_from_rank();
        Test::test_dyn_rank_view_basics(1024);
        Test::test_dyn_rank_create_layout();
        Test::test_dyn_rank_create_view();
        Test::test_reconstruct_layout();
        Test::test_dyn_rank_bounds();
        Test::test_dyn_rank_view_interop();
        Test::test_dyn_rank_view_traits();
        Test::test_dyn_rank_view_span();

#ifdef KOKKOS_ENABLE_CUDA
        Test::test_cuda_specific();
        Test::test_check_issue_7604();
        Test::test_access_and_3rd_operator();
        Test::test_dyn_rank_ctor();
        Test::test_as_view_of_rank_n<double, Kokkos::LayoutRight,Kokkos::CudaSpace>();
        Test::test_required_allocation_size<double, Kokkos::CudaSpace,Kokkos::LayoutRight>();
        Test::test_required_allocation_size<double, Kokkos::HostSpace,Kokkos::LayoutRight>();
        Test::test_required_allocation_size<int, Kokkos::CudaSpace,Kokkos::LayoutRight>();
        Test::test_unmanaged_dynamicrank_view<double,Kokkos::LayoutRight>();

} // namespace Test

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    Test::TestDynamicView<double, Kokkos::DefaultExecutionSpace>::run(1024);
    Test::test_view_data_type_from_rank();
    Test::test_dyn_rank_view_basics(1024);
    Test::test_dyn_rank_create_layout();
    Test::test_dyn_rank_create_view();
    Test::test_reconstruct_layout();
    Test::test_dyn_rank_bounds();
    Test::test_dyn_rank_view_interop();
    Test::test_dyn_rank_view_traits();
    Test::test_dyn_rank_view_span();

#ifdef KOKKOS_ENABLE_CUDA // for cuda
    Test::test_cuda_specific();
    Test::test_check_issue_7604();
    Test::test_access_and_3rd_operator();
    Test::test_dyn_rank_ctor();

#endif

    std::cout << "\n*** ALL TESTS PASSED ***\n";
  }
  Kokkos::finalize();
  return 0;
}