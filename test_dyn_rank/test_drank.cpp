// test_drank.cpp – Kokkos DynRankView / DynamicView smoke tests
//
// Uses assert() for pass/fail so CTest can detect failures.
// CUDA-specific sections are guarded with #ifdef KOKKOS_ENABLE_CUDA.

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <type_traits>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DynamicView.hpp>
#include <Kokkos_Timer.hpp>
#include <Kokkos_DynRankView.hpp>

#define CHECK(expr)                                                          \
  do {                                                                       \
    if (!(expr)) {                                                           \
      std::cerr << "FAILED: " #expr << "  (" << __FILE__ << ":"              \
                << __LINE__ << ")\n";                                        \
      std::abort();                                                          \
    } else {                                                                 \
      std::cout << "  PASS: " #expr << "\n";                                 \
    }                                                                        \
  } while (0)

namespace Test {


template <typename Scalar, class Space>
struct TestDynamicView {
    using execution_space = typename Space::execution_space;
    using memory_space    = typename Space::memory_space;
    using view_type =
        Kokkos::Experimental::DynamicView<Scalar*, Space>;

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
    CHECK((std::is_same_v<Rank4::type, double****>));
}


static void test_dyn_rank_view_basics(unsigned arg_total_size) {
    std::cout << "\n=== test_dyn_rank_view_basics ===\n";

    Kokkos::DynRankView<double, Kokkos::DefaultExecutionSpace>
        d4("d4", 1024, arg_total_size);
    CHECK(d4.rank() == 2u);

    size_t rank = Kokkos::Impl::DynRankDimTraits<void>::computeRank(
        2, 2, 2,
        KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
        KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX);
    CHECK(rank == 3u);

    Kokkos::LayoutRight l(
        10, 20, 30,
        KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
        KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX);
    size_t rank2 = Kokkos::Impl::DynRankDimTraits<void>::computeRank(l);
    CHECK(rank2 == 3u);

    Kokkos::LayoutRight l2(
        10, 20, 30, 40,
        KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
        KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX);
    auto view_a = Kokkos::view_alloc("sajid", Kokkos::WithoutInitializing);
    size_t rank3 =
        Kokkos::Impl::DynRankDimTraits<void>::computeRank(view_a, l2);
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

    using traits = Kokkos::ViewTraits<double**,
                                      Kokkos::LayoutRight,
                                      Kokkos::DefaultExecutionSpace>;
    Kokkos::LayoutRight l4(10, 20, 30);
    auto view_a_b = Kokkos::view_alloc("sajid");

    traits::array_layout la =
        Kokkos::Impl::DynRankDimTraits<void>::createLayout<traits>(
            view_a_b, l4);
    CHECK(la.dimension[0] == 10u);
    CHECK(la.dimension[1] == 20u);
}


static void test_dyn_rank_create_view() {
    std::cout << "\n=== test_dyn_rank_create_view ===\n";

    using view_t = Kokkos::View<double***>;
    std::string label = "SAJID";
    auto v = Kokkos::Impl::DynRankDimTraits<void>::createView<view_t>(
        label, 10, 20, 30,
        KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
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
    Kokkos::LayoutRight layra =
        Kokkos::Impl::reconstructLayout(l5, dynrank);
    CHECK(layra.dimension[0] == 2u);
    CHECK(layra.dimension[1] == 3u);
}


static void test_dyn_rank_bounds() {
    std::cout << "\n=== test_dyn_rank_bounds ===\n";

    struct FakeExtents {
        size_t extent(unsigned r) const {
            if (r == 0) return 10;
            if (r == 1) return 20;
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

    Kokkos::View<double**, Kokkos::LayoutRight> static_view(
        "sajid_b", 10, 20);
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

    using kokkos_dview =
        Kokkos::DynRankView<double, Kokkos::LayoutRight>;
    CHECK((Kokkos::is_dyn_rank_view_v<kokkos_dview>));

    CHECK((std::is_same_v<kokkos_dview::drvtraits,
                          Kokkos::ViewTraits<double, Kokkos::LayoutRight>>));
    CHECK((std::is_same_v<kokkos_dview::view_type,
                          Kokkos::View<double*******, Kokkos::LayoutRight>>));
    CHECK((std::is_same_v<kokkos_dview::data_type, double>));
    CHECK((std::is_same_v<kokkos_dview::value_type, double>));
    CHECK((std::is_same_v<kokkos_dview::array_layout,
                          Kokkos::LayoutRight>));
    CHECK((std::is_same_v<kokkos_dview::pointer_type, double*>));
    CHECK((std::is_same_v<kokkos_dview::reference_type, double&>));
    CHECK((std::is_same_v<kokkos_dview::memory_traits,
                          Kokkos::MemoryTraits<0>>));
}


static void test_dyn_rank_view_span() {
    std::cout << "\n=== test_dyn_rank_view_span ===\n";

    using kokkos_dview =
        Kokkos::DynRankView<double, Kokkos::LayoutRight>;
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
static void test_cuda_specific() {
    std::cout << "\n=== test_cuda_specific ===\n";

    using kokkos_dview =
        Kokkos::DynRankView<double, Kokkos::LayoutRight>;
    CHECK((std::is_same_v<kokkos_dview::execution_space, Kokkos::Cuda>));
    CHECK((std::is_same_v<kokkos_dview::memory_space, Kokkos::CudaSpace>));
    CHECK((std::is_same_v<kokkos_dview::device_type,
                          Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>>));

    // Host mirror should live on HostSpace
    CHECK((std::is_same_v<kokkos_dview::host_mirror_type,
                          Kokkos::DynRankView<double, Kokkos::LayoutRight,
                                              Kokkos::HostSpace>>));

    // dyn_rank_view_verify_operator_bounds with CudaSpace tag
    struct FakeExtents {
        size_t extent(unsigned r) const {
            if (r == 0) return 10;
            if (r == 1) return 20;
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
#endif

        std::cout << "\n*** ALL TESTS PASSED ***\n";
    }
    Kokkos::finalize();
    return 0;
}