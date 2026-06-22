

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DynamicView.hpp>
#include <Kokkos_Timer.hpp>
#include <Kokkos_DynRankView.hpp>

namespace Test {
    template <typename Scalar, class Space>
    struct TestDynamicView {
        using execution_space = typename Space::execution_space;
        using memory_space = typename Space::memory_space;
        using view_type = Kokkos::Experimental::DynamicView<Scalar*, Space>;
        using value_type = double;

        static void run(unsigned arg_total_size){
            {
                {
                    view_type d1;
                    std::cout<<"Check: d1.is_allocated() = "<<d1.is_allocated()<<std::endl;
                    d1 = view_type("d1", 1204, arg_total_size);
                    view_type d2(d1);
                    view_type d3("d3", 1024,arg_total_size);
                    std::cout<<"Check: d2.is_allocated() = "<<d2.is_allocated()<<std::endl;
                    std::cout<<"Check: d3.is_allocated() = "<<d3.is_allocated()<<std::endl;
                    unsigned d_size = arg_total_size / 8;
                    d1.resize_serial(d_size);
                    d2.resize_serial(d_size);
                    d3.resize_serial(d_size);
                    std::cout<<"Check: d1.is_allocated() = "<<d1.is_allocated()<<std::endl;
                    std::cout<<"Check: d2.is_allocated() = "<<d2.is_allocated()<<std::endl;
                    std::cout<<"Check: d3.is_allocated() = "<<d3.is_allocated()<<std::endl;
                }
            }
            using Sajid  = Kokkos::Impl::ViewDataTypeFromRank<double, 4>;
            std::cout<<"check double**** = "<<(std::is_same_v<Sajid::type, double****>)<<std::endl;
            Kokkos::DynRankView<double, Kokkos::DefaultExecutionSpace> d4("d4", 1024, arg_total_size);
            std::cout<<"kokko dynamic rank = "<<d4.rank()<<std::endl;
            //auto s = Kokkos::Impl::as_view_of_rank_n<2>(d4);
            //using s_type = decltype(s);
            //std::cout<<"kokko dynamic rank = "<<s.rank()<<std::endl;
            //std::cout<<"decltype(asview_of_rank_n<2>(d4)) = "<<typeid(decltype(Kokkos::Impl::as_view_of_rank_n<2>(d4))).name()<<std::endl;
            //std::cout<<"decltype(asview_of_rank_n<2>(d4)) = "<<std::endl;
            using view__ = Kokkos::View<double**>;
            ///std::cout<<std::is_same_v<s,view__> <<std::endl;
            size_t rank = Kokkos::Impl::DynRankDimTraits<void>::computeRank(2,2,2,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX);
            std::cout<<"Computed rank = "<<rank<<std::endl;
            Kokkos::LayoutRight l(10,20,30,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX);
            //std::cout<<"LayoutRight rank = "<<l.rank()<<std::endl;
            size_t rank2 = Kokkos::Impl::DynRankDimTraits<void>::computeRank(l);
            std::cout<<"Computed rank from layout = "<<rank2<<std::endl;
            Kokkos::LayoutRight l2(10,20,30,40,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX);
            auto view_a = Kokkos::view_alloc("sajid", Kokkos::WithoutInitializing);
            size_t rank3 = Kokkos::Impl::DynRankDimTraits<void>::computeRank(view_a,l2);
            std::cout<<"Computed rank from view and layout = "<<rank3<<std::endl;
            Kokkos::LayoutRight l3(10,20,30,40,50/*,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX*/);
            Kokkos::LayoutRight lr = Kokkos::Impl::DynRankDimTraits<void>::createLayout(l3);
            size_t rank4 = Kokkos::Impl::DynRankDimTraits<void>::computeRank(lr);
            std::cout<<"Computed rank from created layout = "<<rank4<<std::endl;
            using traits = Kokkos::ViewTraits<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
            Kokkos::LayoutRight l4(10,20,30);
            auto view_a_b = Kokkos::view_alloc("sajid");
            traits::array_layout la = Kokkos::Impl::DynRankDimTraits<void>::createLayout<traits>(view_a_b,l4);
            std::cout<<"padded dim 0 = "<<la.dimension[0]<<std::endl;
            std::cout<<"padded dim 1 = "<<la.dimension[1]<<std::endl;
            std::cout<<"padded dim 2 = "<<la.dimension[2]<<std::endl;
            std::cout<<"padded dim 3 = "<<la.dimension[3]<<std::endl;
            using view_sajid = Kokkos::View<double***>;
            std::string label = "SAJID";
            auto view_sajid_a = Kokkos::Impl::DynRankDimTraits<void>::createView<view_sajid>(label,10,20,30,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX,KOKKOS_INVALID_INDEX);
            std::cout<<"view label = "<<view_sajid_a.label()<<std::endl;
            std::cout<<"view rank = "<<view_sajid_a.rank()<<std::endl;
            std::cout<<"view dim 0 = "<<view_sajid_a.extent(0)<<std::endl;
            std::cout<<"view dim 1 = "<<view_sajid_a.extent(1)<<std::endl;
            std::cout<<"view dim 2 = "<<view_sajid_a.extent(2)<<std::endl;
            std::cout<<"view dim 3 = "<<view_sajid_a.extent(3)<<std::endl;
            Kokkos::LayoutRight l5(2,3,4);
            int dynrank =2;
            Kokkos::LayoutRight layra = Kokkos::Impl::reconstructLayout(l5,dynrank);
            std::cout<<"reconstructed layout = "<<layra.dimension[0]<<" "<<layra.dimension[1]<<" "<<std::endl;
        }
    };

}
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        Test::TestDynamicView<double, Kokkos::DefaultExecutionSpace>::run(1024);
    }
    Kokkos::finalize();
    return 0;
}