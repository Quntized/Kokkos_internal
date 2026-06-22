

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
            struct SajidS {
                size_t extent(unsigned r) const {
                    if(r == 0) return 10;
                    if(r == 1) return 20;
                    return 1;
                }
            };
            SajidS sajid;
            unsigned dyn_rank = 2;
            bool ok =  Kokkos::Impl::dyn_rank_view_verify_operator_bounds<0>(dyn_rank, sajid,5,10,0,0,0,0,0);
            std::cout<<"dyn_rank_view_verify_operator_bounds = "<<ok<<std::endl;
            unsigned rankk = 2;
            Kokkos::Impl::SharedAllocationTracker dummy_tracker;
            Kokkos::Impl::dyn_rank_view_verify_operator_bounds<Kokkos::CudaSpace>(3,rankk, dummy_tracker,sajid,5,10,0,0,0,0,0);
            Kokkos::View<double**,Kokkos::LayoutRight> static_view("sajid_b",10,20);
            Kokkos::DynRankView<double,Kokkos::LayoutRight> dyn_view=static_view;
            auto host_mirror = Kokkos::create_mirror_view(dyn_view);
            host_mirror(0,0) = 42.0;
            Kokkos::deep_copy(dyn_view, host_mirror);
            std::cout<<"dyn_view(0,0) = "<<host_mirror(0,0)<<std::endl;
            if(dyn_view.data() == static_view.data()) {
                std::cout<<"Data pointers are the same, no copy needed."<<std::endl;
            } else {
                std::cout<<"Data pointers are different, copy was made."<<std::endl;
            }
            using ddyn_view = Kokkos::DynRankView<double,Kokkos::LayoutRight>;
            bool is_dyn_rank_view_b = Kokkos::is_dyn_rank_view_v<ddyn_view>;
            std::cout<<"is_dyn_rank_view = "<<is_dyn_rank_view_b<<std::endl;
            using kokkos_dview = Kokkos::DynRankView<double,Kokkos::LayoutRight>;
            kokkos_dview drank_m("drank_m", 10, 20, 30);
            //size_t rank_m = drank_m.m_rank; this is inaccessible as PRIVATE;
            //std::cout<<"rank of drank_m = "<<rank_m<<std::endl;
            std::cout<<"is same Kokkos::DynRankView<double,Kokkos::LayoutRight>::viewtraits == viewtraits = "<<(std::is_same_v<kokkos_dview::drvtraits,Kokkos::ViewTraits<double,Kokkos::LayoutRight>>)<<std::endl;
            std::cout<<"is same Kokkos::DynRankView<double,Kokkos::LayoutRight>::view_type == view = "<<(std::is_same_v<kokkos_dview::view_type,Kokkos::View<double*******,Kokkos::LayoutRight>>)<<std::endl;
            std::cout<<"is_same_v<kokkos_dview::data_type, double> = "<<(std::is_same_v<kokkos_dview::data_type, double>)<<std::endl;
            std::cout<<"is_same_v<kokkos_dview::value_type, double> = "<<(std::is_same_v<kokkos_dview::value_type, double>)<<std::endl;
            std::cout<<"is_same_v<kokkos_dview::array_layout, Kokkos::LayoutRight> = "<<(std::is_same_v<kokkos_dview::array_layout, Kokkos::LayoutRight>)<<std::endl;
            std::cout<<"is_same_v<kokkos_dview::execution_space, Kokkos::CudaSpace> = "<<(std::is_same_v<kokkos_dview::execution_space, Kokkos::Cuda>)<<std::endl;
            std::cout<<"is_same_v<kokkos_dview::memory_space, Kokkos::CudaSpace> = "<<(std::is_same_v<kokkos_dview::memory_space, Kokkos::CudaSpace>)<<std::endl;
            std::cout<<"is_same_v<kokkos_dview::device_type, Kokkos::Device<Kokkos::CudaSpace, Kokkos::CudaSpace>> = "<<(std::is_same_v<kokkos_dview::device_type, Kokkos::Device<Kokkos::Cuda,Kokkos::CudaSpace>>)<<std::endl;
            std::cout<<"is_same_v<kokkos_dview::pointer_type, double*> = "<<(std::is_same_v<kokkos_dview::pointer_type, double*>)<<std::endl;
            std::cout<<"is_same_v<kokkos_dview::reference_type, double&> = "<<(std::is_same_v<kokkos_dview::reference_type, double&>)<<std::endl;
            std::cout<<"is_same_v<kokkos_dview::memory_traits, Kokkos::MemoryTraits<>> = "<<(std::is_same_v<kokkos_dview::memory_traits, Kokkos::MemoryTraits<>>)<<std::endl;
            std::cout<<"is_same_v<kokkos_dview::host_mirror_type, Kokkos::Serial> = "<<(std::is_same_v<kokkos_dview::host_mirror_type, Kokkos::DynRankView<double, Kokkos::LayoutRight, Kokkos::HostSpace>>)<<std::endl;
            if(drank_m.span_is_contiguous()){
                size_t memory_ = drank_m.span();
                std::cout<<"Total length = "<<drank_m.size()<<std::endl;
                std::cout<<"Span is contiguous, span = "<<memory_<<std::endl;
            }
            std::cout<<"stride = "<<drank_m.stride(0)<<std::endl;
            std::cout<<"stride = "<<drank_m.stride(1)<<std::endl;
            std::cout<<"stride = "<<drank_m.stride(2)<<std::endl;
            std::cout<<"extent = "<<drank_m.extent(0)<<std::endl;
            std::cout<<"extent = "<<drank_m.extent(1)<<std::endl;
            std::cout<<"extent = "<<drank_m.extent(2)<<std::endl;
            if(drank_m.is_allocated()){
                std::cout<<"reference count : "<<drank_m.use_count()<<std::endl;
            }



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