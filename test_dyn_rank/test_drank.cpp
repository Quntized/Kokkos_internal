

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
            std::cout<<std::is_same_v<s,view__> <<std::endl;
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