#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <utility>

namespace Test {
    TEST(kokkos_pair, host_execution){
        Kokkos::pair<int, double> p_default;
        p_default.first = 10;
        p_default.second = 3.14;
        EXPECT_EQ(p_default.first, 10);
        EXPECT_DOUBLE_EQ(p_default.second,3.14);

        Kokkos::pair<int , double> c_param(1,2.0);
        EXPECT_EQ(c_param.first, 1);
        EXPECT_DOUBLE_EQ(c_param.second , 2.0);
        
        Kokkos::pair<int , double> c_copy(c_param);
        EXPECT_EQ(c_copy.first, 1);
        EXPECT_DOUBLE_EQ(c_copy.second , 2.0);

        std::pair<int , double> spair(12,11.0);
        Kokkos::pair<int, double> k_from_spair(spair);
        EXPECT_EQ(k_from_spair.first, 12);
        EXPECT_DOUBLE_EQ(k_from_spair.second, 11.0);

        std::pair<int, double> st_f_k = c_param.to_std_pair();
        EXPECT_EQ(st_f_k.first, 1);
        EXPECT_DOUBLE_EQ(st_f_k.second, 2.0);
    }
    void test_kokkos_pair_device(){
        Kokkos::View<Kokkos::pair<int,double>*> d_result("in device",1);
        Kokkos::parallel_for("TestKokkos",1,KOKKOS_LAMBDA(const int i){
            Kokkos::pair<int,double> p1(3,4.0);
            Kokkos::pair<int,double> p2(p1);
            Kokkos::pair<int,double> p3;
            p3 = p2;
            d_result(i) = p3;
        });
        Kokkos::fence();
        auto h_res = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),d_result);
        EXPECT_EQ(h_res(0).first,3);
        EXPECT_DOUBLE_EQ(h_res(0).second,4.0);
    }
    TEST(kokkos_pair,device_execution){
        test_kokkos_pair_device();
    }
    
}