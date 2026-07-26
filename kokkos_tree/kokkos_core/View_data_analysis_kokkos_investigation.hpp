#include <Kokkos_Core.hpp>
#include <type_traits>
//#include <View/Kokkos_ViewDataAnalysis.hpp>

namespace Test{

TEST(ViewDataType,ViewDataType_linefrom251){
    static_assert(std::is_same_v<Kokkos::Impl::ViewDataType<double,Kokkos::Impl::ViewDimension<>>::type, double>, "ViewDataType<double,Kokkos::Impl::ViewDimension<>>::type should be double");
    static_assert(std::is_same_v<Kokkos::Impl::ViewDataType<double,Kokkos::Impl::ViewDimension<0>>::type, double*>, "ViewDataType<double,Kokkos::Impl::ViewDimension<0>>::type should be double*");
    static_assert(std::is_same_v<Kokkos::Impl::ViewDataType<double,Kokkos::Impl::ViewDimension<0,0>>::type, double**>, "ViewDataType<double,Kokkos::Impl::ViewDimension<0,0>>::type should be double**");
    static_assert(std::is_same_v<Kokkos::Impl::ViewDataType<double,Kokkos::Impl::ViewDimension<0,0,5>>::type, double**[5]>, "ViewDataType<double,Kokkos::Impl::ViewDimension<0,0,5>>::type should be double**[5]");
}
TEST(ViewArrayAnalysis,ViewArrayAnalysis_line_289){
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double>::value_type, double>, "ViewArrayAnalysis<double>::value_type should be double");
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double>::const_value_type, const double>, "ViewArrayAnalysis<double>::const_value_type should be const double");
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<const double>::non_const_value_type, double>, "ViewArrayAnalysis<const double>::non_const_value_type should be double");
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double>::static_dimension, Kokkos::Impl::ViewDimension<>>, "ViewArrayAnalysis<double>::static_dimension should be Kokkos::Impl::ViewDimension<>");
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double>::dynamic_dimension, Kokkos::Impl::ViewDimension<>>, "ViewArrayAnalysis<double>::dynamic_dimension should be Kokkos::Impl::ViewDimension<>");
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double>::dimension, Kokkos::Impl::ViewDimension<>>, "ViewArrayAnalysis<double>::dimension should be Kokkos::Impl::ViewDimension<>");
    //static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double[5]>::nested,Kokkos::Impl::ViewArrayAnalysis<double>>, "ViewArrayAnalysis<double[5]>::nested should be Kokkos::Impl::ViewArrayAnalysis<double>"); this is private can't invoke.
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double[5]>::value_type, double>, "ViewArrayAnalysis<double[5]>::value_type should be double");
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double[5]>::const_value_type, const double>, "ViewArrayAnalysis<double[5]>::const_value_type should be const double");
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double[5]>::non_const_value_type, double>, "ViewArrayAnalysis<double[5]>::non_const_value_type should be double");
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double[5]>::static_dimension, Kokkos::Impl::ViewDimension<5>>, "ViewArrayAnalysis<double[5]>::static_dimension should be Kokkos::Impl::ViewDimension<5>");
    //static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double[5]>::dynamic_dimension, Kokkos::Impl::ViewDimension<>>, "ViewArrayAnalysis<double[5]>::dynamic_dimension should be Kokkos::Impl::ViewDimension<>");
    static_assert(Kokkos::Impl::ViewArrayAnalysis<double[5]>::static_dimension::ArgN0 == 5, "ViewArrayAnalysis<double[5]>::static_dimension::ArgN0 should be 5");
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double[5][4][6]>::value_type,double>, "ViewArrayAnalysis<double[5][4][6]>::value_type should bematch");
    auto constexpr argn0 = Kokkos::Impl::ViewArrayAnalysis<double[5][4][6]>::static_dimension::ArgN0;
    auto constexpr argn1 = Kokkos::Impl::ViewArrayAnalysis<double[5][4][6]>::static_dimension::ArgN1;
    auto constexpr argn2 = Kokkos::Impl::ViewArrayAnalysis<double[5][4][6]>::static_dimension::ArgN2;
    auto constexpr rank = Kokkos::Impl::ViewArrayAnalysis<double[5][4][6]>::static_dimension::rank;
    auto constexpr rank_dyn = Kokkos::Impl::ViewArrayAnalysis<double[5][4][6]>::static_dimension::rank_dynamic;
    std::cout<<"ArgN0 " <<argn0<<"\n";
    std::cout<<"ArgN1 " <<argn1<<"\n";
    std::cout<<"ArgN2 " <<argn2<<"\n";
    std::cout<<"rank "<<rank<<"\n";
    std::cout<<"rank_dun "<<rank_dyn<<"\n";
    using vaa = Kokkos::Impl::ViewArrayAnalysis<double[]>;
    auto constexpr dargn0 = vaa::dynamic_dimension::ArgN0;
    std::cout<<"Dynamic argn0 "<<dargn0<<"\n";
    std::cout<<"Dynamic rank = "<<vaa::dynamic_dimension::rank_dynamic<<"\n";
    using mvaa = Kokkos::Impl::ViewArrayAnalysis<double[]>;
    auto constexpr mdargn0 = mvaa::dynamic_dimension::ArgN0;
    auto constexpr mdargn1 = mvaa::dynamic_dimension::ArgN1;
    auto constexpr mdargn2 = mvaa::dynamic_dimension::ArgN2;
    auto constexpr mdrank = mvaa::dynamic_dimension::rank;
    auto constexpr mdrankdyn = mvaa::dynamic_dimension::rank_dynamic;
    std::cout<<"mdArgN0 " <<mdargn0<<"\n";
    std::cout<<"mdArgN1 " <<mdargn1<<"\n";
    std::cout<<"mdArgN2 " <<mdargn2<<"\n";
    std::cout<<"mdrank "<<mdrank<<"\n";
    std::cout<<"drank_dyn "<<mdrankdyn<<"\n";
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double***>::dynamic_dimension,Kokkos::Impl::ViewDimension<0,0,0>>, "ViewArrayAnalysis<double***>::dynamic_dimension should be Kokkos::Impl::ViewDimension<0,0,0>");
    static_assert(std::is_same_v<Kokkos::Impl::ViewArrayAnalysis<double***>::dynamic_dimension,Kokkos::Impl::ViewArrayAnalysis<double***>::dimension>, "ViewArrayAnalysis<double***>::dynamic_dimension should be Kokkos::Impl::ViewArrayAnalysis<double***>::dimension");

}
}