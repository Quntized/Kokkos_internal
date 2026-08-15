#include <Kokkos_Core.hpp>


using ext = MDSPAN_IMPL_STANDARD_NAMESPACE::extents<int, 2, 3, 4>;
namespace Test {

TEST(extent_inv, extent_line_376){
    static_assert(std::is_same_v<typename ext::index_type,int>,"index_type is not int");
    static_assert(std::is_same_v<typename ext::rank_type,size_t>,"rank_type is not size_t");
    auto constexpr rank = ext::rank();
    static_assert(rank == 3,"rank is not 3");
    auto constexpr drank = ext::rank_dynamic();
    static_assert(drank == 0,"dynamic rank is not 0");
    using dext = MDSPAN_IMPL_STANDARD_NAMESPACE::extents<int, 2, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent, 4>;
    auto constexpr drank2 = dext::rank_dynamic();
    static_assert(drank2 == 1,"dynamic rank is not 1");
    using msa = MDSPAN_IMPL_STANDARD_NAMESPACE::detail::maybe_static_array<int, size_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent,2,3,4,MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent,5>;
    static_assert(std::is_same_v<msa::value_type,int>,"value_type is not int");
    static_assert(std::is_same_v<msa::static_value_type,size_t>,"static_value_type is not size_t");
    static_assert(msa::tag_value == MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent,"tag_value is not dynamic_extent");
    msa m1(23);
    std::array<size_t,1> d_rank = {4};
    msa m2(d_rank);
    auto m1_static = m1.static_value(0);
    EXPECT_EQ(m1_static,2);
    auto m2_dynamic = m2.value(3);
    std::cout<<"m2_dynamic: "<<m2_dynamic<<std::endl;
    std::cout<<"m1 size : "<<m1.size()<<std::endl;
}
TEST(extent_inv, extent_line_main376){
    using extent = MDSPAN_IMPL_STANDARD_NAMESPACE::extents<int, 2, 3, 4, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>;
    static_assert(extent::rank() == 4,"rank is not 4");
    static_assert(extent::rank_dynamic() == 1,"dynamic rank is not 1");
    //static_assert(extent::extent(0) == 2,"extent(0) is not 2");
    ///static_assert(extent::extent(3) == 0,"extent(3) is not 3"); as non static;
    extent ex;
    EXPECT_EQ(ex.extent(0),2);
    EXPECT_EQ(ex.extent(1),3);
    EXPECT_EQ(ex.extent(2),4);
    EXPECT_EQ(ex.extent(3),0);
    static_assert(extent::static_extent(0) == 2,"extent(0) is not 2");
}
}