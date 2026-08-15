#include <Kokkos_Core.hpp>
#include <array>

namespace kokkos_num{
    
    template <class... Dim>
    struct vec_perp{
        static constexpr int dim = sizeof...(Dim);
        std::array<double, dim> data;
        template<typename... Values>
        vec_perp(Values... values) : data{static_cast<double>(values)...} {}
    }
    
    struct Foo{
        int val;
    }

}

using view_foo = Kokkos::View<kokkos_num::Foo*>;
view_foo foo_view("foo_view", 10);
using md = Kokkos::MDRangePolicy<Kokkos::Rank<1>,Kokkos::IndexType<int>>;
Kokkos::parallel_for("test", md({0},{10}),KOKKOS_LAMBDA(int i){
    foo_view(i).val = i;
});

