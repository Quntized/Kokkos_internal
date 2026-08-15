#include <Kokkos_Core.hpp>
#include <array>
#include <vector>


namespace vector_prep{
    template <class ValueType>
    using KokkosView = Kokkos::View<ValueType*,Kokkos::LayoutLeft, Kokkos::CudaSpace>;
    template<class ValueType>
    using Vector = std::vector<ValueType>;

}

namespace vector_approach{
    template<class ValueType>
    ValueType dot_product(vector_prep::Vector<ValueType> const& A, vector_prep::Vector<ValueType> const& B){
    if(A.size() != B.size())
    {
        throw std::invalid_argument("Vectors must be of the same size for dot product.");
    }
    const size_t size = A.size();
    vector_prep::KokkosView<ValueType> A_("A", size);
    vector_prep::KokkosView<ValueType> B_("B", size);
    vector_prep::KokkosView<ValueType> C_("C", size);
    auto host_A = Kokkos::create_mirror_view(A_);
    auto host_B = Kokkos::create_mirror_view(B_);
    for (size_t i = 0; i < size; ++i) {
        host_A(i) = A[i];
        host_B(i) = B[i];
    }
    Kokkos::deep_copy(A_,host_A);
    Kokkos::deep_copy(B_, host_B);
    double sum = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, size), KOKKOS_LAMBDA(int i, double& lsum){
        lsum += A_(i) * B_(i);
    }, sum);
    Kokkos::fence();

    return sum;
}
}

