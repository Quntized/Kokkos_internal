#include "vec_perp.hpp"
#include <iostream>

int main(int argc, char* argv[]){
    Kokkos::initialize(argc, argv);
    std::vector<double> A = {1.0,2.0};
    std::vector<double> B = {3.0,4.0};
    auto result = vector_approach::dot_product<double>(A, B);
    std::cout << "Dot product: " << result << std::endl;

    Kokkos::finalize();
    return 0;
}