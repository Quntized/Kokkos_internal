#include "view.hpp"

int main(int argc,char* argv[]){
    Kokkos::initialize(argc,argv);
    create_required_span_rank<Kokkos::View<double*>> (1000000*8,"view1",1000000);
    Kokkos::finalize();
    return 0;
}