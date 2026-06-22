# Kokkos_internal
This repository contains only the `dynamic_view_rank` files inside `kokkos/containers` . My entirely goal for this project is to debug (and simultaneously understand the behind the scene of kokkos) . 
This repo will have distinct file containing internal test case.

```
#to run this you need to first clone the kokkos repository. Given below:
git clone https://github.com/kokkos/kokkos.git
cmake -B builddir \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ARCH_NATIVE=ON \
    -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF
    -DKokkos_ENABLE_CUDA=ON
#You can change the backend based on your compiler. For my case ENABLE_CUDA and also can be done for host case with SERIAL ON, provided in `https://kokkos.org/kokkos-core-wiki/get-started/building-from-source.html`
cmake --build builddir
#now for this repo, inside specific folder run 
cmake -DKokkos_COMPILE_LAUNCHER=<your_specific_kokkos_bin_folder>/kokkos_launch_compiler       -DKokkos_NVCC_WRAPPER=<your_specific_kokkos_bin_folder>/nvcc_wrapper -DKokkos_DIR=<your_kokkos_build_dir>/build -B build -S .
cmake --build build
./build/tests
```

