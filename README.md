# Kokkos_internal

[![CI](https://github.com/Quntized/Kokkos_internal/actions/workflows/ci.yml/badge.svg)](https://github.com/Quntized/Kokkos_internal/actions/workflows/ci.yml)

This repository contains all the files inside `kokkos`, the repository's test files (apart from Kokkos). My entire goal for this project is to debug in Kokkos and find any potential issue hidden in internal functions. 

This repo will have distinct file containing internal test cases. (Currently it only contains the dynamic rank part, kokkos_pair, raw mdspan (a bit) and further will be added in future)
## Things That has been changed to the UPSTREAM:
List of contributions that have been made to the UPSTREAM repo Kokkos:

This is here for me to track what has been changed through my end.

*kokks:*

1) [Fix ambiguous overload in Kokkos::resize with ExecutionSpace](https://github.com/kokkos/kokkos/pull/9128) .
2) [Fix silent state mutation in tuning variable ID registry](https://github.com/kokkos/kokkos/pull/9096)
3) [Fix: Add bounds check to prevent OOB access in tuner dimension indexing](https://github.com/kokkos/kokkos/pull/9090)
4) [Fix: Correct type reference in IndexTypeTrait error reporting](https://github.com/kokkos/kokkos/pull/9085)
5) [Fix: unset the variable _UPPERCASE_CMAKE_BUILD_TYPE properly due to missing _](https://github.com/kokkos/kokkos/pull/9213)
6) [CMake:Remove dead-code in cuda compiler_id_cmake](https://github.com/kokkos/kokkos/pull/9204)
7) [small typo in kokkos_function.cmake](https://github.com/kokkos/kokkos/pull/9169)
8) [Fix: Address FIXMEs in DynRankView required_allocation_size and add unit tests](https://github.com/kokkos/kokkos/pull/9290)
9) [Fix typos, Bash quoting, and variable initialization in compiler scripts ](https://github.com/kokkos/kokkos/pull/9273)
10) [Refactoring AMD architecture loops](https://github.com/kokkos/kokkos/pull/9217)
11) [A small typo in README.md](https://github.com/kokkos/kokkos/pull/9355)
12) [Duplicate source file accumulation in CMake test generation](https://github.com/kokkos/kokkos/pull/9135)

*Mdspan:*

1) [Remove templated get function from static_array_impl::get ](https://github.com/kokkos/mdspan/pull/459)
2) [Fix return type of layout_padded operator](https://github.com/kokkos/mdspan/pull/465)

*Kokkos_Core_Wiki:*

1) [Docs chapter numbering fix.](https://github.com/kokkos/kokkos-core-wiki/pull/872)
2) [One left out chapter number fix](https://github.com/kokkos/kokkos-core-wiki/pull/875).

## Additional Project Linked to this Repo:

1) [CMake_project](https://github.com/Quntized/CMake_project_preparation):
    In this repository systematically approach which is used in Kokkos cmake files, using official [CMake](https://cmake.org/cmake/help/latest/index.html) and as well as [Kokkos_CMake](https://github.com/kokkos/kokkos/tree/develop/cmake) for proper investigation.
2) [Cuda_code](https://github.com/Quntized/Cuda_code):
    In this repository the kokkos official cuda folder investigation [Kokkos_Cuda](https://github.com/kokkos/kokkos/tree/develop/core/src/Cuda) and as well as [Cuda](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) official documentations used for proper investigation.

3) [DiscDomComKok](https://github.com/Quntized/DiscDomCom) : In this repository I tried some idea and tried some initial phase though not able to perfectly implement it. But I could work in future. If you find it useful, you can knock me. We can discuss grabbing with coffee. I would really appreciate.



## Quick Build (self-contained)


You can build with either way.

The project fetches Kokkos automatically via CMake `FetchContent`:

```bash
cmake -B build -S . \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Build with a pre-installed Kokkos

If you already have Kokkos built locally, point CMake at it:

```bash
# First build Kokkos (one-time):
git clone https://github.com/kokkos/kokkos.git
cmake -B kokkos/build -S kokkos \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ARCH_NATIVE=ON \
    -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF
cmake --build kokkos/build

# Then build this project:
cmake -B build -S . -DKokkos_DIR=kokkos/build
cmake --build build
ctest --test-dir build --output-on-failure
```

**Note** : coding agent was used only for code review as i work independently, i need a tool for checking things, that's why. 

Thank you for your time.

**Reference:** [Kokkos Core Library](https://github.com/kokkos/kokkos)
