# Kokkos_internal

[![CI](https://github.com/Quntized/Kokkos_internal/actions/workflows/ci.yml/badge.svg)](https://github.com/Quntized/Kokkos_internal/actions/workflows/ci.yml)

This repository contains all the files inside `kokkos`, the repository's test files (apart from Kokkos). My entire goal for this project is to debug in Kokkos and find any potential issue hidden in internal functions. 

This repo will have distinct file containing internal test cases. (Currently it only contains the dynamic rank part, kokkos_pair, raw mdspan (a bit) and further will be added in future)

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

**Reference:** [Kokkos Core Library](https://github.com/kokkos/kokkos)
