# Kokkos_internal

[![CI](https://github.com/Quntized/Kokkos_internal/actions/workflows/ci.yml/badge.svg)](https://github.com/Quntized/Kokkos_internal/actions/workflows/ci.yml)

This repository contains only the `dynamic_view_rank` files inside `kokkos/containers` . My entirely goal for this project is to debug (and simultaneously understand the behind the scene of kokkos) . 
This repo will have distinct file containing internal test case, and also for my learning purposes

## Quick Build (self-contained)


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

## CI

Every push and pull request triggers a [GitHub Actions workflow](.github/workflows/ci.yml) that:
1. Fetches Kokkos (cached for speed)
2. Builds with Serial + OpenMP backends
3. Runs the test suite via CTest
