enable_testing()
add_executable(Kokkos_qun_test test_dyn_rank.cpp)
target_link_libraries(Kokkos_qun_test PRIVATE Kokkos::kokkos GTest::gtest)

add_test(
  NAME    Kokkos_qun_test
  COMMAND Kokkos_qun_test
)
