#include <Kokkos_Core.hpp>
#include <desul/atomics/Common.hpp>
#include <type_traits>
#include <gtest/gtest.h>

namespace Test {

TEST(DesulAtomicTest, Dummy16ByteValue_Equality) {
  desul::Dummy16ByteValue val1{100, 200};
  desul::Dummy16ByteValue val2{100, 200};
  desul::Dummy16ByteValue val3{999, 200};
  desul::Dummy16ByteValue val4{100, 999};
  EXPECT_TRUE(val1 == val2);
  EXPECT_FALSE(val1 == val3);
  EXPECT_FALSE(val1 == val4);
}
TEST(DesulAtomicTest, Dummy16ByteValue_Inequality) {
  desul::Dummy16ByteValue val1{2, 3};
  desul::Dummy16ByteValue val2{2, 3};
  desul::Dummy16ByteValue val3{2, 4};
  EXPECT_FALSE(val1 != val2);
  EXPECT_TRUE(val1 != val3);
}
TEST(DesulAtomicTest, Dummy16ByteValueMemoryAlignment) {
  EXPECT_EQ(sizeof(desul::Dummy16ByteValue), 16);
  EXPECT_EQ(alignof(desul::Dummy16ByteValue), 16);
}
TEST(DesulAtomicTest, CmpExchFailureOrder_TypeMapping) {
  EXPECT_TRUE(
      (std::is_same<
          desul::Impl::cmpexch_failure_memory_order<desul::MemoryOrderAcqRel>,
          desul::MemoryOrderAcquire>::value));
  EXPECT_TRUE(
      (std::is_same<
          desul::Impl::cmpexch_failure_memory_order<desul::MemoryOrderRelease>,
          desul::MemoryOrderRelaxed>::value));
  EXPECT_TRUE((std::is_empty<desul::MemoryOrderSeqCst>::value));
  EXPECT_TRUE((std::is_empty<desul::MemoryOrderAcqRel>::value));
  EXPECT_TRUE((std::is_empty<desul::MemoryScopeDevice>::value));
  EXPECT_TRUE((std::is_empty<desul::MemoryScopeSystem>::value));
  EXPECT_EQ((sizeof(desul::MemoryOrderRelaxed)), 1);
  EXPECT_EQ((sizeof(desul::MemoryScopeCore)), 1);
  EXPECT_TRUE((std::is_trivial<desul::MemoryOrderAcquire>::value));
  EXPECT_TRUE(
      (std::is_same<
          desul::Impl::cmpexch_failure_memory_order<desul::MemoryOrderAcqRel>,
          desul::MemoryOrderAcquire>::value));
  EXPECT_TRUE(
      (std::is_same<
          desul::Impl::cmpexch_failure_memory_order<desul::MemoryOrderRelease>,
          desul::MemoryOrderRelaxed>::value));
  EXPECT_TRUE(
      (std::is_same<
          desul::Impl::cmpexch_failure_memory_order<desul::MemoryOrderAcquire>,
          desul::MemoryOrderAcquire>::value));
  EXPECT_TRUE(
      (std::is_same<
          desul::Impl::cmpexch_failure_memory_order<desul::MemoryOrderRelaxed>,
          desul::MemoryOrderRelaxed>::value));
  EXPECT_TRUE(
      (std::is_same<
          desul::Impl::cmpexch_failure_memory_order<desul::MemoryOrderSeqCst>,
          desul::MemoryOrderSeqCst>::value));
}
} // namespace Test