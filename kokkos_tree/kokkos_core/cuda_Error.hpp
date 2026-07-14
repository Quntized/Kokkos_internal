#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Cuda/Kokkos_Cuda_Error.hpp>
#include <stdexcept>
#include <type_traits> // That's for std::is_void_v

namespace Test{
TEST(KokkosCudaError, InternalErrorThrowsRuntimeError){
    EXPECT_THROW({
        Kokkos::Impl::cuda_internal_error_throw(cudaErrorInvalidValue,"PlasmaKernal",__FILE__,__LINE__);
    }, std::runtime_error);
}
struct InternalCudaFunctor{
    Kokkos::View<int*,Kokkos::CudaSpace> data;
    InternalCudaFunctor(Kokkos::View<int*,Kokkos::CudaSpace> d):data(d){}
    KOKKOS_INLINE_FUNCTION void operator()(const int i) const{
        data(i) = i * 2;
    }
};

TEST(Kokkos_internal, InterCudaFunctorCase01){
    const int N=100000;
    using CudaPolicy = Kokkos::RangePolicy<Kokkos::Cuda>;
    Kokkos::View<int*,Kokkos::CudaSpace> data("data",N);
    Kokkos::parallel_for("TestCudaInternal",CudaPolicy(0,N),InternalCudaFunctor(data));
    Kokkos::fence();
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),data);
    for(int i=0;i<N;i++){
        EXPECT_EQ(h_data(i),i*2)<<"Kernel exec failed at index "<<i;
    }
}
TEST(Kokkos_internal, InterCudaFunctorCase02){
    const int N= 1028;
    Kokkos::View<int*,Kokkos::CudaSpace> data("data",N);
    using CudaPolicy = Kokkos::RangePolicy<Kokkos::Cuda>;
    using Functor = InternalCudaFunctor;
    CudaPolicy policy(0,N);
    Functor functor(data);
    using ImplParallelFor = Kokkos::Impl::ParallelFor<Functor,CudaPolicy,Kokkos::Cuda>;
    ImplParallelFor impl_parallel_for(functor,policy);
    impl_parallel_for.execute();
    Kokkos::Cuda().fence("Wait for internal launch to finish");
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),data);
    for(int i=0;i<N;i++){
        EXPECT_EQ(h_data(i),i*2)<<"Kernel exec failed at index "<<i;
    }
}
struct SpecialTag {};
struct TagDispatchFunctor{
    Kokkos::View<int*,Kokkos::CudaSpace> data;
    TagDispatchFunctor(Kokkos::View<int*,Kokkos::CudaSpace> d):data(d){}
    KOKKOS_INLINE_FUNCTION void operator()(const int i) const{
        data(i) = i * 2;
    }
    KOKKOS_INLINE_FUNCTION void operator()(const SpecialTag&,const int i) const{
        data(i) = i * 3;
    }
};
TEST(kokkos_internal, InterCudaFunctorCase03){
    const int N = 12800;
    using KView = Kokkos::View<int*,Kokkos::CudaSpace>;
    using ImplParallelFor = Kokkos::Impl::ParallelFor<TagDispatchFunctor,Kokkos::RangePolicy<Kokkos::Cuda>,Kokkos::Cuda>;
    using CudaPolicy = ImplParallelFor::Policy;
    KView d("data",N);
    ImplParallelFor imppf(TagDispatchFunctor(d),CudaPolicy(0,N));
    imppf.execute();
    Kokkos::Cuda().fence("Wait for void tag launch");
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
    for(int i = 0; i < N; ++i) {
        EXPECT_EQ(h_data(i), i * 2) << "Void overload failed!";
    }
}
}