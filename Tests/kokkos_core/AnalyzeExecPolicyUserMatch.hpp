#include <Kokkos_Core.hpp>
#include <traits/Kokkos_ExecutionSpaceTrait.hpp>
#include <traits/Kokkos_TeamHandleTrait.hpp>
#include <traits/Kokkos_GraphKernelTrait.hpp>
#include <traits/Kokkos_IndexTypeTrait.hpp>
#include <traits/Kokkos_IterationPatternTrait.hpp>
#include <traits/Kokkos_LaunchBoundsTrait.hpp>
#include <traits/Kokkos_StaticBatchSizeTrait.hpp>
#include <traits/Kokkos_OccupancyControlTrait.hpp>
#include <traits/Kokkos_ScheduleTrait.hpp>
#include <traits/Kokkos_WorkItemPropertyTrait.hpp>
#include <traits/Kokkos_WorkTagTrait.hpp>
#include <impl/Kokkos_AnalyzePolicy.hpp>
#include <traits/Kokkos_Traits_fwd.hpp>
#include <iostream>
using Policy = Kokkos::Impl::AnalyzeExecPolicy<void,Kokkos::Cuda>; //searching ExecutionSpaceTrait
namespace Test{
TEST(AnalyzeExecPolicyUserMatch,line_52_AnalyzeP_file){
    using analyzeExecPolicyVoid = Kokkos::Impl::AnalyzeExecPolicy<void>;
    static_assert(analyzeExecPolicyVoid::execution_space_is_defaulted, "Not Ok");
    using analyzedPolicyExec = Kokkos::Impl::AnalyzeExecPolicy<void,Kokkos::Cuda>;
    //auto recv = analyzedPolicyExec::show_execution_space_error_in_compilation_message;
    static_assert(std::is_same_v<analyzedPolicyExec::execution_space,Kokkos::Cuda>,"NOT SAME");
    using analyzedPolicyExecSch = Kokkos::Impl::AnalyzeExecPolicy<void,Kokkos::Cuda,Kokkos::Schedule<Kokkos::Dynamic>>;
    static_assert(std::is_same_v<analyzedPolicyExecSch::base_t::base_t::schedule_type,Kokkos::Schedule<Kokkos::Dynamic>>,"NOT SAME");

}

}