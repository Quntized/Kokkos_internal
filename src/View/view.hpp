#include <Kokkos_Core.hpp>
#include <utility>
template <class ViewType>
KOKKOS_FUNCTION void launch_parallel_kernel(ViewType& view)
{
    Kokkos::parallel_for("",Kokkos::RangePolicy<>(0, view.size()), KOKKOS_LAMBDA(int i){
        view(i) = static_cast<typename ViewType::value_type>(i);
    });
    auto host_view = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(host_view, view);
    for(size_t i = 0; i < host_view.size(); i++)
    {
        if(i%20 == 0){
            printf("view(%zu) = %f\n", i, static_cast<double>(host_view(i)));
        }
    }
}
template <class ViewType, std::integral... Sizes>
void create_required_span_rank(size_t expected_size, std::string label, Sizes... sizes)
{
    ViewType view(label,sizes...);
    size_t req_allocation_size = ViewType::required_allocation_size(sizes...); //this goes to Kokkos::mdspan
    //printf("View %s: expected size = %zu, required allocation size = %zu\n", label.c_str(), expected_size, req_allocation_size);
    if(expected_size == req_allocation_size)
    {
        launch_parallel_kernel(view);
    }
}
template<class ExecutionSpace, class ExtentsType>
auto make_mdrange_from_extents(const ExtentsType &extents,std::index_sequence<0>)
{
    return Kokkos::RangePolicy<ExecutionSpace>{0,static_cast<ExtentsType::index_type>(extents.extent(0))};
}

template<class ExecutionSpace, class ExtentsType, std::size_t... Indices>
auto make_mdrange_from_extents(const ExtentsType &extents, std::index_sequence<Indices...>)
{
    using index_type = typename ExtentsType::index_type;
    using rank_type = ExtentsType::rank_type;
    rank_type rank = ExtentsType::rank();
    return Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<rank>,Kokkos::IndexType<index_type>>{{static_cast<index_type>(0*Indices)...}, {static_cast<index_type>(extents.extent(Indices))...}};
}
template<class T, class ExtentsType,class LayoutType>
auto make_Bview_from_checked_accessor(const ExtentsType &extents){
    using extents_type = ExtentsType;
    using layout_type = LayoutType;
    using accessor_type = Kokkos::Impl::CheckedReferenceCountedAccessor<T, typename ExecutionSpace::memory_space>;
    using view_type = Kokkos::Impl::BV::BasicView<T,extents_type, layout_type, accessor_type>;
    view_type view("Test View", extents);
    for (int r = 0; r<static_cast<int>(view_type::rank()); r++){
        expected_size *= extents.extent(r);
    }
    auto size = view.size();
    return std::pair(view, size, expected_size);
}
template<class T, class ExtentsType, class LayoutType>
auto make_BView_from_mapping_and_padding(const ExtentsType &extents, std::size_t padding){
    using extents_type = ExtentsType;
    using layout_type = LayoutType;
    mapping_type = typename layout_type::template mapping_type<ExtentsType>;
    using accessor_type = Kokkos::Impl::CheckedReferenceCountedAccessor<T,typename ExecutionSpace::memory_space>;
    using view_type = Kokkos::Impl::BV::BasicView<T,extents_type,layout_type, accessor_type>;
    auto mapping = mappint_type(extents, padding);
    view_type("Bview", mapping);
    for (int r =0; r<static_cast<int>(view_type::rank()); r++){
        expected_size *= view.extent(r);
    }
    std::size_t size = view.size();
    return std::pair(view, size, expected_size);
}
template <class ExecutionSpace, class FunctorType,class T, class ExtentsType, class LayoutType>
auto make_parallel_run_from_extents(const ExtentsType &extents){
    using extents_type = ExtentsType;
    using layout_type = LayoutType;
    //using mapping_type = typename layout_type::mapping_type<ExtentsType>;
    using accessor_type = Kokkos::Impl::CheckedReferenceCountedAccessor<T,typename ExecutionSpace::memory_space>;
    using view_type = Kokkos::Impl::BV::BasicView<T,extents_type,layout_type,accessor_type>;
    auto view = view_type("Bview", extents);
    auto mdrange = make_mdrange_from_extents<ExecutionSpace>(extents,std::make_index_sequence<ExtentsType::rank()>{});
    Kokkos::parallel_for("Label", mdrange_policy, mdrange, FuntorType{});
    auto host_view = create_mirror_view(view);
    Kokkos::deep_copy(host_view,view);
    return host_view;
}


