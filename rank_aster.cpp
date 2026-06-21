#include <iostream>
#include <type_traits>

namespace Impl {
    template <class T, size_t Rank>
    struct ViewDataTypeRank {
        using type = typename ViewDataTypeRank<T, Rank - 1>::type*;
    };
    template <class T>
    struct ViewDataTypeRank<T, 0> {
        using type = T;
    };
}

int main() {
    using RankType = Impl::ViewDataTypeRank<int, 3>::type;
    if(std::is_same_v<RankType, int***>) {
        std::cout << "RankType is int***" << std::endl;
    } else {
        std::cout << "RankType is not int***" << std::endl;
    }
}