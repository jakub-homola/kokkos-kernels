// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOSBLAS2_SYMV_SPEC_HPP_
#define KOKKOSBLAS2_SYMV_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosBlas2_symv_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class ExecutionSpace, class AVIT, class XVIT, class YVIT>
struct symv_eti_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization availability
// KokkosBlas::Impl::SYMV.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS2_SYMV_ETI_SPEC_AVAIL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE)                           \
  template <>                                                                                            \
  struct symv_eti_spec_avail<EXEC_SPACE,                                                                 \
                             Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                     \
                             Kokkos::View<const SCALAR*, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,  \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                     \
                             Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,        \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                  \
    enum : bool { value = true };                                                                        \
  };

// Include the actual specialization declarations
#include <KokkosBlas2_symv_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosBlas2_symv_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {

//
// symv
//

// Unification layer
template <class ExecutionSpace, class AVIT, class XVIT, class YVIT,
          bool tpl_spec_avail = symv_tpl_spec_avail<ExecutionSpace, AVIT, XVIT, YVIT>::value,
          bool eti_spec_avail = symv_eti_spec_avail<ExecutionSpace, AVIT, XVIT, YVIT>::value>
struct SYMV {
  static void symv(const ExecutionSpace& space, const char uplo[], typename YVIT::const_value_type& alpha,
                   const AVIT& A, const XVIT& x, typename YVIT::const_value_type& beta, const YVIT& y);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
template <class ExecutionSpace, class AVIT, class XVIT, class YVIT>
struct SYMV<ExecutionSpace, AVIT, XVIT, YVIT, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  static void symv(const ExecutionSpace& /*space*/, const char uplo[], typename YVIT::const_value_type& alpha,
                   const AVIT& A, const XVIT& x, typename YVIT::const_value_type& beta, const YVIT& y) {
    static_assert(Kokkos::is_view<AVIT>::value, "AVIT must be a Kokkos::View.");
    static_assert(Kokkos::is_view<XVIT>::value, "XVIT must be a Kokkos::View.");
    static_assert(Kokkos::is_view<YVIT>::value, "YVIT must be a Kokkos::View.");
    static_assert(static_cast<int>(AVIT::rank) == 2, "AVIT must have rank 2.");
    static_assert(static_cast<int>(XVIT::rank) == 1, "XVIT must have rank 1.");
    static_assert(static_cast<int>(YVIT::rank) == 1, "YVIT must have rank 1.");

    Kokkos::Profiling::pushRegion(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY ? "KokkosBlas::symv[ETI]"
                                                                     : "KokkosBlas::symv[noETI]");

    typename AVIT::host_mirror_type host_A = Kokkos::create_mirror_view(A);
    typename XVIT::host_mirror_type host_x = Kokkos::create_mirror_view(x);
    typename YVIT::host_mirror_type host_y = Kokkos::create_mirror_view(y);

    // Copy A, x to host and y to host
    // no-op if A, x, y MemorySpace is HostSpace
    Kokkos::deep_copy(host_A, A);
    Kokkos::deep_copy(host_x, x);
    Kokkos::deep_copy(host_y, y);

    SerialSymv_Invoke<typename AVIT::host_mirror_type, typename XVIT::host_mirror_type,
                      typename YVIT::host_mirror_type>(uplo, alpha, host_A, host_x, beta, host_y);

    // Copy host_y back to y
    // no-op if y's MemorySpace is HostSpace
    Kokkos::deep_copy(y, host_y);

    Kokkos::Profiling::popRegion();
  }
};
#endif  //! defined(KOKKOSKERNELS_ETI_ONLY) ||
        //! KOKKOSKERNELS_IMPL_COMPILE_LIBRARY

}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization of
// KokkosBlas::Impl::SYMV.  This is NOT for users!!!
// All the declarations of full specializations go in this header
// file.  We may spread out definitions (see _DEF macro below) across
// one or more .cpp files.
//

#define KOKKOSBLAS2_SYMV_ETI_SPEC_DECL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE)                                         \
  extern template struct SYMV<                                                                                       \
      EXEC_SPACE,                                                                                                     \
      Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                         \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                         \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      false, true>;

#include <generated_specializations_hpp/KokkosBlas2_symv_eti_spec_decl.hpp>

#define KOKKOSBLAS2_SYMV_ETI_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE)                                         \
  template struct SYMV<                                                                                              \
      EXEC_SPACE,                                                                                                     \
      Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                         \
      Kokkos::View<const SCALAR*, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                         \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      false, true>;

#include <KokkosBlas2_symv_tpl_spec_decl.hpp>

#endif  // KOKKOSBLAS2_SYMV_SPEC_HPP_
