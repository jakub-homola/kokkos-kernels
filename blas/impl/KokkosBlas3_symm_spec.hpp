// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOSBLAS3_SYMM_SPEC_HPP_
#define KOKKOSBLAS3_SYMM_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosBlas3_symm_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class execution_space, class AVIT, class BVIT, class CVIT>
struct symm_eti_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

//
// This Macro is for readability of the template arguments.
//
#define KOKKOSBLAS3_SYMM_ETI_SPEC_AVAIL_LAYOUT(SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, EXEC_SPACE, MEM_SPACE)  \
  template <>                                                                                             \
  struct symm_eti_spec_avail<EXEC_SPACE,                                                                  \
                             Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                      \
                             Kokkos::View<const SCALAR**, LAYOUTB, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                      \
                             Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,       \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                   \
    enum : bool { value = true };                                                                         \
  };

//
// This Macro provides the ETI specialization of symm
//
#define KOKKOSBLAS3_SYMM_ETI_SPEC_AVAIL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  KOKKOSBLAS3_SYMM_ETI_SPEC_AVAIL_LAYOUT(SCALAR, LAYOUT, LAYOUT, LAYOUT, EXEC_SPACE, MEM_SPACE)

// Include the actual specialization declarations
#include <KokkosBlas3_symm_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosBlas3_symm_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {

//
// symm
//

// Unification layer
template <class execution_space, class AVIT, class BVIT, class CVIT,
          bool tpl_spec_avail = symm_tpl_spec_avail<execution_space, AVIT, BVIT, CVIT>::value,
          bool eti_spec_avail = symm_eti_spec_avail<execution_space, AVIT, BVIT, CVIT>::value>
struct SYMM {
  static void symm(const execution_space& space, const char side[], const char uplo[],
                   typename CVIT::const_value_type& alpha, const AVIT& A, const BVIT& B,
                   typename CVIT::const_value_type& beta, const CVIT& C);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
template <class execution_space, class AVIT, class BVIT, class CVIT>
struct SYMM<execution_space, AVIT, BVIT, CVIT, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  static void symm(const execution_space& /*space*/, const char side[], const char uplo[],
                   typename CVIT::const_value_type& alpha, const AVIT& A, const BVIT& B,
                   typename CVIT::const_value_type& beta, const CVIT& C) {
    static_assert(Kokkos::is_view<AVIT>::value, "AVIT must be a Kokkos::View.");
    static_assert(Kokkos::is_view<BVIT>::value, "BVIT must be a Kokkos::View.");
    static_assert(Kokkos::is_view<CVIT>::value, "CVIT must be a Kokkos::View.");
    static_assert(static_cast<int>(AVIT::rank) == 2, "AVIT must have rank 2.");
    static_assert(static_cast<int>(BVIT::rank) == 2, "BVIT must have rank 2.");
    static_assert(static_cast<int>(CVIT::rank) == 2, "CVIT must have rank 2.");

    Kokkos::Profiling::pushRegion(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY ? "KokkosBlas::symm[ETI]"
                                                                     : "KokkosBlas::symm[noETI]");

    typename AVIT::host_mirror_type host_A = Kokkos::create_mirror_view(A);
    typename BVIT::host_mirror_type host_B = Kokkos::create_mirror_view(B);
    typename CVIT::host_mirror_type host_C = Kokkos::create_mirror_view(C);

    // Copy A, B to host and C to host
    // no-op if A, B, C MemorySpace is HostSpace
    Kokkos::deep_copy(host_A, A);
    Kokkos::deep_copy(host_B, B);
    Kokkos::deep_copy(host_C, C);

    SerialSymm_Invoke<typename AVIT::host_mirror_type, typename BVIT::host_mirror_type,
                      typename CVIT::host_mirror_type>(side, uplo, alpha, host_A, host_B, beta, host_C);

    // Copy host_C back to C
    // no-op if C's MemorySpace is HostSpace
    Kokkos::deep_copy(C, host_C);

    Kokkos::Profiling::popRegion();
  }
};
#endif  //! defined(KOKKOSKERNELS_ETI_ONLY) ||
        //! KOKKOSKERNELS_IMPL_COMPILE_LIBRARY

}  // namespace Impl
}  // namespace KokkosBlas

//
// These Macros are for readability.
//
#define KOKKOSBLAS3_SYMM_ETI_SPEC_DECL_LAYOUTS(SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, EXEC_SPACE, MEM_SPACE)   \
  extern template struct SYMM<EXEC_SPACE,                                                                  \
                              Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                      \
                              Kokkos::View<const SCALAR**, LAYOUTB, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                      \
                              Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,       \
                                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                      \
                              false, true>;

#define KOKKOSBLAS3_SYMM_ETI_SPEC_INST_LAYOUTS(SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, EXEC_SPACE, MEM_SPACE) \
  template struct SYMM<EXEC_SPACE,                                                                       \
                       Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,      \
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                           \
                       Kokkos::View<const SCALAR**, LAYOUTB, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,      \
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                           \
                       Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,            \
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                           \
                       false, true>;

//
// These Macros are only included when we are not compiling libkokkoskernels but
// are auto generating files. These macros provide the explicit instantiation
// declaration and definition of SYMM, potentially reducing user code size. The
// "extern template" skips the implicit instatiation step ensuring that the
// callers code uses this explicit instantiation definition of SYMM.
//
#define KOKKOSBLAS3_SYMM_ETI_SPEC_DECL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  KOKKOSBLAS3_SYMM_ETI_SPEC_DECL_LAYOUTS(SCALAR, LAYOUT, LAYOUT, LAYOUT, EXEC_SPACE, MEM_SPACE)

#include <generated_specializations_hpp/KokkosBlas3_symm_eti_spec_decl.hpp>

#define KOKKOSBLAS3_SYMM_ETI_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  KOKKOSBLAS3_SYMM_ETI_SPEC_INST_LAYOUTS(SCALAR, LAYOUT, LAYOUT, LAYOUT, EXEC_SPACE, MEM_SPACE)

#include <KokkosBlas3_symm_tpl_spec_decl.hpp>

#endif  // KOKKOSBLAS3_SYMM_SPEC_HPP_
