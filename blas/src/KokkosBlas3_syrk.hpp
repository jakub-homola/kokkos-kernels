// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOSBLAS3_SYRK_HPP_
#define KOKKOSBLAS3_SYRK_HPP_

/// \file KokkosBlas3_syrk.hpp

#include "KokkosKernels_Macros.hpp"
#include "KokkosBlas3_syrk_spec.hpp"
#include "KokkosBlas1_scal.hpp"
#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Error.hpp"
#include <sstream>
#include <type_traits>

namespace KokkosBlas {

/// \brief Symmetric rank-k update:
///
///        C = beta*C + alpha*A*A**T   if trans == "N" or "n"
///        C = beta*C + alpha*A**T*A   if trans == "T" or "t"
///
/// Only the triangle of C specified by uplo is referenced and updated;
/// the other triangle of C is left untouched.
///
/// \tparam execution_space a Kokkos execution space to run the kernels on.
/// \tparam AViewType Input matrix, as a 2-D Kokkos::View
/// \tparam CViewType Input/Output symmetric matrix, as a 2-D Kokkos::View
///
/// \param space [in] an execution space instance that may contain a stream
///   or a queue to execute the kernel on, this only works with TPLs at the
///   moment.
/// \param uplo  [in] "U" or "u" indicates only the upper triangle of C is
///                   used/updated
///                   "L" or "l" indicates only the lower triangle of C is
///                   used/updated
/// \param trans [in] "N" or "n" for C := beta*C + alpha*A*A**T
///                   "T" or "t" for C := beta*C + alpha*A**T*A
/// \param alpha [in] Input coefficient of A*A**T or A**T*A
/// \param A [in]     Input matrix, as a 2-D Kokkos::View
///                   If trans == "N" or "n", matrix A is N-by-K
///                   otherwise, matrix A is K-by-N
/// \param beta [in]  Input coefficient of C
/// \param C [in,out] Input/Output matrix, as a nonconst 2-D Kokkos::View
///                   On entry, N-by-N symmetric matrix (only the uplo
///                   triangle is significant)
///                   On exit, overwritten with the result (only the uplo
///                   triangle is updated)
template <class execution_space, class AViewType, class CViewType>
void syrk(const execution_space& space, const char uplo[], const char trans[],
          typename CViewType::const_value_type& alpha, const AViewType& A,
          typename CViewType::const_value_type& beta, const CViewType& C) {
  static_assert(Kokkos::is_execution_space_v<execution_space>,
                "KokkosBlas::syrk: execution_space must be a Kokkos::execution_space.");
  static_assert(Kokkos::is_view_v<AViewType>, "KokkosBlas::syrk: AViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view_v<CViewType>, "KokkosBlas::syrk: CViewType must be a Kokkos::View.");
  static_assert(static_cast<int>(AViewType::rank) == 2, "KokkosBlas::syrk: AViewType must have rank 2.");
  static_assert(static_cast<int>(CViewType::rank) == 2, "KokkosBlas::syrk: CViewType must have rank 2.");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename AViewType::memory_space>::accessible,
                "KokkosBlas::syrk: AViewType must be accessible from execution_space");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename CViewType::memory_space>::accessible,
                "KokkosBlas::syrk: CViewType must be accessible from execution_space");

  // Check validity of indicator argument
  bool valid_uplo  = (uplo[0] == 'U') || (uplo[0] == 'u') || (uplo[0] == 'L') || (uplo[0] == 'l');
  bool valid_trans = (trans[0] == 'N') || (trans[0] == 'n') || (trans[0] == 'T') || (trans[0] == 't');
  if (!valid_uplo) {
    std::ostringstream os;
    os << "KokkosBlas::syrk: uplo = '" << uplo[0] << "'. "
       << "Valid values include 'U' or 'u' (use/update the upper triangle of C), "
          "'L' or 'l' (use/update the lower triangle of C).";
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }
  if (!valid_trans) {
    std::ostringstream os;
    os << "KokkosBlas::syrk: trans = '" << trans[0] << "'. "
       << "Valid values include 'N' or 'n' (C := beta*C + alpha*A*A**T), "
          "'T' or 't' (C := beta*C + alpha*A**T*A).";
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  bool A_not_trans = (trans[0] == 'N' || trans[0] == 'n');

  int64_t A_m = A.extent(0);
  int64_t A_n = A.extent(1);
  int64_t C_m = C.extent(0);
  int64_t C_n = C.extent(1);

  // Return if C is degenerate
  if (C_m == 0 || C_n == 0) return;

  // Ensure that C is square and that A's dimensions are consistent with C
  if (C_m != C_n || (A_not_trans ? A_m : A_n) != C_m) {
    std::ostringstream os;
    os << "KokkosBlas::syrk: Dimensions of A and C do not match: "
       << "trans: " << trans[0] << " A: " << A.extent(0) << " x " << A.extent(1) << " C: " << C.extent(0) << " x "
       << C.extent(1);
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  // Simply scale C if A's contraction dimension (K) is 0
  if ((A_not_trans ? A_n : A_m) == 0) {
    scal(space, C, beta, C);
    return;
  }

  // Create A matrix view type alias
  using AViewInternalType = Kokkos::View<typename AViewType::const_value_type**, typename AViewType::array_layout,
                                         typename AViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  // Create C matrix view type alias
  using CViewInternalType = Kokkos::View<typename CViewType::non_const_value_type**, typename CViewType::array_layout,
                                         typename CViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  KokkosBlas::Impl::SYRK<execution_space, AViewInternalType, CViewInternalType>::syrk(space, uplo, trans, alpha, A,
                                                                                       beta, C);
}

/// \brief Symmetric rank-k update:
///
///        C = beta*C + alpha*A*A**T   if trans == "N" or "n"
///        C = beta*C + alpha*A**T*A   if trans == "T" or "t"
///
/// Only the triangle of C specified by uplo is referenced and updated;
/// the other triangle of C is left untouched.
///
/// \tparam AViewType Input matrix, as a 2-D Kokkos::View
/// \tparam CViewType Input/Output symmetric matrix, as a 2-D Kokkos::View
///
/// \param uplo  [in] "U" or "u" indicates only the upper triangle of C is
///                   used/updated
///                   "L" or "l" indicates only the lower triangle of C is
///                   used/updated
/// \param trans [in] "N" or "n" for C := beta*C + alpha*A*A**T
///                   "T" or "t" for C := beta*C + alpha*A**T*A
/// \param alpha [in] Input coefficient of A*A**T or A**T*A
/// \param A [in]     Input matrix, as a 2-D Kokkos::View
/// \param beta [in]  Input coefficient of C
/// \param C [in,out] Input/Output matrix, as a nonconst 2-D Kokkos::View
template <class AViewType, class CViewType>
void syrk(const char uplo[], const char trans[], typename CViewType::const_value_type& alpha, const AViewType& A,
          typename CViewType::const_value_type& beta, const CViewType& C) {
  syrk(typename CViewType::execution_space{}, uplo, trans, alpha, A, beta, C);
}

}  // namespace KokkosBlas

#endif  // KOKKOSBLAS3_SYRK_HPP_
