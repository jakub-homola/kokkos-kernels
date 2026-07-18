// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOSBLAS3_SYMM_HPP_
#define KOKKOSBLAS3_SYMM_HPP_

/// \file KokkosBlas3_symm.hpp

#include "KokkosKernels_Macros.hpp"
#include "KokkosBlas3_symm_spec.hpp"
#include "KokkosBlas1_scal.hpp"
#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Error.hpp"
#include <sstream>
#include <type_traits>

namespace KokkosBlas {

/// \brief Symmetric matrix-matrix multiply:
///
///        C = beta*C + alpha*A*B   if side == "L" or "l"
///        C = beta*C + alpha*B*A   if side == "R" or "r"
///
/// A is symmetric; only the triangle of A specified by uplo is
/// referenced (the other triangle is inferred by symmetry).
///
/// \tparam execution_space a Kokkos execution space to run the kernels on.
/// \tparam AViewType Input symmetric matrix, as a 2-D Kokkos::View
/// \tparam BViewType Input matrix, as a 2-D Kokkos::View
/// \tparam CViewType Input/Output matrix, as a nonconst 2-D Kokkos::View
///
/// \param space [in] an execution space instance that may contain a stream
///   or a queue to execute the kernel on, this only works with TPLs at the
///   moment.
/// \param side  [in] "L" or "l" indicates matrix A is on the left of B
///                   "R" or "r" indicates matrix A is on the right of B
/// \param uplo  [in] "U" or "u" indicates only the upper triangle of A is
///                   referenced
///                   "L" or "l" indicates only the lower triangle of A is
///                   referenced
/// \param alpha [in] Input coefficient of A*B or B*A
/// \param A [in]     Input matrix, as a 2-D Kokkos::View
///                   If side == "L" or "l", matrix A is a M-by-M symmetric
///                   matrix; otherwise, matrix A is a N-by-N symmetric matrix
/// \param B [in]     Input matrix, as a 2-D Kokkos::View, M-by-N
/// \param beta [in]  Input coefficient of C
/// \param C [in,out] Input/Output matrix, as a nonconst 2-D Kokkos::View,
///                   M-by-N
template <class execution_space, class AViewType, class BViewType, class CViewType>
void symm(const execution_space& space, const char side[], const char uplo[],
          typename CViewType::const_value_type& alpha, const AViewType& A, const BViewType& B,
          typename CViewType::const_value_type& beta, const CViewType& C) {
  static_assert(Kokkos::is_execution_space_v<execution_space>,
                "KokkosBlas::symm: execution_space must be a Kokkos::execution_space.");
  static_assert(Kokkos::is_view_v<AViewType>, "KokkosBlas::symm: AViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view_v<BViewType>, "KokkosBlas::symm: BViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view_v<CViewType>, "KokkosBlas::symm: CViewType must be a Kokkos::View.");
  static_assert(static_cast<int>(AViewType::rank) == 2, "KokkosBlas::symm: AViewType must have rank 2.");
  static_assert(static_cast<int>(BViewType::rank) == 2, "KokkosBlas::symm: BViewType must have rank 2.");
  static_assert(static_cast<int>(CViewType::rank) == 2, "KokkosBlas::symm: CViewType must have rank 2.");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename AViewType::memory_space>::accessible,
                "KokkosBlas::symm: AViewType must be accessible from execution_space");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename BViewType::memory_space>::accessible,
                "KokkosBlas::symm: BViewType must be accessible from execution_space");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename CViewType::memory_space>::accessible,
                "KokkosBlas::symm: CViewType must be accessible from execution_space");

  // Check validity of indicator arguments
  bool valid_side = (side[0] == 'L') || (side[0] == 'l') || (side[0] == 'R') || (side[0] == 'r');
  bool valid_uplo = (uplo[0] == 'U') || (uplo[0] == 'u') || (uplo[0] == 'L') || (uplo[0] == 'l');
  if (!valid_side) {
    std::ostringstream os;
    os << "KokkosBlas::symm: side = '" << side[0] << "'. "
       << "Valid values include 'L' or 'l' (A is on the left of B), "
          "'R' or 'r' (A is on the right of B).";
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }
  if (!valid_uplo) {
    std::ostringstream os;
    os << "KokkosBlas::symm: uplo = '" << uplo[0] << "'. "
       << "Valid values include 'U' or 'u' (the upper triangle of A is referenced), "
          "'L' or 'l' (the lower triangle of A is referenced).";
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  bool is_left = (side[0] == 'L' || side[0] == 'l');

  int64_t A_m = A.extent(0);
  int64_t A_n = A.extent(1);
  int64_t B_m = B.extent(0);
  int64_t B_n = B.extent(1);
  int64_t C_m = C.extent(0);
  int64_t C_n = C.extent(1);

  // Return if C is degenerate
  if (C_m == 0 || C_n == 0) return;

  // Ensure A is square, B and C match, and A's dimension is consistent with the side of the product it's on
  if (A_m != A_n || B_m != C_m || B_n != C_n || A_n != (is_left ? C_m : C_n)) {
    std::ostringstream os;
    os << "KokkosBlas::symm: Dimensions of A, B, and C do not match: "
       << "side: " << side[0] << " A: " << A.extent(0) << " x " << A.extent(1) << " B: " << B.extent(0) << " x "
       << B.extent(1) << " C: " << C.extent(0) << " x " << C.extent(1);
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  // Create A matrix view type alias
  using AViewInternalType = Kokkos::View<typename AViewType::const_value_type**, typename AViewType::array_layout,
                                         typename AViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  // Create B matrix view type alias
  using BViewInternalType = Kokkos::View<typename BViewType::const_value_type**, typename BViewType::array_layout,
                                         typename BViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  // Create C matrix view type alias
  using CViewInternalType = Kokkos::View<typename CViewType::non_const_value_type**, typename CViewType::array_layout,
                                         typename CViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  KokkosBlas::Impl::SYMM<execution_space, AViewInternalType, BViewInternalType, CViewInternalType>::symm(
      space, side, uplo, alpha, A, B, beta, C);
}

/// \brief Symmetric matrix-matrix multiply:
///
///        C = beta*C + alpha*A*B   if side == "L" or "l"
///        C = beta*C + alpha*B*A   if side == "R" or "r"
///
/// A is symmetric; only the triangle of A specified by uplo is
/// referenced (the other triangle is inferred by symmetry).
///
/// \tparam AViewType Input symmetric matrix, as a 2-D Kokkos::View
/// \tparam BViewType Input matrix, as a 2-D Kokkos::View
/// \tparam CViewType Input/Output matrix, as a nonconst 2-D Kokkos::View
///
/// \param side  [in] "L" or "l" indicates matrix A is on the left of B
///                   "R" or "r" indicates matrix A is on the right of B
/// \param uplo  [in] "U" or "u" indicates only the upper triangle of A is
///                   referenced
///                   "L" or "l" indicates only the lower triangle of A is
///                   referenced
/// \param alpha [in] Input coefficient of A*B or B*A
/// \param A [in]     Input matrix, as a 2-D Kokkos::View
/// \param B [in]     Input matrix, as a 2-D Kokkos::View
/// \param beta [in]  Input coefficient of C
/// \param C [in,out] Input/Output matrix, as a nonconst 2-D Kokkos::View
template <class AViewType, class BViewType, class CViewType>
void symm(const char side[], const char uplo[], typename CViewType::const_value_type& alpha, const AViewType& A,
          const BViewType& B, typename CViewType::const_value_type& beta, const CViewType& C) {
  symm(typename CViewType::execution_space{}, side, uplo, alpha, A, B, beta, C);
}

}  // namespace KokkosBlas

#endif  // KOKKOSBLAS3_SYMM_HPP_
