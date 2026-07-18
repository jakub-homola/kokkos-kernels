// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOSBLAS2_SYMV_HPP_
#define KOKKOSBLAS2_SYMV_HPP_

/// \file KokkosBlas2_symv.hpp

#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Error.hpp"
#include <KokkosBlas2_symv_spec.hpp>
#include <sstream>
#include <type_traits>

namespace KokkosBlas {

/// \brief Symmetric matrix-vector multiply: y = beta*y + alpha*A*x.
///
/// A is symmetric; only the triangle of A specified by uplo is
/// referenced (the other triangle is inferred by symmetry).
///
/// \tparam ExecutionSpace The type of execution space
/// \tparam AViewType Input symmetric matrix, as a 2-D Kokkos::View
/// \tparam XViewType Input vector, as a 1-D Kokkos::View
/// \tparam YViewType Output vector, as a nonconst 1-D Kokkos::View
///
/// \param space [in] execution space instance on which to run the
///   kernel. This may contain information about which stream to
///   run on.
/// \param uplo  [in] "U" or "u" indicates only the upper triangle of A is
///                   referenced
///                   "L" or "l" indicates only the lower triangle of A is
///                   referenced
/// \param alpha [in] Input coefficient of A*x
/// \param A [in] Input matrix, as a 2-D Kokkos::View, N-by-N symmetric
/// \param x [in] Input vector, as a 1-D Kokkos::View, length N
/// \param beta [in] Input coefficient of y
/// \param y [in/out] Output vector, as a nonconst 1-D Kokkos::View, length N
template <class ExecutionSpace, class AViewType, class XViewType, class YViewType>
void symv(const ExecutionSpace& space, const char uplo[], typename YViewType::const_value_type& alpha,
          const AViewType& A, const XViewType& x, typename YViewType::const_value_type& beta, const YViewType& y) {
  static_assert(Kokkos::is_execution_space_v<ExecutionSpace>,
                "KokkosBlas::symv: ExecutionSpace must be a valid Kokkos "
                "execution space.");
  static_assert(Kokkos::is_view<AViewType>::value, "KokkosBlas::symv: AViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<XViewType>::value, "KokkosBlas::symv: XViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<YViewType>::value, "KokkosBlas::symv: YViewType must be a Kokkos::View.");
  static_assert(static_cast<int>(AViewType::rank) == 2, "KokkosBlas::symv: AViewType must have rank 2.");
  static_assert(static_cast<int>(XViewType::rank) == 1, "KokkosBlas::symv: XViewType must have rank 1.");
  static_assert(static_cast<int>(YViewType::rank) == 1, "KokkosBlas::symv: YViewType must have rank 1.");
  static_assert(Kokkos::SpaceAccessibility<ExecutionSpace, typename AViewType::memory_space>::accessible,
                "KokkosBlas::symv: AViewType must be accessible from ExecutionSpace");
  static_assert(Kokkos::SpaceAccessibility<ExecutionSpace, typename XViewType::memory_space>::accessible,
                "KokkosBlas::symv: XViewType must be accessible from ExecutionSpace");
  static_assert(Kokkos::SpaceAccessibility<ExecutionSpace, typename YViewType::memory_space>::accessible,
                "KokkosBlas::symv: YViewType must be accessible from ExecutionSpace");

  bool valid_uplo = (uplo[0] == 'U') || (uplo[0] == 'u') || (uplo[0] == 'L') || (uplo[0] == 'l');
  if (!valid_uplo) {
    std::ostringstream os;
    os << "KokkosBlas::symv: uplo = '" << uplo[0] << "'. "
       << "Valid values include 'U' or 'u' (the upper triangle of A is referenced), "
          "'L' or 'l' (the lower triangle of A is referenced).";
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  // Check compatibility of dimensions at run time.
  if (A.extent(0) != A.extent(1) || A.extent(0) != x.extent(0) || A.extent(0) != y.extent(0)) {
    std::ostringstream os;
    os << "KokkosBlas::symv: Dimensions of A, x, and y do not match: "
       << "A: " << A.extent(0) << " x " << A.extent(1) << ", x: " << x.extent(0) << ", y: " << y.extent(0);
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  // Return if y is degenerate
  if (y.extent(0) == 0) return;

  // Create A matrix view type alias
  using AViewInternalType = Kokkos::View<typename AViewType::const_value_type**, typename AViewType::array_layout,
                                         typename AViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  // Create x vector view type alias
  using XViewInternalType = Kokkos::View<typename XViewType::const_value_type*, typename XViewType::array_layout,
                                         typename XViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  // Create y vector view type alias
  using YViewInternalType = Kokkos::View<typename YViewType::non_const_value_type*, typename YViewType::array_layout,
                                         typename YViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  KokkosBlas::Impl::SYMV<ExecutionSpace, AViewInternalType, XViewInternalType, YViewInternalType>::symv(
      space, uplo, alpha, A, x, beta, y);
}

/// \brief Symmetric matrix-vector multiply: y = beta*y + alpha*A*x.
///
/// A is symmetric; only the triangle of A specified by uplo is
/// referenced (the other triangle is inferred by symmetry).
///
/// \tparam AViewType Input symmetric matrix, as a 2-D Kokkos::View
/// \tparam XViewType Input vector, as a 1-D Kokkos::View
/// \tparam YViewType Output vector, as a nonconst 1-D Kokkos::View
///
/// \param uplo  [in] "U" or "u" indicates only the upper triangle of A is
///                   referenced
///                   "L" or "l" indicates only the lower triangle of A is
///                   referenced
/// \param alpha [in] Input coefficient of A*x
/// \param A [in] Input matrix, as a 2-D Kokkos::View
/// \param x [in] Input vector, as a 1-D Kokkos::View
/// \param beta [in] Input coefficient of y
/// \param y [in/out] Output vector, as a nonconst 1-D Kokkos::View
template <class AViewType, class XViewType, class YViewType>
void symv(const char uplo[], typename YViewType::const_value_type& alpha, const AViewType& A, const XViewType& x,
          typename YViewType::const_value_type& beta, const YViewType& y) {
  symv(typename AViewType::execution_space{}, uplo, alpha, A, x, beta, y);
}

}  // namespace KokkosBlas

#endif  // KOKKOSBLAS2_SYMV_HPP_
