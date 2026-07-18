// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS2_SYMV_IMPL_HPP_
#define KOKKOSBLAS2_SYMV_IMPL_HPP_

/// \file KokkosBlas2_symv_impl.hpp
/// \brief Sequential fall-back implementation of the symmetric
/// matrix-vector multiply. This is a simple, unoptimized double loop meant
/// to be run on host mirrors of the input/output views (see
/// KokkosBlas2_symv_spec.hpp), analogous to the serial fall-backs used by
/// KokkosBlas::trsm, KokkosBlas::trmm, KokkosBlas::syrk and
/// KokkosBlas::symm.

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "KokkosKernels_ArithTraits.hpp"

namespace KokkosBlas {
namespace Impl {

template <class AViewType, class XViewType, class YViewType>
void SerialSymv_Invoke(const char uplo[], typename YViewType::const_value_type& alpha, const AViewType& A,
                       const XViewType& x, typename YViewType::const_value_type& beta, const YViewType& y) {
  using YScalar = typename YViewType::non_const_value_type;
  using ATV     = KokkosKernels::ArithTraits<YScalar>;

  const bool is_upper = (uplo[0] == 'U') || (uplo[0] == 'u');
  const int N         = static_cast<int>(A.extent(0));

  // A is symmetric: only the uplo triangle is significant, the other is
  // inferred by symmetry.
  auto a_elem = [&](int p, int q) -> YScalar {
    return ((is_upper && p <= q) || (!is_upper && p >= q)) ? static_cast<YScalar>(A(p, q))
                                                            : static_cast<YScalar>(A(q, p));
  };

  for (int i = 0; i < N; i++) {
    YScalar sum = ATV::zero();
    for (int k = 0; k < N; k++) sum += a_elem(i, k) * static_cast<YScalar>(x(k));
    const YScalar y_val = (beta == ATV::zero()) ? ATV::zero() : static_cast<YScalar>(beta * y(i));
    y(i)                 = y_val + static_cast<YScalar>(alpha * sum);
  }
}

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSBLAS2_SYMV_IMPL_HPP_
