// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS3_SYMM_IMPL_HPP_
#define KOKKOSBLAS3_SYMM_IMPL_HPP_

/// \file KokkosBlas3_symm_impl.hpp
/// \brief Sequential fall-back implementation of the symmetric
/// matrix-matrix multiply. This is a simple, unoptimized triple loop meant
/// to be run on host mirrors of the input/output views (see
/// KokkosBlas3_symm_spec.hpp), analogous to the serial fall-backs used by
/// KokkosBlas::trsm, KokkosBlas::trmm and KokkosBlas::syrk.

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "KokkosKernels_ArithTraits.hpp"

namespace KokkosBlas {
namespace Impl {

template <class AViewType, class BViewType, class CViewType>
void SerialSymm_Invoke(const char side[], const char uplo[], typename CViewType::const_value_type& alpha,
                       const AViewType& A, const BViewType& B, typename CViewType::const_value_type& beta,
                       const CViewType& C) {
  using CScalar = typename CViewType::non_const_value_type;
  using ATV     = KokkosKernels::ArithTraits<CScalar>;

  const bool is_left  = (side[0] == 'L') || (side[0] == 'l');
  const bool is_upper = (uplo[0] == 'U') || (uplo[0] == 'u');

  const int M = static_cast<int>(C.extent(0));
  const int N = static_cast<int>(C.extent(1));

  // A is symmetric: only the uplo triangle is significant, the other is
  // inferred by symmetry.
  auto a_elem = [&](int p, int q) -> CScalar {
    return ((is_upper && p <= q) || (!is_upper && p >= q)) ? static_cast<CScalar>(A(p, q))
                                                            : static_cast<CScalar>(A(q, p));
  };

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      CScalar sum = ATV::zero();
      if (is_left) {
        // C = alpha*A*B + beta*C, A is M-by-M
        for (int k = 0; k < M; k++) sum += a_elem(i, k) * static_cast<CScalar>(B(k, j));
      } else {
        // C = alpha*B*A + beta*C, A is N-by-N
        for (int k = 0; k < N; k++) sum += static_cast<CScalar>(B(i, k)) * a_elem(k, j);
      }
      const CScalar c_val = (beta == ATV::zero()) ? ATV::zero() : static_cast<CScalar>(beta * C(i, j));
      C(i, j)              = c_val + static_cast<CScalar>(alpha * sum);
    }
  }
}

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSBLAS3_SYMM_IMPL_HPP_
