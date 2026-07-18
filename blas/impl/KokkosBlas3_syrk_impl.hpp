// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS3_SYRK_IMPL_HPP_
#define KOKKOSBLAS3_SYRK_IMPL_HPP_

/// \file KokkosBlas3_syrk_impl.hpp
/// \brief Sequential fall-back implementation of the symmetric rank-k
/// update. This is a simple, unoptimized triangle-only triple loop meant
/// to be run on host mirrors of the input/output views (see
/// KokkosBlas3_syrk_spec.hpp), analogous to the serial fall-backs used by
/// KokkosBlas::trsm and KokkosBlas::trmm.

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "KokkosKernels_ArithTraits.hpp"

namespace KokkosBlas {
namespace Impl {

template <class AViewType, class CViewType>
void SerialSyrk_Invoke(const char uplo[], const char trans[], typename CViewType::const_value_type& alpha,
                       const AViewType& A, typename CViewType::const_value_type& beta, const CViewType& C) {
  using CScalar = typename CViewType::non_const_value_type;
  using ATV     = KokkosKernels::ArithTraits<CScalar>;

  const bool is_upper  = (uplo[0] == 'U') || (uplo[0] == 'u');
  const bool not_trans = (trans[0] == 'N') || (trans[0] == 'n');

  const int n = static_cast<int>(C.extent(0));
  const int k = not_trans ? static_cast<int>(A.extent(1)) : static_cast<int>(A.extent(0));

  for (int i = 0; i < n; i++) {
    const int j_begin = is_upper ? i : 0;
    const int j_end    = is_upper ? n : i + 1;
    for (int j = j_begin; j < j_end; j++) {
      CScalar sum = ATV::zero();
      for (int p = 0; p < k; p++) {
        const CScalar a_ip = not_trans ? A(i, p) : A(p, i);
        const CScalar a_jp = not_trans ? A(j, p) : A(p, j);
        sum += a_ip * a_jp;
      }
      const CScalar c_val = (beta == ATV::zero()) ? ATV::zero() : static_cast<CScalar>(beta * C(i, j));
      C(i, j)              = c_val + static_cast<CScalar>(alpha * sum);
    }
  }
}

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSBLAS3_SYRK_IMPL_HPP_
