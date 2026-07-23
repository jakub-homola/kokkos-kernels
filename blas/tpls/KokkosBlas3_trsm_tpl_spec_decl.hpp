// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS3_TRSM_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS3_TRSM_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_DTRSM_BLAS(LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL)                                            \
  template <class ExecSpace>                                                                                           \
  struct TRSM<                                                                                                         \
      ExecSpace,                                                                                                       \
      Kokkos::View<const double**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                          \
      Kokkos::View<double**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      true, ETI_SPEC_AVAIL> {                                                                                          \
    typedef double SCALAR;                                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                     \
        AViewType;                                                                                                     \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,                                      \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                     \
        BViewType;                                                                                                     \
                                                                                                                       \
    static void trsm(const ExecSpace& /*space*/, const char side[], const char uplo[], const char trans[],             \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,               \
                     const BViewType& B) {                                                                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS,double]");                                              \
      const int M = static_cast<int>(B.extent(0));                                                                     \
      const int N = static_cast<int>(B.extent(1));                                                                     \
                                                                                                                       \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                                 \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                                 \
                                                                                                                       \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                                 \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                                 \
                                                                                                                       \
      char side_;                                                                                                      \
      char uplo_;                                                                                                      \
                                                                                                                       \
      if (A_is_ll) {                                                                                                   \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                      \
          side_ = 'L';                                                                                                 \
        else                                                                                                           \
          side_ = 'R';                                                                                                 \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                      \
          uplo_ = 'L';                                                                                                 \
        else                                                                                                           \
          uplo_ = 'U';                                                                                                 \
      } else {                                                                                                         \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                      \
          side_ = 'R';                                                                                                 \
        else                                                                                                           \
          side_ = 'L';                                                                                                 \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                      \
          uplo_ = 'U';                                                                                                 \
        else                                                                                                           \
          uplo_ = 'L';                                                                                                 \
      }                                                                                                                \
                                                                                                                       \
      if (A_is_ll)                                                                                                     \
        HostBlas<double>::trsm(side_, uplo_, trans[0], diag[0], M, N, alpha, A.data(), LDA, B.data(), LDB);            \
      else                                                                                                             \
        HostBlas<double>::trsm(side_, uplo_, trans[0], diag[0], N, M, alpha, A.data(), LDA, B.data(), LDB);            \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define KOKKOSBLAS3_STRSM_BLAS(LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL)                                           \
  template <class ExecSpace>                                                                                          \
  struct TRSM<                                                                                                        \
      ExecSpace,                                                                                                      \
      Kokkos::View<const float**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                         \
      Kokkos::View<float**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      true, ETI_SPEC_AVAIL> {                                                                                         \
    typedef float SCALAR;                                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                               \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        AViewType;                                                                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,                                     \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        BViewType;                                                                                                    \
                                                                                                                      \
    static void trsm(const ExecSpace& /*space*/, const char side[], const char uplo[], const char trans[],            \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,              \
                     const BViewType& B) {                                                                            \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS,float]");                                              \
      const int M = static_cast<int>(B.extent(0));                                                                    \
      const int N = static_cast<int>(B.extent(1));                                                                    \
                                                                                                                      \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                                \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                                \
                                                                                                                      \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                                \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                                \
                                                                                                                      \
      char side_;                                                                                                     \
      char uplo_;                                                                                                     \
                                                                                                                      \
      if (A_is_ll) {                                                                                                  \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                     \
          side_ = 'L';                                                                                                \
        else                                                                                                          \
          side_ = 'R';                                                                                                \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                     \
          uplo_ = 'L';                                                                                                \
        else                                                                                                          \
          uplo_ = 'U';                                                                                                \
      } else {                                                                                                        \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                     \
          side_ = 'R';                                                                                                \
        else                                                                                                          \
          side_ = 'L';                                                                                                \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                     \
          uplo_ = 'U';                                                                                                \
        else                                                                                                          \
          uplo_ = 'L';                                                                                                \
      }                                                                                                               \
                                                                                                                      \
      if (A_is_ll)                                                                                                    \
        HostBlas<float>::trsm(side_, uplo_, trans[0], diag[0], M, N, alpha, A.data(), LDA, B.data(), LDB);            \
      else                                                                                                            \
        HostBlas<float>::trsm(side_, uplo_, trans[0], diag[0], N, M, alpha, A.data(), LDA, B.data(), LDB);            \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS3_ZTRSM_BLAS(LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL)                                 \
  template <class ExecSpace>                                                                                \
  struct TRSM<ExecSpace,                                                                                    \
              Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,  \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                       \
              Kokkos::View<Kokkos::complex<double>**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,        \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                       \
              true, ETI_SPEC_AVAIL> {                                                                       \
    typedef Kokkos::complex<double> SCALAR;                                                                 \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                     \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                          \
        AViewType;                                                                                          \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,                           \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                          \
        BViewType;                                                                                          \
                                                                                                            \
    static void trsm(const ExecSpace& /*space*/, const char side[], const char uplo[], const char trans[],  \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,    \
                     const BViewType& B) {                                                                  \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS,complex<double>]");                          \
      const int M = static_cast<int>(B.extent(0));                                                          \
      const int N = static_cast<int>(B.extent(1));                                                          \
                                                                                                            \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                      \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                      \
                                                                                                            \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                      \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                      \
                                                                                                            \
      char side_;                                                                                           \
      char uplo_;                                                                                           \
                                                                                                            \
      if (A_is_ll) {                                                                                        \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                           \
          side_ = 'L';                                                                                      \
        else                                                                                                \
          side_ = 'R';                                                                                      \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                           \
          uplo_ = 'L';                                                                                      \
        else                                                                                                \
          uplo_ = 'U';                                                                                      \
      } else {                                                                                              \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                           \
          side_ = 'R';                                                                                      \
        else                                                                                                \
          side_ = 'L';                                                                                      \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                           \
          uplo_ = 'U';                                                                                      \
        else                                                                                                \
          uplo_ = 'L';                                                                                      \
      }                                                                                                     \
                                                                                                            \
      const std::complex<double> alpha_val = alpha;                                                         \
      if (A_is_ll)                                                                                          \
        HostBlas<std::complex<double> >::trsm(side_, uplo_, trans[0], diag[0], M, N, alpha_val,             \
                                              reinterpret_cast<const std::complex<double>*>(A.data()), LDA, \
                                              reinterpret_cast<std::complex<double>*>(B.data()), LDB);      \
      else                                                                                                  \
        HostBlas<std::complex<double> >::trsm(side_, uplo_, trans[0], diag[0], N, M, alpha_val,             \
                                              reinterpret_cast<const std::complex<double>*>(A.data()), LDA, \
                                              reinterpret_cast<std::complex<double>*>(B.data()), LDB);      \
      Kokkos::Profiling::popRegion();                                                                       \
    }                                                                                                       \
  };

#define KOKKOSBLAS3_CTRSM_BLAS(LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL)                                \
  template <class ExecSpace>                                                                               \
  struct TRSM<ExecSpace,                                                                                   \
              Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,  \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                      \
              Kokkos::View<Kokkos::complex<float>**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,        \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                      \
              true, ETI_SPEC_AVAIL> {                                                                      \
    typedef Kokkos::complex<float> SCALAR;                                                                 \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                    \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                         \
        AViewType;                                                                                         \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,                          \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                         \
        BViewType;                                                                                         \
                                                                                                           \
    static void trsm(const ExecSpace& /*space*/, const char side[], const char uplo[], const char trans[], \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,   \
                     const BViewType& B) {                                                                 \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS,complex<float>]");                          \
      const int M = static_cast<int>(B.extent(0));                                                         \
      const int N = static_cast<int>(B.extent(1));                                                         \
                                                                                                           \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                     \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                     \
                                                                                                           \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                     \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                     \
                                                                                                           \
      char side_;                                                                                          \
      char uplo_;                                                                                          \
                                                                                                           \
      if (A_is_ll) {                                                                                       \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                          \
          side_ = 'L';                                                                                     \
        else                                                                                               \
          side_ = 'R';                                                                                     \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                          \
          uplo_ = 'L';                                                                                     \
        else                                                                                               \
          uplo_ = 'U';                                                                                     \
      } else {                                                                                             \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                          \
          side_ = 'R';                                                                                     \
        else                                                                                               \
          side_ = 'L';                                                                                     \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                          \
          uplo_ = 'U';                                                                                     \
        else                                                                                               \
          uplo_ = 'L';                                                                                     \
      }                                                                                                    \
                                                                                                           \
      const std::complex<float> alpha_val = alpha;                                                         \
      if (A_is_ll)                                                                                         \
        HostBlas<std::complex<float> >::trsm(side_, uplo_, trans[0], diag[0], M, N, alpha_val,             \
                                             reinterpret_cast<const std::complex<float>*>(A.data()), LDA,  \
                                             reinterpret_cast<std::complex<float>*>(B.data()), LDB);       \
      else                                                                                                 \
        HostBlas<std::complex<float> >::trsm(side_, uplo_, trans[0], diag[0], N, M, alpha_val,             \
                                             reinterpret_cast<const std::complex<float>*>(A.data()), LDA,  \
                                             reinterpret_cast<std::complex<float>*>(B.data()), LDB);       \
      Kokkos::Profiling::popRegion();                                                                      \
    }                                                                                                      \
  };

KOKKOSBLAS3_DTRSM_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS3_DTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS3_STRSM_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS3_STRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS3_ZTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS3_CTRSM_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS3_CTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_BLAS

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_DTRSM_CUBLAS(LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL)                                          \
  template <class ExecSpace>                                                                                           \
  struct TRSM<                                                                                                         \
      ExecSpace,                                                                                                       \
      Kokkos::View<const double**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                          \
      Kokkos::View<double**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      true, ETI_SPEC_AVAIL> {                                                                                          \
    typedef double SCALAR;                                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                     \
        AViewType;                                                                                                     \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,                                      \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                     \
        BViewType;                                                                                                     \
                                                                                                                       \
    static void trsm(const ExecSpace& space, const char side[], const char uplo[], const char trans[],                 \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,               \
                     const BViewType& B) {                                                                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_CUBLAS,double]");                                            \
      const int M = static_cast<int>(B.extent(0));                                                                     \
      const int N = static_cast<int>(B.extent(1));                                                                     \
                                                                                                                       \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                                 \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                                 \
                                                                                                                       \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                                 \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                                 \
                                                                                                                       \
      cublasSideMode_t side_;                                                                                          \
      cublasFillMode_t uplo_;                                                                                          \
      cublasOperation_t trans_;                                                                                        \
      cublasDiagType_t diag_;                                                                                          \
                                                                                                                       \
      if (A_is_ll) {                                                                                                   \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                      \
          side_ = CUBLAS_SIDE_LEFT;                                                                                    \
        else                                                                                                           \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                   \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                      \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                              \
        else                                                                                                           \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                              \
      } else {                                                                                                         \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                      \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                   \
        else                                                                                                           \
          side_ = CUBLAS_SIDE_LEFT;                                                                                    \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                      \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                              \
        else                                                                                                           \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                              \
      }                                                                                                                \
                                                                                                                       \
      if ((trans[0] == 'N') || (trans[0] == 'n'))                                                                      \
        trans_ = CUBLAS_OP_N;                                                                                          \
      else if ((trans[0] == 'T') || (trans[0] == 't'))                                                                 \
        trans_ = CUBLAS_OP_T;                                                                                          \
      else                                                                                                             \
        trans_ = CUBLAS_OP_C;                                                                                          \
      if ((diag[0] == 'U') || (diag[0] == 'u'))                                                                        \
        diag_ = CUBLAS_DIAG_UNIT;                                                                                      \
      else                                                                                                             \
        diag_ = CUBLAS_DIAG_NON_UNIT;                                                                                  \
                                                                                                                       \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                       \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                                \
      if (A_is_ll) {                                                                                                   \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                              \
            cublasDtrsm(s.handle, side_, uplo_, trans_, diag_, M, N, &alpha, A.data(), LDA, B.data(), LDB));           \
      } else {                                                                                                         \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                              \
            cublasDtrsm(s.handle, side_, uplo_, trans_, diag_, N, M, &alpha, A.data(), LDA, B.data(), LDB));           \
      }                                                                                                                \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                               \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define KOKKOSBLAS3_STRSM_CUBLAS(LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL)                                         \
  template <class ExecSpace>                                                                                          \
  struct TRSM<                                                                                                        \
      ExecSpace,                                                                                                      \
      Kokkos::View<const float**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                         \
      Kokkos::View<float**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
      true, ETI_SPEC_AVAIL> {                                                                                         \
    typedef float SCALAR;                                                                                             \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                               \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        AViewType;                                                                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,                                     \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        BViewType;                                                                                                    \
                                                                                                                      \
    static void trsm(const ExecSpace& space, const char side[], const char uplo[], const char trans[],                \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,              \
                     const BViewType& B) {                                                                            \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_CUBLAS,float]");                                            \
      const int M = static_cast<int>(B.extent(0));                                                                    \
      const int N = static_cast<int>(B.extent(1));                                                                    \
                                                                                                                      \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                                \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                                \
                                                                                                                      \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                                \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                                \
                                                                                                                      \
      cublasSideMode_t side_;                                                                                         \
      cublasFillMode_t uplo_;                                                                                         \
      cublasOperation_t trans_;                                                                                       \
      cublasDiagType_t diag_;                                                                                         \
                                                                                                                      \
      if (A_is_ll) {                                                                                                  \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                     \
          side_ = CUBLAS_SIDE_LEFT;                                                                                   \
        else                                                                                                          \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                  \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                     \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                             \
        else                                                                                                          \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                             \
      } else {                                                                                                        \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                                     \
          side_ = CUBLAS_SIDE_RIGHT;                                                                                  \
        else                                                                                                          \
          side_ = CUBLAS_SIDE_LEFT;                                                                                   \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                                     \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                             \
        else                                                                                                          \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                             \
      }                                                                                                               \
                                                                                                                      \
      if ((trans[0] == 'N') || (trans[0] == 'n'))                                                                     \
        trans_ = CUBLAS_OP_N;                                                                                         \
      else if ((trans[0] == 'T') || (trans[0] == 't'))                                                                \
        trans_ = CUBLAS_OP_T;                                                                                         \
      else                                                                                                            \
        trans_ = CUBLAS_OP_C;                                                                                         \
      if ((diag[0] == 'U') || (diag[0] == 'u'))                                                                       \
        diag_ = CUBLAS_DIAG_UNIT;                                                                                     \
      else                                                                                                            \
        diag_ = CUBLAS_DIAG_NON_UNIT;                                                                                 \
                                                                                                                      \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                      \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                               \
      if (A_is_ll) {                                                                                                  \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                             \
            cublasStrsm(s.handle, side_, uplo_, trans_, diag_, M, N, &alpha, A.data(), LDA, B.data(), LDB));          \
      } else {                                                                                                        \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                             \
            cublasStrsm(s.handle, side_, uplo_, trans_, diag_, N, M, &alpha, A.data(), LDA, B.data(), LDB));          \
      }                                                                                                               \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                              \
                                                                                                                      \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS3_ZTRSM_CUBLAS(LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL)                                 \
  template <class ExecSpace>                                                                                  \
  struct TRSM<ExecSpace,                                                                                      \
              Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,    \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                         \
              Kokkos::View<Kokkos::complex<double>**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,          \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                         \
              true, ETI_SPEC_AVAIL> {                                                                         \
    typedef Kokkos::complex<double> SCALAR;                                                                   \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                       \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                            \
        AViewType;                                                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,                             \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                            \
        BViewType;                                                                                            \
                                                                                                              \
    static void trsm(const ExecSpace& space, const char side[], const char uplo[], const char trans[],        \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,      \
                     const BViewType& B) {                                                                    \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_CUBLAS,complex<double>]");                          \
      const int M = static_cast<int>(B.extent(0));                                                            \
      const int N = static_cast<int>(B.extent(1));                                                            \
                                                                                                              \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                        \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                        \
                                                                                                              \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                        \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                        \
                                                                                                              \
      cublasSideMode_t side_;                                                                                 \
      cublasFillMode_t uplo_;                                                                                 \
      cublasOperation_t trans_;                                                                               \
      cublasDiagType_t diag_;                                                                                 \
                                                                                                              \
      if (A_is_ll) {                                                                                          \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                             \
          side_ = CUBLAS_SIDE_LEFT;                                                                           \
        else                                                                                                  \
          side_ = CUBLAS_SIDE_RIGHT;                                                                          \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                             \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                     \
        else                                                                                                  \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                     \
      } else {                                                                                                \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                             \
          side_ = CUBLAS_SIDE_RIGHT;                                                                          \
        else                                                                                                  \
          side_ = CUBLAS_SIDE_LEFT;                                                                           \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                             \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                     \
        else                                                                                                  \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                     \
      }                                                                                                       \
                                                                                                              \
      if ((trans[0] == 'N') || (trans[0] == 'n'))                                                             \
        trans_ = CUBLAS_OP_N;                                                                                 \
      else if ((trans[0] == 'T') || (trans[0] == 't'))                                                        \
        trans_ = CUBLAS_OP_T;                                                                                 \
      else                                                                                                    \
        trans_ = CUBLAS_OP_C;                                                                                 \
      if ((diag[0] == 'U') || (diag[0] == 'u'))                                                               \
        diag_ = CUBLAS_DIAG_UNIT;                                                                             \
      else                                                                                                    \
        diag_ = CUBLAS_DIAG_NON_UNIT;                                                                         \
                                                                                                              \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();              \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                       \
      if (A_is_ll) {                                                                                          \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasZtrsm(s.handle, side_, uplo_, trans_, diag_, M, N,             \
                                                     reinterpret_cast<const cuDoubleComplex*>(&alpha),        \
                                                     reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA, \
                                                     reinterpret_cast<cuDoubleComplex*>(B.data()), LDB));     \
      } else {                                                                                                \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasZtrsm(s.handle, side_, uplo_, trans_, diag_, N, M,             \
                                                     reinterpret_cast<const cuDoubleComplex*>(&alpha),        \
                                                     reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA, \
                                                     reinterpret_cast<cuDoubleComplex*>(B.data()), LDB));     \
      }                                                                                                       \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                      \
                                                                                                              \
      Kokkos::Profiling::popRegion();                                                                         \
    }                                                                                                         \
  };

#define KOKKOSBLAS3_CTRSM_CUBLAS(LAYOUTA, LAYOUTB, MEM_SPACE, ETI_SPEC_AVAIL)                                 \
  template <class ExecSpace>                                                                                  \
  struct TRSM<ExecSpace,                                                                                      \
              Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                         \
              Kokkos::View<Kokkos::complex<float>**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,           \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                         \
              true, ETI_SPEC_AVAIL> {                                                                         \
    typedef Kokkos::complex<float> SCALAR;                                                                    \
    typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>,                       \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                            \
        AViewType;                                                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEM_SPACE>,                             \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                            \
        BViewType;                                                                                            \
                                                                                                              \
    static void trsm(const ExecSpace& space, const char side[], const char uplo[], const char trans[],        \
                     const char diag[], typename BViewType::const_value_type& alpha, const AViewType& A,      \
                     const BViewType& B) {                                                                    \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_CUBLAS,complex<float>]");                           \
      const int M = static_cast<int>(B.extent(0));                                                            \
      const int N = static_cast<int>(B.extent(1));                                                            \
                                                                                                              \
      bool A_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;                                        \
      bool B_is_ll = std::is_same<Kokkos::LayoutLeft, LAYOUTB>::value;                                        \
                                                                                                              \
      const int AST = A_is_ll ? A.stride(1) : A.stride(0), LDA = (AST == 0) ? 1 : AST;                        \
      const int BST = B_is_ll ? B.stride(1) : B.stride(0), LDB = (BST == 0) ? 1 : BST;                        \
                                                                                                              \
      cublasSideMode_t side_;                                                                                 \
      cublasFillMode_t uplo_;                                                                                 \
      cublasOperation_t trans_;                                                                               \
      cublasDiagType_t diag_;                                                                                 \
                                                                                                              \
      if (A_is_ll) {                                                                                          \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                             \
          side_ = CUBLAS_SIDE_LEFT;                                                                           \
        else                                                                                                  \
          side_ = CUBLAS_SIDE_RIGHT;                                                                          \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                             \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                     \
        else                                                                                                  \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                     \
      } else {                                                                                                \
        if ((side[0] == 'L') || (side[0] == 'l'))                                                             \
          side_ = CUBLAS_SIDE_RIGHT;                                                                          \
        else                                                                                                  \
          side_ = CUBLAS_SIDE_LEFT;                                                                           \
        if ((uplo[0] == 'L') || (uplo[0] == 'l'))                                                             \
          uplo_ = CUBLAS_FILL_MODE_UPPER;                                                                     \
        else                                                                                                  \
          uplo_ = CUBLAS_FILL_MODE_LOWER;                                                                     \
      }                                                                                                       \
                                                                                                              \
      if ((trans[0] == 'N') || (trans[0] == 'n'))                                                             \
        trans_ = CUBLAS_OP_N;                                                                                 \
      else if ((trans[0] == 'T') || (trans[0] == 't'))                                                        \
        trans_ = CUBLAS_OP_T;                                                                                 \
      else                                                                                                    \
        trans_ = CUBLAS_OP_C;                                                                                 \
      if ((diag[0] == 'U') || (diag[0] == 'u'))                                                               \
        diag_ = CUBLAS_DIAG_UNIT;                                                                             \
      else                                                                                                    \
        diag_ = CUBLAS_DIAG_NON_UNIT;                                                                         \
                                                                                                              \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();              \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                       \
      if (A_is_ll) {                                                                                          \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasCtrsm(                                                         \
            s.handle, side_, uplo_, trans_, diag_, M, N, reinterpret_cast<const cuComplex*>(&alpha),          \
            reinterpret_cast<const cuComplex*>(A.data()), LDA, reinterpret_cast<cuComplex*>(B.data()), LDB)); \
      } else {                                                                                                \
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasCtrsm(                                                         \
            s.handle, side_, uplo_, trans_, diag_, N, M, reinterpret_cast<const cuComplex*>(&alpha),          \
            reinterpret_cast<const cuComplex*>(A.data()), LDA, reinterpret_cast<cuComplex*>(B.data()), LDB)); \
      }                                                                                                       \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                      \
                                                                                                              \
      Kokkos::Profiling::popRegion();                                                                         \
    }                                                                                                         \
  };

KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)
KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_DTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)
KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_STRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)
KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_ZTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)
KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_CTRSM_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_XTRSM_ROCBLAS(SCALAR_TYPE, ROCBLAS_SCALAR_TYPE, ROCBLAS_FN, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)    \
  template <>                                                                                                         \
  struct TRSM<Kokkos::HIP,                                                                                            \
              Kokkos::View<const SCALAR_TYPE**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                       \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<SCALAR_TYPE**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                             \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              true, ETI_SPEC_AVAIL> {                                                                                 \
    typedef SCALAR_TYPE SCALAR;                                                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        AViewType;                                                                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                                    \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        BViewType;                                                                                                    \
                                                                                                                      \
    static void trsm(const typename BViewType::execution_space& space, const char side[], const char uplo[],          \
                     const char trans[], const char diag[], typename BViewType::const_value_type& alpha,              \
                     const AViewType& A, const BViewType& B) {                                                        \
      Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_ROCBLAS," #SCALAR_TYPE "]");                                \
      const int M = static_cast<int>(B.extent(0));                                                                    \
      const int N = static_cast<int>(B.extent(1));                                                                    \
                                                                                                                      \
      const int AST = A.stride(1), LDA = (AST == 0) ? 1 : AST;                                                        \
      const int BST = B.stride(1), LDB = (BST == 0) ? 1 : BST;                                                        \
                                                                                                                      \
      rocblas_side side_       = side_mode_kk_to_rocblas(side);                                                       \
      rocblas_fill uplo_       = ((uplo[0] == 'L') || (uplo[0] == 'l')) ? rocblas_fill_lower : rocblas_fill_upper;    \
      rocblas_operation trans_ = trans_mode_kk_to_rocblas(trans);                                                     \
      rocblas_diagonal diag_ =                                                                                        \
          ((diag[0] == 'U') || (diag[0] == 'u')) ? rocblas_diagonal_unit : rocblas_diagonal_non_unit;                 \
                                                                                                                      \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                        \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));                            \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(ROCBLAS_FN(s.handle, side_, uplo_, trans_, diag_, M, N,                       \
                                                   reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(&alpha),              \
                                                   reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(A.data()), LDA,       \
                                                   reinterpret_cast<ROCBLAS_SCALAR_TYPE*>(B.data()), LDB));           \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, NULL));                                          \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS3_DTRSM_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_XTRSM_ROCBLAS(double, double, rocblas_dtrsm, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_STRSM_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_XTRSM_ROCBLAS(float, float, rocblas_strsm, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_ZTRSM_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                             \
  KOKKOSBLAS3_XTRSM_ROCBLAS(Kokkos::complex<double>, rocblas_double_complex, rocblas_ztrsm, LAYOUT, MEM_SPACE,   \
                            ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_CTRSM_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                           \
  KOKKOSBLAS3_XTRSM_ROCBLAS(Kokkos::complex<float>, rocblas_float_complex, rocblas_ctrsm, LAYOUT, MEM_SPACE,   \
                            ETI_SPEC_AVAIL)

KOKKOSBLAS3_DTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_DTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_STRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_STRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_ZTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_ZTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_CTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_CTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_DTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_DTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS3_STRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_STRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS3_ZTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_ZTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS3_CTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_CTRSM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

// oneMKL
#if defined(KOKKOSKERNELS_ENABLE_TPL_MKL) && defined(KOKKOS_ENABLE_SYCL)
#include <mkl.h>
#include <oneapi/mkl/blas.hpp>
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_TRSM_ONEMKL(SCALAR, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                            \
  template <class ExecSpace>                                                                                          \
  struct TRSM<ExecSpace,                                                                                              \
              Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Experimental::SYCL, MEM_SPACE>,             \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Experimental::SYCL, MEM_SPACE>,                   \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              true, ETI_SPEC_AVAIL> {                                                                                 \
    using device_type = Kokkos::Device<ExecSpace, MEM_SPACE>;                                                         \
    using mem_traits  = Kokkos::MemoryTraits<Kokkos::Unmanaged>;                                                      \
    using AViewType   = Kokkos::View<const SCALAR**, LAYOUT, device_type, mem_traits>;                                \
    using BViewType   = Kokkos::View<SCALAR**, LAYOUT, device_type, mem_traits>;                                      \
                                                                                                                       \
    static void trsm(const ExecSpace& exec, const char kk_side[], const char kk_uplo[], const char kk_trans[],        \
                     const char kk_diag[], typename BViewType::const_value_type& alpha, const AViewType& A,           \
                     const BViewType& B) {                                                                            \
      std::string label = "KokkosBlas::trsm[TPL_ONEMKL," + KokkosKernels::ArithTraits<SCALAR>::name() + "]";          \
      Kokkos::Profiling::pushRegion(label);                                                                           \
                                                                                                                       \
      const std::int64_t M = B.extent(0);                                                                             \
      const std::int64_t N = B.extent(1);                                                                             \
                                                                                                                       \
      const std::int64_t AST = A.stride(1), LDA = (AST == 0) ? 1 : AST;                                               \
      const std::int64_t BST = B.stride(1), LDB = (BST == 0) ? 1 : BST;                                               \
                                                                                                                       \
      oneapi::mkl::side side       = side_mode_kk_to_onemkl(kk_side);                                                  \
      oneapi::mkl::uplo uplo       = uplo_kk_to_onemkl(kk_uplo[0]);                                                    \
      oneapi::mkl::transpose trans = mode_kk_to_onemkl(kk_trans[0]);                                                   \
      oneapi::mkl::diag diag       = diag_kk_to_onemkl(kk_diag[0]);                                                    \
                                                                                                                       \
      using mag_type    = kokkos_to_std_type_map<SCALAR, KokkosKernels::ArithTraits<SCALAR>::is_complex>::type;       \
      const mag_type* a = reinterpret_cast<const mag_type*>(A.data());                                                \
      mag_type* b        = reinterpret_cast<mag_type*>(B.data());                                                     \
                                                                                                                       \
      oneapi::mkl::blas::column_major::trsm(exec.sycl_queue(), side, uplo, trans, diag, M, N, alpha, a, LDA, b, LDB); \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

KOKKOSBLAS3_TRSM_ONEMKL(double, Kokkos::LayoutLeft, Kokkos::SYCLDeviceUSMSpace, true)
KOKKOSBLAS3_TRSM_ONEMKL(double, Kokkos::LayoutLeft, Kokkos::SYCLDeviceUSMSpace, false)
KOKKOSBLAS3_TRSM_ONEMKL(float, Kokkos::LayoutLeft, Kokkos::SYCLDeviceUSMSpace, true)
KOKKOSBLAS3_TRSM_ONEMKL(float, Kokkos::LayoutLeft, Kokkos::SYCLDeviceUSMSpace, false)
KOKKOSBLAS3_TRSM_ONEMKL(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::SYCLDeviceUSMSpace, true)
KOKKOSBLAS3_TRSM_ONEMKL(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::SYCLDeviceUSMSpace, false)
KOKKOSBLAS3_TRSM_ONEMKL(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::SYCLDeviceUSMSpace, true)
KOKKOSBLAS3_TRSM_ONEMKL(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::SYCLDeviceUSMSpace, false)

KOKKOSBLAS3_TRSM_ONEMKL(double, Kokkos::LayoutLeft, Kokkos::SYCLSharedUSMSpace, true)
KOKKOSBLAS3_TRSM_ONEMKL(double, Kokkos::LayoutLeft, Kokkos::SYCLSharedUSMSpace, false)
KOKKOSBLAS3_TRSM_ONEMKL(float, Kokkos::LayoutLeft, Kokkos::SYCLSharedUSMSpace, true)
KOKKOSBLAS3_TRSM_ONEMKL(float, Kokkos::LayoutLeft, Kokkos::SYCLSharedUSMSpace, false)
KOKKOSBLAS3_TRSM_ONEMKL(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::SYCLSharedUSMSpace, true)
KOKKOSBLAS3_TRSM_ONEMKL(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::SYCLSharedUSMSpace, false)
KOKKOSBLAS3_TRSM_ONEMKL(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::SYCLSharedUSMSpace, true)
KOKKOSBLAS3_TRSM_ONEMKL(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::SYCLSharedUSMSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_MKL && KOKKOS_ENABLE_SYCL

#endif
