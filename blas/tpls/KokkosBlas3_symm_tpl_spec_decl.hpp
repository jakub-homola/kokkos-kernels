// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS3_SYMM_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS3_SYMM_TPL_SPEC_DECL_HPP_

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_SYMM_CUBLAS(SCALAR_TYPE, CUDA_SCALAR_TYPE, CUBLAS_FN, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)          \
  template <class ExecSpace>                                                                                          \
  struct SYMM<ExecSpace,                                                                                              \
              Kokkos::View<const SCALAR_TYPE**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                        \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<const SCALAR_TYPE**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                        \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<SCALAR_TYPE**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              true, ETI_SPEC_AVAIL> {                                                                                 \
    typedef SCALAR_TYPE SCALAR;                                                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        AViewType;                                                                                                    \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        BViewType;                                                                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                                      \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        CViewType;                                                                                                    \
                                                                                                                       \
    static void symm(const ExecSpace& space, const char side[], const char uplo[],                                    \
                     typename CViewType::const_value_type& alpha, const AViewType& A, const BViewType& B,            \
                     typename CViewType::const_value_type& beta, const CViewType& C) {                               \
      Kokkos::Profiling::pushRegion("KokkosBlas::symm[TPL_CUBLAS," #SCALAR_TYPE "]");                                 \
      const int M = static_cast<int>(C.extent(0));                                                                    \
      const int N = static_cast<int>(C.extent(1));                                                                    \
                                                                                                                       \
      const int AST = A.stride(1), LDA = (AST == 0) ? 1 : AST;                                                        \
      const int BST = B.stride(1), LDB = (BST == 0) ? 1 : BST;                                                        \
      const int CST = C.stride(1), LDC = (CST == 0) ? 1 : CST;                                                        \
                                                                                                                       \
      cublasSideMode_t side_  = ((side[0] == 'L') || (side[0] == 'l')) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;        \
      cublasFillMode_t uplo_  = ((uplo[0] == 'L') || (uplo[0] == 'l')) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER; \
                                                                                                                       \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                      \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                               \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                               \
          CUBLAS_FN(s.handle, side_, uplo_, M, N, reinterpret_cast<const CUDA_SCALAR_TYPE*>(&alpha),                 \
                    reinterpret_cast<const CUDA_SCALAR_TYPE*>(A.data()), LDA,                                        \
                    reinterpret_cast<const CUDA_SCALAR_TYPE*>(B.data()), LDB,                                        \
                    reinterpret_cast<const CUDA_SCALAR_TYPE*>(&beta), reinterpret_cast<CUDA_SCALAR_TYPE*>(C.data()), \
                    LDC));                                                                                            \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                              \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS3_DSYMM_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_SYMM_CUBLAS(double, double, cublasDsymm, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_SSYMM_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_SYMM_CUBLAS(float, float, cublasSsymm, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_ZSYMM_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_SYMM_CUBLAS(Kokkos::complex<double>, cuDoubleComplex, cublasZsymm, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_CSYMM_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_SYMM_CUBLAS(Kokkos::complex<float>, cuComplex, cublasCsymm, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

KOKKOSBLAS3_DSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_DSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_DSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_DSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS3_SSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_SSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_SSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_SSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS3_ZSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_ZSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_ZSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_ZSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS3_CSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_CSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_CSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_CSYMM_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_XSYMM_ROCBLAS(SCALAR_TYPE, ROCBLAS_SCALAR_TYPE, ROCBLAS_FN, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)    \
  template <>                                                                                                         \
  struct SYMM<Kokkos::HIP,                                                                                            \
              Kokkos::View<const SCALAR_TYPE**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                       \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<const SCALAR_TYPE**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                       \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<SCALAR_TYPE**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                             \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              true, ETI_SPEC_AVAIL> {                                                                                 \
    typedef SCALAR_TYPE SCALAR;                                                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        AViewType;                                                                                                    \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        BViewType;                                                                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                                    \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        CViewType;                                                                                                    \
                                                                                                                       \
    static void symm(const typename CViewType::execution_space& space, const char side[], const char uplo[],        \
                     typename CViewType::const_value_type& alpha, const AViewType& A, const BViewType& B,            \
                     typename CViewType::const_value_type& beta, const CViewType& C) {                               \
      Kokkos::Profiling::pushRegion("KokkosBlas::symm[TPL_ROCBLAS," #SCALAR_TYPE "]");                                \
      const int M = static_cast<int>(C.extent(0));                                                                    \
      const int N = static_cast<int>(C.extent(1));                                                                    \
                                                                                                                       \
      const int AST = A.stride(1), LDA = (AST == 0) ? 1 : AST;                                                        \
      const int BST = B.stride(1), LDB = (BST == 0) ? 1 : BST;                                                        \
      const int CST = C.stride(1), LDC = (CST == 0) ? 1 : CST;                                                        \
                                                                                                                       \
      rocblas_side side_ = side_mode_kk_to_rocblas(side);                                                             \
      rocblas_fill uplo_ = ((uplo[0] == 'L') || (uplo[0] == 'l')) ? rocblas_fill_lower : rocblas_fill_upper;          \
                                                                                                                       \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                        \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));                            \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(                                                                              \
          ROCBLAS_FN(s.handle, side_, uplo_, M, N, reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(&alpha),             \
                     reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(A.data()), LDA,                                     \
                     reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(B.data()), LDB,                                     \
                     reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(&beta),                                             \
                     reinterpret_cast<ROCBLAS_SCALAR_TYPE*>(C.data()), LDC));                                         \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, NULL));                                          \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS3_DSYMM_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_XSYMM_ROCBLAS(double, double, rocblas_dsymm, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_SSYMM_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_XSYMM_ROCBLAS(float, float, rocblas_ssymm, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_ZSYMM_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                             \
  KOKKOSBLAS3_XSYMM_ROCBLAS(Kokkos::complex<double>, rocblas_double_complex, rocblas_zsymm, LAYOUT, MEM_SPACE,   \
                            ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_CSYMM_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                           \
  KOKKOSBLAS3_XSYMM_ROCBLAS(Kokkos::complex<float>, rocblas_float_complex, rocblas_csymm, LAYOUT, MEM_SPACE,   \
                            ETI_SPEC_AVAIL)

KOKKOSBLAS3_DSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_DSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_SSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_SSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_ZSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_ZSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_CSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_CSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_DSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_DSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS3_SSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_SSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS3_ZSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_ZSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS3_CSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_CSYMM_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

#endif  // KOKKOSBLAS3_SYMM_TPL_SPEC_DECL_HPP_
