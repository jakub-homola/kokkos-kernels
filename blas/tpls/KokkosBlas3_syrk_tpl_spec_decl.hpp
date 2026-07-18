// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS3_SYRK_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS3_SYRK_TPL_SPEC_DECL_HPP_

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_SYRK_CUBLAS(SCALAR_TYPE, CUDA_SCALAR_TYPE, CUBLAS_FN, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)          \
  template <class ExecSpace>                                                                                          \
  struct SYRK<ExecSpace,                                                                                              \
              Kokkos::View<const SCALAR_TYPE**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                        \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<SCALAR_TYPE**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              true, ETI_SPEC_AVAIL> {                                                                                 \
    typedef SCALAR_TYPE SCALAR;                                                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        AViewType;                                                                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                                      \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        CViewType;                                                                                                    \
                                                                                                                       \
    static void syrk(const ExecSpace& space, const char uplo[], const char trans[],                                   \
                     typename CViewType::const_value_type& alpha, const AViewType& A,                                \
                     typename CViewType::const_value_type& beta, const CViewType& C) {                               \
      Kokkos::Profiling::pushRegion("KokkosBlas::syrk[TPL_CUBLAS," #SCALAR_TYPE "]");                                 \
      const bool not_trans = (trans[0] == 'N') || (trans[0] == 'n');                                                  \
      const int N          = static_cast<int>(C.extent(0));                                                          \
      const int K          = static_cast<int>(not_trans ? A.extent(1) : A.extent(0));                                \
                                                                                                                       \
      const int AST = A.stride(1), LDA = (AST == 0) ? 1 : AST;                                                        \
      const int CST = C.stride(1), LDC = (CST == 0) ? 1 : CST;                                                        \
                                                                                                                       \
      cublasFillMode_t uplo_  = ((uplo[0] == 'L') || (uplo[0] == 'l')) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER; \
      cublasOperation_t trans_ = not_trans ? CUBLAS_OP_N : CUBLAS_OP_T;                                                \
                                                                                                                       \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                      \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                               \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                               \
          CUBLAS_FN(s.handle, uplo_, trans_, N, K, reinterpret_cast<const CUDA_SCALAR_TYPE*>(&alpha),                \
                    reinterpret_cast<const CUDA_SCALAR_TYPE*>(A.data()), LDA,                                        \
                    reinterpret_cast<const CUDA_SCALAR_TYPE*>(&beta), reinterpret_cast<CUDA_SCALAR_TYPE*>(C.data()), \
                    LDC));                                                                                            \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                              \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS3_DSYRK_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_SYRK_CUBLAS(double, double, cublasDsyrk, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_SSYRK_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_SYRK_CUBLAS(float, float, cublasSsyrk, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_ZSYRK_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_SYRK_CUBLAS(Kokkos::complex<double>, cuDoubleComplex, cublasZsyrk, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_CSYRK_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_SYRK_CUBLAS(Kokkos::complex<float>, cuComplex, cublasCsyrk, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

KOKKOSBLAS3_DSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_DSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_DSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_DSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS3_SSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_SSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_SSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_SSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS3_ZSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_ZSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_ZSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_ZSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS3_CSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS3_CSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS3_CSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS3_CSYRK_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS3_XSYRK_ROCBLAS(SCALAR_TYPE, ROCBLAS_SCALAR_TYPE, ROCBLAS_FN, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)    \
  template <>                                                                                                         \
  struct SYRK<Kokkos::HIP,                                                                                            \
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
        CViewType;                                                                                                    \
                                                                                                                       \
    static void syrk(const typename CViewType::execution_space& space, const char uplo[], const char trans[],        \
                     typename CViewType::const_value_type& alpha, const AViewType& A,                                \
                     typename CViewType::const_value_type& beta, const CViewType& C) {                               \
      Kokkos::Profiling::pushRegion("KokkosBlas::syrk[TPL_ROCBLAS," #SCALAR_TYPE "]");                                \
      const bool not_trans = (trans[0] == 'N') || (trans[0] == 'n');                                                  \
      const int N          = static_cast<int>(C.extent(0));                                                          \
      const int K          = static_cast<int>(not_trans ? A.extent(1) : A.extent(0));                                \
                                                                                                                       \
      const int AST = A.stride(1), LDA = (AST == 0) ? 1 : AST;                                                        \
      const int CST = C.stride(1), LDC = (CST == 0) ? 1 : CST;                                                        \
                                                                                                                       \
      rocblas_fill uplo_       = ((uplo[0] == 'L') || (uplo[0] == 'l')) ? rocblas_fill_lower : rocblas_fill_upper;    \
      rocblas_operation trans_ = trans_mode_kk_to_rocblas(trans);                                                     \
                                                                                                                       \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                        \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));                            \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(                                                                              \
          ROCBLAS_FN(s.handle, uplo_, trans_, N, K, reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(&alpha),            \
                     reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(A.data()), LDA,                                     \
                     reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(&beta),                                             \
                     reinterpret_cast<ROCBLAS_SCALAR_TYPE*>(C.data()), LDC));                                         \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, NULL));                                          \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS3_DSYRK_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_XSYRK_ROCBLAS(double, double, rocblas_dsyrk, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_SSYRK_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS3_XSYRK_ROCBLAS(float, float, rocblas_ssyrk, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_ZSYRK_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                             \
  KOKKOSBLAS3_XSYRK_ROCBLAS(Kokkos::complex<double>, rocblas_double_complex, rocblas_zsyrk, LAYOUT, MEM_SPACE,   \
                            ETI_SPEC_AVAIL)

#define KOKKOSBLAS3_CSYRK_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                           \
  KOKKOSBLAS3_XSYRK_ROCBLAS(Kokkos::complex<float>, rocblas_float_complex, rocblas_csyrk, LAYOUT, MEM_SPACE,   \
                            ETI_SPEC_AVAIL)

KOKKOSBLAS3_DSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_DSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_SSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_SSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_ZSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_ZSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_CSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS3_CSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS3_DSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_DSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS3_SSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_SSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS3_ZSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_ZSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS3_CSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS3_CSYRK_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

#endif  // KOKKOSBLAS3_SYRK_TPL_SPEC_DECL_HPP_
