// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS2_SYMV_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS2_SYMV_TPL_SPEC_DECL_HPP_

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_SYMV_CUBLAS(SCALAR_TYPE, CUDA_SCALAR_TYPE, CUBLAS_FN, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)          \
  template <class ExecSpace>                                                                                          \
  struct SYMV<ExecSpace,                                                                                              \
              Kokkos::View<const SCALAR_TYPE**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                        \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<const SCALAR_TYPE*, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                         \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<SCALAR_TYPE*, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                               \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              true, ETI_SPEC_AVAIL> {                                                                                 \
    typedef SCALAR_TYPE SCALAR;                                                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        AViewType;                                                                                                    \
    typedef Kokkos::View<const SCALAR*, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        XViewType;                                                                                                    \
    typedef Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,                                       \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        YViewType;                                                                                                    \
                                                                                                                       \
    static void symv(const ExecSpace& space, const char uplo[], typename YViewType::const_value_type& alpha,          \
                     const AViewType& A, const XViewType& x, typename YViewType::const_value_type& beta,             \
                     const YViewType& y) {                                                                            \
      Kokkos::Profiling::pushRegion("KokkosBlas::symv[TPL_CUBLAS," #SCALAR_TYPE "]");                                 \
      const int N = static_cast<int>(A.extent(0));                                                                    \
                                                                                                                       \
      const int AST = A.stride(1), LDA = (AST == 0) ? 1 : AST;                                                        \
      const int XST = x.stride(0), INCX = (XST == 0) ? 1 : XST;                                                       \
      const int YST = y.stride(0), INCY = (YST == 0) ? 1 : YST;                                                       \
                                                                                                                       \
      cublasFillMode_t uplo_ = ((uplo[0] == 'L') || (uplo[0] == 'l')) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER; \
                                                                                                                       \
      KokkosBlas::Impl::CudaBlasSingleton& s = KokkosBlas::Impl::CudaBlasSingleton::singleton();                      \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, space.cuda_stream()));                               \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(                                                                               \
          CUBLAS_FN(s.handle, uplo_, N, reinterpret_cast<const CUDA_SCALAR_TYPE*>(&alpha),                           \
                    reinterpret_cast<const CUDA_SCALAR_TYPE*>(A.data()), LDA,                                        \
                    reinterpret_cast<const CUDA_SCALAR_TYPE*>(x.data()), INCX,                                       \
                    reinterpret_cast<const CUDA_SCALAR_TYPE*>(&beta), reinterpret_cast<CUDA_SCALAR_TYPE*>(y.data()), \
                    INCY));                                                                                           \
      KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSetStream(s.handle, NULL));                                              \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS2_DSYMV_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS2_SYMV_CUBLAS(double, double, cublasDsymv, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS2_SSYMV_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS2_SYMV_CUBLAS(float, float, cublasSsymv, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS2_ZSYMV_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS2_SYMV_CUBLAS(Kokkos::complex<double>, cuDoubleComplex, cublasZsymv, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS2_CSYMV_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS2_SYMV_CUBLAS(Kokkos::complex<float>, cuComplex, cublasCsymv, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

KOKKOSBLAS2_DSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS2_DSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS2_DSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS2_DSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS2_SSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS2_SSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS2_SSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS2_SSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS2_ZSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS2_ZSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS2_ZSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS2_ZSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS2_CSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS2_CSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS2_CSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS2_CSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_XSYMV_ROCBLAS(SCALAR_TYPE, ROCBLAS_SCALAR_TYPE, ROCBLAS_FN, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)    \
  template <>                                                                                                         \
  struct SYMV<Kokkos::HIP,                                                                                            \
              Kokkos::View<const SCALAR_TYPE**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                       \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<const SCALAR_TYPE*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                        \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              Kokkos::View<SCALAR_TYPE*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                                                 \
              true, ETI_SPEC_AVAIL> {                                                                                 \
    typedef SCALAR_TYPE SCALAR;                                                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        AViewType;                                                                                                    \
    typedef Kokkos::View<const SCALAR*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                               \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        XViewType;                                                                                                    \
    typedef Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                                     \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >                                                    \
        YViewType;                                                                                                    \
                                                                                                                       \
    static void symv(const typename YViewType::execution_space& space, const char uplo[],                            \
                     typename YViewType::const_value_type& alpha, const AViewType& A, const XViewType& x,            \
                     typename YViewType::const_value_type& beta, const YViewType& y) {                                \
      Kokkos::Profiling::pushRegion("KokkosBlas::symv[TPL_ROCBLAS," #SCALAR_TYPE "]");                                \
      const int N = static_cast<int>(A.extent(0));                                                                    \
                                                                                                                       \
      const int AST = A.stride(1), LDA = (AST == 0) ? 1 : AST;                                                        \
      const int XST = x.stride(0), INCX = (XST == 0) ? 1 : XST;                                                       \
      const int YST = y.stride(0), INCY = (YST == 0) ? 1 : YST;                                                       \
                                                                                                                       \
      rocblas_fill uplo_ = ((uplo[0] == 'L') || (uplo[0] == 'l')) ? rocblas_fill_lower : rocblas_fill_upper;          \
                                                                                                                       \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                        \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, space.hip_stream()));                            \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(                                                                              \
          ROCBLAS_FN(s.handle, uplo_, N, reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(&alpha),                       \
                     reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(A.data()), LDA,                                     \
                     reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(x.data()), INCX,                                    \
                     reinterpret_cast<const ROCBLAS_SCALAR_TYPE*>(&beta),                                             \
                     reinterpret_cast<ROCBLAS_SCALAR_TYPE*>(y.data()), INCY));                                        \
      KOKKOSBLAS_IMPL_ROCBLAS_SAFE_CALL(rocblas_set_stream(s.handle, NULL));                                          \
      Kokkos::Profiling::popRegion();                                                                                 \
    }                                                                                                                 \
  };

#define KOKKOSBLAS2_DSYMV_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS2_XSYMV_ROCBLAS(double, double, rocblas_dsymv, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS2_SSYMV_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL) \
  KOKKOSBLAS2_XSYMV_ROCBLAS(float, float, rocblas_ssymv, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)

#define KOKKOSBLAS2_ZSYMV_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                             \
  KOKKOSBLAS2_XSYMV_ROCBLAS(Kokkos::complex<double>, rocblas_double_complex, rocblas_zsymv, LAYOUT, MEM_SPACE,   \
                            ETI_SPEC_AVAIL)

#define KOKKOSBLAS2_CSYMV_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                           \
  KOKKOSBLAS2_XSYMV_ROCBLAS(Kokkos::complex<float>, rocblas_float_complex, rocblas_csymv, LAYOUT, MEM_SPACE,   \
                            ETI_SPEC_AVAIL)

KOKKOSBLAS2_DSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS2_DSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS2_SSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS2_SSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS2_ZSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS2_ZSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS2_CSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS2_CSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)

KOKKOSBLAS2_DSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS2_DSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS2_SSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS2_SSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS2_ZSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS2_ZSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

KOKKOSBLAS2_CSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, true)
KOKKOSBLAS2_CSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPManagedSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

#endif  // KOKKOSBLAS2_SYMV_TPL_SPEC_DECL_HPP_
