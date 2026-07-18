// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {

// Specialization struct which defines whether a specialization exists
template <class execution_space, class AVT, class BVT, class CVT>
struct symm_tpl_spec_avail {
  enum : bool { value = false };
};

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#define KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUT, MEMSPACE)                                 \
  template <class ExecSpace>                                                                             \
  struct symm_tpl_spec_avail<ExecSpace,                                                                  \
                             Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>,   \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                     \
                             Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>,   \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                     \
                             Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>,         \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                  \
    enum : bool { value = true };                                                                        \
  };

KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaSpace)

KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)

#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

#define KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT, MEMSPACE)                                \
  template <>                                                                                            \
  struct symm_tpl_spec_avail<Kokkos::HIP,                                                                \
                             Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEMSPACE>, \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                     \
                             Kokkos::View<const SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEMSPACE>, \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                     \
                             Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEMSPACE>,       \
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {                  \
    enum : bool { value = true };                                                                        \
  };

KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::HIPSpace)

KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft, Kokkos::HIPManagedSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft, Kokkos::HIPManagedSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::HIPManagedSpace)
KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::HIPManagedSpace)

#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS3_SYMM_TPL_SPEC_AVAIL_HPP_
