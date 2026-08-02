// _cozy_kernels — TORCH_LIBRARY skeleton (pgw#860).
//
// Probe + build-info ops only; math kernels land per the pgw#862/#863 gates.
// Compiled against the shared cuda base image's exact torch pin; the ABI
// floats with the base-image bump, never independently. Ship: prebuilt .so
// baked into the base image (pgw#859 G0 recommendation) — NOT a wheel extra.
#include <torch/library.h>
#include <torch/version.h>

#include <ATen/ATen.h>

#include <string>

#ifndef COZY_FATBIN_ARCHS
#define COZY_FATBIN_ARCHS "unknown"
#endif

namespace cozy_kernels {

at::Tensor probe_add_one_cuda(const at::Tensor& x);  // probe.cu

// Torch pin + fatbin archs the .so was built with — the dispatch probe
// compares this against the running torch before trusting any op.
std::string build_info() {
  return std::string("torch=") + TORCH_VERSION +
         " archs=" + COZY_FATBIN_ARCHS;
}

TORCH_LIBRARY(cozy_kernels, m) {
  m.def("probe_add_one(Tensor x) -> Tensor");
  m.def("build_info", &build_info);
}

TORCH_LIBRARY_IMPL(cozy_kernels, CUDA, m) {
  m.impl("probe_add_one", &probe_add_one_cuda);
}

}  // namespace cozy_kernels
