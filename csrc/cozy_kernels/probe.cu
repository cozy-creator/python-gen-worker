// The trivial probe kernel: proves fatbin arch coverage + launch + numerics
// on the running device before any real op is trusted (pgw#860).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace cozy_kernels {

__global__ void add_one_kernel(const float* in, float* out, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = in[i] + 1.0f;
}

at::Tensor probe_add_one_cuda(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat,
              "probe_add_one expects a float32 CUDA tensor");
  auto xin = x.contiguous();
  auto out = at::empty_like(xin);
  const int64_t n = xin.numel();
  if (n == 0) return out;
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  add_one_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      xin.data_ptr<float>(), out.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace cozy_kernels
