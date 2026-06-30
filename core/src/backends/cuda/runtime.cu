#include <cuda_runtime.h>
#include <stddef.h>
#include <stdlib.h>

#define CUDA_CHECK(expr) do { cudaError_t err__ = (expr); if (err__ != cudaSuccess) abort(); } while (0)

extern "C" float* i_cuda_tensor_alloc(size_t len) {
  float* data = NULL;
  CUDA_CHECK(cudaMallocManaged((void**)&data, len * sizeof(float)));
  return data;
}

extern "C" void i_cuda_tensor_free(float* data) {
  CUDA_CHECK(cudaFree(data));
}

extern "C" void i_cuda_tensor_copy_from_host(float* dst, const float* src, size_t len) {
  CUDA_CHECK(cudaMemcpy(dst, src, len * sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void i_cuda_tensor_copy_to_host(float* dst, const float* src, size_t len) {
  CUDA_CHECK(cudaMemcpy(dst, src, len * sizeof(float), cudaMemcpyDeviceToHost));
}

extern "C" void i_cuda_tensor_copy(float* dst, const float* src, size_t len) {
  CUDA_CHECK(cudaMemcpy(dst, src, len * sizeof(float), cudaMemcpyDeviceToDevice));
}
