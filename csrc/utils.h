// Copyright (c) 2021 MIT
//
// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#ifndef UTILS_H
#define UTILS_H

#include <memory>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <nccl.h>

#define UNUSED(expr) (void)(expr)

template<typename ... Args>
std::string
format(const std::string& format, Args ... args)
{
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args ...);
  return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

static inline
std::string tsrToStr(torch::Tensor tensor)
{
  std::ostringstream stream;
  stream << tensor;
  return stream.str();
}

static inline
std::string tsrSizeToStr(torch::Tensor tensor)
{
  std::ostringstream stream;
  auto sizes = tensor.sizes();
  stream << "[ ";
  for (auto size : sizes) {
    stream << size << " ";
  }
  stream << "]";
  return stream.str();
}

#define NCCL_API_CALL(apiFuncCall)                                            \
  do {                                                                        \
    ncclResult_t _status = apiFuncCall;                                        \
    if (_status != ncclSuccess) {                                             \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",    \
              __FILE__, __LINE__, #apiFuncCall, _status); \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)

#define CUDA_API_CALL(apiFuncCall)                                            \
  do {                                                                        \
    cudaError_t _status = apiFuncCall;                                        \
    if (_status != cudaSuccess) {                                             \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
              __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)

#define CUDACHECK(cmd) CUDA_API_CALL(cmd)

#endif