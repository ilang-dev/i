#pragma once

#include <stddef.h>

typedef struct i_component i_component;
typedef struct i_program i_program;

typedef struct {
  const float* data;
  const size_t* shape;
  size_t rank;
} i_tensor;

typedef struct {
  float* data;
  const size_t* shape;
  size_t rank;
} i_tensor_mut;

typedef struct {
  float* data;
  size_t* shape;
  size_t rank;
  size_t len;
} i_owned_tensor;

typedef struct {
  i_owned_tensor* tensors;
  size_t count;
} i_outputs;

typedef enum {
  I_DEVICE_CPU = 0,
  I_DEVICE_CUDA = 1,
} i_device;

i_component* i_parse(const char* src);
i_component* i_identity(void);
i_component* i_chain(const i_component* left, const i_component* right);
i_component* i_compose(const i_component* left, const i_component* right);
i_component* i_fanout(const i_component* left, const i_component* right);
i_component* i_pair(const i_component* left, const i_component* right);
i_component* i_swap(const i_component* component);

char* i_code(const i_component* component);
char* i_cuda_code(const i_component* component);
i_program* i_compile(const i_component* component);
i_program* i_cuda_compile(const i_component* component);

i_device i_program_device(const i_program* program);
float* i_cuda_alloc(size_t len);
void i_cuda_free(float* data);
int i_cuda_copy_from_host(float* dst, const float* src, size_t len);
int i_cuda_copy_to_host(float* dst, const float* src, size_t len);

size_t i_output_count(const i_program* program);
int i_output_ranks(const i_program* program, size_t* ranks);
int i_output_shapes(
  const i_program* program,
  const i_tensor* inputs,
  size_t input_count,
  size_t** shapes
);

int i_exec_into(
  const i_program* program,
  const i_tensor* inputs,
  size_t input_count,
  i_tensor_mut* outputs,
  size_t output_count
);

i_outputs i_exec(
  const i_program* program,
  const i_tensor* inputs,
  size_t input_count
);

const char* i_error(void);
void i_component_free(i_component* component);
void i_program_free(i_program* program);
void i_outputs_free(i_outputs outputs);
void i_string_free(char* s);
