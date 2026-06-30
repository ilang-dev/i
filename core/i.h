#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

typedef enum {
  I_INPUT_FREE = 0,
  I_INPUT_BOUND = 1,
} i_input_state;

i_component* i_parse(const char* src);
i_component* i_identity(void);

i_component* i_chain(const i_component* left, const i_component* right);
i_component* i_compose(const i_component* left, const i_component* right);
i_component* i_fanout(const i_component* left, const i_component* right);
i_component* i_pair(const i_component* left, const i_component* right);
i_component* i_swap(const i_component* component);
i_component* i_bind_input(const i_component* component, size_t input);

int i_component_input_count(const i_component* component, size_t* out);
int i_component_output_count(const i_component* component, size_t* out);
int i_component_input_states(const i_component* component, int* states);

char* i_code(const i_component* component, i_device device);
i_program* i_compile(const i_component* component, i_device device);
i_device i_program_device(const i_program* program);

float* i_alloc(i_device device, size_t len);
void i_free(i_device device, float* data);
int i_copy(i_device dst_device, float* dst, i_device src_device, const float* src, size_t len);

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

#ifdef __cplusplus
}
#endif
