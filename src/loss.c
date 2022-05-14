#include "loss.h"

float128_t __square(float128_t v) { return v * v; }

float128_t mse_forward(mat_t* predict, mat_t* answer) {
  mat_t copy_predict, copy_answer;
  mat_cpy(&copy_predict, predict);
  mat_cpy(&copy_answer, answer);
  mat_mul_const(&copy_answer, answer, -1);
  mat_add(&copy_predict, &copy_answer, predict);
  mat_map(&copy_predict, &copy_predict, __square);
  return mat_sum(&copy_predict);
}

float128_t mse_backward(mat_t* predict, mat_t* answer) {
  return 0;
}

float128_t binary_cross_entropy_forward(mat_t* predict, mat_t* answer) {
  mat_t copy_predict, copy_answer;
  mat_cpy(&copy_predict, predict);
  mat_cpy(&copy_answer, answer);
  mat_map(&copy_predict, &copy_predict, log);
  mat_hadamard_mul(&copy_answer, &copy_predict, &copy_answer);
  return mat_sum(&copy_answer);
}

float128_t binary_cross_entropy_backward(mat_t* predict, mat_t* answer) {
  return 0;
}

loss_t mse = {mse_forward, mse_backward};
loss_t binary_cross_entropy = {binary_cross_entropy_forward, binary_cross_entropy_backward};