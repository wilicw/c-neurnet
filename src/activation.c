#include "activation.h"

float128_t __sigmoid(float128_t v) { return 1 / (1 + exp(-v)); }

void sigmoid_forward(mat_t* mat) {
  mat_map(mat, mat, __sigmoid);
}

void sigmoid_backward(mat_t* mat) {
}

void softmax_forward(mat_t* mat) {
  mat_exp(mat, mat);
  float128_t sum = mat_sum(mat);
  mat_mul_const(mat, mat, 1 / sum);
}

void softmax_backward(mat_t* mat) {
}

activation_t sigmoid = {sigmoid_forward, sigmoid_backward};
activation_t softmax = {softmax_forward, softmax_backward};