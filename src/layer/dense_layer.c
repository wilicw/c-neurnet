#include "dense_layer.h"

void __dense_forward(void *layer, mat_t *dst, mat_t *src) {
  dense_t *l = (dense_t *)(layer);
  mat_create(dst, l->weight.row, src->column);
  mat_mul(dst, &(l->weight), src);
  l->act.forward(dst);
}

void dense_create(dense_t *layer, uint64_t input, uint64_t output, activation_t act) {
  mat_create(&(layer->weight), output, input);
  mat_fill(&(layer->weight), MAT_RANDOM);
  mat_add_const(&(layer->weight), &(layer->weight), -0.5);
  layer->base.forward = __dense_forward;
  layer->act = act;
}