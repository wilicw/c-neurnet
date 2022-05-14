#include "nn.h"

void nn_create(nn_t *nn, uint64_t net_count, loss_t loss) {
  nn->net_count = net_count;
  nn->network = malloc(sizeof(abstract_layer_t *) * net_count);
  nn->loss = loss;
}

void nn_set_layer(nn_t*nn,abstract_layer_t *layer, uint64_t index) {
  nn->network[index] = layer;
}

void nn_evaulate(nn_t *nn, mat_t *output, mat_t *input) {
  mat_t __input;
  mat_cpy(&__input, input);
  for (uint64_t i = 0; i < nn->net_count; i++) {
    nn->network[i]->base.forward(nn->network[i], output, &__input);
    mat_cpy(&__input, output);
  }
}

float128_t nn_loss(nn_t *nn, mat_t *predict, mat_t *answer) {
  return nn->loss.forward(predict, answer);
}
