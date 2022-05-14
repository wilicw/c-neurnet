#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "activation.h"
#include "dense_layer.h"
#include "layer.h"
#include "loss.h"
#include "mat.h"
#include "nn.h"

void repr(mat_t *m) {
  for (uint64_t i = 0; i < m->row; i++) {
    for (uint64_t j = 0; j < m->column; j++) {
      printf("%.3Lf ", m->args[i * m->column + j]);
    }
    puts("");
  }
  puts("");
}

void create(nn_t *nn) {
  nn_create(nn, 4, binary_cross_entropy);
  {
    dense_t *layer = malloc(sizeof(dense_t));
    dense_create(layer, 28 * 28, 300, sigmoid);
    nn_set_layer(nn, (abstract_layer_t *)layer, 0);
  }

  {
    dense_t *layer = malloc(sizeof(dense_t));
    dense_create(layer, 300, 100, sigmoid);
    nn_set_layer(nn, (abstract_layer_t *)layer, 1);
  }

  {
    dense_t *layer = malloc(sizeof(dense_t));
    dense_create(layer, 100, 50, sigmoid);
    nn_set_layer(nn, (abstract_layer_t *)layer, 2);
  }

  {
    dense_t *layer = malloc(sizeof(dense_t));
    dense_create(layer, 50, 10, softmax);
    nn_set_layer(nn, (abstract_layer_t *)layer, 3);
  }
}

int main() {
  srand(time(NULL));

  nn_t nn;
  create(&nn);

  mat_t a;
  mat_create(&a, 28 * 28, 1);
  mat_fill(&a, MAT_RANDOM);
  repr(&a);

  mat_t nn_output;
  nn_evaulate(&nn, &nn_output, &a);
  repr(&nn_output);

  mat_t answer;
  mat_create(&answer, 10, 1);
  mat_fill(&answer, MAT_ZERO);
  mat_set(&answer, 5, 0, 1);

  printf("%Lf\n", nn_loss(&nn, &nn_output, &answer));
  return 0;
}