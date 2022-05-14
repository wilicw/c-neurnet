#ifndef __NN__
#define __NN__

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layer.h"
#include "loss.h"
#include "mat.h"

typedef struct {
  uint64_t net_count;
  abstract_layer_t **network;
  loss_t loss;
} nn_t;

void nn_create(nn_t *, uint64_t, loss_t);
void nn_set_layer(nn_t *, abstract_layer_t *, uint64_t);
void nn_evaulate(nn_t *, mat_t *, mat_t *);
float128_t nn_loss(nn_t *, mat_t *, mat_t *);

#endif