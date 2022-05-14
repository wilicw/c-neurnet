#ifndef __D_LAYER__
#define __D_LAYER__

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "mat.h"
#include "activation.h"

typedef struct {
  layer_t base;
  mat_t weight;
  activation_t act;
} dense_t;

void dense_create(dense_t*, uint64_t, uint64_t, activation_t);

#endif