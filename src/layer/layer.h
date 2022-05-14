#ifndef __LAYER__
#define __LAYER__

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mat.h"

typedef void (*layer_forward_f)(void*,mat_t*, mat_t*);
typedef void (*layer_backward_f)(void*,mat_t*, mat_t*);

typedef struct {
  layer_forward_f forward;
  layer_backward_f backward;
} layer_t;

typedef struct {
  layer_t base;
} abstract_layer_t;

#endif