#ifndef __LOSS__
#define __LOSS__

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "mat.h"

typedef float128_t (*loss_forward_f)(mat_t *, mat_t *);
typedef float128_t (*loss_baclward_f)(mat_t *, mat_t *);

typedef struct {
  loss_forward_f forward;
  loss_baclward_f backward;
} loss_t;

extern loss_t mse;
extern loss_t binary_cross_entropy;

#endif