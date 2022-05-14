#ifndef __ACTIVATION__
#define __ACTIVATION__

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mat.h"

typedef void (*activation_ff)(mat_t*);
typedef void (*activation_bf)(mat_t*);

typedef struct {
  activation_ff forward;
  activation_bf backward;
} activation_t;

extern activation_t sigmoid;
extern activation_t softmax;

#endif