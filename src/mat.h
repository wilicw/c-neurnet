#ifndef __MAT__
#define __MAT__

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAT_ZERO 0
#define MAT_RANDOM 1

#define INDEX_ERR (uint64_t)(-1)

#define MAT_CMP_MAX(x, y) (x) > (y) ? (x) : (y)

typedef long double float128_t;

typedef struct {
  uint64_t row;
  uint64_t column;
  float128_t *args;
} mat_t;

typedef struct {
  uint64_t i;
  uint64_t j;
} index_t;

void mat_create(mat_t *, uint64_t, uint64_t);
void mat_destroy(mat_t *);
void mat_set(mat_t *, uint64_t, uint64_t, float128_t);
float128_t mat_get(mat_t *, uint64_t, uint64_t);
void mat_fill(mat_t *, int);
void mat_fill_const(mat_t *, float128_t);
void mat_cpy(mat_t *, mat_t *);
void mat_add(mat_t *, mat_t *, mat_t *);
void mat_mul(mat_t *, mat_t *, mat_t *);
void mat_add_const(mat_t *, mat_t *, float128_t);
void mat_mul_const(mat_t *, mat_t *, float128_t);
void mat_hadamard_mul(mat_t *, mat_t *, mat_t *);
void mat_T(mat_t *, mat_t *);
void mat_sqrt(mat_t *, mat_t *);
float128_t mat_sum(mat_t *);
float128_t mat_mean(mat_t *);
void mat_exp(mat_t *, mat_t *);
void mat_reshape(mat_t *, mat_t *, uint64_t, uint64_t);
void mat_map(mat_t *, mat_t *, float128_t (*)(float128_t));
float128_t mat_max(mat_t *);
void mat_index(index_t *, mat_t *, float128_t);

#endif