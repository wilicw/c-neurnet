#include "mat.h"

#include <math.h>

void mat_create(mat_t* m, uint64_t r, uint64_t c) {
  if (r == 0 || c == 0) return;
  m->row = r;
  m->column = c;
  m->args = malloc(sizeof(float128_t) * r * c);
}

void mat_destroy(mat_t* m) {
  free(m->args);
  free(m);
  m = NULL;
}

void mat_set(mat_t* m, uint64_t r, uint64_t c, float128_t v) {
  if (r >= m->row || c >= m->column) return;
  m->args[r * m->column + c] = v;
}

float128_t mat_get(mat_t* m, uint64_t r, uint64_t c) {
  if (r >= m->row || c >= m->column) return -1;
  return m->args[r * m->column + c];
}

void mat_fill(mat_t* m, int random) {
  for (uint64_t i = 0; i < m->row; i++)
    for (uint64_t j = 0; j < m->column; j++)
      mat_set(m, i, j, (float128_t)(random == MAT_ZERO ? 0 : (rand() % 1000) / 1000.0));
}

void mat_fill_const(mat_t* m, float128_t constant) {
  for (uint64_t i = 0; i < m->row; i++)
    for (uint64_t j = 0; j < m->column; j++)
      mat_set(m, i, j, constant);
}

void mat_cpy(mat_t* dst, mat_t* src) {
  mat_t zero;
  mat_create(&zero, src->row, src->column);
  mat_fill(&zero, MAT_ZERO);
  mat_create(dst, src->row, src->column);
  mat_add(dst, src, &zero);
}

void mat_add(mat_t* dst, mat_t* a, mat_t* b) {
  if ((a->row ^ b->row) == 0 && (a->column ^ b->column) != 0) {
  } else if ((a->row ^ b->row) != 0 && (a->column ^ b->column) == 0) {
  } else if ((a->row ^ b->row) == 0 && (a->column ^ b->column) == 0) {
    for (uint64_t i = 0; i < a->row; i++) {
      for (uint64_t j = 0; j < a->column; j++) {
        float128_t v = mat_get(a, i, j) + mat_get(b, i, j);
        mat_set(dst, i, j, v);
      }
    }
  } else {
    return;
  }
}

void mat_mul(mat_t* dst, mat_t* a, mat_t* b) {
  for (uint64_t i = 0; i < dst->row; i++) {
    for (uint64_t j = 0; j < dst->column; j++) {
      float128_t v = 0;
      for (uint64_t k = 0; k < a->column; k++) {
        v += mat_get(a, i, k) * mat_get(b, k, j);
      }
      mat_set(dst, i, j, v);
    }
  }
}

void mat_add_const(mat_t* dst, mat_t* a, float128_t constant) {
  for (uint64_t i = 0; i < a->row; i++)
    for (uint64_t j = 0; j < a->column; j++)
      mat_set(dst, i, j, mat_get(a, i, j) + constant);
}

void mat_mul_const(mat_t* dst, mat_t* a, float128_t constant) {
  for (uint64_t i = 0; i < a->row; i++)
    for (uint64_t j = 0; j < a->column; j++)
      mat_set(dst, i, j, mat_get(a, i, j) * constant);
}

void mat_hadamard_mul(mat_t* dst, mat_t* a, mat_t* b) {
  if (a->row != b->row || a->column != b->column) return;
  for (uint64_t i = 0; i < a->row; i++)
    for (uint64_t j = 0; j < a->column; j++)
      mat_set(dst, i, j, mat_get(a, i, j) * mat_get(b, i, j));
}

void mat_T(mat_t* dst, mat_t* src) {
  for (uint64_t i = 0; i < src->row; i++)
    for (uint64_t j = 0; j < src->column; j++)
      mat_set(dst, j, i, mat_get(src, i, j));
}

void mat_sqrt(mat_t* dst, mat_t* src) { mat_map(dst, src, sqrt); }

float128_t mat_sum(mat_t* src) {
  float128_t v = 0;
  for (uint64_t i = 0; i < src->row; i++)
    for (uint64_t j = 0; j < src->column; j++)
      v += mat_get(src, i, j);
  return v;
}

float128_t mat_mean(mat_t* src) { return mat_sum(src) / (src->row * src->column); }

void mat_exp(mat_t* dst, mat_t* src) { mat_map(dst, src, exp); }

void mat_reshape(mat_t* dst, mat_t* src, uint64_t r, uint64_t c) {
  if (r * c == 0 || r * c != src->row * src->column) return;
  mat_cpy(dst, src);
  dst->row = r;
  dst->column = c;
}
void mat_map(mat_t* dst, mat_t* src, float128_t (*func)(float128_t)) {
  for (uint64_t i = 0; i < src->row; i++)
    for (uint64_t j = 0; j < src->column; j++)
      mat_set(dst, i, j, func(mat_get(src, i, j)));
}

float128_t mat_max(mat_t* a) {
  float128_t v = mat_get(a, 0, 0);
  for (uint64_t i = 0; i < a->row; i++)
    for (uint64_t j = 0; j < a->column; j++)
      v = MAT_CMP_MAX(v, mat_get(a, i, j));
  return v;
}

void mat_index(index_t* index, mat_t* a, float128_t v) {
  index->i = index->j = INDEX_ERR;
  for (uint64_t i = 0; i < a->row; i++) {
    for (uint64_t j = 0; j < a->column; j++) {
      if (mat_get(a, i, j) == v) {
        index->i = i;
        index->j = j;
        return;
      }
    }
  }
}