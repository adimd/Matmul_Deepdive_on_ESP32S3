/**
 * matmul_naive.c
 *
 * Plain triple-nested loop matrix multiplication.
 * Compiled at -O0 — zero compiler optimization.
 * This is our baseline — the worst case.
 *
 * Algorithm:
 *   for i in 0..n
 *     for j in 0..n
 *       for k in 0..n
 *         C[i][j] += A[i][k] * B[k][j]
 *
 * Memory access pattern:
 *   A[i][k] — row major, sequential     ✓ cache friendly
 *   B[k][j] — column major, stride = n  ✗ cache unfriendly
 *   C[i][j] — row major, sequential     ✓ cache friendly
 *
 * At large N, the B access pattern causes cache thrashing.
 * This is intentional — it's what we're measuring.
 */

#include <stdint.h>
#include "benchmark.h"

// ─────────────────────────────────────────────────────────────
//  int16 variant
//  inputs:  int16_t
//  acc:     int32_t  (prevents overflow for our value range)
//  output:  int16_t
// ─────────────────────────────────────────────────────────────
void matmul_naive_i16(const mat_i16 *A,
                      const mat_i16 *B,
                            mat_i16 *C,
                      int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int k = 0; k < n; k++) {
                acc += (int32_t)A[i*n + k] * (int32_t)B[k*n + j];
            }
            C[i*n + j] = (mat_i16)acc;
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  int32 variant
//  inputs:  int32_t
//  acc:     int64_t
//  output:  int32_t
// ─────────────────────────────────────────────────────────────
void matmul_naive_i32(const mat_i32 *A,
                      const mat_i32 *B,
                            mat_i32 *C,
                      int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int64_t acc = 0;
            for (int k = 0; k < n; k++) {
                acc += (int64_t)A[i*n + k] * (int64_t)B[k*n + j];
            }
            C[i*n + j] = (mat_i32)acc;
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  float variant
//  inputs:  float
//  acc:     float
//  output:  float
//  NOTE: LX7 has no FPU — float is software emulated
//        expect this to be significantly slower than int
// ─────────────────────────────────────────────────────────────
void matmul_naive_f32(const mat_f32 *A,
                      const mat_f32 *B,
                            mat_f32 *C,
                      int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < n; k++) {
                acc += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = acc;
        }
    }
}