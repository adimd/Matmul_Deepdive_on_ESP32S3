/**
 * matmul_tiled.c
 *
 * Cache-blocked matrix multiplication.
 * Compiled at -O3.
 *
 * Key idea: divide matrices into T×T tiles that fit in L1 cache.
 * Process one tile-triple at a time so the inner loop only
 * touches data that's already in cache.
 *
 * Hardware target:
 *   L1 D-cache: 16KB
 *   Cache line: 32 bytes
 *   Tile size:  32×32
 *
 * Memory per tile-triple:
 *   i16: 3 × 32² × 2 =  6KB  (fits in 16KB L1)
 *   i32: 3 × 32² × 4 = 12KB  (fits in 16KB L1)
 *   f32: 3 × 32² × 4 = 12KB  (fits in 16KB L1)
 *
 * Access pattern vs naive:
 *   naive:  B[k][j] strides across columns → cache miss every step
 *   tiled:  entire T×T tile of B stays in cache for T² inner ops
 */

#include <stdint.h>
#include "benchmark.h"

#define TILE 32

// ─────────────────────────────────────────────────────────────
//  int16 tiled
// ─────────────────────────────────────────────────────────────
void matmul_tiled_i16(const mat_i16 *A,
                      const mat_i16 *B,
                            mat_i16 *C,
                      int n)
{
    for (int ii = 0; ii < n; ii += TILE) {
        for (int jj = 0; jj < n; jj += TILE) {
            for (int kk = 0; kk < n; kk += TILE) {

                int i_end = ii + TILE < n ? ii + TILE : n;
                int j_end = jj + TILE < n ? jj + TILE : n;
                int k_end = kk + TILE < n ? kk + TILE : n;

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        int32_t acc = C[i*n + j];
                        for (int k = kk; k < k_end; k++) {
                            acc += (int32_t)A[i*n + k] *
                                   (int32_t)B[k*n + j];
                        }
                        C[i*n + j] = (mat_i16)acc;
                    }
                }

            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  int32 tiled
// ─────────────────────────────────────────────────────────────
void matmul_tiled_i32(const mat_i32 *A,
                      const mat_i32 *B,
                            mat_i32 *C,
                      int n)
{
    for (int ii = 0; ii < n; ii += TILE) {
        for (int jj = 0; jj < n; jj += TILE) {
            for (int kk = 0; kk < n; kk += TILE) {

                int i_end = ii + TILE < n ? ii + TILE : n;
                int j_end = jj + TILE < n ? jj + TILE : n;
                int k_end = kk + TILE < n ? kk + TILE : n;

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        int64_t acc = C[i*n + j];
                        for (int k = kk; k < k_end; k++) {
                            acc += (int64_t)A[i*n + k] *
                                   (int64_t)B[k*n + j];
                        }
                        C[i*n + j] = (mat_i32)acc;
                    }
                }

            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  float tiled
// ─────────────────────────────────────────────────────────────
void matmul_tiled_f32(const mat_f32 *A,
                      const mat_f32 *B,
                            mat_f32 *C,
                      int n)
{
    for (int ii = 0; ii < n; ii += TILE) {
        for (int jj = 0; jj < n; jj += TILE) {
            for (int kk = 0; kk < n; kk += TILE) {

                int i_end = ii + TILE < n ? ii + TILE : n;
                int j_end = jj + TILE < n ? jj + TILE : n;
                int k_end = kk + TILE < n ? kk + TILE : n;

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        float acc = C[i*n + j];
                        for (int k = kk; k < k_end; k++) {
                            acc += A[i*n + k] * B[k*n + j];
                        }
                        C[i*n + j] = acc;
                    }
                }

            }
        }
    }
}