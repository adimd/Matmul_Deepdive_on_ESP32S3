/**
 * matmul_tiled.c
 *
 * Cache-blocked matrix multiplication.
 * Compiled at -O3.
 *
 * Key idea: divide matrices into T×T tiles that fit in the L1 data cache.
 * Process one tile-triple at a time so the inner loop only touches data
 * that is already resident in cache, eliminating PSRAM stall cycles.
 *
 * Hardware target (XH-S3E N16R8, ESP32-S3R8):
 *   L1 D-cache : 32 KB (8-way set associative, confirmed via sdkconfig)
 *   Cache line : 32 bytes
 *   Tile size  : 32x32
 *
 * Why T=32 and not T=64?
 *   A tile-triple at T=64 for i16 uses 3 x 64^2 x 2 = 24 KB on paper,
 *   which looks like it fits in 32 KB. However the cache is shared with
 *   FreeRTOS, benchmark harness code, stack, and program instructions.
 *   Benchmarking confirmed that T=64 is 2.3x SLOWER than T=32 at n=256
 *   because the effective usable data cache is well below 32 KB in a
 *   real firmware environment. T=32 uses only 6 KB per tile-triple and
 *   stays comfortably resident regardless of system overhead.
 *   This is why you measure instead of just calculate.
 *
 * Memory footprint per tile-triple at T=32:
 *   i16: 3 x 32^2 x 2 =  6 KB  (fits comfortably in usable cache)
 *   i32: 3 x 32^2 x 4 = 12 KB  (fits comfortably in usable cache)
 *   f32: 3 x 32^2 x 4 = 12 KB  (fits comfortably in usable cache)
 *
 * Access pattern vs naive:
 *   naive : B[k][j] strides across columns -> cache miss on every k step
 *   tiled : entire T×T tile of B stays in cache for T^2 inner operations
 */

#include <stdint.h>
#include "benchmark.h"

#define TILE_I16 32
#define TILE_I32 32
#define TILE_F32 32

// ─────────────────────────────────────────────────────────────
//  int16 tiled  (T=64, fits in 32KB cache)
// ─────────────────────────────────────────────────────────────
void matmul_tiled_i16(const mat_i16 *A,
                      const mat_i16 *B,
                            mat_i16 *C,
                      int n)
{
    for (int ii = 0; ii < n; ii += TILE_I16) {
        for (int jj = 0; jj < n; jj += TILE_I16) {
            for (int kk = 0; kk < n; kk += TILE_I16) {

                int i_end = ii + TILE_I16 < n ? ii + TILE_I16 : n;
                int j_end = jj + TILE_I16 < n ? jj + TILE_I16 : n;
                int k_end = kk + TILE_I16 < n ? kk + TILE_I16 : n;

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
//  int32 tiled  (T=32, 3×32²×4 = 12KB fits in 32KB cache)
// ─────────────────────────────────────────────────────────────
void matmul_tiled_i32(const mat_i32 *A,
                      const mat_i32 *B,
                            mat_i32 *C,
                      int n)
{
    for (int ii = 0; ii < n; ii += TILE_I32) {
        for (int jj = 0; jj < n; jj += TILE_I32) {
            for (int kk = 0; kk < n; kk += TILE_I32) {

                int i_end = ii + TILE_I32 < n ? ii + TILE_I32 : n;
                int j_end = jj + TILE_I32 < n ? jj + TILE_I32 : n;
                int k_end = kk + TILE_I32 < n ? kk + TILE_I32 : n;

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
//  float tiled  (T=32, 3×32²×4 = 12KB fits in 32KB cache)
// ─────────────────────────────────────────────────────────────
void matmul_tiled_f32(const mat_f32 *A,
                      const mat_f32 *B,
                            mat_f32 *C,
                      int n)
{
    for (int ii = 0; ii < n; ii += TILE_F32) {
        for (int jj = 0; jj < n; jj += TILE_F32) {
            for (int kk = 0; kk < n; kk += TILE_F32) {

                int i_end = ii + TILE_F32 < n ? ii + TILE_F32 : n;
                int j_end = jj + TILE_F32 < n ? jj + TILE_F32 : n;
                int k_end = kk + TILE_F32 < n ? kk + TILE_F32 : n;

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