/**
 * matmul_opt.c
 *
 * Same triple-nested loop as matmul_naive.c
 * but compiled at -O3 with full compiler optimization.
 *
 * What -O3 enables over -O0:
 *   - Loop unrolling
 *   - Instruction scheduling
 *   - Auto-vectorization (limited on Xtensa)
 *   - Register allocation optimization
 *   - Inlining
 *   - Strength reduction
 *
 * This tells us how much the compiler can extract
 * from naive code without any algorithmic changes.
 */

#include <stdint.h>
#include "benchmark.h"

void matmul_opt_i16(const mat_i16 *A,
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

void matmul_opt_i32(const mat_i32 *A,
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

void matmul_opt_f32(const mat_f32 *A,
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