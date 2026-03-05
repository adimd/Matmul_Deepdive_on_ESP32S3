#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "benchmark.h"
#include "dsps_dotprod.h"

static void transpose_i16(const int16_t *B, int16_t *Bt, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Bt[j*n + i] = B[i*n + j];
}

void matmul_espdsp_i16(const mat_i16 *A,
                       const mat_i16 *B,
                             mat_i16 *C,
                       int n)
{
    mat_i16 *Bt = (mat_i16*)malloc(n * n * sizeof(mat_i16));
    if (!Bt) return;
    transpose_i16(B, Bt, n);

    int16_t tmp;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            dsps_dotprod_s16(A + i*n, Bt + j*n, &tmp, n, 0);
            C[i*n + j] = tmp;
        }

    free(Bt);
}

// esp-dsp has no SIMD path for i32/f32 — scalar tiled fallback
void matmul_espdsp_i32(const mat_i32 *A, const mat_i32 *B, mat_i32 *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int64_t acc = 0;
            for (int k = 0; k < n; k++)
                acc += (int64_t)A[i*n+k] * (int64_t)B[k*n+j];
            C[i*n+j] = (mat_i32)acc;
        }
}

void matmul_espdsp_f32(const mat_f32 *A, const mat_f32 *B, mat_f32 *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < n; k++)
                acc += A[i*n+k] * B[k*n+j];
            C[i*n+j] = acc;
        }
}