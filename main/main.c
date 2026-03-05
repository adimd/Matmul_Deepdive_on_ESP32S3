#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "benchmark.h"

// ── Forward declarations ──────────────────────────────────────
void matmul_naive_i16(const mat_i16 *A, const mat_i16 *B, mat_i16 *C, int n);
void matmul_naive_i32(const mat_i32 *A, const mat_i32 *B, mat_i32 *C, int n);
void matmul_naive_f32(const mat_f32 *A, const mat_f32 *B, mat_f32 *C, int n);

void matmul_opt_i16(const mat_i16 *A, const mat_i16 *B, mat_i16 *C, int n);
void matmul_opt_i32(const mat_i32 *A, const mat_i32 *B, mat_i32 *C, int n);
void matmul_opt_f32(const mat_f32 *A, const mat_f32 *B, mat_f32 *C, int n);

void matmul_tiled_i16(const mat_i16 *A, const mat_i16 *B, mat_i16 *C, int n);
void matmul_tiled_i32(const mat_i32 *A, const mat_i32 *B, mat_i32 *C, int n);
void matmul_tiled_f32(const mat_f32 *A, const mat_f32 *B, mat_f32 *C, int n);

void matmul_espdsp_i16(const mat_i16 *A, const mat_i16 *B, mat_i16 *C, int n);
void matmul_espdsp_i32(const mat_i32 *A, const mat_i32 *B, mat_i32 *C, int n);
void matmul_espdsp_f32(const mat_f32 *A, const mat_f32 *B, mat_f32 *C, int n);

// ── Sort helpers ──────────────────────────────────────────────
static void sort_u32(uint32_t *arr, int n) {
    for (int i = 1; i < n; i++) {
        uint32_t key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = key;
    }
}

static void sort_i64(int64_t *arr, int n) {
    for (int i = 1; i < n; i++) {
        int64_t key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = key;
    }
}

// ── Median helpers ────────────────────────────────────────────
static uint32_t median_u32(uint32_t *arr, int n) {
    uint32_t tmp[BENCH_RUNS];
    memcpy(tmp, arr, n * sizeof(uint32_t));
    sort_u32(tmp, n);
    return tmp[n / 2];
}

static int64_t median_i64(int64_t *arr, int n) {
    int64_t tmp[BENCH_RUNS];
    memcpy(tmp, arr, n * sizeof(int64_t));
    sort_i64(tmp, n);
    return tmp[n / 2];
}

// ── Matrix fill ───────────────────────────────────────────────
static void fill_i16(mat_i16 *M, int elems, uint32_t seed) {
    uint32_t s = seed;
    for (int i = 0; i < elems; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        M[i] = (mat_i16)((s & 0x7F) - 64);
    }
}

static void fill_i32(mat_i32 *M, int elems, uint32_t seed) {
    uint32_t s = seed;
    for (int i = 0; i < elems; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        M[i] = (mat_i32)((int)((s & 0x7F) - 64));
    }
}

static void fill_f32(mat_f32 *M, int elems, uint32_t seed) {
    uint32_t s = seed;
    for (int i = 0; i < elems; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        M[i] = (mat_f32)((int)((s & 0x7F) - 64));
    }
}

// ── Print one result row ──────────────────────────────────────
static void print_result(const char *dtype, bench_samples_t *s) {
    uint32_t med_c = median_u32(s->cycles, BENCH_RUNS);
    int64_t  med_u = median_i64(s->us,     BENCH_RUNS);
    printf("  %-4s | %12lu cycles | %10lld us\n",
           dtype,
           (unsigned long)med_c,
           (long long)med_u);
}

// ── Run benchmark for one impl, one size ─────────────────────
static void run_bench_size(
    const char *label, int n,
    void (*fn_i16)(const mat_i16*, const mat_i16*, mat_i16*, int),
    void (*fn_i32)(const mat_i32*, const mat_i32*, mat_i32*, int),
    void (*fn_f32)(const mat_f32*, const mat_f32*, mat_f32*, int))
{
    int elems = n * n;
    bench_samples_t s;

    printf("\n[%s]  n = %d x %d\n", label, n, n);
    printf("  type | median cycles      | median us\n");
    printf("  -----|--------------------|-----------\n");

    // ── int16 ──────────────────────────────────────────────
    {
        mat_i16 *A = malloc(elems * sizeof(mat_i16));
        mat_i16 *B = malloc(elems * sizeof(mat_i16));
        mat_i16 *C = malloc(elems * sizeof(mat_i16));

        if (!A || !B || !C) {
            printf("  i16  | malloc failed for n=%d\n", n);
        } else {
            fill_i16(A, elems, 0xAABB);
            fill_i16(B, elems, 0xCCDD);
            for (int r = 0; r < BENCH_RUNS; r++) {
                memset(C, 0, elems * sizeof(mat_i16));
                BENCH_CYCLES_START(); BENCH_US_START();
                fn_i16(A, B, C, n);
                s.cycles[r] = BENCH_CYCLES_STOP();
                s.us[r]     = BENCH_US_STOP();
            }
            print_result("i16", &s);
        }
        free(A); free(B); free(C);
    }

    // ── int32 ──────────────────────────────────────────────
    {
        mat_i32 *A = malloc(elems * sizeof(mat_i32));
        mat_i32 *B = malloc(elems * sizeof(mat_i32));
        mat_i32 *C = malloc(elems * sizeof(mat_i32));

        if (!A || !B || !C) {
            printf("  i32  | malloc failed for n=%d\n", n);
        } else {
            fill_i32(A, elems, 0xAABB);
            fill_i32(B, elems, 0xCCDD);
            for (int r = 0; r < BENCH_RUNS; r++) {
                memset(C, 0, elems * sizeof(mat_i32));
                BENCH_CYCLES_START(); BENCH_US_START();
                fn_i32(A, B, C, n);
                s.cycles[r] = BENCH_CYCLES_STOP();
                s.us[r]     = BENCH_US_STOP();
            }
            print_result("i32", &s);
        }
        free(A); free(B); free(C);
    }

    // ── float ──────────────────────────────────────────────
    {
        mat_f32 *A = malloc(elems * sizeof(mat_f32));
        mat_f32 *B = malloc(elems * sizeof(mat_f32));
        mat_f32 *C = malloc(elems * sizeof(mat_f32));

        if (!A || !B || !C) {
            printf("  f32  | malloc failed for n=%d\n", n);
        } else {
            fill_f32(A, elems, 0xAABB);
            fill_f32(B, elems, 0xCCDD);
            for (int r = 0; r < BENCH_RUNS; r++) {
                memset(C, 0, elems * sizeof(mat_f32));
                BENCH_CYCLES_START(); BENCH_US_START();
                fn_f32(A, B, C, n);
                s.cycles[r] = BENCH_CYCLES_STOP();
                s.us[r]     = BENCH_US_STOP();
            }
            print_result("f32", &s);
        }
        free(A); free(B); free(C);
    }
}

// ── Dummy infrastructure self-test ───────────────────────────
static void run_dummy_benchmark(void) {
    bench_samples_t s;
    const uint32_t ITERS = 10000;
    volatile uint32_t x = 0;

    printf("\n=== Benchmark Infrastructure Self-Test ===\n");
    for (int r = 0; r < BENCH_RUNS; r++) {
        BENCH_CYCLES_START(); BENCH_US_START();
        for (uint32_t i = 0; i < ITERS; i++) x += i;
        s.cycles[r] = BENCH_CYCLES_STOP();
        s.us[r]     = BENCH_US_STOP();
    }

    uint32_t med_c = median_u32(s.cycles, BENCH_RUNS);
    int64_t  med_u = median_i64(s.us,     BENCH_RUNS);
    printf("Median cycles: %lu | Median us: %lld | Cycles/us: %.1f\n",
           (unsigned long)med_c,
           (long long)med_u,
           (double)med_c / (double)med_u);
    (void)x;
}

// ── Main ─────────────────────────────────────────────────────
void app_main(void) {
    printf("\nMatmul_Deepdive_on_ESP32S3\n");
    vTaskDelay(pdMS_TO_TICKS(100));

    run_dummy_benchmark();

    printf("\n=== matmul_naive (-O0) ===\n");
    for (int si = 0; si < NUM_SIZES; si++) {
        run_bench_size("naive_O0", MATRIX_SIZES[si],
                       matmul_naive_i16,
                       matmul_naive_i32,
                       matmul_naive_f32);
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    printf("\n=== matmul_opt (-O3) ===\n");
    for (int si = 0; si < NUM_SIZES; si++) {
        run_bench_size("opt_O3", MATRIX_SIZES[si],
                       matmul_opt_i16,
                       matmul_opt_i32,
                       matmul_opt_f32);
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    printf("\n=== matmul_tiled (T=32, -O3) ===\n");
for (int si = 0; si < NUM_SIZES; si++) {
    run_bench_size("tiled_O3", MATRIX_SIZES[si],
                   matmul_tiled_i16,
                   matmul_tiled_i32,
                   matmul_tiled_f32);
    vTaskDelay(pdMS_TO_TICKS(10));
}

printf("\n=== matmul_espdsp (PIE SIMD via esp-dsp, i16 only) ===\n");
for (int si = 0; si < NUM_SIZES; si++) {
    run_bench_size("espdsp", MATRIX_SIZES[si],
                   matmul_espdsp_i16,
                   matmul_espdsp_i32,
                   matmul_espdsp_f32);
    vTaskDelay(pdMS_TO_TICKS(10));
}

    printf("\n=== Done ===\n");
}