#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_timer.h"
#include "esp_cpu.h"

// ─────────────────────────────────────────────
//  Matrix sizes to benchmark
// ─────────────────────────────────────────────
#define NUM_SIZES                6
#define MAX_N                    256

static const int MATRIX_SIZES[NUM_SIZES] = {8, 16, 32, 64, 128, 256};

// ─────────────────────────────────────────────
//  Number of runs per measurement
// ─────────────────────────────────────────────
#define BENCH_RUNS               10

// ─────────────────────────────────────────────
//  Supported data types
//  Each gets its own experiment pass
// ─────────────────────────────────────────────
typedef int16_t  mat_i16;   // 8-wide PIE SIMD
typedef int32_t  mat_i32;   // 4-wide PIE SIMD
typedef float    mat_f32;   // software FPU, slowest

// ─────────────────────────────────────────────
//  Timing macros
//  CCOUNT = hardware cycle counter (CCOUNT reg)
//  reads directly, zero overhead
// ─────────────────────────────────────────────
#define BENCH_CYCLES_START()   uint32_t _c0 = esp_cpu_get_cycle_count()
#define BENCH_CYCLES_STOP()    (esp_cpu_get_cycle_count() - _c0)

#define BENCH_US_START()       int64_t _t0 = esp_timer_get_time()
#define BENCH_US_STOP()        (esp_timer_get_time() - _t0)

// ─────────────────────────────────────────────
//  Raw sample storage (per run)
// ─────────────────────────────────────────────
typedef struct {
    uint32_t cycles[BENCH_RUNS];   // raw cycle counts
    int64_t  us[BENCH_RUNS];       // raw microsecond counts
} bench_samples_t;

// ─────────────────────────────────────────────
//  Final result after median computation
// ─────────────────────────────────────────────
typedef struct {
    const char *impl_name;         // e.g. "naive_i16"
    const char *dtype_name;        // e.g. "int16"
    int         n;                 // matrix size
    uint32_t    median_cycles;
    int64_t     median_us;
    double      gflops;            // 2*N^3 / time
} bench_result_t;