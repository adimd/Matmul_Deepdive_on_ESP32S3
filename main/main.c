#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "benchmark.h"

// ── Simple sort for median (insertion sort, 10 elements) ──────────
static void sort_u32(uint32_t *arr, int n) {
    for (int i = 1; i < n; i++) {
        uint32_t key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}

static void sort_i64(int64_t *arr, int n) {
    for (int i = 1; i < n; i++) {
        int64_t key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}

// ── Known-cost dummy workload ─────────────────────────────────────
// Volatile so compiler cannot optimize it away
static void dummy_workload(volatile uint32_t iters) {
    volatile uint32_t x = 0;
    for (uint32_t i = 0; i < iters; i++) {
        x += i;
    }
    (void)x;
}

// ── Run benchmark on dummy workload ──────────────────────────────
static void run_dummy_benchmark(void) {
    bench_samples_t s;
    const uint32_t ITERS = 10000;

    printf("\n=== Benchmark Infrastructure Self-Test ===\n");
    printf("Workload: dummy loop with %lu iterations\n\n", (unsigned long)ITERS);

    // Collect 10 raw samples
    for (int r = 0; r < BENCH_RUNS; r++) {
        BENCH_CYCLES_START();
        BENCH_US_START();

        dummy_workload(ITERS);

        s.cycles[r] = BENCH_CYCLES_STOP();
        s.us[r]     = BENCH_US_STOP();
    }

    // Print all raw samples
    printf("Run  |  Cycles   |  us\n");
    printf("-----|-----------|------\n");
    for (int r = 0; r < BENCH_RUNS; r++) {
        printf(" %2d  |  %8lu  |  %lld\n",
               r + 1,
               (unsigned long)s.cycles[r],
               (long long)s.us[r]);
    }

    // Compute and print median
    uint32_t c_sorted[BENCH_RUNS];
    int64_t  u_sorted[BENCH_RUNS];
    for (int i = 0; i < BENCH_RUNS; i++) {
        c_sorted[i] = s.cycles[i];
        u_sorted[i] = s.us[i];
    }
    sort_u32(c_sorted, BENCH_RUNS);
    sort_i64(u_sorted, BENCH_RUNS);

    uint32_t median_cycles = c_sorted[BENCH_RUNS / 2];
    int64_t  median_us     = u_sorted[BENCH_RUNS / 2];

    printf("\nMedian cycles : %lu\n",  (unsigned long)median_cycles);
    printf("Median us     : %lld\n",  (long long)median_us);

    // Sanity check: at 240MHz, cycles/us should be ~240
    if (median_us > 0) {
        double cycles_per_us = (double)median_cycles / (double)median_us;
        printf("Cycles/us     : %.1f  (expect ~240 at 240MHz)\n", cycles_per_us);
    }

    printf("\n==========================================\n");
}

void app_main(void) {
    printf("Matmul_Deepdive_on_ESP32S3\n");
    vTaskDelay(pdMS_TO_TICKS(100));  // let UART settle
    run_dummy_benchmark();
}