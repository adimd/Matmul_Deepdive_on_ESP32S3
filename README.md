> **Note:** This project reflects my current understanding of the hardware and results. I am still learning and may have missed a few things — if you spot anything incorrect or incomplete, I would genuinely appreciate the feedback.

# Matmul Deepdive on ESP32-S3

A ground-up benchmarking study of matrix multiplication on the ESP32-S3, investigating whether Espressif's "AI-capable" claim holds up under real measurement. Four implementations are built from scratch, each one exposing a different layer of the hardware, and measured with cycle-accurate timing on real silicon.


## The Question

Espressif markets the ESP32-S3 as an AI-capable chip. It costs about three dollars. It has a dedicated SIMD unit called PIE (Processor Instruction Extensions) that can multiply eight 16-bit numbers simultaneously in a single clock cycle. The question this project asks is simple: does any of that actually matter in practice, and if so, when and why?

Matrix multiplication is the right operation to test this with. It is the core of every neural network layer. If the chip struggles here, it struggles everywhere that matters for AI workloads.


## Hardware

The board used is the XH-S3E N16R8, built around the ESP32-S3R8 chip. The full configuration that matters for these benchmarks:

| Property | Value |
|---|---|
| Chip | ESP32-S3R8 (Xtensa LX7 dual-core) |
| Clock | 240 MHz |
| Internal SRAM | 512 KB |
| L1 Data Cache | 32 KB, 8-way set associative, 32-byte lines |
| L1 Instruction Cache | 16 KB |
| External PSRAM | 8 MB Octal SPI @ 80 MHz |
| Flash | 16 MB |

The memory hierarchy is the central character of this story. Internal SRAM is fast. The L1 cache is a 32 KB window into that fast world. PSRAM is 8 MB but sits on an external bus and is roughly 10x slower to access. Everything interesting in these benchmarks comes from data moving between these three levels.


## The Four Implementations

Each implementation isolates exactly one variable so that the contribution of each layer is visible in the numbers.

**matmul_naive** is the textbook triple-nested loop compiled at -O0 with no optimization. B is accessed in column-major order (B[k][j]), which strides across memory and causes a cache miss on almost every inner-loop step. This is the honest baseline — what you get if you write the obvious code and do nothing else.

```c
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            C[i*n+j] += A[i*n+k] * B[k*n+j];  // B access is cache-unfriendly
```

**matmul_opt** is the same algorithm compiled at -O3. Loop unrolling, instruction scheduling, register allocation. This isolates exactly what the compiler can contribute without any algorithmic changes from our side.

**matmul_tiled** introduces cache-blocking. The matrices are divided into 32x32 tiles and the computation processes one tile-triple at a time. The inner loop only touches data already resident in cache, eliminating the PSRAM stall cycles that hurt the naive implementation so badly.

```c
#define TILE 32
for (int ii = 0; ii < n; ii += TILE)
    for (int jj = 0; jj < n; jj += TILE)
        for (int kk = 0; kk < n; kk += TILE)
            // inner 3 loops operate entirely within one tile-triple (6 KB for i16)
```

One decision worth explaining here: the data cache is 32 KB and a T=64 tile-triple for i16 uses only 24 KB, so it looks like T=64 should be better. Benchmarking proved otherwise — T=64 was 2.3x slower than T=32 at n=256. The reason is that the cache is not exclusively yours. FreeRTOS, the benchmark harness, the stack, and program instructions all compete for the same 32 KB. The effective usable data cache in a real firmware environment is well below 32 KB on paper. T=32 uses only 6 KB per tile-triple and stays resident regardless of system overhead. This is why you measure instead of just calculate.

**matmul_espdsp** uses Espressif's production esp-dsp library which internally calls `EE.VMULAS.S16.QACC` — the PIE instruction that accumulates 8 lane-wise products per clock cycle. B is transposed first so both row vectors are contiguous in memory, enabling SIMD-friendly sequential loads. Only i16 benefits from this path. There is no PIE multiply-accumulate for i32 or f32, so those types fall back to the same scalar code as opt_O3.

A note on the assembly path: the original plan was to write custom Xtensa PIE assembly directly. This turned out to be harder than expected. The Xtensa assembler does not support the `.option` directive (that is RISC-V syntax), and GCC's inline asm constraint system does not recognize Q registers as named clobbers. After several failed approaches the decision was made to use esp-dsp instead, which is Espressif's own production-tested implementation of exactly this operation. The label in the code reflects this honestly — it is called matmul_espdsp, not matmul_asm.


## Results

All measurements taken at 240 MHz. Median of 10 runs per data point. Cycle counter via the CCOUNT register, cross-validated against esp_timer.

**i16 full benchmark (cycles)**

| n | naive_O0 | opt_O3 | tiled_O3 | espdsp |
|---|---|---|---|---|
| 8x8 | 25,288 | 5,994 | 4,441 | 6,648 |
| 16x16 | 190,248 | 41,458 | 26,745 | 22,256 |
| 32x32 | 1,481,994 | 312,066 | 187,321 | 98,248 |
| 64x64 | 11,694,619 | 2,427,644 | 1,498,096 | 516,551 |
| 128x128 | 103,655,528 | 29,446,056 | 13,270,140 | 4,541,825 |
| 256x256 | 2,438,528,277 | 2,058,759,211 | 106,912,458 | 148,902,291 |

**Speedup over naive_O0 (i16)**

| n | opt_O3 | tiled_O3 | espdsp |
|---|---|---|---|
| 8x8 | 4.2x | 5.7x | 3.8x |
| 16x16 | 4.6x | 7.1x | 8.5x |
| 32x32 | 4.7x | 7.9x | 15.1x |
| 64x64 | 4.8x | 7.8x | 22.6x |
| 128x128 | 3.5x | 7.8x | 22.8x |
| 256x256 | 1.2x | 22.8x | 16.4x |

**256x256 across all types (cycles)**

| impl | i16 | i32 | f32 |
|---|---|---|---|
| naive_O0 | 2,438M | 2,715M | 2,439M |
| opt_O3 | 2,058M | 2,072M | 2,060M |
| tiled_O3 | 106M | 442M | 430M |
| espdsp | 148M | 2,072M | 2,072M |


## What the Numbers Say

**The compiler hits a wall at 256x256.** Up to 64x64, -O3 gives a consistent 4.7-4.8x speedup. At 256x256 the three matrices occupy 192 KB total, which cannot fit in 32 KB cache or even 512 KB SRAM, so almost every B[k][j] access becomes a PSRAM fetch. No amount of loop unrolling helps when the CPU is stalled waiting for memory. The speedup collapses to 1.2x.

**Cache-blocking is the biggest single win.** Tiling delivers 22.8x over naive and 19x over -O3 alone at 256x256. No new hardware, no special instructions — just a smarter access pattern. Memory layout mattered more than hardware acceleration. This is the result that surprised me most going into this.

**The PSRAM cliff is real and dramatic.** For sizes up to 128x128 the naive implementation scales almost exactly 8x per doubling of n, as expected for an O(n^3) algorithm. At 256x256 it scales 23.5x instead because the working set leaves SRAM entirely and every inner-loop access hits PSRAM. The cliff is visible in every implementation except tiled, which avoids it by design.

| Transition | naive i16 scaling | Expected |
|---|---|---|
| 8 to 16 | 7.5x | 8x |
| 16 to 32 | 7.8x | 8x |
| 32 to 64 | 7.9x | 8x |
| 64 to 128 | 8.9x | 8x |
| 128 to 256 | 23.5x | 8x |

**PIE SIMD gives 2.9x over tiling in the cache-resident regime.** At n=64 and n=128 where data fits in cache and the SIMD unit can be fed continuously, esp-dsp is 2.9x faster than tiled. At 128x128 this works out to 4.5M cycles vs 13.3M, completing in 18.9 ms at 240 MHz. That is real-time territory for small inference layers.

**But SIMD loses to tiling at 256x256.** At n=256, espdsp (148M cycles) is slower than tiled (106M cycles). The dot product function is called 65,536 times — once per output element — and the per-call overhead combined with PSRAM latency outweighs the benefit of the SIMD unit. A SIMD engine is only as fast as the data pipeline feeding it.

**i32 and f32 gain nothing from espdsp.** The PIE unit has no multiply-accumulate for 32-bit integers or floats, so espdsp i32 and f32 numbers are identical to opt_O3. For tiled_O3, i16 is about 4x faster than i32/f32 at 256x256 simply because smaller elements mean better cache utilization per tile.


## What This Means

The ESP32-S3 earns its AI label honestly but narrowly. For matrix sizes up to 128x128, the combination of cache-aware code and PIE SIMD delivers a 22.8x speedup over naive and completes in under 20 ms. That is fast enough for keyword spotting, gesture detection, anomaly sensing, and lightweight edge inference — exactly the applications Espressif designed it for.

Beyond that size, the chip hits a fundamental memory wall. PSRAM bandwidth cannot feed the CPU fast enough regardless of how clever the code is. This is not a criticism — the chip was never designed to run large models. It was designed for small always-on inference at minimal power and cost, and it does that well.

The deepest lesson here is architectural. On memory-constrained embedded systems the bottleneck is almost never the CPU. Optimizing memory access patterns gave a larger speedup than either compiler optimization or hardware SIMD. This is the memory wall problem that dominates real embedded AI systems, visible here in cycle-accurate measurements on a three dollar chip.


## Project Structure

```
main/
├── benchmark.h          # Timing macros, matrix typedefs, size constants
├── matmul_naive.c       # Triple-nested loop, compiled -O0
├── matmul_opt.c         # Same algorithm, compiled -O3
├── matmul_tiled.c       # Cache-blocked T=32, compiled -O3
├── matmul_espdsp.c      # PIE SIMD via esp-dsp dsps_dotprod_s16, compiled -O3
└── main.c               # Benchmark harness, dynamic allocation, median timing
CMakeLists.txt           # esp_lcd excluded to work around GCC 14.2.0 IRA pass bug
main/idf_component.yml   # esp-dsp dependency declaration
sdkconfig                # 32KB data cache, 240MHz, octal PSRAM 80MHz confirmed
```


## Build and Run

Requires ESP-IDF v5.x and an ESP32-S3 board with octal PSRAM.

```bash
idf.py set-target esp32s3
idf.py build flash monitor
```


## Build Issues Worth Documenting

**GCC 14.2.0 internal compiler error** — An IRA (Integrated Register Allocator) pass crash in `esp_lcd_panel_rgb.c` when combining -mdisable-hardware-atomics, octal PSRAM, and -Og. Fixed by excluding the esp_lcd component entirely in the root CMakeLists.txt. This is a known toolchain bug, not a project configuration error.

**Xtensa PIE inline assembly** — Multiple failed attempts to write custom PIE assembly before settling on esp-dsp. The Xtensa assembler rejects the `.option` directive (RISC-V syntax), and GCC inline asm cannot name Q registers as clobbers because they are TIE extension registers outside the standard constraint system. The esp-dsp library solves this cleanly and is the right choice for production use anyway.


## Environment

| Component | Version |
|---|---|
| ESP-IDF | v5.5.3 |
| Toolchain | xtensa-esp-elf GCC 14.2.0 |
| esp-dsp | ^1.0.0 |
| Target | esp32s3 |