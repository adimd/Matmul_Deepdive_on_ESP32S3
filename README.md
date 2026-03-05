# Matmul Deepdive on ESP32-S3

A ground-up benchmarking study of matrix multiplication on the ESP32-S3, investigating whether Espressif's "AI-capable" claim holds up under real measurement. Four implementations are built from scratch — each one exposing a different layer of the hardware — and measured with cycle-accurate timing on real silicon.

---

## The Question

Espressif markets the ESP32-S3 as an AI-capable chip. It costs about three dollars. It has a dedicated SIMD unit called PIE (Processor Instruction Extensions) that can multiply eight 16-bit numbers simultaneously in a single clock cycle. The question this project asks is simple: does any of that actually matter in practice, and if so, when and why?

---

## Hardware

| Property | Value |
|---|---|
| Board | XH-S3E N16R8 |
| Chip | ESP32-S3R8 (Xtensa LX7 dual-core) |
| Clock | 240 MHz |
| Internal SRAM | 512 KB |
| L1 Data Cache | 32 KB, 8-way set associative, 32-byte lines |
| L1 Instruction Cache | 16 KB |
| External PSRAM | 8 MB Octal SPI @ 80 MHz |
| Flash | 16 MB |

The memory hierarchy is the central character of this story. Internal SRAM is fast. The L1 cache is a 32 KB window into that fast world. PSRAM is 8 MB but sits on an external bus and is roughly 10x slower to access. Everything interesting in these benchmarks comes from data moving between these three levels.

---

## Implementations

Four implementations of square matrix multiplication (C = A × B), each isolating one variable.

### 1. `matmul_naive` — compiled at `-O0`

The textbook triple-nested loop with no tricks. B is accessed in column-major order (B[k][j]), which strides across memory and causes a cache miss on almost every inner-loop step. Compiled without optimization so the compiler adds nothing. This is the honest baseline.

```c
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            C[i*n+j] += A[i*n+k] * B[k*n+j];  // B access is cache-unfriendly
```

### 2. `matmul_opt` — compiled at `-O3`

Identical algorithm to naive, but compiled with full compiler optimization. Loop unrolling, instruction scheduling, register allocation. This isolates exactly what the compiler can contribute without any algorithmic changes.

### 3. `matmul_tiled` — compiled at `-O3`

Cache-blocking: the matrices are divided into 32×32 tiles and the computation processes one tile-triple at a time. The inner loop only touches data already resident in cache, eliminating PSRAM stall cycles.

```c
#define TILE 32
// Tile-triple footprint: 3 × 32² × 2 = 6 KB (i16) — fits in usable cache
for (int ii = 0; ii < n; ii += TILE)
    for (int jj = 0; jj < n; jj += TILE)
        for (int kk = 0; kk < n; kk += TILE)
            // inner 3 loops operate entirely within one tile
```

**Why T=32 and not T=64?** The data cache is 32 KB on paper, and a T=64 tile-triple for i16 uses 24 KB, which looks like it fits. Benchmarking proved this wrong — T=64 was 2.3× slower than T=32 at n=256. The cache is shared with FreeRTOS, benchmark harness code, the stack, and program instructions. The effective usable data cache is well below 32 KB in a real firmware environment. T=32 uses only 6 KB per tile-triple and stays comfortably resident regardless of system overhead. This is why you measure instead of just calculate.

### 4. `matmul_espdsp` — PIE SIMD via Espressif's esp-dsp library

Uses Espressif's production-grade `dsps_dotprod_s16` which internally uses `EE.VMULAS.S16.QACC` — the PIE instruction that accumulates 8 lane-wise products per clock cycle. B is transposed first so both row vectors are contiguous in memory, enabling SIMD-friendly sequential loads.

Only i16 benefits from PIE SIMD. i32 and f32 have no equivalent PIE path and fall back to scalar.

---

## Results

All measurements taken at 240 MHz. Median of 10 runs. Cycle counter via CCOUNT register cross-validated against `esp_timer`.

### i16 — Full Benchmark Table (cycles)

| n | naive_O0 | opt_O3 | tiled_O3 | espdsp |
|---|---|---|---|---|
| 8×8 | 25,288 | 5,994 | 4,441 | 6,648 |
| 16×16 | 190,248 | 41,458 | 26,745 | 22,256 |
| 32×32 | 1,481,994 | 312,066 | 187,321 | 98,248 |
| 64×64 | 11,694,619 | 2,427,644 | 1,498,096 | 516,551 |
| 128×128 | 103,655,528 | 29,446,056 | 13,270,140 | 4,541,825 |
| 256×256 | 2,438,528,277 | 2,058,759,211 | 106,912,458 | 148,902,291 |

### Speedup over naive_O0 (i16)

| n | opt_O3 | tiled_O3 | espdsp |
|---|---|---|---|
| 8×8 | 4.2× | 5.7× | 3.8× |
| 16×16 | 4.6× | 7.1× | 8.5× |
| 32×32 | 4.7× | 7.9× | 15.1× |
| 64×64 | 4.8× | 7.8× | 22.6× |
| 128×128 | 3.5× | 7.8× | **22.8×** |
| 256×256 | 1.2× | **22.8×** | 16.4× |

### dtype comparison at 256×256

| impl | i16 | i32 | f32 |
|---|---|---|---|
| naive_O0 | 2,438M | 2,715M | 2,439M |
| opt_O3 | 2,058M | 2,072M | 2,060M |
| tiled_O3 | 106M | 442M | 430M |
| espdsp | **148M** | 2,072M | 2,072M |

---

## Key Findings

### 1. The compiler hits a wall at 256×256 (1.2× speedup)

`-O3` gives a genuine 4.7–4.8× speedup for sizes up to 64×64 where data fits in cache. At 256×256 the three matrices are 192 KB — they cannot fit in the 32 KB L1 cache or 512 KB internal SRAM — so almost every B[k][j] access becomes a PSRAM fetch. No amount of loop unrolling or instruction scheduling helps when the CPU is stalled waiting for memory. The speedup collapses to 1.2×.

### 2. Cache-blocking is the biggest single win (22.8× over naive at 256×256)

By reorganizing the computation into 32×32 tiles that fit in cache, tiling nearly eliminates PSRAM stalls entirely. At 256×256 it is 19.3× faster than `-O3` alone and 22.8× faster than the naive baseline. No new hardware, no special instructions — just a smarter access pattern. **Memory layout mattered more than hardware acceleration.**

### 3. The PSRAM cliff — an unexpected scaling discontinuity

For sizes up to 128×128 the naive implementation scales almost exactly 8× per doubling of n, as expected for an O(n³) algorithm. At 256×256 it scales 23.5× instead. This is the moment the working set leaves SRAM entirely and every inner-loop access hits PSRAM. The cliff is visible in every implementation except tiled, which avoids it by design.

| Transition | naive i16 scaling | Expected |
|---|---|---|
| 8→16 | 7.5× | 8× |
| 16→32 | 7.8× | 8× |
| 32→64 | 7.9× | 8× |
| 64→128 | 8.9× | 8× |
| 128→256 | **23.5×** | 8× |

### 4. PIE SIMD gives 2.9× over tiling in the cache-resident regime

For n=64 and n=128, where data fits in cache and the SIMD unit can be fed continuously, esp-dsp's PIE implementation is 2.9× faster than tiled. At 128×128 this means 4.5M cycles vs 13.3M — completing in 18.9 ms at 240 MHz. That is real-time territory for small neural network inference layers.

### 5. PIE SIMD loses to tiling at 256×256

At n=256 the espdsp implementation (148M cycles) is slower than tiled (106M cycles). The esp-dsp dot product function is called once per output element — 65,536 times for a 256×256 matrix. The per-call overhead and the fact that the transposed Bt matrix is not 16-byte aligned combine to outweigh the SIMD benefit when data is in PSRAM. The SIMD unit is only as fast as the data pipeline feeding it.

### 6. i32 and f32 SIMD gain nothing

The PIE unit has no multiply-accumulate instruction for 32-bit integers or floats. The espdsp i32 and f32 numbers are identical to opt_O3 — confirming the SIMD path is i16-only. For tiled_O3, i16 is 4× faster than i32/f32 at 256×256 because the smaller element size means better cache utilization per tile.

---

## What This Means

The ESP32-S3 earns its AI label honestly but narrowly. For small matrix sizes up to 128×128 the combination of cache-aware code and PIE SIMD delivers a 22.8× speedup over naive and completes in under 20 ms. That is fast enough for keyword spotting, gesture detection, anomaly sensing, and lightweight edge inference — exactly the applications Espressif designed it for.

Beyond that size, the chip runs into a fundamental memory wall. The PSRAM bandwidth cannot feed the CPU fast enough regardless of how clever the code is. The ESP32-S3 is not a competitor to dedicated edge AI accelerators or even a Raspberry Pi for larger models. It was never meant to be.

The deepest lesson is architectural: on memory-constrained embedded systems, the bottleneck is almost never the CPU. Optimizing memory access patterns gave a larger speedup than either compiler optimization or hardware SIMD. This is the memory wall problem that dominates real embedded AI systems, visible here in cycle-accurate measurements on a three dollar chip.

---

## Project Structure

```
main/
├── benchmark.h          # Timing macros, matrix typedefs, size constants
├── matmul_naive.c       # Triple-nested loop, compiled -O0
├── matmul_opt.c         # Same algorithm, compiled -O3
├── matmul_tiled.c       # Cache-blocked T=32, compiled -O3
├── matmul_espdsp.c      # PIE SIMD via esp-dsp dsps_dotprod_s16, compiled -O3
└── main.c               # Benchmark harness, dynamic allocation, median timing
CMakeLists.txt           # esp_lcd excluded (GCC 14.2.0 IRA pass bug workaround)
main/idf_component.yml   # esp-dsp dependency
sdkconfig                # 32KB data cache, 240MHz, octal PSRAM 80MHz
```

---

## Build and Run

Requires ESP-IDF v5.x and an ESP32-S3 board with octal PSRAM.

```bash
idf.py set-target esp32s3
idf.py build flash monitor
```

---

## Notable Build Issues Encountered

**GCC 14.2.0 internal compiler error** — An IRA (Integrated Register Allocator) pass crash in `esp_lcd_panel_rgb.c` when combining `-mdisable-hardware-atomics`, octal PSRAM, and `-Og`. Fixed by excluding the `esp_lcd` component entirely via `set(EXCLUDE_COMPONENTS "esp_lcd")` in the root CMakeLists.txt.

**Xtensa PIE inline assembly** — The Xtensa assembler does not support the `.option` directive (that is RISC-V syntax). GCC's inline asm constraint system does not recognize Q registers (TIE extension registers) as named clobbers. After several failed approaches, the decision was made to use Espressif's battle-tested `esp-dsp` library which provides verified PIE SIMD through a clean C API, rather than continuing to fight the toolchain.

---

## Environment

| Component | Version |
|---|---|
| ESP-IDF | v5.5.3 |
| Toolchain | xtensa-esp-elf GCC 14.2.0 |
| esp-dsp | ^1.0.0 |
| Target | esp32s3 |