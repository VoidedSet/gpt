# CPU Performance Profiling & Optimization Metrics

This log records the performance metrics and memory cache profiling of the C++ GPT-2 training engine on the CPU before and after applying optimizations.

---

## 1. Metric Comparison

| Metric | Unoptimized (Sequential CPU) | Optimized (OpenMP + AVX2 FMA) | Speedup / Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | layers=4, dim=128, heads=4 | layers=4, dim=128, heads=4 | - |
| **Average Latency** | ~575 ms/step | **~458 ms/step** | **~21% speedup** |
| **Throughput** | ~440 tokens/sec | **~550 tokens/sec** | **~25% higher throughput** |
| **Total Training Time** | ~349.8 seconds | **~282.9 seconds** | **~19.1% time saved** |
| **Cache Misses** | 1,408,018,276 | **1,113,534,770** | **~20.9% fewer misses** (294M saved) |
| **L1-dcache Load Misses** | 26,177,696,806 | **25,964,577,992** | Slight reduction (~0.8%) |
| **CPU Utilization** | ~8.3% (1 core active) | **~30%** (12 threads shared) | Multi-core scaling active |

---

## 2. Optimizations Applied

1. **Bypassing Indexing Overhead:** Removed high-overhead `operator()` stride/boundary calculation calls inside the inner loops of `matmul` and `matmul_transposed_b`, replacing them with raw float pointers and contiguous index increments.
2. **AVX2 SIMD Vectorization:** Utilized 256-bit Intel registers to execute 8 floating-point multiplications and additions simultaneously in vector registers.
3. **Fused Multiply-Add (FMA):** Applied `_mm256_fmadd_ps` to calculate dot products directly on CPU hardware registers, avoiding repeated L1 cache read/write roundtrips.
4. **OpenMP Multi-threading:** Used `#pragma omp parallel for` compiler directives to distribute matrix row computations across all available CPU threads.

---

## 3. Raw Profiling Output (Optimized Run)

```text
kshayik@kshayiks-hp-victus:/media/kshayik/New Volume/Projects/GPT$ perf stat -e cache-misses,L1-dcache-load-misses ./build/engine
[*] Running GPT Training Loop...
[+] Loaded dataset/input.txt (1115394 bytes)
[+] Vocab size: 65 unique characters
[+] Encoded 1115394 tokens.
Creating GPT Model (vocab_size=65, layers=4, dim=128)...
Number of learnable parameters: 52

--- Generating with untrained model ---
The .j-PDi3MLf-v
gC
lFJriH;e'JN!l&muovK'dUU$?c
BICC-Xa Fx?x$ZM,KQuGHqeNRBwXOaZgxqv,MW.gVIeZv&gHWbbrRGFzT
---------------------------------------

[*] Training starting...
  Step 0 | Loss: 4.17317 | Speed: 466.152 ms/step (549.177 tok/sec)
  Step 50 | Loss: 3.12853 | Speed: 453.732 ms/step (564.209 tok/sec)
  Step 100 | Loss: 2.74551 | Speed: 493.791 ms/step (518.438 tok/sec)
  Step 150 | Loss: 2.56544 | Speed: 476.19 ms/step (537.6 tok/sec)
  Step 200 | Loss: 2.52312 | Speed: 467.68 ms/step (547.383 tok/sec)
  Step 250 | Loss: 2.51062 | Speed: 469.054 ms/step (545.78 tok/sec)
  Step 300 | Loss: 2.43287 | Speed: 466.429 ms/step (548.85 tok/sec)

  --- [Step 300] Intermediate Generation snippet ---
The dous prves ay pfoferthel,

LT
viovend qe y a teryelrdoneve he awtendilit s gli l
  -----------------------------------------------------

  Step 350 | Loss: 2.36968 | Speed: 463.444 ms/step (552.386 tok/sec)
  Step 400 | Loss: 2.34493 | Speed: 458.013 ms/step (558.936 tok/sec)
  Step 450 | Loss: 2.4893 | Speed: 456.649 ms/step (560.606 tok/sec)
  Step 500 | Loss: 2.16121 | Speed: 456.137 ms/step (561.234 tok/sec)
  Step 550 | Loss: 2.31568 | Speed: 458.473 ms/step (558.375 tok/sec)
  Step 599 | Loss: 2.25733 | Speed: 464.985 ms/step (550.555 tok/sec)

  --- [Step 599] Intermediate Generation snippet ---
The cong tece thate
E many to ach pplt: s conqurgopy vees d cher h, fotrvishiat t it
  -----------------------------------------------------


[+] Training completed in 282.894 seconds.

--- Generating with final trained model ---
The bor rgrovee tou ont, prcos;
Wan ouge, I ch w avuss, llla y w yser t tiichey'sur tan slenewisurerigouuth he ter if sig mo arwive?
Hue, me, oven o y d t ave
heryh ys, h w tin ban he erebr t th y anefree
-------------------------------------

[+] Main Completed in 286.09 seconds.

 Performance counter stats for './build/engine':

     1,113,534,770      cache-misses                                                          
    25,964,577,992      L1-dcache-load-misses                                                 

     286.096414105 seconds time elapsed

    1214.753786000 seconds user
       2.004785000 seconds sys
```
