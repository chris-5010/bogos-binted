/*
 * bogo_hip.cpp — HIP-native bogosort compute engine for AMD GPUs (ROCm)
 *
 * Same seeding/shuffle logic as bogo_gpu.cpp (OpenCL) — fully verifiable
 * by the server. HIP compiles directly to RDNA ISA, no OpenCL translation.
 *
 * Build:
 *   hipcc -O3 -std=c++17 bogo_hip.cpp -o bogo_hip
 *
 * Two modes:
 *
 *   One-shot:
 *     ./bogo_hip [seed_lo] [seed_hi] [total] [work_items] [index_offset]
 *
 *   Daemon (persistent — use this with bogo_bridge.py --binary ./bogo_hip):
 *     ./bogo_hip --daemon [--work-items N] [--block-size N]
 *     Reads "seed_lo seed_hi batch index_offset\n" from stdin.
 *     Writes one JSON result line to stdout per request.
 *     Send "quit\n" to exit.
 *
 * Tuning for 7900 XTX (gfx1100, 96 CUs):
 *   --work-items 655360  --block-size 256   (default)
 *   Try 1310720 / 512 for higher occupancy if temps allow.
 */

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                     \
    do {                                                                     \
        hipError_t _e = (expr);                                              \
        if (_e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s at %s:%d\n",                      \
                    hipGetErrorString(_e), __FILE__, __LINE__);              \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ── Device-side RNG (identical to JS worker + OpenCL kernel) ─────────────

struct Xo128State { uint32_t s[4]; };

__device__ __forceinline__ Xo128State xo_seed(uint64_t si) {
    uint64_t z, a, b;
    z = si + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    a = z ^ (z >> 31);
    z = si + 2ULL * 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    b = z ^ (z >> 31);
    Xo128State st;
    st.s[0] = (uint32_t)(a & 0xffffffffULL); st.s[1] = (uint32_t)(a >> 32);
    st.s[2] = (uint32_t)(b & 0xffffffffULL); st.s[3] = (uint32_t)(b >> 32);
    if (!st.s[0] && !st.s[1] && !st.s[2] && !st.s[3]) st.s[0] = 1;
    return st;
}

__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t k) {
    return (x << k) | (x >> (32u - k));
}

__device__ __forceinline__ uint32_t xo_next(Xo128State *st) {
    uint32_t res = rotl32(st->s[0] + st->s[3], 7u) + st->s[0];
    uint32_t t   = st->s[1] << 9u;
    st->s[2] ^= st->s[0]; st->s[3] ^= st->s[1];
    st->s[1] ^= st->s[2]; st->s[0] ^= st->s[3];
    st->s[2] ^= t; st->s[3] = rotl32(st->s[3], 11u);
    return res;
}

__device__ __forceinline__ uint32_t xo_uint(Xo128State *st, uint32_t max) {
    // Rejection sampling — matches JS xint() exactly (required for server verify)
    uint32_t threshold = (uint32_t)(0x100000000ULL % (uint64_t)max);
    uint32_t x;
    do { x = xo_next(st); } while (x < threshold);
    return x % max;
}

// ── Kernel ────────────────────────────────────────────────────────────────
// Result layout per thread: 35 bytes
//   [0]      best_correct (0xFF = no result)
//   [1..25]  best_arr
//   [26..33] best_index (little-endian u64)
//   [34]     padding

__global__ void bogosort_batch(
    uint32_t seed_lo, uint32_t seed_hi,
    uint32_t batch_per_item, uint64_t index_offset,
    uint8_t * __restrict__ results
) {
    const int N = 25;
    uint64_t gid     = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t seed64  = ((uint64_t)seed_hi << 32) | (uint64_t)seed_lo;

    int      best_correct = -1;
    uint8_t  best_arr[25];
    uint64_t best_index   = 0;

    #pragma unroll 1
    for (uint32_t iter = 0; iter < batch_per_item; iter++) {
        uint64_t global_index = index_offset + gid * (uint64_t)batch_per_item + iter;
        uint64_t si = seed64 + global_index * 0x9e3779b97f4a7c15ULL;

        Xo128State st = xo_seed(si);

        uint8_t arr[25];
        #pragma unroll
        for (int i = 0; i < N; i++) arr[i] = (uint8_t)(i + 1);

        // Fisher-Yates — must match JS/WASM exactly
        #pragma unroll 1
        for (int i = N - 1; i > 0; i--) {
            uint32_t j = xo_uint(&st, (uint32_t)(i + 1));
            uint8_t tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        }

        int correct = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) if (arr[i] == (uint8_t)(i + 1)) correct++;

        if (correct > best_correct) {
            best_correct = correct;
            best_index   = global_index;
            #pragma unroll
            for (int i = 0; i < N; i++) best_arr[i] = arr[i];
            if (correct == N) break;
        }
    }

    uint64_t base = gid * 35;
    results[base] = (best_correct < 0) ? 0xFF : (uint8_t)best_correct;
    #pragma unroll
    for (int i = 0; i < 25; i++) results[base + 1 + i] = best_arr[i];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        results[base + 26 + i] = (uint8_t)((best_index >> (8 * i)) & 0xff);
    results[base + 34] = 0;
}

// ── GPU state (kept alive across daemon dispatches) ───────────────────────

struct GpuState {
    uint8_t *d_results = nullptr;   // device result buffer
    uint8_t *h_results = nullptr;   // pinned host result buffer (fast DMA)
    size_t   work_items  = 0;
    int      block_size  = 256;
    bool     unified     = false;
};

static void init_gpu(GpuState &g, size_t work_items, int block_size) {
    g.work_items = work_items;
    g.block_size = block_size;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    g.unified = prop.unifiedAddressing && (prop.totalGlobalMem == 0 ||
                strstr(prop.name, "Radeon RX") == nullptr);
    fprintf(stderr, "[hip] device:   %s\n", prop.name);
    fprintf(stderr, "[hip] unified:  %s\n", g.unified ? "yes (iGPU)" : "no (discrete)");
    fprintf(stderr, "[hip] CUs:      %d\n", prop.multiProcessorCount);

    size_t buf = work_items * 35;
    HIP_CHECK(hipMalloc(&g.d_results, buf));
    // Pinned host memory for fast DMA transfer
    HIP_CHECK(hipHostMalloc(&g.h_results, buf, hipHostMallocDefault));
}

static void destroy_gpu(GpuState &g) {
    if (g.d_results) hipFree(g.d_results);
    if (g.h_results) hipHostFree(g.h_results);
}

// ── One dispatch ──────────────────────────────────────────────────────────

struct BatchResult {
    int      best_correct;
    uint8_t  best_arr[25];
    uint64_t best_index;
};

static BatchResult run_dispatch(GpuState &g,
                                uint32_t seed_lo, uint32_t seed_hi,
                                uint32_t batch_per_item, uint64_t index_offset) {
    int   grid = (int)((g.work_items + g.block_size - 1) / g.block_size);

    bogosort_batch<<<grid, g.block_size>>>(
        seed_lo, seed_hi, batch_per_item, index_offset, g.d_results);

    // Copy results back via pinned memory (async-then-sync for overlap)
    HIP_CHECK(hipMemcpy(g.h_results, g.d_results,
                        g.work_items * 35, hipMemcpyDeviceToHost));

    BatchResult best; best.best_correct = -1; best.best_index = 0;
    memset(best.best_arr, 0, 25);

    for (size_t i = 0; i < g.work_items; i++) {
        uint8_t sb = g.h_results[i * 35];
        if (sb == 0xFF) continue;
        int c = (int)sb;
        if (c > best.best_correct) {
            best.best_correct = c;
            memcpy(best.best_arr, &g.h_results[i * 35 + 1], 25);
            uint64_t idx = 0;
            for (int b = 0; b < 8; b++)
                idx |= (uint64_t)g.h_results[i * 35 + 26 + b] << (8 * b);
            best.best_index = idx;
        }
    }
    return best;
}

// ── Daemon loop ───────────────────────────────────────────────────────────

static void daemon_loop(GpuState &g) {
    fprintf(stderr, "[hip-daemon] ready  work_items=%zu  block_size=%d\n",
            g.work_items, g.block_size);
    fflush(stderr);

    char line[512];
    while (fgets(line, sizeof(line), stdin)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (strncmp(line, "quit", 4) == 0) break;

        unsigned long long sl, sh, batch_req, offset;
        if (sscanf(line, "%llu %llu %llu %llu", &sl, &sh, &batch_req, &offset) != 4) {
            fprintf(stderr, "[hip-daemon] bad line: '%s'\n", line);
            fflush(stderr);
            continue;
        }

        uint32_t seed_lo       = (uint32_t)sl;
        uint32_t seed_hi       = (uint32_t)sh;
        uint64_t index_offset  = (uint64_t)offset;
        uint32_t batch_per_item = (uint32_t)((batch_req + g.work_items - 1) / g.work_items);
        uint64_t actual = (uint64_t)g.work_items * batch_per_item;

        double t0 = now_s();
        BatchResult r = run_dispatch(g, seed_lo, seed_hi, batch_per_item, index_offset);
        double elapsed = now_s() - t0;

        printf("{\"best_correct\":%d,\"best_arr\":[", r.best_correct);
        for (int i = 0; i < 25; i++) printf("%s%d", i?",":"", r.best_arr[i]);
        printf("],\"best_index\":%llu,\"total_done\":%llu,\"elapsed\":%.6f,\"rate\":%.0f}\n",
               (unsigned long long)r.best_index,
               (unsigned long long)actual,
               elapsed,
               elapsed > 0 ? actual / elapsed : 0.0);
        fflush(stdout);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    bool        daemon_mode    = false;
    size_t      work_items_flag = 0;
    int         block_size      = 256;
    std::vector<std::string> pos_args;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--daemon")                    { daemon_mode = true; }
        else if (a == "--work-items" && i+1 < argc)  { work_items_flag = std::stoull(argv[++i]); }
        else if (a == "--block-size" && i+1 < argc)  { block_size = std::stoi(argv[++i]); }
        else if (!a.empty() && a[0] != '-')          { pos_args.push_back(a); }
    }

    uint32_t seed_lo          = pos_args.size() > 0 ? (uint32_t)std::stoull(pos_args[0]) : 0x12345678u;
    uint32_t seed_hi          = pos_args.size() > 1 ? (uint32_t)std::stoull(pos_args[1]) : 0xdeadbeef;
    uint64_t total_target     = pos_args.size() > 2 ? std::stoull(pos_args[2])            : 10'000'000ULL;
    size_t   work_items_pos   = pos_args.size() > 3 ? std::stoull(pos_args[3])            : 0;
    uint64_t index_offset_pos = pos_args.size() > 4 ? std::stoull(pos_args[4])            : 0;

    // Default work_items: tuned for 7900 XTX (96 CUs × 64 × 4 waves × 4 = 98304 minimum,
    // use 655360 for deep queuing and hide memory latency)
    size_t work_items = work_items_flag ? work_items_flag :
                        work_items_pos  ? work_items_pos  : 655360;
    // Align to block size
    work_items = ((work_items + block_size - 1) / block_size) * block_size;

    GpuState g;
    init_gpu(g, work_items, block_size);

    if (daemon_mode) {
        daemon_loop(g);
        destroy_gpu(g);
        return 0;
    }

    // One-shot mode
    uint32_t batch_per_item = (uint32_t)((total_target + work_items - 1) / work_items);
    uint64_t actual_total   = (uint64_t)work_items * batch_per_item;
    uint64_t index_offset   = index_offset_pos;

    fprintf(stderr, "[hip] work_items=%zu  batch_per_item=%u  total=%llu\n",
            work_items, batch_per_item, (unsigned long long)actual_total);

    uint64_t shuffles_done = 0, dispatches = 0;
    int      global_best  = -1;
    uint8_t  global_arr[25] = {};
    uint64_t global_index = 0;
    double   t0 = now_s();

    while (shuffles_done < actual_total) {
        BatchResult r = run_dispatch(g, seed_lo, seed_hi, batch_per_item, index_offset);
        if (r.best_correct > global_best) {
            global_best  = r.best_correct;
            global_index = r.best_index;
            memcpy(global_arr, r.best_arr, 25);
        }
        uint64_t bt = (uint64_t)work_items * batch_per_item;
        index_offset  += bt;
        shuffles_done += bt;
        dispatches++;
        if (dispatches % 16 == 0) {
            double el = now_s() - t0;
            fprintf(stderr, "\r[hip] %.2fM done  %.0fM/s  best=%d/25   ",
                    shuffles_done/1e6, shuffles_done/el/1e6, global_best);
            fflush(stderr);
        }
        if (global_best == 25) break;
    }

    double elapsed = now_s() - t0;
    fprintf(stderr, "\n[hip] done. %llu shuffles in %.3fs = %.0fM/s\n",
            (unsigned long long)shuffles_done, elapsed, shuffles_done/elapsed/1e6);

    uint64_t seed64 = ((uint64_t)seed_hi << 32) | seed_lo;
    printf("{\n");
    printf("  \"seed\": \"%llu\",\n",     (unsigned long long)seed64);
    printf("  \"total_done\": %llu,\n",   (unsigned long long)total_target);
    printf("  \"best_correct\": %d,\n",   global_best);
    printf("  \"best_index\": %llu,\n",   (unsigned long long)global_index);
    printf("  \"best_arr\": [");
    for (int i = 0; i < 25; i++) printf("%s%d", i?",":"", global_arr[i]);
    printf("],\n");
    printf("  \"elapsed\": %.6f,\n",      elapsed);
    printf("  \"rate\": %.0f,\n",         shuffles_done/elapsed);
    printf("  \"unified_memory\": %s\n",  g.unified ? "true" : "false");
    printf("}\n");

    destroy_gpu(g);
    return 0;
}
