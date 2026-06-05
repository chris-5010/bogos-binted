// bogo_hip2.cpp - HIP bogosort daemon for AMD Instinct (MI300X, MI355X)
// build: hipcc -O3 -std=c++17 --offload-arch=gfx942 bogo_hip2.cpp -o bogo_hip2
//        (check your arch: rocminfo | grep -o 'gfx[0-9]*' | grep -v gfx0 | head -1)
// daemon: echo "seed_lo seed_hi batch offset" | ./bogo_hip2 --daemon [--work-items N]
// note: wave64 on CDNA, so block sizes should be multiples of 64

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(e) do { hipError_t _r=(e); if(_r!=hipSuccess){ \
    fprintf(stderr,"HIP error %s at %s:%d\n",hipGetErrorString(_r),__FILE__,__LINE__); \
    exit(1);} } while(0)

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// chained seeding (1 hash/shuffle instead of 2), unrolled FY + scoring,
// best_arr in registers, uint8_t arrays (native on CDNA)
__device__ static __forceinline__ uint64_t sm64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ static __forceinline__ uint32_t rotl32(uint32_t x, uint32_t k) {
    return (x << k) | (x >> (32u - k));
}

struct Xo128 { uint32_t s0, s1, s2, s3; };

__device__ static __forceinline__ uint32_t xnext(Xo128 &st) {
    uint32_t res = rotl32(st.s0 + st.s3, 7u) + st.s0;
    uint32_t t   = st.s1 << 9u;
    st.s2 ^= st.s0; st.s3 ^= st.s1;
    st.s1 ^= st.s2; st.s0 ^= st.s3;
    st.s2 ^= t; st.s3 = rotl32(st.s3, 11u);
    return res;
}

// compile-time constant max → compiler replaces % with multiply-shift
#define XU(st,m) ({ \
    uint32_t _thr=(uint32_t)(0x100000000ULL%(uint64_t)(m)); \
    uint32_t _x; do{_x=xnext(st);}while(_x<_thr); \
    _x%(m); })

// pre-capture index to avoid evaluating XU twice (it advances RNG state)
#define FY(st,arr,i,m) { uint32_t _j=XU(st,m); \
    uint8_t _t=arr[i]; arr[i]=arr[_j]; arr[_j]=_t; }

// Per work-item result: 35 bytes = [0]=score, [1..25]=arr, [26..33]=index_le64, [34]=pad
#define STRIDE 35

__global__ void bogosort_batch(
    uint32_t seed_lo, uint32_t seed_hi,
    uint32_t batch_per_item, uint64_t index_offset,
    uint8_t * __restrict__ results
) {
    const uint64_t C    = 0x9e3779b97f4a7c15ULL;
    uint32_t gid        = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t seed64     = ((uint64_t)seed_hi << 32) | (uint64_t)seed_lo;
    uint64_t base       = index_offset + (uint64_t)gid * (uint64_t)batch_per_item;

    // Chained seeding
    uint64_t z  = seed64 + base * C + C;
    uint64_t ha = sm64(z); z += C;
    uint64_t hb = sm64(z); z += C;

    int      best_correct = -1;
    uint8_t  best_arr[25];
    uint32_t best_iter    = 0;

    for (uint32_t iter = 0; iter < batch_per_item; ++iter) {
        Xo128 st;
        st.s0 = (uint32_t)ha;        st.s1 = (uint32_t)(ha >> 32);
        st.s2 = (uint32_t)hb;        st.s3 = (uint32_t)(hb >> 32);
        if (!(st.s0 | st.s1 | st.s2 | st.s3)) st.s0 = 1;

        uint8_t arr[25];
        arr[ 0]= 1; arr[ 1]= 2; arr[ 2]= 3; arr[ 3]= 4; arr[ 4]= 5;
        arr[ 5]= 6; arr[ 6]= 7; arr[ 7]= 8; arr[ 8]= 9; arr[ 9]=10;
        arr[10]=11; arr[11]=12; arr[12]=13; arr[13]=14; arr[14]=15;
        arr[15]=16; arr[16]=17; arr[17]=18; arr[18]=19; arr[19]=20;
        arr[20]=21; arr[21]=22; arr[22]=23; arr[23]=24; arr[24]=25;

        // Fully unrolled Fisher-Yates
        FY(st,arr,24,25) FY(st,arr,23,24) FY(st,arr,22,23) FY(st,arr,21,22)
        FY(st,arr,20,21) FY(st,arr,19,20) FY(st,arr,18,19) FY(st,arr,17,18)
        FY(st,arr,16,17) FY(st,arr,15,16) FY(st,arr,14,15) FY(st,arr,13,14)
        FY(st,arr,12,13) FY(st,arr,11,12) FY(st,arr,10,11) FY(st,arr, 9,10)
        FY(st,arr, 8, 9) FY(st,arr, 7, 8) FY(st,arr, 6, 7) FY(st,arr, 5, 6)
        FY(st,arr, 4, 5) FY(st,arr, 3, 4) FY(st,arr, 2, 3) FY(st,arr, 1, 2)

        int correct =
            (arr[ 0]== 1)+(arr[ 1]== 2)+(arr[ 2]== 3)+(arr[ 3]== 4)+(arr[ 4]== 5)+
            (arr[ 5]== 6)+(arr[ 6]== 7)+(arr[ 7]== 8)+(arr[ 8]== 9)+(arr[ 9]==10)+
            (arr[10]==11)+(arr[11]==12)+(arr[12]==13)+(arr[13]==14)+(arr[14]==15)+
            (arr[15]==16)+(arr[16]==17)+(arr[17]==18)+(arr[18]==19)+(arr[19]==20)+
            (arr[20]==21)+(arr[21]==22)+(arr[22]==23)+(arr[23]==24)+(arr[24]==25);

        if (correct > best_correct) {
            best_correct = correct;
            best_iter    = iter;
            for (int k = 0; k < 25; ++k) best_arr[k] = arr[k];
            if (correct == 25) break;
        }

        ha = hb;
        hb = sm64(z);
        z += C;
    }

    uint64_t best_index = base + (uint64_t)best_iter;
    uint64_t base_out   = (uint64_t)gid * STRIDE;
    results[base_out] = (best_correct < 0) ? 0xFF : (uint8_t)best_correct;
    for (int k = 0; k < 25; ++k) results[base_out + 1 + k] = best_arr[k];
    for (int k = 0; k < 8;  ++k)
        results[base_out + 26 + k] = (uint8_t)((best_index >> (8*k)) & 0xff);
    results[base_out + 34] = 0;
}

struct BatchResult { int best_correct; uint8_t best_arr[25]; uint64_t best_index; };

struct GpuCtx {
    uint8_t *dev_results = nullptr;
    uint8_t *host_results = nullptr;
    size_t   work_items  = 0;
    int      block_size  = 256;
};

static void init_gpu(GpuCtx &ctx, size_t work_items, int block_size) {
    ctx.work_items = work_items;
    ctx.block_size = block_size;

    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    fprintf(stderr, "[hip] device:  %s\n", prop.name);
    fprintf(stderr, "[hip] CUs:     %d\n", prop.multiProcessorCount);

    size_t buf = work_items * STRIDE;
    HIP_CHECK(hipMalloc(&ctx.dev_results, buf));
    HIP_CHECK(hipHostMalloc(&ctx.host_results, buf, hipHostMallocDefault));
    HIP_CHECK(hipMemset(ctx.dev_results, 0xFF, buf));
}

static BatchResult run_dispatch(GpuCtx &ctx,
                                uint32_t seed_lo, uint32_t seed_hi,
                                uint32_t batch_per_item, uint64_t index_offset) {
    uint32_t blocks = (uint32_t)((ctx.work_items + ctx.block_size - 1) / ctx.block_size);
    bogosort_batch<<<blocks, ctx.block_size>>>(
        seed_lo, seed_hi, batch_per_item, index_offset, ctx.dev_results);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(ctx.host_results, ctx.dev_results,
                        ctx.work_items * STRIDE, hipMemcpyDeviceToHost));

    BatchResult best; best.best_correct = -1; best.best_index = 0;
    memset(best.best_arr, 0, 25);
    for (size_t i = 0; i < ctx.work_items; ++i) {
        uint8_t sb = ctx.host_results[i * STRIDE];
        if (sb == 0xFF) continue;
        int c = (int)sb;
        if (c > best.best_correct) {
            best.best_correct = c;
            memcpy(best.best_arr, ctx.host_results + i*STRIDE + 1, 25);
            uint64_t idx = 0;
            for (int b = 0; b < 8; ++b)
                idx |= (uint64_t)ctx.host_results[i*STRIDE+26+b] << (8*b);
            best.best_index = idx;
        }
    }
    return best;
}

static void daemon_loop(GpuCtx &ctx) {
    fprintf(stderr, "[hip-daemon] ready  work_items=%zu  block_size=%d\n",
            ctx.work_items, ctx.block_size);
    fflush(stderr);
    char line[512];
    while (fgets(line, sizeof(line), stdin)) {
        size_t len = strlen(line);
        while (len && (line[len-1]=='\n'||line[len-1]=='\r')) line[--len]=0;
        if (!strncmp(line,"quit",4)) break;
        unsigned long long sl, sh, batch, offset;
        if (sscanf(line, "%llu %llu %llu %llu", &sl, &sh, &batch, &offset) != 4) {
            fprintf(stderr, "[hip-daemon] bad line: '%s'\n", line); continue;
        }
        uint32_t bpi = (uint32_t)((batch + ctx.work_items - 1) / ctx.work_items);
        uint64_t actual = (uint64_t)ctx.work_items * bpi;
        double t0 = now_s();
        BatchResult r = run_dispatch(ctx, (uint32_t)sl, (uint32_t)sh,
                                     bpi, (uint64_t)offset);
        double elapsed = now_s() - t0;
        printf("{\"best_correct\":%d,\"best_arr\":[", r.best_correct);
        for (int i = 0; i < 25; i++) printf("%s%d", i?",":"", r.best_arr[i]);
        printf("],\"best_index\":%llu,\"total_done\":%llu,\"elapsed\":%.6f,\"rate\":%.0f}\n",
               (unsigned long long)r.best_index, (unsigned long long)actual,
               elapsed, elapsed > 0 ? actual/elapsed : 0.0);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    bool   daemon_mode = false;
    size_t work_items  = 1048576;   // good starting point for 304-CU MI355X
    int    block_size  = 512;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--daemon")                    daemon_mode = true;
        else if (a == "--work-items" && i+1<argc)    work_items  = std::stoull(argv[++i]);
        else if (a == "--block-size" && i+1<argc)    block_size  = std::stoi(argv[++i]);
        else if (!a.empty() && a[0] != '-')          pos.push_back(a);
    }
    work_items = ((work_items + block_size - 1) / block_size) * block_size;

    GpuCtx ctx;
    init_gpu(ctx, work_items, block_size);

    if (daemon_mode) { daemon_loop(ctx); return 0; }

    // One-shot benchmark
    uint32_t seed_lo = pos.size()>0 ? (uint32_t)std::stoull(pos[0]) : 0x12345678u;
    uint32_t seed_hi = pos.size()>1 ? (uint32_t)std::stoull(pos[1]) : 0xdeadbeef;
    uint64_t total   = pos.size()>2 ? std::stoull(pos[2]) : 2'000'000'000ULL;
    uint32_t bpi     = (uint32_t)((total + work_items - 1) / work_items);
    double t0 = now_s();
    BatchResult r = run_dispatch(ctx, seed_lo, seed_hi, bpi, 0);
    double elapsed = now_s() - t0;
    uint64_t seed64 = ((uint64_t)seed_hi << 32) | seed_lo;
    printf("{\n  \"seed\":\"%llu\",\n  \"best_correct\":%d,\n  \"best_index\":%llu,\n  \"best_arr\":[",
           (unsigned long long)seed64, r.best_correct, (unsigned long long)r.best_index);
    for (int i = 0; i < 25; i++) printf("%s%d", i?",":"", r.best_arr[i]);
    printf("],\n  \"elapsed\":%.6f,\n  \"rate\":%.0f\n}\n",
           elapsed, (uint64_t)work_items * bpi / elapsed);
    return 0;
}
