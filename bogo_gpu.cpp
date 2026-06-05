// bogo_gpu.cpp - OpenCL bogosort compute engine
// build: g++ -O2 -std=c++17 bogo_gpu.cpp -o bogo_gpu -lOpenCL
// daemon mode: echo "seed_lo seed_hi batch offset" | ./bogo_gpu --daemon [--work-items N]

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

#define CL_CHECK(expr) \
    do { cl_int _e = (expr); if (_e != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s:%d\n", _e, __FILE__, __LINE__); \
        std::exit(1); } } while(0)


static const char *KERNEL_INLINE = R"CL(
static ulong sm64(ulong z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}
typedef struct { uint s[4]; } Xo128State;
static uint rotl32(uint x, uint k) { return (x << k) | (x >> (32u - k)); }
static uint xo_next(Xo128State *st) {
    uint res = rotl32(st->s[0] + st->s[3], 7u) + st->s[0];
    uint t   = st->s[1] << 9u;
    st->s[2] ^= st->s[0]; st->s[3] ^= st->s[1];
    st->s[1] ^= st->s[2]; st->s[0] ^= st->s[3];
    st->s[2] ^= t; st->s[3] = rotl32(st->s[3], 11u);
    return res;
}
#define XO_UINT(st, max) ({ \
    uint _thr = (uint)(((ulong)0x100000000UL % (ulong)(max))); \
    uint _x;  do { _x = xo_next(st); } while (_x < _thr); \
    _x % (max); })
#define FY_SWAP(arr, i, j) do { uchar _t = arr[i]; arr[i] = arr[j]; arr[j] = _t; } while(0)
#define DO_SHUFFLE(st, arr) do { \
    uint _j; \
    _j=XO_UINT(st,25); FY_SWAP(arr,24,_j); \
    _j=XO_UINT(st,24); FY_SWAP(arr,23,_j); \
    _j=XO_UINT(st,23); FY_SWAP(arr,22,_j); \
    _j=XO_UINT(st,22); FY_SWAP(arr,21,_j); \
    _j=XO_UINT(st,21); FY_SWAP(arr,20,_j); \
    _j=XO_UINT(st,20); FY_SWAP(arr,19,_j); \
    _j=XO_UINT(st,19); FY_SWAP(arr,18,_j); \
    _j=XO_UINT(st,18); FY_SWAP(arr,17,_j); \
    _j=XO_UINT(st,17); FY_SWAP(arr,16,_j); \
    _j=XO_UINT(st,16); FY_SWAP(arr,15,_j); \
    _j=XO_UINT(st,15); FY_SWAP(arr,14,_j); \
    _j=XO_UINT(st,14); FY_SWAP(arr,13,_j); \
    _j=XO_UINT(st,13); FY_SWAP(arr,12,_j); \
    _j=XO_UINT(st,12); FY_SWAP(arr,11,_j); \
    _j=XO_UINT(st,11); FY_SWAP(arr,10,_j); \
    _j=XO_UINT(st,10); FY_SWAP(arr, 9,_j); \
    _j=XO_UINT(st, 9); FY_SWAP(arr, 8,_j); \
    _j=XO_UINT(st, 8); FY_SWAP(arr, 7,_j); \
    _j=XO_UINT(st, 7); FY_SWAP(arr, 6,_j); \
    _j=XO_UINT(st, 6); FY_SWAP(arr, 5,_j); \
    _j=XO_UINT(st, 5); FY_SWAP(arr, 4,_j); \
    _j=XO_UINT(st, 4); FY_SWAP(arr, 3,_j); \
    _j=XO_UINT(st, 3); FY_SWAP(arr, 2,_j); \
    _j=XO_UINT(st, 2); FY_SWAP(arr, 1,_j); \
} while(0)

__kernel void bogosort_batch(
    uint seed_lo, uint seed_hi, uint batch_per_item, ulong index_offset,
    __global uchar *results
) {
    const ulong C = 0x9e3779b97f4a7c15UL;
    uint  gid    = (uint)get_global_id(0);
    ulong seed64 = ((ulong)seed_hi << 32) | (ulong)seed_lo;
    ulong base   = index_offset + (ulong)gid * (ulong)batch_per_item;

    ulong z  = seed64 + base * C + C;
    ulong ha = sm64(z); z += C;
    ulong hb = sm64(z); z += C;

    int   best_correct = -1;
    uchar best_arr[25];
    uint  best_iter    = 0;

    for (uint iter = 0; iter < batch_per_item; iter++) {
        Xo128State st;
        st.s[0] = (uint)(ha & 0xffffffffUL); st.s[1] = (uint)(ha >> 32);
        st.s[2] = (uint)(hb & 0xffffffffUL); st.s[3] = (uint)(hb >> 32);
        if (!st.s[0] && !st.s[1] && !st.s[2] && !st.s[3]) st.s[0] = 1;

        uchar arr[25];
        arr[ 0]= 1; arr[ 1]= 2; arr[ 2]= 3; arr[ 3]= 4; arr[ 4]= 5;
        arr[ 5]= 6; arr[ 6]= 7; arr[ 7]= 8; arr[ 8]= 9; arr[ 9]=10;
        arr[10]=11; arr[11]=12; arr[12]=13; arr[13]=14; arr[14]=15;
        arr[15]=16; arr[16]=17; arr[17]=18; arr[18]=19; arr[19]=20;
        arr[20]=21; arr[21]=22; arr[22]=23; arr[23]=24; arr[24]=25;

        DO_SHUFFLE(&st, arr);

        int correct =
            (arr[ 0]== 1)+(arr[ 1]== 2)+(arr[ 2]== 3)+(arr[ 3]== 4)+(arr[ 4]== 5)+
            (arr[ 5]== 6)+(arr[ 6]== 7)+(arr[ 7]== 8)+(arr[ 8]== 9)+(arr[ 9]==10)+
            (arr[10]==11)+(arr[11]==12)+(arr[12]==13)+(arr[13]==14)+(arr[14]==15)+
            (arr[15]==16)+(arr[16]==17)+(arr[17]==18)+(arr[18]==19)+(arr[19]==20)+
            (arr[20]==21)+(arr[21]==22)+(arr[22]==23)+(arr[23]==24)+(arr[24]==25);

        if (correct > best_correct) {
            best_correct = correct;
            best_iter    = iter;
            for (int i = 0; i < 25; i++) best_arr[i] = arr[i];
            if (correct == 25) break;
        }

        ha = hb;
        hb = sm64(z);
        z += C;
    }

    ulong best_index = base + (ulong)best_iter;
    size_t base_out  = (size_t)gid * 35;
    results[base_out] = (best_correct < 0) ? 0xFF : (uchar)best_correct;
    for (int i = 0; i < 25; i++) results[base_out + 1 + i] = best_arr[i];
    for (int i = 0; i < 8; i++)
        results[base_out + 26 + i] = (uchar)((best_index >> (8*i)) & 0xff);
    results[base_out + 34] = 0;
}
)CL";


struct GpuContext {
    cl_platform_id   platform    = nullptr;
    cl_device_id     device      = nullptr;
    cl_context       ctx         = nullptr;
    cl_command_queue queue       = nullptr;
    cl_program       program     = nullptr;
    cl_kernel        kernel      = nullptr;
    cl_mem           results_buf = nullptr;
    size_t           work_items  = 0;
    bool             unified_memory = false;
};

static void pick_device(GpuContext &g) {
    cl_uint np = 0; clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    struct C { cl_platform_id p; cl_device_id d; int s; };
    C best = {nullptr, nullptr, -1};
    for (auto &pl : plats) {
        char v[256] = {}; clGetPlatformInfo(pl, CL_PLATFORM_VENDOR, 256, v, nullptr);
        bool amd = strstr(v,"AMD") || strstr(v,"Advanced Micro");
        cl_uint nd = 0; clGetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
        std::vector<cl_device_id> devs(nd);
        clGetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
        for (auto &d : devs) {
            cl_device_type dt = 0; clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(dt), &dt, nullptr);
            int s = (amd?2:0) + ((dt & CL_DEVICE_TYPE_GPU)?1:0);
            if (s > best.s) best = {pl, d, s};
        }
    }
    if (!best.d) { fprintf(stderr, "No OpenCL device found.\n"); std::exit(1); }
    g.platform = best.p; g.device = best.d;
    char name[256]={}, pname[256]={};
    clGetDeviceInfo(g.device, CL_DEVICE_NAME, 256, name, nullptr);
    clGetPlatformInfo(g.platform, CL_PLATFORM_NAME, 256, pname, nullptr);
    cl_bool uni = CL_FALSE;
    clGetDeviceInfo(g.device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(uni), &uni, nullptr);
    bool igpu = (strstr(name,"Radeon RX")==nullptr) &&
                (strstr(name,"Radeon Graphics")!=nullptr || strstr(name,"Raphael")!=nullptr ||
                 strstr(name,"Rembrandt")!=nullptr || strstr(name,"Phoenix")!=nullptr ||
                 strstr(name,"Hawk Point")!=nullptr || strstr(name,"Strix")!=nullptr);
    g.unified_memory = (uni == CL_TRUE) || igpu;
    fprintf(stderr, "[gpu] platform: %s\n[gpu] device:   %s\n[gpu] unified_memory: %s\n",
            pname, name, g.unified_memory ? "yes (iGPU)" : "no (discrete)");
}

static void init_gpu(GpuContext &g, size_t work_items) {
    // disable AMD kernel cache (prevents stale compiled kernels)
    putenv((char*)"AMDOCL_DISABLE_SHADER_CACHE=1");
    putenv((char*)"GPU_ENABLE_SHADER_CACHE=0");
    g.work_items = work_items;
    pick_device(g);
    cl_int err = 0;
    g.ctx   = clCreateContext(nullptr, 1, &g.device, nullptr, nullptr, &err); CL_CHECK(err);
    g.queue = clCreateCommandQueueWithProperties(g.ctx, g.device, nullptr, &err); CL_CHECK(err);
    const char *src = KERNEL_INLINE; size_t slen = strlen(src);
    g.program = clCreateProgramWithSource(g.ctx, 1, &src, &slen, &err); CL_CHECK(err);
    err = clBuildProgram(g.program, 1, &g.device,
                         "-cl-std=CL2.0 -cl-fast-relaxed-math -cl-mad-enable -DV8",
                         nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t ls = 0; clGetProgramBuildInfo(g.program, g.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ls);
        std::string log(ls,'\0'); clGetProgramBuildInfo(g.program, g.device, CL_PROGRAM_BUILD_LOG, ls, log.data(), nullptr);
        fprintf(stderr, "Build log:\n%s\n", log.c_str()); std::exit(1);
    }
    g.kernel = clCreateKernel(g.program, "bogosort_batch", &err); CL_CHECK(err);
    cl_uint nargs = 0; clGetKernelInfo(g.kernel, CL_KERNEL_NUM_ARGS, sizeof(nargs), &nargs, nullptr);
    fprintf(stderr, "[gpu] kernel arg count (should be 5): %u\n", nargs);
    g.results_buf = clCreateBuffer(g.ctx, CL_MEM_READ_WRITE, work_items * 35, nullptr, &err);
    CL_CHECK(err);
}

static void destroy_gpu(GpuContext &g) {
    if (g.results_buf) clReleaseMemObject(g.results_buf);
    if (g.kernel)      clReleaseKernel(g.kernel);
    if (g.program)     clReleaseProgram(g.program);
    if (g.queue)       clReleaseCommandQueue(g.queue);
    if (g.ctx)         clReleaseContext(g.ctx);
}


struct BatchResult {
    int      best_correct;
    uint8_t  best_arr[25];
    uint64_t best_index;
};

static BatchResult run_dispatch(GpuContext &g,
                                uint32_t seed_lo, uint32_t seed_hi,
                                uint32_t batch_per_item, uint64_t index_offset) {
    cl_ulong off = (cl_ulong)index_offset;
    CL_CHECK(clSetKernelArg(g.kernel, 0, sizeof(cl_uint),  &seed_lo));
    CL_CHECK(clSetKernelArg(g.kernel, 1, sizeof(cl_uint),  &seed_hi));
    CL_CHECK(clSetKernelArg(g.kernel, 2, sizeof(cl_uint),  &batch_per_item));
    CL_CHECK(clSetKernelArg(g.kernel, 3, sizeof(cl_ulong), &off));
    CL_CHECK(clSetKernelArg(g.kernel, 4, sizeof(cl_mem),   &g.results_buf));
    size_t global = g.work_items;
    CL_CHECK(clEnqueueNDRangeKernel(g.queue, g.kernel, 1, nullptr,
                                    &global, nullptr, 0, nullptr, nullptr));
    CL_CHECK(clFinish(g.queue));
    std::vector<uint8_t> raw(g.work_items * 35);
    CL_CHECK(clEnqueueReadBuffer(g.queue, g.results_buf, CL_TRUE, 0,
                                 raw.size(), raw.data(), 0, nullptr, nullptr));
    BatchResult best; best.best_correct = -1; best.best_index = 0;
    memset(best.best_arr, 0, 25);
    for (size_t i = 0; i < g.work_items; i++) {
        uint8_t sb = raw[i*35];
        if (sb == 0xFF) continue;
        int c = (int)sb;
        if (c > best.best_correct) {
            best.best_correct = c;
            memcpy(best.best_arr, &raw[i*35+1], 25);
            uint64_t idx = 0;
            for (int b = 0; b < 8; b++) idx |= (uint64_t)raw[i*35+26+b] << (8*b);
            best.best_index = idx;
        }
    }
    return best;
}


static void print_result_json(const BatchResult &r, uint64_t total_done,
                               double elapsed, bool newline = true) {
    printf("{\"best_correct\":%d,\"best_arr\":[", r.best_correct);
    for (int i = 0; i < 25; i++) printf("%s%d", i?",":"", r.best_arr[i]);
    printf("],\"best_index\":%llu,\"total_done\":%llu,\"elapsed\":%.6f,\"rate\":%.0f,\"unified_memory\":%s}",
           (unsigned long long)r.best_index,
           (unsigned long long)total_done,
           elapsed,
           elapsed > 0 ? total_done / elapsed : 0.0,
           r.best_correct >= 0 ? "false" : "false");  // unified_memory always false in daemon (discrete)
    if (newline) printf("\n");
    fflush(stdout);
}


struct LoadPreset { size_t work_items; int sleep_ms; const char *name; };
static LoadPreset get_preset(const char *s) {
    if (!s) s = "max";
    std::string ls = s;
    if (ls=="low")    return {262144, 200, "low"};
    if (ls=="medium" || ls=="med") return {262144, 50, "medium"};
    return {262144, 0, "max"};
}

static void daemon_loop(GpuContext &g) {
    fprintf(stderr, "[gpu-daemon] ready  work_items=%zu\n", g.work_items);
    fflush(stderr);

    char line[512];
    while (fgets(line, sizeof(line), stdin)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;

        if (strncmp(line, "quit", 4) == 0) break;

        unsigned long long sl, sh, batch_req, offset;
        if (sscanf(line, "%llu %llu %llu %llu", &sl, &sh, &batch_req, &offset) != 4) {
            fprintf(stderr, "[gpu-daemon] bad line: '%s'\n", line);
            fflush(stderr);
            continue;
        }

        uint32_t seed_lo = (uint32_t)sl;
        uint32_t seed_hi = (uint32_t)sh;
        uint64_t index_offset = (uint64_t)offset;
        uint32_t batch_per_item = (uint32_t)((batch_req + g.work_items - 1) / g.work_items);
        uint64_t actual = (uint64_t)g.work_items * batch_per_item;

        double t0 = now_s();
        BatchResult r = run_dispatch(g, seed_lo, seed_hi, batch_per_item, index_offset);
        double elapsed = now_s() - t0;

        print_result_json(r, actual, elapsed);
    }
}


int main(int argc, char **argv) {
    bool        daemon_mode     = false;
    const char *load_str        = "max";
    size_t      work_items_flag = 0;
    std::vector<std::string> pos_args;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--daemon")                    { daemon_mode = true; }
        else if (a == "--load"       && i+1 < argc)  { load_str        = argv[++i]; }
        else if (a == "--work-items" && i+1 < argc)  { work_items_flag = std::stoull(argv[++i]); }
        else if (!a.empty() && a[0] != '-')          { pos_args.push_back(a); }
    }

    uint32_t seed_lo          = pos_args.size() > 0 ? (uint32_t)std::stoull(pos_args[0]) : 0x12345678u;
    uint32_t seed_hi          = pos_args.size() > 1 ? (uint32_t)std::stoull(pos_args[1]) : 0xdeadbeef;
    uint64_t total_target     = pos_args.size() > 2 ? std::stoull(pos_args[2])            : 10'000'000ULL;
    size_t   work_items_pos   = pos_args.size() > 3 ? std::stoull(pos_args[3])            : 0;
    uint64_t index_offset_pos = pos_args.size() > 4 ? std::stoull(pos_args[4])            : 0;

    LoadPreset preset = get_preset(load_str);
    size_t work_items = work_items_flag  ? work_items_flag  :
                        work_items_pos   ? work_items_pos   :
                        preset.work_items;
    work_items = ((work_items + 63) / 64) * 64;

    GpuContext g;
    init_gpu(g, work_items);

    if (daemon_mode) {
        daemon_loop(g);
        destroy_gpu(g);
        return 0;
    }

    uint32_t batch_per_item = (uint32_t)((total_target + work_items - 1) / work_items);
    uint64_t actual_total   = (uint64_t)work_items * batch_per_item;

    fprintf(stderr, "[gpu] load=%s  work_items=%zu  batch_per_item=%u  total=%llu\n",
            preset.name, work_items, batch_per_item, (unsigned long long)actual_total);

    uint64_t shuffles_done = 0, dispatches = 0;
    int      global_best  = -1;
    uint8_t  global_arr[25] = {};
    uint64_t global_index = 0;
    uint64_t index_offset = index_offset_pos;
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
        if (dispatches % 32 == 0) {
            double el = now_s() - t0;
            fprintf(stderr, "\r[gpu] %.2fM done  %.0fM/s  best=%d/25   ",
                    shuffles_done/1e6, shuffles_done/el/1e6, global_best);
            fflush(stderr);
        }
        if (global_best == 25) break;
        if (preset.sleep_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(preset.sleep_ms));
    }

    double elapsed = now_s() - t0;
    fprintf(stderr, "\n[gpu] done. %llu shuffles in %.3fs = %.0fM/s\n",
            (unsigned long long)shuffles_done, elapsed, shuffles_done/elapsed/1e6);

    uint64_t seed64 = ((uint64_t)seed_hi << 32) | seed_lo;
    printf("{\n");
    printf("  \"seed\": \"%llu\",\n",      (unsigned long long)seed64);
    printf("  \"total_done\": %llu,\n",    (unsigned long long)total_target);
    printf("  \"best_correct\": %d,\n",    global_best);
    printf("  \"best_index\": %llu,\n",    (unsigned long long)global_index);
    printf("  \"best_arr\": [");
    for (int i = 0; i < 25; i++) printf("%s%d", i?",":"", global_arr[i]);
    printf("],\n");
    printf("  \"elapsed\": %.6f,\n",       elapsed);
    printf("  \"rate\": %.0f,\n",          shuffles_done/elapsed);
    printf("  \"unified_memory\": %s\n",   g.unified_memory ? "true" : "false");
    printf("}\n");

    destroy_gpu(g);
    return 0;
}
