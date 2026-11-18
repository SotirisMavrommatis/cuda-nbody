// nbody.cu
// Gravitational N-body simulation: every body attracts every other body.
// Cost per step is O(N^2) interactions, which makes it a perfect GPU workload.
//
// The program does three things:
//   1. VERIFY    - runs the same small simulation on CPU and GPU and compares positions
//   2. BENCHMARK - times one simulation step on CPU, GPU naive and GPU tiled for several N
//   3. SIMULATE  - two colliding galaxies on the GPU, saving frames to frames.bin
//                  (render.py turns them into an animated GIF)
//
// Build:  nvcc -O3 -o nbody nbody.cu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Physics constants (simulation units, G = 1)
// ---------------------------------------------------------------------------
constexpr float G = 1.0f;
constexpr float SOFTENING2 = 0.0025f;  // eps^2: avoids infinite force when two bodies get very close
constexpr int   BLOCK = 256;
const double FLOPS_PER_INTERACTION = 20.0;  // common convention for N-body benchmarks

// Bodies are stored as float4:  pos = (x, y, z, mass),  vel = (vx, vy, vz, unused)

// Acceleration that body j applies to body i, added to `a`.
// Shared by CPU and GPU code (__host__ __device__).
__host__ __device__ inline float3 bodyInteraction(float4 bi, float4 bj, float3 a) {
    float rx = bj.x - bi.x;
    float ry = bj.y - bi.y;
    float rz = bj.z - bi.z;
    float dist2 = rx * rx + ry * ry + rz * rz + SOFTENING2;
#ifdef __CUDA_ARCH__
    float invDist = rsqrtf(dist2);             // fast hardware 1/sqrt on the GPU
#else
    float invDist = 1.0f / sqrtf(dist2);
#endif
    float s = G * bj.w * invDist * invDist * invDist;   // G * m_j / r^3
    a.x += rx * s;
    a.y += ry * s;
    a.z += rz * s;
    return a;
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// Naive: thread i loops over all N bodies, reading each one from global memory.
__global__ void forcesNaive(const float4* pos, float4* acc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 bi = pos[i];
    float3 a = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < n; ++j) {
        a = bodyInteraction(bi, pos[j], a);
    }
    acc[i] = make_float4(a.x, a.y, a.z, 0.0f);
}

// Tiled: the block loads BLOCK bodies at a time into shared memory, then every
// thread computes its interactions with that tile from fast on-chip memory.
// Each body loaded from global memory is reused by all BLOCK threads.
__global__ void forcesTiled(const float4* pos, float4* acc, int n) {
    __shared__ float4 tile[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 bi = (i < n) ? pos[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 a = make_float3(0.0f, 0.0f, 0.0f);

    for (int start = 0; start < n; start += BLOCK) {
        int j = start + threadIdx.x;
        // Out-of-range slots get mass 0, so they add no force.
        tile[threadIdx.x] = (j < n) ? pos[j] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < BLOCK; ++k) {
            a = bodyInteraction(bi, tile[k], a);
        }
        __syncthreads();
    }

    if (i < n) acc[i] = make_float4(a.x, a.y, a.z, 0.0f);
}

// Semi-implicit Euler: update velocity with the new acceleration, then position.
__global__ void integrate(float4* pos, float4* vel, const float4* acc, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 p = pos[i], v = vel[i], a = acc[i];
    v.x += a.x * dt;  v.y += a.y * dt;  v.z += a.z * dt;
    p.x += v.x * dt;  p.y += v.y * dt;  p.z += v.z * dt;
    pos[i] = p;
    vel[i] = v;
}

// ---------------------------------------------------------------------------
// CPU reference (single thread)
// ---------------------------------------------------------------------------
void stepCPU(float4* pos, float4* vel, float4* acc, int n, float dt) {
    for (int i = 0; i < n; ++i) {
        float3 a = make_float3(0.0f, 0.0f, 0.0f);
        for (int j = 0; j < n; ++j) a = bodyInteraction(pos[i], pos[j], a);
        acc[i] = make_float4(a.x, a.y, a.z, 0.0f);
    }
    for (int i = 0; i < n; ++i) {
        vel[i].x += acc[i].x * dt;  vel[i].y += acc[i].y * dt;  vel[i].z += acc[i].z * dt;
        pos[i].x += vel[i].x * dt;  pos[i].y += vel[i].y * dt;  pos[i].z += vel[i].z * dt;
    }
}

// ---------------------------------------------------------------------------
// Initial conditions: two rotating disk galaxies on a collision course.
// Each galaxy = one heavy central body + a thin disk of light stars
// on (roughly) circular orbits.
// ---------------------------------------------------------------------------
float frand() { return rand() / (float)RAND_MAX; }

void initGalaxies(std::vector<float4>& pos, std::vector<float4>& vel, int n) {
    srand(42);
    const float centerMass = 1.0f;
    const float diskMass   = 0.1f;
    const float rMin = 0.2f, rMax = 1.2f;
    const float PI = 3.14159265f;

    int half = n / 2;
    for (int g = 0; g < 2; ++g) {
        int first = (g == 0) ? 0 : half;
        int count = (g == 0) ? half : n - half;
        float cx = (g == 0) ? -3.0f : 3.0f;
        float cy = (g == 0) ? -1.0f : 1.0f;
        float vx = (g == 0) ? 0.3f : -0.3f;
        float spin = (g == 0) ? 1.0f : -1.0f;   // the two galaxies rotate in opposite directions

        pos[first] = make_float4(cx, cy, 0.0f, centerMass);
        vel[first] = make_float4(vx, 0.0f, 0.0f, 0.0f);

        float starMass = diskMass / (count - 1);
        for (int k = 1; k < count; ++k) {
            // Uniform density over the ring area between rMin and rMax.
            float u = frand();
            float r = sqrtf(rMin * rMin + u * (rMax * rMax - rMin * rMin));
            float theta = 2.0f * PI * frand();
            float z = 0.02f * (frand() - 0.5f);

            // Speed for a circular orbit around the mass inside radius r.
            float enclosed = centerMass + diskMass * u;
            float v = sqrtf(G * enclosed / r);

            float c = cosf(theta), s = sinf(theta);
            pos[first + k] = make_float4(cx + r * c, cy + r * s, z, starMass);
            vel[first + k] = make_float4(vx - spin * v * s, spin * v * c, 0.0f, 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU simulation state
// ---------------------------------------------------------------------------
struct GpuSim {
    float4 *pos = nullptr, *vel = nullptr, *acc = nullptr;
    int n = 0;

    void upload(const std::vector<float4>& hp, const std::vector<float4>& hv) {
        n = (int)hp.size();
        size_t bytes = n * sizeof(float4);
        CUDA_CHECK(cudaMalloc(&pos, bytes));
        CUDA_CHECK(cudaMalloc(&vel, bytes));
        CUDA_CHECK(cudaMalloc(&acc, bytes));
        CUDA_CHECK(cudaMemcpy(pos, hp.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vel, hv.data(), bytes, cudaMemcpyHostToDevice));
    }
    void step(bool tiled, float dt) {
        int blocks = (n + BLOCK - 1) / BLOCK;
        if (tiled) forcesTiled<<<blocks, BLOCK>>>(pos, acc, n);
        else       forcesNaive<<<blocks, BLOCK>>>(pos, acc, n);
        integrate<<<blocks, BLOCK>>>(pos, vel, acc, n, dt);
    }
    void downloadPos(std::vector<float4>& hp) {
        CUDA_CHECK(cudaMemcpy(hp.data(), pos, n * sizeof(float4), cudaMemcpyDeviceToHost));
    }
    void release() {
        cudaFree(pos); cudaFree(vel); cudaFree(acc);
    }
};

// Average milliseconds per GPU step.
float timeGpuSteps(int n, bool tiled, int steps) {
    std::vector<float4> hp(n), hv(n);
    initGalaxies(hp, hv, n);
    GpuSim sim;
    sim.upload(hp, hv);

    sim.step(tiled, 0.001f);                 // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int s = 0; s < steps; ++s) sim.step(tiled, 0.001f);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    sim.release();
    return ms / steps;
}

double timeCpuStep(int n) {
    std::vector<float4> hp(n), hv(n), ha(n);
    initGalaxies(hp, hv, n);
    auto t0 = std::chrono::high_resolution_clock::now();
    stepCPU(hp.data(), hv.data(), ha.data(), n, 0.001f);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// 1. Verify: CPU and GPU (naive and tiled) must agree after a few steps.
// ---------------------------------------------------------------------------
bool verify() {
    const int n = 2048, steps = 10;
    const float dt = 0.005f;
    printf("== 1. Verification (N = %d, %d steps) ==\n", n, steps);

    std::vector<float4> p0(n), v0(n);
    initGalaxies(p0, v0, n);

    std::vector<float4> cp = p0, cv = v0, ca(n);
    for (int s = 0; s < steps; ++s) stepCPU(cp.data(), cv.data(), ca.data(), n, dt);

    bool allOk = true;
    for (int tiled = 0; tiled <= 1; ++tiled) {
        GpuSim sim;
        sim.upload(p0, v0);
        for (int s = 0; s < steps; ++s) sim.step(tiled != 0, dt);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float4> gp(n);
        sim.downloadPos(gp);
        sim.release();

        float maxDiff = 0.0f;
        for (int i = 0; i < n; ++i) {
            maxDiff = fmaxf(maxDiff, fabsf(gp[i].x - cp[i].x));
            maxDiff = fmaxf(maxDiff, fabsf(gp[i].y - cp[i].y));
            maxDiff = fmaxf(maxDiff, fabsf(gp[i].z - cp[i].z));
        }
        bool ok = maxDiff < 1e-3f;
        allOk = allOk && ok;
        printf("  GPU %-6s vs CPU: max position difference = %.2e  -> %s\n",
               tiled ? "tiled" : "naive", maxDiff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");
    return allOk;
}

// ---------------------------------------------------------------------------
// 2. Benchmark one simulation step for several N.
// ---------------------------------------------------------------------------
void benchmark() {
    printf("== 2. Benchmark (time per simulation step) ==\n");
    printf("%8s | %12s | %12s | %12s | %10s | %10s\n",
           "N", "CPU (ms)", "GPU naive", "GPU tiled", "tiled GFLOPS", "vs CPU");
    printf("---------+--------------+--------------+--------------+--------------+-----------\n");

    const int sizes[] = {1024, 4096, 16384, 65536};
    for (int n : sizes) {
        double interactions = (double)n * n;
        float naive = timeGpuSteps(n, false, 10);
        float tiled = timeGpuSteps(n, true, 10);
        double gflops = interactions * FLOPS_PER_INTERACTION / (tiled * 1e6);

        if (n <= 16384) {
            double cpu = timeCpuStep(n);
            printf("%8d | %12.2f | %12.3f | %12.3f | %12.1f | %9.0fx\n",
                   n, cpu, naive, tiled, gflops, cpu / tiled);
        } else {
            printf("%8d | %12s | %12.3f | %12.3f | %12.1f | %10s\n",
                   n, "(too slow)", naive, tiled, gflops, "-");
        }
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// 3. Galaxy collision: run on the GPU and save x,y of every body every few steps.
//    File format: int32 n, int32 frames, then frames * n * (float32 x, float32 y)
// ---------------------------------------------------------------------------
void simulate(const char* path) {
    const int n = 8192;
    const int steps = 1500, saveEvery = 6;
    const float dt = 0.01f;
    const int frames = steps / saveEvery;
    printf("== 3. Galaxy collision (N = %d, %d steps, %d frames) ==\n", n, steps, frames);

    std::vector<float4> hp(n), hv(n);
    initGalaxies(hp, hv, n);
    GpuSim sim;
    sim.upload(hp, hv);

    FILE* f = fopen(path, "wb");
    if (!f) { printf("  ERROR: cannot write %s\n", path); return; }
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&frames, sizeof(int), 1, f);

    std::vector<float> xy(2 * n);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 1; s <= steps; ++s) {
        sim.step(true, dt);
        if (s % saveEvery == 0) {
            sim.downloadPos(hp);
            for (int i = 0; i < n; ++i) { xy[2 * i] = hp[i].x; xy[2 * i + 1] = hp[i].y; }
            fwrite(xy.data(), sizeof(float), xy.size(), f);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    auto t1 = std::chrono::high_resolution_clock::now();
    fclose(f);
    sim.release();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("  Simulated %d steps in %.2f s (including saving frames)\n", steps, sec);
    printf("  Saved frames to %s\n\n", path);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs)\n\n", prop.name, prop.multiProcessorCount);

    bool ok = verify();
    benchmark();
    simulate("frames.bin");
    return ok ? 0 : 1;
}
