// nbody.cu
// Gravitational N-body simulation: every body attracts every other body.
// Cost per step is O(N^2) interactions, which makes it a perfect GPU workload.
//
// Usage:
//   nbody bench                        verify CPU vs GPU, then benchmark several N
//   nbody <scenario.txt> [frames.bin]  run a scenario on the GPU and save frames
//                                      (render.py turns them into an animated GIF)
//
// Scenario files describe what collides with what. See scenarios/README.md.
//
// Build:  nvcc -O3 -o nbody nbody.cu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
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
// constexpr so the GPU code can use them too.
// ---------------------------------------------------------------------------
constexpr float G = 1.0f;
constexpr float SOFTENING2 = 0.0025f;  // eps^2: avoids infinite force when two bodies get very close
constexpr int   BLOCK = 256;
const double FLOPS_PER_INTERACTION = 20.0;  // common convention for N-body benchmarks
const float PI = 3.14159265f;

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

// ===========================================================================
// Scenarios: what is in the simulation and how it moves
// ===========================================================================

enum ObjectType { GALAXY = 0, CLUSTER = 1, BODY = 2 };

struct Object {
    ObjectType type = GALAXY;
    int   count = 4096;           // number of bodies (BODY is always 1)
    float x = 0, y = 0, z = 0;    // centre position
    float vx = 0, vy = 0, vz = 0; // centre velocity
    float mass = 1.0f;            // GALAXY: central mass, CLUSTER: total mass, BODY: its mass
    float diskMass = 0.1f;        // GALAXY only: total mass of the stars in the disk
    float radius = 1.2f;          // GALAXY: outer disk radius, CLUSTER: Plummer scale radius
    float inner = 0.2f;           // GALAXY only: inner disk radius
    float spin = 1.0f;            // GALAXY only: +1 counter-clockwise, -1 clockwise
    float tilt = 0.0f;            // GALAXY only: tilt of the disk around the x axis, degrees
};

struct Scenario {
    std::string name = "scenario";
    int   steps = 1500;
    int   saveEvery = 6;
    float dt = 0.01f;
    float view = 5.0f;            // the GIF shows x and y from -view to +view
    unsigned seed = 42;
    std::vector<Object> objects;

    int totalBodies() const {
        int n = 0;
        for (const Object& o : objects) n += (o.type == BODY) ? 1 : o.count;
        return n;
    }
};

float frand() { return rand() / (float)RAND_MAX; }

// Galaxy: heavy central body + thin disk of stars on circular orbits.
void addGalaxy(const Object& o, std::vector<float4>& pos, std::vector<float4>& vel) {
    float tilt = o.tilt * PI / 180.0f;
    float ct = cosf(tilt), st = sinf(tilt);

    pos.push_back(make_float4(o.x, o.y, o.z, o.mass));
    vel.push_back(make_float4(o.vx, o.vy, o.vz, 0.0f));

    int stars = o.count - 1;
    float starMass = (stars > 0) ? o.diskMass / stars : 0.0f;
    for (int k = 0; k < stars; ++k) {
        // Uniform density over the ring area between inner and radius.
        float u = frand();
        float r = sqrtf(o.inner * o.inner + u * (o.radius * o.radius - o.inner * o.inner));
        float theta = 2.0f * PI * frand();
        float h = 0.01f * (frand() - 0.5f) * 2.0f;          // small disk thickness

        // Speed for a circular orbit around the mass inside radius r.
        float v = sqrtf(G * (o.mass + o.diskMass * u) / r);

        float c = cosf(theta), s = sinf(theta);
        // Position and velocity in the disk's own plane...
        float px = r * c, py = r * s, pz = h;
        float qx = -o.spin * v * s, qy = o.spin * v * c, qz = 0.0f;
        // ...then tilted around the x axis.
        float py2 = py * ct - pz * st, pz2 = py * st + pz * ct;
        float qy2 = qy * ct - qz * st, qz2 = qy * st + qz * ct;

        pos.push_back(make_float4(o.x + px, o.y + py2, o.z + pz2, starMass));
        vel.push_back(make_float4(o.vx + qx, o.vy + qy2, o.vz + qz2, 0.0f));
    }
}

// Random direction on a sphere, scaled to length `len`.
float3 randomDirection(float len) {
    float zc = 2.0f * frand() - 1.0f;
    float phi = 2.0f * PI * frand();
    float rxy = sqrtf(fmaxf(0.0f, 1.0f - zc * zc));
    return make_float3(len * rxy * cosf(phi), len * rxy * sinf(phi), len * zc);
}

// Star cluster: a Plummer sphere, a classic model of a ball of stars that is in
// equilibrium (it neither collapses nor flies apart on its own).
// Sampling method from Aarseth, Henon & Wielen (1974).
void addCluster(const Object& o, std::vector<float4>& pos, std::vector<float4>& vel) {
    float a = o.radius;
    float m = o.mass / o.count;
    for (int k = 0; k < o.count; ++k) {
        // Radius from the Plummer cumulative mass profile (cut off at 8a).
        float r;
        do {
            float u = fmaxf(frand(), 1e-6f);
            r = a / sqrtf(powf(u, -2.0f / 3.0f) - 1.0f);
        } while (r > 8.0f * a);
        float3 p = randomDirection(r);

        // Speed as a fraction q of the local escape speed, by rejection sampling.
        float q, y;
        do {
            q = frand();
            y = 0.1f * frand();
        } while (y > q * q * powf(1.0f - q * q, 3.5f));
        float vesc = sqrtf(2.0f * G * o.mass) * powf(r * r + a * a, -0.25f);
        float3 v = randomDirection(q * vesc);

        pos.push_back(make_float4(o.x + p.x, o.y + p.y, o.z + p.z, m));
        vel.push_back(make_float4(o.vx + v.x, o.vy + v.y, o.vz + v.z, 0.0f));
    }
}

// Build all bodies. group[i] tells which object body i belongs to (for colours).
void buildScenario(const Scenario& sc, std::vector<float4>& pos, std::vector<float4>& vel,
                   std::vector<int>& group) {
    srand(sc.seed);
    pos.clear(); vel.clear(); group.clear();
    for (size_t g = 0; g < sc.objects.size(); ++g) {
        const Object& o = sc.objects[g];
        size_t before = pos.size();
        if (o.type == GALAXY)       addGalaxy(o, pos, vel);
        else if (o.type == CLUSTER) addCluster(o, pos, vel);
        else {
            pos.push_back(make_float4(o.x, o.y, o.z, o.mass));
            vel.push_back(make_float4(o.vx, o.vy, o.vz, 0.0f));
        }
        group.resize(pos.size(), (int)g);
        (void)before;
    }
}

// The built-in scene used by `bench` and the verification: two galaxies, n bodies total.
Scenario twoGalaxies(int n) {
    Scenario sc;
    sc.name = "two_galaxies";
    Object a, b;
    a.count = n / 2;     a.x = -3; a.y = -1; a.vx = 0.3f;  a.spin = 1;
    b.count = n - n / 2; b.x = 3;  b.y = 1;  b.vx = -0.3f; b.spin = -1;
    sc.objects = {a, b};
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario file parser
//
//   # comment
//   steps = 1500            global settings: steps, dt, save_every, view, seed
//   galaxy  count=4096 x=-3 y=-1 vx=0.3 spin=1
//   cluster count=2048 x=3 mass=0.5 radius=0.3
//   body    x=0 y=4 vy=-0.5 mass=3
// ---------------------------------------------------------------------------
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// "a = 1   b= 2" -> "a=1 b=2" so every setting is a single token.
std::string squeezeEquals(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == ' ' || s[i] == '\t') {
            size_t j = i;
            while (j < s.size() && (s[j] == ' ' || s[j] == '\t')) ++j;
            bool nextIsEq = (j < s.size() && s[j] == '=');
            bool prevIsEq = (!out.empty() && out.back() == '=');
            if (nextIsEq || prevIsEq) { i = j - 1; continue; }
        }
        out += s[i];
    }
    return out;
}

bool setObjectField(Object& o, const std::string& key, float v) {
    if (key == "count")          o.count = (int)v;
    else if (key == "x")         o.x = v;
    else if (key == "y")         o.y = v;
    else if (key == "z")         o.z = v;
    else if (key == "vx")        o.vx = v;
    else if (key == "vy")        o.vy = v;
    else if (key == "vz")        o.vz = v;
    else if (key == "mass")      o.mass = v;
    else if (key == "disk_mass") o.diskMass = v;
    else if (key == "radius")    o.radius = v;
    else if (key == "inner")     o.inner = v;
    else if (key == "spin")      o.spin = v;
    else if (key == "tilt")      o.tilt = v;
    else return false;
    return true;
}

bool loadScenario(const char* path, Scenario& sc) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("ERROR: cannot open scenario file %s\n", path); return false; }

    // Scenario name = file name without folder and extension.
    std::string p = path;
    size_t slash = p.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    sc.name = (dot == std::string::npos) ? base : base.substr(0, dot);

    char buf[1024];
    int lineNo = 0;
    bool ok = true;
    while (fgets(buf, sizeof(buf), f)) {
        ++lineNo;
        std::string line = buf;
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(squeezeEquals(trim(line)));
        if (line.empty()) continue;

        // Split into tokens.
        std::vector<std::string> tok;
        size_t i = 0;
        while (i < line.size()) {
            while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
            size_t j = i;
            while (j < line.size() && line[j] != ' ' && line[j] != '\t') ++j;
            if (j > i) tok.push_back(line.substr(i, j - i));
            i = j;
        }

        const std::string& first = tok[0];
        if (first == "galaxy" || first == "cluster" || first == "body") {
            Object o;
            if (first == "galaxy")  { o.type = GALAXY; }
            if (first == "cluster") { o.type = CLUSTER; o.count = 2048; o.radius = 0.3f; }
            if (first == "body")    { o.type = BODY; o.count = 1; }
            for (size_t t = 1; t < tok.size(); ++t) {
                size_t eq = tok[t].find('=');
                if (eq == std::string::npos) {
                    printf("  line %d: expected key=value, got '%s'\n", lineNo, tok[t].c_str());
                    ok = false; continue;
                }
                std::string key = tok[t].substr(0, eq);
                float v = (float)atof(tok[t].substr(eq + 1).c_str());
                if (!setObjectField(o, key, v)) {
                    printf("  line %d: unknown setting '%s' for %s\n", lineNo, key.c_str(), first.c_str());
                    ok = false;
                }
            }
            if (o.type != BODY && o.count < 2) {
                printf("  line %d: count must be at least 2\n", lineNo);
                ok = false;
            }
            sc.objects.push_back(o);
        } else {
            // Global setting(s): key=value
            for (const std::string& t : tok) {
                size_t eq = t.find('=');
                if (eq == std::string::npos) {
                    printf("  line %d: unknown line '%s'\n", lineNo, line.c_str());
                    ok = false; break;
                }
                std::string key = t.substr(0, eq);
                double v = atof(t.substr(eq + 1).c_str());
                if (key == "steps")           sc.steps = (int)v;
                else if (key == "dt")         sc.dt = (float)v;
                else if (key == "save_every") sc.saveEvery = (int)v;
                else if (key == "view")       sc.view = (float)v;
                else if (key == "seed")       sc.seed = (unsigned)v;
                else { printf("  line %d: unknown setting '%s'\n", lineNo, key.c_str()); ok = false; }
            }
        }
    }
    fclose(f);

    if (sc.objects.empty()) { printf("ERROR: the scenario has no objects\n"); ok = false; }
    if (sc.saveEvery < 1) sc.saveEvery = 1;
    return ok;
}

// ===========================================================================
// GPU simulation state
// ===========================================================================
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
    std::vector<float4> hp, hv;
    std::vector<int> group;
    buildScenario(twoGalaxies(n), hp, hv, group);
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
    std::vector<float4> hp, hv;
    std::vector<int> group;
    buildScenario(twoGalaxies(n), hp, hv, group);
    std::vector<float4> ha(hp.size());
    auto t0 = std::chrono::high_resolution_clock::now();
    stepCPU(hp.data(), hv.data(), ha.data(), n, 0.001f);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Verify: CPU and GPU (naive and tiled) must agree after a few steps.
// ---------------------------------------------------------------------------
bool verify() {
    const int n = 2048, steps = 10;
    const float dt = 0.005f;
    printf("== Verification (N = %d, %d steps) ==\n", n, steps);

    std::vector<float4> p0, v0;
    std::vector<int> group;
    buildScenario(twoGalaxies(n), p0, v0, group);

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
// Benchmark one simulation step for several N.
// ---------------------------------------------------------------------------
void benchmark() {
    printf("== Benchmark (time per simulation step) ==\n");
    printf("%8s | %12s | %12s | %12s | %12s | %10s\n",
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
// Run a scenario on the GPU and save x,y of every body every few steps.
//
// frames.bin format (all little-endian):
//   int32 n, int32 frames, int32 numObjects, float32 view
//   int32 objectType[numObjects]     0 = galaxy, 1 = cluster, 2 = body
//   int32 group[n]                   which object each body belongs to
//   frames * n * (float32 x, float32 y)
// ---------------------------------------------------------------------------
bool simulate(const Scenario& sc, const char* path) {
    std::vector<float4> hp, hv;
    std::vector<int> group;
    buildScenario(sc, hp, hv, group);
    int n = (int)hp.size();
    int frames = sc.steps / sc.saveEvery;
    int numObjects = (int)sc.objects.size();

    printf("== Scenario '%s' ==\n", sc.name.c_str());
    const char* typeName[] = {"galaxy", "cluster", "body"};
    for (int g = 0; g < numObjects; ++g) {
        const Object& o = sc.objects[g];
        printf("  %-7s  bodies=%-6d pos=(%.2f, %.2f, %.2f)  vel=(%.2f, %.2f, %.2f)  mass=%.2f\n",
               typeName[o.type], (o.type == BODY) ? 1 : o.count,
               o.x, o.y, o.z, o.vx, o.vy, o.vz, o.mass);
    }
    printf("  total bodies = %d, steps = %d, dt = %.4f, frames = %d\n", n, sc.steps, sc.dt, frames);
    if (n > 65536) printf("  (large N: this will take a while)\n");

    GpuSim sim;
    sim.upload(hp, hv);

    FILE* f = fopen(path, "wb");
    if (!f) { printf("  ERROR: cannot write %s\n", path); sim.release(); return false; }
    std::vector<int> types(numObjects);
    for (int g = 0; g < numObjects; ++g) types[g] = (int)sc.objects[g].type;
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&frames, sizeof(int), 1, f);
    fwrite(&numObjects, sizeof(int), 1, f);
    fwrite(&sc.view, sizeof(float), 1, f);
    fwrite(types.data(), sizeof(int), numObjects, f);
    fwrite(group.data(), sizeof(int), n, f);

    std::vector<float> xy(2 * (size_t)n);
    auto t0 = std::chrono::high_resolution_clock::now();
    int written = 0;
    for (int s = 1; s <= sc.steps && written < frames; ++s) {
        sim.step(true, sc.dt);
        if (s % sc.saveEvery == 0) {
            sim.downloadPos(hp);
            for (int i = 0; i < n; ++i) { xy[2 * i] = hp[i].x; xy[2 * i + 1] = hp[i].y; }
            fwrite(xy.data(), sizeof(float), xy.size(), f);
            ++written;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    auto t1 = std::chrono::high_resolution_clock::now();
    fclose(f);
    sim.release();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("  Simulated %d steps in %.2f s (including saving frames)\n", sc.steps, sec);
    printf("  Saved %d frames to %s\n\n", written, path);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("  nbody bench                        verify + benchmark\n");
        printf("  nbody <scenario.txt> [frames.bin]  run a scenario\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs)\n\n", prop.name, prop.multiProcessorCount);

    if (strcmp(argv[1], "bench") == 0) {
        bool ok = verify();
        benchmark();
        return ok ? 0 : 1;
    }

    Scenario sc;
    if (!loadScenario(argv[1], sc)) {
        printf("Fix the scenario file and try again.\n");
        return 1;
    }
    const char* out = (argc > 2) ? argv[2] : "frames.bin";
    return simulate(sc, out) ? 0 : 1;
}
