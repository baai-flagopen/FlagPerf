#include <stdio.h>
#include <musa_runtime.h>
#include <string.h>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (1ULL * GB)
#define WARMUP_ITERATIONS 1
#define ITERATIONS 50

void checkMusaError(musaError_t err, const char* msg) {
    if (err != musaSuccess) {
        fprintf(stderr, "MUSA Error: %s: %s\n", msg, musaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define LOOP_NUM 4
#define UNROLL_NUM 4

// read only
__global__ void read_only(const float4* src, float4* dst) {
    int id, dist;
    id = blockIdx.x * blockDim.x * LOOP_NUM * UNROLL_NUM + threadIdx.x;
    dist = blockDim.x;

    float4 tmp = {0.f, 0.f, 0.f, 0.f};

#pragma unroll 1
    for (int i = 0; i < LOOP_NUM; i++) {
#pragma unroll
        for (int j = 0; j < UNROLL_NUM; j++) {
            float4 val = src[id];
            tmp.x += val.x;
            tmp.y += val.y;
            tmp.z += val.z;
            tmp.w += val.w;
            id += dist;
        }
    }

    if (tmp.x + tmp.y + tmp.z + tmp.w < -1e10f) {
        dst[blockIdx.x * blockDim.x + threadIdx.x] = tmp;
    }
}

// write only
#define LOOP_NUM_W 8
#define UNROLL_NUM_W 8

__global__ void write_only(float4 *dst) {
    int id, dist;
    id = blockIdx.x * blockDim.x * LOOP_NUM_W * UNROLL_NUM_W + threadIdx.x;
    dist = blockDim.x;

    float4 val = {1.0f, 2.0f, 3.0f, 4.0f}; 

#pragma unroll 1
    for (int i = 0; i < LOOP_NUM_W; i++) {
#pragma unroll
        for (int j = 0; j < UNROLL_NUM_W; j++) {
            dst[id] = val;
            id += dist;
        }
    }
}

/* d2d */
#define LOOP_NUM_D2D 1
#define UNROLL_NUM_D2D 1

__global__ void global_bandwidth(float4 *dst, float4 *src) {
  int id, dist;
  id = blockIdx.x * blockDim.x * LOOP_NUM_D2D * UNROLL_NUM_D2D + threadIdx.x;
  dist = blockDim.x;
  // id = blockIdx.x * blockDim.x + threadIdx.x;
  // dist = gridDim.x * blockDim.x;
#pragma unroll 1
  for (int i = 0; i < LOOP_NUM_D2D; i++) {
#pragma unroll
    for (int j = 0; j < UNROLL_NUM_D2D; j++) {
      dst[id] = src[id];
      id += dist;
    }
  }
}

void runReadOnly(void *dst, void *src, size_t total_size) {
    dim3 block_size(256),
    block_num(total_size / sizeof(float4) / LOOP_NUM / UNROLL_NUM / 256);
    read_only<<<block_num, block_size>>>((float4 *)dst, (float4 *)src);
}

void runWriteOnly(float* dst, size_t total_size) {
    dim3 block_size(256),
    block_num(total_size / sizeof(float4) / LOOP_NUM_W / UNROLL_NUM_W / 256);
    write_only<<<block_num, block_size>>>((float4 *)dst);
}

void runMemcpy(void *dst, void *src, size_t total_size) {
    dim3 block_size(256),
    block_num(total_size / sizeof(float4) / LOOP_NUM_D2D / UNROLL_NUM_D2D / 256);
    global_bandwidth<<<block_num, block_size>>>((float4 *)dst, (float4 *)src);
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "readwrite";

    float* d_src, * d_dst;
    musaEvent_t start, end;
    float elapsed_time;

    checkMusaError(musaMalloc(&d_src, SIZE), "musaMalloc");
    checkMusaError(musaMalloc(&d_dst, SIZE), "musaMalloc");

    checkMusaError(musaEventCreate(&start), "musaEventCreate");
    checkMusaError(musaEventCreate(&end), "musaEventCreate");

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        if (strcmp(mode, "read") == 0) {
            runReadOnly(d_src, d_dst, SIZE);
        } else if (strcmp(mode, "write") == 0) {
            runWriteOnly(d_dst, SIZE);
        } else {
            runMemcpy(d_dst, d_src, SIZE);
        }
    }

    checkMusaError(musaEventRecord(start), "musaEventRecord");

    
    for (int i = 0; i < ITERATIONS; ++i) {
        if (strcmp(mode, "read") == 0) {
            runReadOnly(d_src, d_dst, SIZE);
        } else if (strcmp(mode, "write") == 0) {
            runWriteOnly(d_dst, SIZE);
        } else {
            runMemcpy(d_dst, d_src, SIZE);
        }
    }

    checkMusaError(musaEventRecord(end), "musaEventRecord");
    checkMusaError(musaEventSynchronize(end), "musaEventSynchronize");

    checkMusaError(musaEventElapsedTime(&elapsed_time, start, end), "musaEventElapsedTime");

    double bidirectional = (strcmp(mode, "readwrite") == 0) ? 2.0 : 1.0;
    double bandwidth = bidirectional * SIZE * ITERATIONS / (elapsed_time / 1000.0);
    

    printf("\n[FlagPerf Result][%s]main_memory-bandwidth=%.2fGiB/s\n", mode, bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result][%s]main_memory-bandwidth=%.2fGB/s\n", mode, bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkMusaError(musaFree(d_src), "musaFree");
    checkMusaError(musaFree(d_dst), "musaFree");
    checkMusaError(musaEventDestroy(start), "musaEventDestroy");
    checkMusaError(musaEventDestroy(end), "musaEventDestroy");

    return 0;
}