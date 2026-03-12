#include "cuda_example.h"

// Simple kernel to increment data
__global__ void incrementKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1.0f;
    }
}

void runGpuTask(int threadId, int N) {
    float *d_data;
    size_t size = N * sizeof(float);

    // 1. Allocate Memory
    cudaMalloc(&d_data, size);
    cudaMemset(d_data, 0, size);

    // 2. Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "CPU Thread " << threadId << " launching kernel..." << std::endl;
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // 3. IMPORTANT: Synchronization
    // Ensures this CPU thread waits until its GPU work is done
    cudaDeviceSynchronize(); 

    std::cout << "CPU Thread " << threadId << " kernel finished!" << std::endl;

    // 4. Cleanup
    cudaFree(d_data);
}

int main() {
    int numThreads = 4;
    int dataSize = 1000000;
    std::vector<std::thread> cpuThreads;

    // Launch multiple CPU threads
    for (int i = 0; i < numThreads; ++i) {
        cpuThreads.emplace_back(runGpuTask, i, dataSize);
    }

    // Join threads (wait for CPU threads to finish)
    for (auto& t : cpuThreads) {
        t.join();
    }

    std::cout << "All multi-threaded tasks complete." << std::endl;
    return 0;
}