#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <cublas_v2.h>

//#include <cuda_runtime_api.h>

// #ifdef __cplusplus
//   extern "C" {
// #endif  

// #define MM_BLOCK_SIZE 16
// #define MM_GRID_SIZE 4

// #define ELEM_PER_THREAD 4

__device__ constexpr int MM_BLOCK_SIZE = 16;
__device__ constexpr int MM_GRID_SIZE = 4;

__device__ constexpr int ELEM_PER_THREAD = 4;
__device__ constexpr int STRIDE = MM_BLOCK_SIZE / ELEM_PER_THREAD;

#define CUDA_KERNEL_CHECK			if( cudaDeviceSynchronize() != cudaSuccess ) {\
                                    printf("\n%s (%d): [CUDA KERNEL FAILED]\nCudaErr = %s\n",\
                                    __FILE__,__LINE__,cudaGetErrorString (cudaGetLastError())); \
									exit(-1); }

#define CUDA_CALL_FL(A,F,L)			do { if( (A) != cudaSuccess ) { \
                                    cudaDeviceSynchronize(); \
                                    printf("\n%s (%d): %s [CUDA CALL FAILED]\nCudaErr = %s\n",\
                                    F,L,#A,cudaGetErrorString (cudaGetLastError())); \
                                    exit(-1); } } while (0)
#define CUDA_CALL(A)				CUDA_CALL_FL(A,__FILE__,__LINE__)

// for cudnn
#define CUDNN_CALL_FL(A,F,L)        do { cudnnStatus_t status; \
                                    if ( (status = A) != CUDNN_STATUS_SUCCESS) {\
                                    printf("\n%s (%d): %s [CUDNN CALL FAILED]\nCudaErr = %s\n",\
                                    F,L,#A,cudnnGetErrorString(status)); \
                                    exit(-1); } \
                                    } while (0)

#define CUDNN_CALL(A)               CUDNN_CALL_FL(A,__FILE__,__LINE__)

#define CUBLAS_CALL_FL(A,F,L)        do { cublasStatus_t status = (A); \
                                    if (status != CUBLAS_STATUS_SUCCESS) {\
                                    printf("\n%s (%d): %s [CUBLAS CALL FAILED]\nError String: %s\n",\
                                    F, L, #A, cublasGetStatusString(status)); \
                                    exit(-1); } \
                                    } while (0)

#define CUBLAS_CALL(A)               CUBLAS_CALL_FL(A,__FILE__,__LINE__)

typedef void * cuBlasHandle;
cuBlasHandle InitCuBlas(bool useTensorCore);

// A is 2D matrix: col_a x row_a, dimA = { col_a x row_a x 1 }
// B is 3D tensor: col_b x row_b x depth_b, dimB = { col_b x row_b x depth_b }
// common dimension: col_a = row_b when both are not transposed -> not checked
// C = A * B -> dimC = { col_b x row_a x depth_b }
// A and B are Row major, cuBlas is Col major, we will perform B * A instead
// int3 is used as { Col, Row, Depth }
void BatchMatMul(const float * A, const int3 &dimA, bool transA,
                 const float * B, const int3 &dimB, bool transB,
                 float * C, cuBlasHandle cuBlasHandle);

// op = A x B
void gpuBatchMatMul(const float * ip, const int3 & ipSize, // A
                    const float * kernel, const int3 & kerSize, // B
                    float * op, cudaStream_t stream = 0);

// op = A x B
void gpuBatchMatMulScatter(const float * kernel, const int3 & kerSize, // A
                           const float * ip, const int3 & ipSize, // B
                           float * op, cudaStream_t stream);

__global__ void high_memory_kernel(float* output, int output_size, int share_mem__size);


// #ifdef __cplusplus
//   }
// #endif