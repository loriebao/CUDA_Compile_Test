#include "cuda_example.h"

cuBlasHandle InitCuBlas(bool useTensorCore)
{
    cublasHandle_t handle = nullptr;
    CUBLAS_CALL(cublasCreate(&handle));
    CUBLAS_CALL(cublasSetMathMode(handle, useTensorCore ? CUBLAS_TF32_TENSOR_OP_MATH :
                                                          CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    // Blackwell Optimization: If you want to use Tensor Cores (TC), 
    // you often need to set the math mode explicitly.
    if (useTensorCore) {
        CUBLAS_CALL(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }
    
    return (cuBlasHandle)handle;
}

// Batch matrix multiplication kernel
// matrix description is Width x Height -> NumCol x NumRow, not regular math notations of NumRow x NumCol
// Operation is : A x B = C where
//                                     W x H
//                              (A) is K X M,
//                              (B) is N x K,
//                              (C) is N x M.
// A and C are assumed to contain batches of matrices, B is assumed a single matrix
// not efficient, for now, use shared memory later after verifying results
__global__ void batchMatMulKernel(const float * A, const float * B, float * C, int M, int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // col
    int j = blockIdx.y * ELEM_PER_THREAD * blockDim.y + threadIdx.y; // row

    int bi = threadIdx.x;
    int bj = threadIdx.y;

    int batch = blockIdx.z;

    __shared__ float subA[MM_BLOCK_SIZE * MM_BLOCK_SIZE];
    __shared__ float subB[MM_BLOCK_SIZE * MM_BLOCK_SIZE];

    if (i >= N || j >= M)
        return;

    // move to specific batch for input and output
    A += (K*M)*batch;
    C += (N*M)*batch;

    // access helper functions
    auto get = [](const int & c /* X */, const int & r /* Y */, const int & Width) -> int { return Width*r + c; };

    int bk_max = min(K, MM_BLOCK_SIZE); // in case K < MM_BLOCK_SIZE

    float sum[ELEM_PER_THREAD] = {0.f};
    float tmpB;

    for (int k = 0; k < K; k += MM_BLOCK_SIZE)
    {
    #pragma unroll ELEM_PER_THREAD
        for (int m = 0; m < ELEM_PER_THREAD; m++)
        {
            if (bi < K) // in case K < MM_BLOCK_SIZE
                subA[get(bi, bj + m * STRIDE, MM_BLOCK_SIZE)] = A[get(k + bi, j + m  * STRIDE,     K)];
            if (bj + m * STRIDE < K)
                subB[get(bi, bj + m * STRIDE, MM_BLOCK_SIZE)] = B[get(i,      k + bj + m * STRIDE, N)];
        }
        __syncthreads();

        #pragma unroll
        for (int bk = 0; bk < bk_max; bk++)
        {
            tmpB = subB[get(bi, bk, MM_BLOCK_SIZE)];
            #pragma unroll ELEM_PER_THREAD
            for (int m = 0; m < ELEM_PER_THREAD; m++)
                sum[m] += subA[get(bk, bj + m * STRIDE, MM_BLOCK_SIZE)] * tmpB;
        }
        __syncthreads();
    }

    #pragma unroll ELEM_PER_THREAD
    for (int m = 0; m < ELEM_PER_THREAD; m++)
        C[get(i, j + m * STRIDE, N)] = sum[m];
}


void gpuMatlMul(float * d_output, const float * d_input, const int & input_size,
                const float * d_kernel, const int & kernel_size, cudaStream_t stream)
{
    // input is K x M = w x h
    int M = input_size;
    int K = input_size;
    // kernel is N x K = w x h
    int N = kernel_size;

    int num_batch = input_size * input_size;

    dim3 thread_size(MM_BLOCK_SIZE, MM_BLOCK_SIZE / ELEM_PER_THREAD, 1);
    int bsx = ceilf(N/float(MM_BLOCK_SIZE));
    int bsy = ceilf(M/float(MM_BLOCK_SIZE));
    dim3 grid_size(bsx, bsy, num_batch);

    batchMatMulKernel<<< grid_size, thread_size, 0, stream >>>( d_input, d_kernel, d_output, M, N, K);
    CUDA_KERNEL_CHECK;
}

void BatchMatMul(const float * A, const int3 & dimA, bool transA,
                 const float * B, const int3 & dimB, bool transB,
                 float * C, cuBlasHandle cuBlasHandle)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    int32_t opA_rows = transA ? dimA.x : dimA.y;
    int32_t opA_cols = transA ? dimA.y : dimA.x;
    int32_t opB_rows = transB ? dimB.x : dimB.y;
    int32_t opB_cols = transB ? dimB.y : dimB.x;

    if (opA_cols != opB_rows)
    {
        printf("Inner dimensions (%d # %d) don't match\n", opA_cols, opB_rows);
        return;
    }

    int32_t strideA = dimA.x * dimA.y;
    int32_t strideB = dimB.x * dimB.y;
    int32_t nBatch = 1;

    if (dimA.z != 1 && dimB.z != 1)
    {
        if (dimA.z != dimB.z)
        {
            printf("Batch size mismatch between A (%d) & B (%d)\n", dimA.z, dimB.z);
            return;
        }
        nBatch = dimA.z; // both equal size: multiply each A matrix by each B matrix
    }
    else
    {
        if (dimA.z == 1) // broadcast A to set of B matrices
        {
            strideA = 0;
            nBatch = dimB.z;
        }
        else if (dimB.z == 1) // broadcast B to set of A matrices
        {
            strideB = 0;
            nBatch = dimA.z;
        }
    }

    int32_t comm = opA_cols; // common dimension
    int3 dimC = { opB_cols, opA_rows, nBatch };

    // cuBlas is col-major to match Fortran !! So, to perform A x B, we have to rearrange the data, which is slow !
    // Instead, we use the property (A⋅B)^T = B^T x A^T. In cuBLAS, this means we swap the positions of A and B in the
    // function call and interpret our Row-Major data as Column-Major.
    // Leading Dimension (lda, ldb, ldc): This is the number of rows in the allocated matrix (including any padding).
    // For a standard packed column-major matrix, lda is the number of rows.
    // For a row-major matrix, lda becomes the number of columns.

    // following description is Row x Col (math notation) !! Sorry for confusion
    // A is M x N x L, for transpose, A is N x M, in both cases LDB is set to columns of A : dimA.x
    // B is N x K x L, for transpose, B is K x N, in both cases LDA is set to columns of B : dimB.x
    // C is M x K x L
    // To multiply a Row-Major Matrix A (M×NxL) by a Row-Major Tensor B (N×K×L):
    //  Swap A and B: We treat the operation as B×A.
    //  Dimensions: The dimensions passed to the function become (K,M,N).
    //  Leading Dimensions: These become the number of columns of the original matrices.
    //  Parameter	Row-Major Value     Why?
    //      m           K               The number of columns in the original tensor B.
    //      n           M               The number of rows in the original matrix A.
    //      k           N               The common dimension (columns of A / rows of B).
    //      lda         K (N trans)     Leading dimension of B (width of B).
    //      ldb         N (M trans)     Leading dimension of A (width of A).
    //      ldc         K               Leading dimension of C (width of C).

    // Result = A * B (Row Major) -> cuBLAS call interpreted as (R^T)^T = (B_T * A_T)^T
    CUBLAS_CALL( cublasSgemmStridedBatched( (cublasHandle_t) cuBlasHandle,
                                            transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                                            transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                                            dimC.x, dimC.y, comm,       // Cols of C, Rows of C, Common Dim
                                            &alpha,
                                            B, dimB.x, strideB,         // LDA = cols of B.
                                            A, dimA.x, strideA,         // LDB = cols of A.
                                            &beta,
                                            C, dimC.x, dimC.x * dimC.y, // LDC = cols of C. Stride is M*K.
                                            dimC.z ));                  // batch dimension

}


__device__ constexpr int THREADS = 256;
__device__ constexpr int MAX_MAT_SIZE = 64;
//__device__ constexpr int ELEM_PER_THREAD = 2;

__constant__ float A_KERNEL[THREADS];
__constant__ float B_KERNEL[THREADS];


// Special Batch matrix multiplication with a kernel and scattering the output
// matrix description is Width x Height -> NumCol x NumRow, not regular math notations of NumRow x NumCol
// Operation is : A x B = C where
//                                     W x H
//        kernel matrix         (A) is K X M
//                              (B) is N x K x nBatch
//                              (C) is M x N            -> scatter each element separated by nBatch elements
__global__ void batchMatMulScatterKernel(const float * __restrict__ B, float * __restrict__ C,
                                         int M, int K, int N, int nBatch)
{
    int batch = (blockIdx.x * blockDim.x + threadIdx.x);
    if (batch >= nBatch) return;

    // access helper functions
    auto get = [](const int c /* X */, const int r /* Y */, const int W) -> int { return W*r + c; };

    // move to specific batch for input and output
    B += (K*N)*batch;
    C += batch; // scatter effect, don't use M*batch

    float BB[MAX_MAT_SIZE];

    // store matrix localy, small B
    if (K % 4 == 0) // maximize bandwidth throughput
    {
        for (int m = 0; m < N*K/4; m++) ((float4 *) BB)[m] = ((const float4 *) B)[m];
    }
    else
    {
        for (int m = 0; m < N*K; m++) BB[m] = B[m];
    }

    float sum;

    for (int n = 0; n < N; n++) // cols of B
    {
        for (int m = 0; m < M; m++) // rows of kernel A
        {
            sum = 0.f;
            for (int k = 0; k < K; k++) // cols of A = rows of B
            {
                sum += A_KERNEL[get(k, m, K)] * BB[get(n, k, N)];
            }

            *C = sum;
            C += nBatch; // scatter
        }
    }
}

// op = A x B
void gpuBatchMatMulScatter(const float * kernel, const int3 & kerSize, // A
                           const float * ip, const int3 & ipSize, // B
                           float * op, cudaStream_t stream)
{
    if ( kerSize.x * kerSize.y > THREADS )
    {
        printf("batchMatMulKernel ERROR: Unsupported Kernel (%d x %d) dimension, total elements must be < %d\n",
               kerSize.x, kerSize.y, THREADS);
        return;
    }

    if (kerSize.x != ipSize.y)
    {
        printf("batchMatMulKernel ERROR: Inner dimension mismatch: Kernel (%d x %d) and Input (%d x %d)\n",
               kerSize.x, kerSize.y, ipSize.x, ipSize.y);
        return;
    }

    if (ipSize.x * ipSize.y > MAX_MAT_SIZE)
    {
        printf("batchMatMulKernel ERROR: Input number of elements > max supported (%d)\n", MAX_MAT_SIZE);
        return;
    }

    // kernel is K x M = w x h
    int K = kerSize.x;
    int M = kerSize.y;
    // input is N x K = w x h
    int N = ipSize.x;

    int nBatch = ipSize.z;

    CUDA_CALL(cudaMemcpyToSymbol(A_KERNEL, kernel, sizeof(float) * kerSize.x * kerSize.y, 0, cudaMemcpyDeviceToDevice));

    batchMatMulScatterKernel<<< (nBatch + THREADS - 1) / THREADS, THREADS, 0, stream >>>( ip, op, M, K, N, nBatch);
    CUDA_KERNEL_CHECK;
}

// Special Batch matrix multiplication with a kernel
// matrix description is Width x Height -> NumCol x NumRow, not regular math notations of NumRow x NumCol
// Operation is : A x B = C where
//                                     W x H
//                              (A) is K X M x nBatch
//        kernel matrix         (B) is 1 x K x N                broadcast each col for all K x M x nBatch of A, repeat N times
//                              (C) is M x 1 x nBatch x N
// A and C are assumed to contain batches of matrices, B is assumed a single matrix
// each thread will multiply a small matrix A by a kernel B and broadcast the result in a non-contiguous way
// each column result will belong to a different volume separated by nBatch elements
// The kernel B matrix is stored in constant memory KERNEL for faster access since it is common with all threads
// that is a faster approach than using shared memory since it is small and better cached and avoids thread syncing
__global__ void batchMatMulKernel(const float * __restrict__ A, float * __restrict__ C, int M, int K, int N, int nBatch)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= nBatch) return;

    // access helper functions
    auto get = [](const int c /* X */, const int r /* Y */, const int W) -> int { return W*r + c; };

    // move to specific batch for input and output
    A += (K*M)*batch;
    C += M*batch;

    float AA[MAX_MAT_SIZE];

    // store matrix localy, small A
    for (int m = 0; m < M*K; m++) AA[m] = A[m];

    float sum ;

    for (int n = 0; n < N; n++) // volume number -> cols of B kernel
    {
        for (int m = 0; m < M; m++) // rows of A
        {
            sum = 0.f;
            for (int k = 0; k < K; k++) // cols of A = rows of B
            {
                sum += AA[get(k, m, K)] * B_KERNEL[get(n, k, N)];

                /*if (batch == 0) // && n == 0)
                    printf("%d: AA(%d, %d) (%f) * BB(%d, %d) (%f) = %f\n",
                           m, k, m, AA[get(k, m, K)], n, k, BB[get(n, k, N)], sum[m]);*/
            }

            C[m] = sum;
        }
        //if (batch == 0) printf("\n");

        C += M*nBatch;
    }
}

// op = A x B
void gpuBatchMatMul(const float * ip, const int3 & ipSize, // A
                    const float * kernel, const int3 & kerSize, // B
                    float * op, cudaStream_t stream)
{
    if ( kerSize.x * kerSize.y > THREADS )
    {
        printf("batchMatMulKernel ERROR: Unsupported Kernel (%d x %d) dimension, total elements must be < %d\n",
               kerSize.x, kerSize.y, THREADS);
        return;
    }

    if (ipSize.x != kerSize.y)
    {
        printf("batchMatMulKernel ERROR: Inner dimension mismatch: Input (%d x %d) and Kernel (%d x %d)\n",
               ipSize.x, ipSize.y, kerSize.x, kerSize.y);
        return;
    }

    if (ipSize.x * ipSize.y > MAX_MAT_SIZE)
    {
        printf("batchMatMulKernel ERROR: Input number of elements > max supported (%d)\n", MAX_MAT_SIZE);
        return;
    }

    // input is K x M = w x h
    int K = ipSize.x;
    int M = ipSize.y;
    // kernel is N x K = w x h
    int N = kerSize.x;

    int nBatch = ipSize.z;

    CUDA_CALL(cudaMemcpyToSymbol(B_KERNEL, kernel, sizeof(float) * kerSize.x * kerSize.y, 0, cudaMemcpyDeviceToDevice));

    batchMatMulKernel<<< (nBatch + THREADS - 1) / THREADS, THREADS, 0, stream >>>( ip, op, M, K, N, nBatch);
    CUDA_KERNEL_CHECK;
}

// Kernel using Dynamic Shared Memory
__global__ void high_memory_kernel(float* output, int output_size, int share_mem__size) {
    // 'extern' indicates the size is defined at launch time
    extern __shared__ float shared_data[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Use the shared memory (e.g., as an 16,384 element float array = 64 KB)
    shared_data[tid] = (float)tid;
    
    __syncthreads();

    //if (tid == 16) 
        output[0] = shared_data[tid]; // Accessing data beyond the 48KB (12k float) mark
    
    printf("high_memory_kernel from GPU thread %d: output[0] %f, output_size %d, share_mem__size %dKB!\n", 
            tid, output[0], output_size, share_mem__size/1024);
}

void cuBlas_overlap_test(float *h_A, float *h_B, float *h_C, const int M, const int K, const int N)
{
    constexpr int max_num_streams = 4;

    int num_streams = max_num_streams;


    
    // Calculate rows per stream
    std::div_t result = std::div(M, num_streams);
    int rows_per_stream = (result.rem==0)?result.quot:(result.quot+1);

    if(M<max_num_streams)
    {
        num_streams = 1;
        rows_per_stream = M;
    }
    else
    {
        num_streams = (M+rows_per_stream-1)/rows_per_stream;
    }
    std::cout << "Number of streams: " << num_streams << std::endl;
    std::cout << "Rows per stream: " << rows_per_stream << std::endl;

    // 2. Device Memory
    float *d_A, *d_B, *d_C;
    CUDA_KERNEL_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_KERNEL_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_KERNEL_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 3. Create Streams and cuBLAS Handles
    cudaStream_t streams[max_num_streams];
    cublasHandle_t handles[max_num_streams];
    for (int i = 0; i < max_num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cublasCreate(&handles[i]);
        cublasSetStream(handles[i], streams[i]);
    }
    float alpha = 1.0f, beta = 0.0f;

    CUDA_KERNEL_CHECK(cudaMemset(&d_A, 0, M * K * sizeof(float)));
    CUDA_KERNEL_CHECK(cudaMemset(&d_B, 0, K * N * sizeof(float)));
    CUDA_KERNEL_CHECK(cudaMemset(&d_C, 0, M * N * sizeof(float)));

    // Start Timing
    cudaEventRecord(start);
    for(int ii=0; ii<1; ii++)
    {
    // Pre-copy Matrix B (Common to all streams)
    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, streams[0]);

    // 4. Overlap Loop
    for (int i = 0; i < num_streams; i++) {
        
        int a_offset = i * rows_per_stream * K;
        int c_offset = i * rows_per_stream * N;

        if(i == num_streams -1)
        {   
            rows_per_stream = M - i * rows_per_stream;
            //std::cout << "Last stream processing: " << rows_per_stream << " rows." << std::endl;
        }
        // Asynchronous H2D for a chunk of A
        cudaMemcpyAsync(d_A + a_offset, h_A + a_offset, 
                        rows_per_stream * K * sizeof(float), 
                        cudaMemcpyHostToDevice, streams[i]);

        // Compute: C_chunk = A_chunk * B
        // Row-Major swap: C^T = B^T * A^T
        cublasSgemm(handles[i], CUBLAS_OP_N, CUBLAS_OP_N,
                    N, rows_per_stream, K,
                    &alpha,
                    d_B, N,
                    d_A + a_offset, K,
                    &beta,
                    d_C + c_offset, N);

        // Asynchronous D2H for the result chunk
        cudaMemcpyAsync(h_C + c_offset, d_C + c_offset, 
                        rows_per_stream * N * sizeof(float), 
                        cudaMemcpyDeviceToHost, streams[i]);
    }
    }
    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) cudaStreamSynchronize(streams[i]);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate TFLOPS: (2 * M * N * K) / (time * 10^12)
    double flops = 2.0 * M * N * K * 1000;
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time elapsed (M " << M << ", N " << N << ", K " << K << "): " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // Cleanup
    for (int i = 0; i < max_num_streams; i++) {
        cublasDestroy(handles[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() 
{
    // Row major Matrix dimensions: C(M,N) = A(M,K) * B(K,N)
    const int M = 64, K = 20, N = 64;
    //const int M = 1024, K = 320, N = 1024;

    // 1. Allocate Pinned Host Memory (Page-locked)
    float *h_A, *h_B, *h_C, *h_D;
    CUDA_KERNEL_CHECK(cudaHostAlloc(&h_A, M * K * sizeof(float), cudaHostAllocDefault));
    CUDA_KERNEL_CHECK(cudaHostAlloc(&h_B, K * N * sizeof(float), cudaHostAllocDefault));
    CUDA_KERNEL_CHECK(cudaHostAlloc(&h_C, M * N * sizeof(float), cudaHostAllocDefault));
    CUDA_KERNEL_CHECK(cudaHostAlloc(&h_D, M * N * sizeof(float), cudaHostAllocDefault));

    // 2. Initialize Host Memory
    for (int i = 0; i < M * K; i++) h_A[i] = i + 1;
    for (int i = 0; i < K * N; i++) h_B[i] = i + 1;
    for (int i = 0; i < M * N; i++) h_C[i] = 0;
    for (int i = 0; i < M * N; i++) h_D[i] = 0;

    // 3. Call Batch Matrix Multiplication Kernel
    cuBlas_overlap_test(h_A, h_B, h_C, M, K, N);

#if 1
    float *d_A, *d_B, *d_C;
    CUDA_KERNEL_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_KERNEL_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_KERNEL_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_KERNEL_CHECK(cudaMemset(&d_A, 0, M * K * sizeof(float)));
    CUDA_KERNEL_CHECK(cudaMemset(&d_B, 0, K * N * sizeof(float)));
    CUDA_KERNEL_CHECK(cudaMemset(&d_C, 0, M * N * sizeof(float)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;
    // Start Timing
    cudaEventRecord(start);

    for(int ii=0; ii<1; ii++)
    {
        CUDA_KERNEL_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_KERNEL_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

        /* Logic: C = A * B in Row-Major is equivalent to C^T = B^T * A^T in Col-Major.
        cuBLAS interprets our Row-Major A(m,k) as a Col-Major A^T(k,m).
        We pass: handle, transB, transA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc
        */
        CUBLAS_CALL(cublasSgemm(handle, 
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                N, M, K, 
                                &alpha, 
                                d_B, N,    // Leading dimension of B is 'n' columns
                                d_A, K,    // Leading dimension of A is 'k' columns
                                &beta, 
                                d_C, N));  // Leading dimension of C is 'n' columns

        CUDA_KERNEL_CHECK(cudaMemcpy(h_D, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate TFLOPS: (2 * M * N * K) / (time * 10^12)
    double flops = 2.0 * M * N * K * 1000;
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time elapsed (M " << M << ", N " << N << ", K " << K << "): " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
#endif

#if 0
    //print h_A
    std::cout << "Input A (Row-Major):" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << h_A[i * K + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //print h_B
    std::cout << "Input B (Row-Major):" << std::endl;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {   
            std::cout << h_B[i * N + j] << " ";
        }
        std::cout << std::endl;
    } 
    std::cout << std::endl;
#endif

    if(memcmp(h_C, h_D, M * N * sizeof(float)) !=0)
    {
        std::cout << "Error: Host output Memory Mismatch" << std::endl;
        //print diff
        std::cout << "Result C-D (Row-Major):" << std::endl;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << h_C[i * N + j] - h_D[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        #if 0
        //print h_C
        std::cout << "Result C (Row-Major):" << std::endl;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << h_C[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        //print h_D
        std::cout << "Result D (Row-Major):" << std::endl;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << h_D[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        #endif
    }
    // 4. Cleanup
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C); cudaFreeHost(h_D);
    return 0;
}

#if 0
int main() {
    int device = 0;
    int kb_per_block = 64;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    

    // Check if the hardware actually supports more than 48 KB
    if (prop.sharedMemPerBlockOptin < 65536) {
        printf("Device does not support 64KB shared memory per block.\n");
        return 0;
    }
    else
    {
        kb_per_block = prop.sharedMemPerBlockOptin/1024;
        printf("Device supports %dKB shared memory per block.\n", (int)prop.sharedMemPerBlockOptin/1024);
    }
        

    float *d_out;
    cudaMalloc(&d_out, sizeof(float));

    size_t shared_mem_size = 48 * 1024;  //kb_per_block * 1024;

    // IMPORTANT: You must opt-in to use more than 48 KB
    // This attribute is what allows Blackwell/Ada to use their full L1 capacity
    cudaFuncSetAttribute(high_memory_kernel, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 
                         shared_mem_size);

    // Launch with 64 KB specified in the 3rd execution configuration parameter
    high_memory_kernel<<<MM_GRID_SIZE, MM_BLOCK_SIZE, shared_mem_size>>>(d_out, (int)sizeof(float), (int)shared_mem_size);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    else
        printf("Kernel launched successfully with %zu bytes of shared memory.\n", shared_mem_size);

    cudaFree(d_out);
    return 0;
}

int main() {
    const int max_num_streams = 4;
    cudaStream_t streams[max_num_streams];
    cublasHandle_t handles[max_num_streams];
    cudaEvent_t sync_events[max_num_streams]; // Event Pool
    cudaEvent_t start, stop;

    // Initialization
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < max_num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cublasCreate(&handles[i]);
        cublasSetStream(handles[i], streams[i]);
        cudaEventCreate(&sync_events[i]);
    }

    // 1. Record start on the default stream
    cudaEventRecord(start, 0);

    // 2. Launch asynchronous work across streams
    for (int i = 0; i < max_num_streams; i++) {
        // [Insert H2D and cublasSgemm calls here as per previous logic]
        
        // Record an event in this specific stream's queue after its work
        cudaEventRecord(sync_events[i], streams[i]);
    }

    // 3. The Join Point: Make the NULL stream wait for ALL worker streams
    for (int i = 0; i < max_num_streams; i++) {
        // GPU-side wait: Stream 0 stays idle until sync_events[i] is reached
        cudaStreamWaitEvent(0, sync_events[i], 0);
    }

    // 4. Record stop on the NULL stream (now guaranteed to be the last thing)
    cudaEventRecord(stop, 0);

    // 5. CPU Synchronize
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total Multi-Stream Time: " << ms << " ms" << std::endl;

    // Cleanup
    for (int i = 0; i < max_num_streams; i++) {
        cudaStreamDestroy(streams[i]);
        cublasDestroy(handles[i]);
        cudaEventDestroy(sync_events[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
#endif