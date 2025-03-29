#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
// #include "/home/pp24/pp24s036/firefly///nvtx/c/include///nvtx3///nvtx3.hpp"


#define BLOCK_SIZE 64  //64,32,16,8
static const int INF = ((1 << 30) - 1);


#define B  BLOCK_SIZE            // tile大小
#define T  (BLOCK_SIZE / 2)     // tile大小的一半 (用來做 2x2 載入/更新)
#define S  (BLOCK_SIZE*BLOCK_SIZE)  // B*B
#define S2 (2*BLOCK_SIZE*BLOCK_SIZE)


int  n, m;          // 圖的大小、邊的數量 (原始)
int  n_padded;      // 將 n 向上對齊到 B 的倍數後的大小
int* Dist_h = nullptr;  // Host 上的距離矩陣, 大小 n_padded * n_padded
int* Dist_d = nullptr;  // GPU 上的距離矩陣, 大小 n_padded * n_padded


void input(const char* infile);
void output(const char* outfile);
int  ceilDiv(int a, int b);
void block_FW();

// 三個 kernel
__global__ void phase1_kernel(int* dist, int n_padded, int r);
__global__ void phase2_kernel(int* dist, int n_padded, int r);
__global__ void phase3_kernel(int* dist, int n_padded, int r);

int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        return 1;
    }

    // 1. 讀檔
    input(argv[1]);

    // 2. 配置 GPU 記憶體並拷貝資料
    size_t distSize = (size_t)n_padded * (size_t)n_padded * sizeof(int);
    cudaMalloc((void**)&Dist_d, distSize);
    cudaMemcpy(Dist_d, Dist_h, distSize, cudaMemcpyHostToDevice);

    // 3. 執行 Blocked Floyd-Warshall
    block_FW();

    // 4. 拷回結果到 host
    cudaMemcpy(Dist_h, Dist_d, distSize, cudaMemcpyDeviceToHost);

    // 5. 輸出檔案 (只輸出原本 n x n 的部分)
    output(argv[2]);

    // 收尾
    cudaFree(Dist_d);
    free(Dist_h);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Program execution time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}


void input(const char* infile) {
    //nvtxRangePushA("input");
    FILE* file = fopen(infile, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open input file %s\n", infile);
        exit(1);
    }

    // 讀入 n, m
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // 將 n 向上對齊到 B 的倍數
    // 例如 n=130, B=64 => n_padded=192
    n_padded = n + (B - (n % B)) % B;

    // 分配 host memory
    Dist_h = (int*)malloc((size_t)n_padded * (size_t)n_padded * sizeof(int));
    if (!Dist_h) {
        fprintf(stderr, "Host Dist_h malloc failed\n");
        fclose(file);
        exit(1);
    }

    // 初始化 
    //   i==j => 0
    //   其餘 => INF
    //   超過 n 的行列也設為 INF
    for (int i = 0; i < n_padded; i++) {
        for (int j = 0; j < n_padded; j++) {
            if (i < n && j < n) {
                Dist_h[i*n_padded + j] = (i == j) ? 0 : INF;
            } else {
                Dist_h[i*n_padded + j] = INF;
            }
        }
    }

    // 讀取邊
    int edge[3];
    for (int i = 0; i < m; i++) {
        fread(edge, sizeof(int), 3, file);
        int a = edge[0];
        int b = edge[1];
        int w = edge[2];
        Dist_h[a*n_padded + b] = w;
    }
    fclose(file);
    //nvtxRangePop();
}


void output(const char* outfile) {
    //nvtxRangePushA("output");
    FILE* out = fopen(outfile, "wb");
    if (!out) {
        fprintf(stderr, "Cannot open output file %s\n", outfile);
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (Dist_h[i*n_padded + j] >= INF) {
                Dist_h[i*n_padded + j] = INF;
            }
        }
        fwrite(&Dist_h[i*n_padded], sizeof(int), n, out);
    }
    fclose(out);
    //nvtxRangePop();
}


void block_FW() {
    //nvtxRangePushA("block_FW");

    int round = n_padded / B;

    // 決定一次 launch 幾個 thread (T x T)，
    // 例如 B=64 時 T=32 => dim3(32,32)
    // 例如 B=32 時 T=16 => dim3(16,16)
    dim3 threads(T, T);

    for (int r = 0; r < round; r++) {
        // Phase1
        phase1_kernel<<<1, threads>>>(Dist_d, n_padded, r);

        // Phase2
        if (round > 1) {
            // grid2 = (2, round - 1)
            dim3 grid2(2, round - 1);
            phase2_kernel<<<grid2, threads>>>(Dist_d, n_padded, r);
        }

        // Phase3
        if (round > 1) {
            // grid3 = (round - 1, round - 1)
            dim3 grid3(round - 1, round - 1);
            phase3_kernel<<<grid3, threads>>>(Dist_d, n_padded, r);
        }
    }

    //nvtxRangePop();
}


__global__ void phase1_kernel(int* dist, int n_padded, int r) {
    // 用 shared memory 存 B x B
    __shared__ int s[B * B];

    // pivot 區塊左上角在 dist[] 裏的 index
    int b_i = r * B;
    int b_j = r * B;

    // 以 (threadIdx.x, threadIdx.y) 索引 tile 內的位置
    int i = threadIdx.y; 
    int j = threadIdx.x;

    // 載入 2x2 的方式
    s[i * B + j]           = dist[(b_i + i) * n_padded + (b_j + j)];
    s[i * B + (j + T)]     = dist[(b_i + i) * n_padded + (b_j + j + T)];
    s[(i + T) * B + j]     = dist[(b_i + i + T) * n_padded + (b_j + j)];
    s[(i + T) * B + (j+T)] = dist[(b_i + i + T) * n_padded + (b_j + j + T)];

    // 利用 k 在 [0..(B-1)] 反覆更新
    #pragma unroll
    for (int k = 0; k < B; k++) {
        __syncthreads();
        s[i*B + j] = min(s[i*B + j], 
                         s[i*B + k] + s[k*B + j]);
        s[i*B + j + T] = min(s[i*B + j + T],
                             s[i*B + k] + s[k*B + (j + T)]);
        s[(i+T)*B + j] = min(s[(i+T)*B + j],
                             s[(i+T)*B + k] + s[k*B + j]);
        s[(i+T)*B + (j+T)] = min(s[(i+T)*B + (j+T)],
                                 s[(i+T)*B + k] + s[k*B + (j+T)]);
    }

    // 寫回 global memory
    dist[(b_i + i) * n_padded + (b_j + j)]           = s[i * B + j];
    dist[(b_i + i) * n_padded + (b_j + j + T)]       = s[i * B + (j + T)];
    dist[(b_i + i + T) * n_padded + (b_j + j)]       = s[(i + T) * B + j];
    dist[(b_i + i + T) * n_padded + (b_j + j + T)]   = s[(i + T) * B + (j + T)];
}

//==============================================================================
// phase2_kernel: 同 row 與同 column 的區塊
//   gridDim = (2, round-1)
//   blockIdx.x = 0 => column blocks
//   blockIdx.x = 1 => row blocks
//------------------------------------------------------------------------------
__global__ void phase2_kernel(int* dist, int n_padded, int r) {
    // s[] 前半存 (row-part)，後半存 (col-part)
    __shared__ int s[S2];

    // blockIdx.y from [0..(round-2)], 需要跳過 r
    int actual_b = blockIdx.y + (blockIdx.y >= r);

    // pivot block 的起始點
    int b_k = r * B;

    // 依照 blockIdx.x 來決定要處理 row-block 或 col-block
    // row blocks: row = r, col = actual_b
    // col blocks: row = actual_b, col = r
    int b_i = (blockIdx.x ? r         : actual_b) * B; 
    int b_j = (blockIdx.x ? actual_b  : r        ) * B; 

    // 每個 thread 的 (i, j) 索引
    int i = threadIdx.y; 
    int j = threadIdx.x;

    // 讀取「自己」區塊的 4 個值
    int val0 = dist[(b_i + i)       * n_padded + (b_j + j)];
    int val1 = dist[(b_i + i)       * n_padded + (b_j + j + T)];
    int val2 = dist[(b_i + i + T)   * n_padded + (b_j + j)];
    int val3 = dist[(b_i + i + T)   * n_padded + (b_j + j + T)];

    // 前半 s[0 ~ B*B-1]: (b_i, b_k)
    // 後半 s[B*B ~ 2*B*B-1]: (b_k, b_j)
    s[i*B + j]               = dist[(b_i + i)       * n_padded + (b_k + j)];
    s[i*B + (j+T)]           = dist[(b_i + i)       * n_padded + (b_k + (j+T))];
    s[(i+T)*B + j]           = dist[(b_i + i + T)   * n_padded + (b_k + j)];
    s[(i+T)*B + (j+T)]       = dist[(b_i + i + T)   * n_padded + (b_k + (j+T))];

    s[S + i*B + j]           = dist[(b_k + i)       * n_padded + (b_j + j)];
    s[S + i*B + (j+T)]       = dist[(b_k + i)       * n_padded + (b_j + (j+T))];
    s[S + (i+T)*B + j]       = dist[(b_k + i + T)   * n_padded + (b_j + j)];
    s[S + (i+T)*B + (j+T)]   = dist[(b_k + i + T)   * n_padded + (b_j + (j+T))];

    __syncthreads();

    // FW 更新
    #pragma unroll
    for (int k = 0; k < B; k++) {
        val0 = min(val0, s[i*B + k] + s[S + k*B + j]);
        val1 = min(val1, s[i*B + k] + s[S + k*B + (j+T)]);
        val2 = min(val2, s[(i+T)*B + k] + s[S + k*B + j]);
        val3 = min(val3, s[(i+T)*B + k] + s[S + k*B + (j+T)]);
    }

    // 寫回
    dist[(b_i + i)     * n_padded + (b_j + j)]        = val0;
    dist[(b_i + i)     * n_padded + (b_j + (j+T))]    = val1;
    dist[(b_i + i + T) * n_padded + (b_j + j)]        = val2;
    dist[(b_i + i + T) * n_padded + (b_j + (j+T))]    = val3;
}

//==============================================================================
// phase3_kernel: 其餘區塊 (doubly dependent)
//   gridDim = (round-1, round-1)
//   blockIdx.x, blockIdx.y 皆要跳過 r
//------------------------------------------------------------------------------
__global__ void phase3_kernel(int* dist, int n_padded, int r) {
    // s[] 前半存 (b_i, b_k)，後半存 (b_k, b_j)
    __shared__ int s[S2];

    int b_x = blockIdx.x + (blockIdx.x >= r);
    int b_y = blockIdx.y + (blockIdx.y >= r);

    int b_i = b_y * B;  // row block
    int b_j = b_x * B;  // col block
    int b_k = r   * B;  // pivot

    int i = threadIdx.y;
    int j = threadIdx.x;

    // 讀取「自己」區塊
    int val0 = dist[(b_i + i)       * n_padded + (b_j + j)];
    int val1 = dist[(b_i + i)       * n_padded + (b_j + j + T)];
    int val2 = dist[(b_i + i + T)   * n_padded + (b_j + j)];
    int val3 = dist[(b_i + i + T)   * n_padded + (b_j + j + T)];

    // s[] 前半 (b_i, b_k), 後半 (b_k, b_j)
    s[i*B + j]               = dist[(b_i + i)       * n_padded + (b_k + j)];
    s[i*B + (j+T)]           = dist[(b_i + i)       * n_padded + (b_k + (j+T))];
    s[(i+T)*B + j]           = dist[(b_i + i + T)   * n_padded + (b_k + j)];
    s[(i+T)*B + (j+T)]       = dist[(b_i + i + T)   * n_padded + (b_k + (j+T))];

    s[S + i*B + j]           = dist[(b_k + i)       * n_padded + (b_j + j)];
    s[S + i*B + (j+T)]       = dist[(b_k + i)       * n_padded + (b_j + (j+T))];
    s[S + (i+T)*B + j]       = dist[(b_k + i + T)   * n_padded + (b_j + j)];
    s[S + (i+T)*B + (j+T)]   = dist[(b_k + i + T)   * n_padded + (b_j + (j+T))];

    __syncthreads();

    // FW 更新
    #pragma unroll
    for (int k = 0; k < B; k++) {
        val0 = min(val0, s[i*B + k] + s[S + k*B + j]);
        val1 = min(val1, s[i*B + k] + s[S + k*B + (j+T)]);
        val2 = min(val2, s[(i+T)*B + k] + s[S + k*B + j]);
        val3 = min(val3, s[(i+T)*B + k] + s[S + k*B + (j+T)]);
    }

    // 寫回
    dist[(b_i + i)     * n_padded + (b_j + j)]         = val0;
    dist[(b_i + i)     * n_padded + (b_j + j + T)]     = val1;
    dist[(b_i + i + T) * n_padded + (b_j + j)]         = val2;
    dist[(b_i + i + T) * n_padded + (b_j + j + T)]     = val3;
}


int ceilDiv(int a, int b) {
    //nvtxRangePushA("ceilDiv");
    int c = (a + b - 1) / b;
    //nvtxRangePop();
    return c;
}
