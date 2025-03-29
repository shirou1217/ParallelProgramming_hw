#include <iostream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
// #include "/home/pp24/pp24s036/firefly///nvtx/c/include///nvtx3///nvtx3.hpp"


static const int INF = ((1 << 30) - 1);
#define BLOCK_SIZE 64  // 以 64x64 為一個 tile


int  n, m;          // 圖的大小、邊的數量 (原始)
int  n_padded;      // 向上對齊到 64 的倍數後的 n
int* Dist_h = nullptr;  // Host 上的 Dist，大小 n_padded * n_padded

// 下面這兩個指標分別是兩張 GPU 上的 Dist
int* Dist_d0 = nullptr;  // GPU #0 上的 Dist
int* Dist_d1 = nullptr;  // GPU #1 上的 Dist


void input(const char* infile);
void output(const char* outfile);
int  ceilDiv(int a, int b);

// 執行 Blocked Floyd-Warshall (兩卡版本)
void block_FW_multiGPU();

// 三個 kernel：將以 64x64 做分塊，每個 block 執行一個子區塊
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

    // 2. 分別在 GPU #0 與 GPU #1 上配置記憶體並拷貝初始資料
    size_t distSize = (size_t)n_padded * (size_t)n_padded * sizeof(int);

    // -- 配置 GPU #0
    cudaSetDevice(0);  // 選擇 GPU0
    cudaMalloc((void**)&Dist_d0, distSize);
    cudaMemcpy(Dist_d0, Dist_h, distSize, cudaMemcpyHostToDevice);

    // -- 配置 GPU #1
    cudaSetDevice(1);  // 選擇 GPU1
    cudaMalloc((void**)&Dist_d1, distSize);
    cudaMemcpy(Dist_d1, Dist_h, distSize, cudaMemcpyHostToDevice);

    // 3. 執行 Blocked Floyd-Warshall (多 GPU 版本)
    block_FW_multiGPU();

    // 4. 把 GPU #0 上的結果拷回 Host（或 #1 也行，因為已同步）
    cudaSetDevice(0);
    cudaMemcpy(Dist_h, Dist_d0, distSize, cudaMemcpyDeviceToHost);

    // 5. 輸出檔案 (只輸出原本 n x n 的部分)
    output(argv[2]);

    // 收尾
    cudaFree(Dist_d0);
    cudaFree(Dist_d1);
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

    // 將 n 向上對齊到 64 的倍數
    // 例如 n=130 時，n_padded 會成為 192 (即 3 * 64)
    n_padded = n + (BLOCK_SIZE - (n % BLOCK_SIZE)) % BLOCK_SIZE;

    // 分配 host memory (大小為 n_padded * n_padded)
    Dist_h = (int*)malloc((size_t)n_padded * (size_t)n_padded * sizeof(int));
    if (!Dist_h) {
        fprintf(stderr, "Host Dist_h malloc failed\n");
        fclose(file);
        exit(1);
    }

    // 初始化:
    for (int i = 0; i < n_padded; i++) {
        for (int j = 0; j < n_padded; j++) {
            if (i < n && j < n) {
                Dist_h[i*n_padded + j] = (i == j) ? 0 : INF;
            } else {
                Dist_h[i*n_padded + j] = INF;
            }
        }
    }

    // 讀取邊資訊，寫入 Dist_h[a, b]
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
        // 把大於等於 INF 的都寫成 INF
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

//==============================================================================
// block_FW_multiGPU: 進行 Blocked Floyd-Warshall，但改成使用兩張 GPU 輪流執行
//   - round = n_padded / 64
//   - 每個 round 先選擇 GPU0 或 GPU1，執行 phase1, phase2, phase3
//   - 執行完後，把更新後的 Dist (整個矩陣) 從該卡拷貝到另一卡，以保持同步
//==============================================================================
void block_FW_multiGPU() {
    //nvtxRangePushA("block_FW_multiGPU");

    int round = n_padded / BLOCK_SIZE;  // 例如 n_padded=192 => round=3

    // kernel 的 thread 設定 (與原程式相同)
    dim3 threads(32, 32);

    for (int r = 0; r < round; r++) {
        // decide which GPU to use this round
        // 輪流: 偶數輪用 GPU0, 奇數輪用 GPU1
        int this_gpu = (r % 2 == 0) ? 0 : 1;
        int other_gpu = 1 - this_gpu;

        cudaSetDevice(this_gpu);

        // Phase1: 對角區塊
        phase1_kernel<<<1, threads>>>( 
            (this_gpu == 0 ? Dist_d0 : Dist_d1), 
            n_padded, 
            r
        );

        // Phase2: 同 row 和同 column 的所有區塊
        if (round > 1) {
            dim3 grid2(2, round - 1);
            phase2_kernel<<<grid2, threads>>>(
                (this_gpu == 0 ? Dist_d0 : Dist_d1), 
                n_padded, 
                r
            );
        }

        // Phase3: 其餘區塊
        if (round > 1) {
            dim3 grid3(round - 1, round - 1);
            phase3_kernel<<<grid3, threads>>>(
                (this_gpu == 0 ? Dist_d0 : Dist_d1), 
                n_padded, 
                r
            );
        }

        // 等待該 GPU 結束這輪計算
        cudaDeviceSynchronize();

        // 接著把更新後的整個 Dist (大小 n_padded*n_padded) 
        // 從 this_gpu 拷貝到 other_gpu，確保兩卡資料一致。
        size_t distSize = (size_t)n_padded * (size_t)n_padded * sizeof(int);
        cudaMemcpyPeer( (other_gpu == 0 ? Dist_d0 : Dist_d1), other_gpu,
                        (this_gpu  == 0 ? Dist_d0 : Dist_d1), this_gpu,
                        distSize );
    }

    //nvtxRangePop();
}

//==============================================================================
// phase1_kernel: 對角區塊 (64x64) 的 Floyd-Warshall
//   blockIdx.x = blockIdx.y = 0 => 只處理第 r 個對角
//------------------------------------------------------------------------------
__global__ void phase1_kernel(int* dist, int n_padded, int r) {
    __shared__ int s[BLOCK_SIZE*BLOCK_SIZE];

    int b_i = r << 6;  // r*64
    int b_j = r << 6;  // r*64

    int i = threadIdx.y;
    int j = threadIdx.x;

    // 載入 4 個位置
    s[i*64 + j]             = dist[(b_i + i)*n_padded + (b_j + j)];
    s[i*64 + (j + 32)]      = dist[(b_i + i)*n_padded + (b_j + j + 32)];
    s[(i + 32)*64 + j]      = dist[(b_i + i + 32)*n_padded + (b_j + j)];
    s[(i + 32)*64 + (j+32)] = dist[(b_i + i + 32)*n_padded + (b_j + j + 32)];

    // 開始 FW
    #pragma unroll
    for (int k = 0; k < 64; k++) {
        __syncthreads();
        s[i*64 + j] = min(s[i*64 + j], 
                          s[i*64 + k] + s[k*64 + j]);
        s[i*64 + (j+32)] = min(s[i*64 + (j+32)], 
                               s[i*64 + k] + s[k*64 + (j+32)]);
        s[(i+32)*64 + j] = min(s[(i+32)*64 + j], 
                               s[(i+32)*64 + k] + s[k*64 + j]);
        s[(i+32)*64 + (j+32)] = min(s[(i+32)*64 + (j+32)], 
                                    s[(i+32)*64 + k] + s[k*64 + (j+32)]);
    }

    // 寫回
    dist[(b_i + i)*n_padded + (b_j + j)]             = s[i*64 + j];
    dist[(b_i + i)*n_padded + (b_j + j + 32)]        = s[i*64 + (j + 32)];
    dist[(b_i + i + 32)*n_padded + (b_j + j)]        = s[(i + 32)*64 + j];
    dist[(b_i + i + 32)*n_padded + (b_j + j + 32)]   = s[(i + 32)*64 + (j + 32)];
}

//==============================================================================
// phase2_kernel: 同 row 與同 column 的區塊
//   gridDim = (2, round-1)
//------------------------------------------------------------------------------
__global__ void phase2_kernel(int* dist, int n_padded, int r) {
    __shared__ int s[2 * BLOCK_SIZE * BLOCK_SIZE];

    int actual_b = blockIdx.y + (blockIdx.y >= r);

    int b_k = r << 6;
    int b_i = (blockIdx.x ? r         : actual_b) << 6;
    int b_j = (blockIdx.x ? actual_b : r        ) << 6;

    int i = threadIdx.y;
    int j = threadIdx.x;

    // 讀取當前 tile
    int val0 = dist[(b_i + i)*n_padded + (b_j + j)];
    int val1 = dist[(b_i + i)*n_padded + (b_j + j + 32)];
    int val2 = dist[(b_i + i + 32)*n_padded + (b_j + j)];
    int val3 = dist[(b_i + i + 32)*n_padded + (b_j + j + 32)];

    // s[0..4095] => (b_i,b_k), s[4096..8191] => (b_k,b_j)
    s[i*64 + j] = dist[(b_i + i)*n_padded + (b_k + j)];
    s[i*64 + (j+32)] = dist[(b_i + i)*n_padded + (b_k + (j+32))];
    s[(i+32)*64 + j] = dist[(b_i + i + 32)*n_padded + (b_k + j)];
    s[(i+32)*64 + (j+32)] = dist[(b_i + i + 32)*n_padded + (b_k + (j+32))];

    s[4096 + i*64 + j] = dist[(b_k + i)*n_padded + (b_j + j)];
    s[4096 + i*64 + (j+32)] = dist[(b_k + i)*n_padded + (b_j + (j+32))];
    s[4096 + (i+32)*64 + j] = dist[(b_k + i + 32)*n_padded + (b_j + j)];
    s[4096 + (i+32)*64 + (j+32)] = dist[(b_k + i + 32)*n_padded + (b_j + (j+32))];

    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 64; k++) {
        val0 = min(val0, s[i*64 + k] + s[4096 + k*64 + j]);
        val1 = min(val1, s[i*64 + k] + s[4096 + k*64 + (j+32)]);
        val2 = min(val2, s[(i+32)*64 + k] + s[4096 + k*64 + j]);
        val3 = min(val3, s[(i+32)*64 + k] + s[4096 + k*64 + (j+32)]);
    }

    dist[(b_i + i)*n_padded + (b_j + j)] = val0;
    dist[(b_i + i)*n_padded + (b_j + (j+32))] = val1;
    dist[(b_i + i + 32)*n_padded + (b_j + j)] = val2;
    dist[(b_i + i + 32)*n_padded + (b_j + (j+32))] = val3;
}

//==============================================================================
// phase3_kernel: 其餘區塊 (doubly dependent)
//   gridDim = (round-1, round-1)
//------------------------------------------------------------------------------
__global__ void phase3_kernel(int* dist, int n_padded, int r) {
    __shared__ int s[2 * BLOCK_SIZE * BLOCK_SIZE];

    int b_x = blockIdx.x + (blockIdx.x >= r);
    int b_y = blockIdx.y + (blockIdx.y >= r);

    int b_i = b_y << 6;
    int b_j = b_x << 6;
    int b_k = r   << 6;

    int i = threadIdx.y;
    int j = threadIdx.x;

    // 當前 tile (4 個值)
    int val0 = dist[(b_i + i)*n_padded + (b_j + j)];
    int val1 = dist[(b_i + i)*n_padded + (b_j + j + 32)];
    int val2 = dist[(b_i + i + 32)*n_padded + (b_j + j)];
    int val3 = dist[(b_i + i + 32)*n_padded + (b_j + j + 32)];

    // 前半 (b_i, b_k)，後半 (b_k, b_j)
    s[i*64 + j] = dist[(b_i + i)*n_padded + (b_k + j)];
    s[i*64 + (j+32)] = dist[(b_i + i)*n_padded + (b_k + (j+32))];
    s[(i+32)*64 + j] = dist[(b_i + i + 32)*n_padded + (b_k + j)];
    s[(i+32)*64 + (j+32)] = dist[(b_i + i + 32)*n_padded + (b_k + (j+32))];

    s[4096 + i*64 + j] = dist[(b_k + i)*n_padded + (b_j + j)];
    s[4096 + i*64 + (j+32)] = dist[(b_k + i)*n_padded + (b_j + (j+32))];
    s[4096 + (i+32)*64 + j] = dist[(b_k + i + 32)*n_padded + (b_j + j)];
    s[4096 + (i+32)*64 + (j+32)] = dist[(b_k + i + 32)*n_padded + (b_j + (j+32))];

    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 64; k++) {
        val0 = min(val0, s[i*64 + k] + s[4096 + k*64 + j]);
        val1 = min(val1, s[i*64 + k] + s[4096 + k*64 + (j+32)]);
        val2 = min(val2, s[(i+32)*64 + k] + s[4096 + k*64 + j]);
        val3 = min(val3, s[(i+32)*64 + k] + s[4096 + k*64 + (j+32)]);
    }

    dist[(b_i + i)*n_padded + (b_j + j)] = val0;
    dist[(b_i + i)*n_padded + (b_j + j + 32)] = val1;
    dist[(b_i + i + 32)*n_padded + (b_j + j)] = val2;
    dist[(b_i + i + 32)*n_padded + (b_j + j + 32)] = val3;
}


int ceilDiv(int a, int b) {
    //nvtxRangePushA("ceilDiv");
    int c = (a + b - 1) / b;
    //nvtxRangePop();
    return c;
}
