#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "/home/pp24/pp24s036/firefly/NVTX/c/include/nvtx3/nvtx3.hpp"

// CUDA相關包含
#include <cuda.h>
#include <cuda_runtime.h>

// 函式宣告
void input(char *input_filename);
void output(char *output_filename);
void self_attention(float *q, float *k, float *v, float *o);

void SoftMax(float *out, float *in);
void MulAttV(float *out, float *att, float *v);

// CUDA核函式宣告
__global__ void QKDotAndScalarKernel(float *d_out, const float *d_q, const float *d_k, float scalar, int N, int d);
__global__ void MulAttVKernel(float *d_out, const float *d_att, const float *d_v, int N, int d);

// 輔助函式
float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    double start, end;
    start = getTimeStamp();

    // 對每個批次執行自注意力
    for (int i = 0; i < B; i++) {
        self_attention(
            Q + (i * N * d), 
            K + (i * N * d), 
            V + (i * N * d), 
            O + (i * N * d)
        );
    }

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}

void input(char *input_filename) {
    nvtxRangePushA("input");
    FILE *file = fopen(input_filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open input file %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
    nvtxRangePop();
}

void output(char *output_filename) {
    nvtxRangePushA("output");
    FILE *file = fopen(output_filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open output file %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
    nvtxRangePop();
}

void self_attention(float *q, float *k, float *v, float *o) {
    nvtxRangePushA("self_attention");

    // 分配主機端注意力矩陣
    float *Attn = (float *)malloc(N * N * sizeof(float));
    memset(Attn, 0x00, N * N * sizeof(float));

    // CUDA相關變數
    float *d_q, *d_k, *d_att, *d_v, *d_out;
    size_t size_qk = N * d * sizeof(float);
    size_t size_att = N * N * sizeof(float);
    size_t size_v = N * d * sizeof(float);
    size_t size_o = N * d * sizeof(float);

    cudaError_t err;

    nvtxRangePushA("QKDotAndScalar");
    // 分配裝置端記憶體
    err = cudaMalloc((void **)&d_q, size_qk);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc d_q failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_k, size_qk);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc d_k failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_att, size_att);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc d_att failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 將Q和K拷貝到裝置端
    err = cudaMemcpy(d_q, q, size_qk, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy Q to device failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_k, k, size_qk, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy K to device failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 計算縮放因子
    float scalar = 1.0f / sqrtf((float)d);

    // 定義CUDA核函式的執行配置
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    // 啟動QKDotAndScalarKernel
    QKDotAndScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_att, d_q, d_k, scalar, N, d);

    // 檢查是否有錯誤
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA QKDotAndScalarKernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 將Attn從裝置端拷貝回主機端
    err = cudaMemcpy(Attn, d_att, size_att, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy Attn from device failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    nvtxRangePop();
    // 釋放部分裝置端記憶體
    cudaFree(d_q);
    cudaFree(d_k);

    // 應用 SoftMax
    SoftMax(Attn, Attn);

    nvtxRangePushA("QKDotAndScalar");
    // 現在將SoftMax後的Attn和V傳到裝置端以進行MulAttV
    err = cudaMalloc((void **)&d_v, size_v);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc d_v failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_out, size_o);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc d_out failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 將V和Attn拷貝到裝置端
    err = cudaMemcpy(d_v, v, size_v, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy V to device failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_att, Attn, size_att, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy Attn to device failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 定義MulAttVKernel的執行配置
    int threadsPerBlockMV = 256;
    int blocksPerGridMV = (N * d + threadsPerBlockMV - 1) / threadsPerBlockMV;

    // 啟動MulAttVKernel
    MulAttVKernel<<<blocksPerGridMV, threadsPerBlockMV>>>(d_out, d_att, d_v, N, d);

    // 檢查是否有錯誤
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA MulAttVKernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 將結果從裝置端拷貝回主機端
    err = cudaMemcpy(o, d_out, size_o, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy O from device failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    nvtxRangePop();
    // 釋放裝置端記憶體
    cudaFree(d_att);
    cudaFree(d_v);
    cudaFree(d_out);

    // 釋放主機端記憶體
    free(Attn);
    nvtxRangePop();
}

__global__ void QKDotAndScalarKernel(float *d_out, const float *d_q, const float *d_k, float scalar, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;

    if (idx < total) {
        int i = idx / N; // 查詢位置
        int j = idx % N; // 鍵位置
        float sum = 0.0f;
        for (int t = 0; t < d; t++) {
            sum += d_q[i * d + t] * d_k[j * d + t];
        }
        d_out[idx] = sum * scalar;
    }
}

__global__ void MulAttVKernel(float *d_out, const float *d_att, const float *d_v, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * d;

    if (idx < total) {
        int i = idx / d; // 查詢位置
        int j = idx % d; // 維度位置
        float sum = 0.0f;
        for (int t = 0; t < N; t++) {
            sum += d_att[i * N + t] * d_v[t * d + j];
        }
        d_out[i * d + j] = sum;
    }
}

void SoftMax(float *out, float *in) {
    nvtxRangePushA("SoftMax");
    for (int i = 0; i < N; i++) {
        float max_value = in[i * N];
        for (int j = 0; j < N; j++) {
            max_value = _max(max_value, in[i * N + j]);
        }
        for (int j = 0; j < N; j++) {
            out[i * N + j] = expf(in[i * N + j] - max_value);
        }

        float sum_value = 0.0F;
        for (int j = 0; j < N; j++) {
            sum_value += out[i * N + j];
        }
        for (int j = 0; j < N; j++) {
            out[i * N + j] = out[i * N + j] / sum_value;
        }
    }
    nvtxRangePop();
}
