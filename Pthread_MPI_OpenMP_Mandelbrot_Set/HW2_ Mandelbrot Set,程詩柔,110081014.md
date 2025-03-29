---
title: 'HW2: Mandelbrot Set,程詩柔,110081014'

---

# HW2: Mandelbrot Set
## Implementation
### pthread
- 取得現在可使用的thread總數建立對應數量的threads[]thread_data[]
![image](https://hackmd.io/_uploads/ByekyzSWkl.png)
![image](https://hackmd.io/_uploads/H1RbJfBbye.png)
- 將height平分給所有threads，讓所有thread負責一段height區間也就是start_row和end_row區間，並且將width,height,iters,left,right,lower,upper,image都傳給每個thread的thread_data。並呼叫 mandelbrot_thread function計算出每個thread自己的image
![image](https://hackmd.io/_uploads/ByjWlGr-1l.png)
- 每個thread在自己的start_row和end_row區間運算mandelbrot，並將運算結果存在data->image中
![image](https://hackmd.io/_uploads/BJlPZfSZJg.png)
- 使用pthread_join將所有thread執行結果合併後呼叫write_png()
![image](https://hackmd.io/_uploads/Bk55bGrbyg.png)
![image](https://hackmd.io/_uploads/SyTRWfBbkx.png)
### hybrid
#### MPI
- 將 height平分給所有process做運算，每個rank處理自己的local_height區間也就是分配到的start_row和end_row區間，若遇到沒辦法整除的情況，將交由最後一個rank處理，最後計算結果存在local_image中。
![image](https://hackmd.io/_uploads/BJPVVGB-kx.png)
- 每個MPI根據自己分配到的區間計算mandelbrot set並將結果存在local_image中
![image](https://hackmd.io/_uploads/r1glBMSW1e.png)
- 透過MPI_Gather將每個rank的local height記錄在local_heights中
![image](https://hackmd.io/_uploads/Sy2qDMSbJe.png)
- 計算recvcounts紀錄每個rank算出來的image大小，計算displs紀錄每個rank的起始位置，使用MPI_Gather合併來自所有rank的local_image, 由rank0呼叫write_png()
![image](https://hackmd.io/_uploads/SJFJFGBbJe.png)
#### OpenMP
- 使用`#pragma omp parallel for schedule(static)`平行化mandelbrot外層for loop以及write_png()內層for loop
- ![image](https://hackmd.io/_uploads/HkWuSQHZ1l.png)
- ![image](https://hackmd.io/_uploads/HJH-U7Bbye.png)



## Methodology
### System Spec
- QCT server,INTEL(R) XEON(R) PLATINUM 8568Y+
### Performance Metrics
- 使用intel vtune進行profile
## Plots: Scalability & Load Balancing & Profile
### Experimental Method
- Test Case Description:使用strict23.txt作為testcase data
- Parallel Configurations:使用一個process 12個threads
### Performance Measurement
- hw2a : 16.391s
- hw2b : 17.091s
### Analysis of Results
#### hw2a
- ![image](https://hackmd.io/_uploads/B1pYImrb1x.png)
- ![image](https://hackmd.io/_uploads/BJBnI7SW1l.png)
- time profile (固定process=1改變threads總數)
![chart (2)](https://hackmd.io/_uploads/SkUmfl8Z1l.png)
- strong scalability(固定process=1改變threads總數)
![chart (3)](https://hackmd.io/_uploads/SyKIMgLWJl.png)
- load balance(1 process and 12 threads)
![chart (6)](https://hackmd.io/_uploads/ry8MYl8Z1x.png)

#### hw2b
- ![image](https://hackmd.io/_uploads/BJuoNXHbyg.png)
- ![image](https://hackmd.io/_uploads/rycA47SZJe.png)
- 由profile結果可以得知mandelbrot function的執行時間最久為bottlenecks
- time profile(固定threads=12改變process總數)
![chart (4)](https://hackmd.io/_uploads/B177UeUbkg.png)
- strong scalability(固定threads=12改變process總數)
![chart (5)](https://hackmd.io/_uploads/SJ448xLbkg.png)
- time profile(固定process=1改變threads總數)
![chart (8)](https://hackmd.io/_uploads/Hk1KV-8byl.png)
- strong scalability(固定process=1改變threads總數)
![chart (9)](https://hackmd.io/_uploads/SJ5n4Z8Wyl.png)
- load balance(6 processes and 12 threads)
![chart (7)](https://hackmd.io/_uploads/rkeJ1ZU-Jg.png)

### discussions 
- scalabilty: 根據hw2a,hw2b三項實驗結果發現，當所使用的threads,process總數增加，運算時間越短，scalabilty表現佳
- load balance : 可以發現hw1a,hw1bload到每個threads和process的工作量並沒有非常平均，這是因為我的implementation方式是讓最後一個thread/rank處理無法整除多出來的工作量，因此都是最後一個thread/rank執行時間會比較久。或許可以改用Dynamic Work Distribution的方式，讓已經提前做完的thread/rank去分擔其他thread還沒做完的工作。

### Optimization Strategies

- 經過vtune的profile結果可以得知，不論是hw2a或hw2b performance bottlenexks都是mandelbrot set function，因此決定以AVX512 SIMD instruction來加速該function中的數學運算。

- 內層迴圈用於處理每一行中的像素。由於使用 SIMD 指令集 AVX-512，每次迴圈處理 8 個點（像素），因此 i 每次遞增 8。
![image](https://hackmd.io/_uploads/S1FO3GSbyl.png)
- 使用 _mm512_set_pd 指令初始化 AVX-512 向量 x0，這個向量包含了 8 個連續的複數實部值，每個值對應於圖像中一個像素的 x 坐標。
![image](https://hackmd.io/_uploads/ByZ9hzBZJl.png)
- y0_vec 使用 _mm512_set1_pd 初始化，將 y0 的值複製到 SIMD 向量中，這樣 8 個元素都擁有相同的 y 坐標。x 和 y 分別初始化為 0，表示初始的複數值為 (0, 0)。length_squared 也初始化為 0，用來儲存 x 和 y 的平方和。repeats 初始化為 0，將用來計算每個點的迭代次數。
![image](https://hackmd.io/_uploads/Sk_NCMBWJg.png)
- 迴圈執行 Mandelbrot 集合的最大迭代次數。每次迭代中，計算每個點的下一個複數值，並檢查其是否發散。
![image](https://hackmd.io/_uploads/ByQXeXrWJe.png)
- 使用 _mm512_mul_pd 指令來同時計算 8 個點的 x^2 和 y^2
- _mm512_add_pd 用於計算 8 個點的 x^2 + y^2，這是用來判斷每個點是否發散的標準。
- _mm512_cmp_pd_mask 用來比較 length_squared 是否小於 4。如果小於 4，則該點還在集合內部。這會返回一個遮罩 mask，用來標記哪些點需要繼續迭代。
- 如果 mask 為 0，表示所有 8 個點都已經發散，則可以提前結束迭代，節省計算時間。
![image](https://hackmd.io/_uploads/HyXwgQrbyl.png)
- xy 計算 x * y。y = 2 * x * y + y0 使用 _mm512_fmadd_pd 完成，這是一個融合乘加指令，減少計算次數。x = x^2 - y^2 + x0 計算下一步的 x 值。
![image](https://hackmd.io/_uploads/HyE5eXSW1x.png)
- 使用 mask 來控制只有那些還未發散的點的 repeats 計數器加 1。這樣可以準確記錄每個點在發散之前的迭代次數。
![image](https://hackmd.io/_uploads/S1tox7HWyg.png)
- 使用 _mm512_store_epi32 將 SIMD 向量中的 8 個迭代次數存入普通陣列 repeats_array，以便後續儲存到圖像緩衝區中。將repeats_array 中的迭代次數值複製到圖像的緩衝區 data->image 中，以便後續生成圖像。這裡檢查 (i + k) < data->width 以避免超出圖像寬度的邊界。
![image](https://hackmd.io/_uploads/SJaybXSWJx.png)
### non-SIMD vs SIMD 
- hw2a
![chart (11)](https://hackmd.io/_uploads/BkN39-L-1g.png)
- hw2b
![6 processes 12 threads](https://hackmd.io/_uploads/r1O0j-U-1l.png)

- 根據前後對比可以發現，使用AVX512優化mandelbrot set function數學運算確實可以有很大幅度的效能優化!!!

## Experience & Conclusion
### Your conclusion of this assignment.
- 此次作業的scalablity佳，可以透過使用更多的process/thread得到更佳的效能表現
- 此次作業的效能瓶頸為mandelbrot set function，透過AVX512一次運算多筆資料也可以有大幅的效能提升
### What have you learned from this assignment?
- 學會使用pthread,MPI,OpenMP跨process,thread運算
- 學會使用AVX512優化效能
### What difficulties did you encounter in this assignment?
- implementent SIMD時花比較久時間研究。
