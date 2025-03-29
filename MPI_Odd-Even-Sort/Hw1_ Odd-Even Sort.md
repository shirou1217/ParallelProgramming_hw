---
title: 'Hw1: Odd-Even Sort'

---

# Hw1: Odd-Even Sort,程詩柔,110081014
## Implementation
### **How do you handle an arbitrary number of input items and processes?**

- 首先將input array size 平分給所有rank，若遇到沒法整除的情況，會讓最後一個rank多處理餘數(remainder)筆的資料
- ![image](https://hackmd.io/_uploads/r1VM86XZkx.png)

### **How do you sort in your program?**
- 當每個拿到分配到的local_data時會先使用std::sort進行sorting，接著會進到執行odd-even-sort的for loop
- odd-even-sort 會在 for loop i=1 to size內完成，for loop 內則會先進行odd sort 再進行even sort
- odd rank 及 even rank 會使用 MPI_Send以及MPI_Recv 將自己local_data,local_size傳給對方，並將對方傳來的資料存為neighbor data,neighbor_size
![image](https://hackmd.io/_uploads/HkVaG-Bb1e.png)


- 將neighbor data與自己的local_data使用std::merge成一個array並做sorting，以odd-sort為例，merge完後odd rank會取merge_data的前半，而even rank會取merge data的後半，evensort則反之。
- ![image](https://hackmd.io/_uploads/rJBadTXWke.png)
- ![image](https://hackmd.io/_uploads/ByGC_pQZ1x.png)
### **Other efforts you’ve made in your program.**
- 除此之外，這裡有個小優化，由於每個rank的local_data一定都是已經排序好的狀態，因此可以在進行merge sort前先判斷，，前面rank local_data最後一個值是否小於後面rank local_data的第一個值，如果是，代表這兩個相鄰rank已經sort好了不需要再merge。
- ![image](https://hackmd.io/_uploads/BkSQ7brZkg.png)



## Experiment & Analysis
### System Spec 
- QCT server,INTEL(R) XEON(R) PLATINUM 8568Y+
### Performance Metrics:
- 使用intel aps進行profile
- 將MPI_Send,MPI_Recv視為Comm time
- MPI_File_write_at,MPI_File_read_at,MPI_File_close,MPI_File_open視為I/O time
- 剩下的Elapsed Time - Comm time - I/O time視為CPU time
- 透過`aps-report aps_result -f` 可以印出總rank 執行 MPI Function 的時間，因此只要將 Time/(總rank數)就是該MPI Function實際在Elapsed Time中所佔的執行時間。
- ![SR_N1n12](https://hackmd.io/_uploads/r14Tm-SW1g.png)
### Experimental Method: 
- Test Case Description: 使用testcases中 33.in，因為array size最大，執行時間最久，比較能看出performance上的變化
- Parallel Configurations: 1 node 12 processes

### Performance Measurement
- execution time 7.27s
- communication time 2.82s
- IO time 2.22s
- CPU time 2.23s
### Analysis of Results
![image](https://hackmd.io/_uploads/HJMEaMUbJl.png)
![image](https://hackmd.io/_uploads/HJUBpGLb1l.png)
- ==**MPI Time占比時間最久，其中又以MPI_Recv執行時間最久為bottlenecks**==
- Time profile 
![chart (12)](https://hackmd.io/_uploads/H1IlOfUZ1e.png)
- Speedup
![chart (13)](https://hackmd.io/_uploads/ryOfdzIbye.png)


### Optimization Strategies
- odd rank 及 even rank 改成使用 MPI_Sendrecv() 傳輸資料
- ![image](https://hackmd.io/_uploads/S1Qe_6mZ1x.png)
- 並且在for loop的最後加上MPI_Allreduce()判斷是否每個rank都跳過skip==1不需要merge，如果是我們就可以提前break結束迴圈。
- ![image](https://hackmd.io/_uploads/Hyq_9TQWJe.png)

#### before(MPI_Send&MPI_Recv) vs After(MPI_Sendrecv&MPI_Allreduce)
![1 node 96 processes](https://hackmd.io/_uploads/Sy1eafLZ1l.png)
- ==改成MPI_Sendrecv()能夠有效降低一些MPI_Comm的時間，達到約2s的效能優化==
## Discussion
- 此次作業MPI的Communication為bottlenecks
- 就scalability的結果圖來看，scale差，雖然從process數量從12增加到24，runtime有稍微降低，但接著用越多process反而會變慢，MPI Comm越久。
- 如果能節省MPI_Sendrecv 的Comm時間效能上或許就能有更大的進展，以下列出兩點或許可以改進的空間:
    - 使用非阻塞通信 (Non-blocking Communication)MPI_Isend 和 MPI_Irecv 非阻塞的發送和接收操作，允許rank在等待數據傳輸完成的同時繼續執行其他操作。這可以減少等待時間並提高程序的並行性。
    - 使用 MPI_Allgather 聚集所有資料若數據量不大，另一種方法是使用 MPI_Allgather 將所有rank的資料收集到每個rank上。這樣每個rank都能獲得完整的資料，無需逐對交換數據。
## Experiences / Conclusion
### Your conclusion of this assignment.
- 本次作業的scale並不好，使用越多process反而會使MPI溝通時間越久降低效能
- 單獨呼叫一次MPI_SendRecv的溝通成本時間比分別呼叫MPI_Send和MPI_Recv來的低
### What have you learned from this assignment?
- 學會odd-even-sort演算法邏輯、更熟悉MPI programming
- 知道如runtime還能再細分為CPU,Comm,I/O time做更進一步的分析
- 知道該如何用aps分析MPI function執行時間
### What difficulties did you encounter in this assignment?
- 一開始沒有使用c++的std:sort都自己做sequential sorting導致程式運行很慢
- 研究讓兩個rank該如何交換data蠻久，由於我的作法是讓最後一個rank處理所以剩餘的部分，所以最後一個rank的data傳輸必須特例處理，因為他拿到的data size比較大，起初這邊沒有處理好，和解答對比時會發現後面排序輸出都是0。

