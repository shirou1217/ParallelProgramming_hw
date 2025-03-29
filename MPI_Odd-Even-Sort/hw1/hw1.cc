#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <fstream>  // For file output
//debug sendrecive
void write_to_file(const std::string& filename, const std::string& label, const std::vector<float>& vec) {
    std::ofstream outfile;
    outfile.open(filename, std::ios_base::app);  // Open the file in append mode
    outfile << label;
    for (const auto& val : vec) {
        outfile << val << " ";
    }
    outfile << std::endl;
    outfile.close();
}
void write_compare_to_file(const std::string& filename, const std::string& label, int rank, const std::string& text1, float local_end, const std::string& text2, float neighbor_begin) {
    std::ofstream outfile;
    outfile.open(filename, std::ios_base::app);  // Open in append mode

    outfile << label << " " << rank << " " << text1 << " " << local_end << " " << text2 << " " << neighbor_begin << "\n";

    outfile.close();
}
void print(const std::string& label, const std::vector<float>& vec) {
    std::cout << label;
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
int main(int argc, char **argv) {
    double starttime, endtime;
	starttime = MPI_Wtime();
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <array_size> <input_file> <output_file>\n", argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    //C++ boost
    // Divide the array into chunks for each process
    int local_size = n / size;
    int remainder = n % size;
    if (n % size != 0 && rank == size - 1) {
        // The last process takes care of any remainder
        local_size += remainder;
    }
    if(rank==size-1) printf("rank%d local_size=%d\n",rank,local_size);
    std::vector<float> local_data(local_size);

    // MPI File Read
    MPI_File input_file;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, rank * (n / size) * sizeof(float), local_data.data(), local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    // Sort the local array using C++ sort
    std::sort(local_data.begin(), local_data.end());
    // if(rank==size-1) {
    //     std::string filename = "output_rank_" + std::to_string(rank) + ".txt";
    //     write_to_file(filename, "Read in data: ", local_data);
    // }
    

    //debug here
    // int skip_odd = 0;
    // int skip_even =0;
  
    if(rank==size-1){
            printf("start odd even sort");
    }
    for(int i=0;i<size;i++){
        // skip_odd = 0;
        // skip_even =0;
    
        //odd
        if (rank % 2 == 1 && rank + 1 < size) {
            int neighbor_size;
            MPI_Recv( &neighbor_size, 1, MPI_INT, rank + 1, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&local_size , 1, MPI_INT, rank + 1, 10, MPI_COMM_WORLD);
            int total_size = neighbor_size + local_size;
            if(rank==size-1){
                   printf("total size variable: %d\n", total_size);
            }
            std::vector<float> neighbor_data(neighbor_size);
            MPI_Recv(neighbor_data.data(), neighbor_size, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(rank==size-1){
                   printf("receive neighbor data\n");
            }
            MPI_Send(local_data.data() , local_size, MPI_FLOAT, rank + 1, 11, MPI_COMM_WORLD);
            if(rank==size-1){
                   printf("send local data data\n");
            }
            // if(*(local_data.end()-1) < *(neighbor_data.begin())) {
            //     skip_odd = 1 ;
            //     std::string filename = "compare.txt";
            //     write_compare_to_file(filename, "rank:", rank ,"local_end:", *(local_data.end()-1) ,"neighbor_begin:", *(neighbor_data.begin()));
            // }
            //printf("rank: %d local_end: %f neighbor_begin: %f\n", rank, *(local_data.end() - 1), *neighbor_data.begin());
            // MPI_Send(&skip_odd , 1, MPI_INT, rank + 1, 3, MPI_COMM_WORLD);
            if(*(local_data.end()-1) > *(neighbor_data.begin())) {
                std::vector<float> merged_data(local_size +neighbor_size);
                std::merge(local_data.begin(), local_data.end(), neighbor_data.begin(), neighbor_data.end(), merged_data.begin());
                if(rank==size-1){
                   printf("finish merge data\n");
               }
                std::sort(merged_data.begin(), merged_data.end()); //????
                local_data.assign(merged_data.begin(), merged_data.begin() + local_size);
            }
            
           
        } else if (rank % 2 == 0 && rank > 0) {
            MPI_Send(&local_size , 1, MPI_INT, rank - 1, 9, MPI_COMM_WORLD);
            int neighbor_size;
            MPI_Recv( &neighbor_size, 1, MPI_INT, rank - 1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int total_size = neighbor_size + local_size;
            std::vector<float> neighbor_data(neighbor_size);
            MPI_Send(local_data.data() , local_size, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
            MPI_Recv( neighbor_data.data(), neighbor_size, MPI_FLOAT, rank - 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(*(neighbor_data.end()-1) > *(local_data.begin())) {
                std::vector<float> merged_data(local_size +neighbor_size);
                std::merge(local_data.begin(), local_data.end(), neighbor_data.begin(), neighbor_data.end(), merged_data.begin());
                std::sort(merged_data.begin(), merged_data.end()); //????
                merged_data.erase(merged_data.begin(), merged_data.begin() + neighbor_size);
                local_data.assign(merged_data.begin(), merged_data.begin() + local_size);
            }
            // int received_skip_odd;
            // MPI_Recv(&received_skip_odd , 1, MPI_INT, rank - 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("Even rank received skip_odd : %d\n",received_skip_odd);
            // if(!received_skip_odd){
                // std::vector<float> received_data(total_size);
                // Receive the second half from odd neighbor
                // MPI_Recv(received_data.data(), total_size, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //printf("Even rank received second half from odd rank.\n");
                // received_data.erase(received_data.begin(), received_data.begin() + neighbor_size);
                // local_data.assign(received_data.begin(), received_data.begin() + local_size);
                // std::copy(received_data.begin() + neighbor_size + 1 ,received_data.end(), local_data.begin());
                //print("Even rank updated local_data: ", local_data);
            // }
          
        }

        //even
        if (rank % 2 == 0 && rank + 1 < size) { 
            int neighbor_size;
            MPI_Recv( &neighbor_size, 1, MPI_INT, rank + 1, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&local_size , 1, MPI_INT, rank + 1, 8, MPI_COMM_WORLD);
            int total_size = neighbor_size + local_size;
            
            // if(rank==size-2){
            //     std::string filename = "output_rank_" + std::to_string(rank) + ".txt";
            //     write_to_file(filename, "local array ", local_data);
            // }
            std::vector<float> neighbor_data(neighbor_size);
            MPI_Recv( neighbor_data.data(), neighbor_size, MPI_FLOAT, rank + 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(local_data.data() , local_size, MPI_FLOAT, rank + 1, 12, MPI_COMM_WORLD);
            // if(rank==size-2){
            //     std::string filename = "output_rank_" + std::to_string(rank) + ".txt";
            //     write_to_file(filename, "neighbor array ", neighbor_data);
            // }
            // if(*(local_data.end()-1) < *(neighbor_data.begin())){
            //     skip_even = 1 ;
            //     std::string filename = "compare.txt";
            //     write_compare_to_file(filename, "rank:", rank ,"local_end:", *(local_data.end()-1) ,"neighbor_begin:", *(neighbor_data.begin()));
            //     // printf("rank: %d local_end: %f neighbor_begin: %f\n", rank, *(local_data.end() - 1), *neighbor_data.begin());
            // } 
            // MPI_Send(&skip_even , 1, MPI_INT, rank + 1, 5, MPI_COMM_WORLD);
            if(*(local_data.end()-1) > *(neighbor_data.begin())) {
                std::vector<float> merged_data(local_size +neighbor_size);
                std::merge(local_data.begin(), local_data.end(), neighbor_data.begin(), neighbor_data.end(), merged_data.begin());
                std::sort(merged_data.begin(), merged_data.end());
                local_data.assign(merged_data.begin(), merged_data.begin() + local_size);
            }
        
                // if(rank==size-2){
                //     std::string filename = "output_rank_" + std::to_string(rank) + ".txt";
                //     write_to_file(filename, "After merge ", merged_data);
                //  }
                //print("After merging: ", merged_data);
                // std::copy(merged_data.begin(), merged_data.begin() + local_size, local_data.begin());
               
                //  // Send the second half of the merged data to the neighbor even rank
                //  if(rank==size-2){
                //     std::string filename = "output_rank_" + std::to_string(rank) + ".txt";
                //     write_to_file(filename, "new data", local_data);
                //  }
                // MPI_Send(merged_data.data() , total_size, MPI_FLOAT, rank + 1, 6, MPI_COMM_WORLD);
                //printf("even rank sent second half to odd rank.\n");

        } else if (rank % 2 == 1 && rank > 0) {
            MPI_Send(&local_size , 1, MPI_INT, rank - 1, 7, MPI_COMM_WORLD);
            // int received_skip_even;
            // MPI_Recv(&received_skip_even , 1, MPI_INT, rank - 1, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("odd rank received skip_even : %d\n",received_skip_even);
            int neighbor_size;
            MPI_Recv( &neighbor_size, 1, MPI_INT, rank - 1, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<float> neighbor_data(neighbor_size);
            MPI_Send(local_data.data() , local_size, MPI_FLOAT, rank - 1, 4, MPI_COMM_WORLD);
            MPI_Recv( neighbor_data.data(), neighbor_size, MPI_FLOAT, rank -1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int total_size = neighbor_size + local_size;
            if(*(neighbor_data.end()-1) > *(local_data.begin())) {
                std::vector<float> merged_data(local_size +neighbor_size);
                std::merge(local_data.begin(), local_data.end(), neighbor_data.begin(), neighbor_data.end(), merged_data.begin());
                std::sort(merged_data.begin(), merged_data.end()); //????
                merged_data.erase(merged_data.begin(), merged_data.begin() + neighbor_size);
                local_data.assign(merged_data.begin(), merged_data.begin() + local_size);
            }
            
            
            // if(rank==size-1){
            //     std::string filename = "output_rank_" + std::to_string(rank) + ".txt";
            //     write_to_file(filename, "local array ",local_data );
            // }
            
            // if(!received_skip_even){
                // std::vector<float> received_data(total_size);
                // Receive the second half from odd neighbor
                // if(rank==size-1){
                //     printf("neighbor_size: %d\n", neighbor_size);
                //     printf("local_size: %d\n", local_size);
                //     printf("receive merge data total size: %d\n", total_size);
                //  }
                // MPI_Recv(received_data.data(), total_size, MPI_FLOAT, rank - 1, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // if(rank==size-1){
                //     std::string filename = "output_rank_" + std::to_string(rank) + ".txt";
                //     write_to_file(filename, "Receive merge data: ", received_data);
                //  }
                //  received_data.erase(received_data.begin(), received_data.begin() + neighbor_size);
                //  if(rank==size-1){
                //     std::string filename = "output_rank_" + std::to_string(rank) + ".txt";
                //     write_to_file(filename, "After earse first half: ", received_data);
                //  }
                //printf("odd rank received second half from even rank.\n");

                // std::copy(received_data.begin()  ,received_data.begin()+ local_size, local_data.begin());
                // local_data.assign(received_data.begin(), received_data.begin() + local_size);
                //print("odd rank updated local_data: ", local_data);
            // }
            // if(rank==size-1){
            //     std::string filename = "output_rank_" + std::to_string(rank) + ".txt";
            //     write_to_file(filename, "new data ", local_data);
            // }
        }

        // Synchronize after each phase
        // MPI_Barrier(MPI_COMM_WORLD);

        // // --- Check if all ranks can skip ---
        // int global_skip_odd, global_skip_even;
        // MPI_Allreduce(&skip_odd, &global_skip_odd, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        // MPI_Allreduce(&skip_even, &global_skip_even, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        // // If all ranks have skip_odd and skip_even as 1, break the loop
        // if (global_skip_odd == 1 && global_skip_even == 1) {
        //     printf("All ranks agree to skip, breaking the loop.\n");
        //     break;
        // }
    }
    // MPI File Write
    MPI_File output_file;
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, rank * (n / size) * sizeof(float), local_data.data(), local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    endtime = MPI_Wtime();

    printf("rank %d took %f seconds\n",rank,endtime - starttime);

    return 0;
}