#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <fstream>

void write_to_file(const std::string& filename, const std::string& label, const std::vector<float>& vec) {
    std::ofstream outfile;
    outfile.open(filename, std::ios_base::app);
    outfile << label;
    for (const auto& val : vec) {
        outfile << val << " ";
    }
    outfile << std::endl;
    outfile.close();
}

// void print(const std::string& label, const std::vector<float>& vec) {
//     std::cout << label;
//     for (const auto& val : vec) {
//         std::cout << val << " ";
//     }
//     std::cout << std::endl;
// }

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

    int local_size = n / size;
    int remainder = n % size;
    if (n % size != 0 && rank == size - 1) {
        local_size += remainder;
    }

    std::vector<float> local_data(local_size);

    MPI_File input_file;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, rank * (n / size) * sizeof(float), local_data.data(), local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    std::sort(local_data.begin(), local_data.end());
    int skip = 0;
    for (int i = 0; i < size; i++) {
        // Odd phase
        skip = 0;
        if (rank % 2 == 1 && rank + 1 < size) {
            int neighbor_size;
            MPI_Sendrecv(&local_size, 1, MPI_INT, rank + 1, 10,
                         &neighbor_size, 1, MPI_INT, rank + 1, 9,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<float> neighbor_data(neighbor_size);
            MPI_Sendrecv(local_data.data(), local_size, MPI_FLOAT, rank + 1, 11,
                         neighbor_data.data(), neighbor_size, MPI_FLOAT, rank + 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (*(local_data.end() - 1) > *(neighbor_data.begin())) {
                std::vector<float> merged_data(local_size + neighbor_size);
                std::merge(local_data.begin(), local_data.end(), neighbor_data.begin(), neighbor_data.end(), merged_data.begin());
                //std::sort(merged_data.begin(), merged_data.end());
                local_data.assign(merged_data.begin(), merged_data.begin() + local_size);
            }else skip = 1;

        } else if (rank % 2 == 0 && rank > 0) {
            int neighbor_size;
            MPI_Sendrecv(&local_size, 1, MPI_INT, rank - 1, 9,
                         &neighbor_size, 1, MPI_INT, rank - 1, 10,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<float> neighbor_data(neighbor_size);
            MPI_Sendrecv(local_data.data(), local_size, MPI_FLOAT, rank - 1, 1,
                         neighbor_data.data(), neighbor_size, MPI_FLOAT, rank - 1, 11,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (*(neighbor_data.end() - 1) > *(local_data.begin())) {
                std::vector<float> merged_data(local_size + neighbor_size);
                std::merge(local_data.begin(), local_data.end(), neighbor_data.begin(), neighbor_data.end(), merged_data.begin());
                //std::sort(merged_data.begin(), merged_data.end());
                merged_data.erase(merged_data.begin(), merged_data.begin() + neighbor_size);
                local_data.assign(merged_data.begin(), merged_data.begin() + local_size);
            }else skip = 1;
        }

        // Even phase
        if (rank % 2 == 0 && rank + 1 < size) {
            int neighbor_size;
            MPI_Sendrecv(&local_size, 1, MPI_INT, rank + 1, 8,
                         &neighbor_size, 1, MPI_INT, rank + 1, 7,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<float> neighbor_data(neighbor_size);
            MPI_Sendrecv(local_data.data(), local_size, MPI_FLOAT, rank + 1, 12,
                         neighbor_data.data(), neighbor_size, MPI_FLOAT, rank + 1, 4,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (*(local_data.end() - 1) > *(neighbor_data.begin())) {
                std::vector<float> merged_data(local_size + neighbor_size);
                std::merge(local_data.begin(), local_data.end(), neighbor_data.begin(), neighbor_data.end(), merged_data.begin());
                //std::sort(merged_data.begin(), merged_data.end());
                local_data.assign(merged_data.begin(), merged_data.begin() + local_size);
            }else skip = 1;

        } else if (rank % 2 == 1 && rank > 0) {
            int neighbor_size;
            MPI_Sendrecv(&local_size, 1, MPI_INT, rank - 1, 7,
                         &neighbor_size, 1, MPI_INT, rank - 1, 8,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<float> neighbor_data(neighbor_size);
            MPI_Sendrecv(local_data.data(), local_size, MPI_FLOAT, rank - 1, 4,
                         neighbor_data.data(), neighbor_size, MPI_FLOAT, rank - 1, 12,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (*(neighbor_data.end() - 1) > *(local_data.begin())) {
                std::vector<float> merged_data(local_size + neighbor_size);
                std::merge(local_data.begin(), local_data.end(), neighbor_data.begin(), neighbor_data.end(), merged_data.begin());
                //std::sort(merged_data.begin(), merged_data.end());
                merged_data.erase(merged_data.begin(), merged_data.begin() + neighbor_size);
                local_data.assign(merged_data.begin(), merged_data.begin() + local_size);
            }else skip = 1;
        }
        int global_skip;
        MPI_Allreduce(&skip, &global_skip, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (global_skip) {
            // printf("All ranks agree to skip, breaking the loop.\n");
            break;
        }
    }

    MPI_File output_file;
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, rank * (n / size) * sizeof(float), local_data.data(), local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    endtime = MPI_Wtime();

    // printf("rank %d took %f seconds\n", rank, endtime - starttime);

    return 0;
}
