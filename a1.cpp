#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "a1.hpp"

#ifdef DEBUG
#include <unistd.h>
#endif

void erdos_renyi_2D(std::vector<signed char>& A, int n, int M, int q, MPI_Comm comm, int seed = 13) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int col = rank % q;
    int row = rank / q;

    int b = n / q;

    // expected number of edged per block
    int ne = ((2.0 * M * b * b) / (n * n - n));

    // initialize, we cut corner here: only non-diagonal blocks have edges
    // this affects our estimate (but not too drastically if p is large)
    A.resize(b * b, 0);

    if (col == row) for (int i = 0; i < b; ++i) A[i * b + i] = 1;
    else {
        std::random_device rd;
        if (seed == -1) seed = rd();

        std::mt19937 g(seed);

        for (int i = 0; i < std::min(ne, b * b); ++i) A[i] = 1;
        std::shuffle(std::begin(A), std::end(A), g);
    }

    // make A symmetric 
    // processor (row, col) sends to (col, row)
    int addr = col * q + row;

    if (row < col) MPI_Send(A.data(), b * b, MPI_CHAR, addr, 111, comm);

    if (col < row) {
        std::vector<signed char> buf(b * b);

        MPI_Status stat;
        MPI_Recv(buf.data(), b * b, MPI_CHAR, addr, 111, comm, &stat);

        for (int i = 0; i < b; ++i) {
            for (int j = 0; j < b; ++j) A[j * b + i] = buf[i * b + j];
        }
    } // if col < row

    
    #ifdef DEBUG
    sleep(rank);
    std::cout << "(" << row << "," << col << ")" << std::endl;
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < b; ++j) std::cout << static_cast<int>(A[i * b + j]) << " ";
        std::cout << std::endl;
    }
    #endif
} // erdos_renyi_2D


int main(int argc, char* argv[]) {
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 4) {
        if (rank == 0) {
            std::cout << "usage: [mpiexec -np p] " << argv[0] << " n M outfile" << std::endl;
            std::cout << "p - numver of MPI ranks" << std::endl;
            std::cout << "n - number of graph nodes" << std::endl;
            std::cout << "M - expected number of edges" << std::endl;
            std::cout << "outfile - path to output file" << std::endl;
        }
        return MPI_Finalize();
    }

    // parsing
    int n = std::atoi(argv[1]);
    int M = std::atof(argv[2]);
    std::string out = argv[3];

    int q = std::sqrt(size);
    int p = q * q;

    // sanity check
    if (q < 2) {
        if (rank == 0) std::cout << "too few ranks" << std::endl;
        return MPI_Finalize();
    }

    if (rank == 0) std::cout << "using p=" << p << " ranks" << std::endl;

    if (n % q) {
        if (rank == 0) std::cout << "sqrt(p) must divide n" << std::endl;
        return MPI_Finalize();
    }

    
    if (rank == 0) {
        double np = (n * 2.0 * M) / (n * n - n);
        if (np < 1) std::cout << "connected components likely no larger than: " << std::log10(n) << std::endl;
        if (np < std::log(n)) std::cout << "likely more than one component" << std::endl;
        else std::cout << "likely one component" << std::endl;
    }

    MPI_Comm new_world;
    MPI_Comm_split(MPI_COMM_WORLD, (rank < p), rank, &new_world);

    std::vector<signed char> A;

    if (rank < p) {
        if (rank == 0) std::cout << "preparing graph..." << std::endl;
        erdos_renyi_2D(A, n, M, q, new_world);

        MPI_Barrier(new_world);
        double t0 = MPI_Wtime();

        int cc = connected_components(A, n, q, out.c_str(), new_world);

        MPI_Barrier(new_world);
        double t1 = MPI_Wtime();

        if (rank == 0) {
            std::cout << "#components: " << cc << std::endl;
            std::cout << "time[s]: " << (t1 - t0) << std::endl;
        }
    }

    MPI_Comm_free(&new_world);

    return MPI_Finalize();
} // main
