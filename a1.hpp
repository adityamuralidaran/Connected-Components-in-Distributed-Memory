/*  First Name: Aditya Subramanian
 *  Last Name: Muralidaran
 *  UBIT Name: adityasu
 */

#ifndef A1_HPP
#define A1_HPP

#include <vector>
#include <mpi.h>
#include <algorithm>


int connected_components(std::vector<signed char>& A, int n, int q, const char* out, MPI_Comm comm) {
    // ...
    int k = n/q;
    int sol = -1;
	std::vector<signed char> P;
	std::vector<signed char> P1;
	std::vector<signed char> P_prime;
	std::vector<signed char> P_temp;

	std::vector<signed char> M;
    std::vector<signed char> M1;
	std::vector<signed char> Q;
	std::vector<signed char> Q1;
    std::vector<signed char> V;
    std::vector<signed char> Z;

	P.resize(k*k,0);
	P1.resize(k*k,0);
    Z.resize(k*k,0);

    /* start of Init step */
	for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
        	if(A[i*k+j] == 1){
        		P[i*k+j] = i;
        	}
        }
    }


    for (int i = 0; i < k; ++i){
    	signed char max = 0;
    	for (int j = 0; j < k; ++j){
    		if (P[j*k+i] > max){
    			max = P[j*k+i];
    		}
    	}
    	P[i] = max;
    	for (int j = 1; j < k; ++j){
    		P[j*k+i] = P[i];
    	}
    }


	int rank;
    MPI_Comm_rank(comm, &rank);

    int col = rank % q;
    int row = rank / q;

    MPI_Comm col_com;
    MPI_Comm row_com;

    MPI_Comm_split(comm, col, rank, &col_com);
    MPI_Comm_split(comm, row, rank, &row_com);

    MPI_Allreduce(P.data(), P1.data(), k*k, MPI_CHAR, MPI_MAX, col_com);
    /* End of Init step */


    while(P1 != Z) {
        Z = P1;
        /* step 2 */
        /* Forming Helper Matrix 'M' */
        M.resize(k*k,0);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
            	if(A[i*k+j] == 1){
            		M[i*k+j] = P1[i*k+j];
            	}
            }
        }

        /* Local row wise reduce(op:max, row) of 'M' to 'Q' */
        Q.resize(k*k,0);
        for (int i = 0; i < k; ++i){
        	signed char max = 0;
        	for (int j = 0; j < k; ++j){
        		if (M[i*k+j] > max){
        			max = M[i*k+j];
        		}
        	}
        	Q[i*k] = max;
        	for (int j = 1; j < k; ++j){
        		Q[i*k+j] = Q[i*k];
        	}
        }

        /* Global AllReduce(op:max,row) of 'Q' to 'Q1' */
        Q1.resize(k*k,0);
        MPI_Allreduce(Q.data(), Q1.data(), k*k, MPI_CHAR, MPI_MAX, row_com);

        /* Setting helper matrix 'M' based on values of 'Q1' */
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
            	if(Q1[i*k+j] == j){
            		M[i*k+j] = P1[i*k+j];
            	}
            }
        }

        /* Local AllReduce(op:max,row) on 'M' into 'P_temp' */
        P_temp.resize(k*k,0);
        for (int i = 0; i < k; ++i){
        	signed char max = 0;
        	for (int j = 0; j < k; ++j){
        		if (M[i*k+j] > max){
        			max = M[i*k+j];
        		}
        	}
        	P_temp[i*k] = max;
        	for (int j = 1; j < k; ++j){
        		P_temp[i*k+j] = P_temp[i*k];
        	}
        } 

        /* Global AllReduce(op:max,row) of 'P_temp' to 'P_prime' */
        P_prime.resize(k*k,0);
        MPI_Allreduce(P_temp.data(), P_prime.data(), k*k, MPI_CHAR, MPI_MAX, row_com);

        /* step 3 */
        /* Setting 'M1' using 'P_prime' */
        M1.resize(k*k,0);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                if(P1[i*k+j] == i){
                    M1[i*k+j] = P_prime[i*k+j];
                }
            }
        }

        /* Local AllReduce(op:max) on 'M1' */
        for (int i = 0; i < k; ++i){
            signed char max = 0;
            for (int j = 0; j < k; ++j){
                if (M1[i*k+j] > max){
                    max = M1[i*k+j];
                }
            }
            M1[i*k] = max;
            for (int j = 1; j < k; ++j){
                M1[i*k+j] = M1[i*k];
            }
        } 

        /* Global AllReduce(op:max) on 'M1' into 'V' */
        V.resize(k*k,0);
        MPI_Allreduce(M1.data(), V.data(), k*k, MPI_CHAR, MPI_MAX, row_com);

        /* Calculating 'P' */
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                if(P_prime[i*k+j] >= V[i*k+j]) {
                    P_prime[i*k+j] = P_prime[i*k+j];
                }
                else {
                    P_prime[i*k+j] = V[i*k+j];
                }
            }
        }


        /* Switching column and row operations */

        /* Forming Helper Matrix 'M' */
        M.resize(k*k,0);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                if(i==j){
                    M[i*k+j] = P_prime[i*k+j];
                }
            }
        }

        /* Local reduce(op:max,column) of 'M' to 'Q' */
        Q.resize(k*k,0);
        for (int i = 0; i < k; ++i){
            signed char max = 0;
            for (int j = 0; j < k; ++j){
                if (M[j*k+i] > max){
                    max = M[j*k+i];
                }
            }
            Q[i] = max;
            for (int j = 1; j < k; ++j){
                Q[j*k+i] = Q[i];
            }
        }

        /* Global AllReduce(op:max,column) of 'Q' to 'Q1' */
        Q1.resize(k*k,0);
        MPI_Allreduce(Q.data(), Q1.data(), k*k, MPI_CHAR, MPI_MAX, col_com);

        /* Setting helper matrix 'M' based on values of 'Q1' */
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                if(Q1[i*k+j] == i) {
                    M[i*k+j] = P_prime[i*k+j];
                }
            }
        }

        /* Local AllReduce(op:max,column)  on 'M' into 'P_temp' */
        P_temp.resize(k*k,0);
        for (int i = 0; i < k; ++i) {
            signed char max = 0;
            for (int j = 0; j < k; ++j){
                if (M[j*k+i] > max){
                    max = M[j*k+i];
                }
            }
            P_temp[i] = max;
            for (int j = 1; j < k; ++j){
                P_temp[j*k+i] = P_temp[i];
            }
        } 

        /* Global AllReduce(op:max) of 'P_temp' to 'P_prime' */
        P1.resize(k*k,0);
        MPI_Allreduce(P_temp.data(), P1.data(), k*k, MPI_CHAR, MPI_MAX, col_com);
    }

    MPI_Comm_free(&col_com);
    MPI_Comm_free(&row_com);

    
    std::sort(P1.begin(), P1.end());
    sol = std::unique(P1.begin(), P1.end()) - P1.begin();
    


    return sol;
} // connected_components

#endif // A1_HPP
