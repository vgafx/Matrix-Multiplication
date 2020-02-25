/* Main
 * Multiplies a M x N matrix named A with a N x P matrix named B.
 * The results are store in a M x P matrix named C. Note that A and B
 * must have a 'common' dimension. Sensible datatype should be used
 * for templated functions.
*/
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <omp.h>
#include "ScopedTimer.h"
#include "mm_kernels.h"

enum m_name{
    MA = 0 , MB, MC
};

template<typename T>
void PrintMatrix( T* data, int height, int width) {
    std::cout << "Matrix data: \n";
    //std::cout << std::setprecision(16);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << data[i* width +j] << " ";
        }
        std::cout << "\n";
    }
}


template<typename T>
void InitMatrix(T* matrix,  int height, int width, m_name name){
    T lower_bound = 1;
    T upper_bound = 8;
    std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
    std::default_random_engine re;

    switch (name){
    case MA:
        for (int i = 0; i < width * height; ++i) {
            matrix[i] = unif(re);
        }
        break;
    case MB:
        for (int i = 0; i < width * height; ++i) {
            matrix[i] = unif(re);
        }
        break;
    case MC:
        for (int i = 0; i < width * height; ++i) {
            matrix[i] = T(0);
        }
        break;
    }
}


template<typename T>
void TransposeMatrix(T* data, T* data_t, int height, int width){
    //N x P
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            data_t[j * height +i] = data[i * width +j];
        }
    }
}


template<typename T>
int CompareMatrices(T* data1, T* data2, int height, int width){
    int diff = 0;

    for (int i = 0; i < height * width; ++i) {
        if (data1[i] != data2[i]){
            //std::cout << "Diff at: " << i << "\n";
            diff++;
        }
    }

    return diff;
}


int main()
{
    using myType = double;

    /*Set input*/
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int P = 4096;
    std::srand(42);
    constexpr int threads = 4;
    omp_set_num_threads(threads);

    /*The size of the cacheline in bytes*/
    constexpr int CL_SIZE = 64;

    /*How many elements we can fit per CL*/
    constexpr int E_P_CL = CL_SIZE / sizeof (myType);

    /*How many doubles we can load*/
    constexpr int SIMD_WIDTH = 256 / (sizeof (myType)*8);

    /*Switch between E_P_CLINE and SIMD_WIDTH*/
    constexpr int ELEMENTS = SIMD_WIDTH;

    /*Heap-allocate the arrays so that
     *we can test for larger input if desired.*/
    myType* A = new myType[M*N];
    myType* B = new myType[N*P];
    myType* BT = new myType[N*P];
    myType* C = new myType[M*P];

    m_name name_a = MA;
    InitMatrix<myType>(A, M, N, name_a);

    m_name name_b = MB;
    InitMatrix<myType>(B, N, P, name_b);

    m_name name_c = MC;
    InitMatrix<myType>(C, M, P, name_c);
    InitMatrix<myType>(BT, N, P, name_c);

    TransposeMatrix<myType>(B, BT, N, P);

    constexpr int M_END = (M / ELEMENTS) * ELEMENTS;
    constexpr int N_END = (N / ELEMENTS) * ELEMENTS;
    constexpr int P_END = (P / ELEMENTS) * ELEMENTS;
    std::cout << "M_END: " << M_END << ", N_END: " << N_END << ", P_END: " << P_END << "\n";

    /*Calculate leftovers(if any)*/
    constexpr int L_M = M % ELEMENTS;
    constexpr int L_N = N % ELEMENTS;
    constexpr int L_P = P % ELEMENTS;
    std::cout << "L_M: " << L_M << ", L_N: " << L_N << ", L_P: " << L_P << "\n";


    /*Select the version to be timed*/
    std::string version_name = "Transposed";

    {
        ScopedTimer timer(version_name);
        //mm_sequential<double>(M, N, P, A, B, C);
        //mm_seq_transposed<myType>(M, N, P, A, BT, C);
        //mm_seq_tiled<myType>(M, N, P, E_P_CL, A, B, C);
        //mm_seq_tiled_opt<myType>(M, N, P, M_END, N_END, P_END, E_P_CL, A, B, C);
        //mm_seq_tiled_squared<myType>(M, N, P, E_P_CL, A, B, C);
        //mm_seq_SIMD<myType>(M, N, P, M_END, N_END, P_END, SIMD_WIDTH, A, B, C);
        //mm_SIMD(M, N, P, N_END, SIMD_WIDTH, A, BT, C);
        mm_SIMD_OMP(M, N, P, N_END, SIMD_WIDTH, A, BT, C);
    }

    //PrintMatrix(C, M, P);

    /*Extra array for verification against sequential*/
    myType* D = new myType[M*P];
    InitMatrix(D, M, P, name_c);

    {
        version_name = "Verify";
        ScopedTimer timer(version_name);
        mm_sequential<myType>(M, N, P, A, B, D);
    }

    /*Check the results*/
    int res = CompareMatrices(C, D, M, P);
    //PrintMatrix(D, M, P);
    if (!res){
        std::cout << "Results OK!\n";
    } else {
        std::cout << "ERROR: found " << res << " differencies in verification\n";
    }

    /*Clean up*/
    delete[] A;
    delete[] B;
    delete[] BT;
    delete[] C;
    delete[] D;
    return 0;

}
