/* Different implementations of Matrix Multiplication. Most functions
 * are templated so switching between datatypes is easier.
 *
*/
#pragma once
#include "assert.h"
#include <immintrin.h>
#include <emmintrin.h>
#include <omp.h>
#include <iostream>

/*Baseline*/
template<typename T>
void mm_sequential(const int m, const int n, const int p, T* __restrict A, T* __restrict B, T* __restrict C){
    int i, j, k;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < p; ++j) {
            T temp = 0;
            for (k = 0; k < n; ++k) {
                temp += A[i*n+k]*B[k*p+j];
            }
            C[i*p+j] = temp;
        }
    }

};


/*Assumes that Matrix B is transposed*/
template<typename T>
void mm_seq_transposed(const int m, const int n, const int p, T* __restrict A, T* __restrict B, T* __restrict C){
    int i, j, k;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < p; ++j) {
            T temp = 0.0;
            for (k = 0; k < n; ++k) {
                temp += A[i*n+k]*B[k+j*n];
            }
            C[i*p+j] = temp;
        }
    }

};



/*Tiled implementation for better data reuse.
 *Works for all sizes.
 */
template<typename T>
void mm_seq_tiled(const int m, const int n, const int p, const int el, T* __restrict A, T* __restrict B, T* __restrict C){
    int i, j, k;
    int x, y, z;

    for (i = 0; i < m; i += el) {
        for (j = 0; j < p; j += el) {
            for (k = 0; k < n; k += el) {
                for (x = i; x < std::min(i + el, m); ++x) {
                    for (y = j; y < std::min(j + el, p); ++y) {
                        for (z = k; z < std::min(k + el, n); ++z) {
                            C[x * p + y] += A[x * n + z ] * B[z * p  + y];
                        }
                    }
                }
            }
        }
    }


};



/*Assumes m, p, n are all perfectly divisible by 'el'.In the case of doubles:
 * el = CL_BYTES / sizeof(double) i.e. el = 8 so this only works for
 * sizes where m(n,p) % 8 == 0
*/
template<typename T>
void mm_seq_tiled_opt(const int m, const int n, const int p, const int m_end, const int n_end, const int p_end,
                      const int el, T* __restrict A, T* __restrict B, T* __restrict C){
    int i, j, k;
    int x, y, z;

    for (i = 0; i < m_end; i += el) {
        for (j = 0; j < p_end; j += el) {
            for (k = 0; k < n_end; k += el) {
                for (x = i; x < i + el; ++x) {
                    for (y = j; y < j + el; ++y) {
                        for (z = k; z < k + el; ++z) {
                            C[x * p + y] += A[x * n + z ] * B[z * p  + y];
                        }
                    }
                }
            }
        }
    }
};


/*Only works for matrix sizes divisible by 'el'. In the case of doubles:
 * el = CL_BYTES / sizeof(double) i.e. el = 8 so this only works for
 * sizes where m(n,p) % 8 == 0
 */
template<typename T>
void mm_seq_tiled_squared(const int m, const int n, const int p, const int el, T* __restrict A, T* __restrict B, T* __restrict C){
    int i, j, k;
    int i2, j2, k2;

    T* __restrict ma, * __restrict mb, * __restrict mc;

    for (i = 0; i < m; i += el) {
        for (j = 0; j < p; j += el) {
            for (k = 0; k < n; k += el) {
                for (i2 = 0, mc = &C[i*p+j], ma = &A[i*n+k]; i2 < el; ++i2, mc += m, ma += m) {
                    for (k2 = 0, mb = &B[k*p+j]; k2 < el; ++k2, mb +=n) {
                        for (j2 = 0; j2 < el; ++j2) {
                            mc[j2] += ma[k2] * mb[j2];
                        }
                    }
                }
            }
        }
    }

};



/*Hand vectorized SIMD version. Matrix B is expected to be transposed.
 *Essentially multiplies rows of A with rows of B. Some percision loss
 * was noticed: ~@ the 15th digit. Would have to investigate further.
 * Works for all matrix sizes.
 * TO-DO: Implement variant that does not require horizontal additions.
*/
void mm_SIMD(const int m, const int n, const int p, const int n_end, const int el,
                        double* __restrict A, double* __restrict B, double* __restrict C){
    int i, j, k, l;
    double buffer[4];

    for (i = 0; i < m; ++i) {
        for (j = 0; j < p; ++j) {
            __m256d dst1 = _mm256_set1_pd(0.0);
            for (k = 0; k < n_end; k+= el) {
                __m256d A_values1 = _mm256_loadu_pd(&A[i * n + k]);
                __m256d B_values1 = _mm256_loadu_pd(&B[k + j * n]);
                dst1 += _mm256_mul_pd(A_values1, B_values1);
            }

            __m128d dst_a = _mm256_extractf128_pd(dst1, 0);
            __m128d dst_b = _mm256_extractf128_pd(dst1, 1);

            _mm_storel_pd(&buffer[0], dst_a);
            _mm_storeh_pd(&buffer[1], dst_a);
            _mm_storel_pd(&buffer[2], dst_b);
            _mm_storeh_pd(&buffer[3], dst_b);
            for (int b = 0; b < 4; ++b) {
                C[i * n + j] += buffer[b];
            }

            for (int l = n_end; l < n; ++l) {
                C[i*p+j] += A[i*n+l]*B[l+j*n];
            }
        }
    }
};


/*Similar to the mm_SIMD version with the outer loops parallelised with OpenMP.
 *The number of threads and the affinity should be tailored to the target architecture.
 */
void mm_SIMD_OMP(const int m, const int n, const int p, const int n_end,const int el,
                        double* __restrict A, double* __restrict B, double* __restrict C){
    int i, j, k;
    double buffer[4];

    #pragma omp parallel for firstprivate(buffer) num_threads(4)
    for (i = 0; i < m; ++i) {
        for (j = 0; j < p; ++j) {
            __m256d dst1 = _mm256_set1_pd(0.0);
            for (k = 0; k < n_end; k+= el) {
                __m256d A_values1 = _mm256_loadu_pd(&A[i * n + k]);
                __m256d B_values1 = _mm256_loadu_pd(&B[k + j * n]);
                dst1 += _mm256_mul_pd(A_values1, B_values1);
            }

            __m128d dst_a = _mm256_extractf128_pd(dst1, 0);
            __m128d dst_b = _mm256_extractf128_pd(dst1, 1);

            _mm_storel_pd(&buffer[0], dst_a);
            _mm_storeh_pd(&buffer[1], dst_a);
            _mm_storel_pd(&buffer[2], dst_b);
            _mm_storeh_pd(&buffer[3], dst_b);
            for (int b = 0; b < 4; ++b) {
                C[i * n + j] += buffer[b];
            }
        }
    }
};

