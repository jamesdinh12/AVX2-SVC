#include <iostream>
#include <immintrin.h>
#include <cmath>

int main() {
    // Initialize some data
    double a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double b[8] = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double c[8] = {0.0};
    double d[8] = {0.0};
    double e[8] = {0.0};

    // Load data into vectors
    __m512d vec_a = _mm512_loadu_pd(a);
    __m512d vec_b = _mm512_loadu_pd(b);
    __m512d vec_c = _mm512_setzero_pd();
    __m512d vec_d = _mm512_setzero_pd();
    __m512d vec_e = _mm512_setzero_pd();

    // Perform vector operations
    vec_c = _mm512_add_pd(vec_a, vec_b);  // Addition
    vec_d = _mm512_sub_pd(vec_a, vec_b);  // Subtraction
    vec_e = _mm512_mul_pd(vec_a, vec_b);  // Multiplication

    // Use masking and blending
    __mmask8 mask = _mm512_cmpneq_pd_mask(vec_a, _mm512_set1_pd(4.0));  // Create a mask based on condition vec_a != 4.0
    vec_c = _mm512_mask_add_pd(vec_c, mask, vec_a, vec_b);  // Masked addition

    // Fused Multiply-Add (FMA)
   vec_d = _mm512_mask3_fmadd_pd(vec_a, vec_b, vec_d, mask);  // Masked FMA: vec_d = (vec_a * vec_b) + vec_d

    // Store results back to arrays
    _mm512_storeu_pd(c, vec_c);
    _mm512_storeu_pd(d, vec_d);
    _mm512_storeu_pd(e, vec_e);

    // Print results
    std::cout << "a = ";
    for (int i = 0; i < 8; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << "\nb = ";
    for (int i = 0; i < 8; ++i) {
        std::cout << b[i] << " ";
    }
    std::cout << "\nc = a + b = ";
    for (int i = 0; i < 8; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << "\nd = a - b = ";
    for (int i = 0; i < 8; ++i) {
        std::cout << d[i] << " ";
    }
    std::cout << "\ne = a * b = ";
    for (int i = 0; i < 8; ++i) {
        std::cout << e[i] << " ";
    }
    std::cout << std::endl;

    //Print execution time
    std::cout << "Execution time: " << clock() / (double)CLOCKS_PER_SEC << " seconds" << std::endl;

    return 0;
}