#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>

// AVX2 approximate sine function
inline __m256 avx2_sin_ps(__m256 x) {
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 sin_mask = _mm256_set1_ps(1.27323954f);
    __m256 cos_mask = _mm256_set1_ps(0.405284735f);
    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);

    __m256 abs_x = _mm256_andnot_ps(sign_mask, x);
    __m256 y = _mm256_mul_ps(x, _mm256_set1_ps(0.15915494309189535f)); // x * (1/2pi)
    y = _mm256_sub_ps(y, _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT));

    __m256 sign = _mm256_and_ps(x, sign_mask);
    __m256 c = _mm256_andnot_ps(sign_mask, y);

    __m256 res = _mm256_fmadd_ps(sin_mask, c, _mm256_mul_ps(cos_mask, _mm256_mul_ps(c, _mm256_sub_ps(one, c))));
    res = _mm256_xor_ps(res, sign);

    return res;
}

// AVX2 function
void avx2_complex_op(float* a, float* b, float* result, int size) {
    const __m256 scale = _mm256_set1_ps(0.1f);
    for (int i = 0; i < size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        
        // Dot product
        __m256 mul = _mm256_mul_ps(va, vb);
        __m256 dot = _mm256_hadd_ps(mul, mul);
        dot = _mm256_hadd_ps(dot, dot);
        
        // Scale the result
        dot = _mm256_mul_ps(dot, scale);
        
        // Apply sine function
        __m256 sin_result = avx2_sin_ps(dot);
        
        _mm256_storeu_ps(&result[i], sin_result);
    }
}

// Non-AVX2 function
void regular_complex_op(float* a, float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        float dot = 0;
        for (int j = 0; j < 8; j++) {
            dot += a[i+j] * b[i+j];
        }
        result[i] = std::sin(dot * 0.1f);
    }
}

int main() {
    const int size = 10000000;
    float* a = new float[size];
    float* b = new float[size];
    float* result_avx2 = new float[size];
    float* result_regular = new float[size];

    // Initialize arrays with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < size; i++) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    // Measure AVX2 performance
    auto start = std::chrono::high_resolution_clock::now();
    avx2_complex_op(a, b, result_avx2, size);
    auto end = std::chrono::high_resolution_clock::now();
    auto avx2_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Measure regular performance
    start = std::chrono::high_resolution_clock::now();
    regular_complex_op(a, b, result_regular, size);
    end = std::chrono::high_resolution_clock::now();
    auto regular_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "AVX2 time: " << avx2_duration.count() << " microseconds\n";
    std::cout << "Regular time: " << regular_duration.count() << " microseconds\n";

    // Verify results
    float max_diff = 0;
    for (int i = 0; i < size; i++) {
        max_diff = std::max(max_diff, std::abs(result_avx2[i] - result_regular[i]));
    }
    std::cout << "Max difference: " << max_diff << std::endl;

    // Clean up
    delete[] a;
    delete[] b;
    delete[] result_avx2;
    delete[] result_regular;

    return 0;
}