#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>

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
        __m256 sin_result = _mm256_sin_ps(dot);
        
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