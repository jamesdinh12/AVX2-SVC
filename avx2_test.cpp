#include <immintrin.h>
#include <iostream>
#include <chrono>

// AVX2 function
void avx2_add(int* a, int* b, int* result, int size) {
    for (int i = 0; i < size; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i vr = _mm256_add_epi32(va, vb);
        _mm256_storeu_si256((__m256i*)&result[i], vr);
    }
}

// Non-AVX2 function
void regular_add(int* a, int* b, int* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

int main() {
    const int size = 1000000;
    int* a = new int[size];
    int* b = new int[size];
    int* result_avx2 = new int[size];
    int* result_regular = new int[size];

    // Initialize arrays
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Measure AVX2 performance
    auto start = std::chrono::high_resolution_clock::now();
    avx2_add(a, b, result_avx2, size);
    auto end = std::chrono::high_resolution_clock::now();
    auto avx2_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Measure regular performance
    start = std::chrono::high_resolution_clock::now();
    regular_add(a, b, result_regular, size);
    end = std::chrono::high_resolution_clock::now();
    auto regular_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "AVX2 time: " << avx2_duration.count() << " microseconds\n";
    std::cout << "Regular time: " << regular_duration.count() << " microseconds\n";

    // Clean up
    delete[] a;
    delete[] b;
    delete[] result_avx2;
    delete[] result_regular;

    return 0;
}

