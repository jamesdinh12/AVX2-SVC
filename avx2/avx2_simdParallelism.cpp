#include <immintrin.h>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

#define ARRAY_SIZE 1024

// Helper function to sum elements of a 512-bit double vector
double m512d_sum(__m512d v) {
    double sum = 0;
    double* ptr = reinterpret_cast<double*>(&v);
    for (int i = 0; i < 8; i++) {
        sum += ptr[i];
    }
    return sum;
}

void perform_computations(double* a, double* b, double* c, int32_t* d, float* e) {
    __m512d va = _mm512_load_pd(a);
    __m512d vb = _mm512_load_pd(b);
    __m512d vc = _mm512_load_pd(c);
    __m256 ve = _mm256_load_ps(e);

    __mmask8 mask = _mm512_cmpneq_pd_mask(va, _mm512_set1_pd(0.0));

    __m512d vadd = _mm512_mask_add_pd(va, mask, va, vb);
    __m512d vfma = _mm512_mask3_fmadd_pd(va, vb, vc, mask);

    __m512d vmax = _mm512_max_pd(va, vb);
    __m512d vmin = _mm512_min_pd(va, vb);
    __m512d vmul = _mm512_mul_pd(va, vb);
    __m512d vcvt = _mm512_cvt_roundps_pd(ve, _MM_FROUND_NO_EXC);
    __m256i vcvti = _mm512_cvt_roundpd_epi32(vc, _MM_FROUND_NO_EXC);
    __m512d vfmadd = _mm512_fmadd_pd(va, vb, vc);
    __m512d vsub = _mm512_sub_pd(va, vb);
    __m512d vcast = _mm512_castsi512_pd(_mm512_castpd_si512(vc));
    __m512i vcvt64 = _mm512_cvtpd_epi64(vc);

    double sum = m512d_sum(vc);
    __m512d vset = _mm512_set1_pd(sum);

    // Store results back to memory
    _mm512_store_pd(a, vadd);
    _mm512_store_pd(b, vfma);
    _mm512_store_pd(c, vset);
}

int main() {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-100.0, 100.0);

    // Initialize arrays
    alignas(64) double a[ARRAY_SIZE];
    alignas(64) double b[ARRAY_SIZE];
    alignas(64) double c[ARRAY_SIZE];
    alignas(32) int32_t d[ARRAY_SIZE];
    alignas(32) float e[ARRAY_SIZE];

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start + std::chrono::seconds(1);

    int iterations = 0;

    while (std::chrono::high_resolution_clock::now() < end) {
        // Fill arrays with random values
        for (int i = 0; i < ARRAY_SIZE; i++) {
            a[i] = dis(gen);
            b[i] = dis(gen);
            c[i] = dis(gen);
            d[i] = static_cast<int32_t>(dis(gen));
            e[i] = static_cast<float>(dis(gen));
        }

        perform_computations(a, b, c, d, e);
        iterations++;
    }

    std::cout << "Script was run " << iterations << " times in one second." << std::endl;

    return 0;
}