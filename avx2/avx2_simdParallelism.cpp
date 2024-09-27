#include <immintrin.h>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

#define ARRAY_SIZE 1024

// Helper function to print a 512-bit double vector
void print_vector(__m512d v) {
    double* ptr = reinterpret_cast<double*>(&v);
    std::cout << "[";
    for (int i = 0; i < 8; i++) {
        std::cout << std::setprecision(3) << ptr[i] << ", ";
    }
    std::cout << "]\n";
}

// Helper function to print a 256-bit integer vector
void print_vector(__m256i v) {
    int32_t* ptr = reinterpret_cast<int32_t*>(&v);
    std::cout << "[";
    for (int i = 0; i < 8; i++) {
        std::cout << ptr[i] << ", ";
    }
    std::cout << "]\n";
}

// Helper function to print a 512-bit integer vector
void print_vector(__m512i v) {
    int64_t* ptr = reinterpret_cast<int64_t*>(&v);
    std::cout << "[";
    for (int i = 0; i < 8; i++) {
        std::cout << ptr[i] << ", ";
    }
    std::cout << "]\n";
}

// Helper function to sum elements of a 512-bit double vector
double m512d_sum(__m512d v) {
    double sum = 0;
    double* ptr = reinterpret_cast<double*>(&v);
    for (int i = 0; i < 8; i++) {
        sum += ptr[i];
    }
    return sum;
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

    // Fill arrays with random values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = dis(gen);
        b[i] = dis(gen);
        c[i] = dis(gen);
        d[i] = static_cast<int32_t>(dis(gen));
        e[i] = static_cast<float>(dis(gen));
    }

    // Load data into vectors
    __m512d va = _mm512_load_pd(a);
    __m512d vb = _mm512_load_pd(b);
    __m512d vc = _mm512_load_pd(c);
    __m256 ve = _mm256_load_ps(e);

    // Compute mask
    __mmask8 mask = _mm512_cmpneq_pd_mask(va, _mm512_set1_pd(0.0));

    // AAD computations
    __m512d vadd = _mm512_mask_add_pd(va, mask, va, vb);
    __m512d vfma = _mm512_mask3_fmadd_pd(va, vb, vc, mask, _mm512_set1_pd(0.5));

    // Fast Exp computations
    __m512d vmax = _mm512_max_pd(va, vb);
    __m512d vmin = _mm512_min_pd(va, vb);
    __m512d vmul = _mm512_mul_pd(va, vb);
    __m512d vcvt = _mm512_cvt_roundps_pd(ve, _MM_FROUND_NO_EXC);
    __m256i vcvti = _mm512_cvt_roundpd_epi32(vc, _MM_FROUND_NO_EXC);
    __m512d vfmadd = _mm512_fmadd_pd(va, vb, vc);
    __m512d vsub = _mm512_sub_pd(va, vb);
    __m512d vcast = _mm512_castsi512_pd(_mm512_castpd_si512(vc));
    __m512i vcvt64 = _mm512_cvtpd_epi64(vc);

    // State Fill computations
    double sum = m512d_sum(vc);
    __m512d vset = _mm512_set1_pd(sum);

    // Print results
    std::cout << "Input vectors:\n";
    print_vector(va);
    print_vector(vb);
    print_vector(vc);

    std::cout << "\nAAD results:\n";
    print_vector(vadd);
    print_vector(vfma);

    std::cout << "\nFast Exp results:\n";
    print_vector(vmax);
    print_vector(vmin);
    print_vector(vmul);
    print_vector(vcvt);
    std::cout << "vcvti: ";
    print_vector(vcvti);
    print_vector(vfmadd);
    print_vector(vsub);
    print_vector(vcast);
    std::cout << "vcvt64: ";
    print_vector(vcvt64);

    std::cout << "\nState Fill result:\n";
    print_vector(vset);

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform computations (you can add more operations here)
    for (int i = 0; i < 1000; i++) {
        vadd = _mm512_mask_add_pd(va, mask, va, vb);
        vfma = _mm512_mask3_fmadd_pd(va, vb, vc, mask, _mm512_set1_pd(0.5));
        vmax = _mm512_max_pd(va, vb);
        vmin = _mm512_min_pd(va, vb);
        vmul = _mm512_mul_pd(va, vb);
        vcvt = _mm512_cvt_roundps_pd(ve, _MM_FROUND_NO_EXC);
        vcvti = _mm512_cvt_roundpd_epi32(vc, _MM_FROUND_NO_EXC);
        vfmadd = _mm512_fmadd_pd(va, vb, vc);
        vsub = _mm512_sub_pd(va, vb);
        vcast = _mm512_castsi512_pd(_mm512_castpd_si512(vc));
        vcvt64 = _mm512_cvtpd_epi64(vc);
        sum = m512d_sum(vc);
        vset = _mm512_set1_pd(sum);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "\nExecution time: " << duration << " nanoseconds\n";

    return 0;
}