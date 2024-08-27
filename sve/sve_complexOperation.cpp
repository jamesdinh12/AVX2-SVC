#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <arm_sve.h>

const int ARRAY_SIZE = 1000000;
const float SCALE_FACTOR = 0.5f;

void regular_operation(std::vector<float>& a, std::vector<float>& b, std::vector<float>& result) {
    float dot_product = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        dot_product += a[i] * b[i];
    }
    
    float scaled_result = dot_product * SCALE_FACTOR;
    
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        result[i] = std::sin(scaled_result * a[i]);
    }
}

void sve_operation(std::vector<float>& a, std::vector<float>& b, std::vector<float>& result) {
    svfloat32_t vdot = svdup_n_f32(0.0f);
    
    for (int i = 0; i < ARRAY_SIZE; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, ARRAY_SIZE);
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vb = svld1(pg, &b[i]);
        vdot = svmla_f32_z(pg, vdot, va, vb);
    }
    
    float dot_product = svaddv_f32(svptrue_b32(), vdot);
    float scaled_result = dot_product * SCALE_FACTOR;
    
    svfloat32_t vscaled = svdup_n_f32(scaled_result);
    
    for (int i = 0; i < ARRAY_SIZE; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, ARRAY_SIZE);
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vresult = svmul_f32_z(pg, vscaled, va);
        // Replace SVE sine with scalar sine
        svst1(pg, &result[i], vresult);
    }

    // Apply sine function separately
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        result[i] = std::sin(result[i]);
    }
}

int main() {
    std::vector<float> a(ARRAY_SIZE);
    std::vector<float> b(ARRAY_SIZE);
    std::vector<float> result_regular(ARRAY_SIZE);
    std::vector<float> result_sve(ARRAY_SIZE);

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        a[i] = static_cast<float>(i) / ARRAY_SIZE;
        b[i] = static_cast<float>(ARRAY_SIZE - i) / ARRAY_SIZE;
    }

    auto start = std::chrono::high_resolution_clock::now();
    regular_operation(a, b, result_regular);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_regular = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    sve_operation(a, b, result_sve);
    end = std::chrono::high_resolution_clock::now();
    auto duration_sve = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Regular operation time: " << duration_regular.count() << " microseconds\n";
    std::cout << "SVE operation time: " << duration_sve.count() << " microseconds\n";

    return 0;
}