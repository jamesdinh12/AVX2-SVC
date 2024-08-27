#include <iostream>
#include <vector>
#include <chrono>
#include <arm_sve.h>

const int ARRAY_SIZE = 10000000;

void regular_addition(std::vector<float>& a, std::vector<float>& b, std::vector<float>& result) {
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        result[i] = a[i] + b[i];
    }
}

void sve_addition(std::vector<float>& a, std::vector<float>& b, std::vector<float>& result) {
    for (int i = 0; i < ARRAY_SIZE; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, ARRAY_SIZE);
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vb = svld1(pg, &b[i]);
        svfloat32_t vresult = svadd_f32_z(pg, va, vb);
        svst1(pg, &result[i], vresult);
    }
}

int main() {
    std::vector<float> a(ARRAY_SIZE, 1.0f);
    std::vector<float> b(ARRAY_SIZE, 2.0f);
    std::vector<float> result_regular(ARRAY_SIZE);
    std::vector<float> result_sve(ARRAY_SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    regular_addition(a, b, result_regular);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_regular = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    sve_addition(a, b, result_sve);
    end = std::chrono::high_resolution_clock::now();
    auto duration_sve = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Regular addition time: " << duration_regular.count() << " microseconds\n";
    std::cout << "SVE addition time: " << duration_sve.count() << " microseconds\n";

    return 0;
}