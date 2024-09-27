CXX = g++
CXXFLAGS = -mavx2 -O3 -std=c++11

all: avx2_simpleAddition avx2_complexOperation avx2_simdParallelism

avx2_simpleAddition: avx2_simpleAddition.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

avx2_complexOperation: avx2_complexOperation.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

avx2_simdParallelism: avx2_simdParallelism.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f avx2_simpleAddition avx2_complexOperation avx2_simdParallelism
