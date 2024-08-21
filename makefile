CXX = g++
CXXFLAGS = -mavx2 -O3 -std=c++11

all: avx2_test

avx2_test: avx2_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f avx2_test
