# Compiler settings
CXX = g++
CXXFLAGS = -march=armv8-a+sve -O3 -std=c++11

# Source files
SOURCES = sve_simpleAddition.cpp sve_complexOperation.cpp

# Executables
EXECUTABLES = sve_simpleAddition sve_complexOperation

# Default target
all: $(EXECUTABLES)

# Rule for building executables
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Clean target
clean:
	rm -f $(EXECUTABLES)

# Phony targets
.PHONY: all clean