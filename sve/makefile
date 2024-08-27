# Compiler settings
CXX = g++
CXXFLAGS = -march=armv8-a+sve -O3 -std=c++11

# Source file
SOURCES = sve_simpleAddition.cpp sve_complexOperation.cpp

# Object file
OBJECTS = $(SOURCES:.cpp=.o)

# Executable
EXECUTABLES = sve_simpleAddition sve_complexOperation

# Default target
all: $(EXECUTABLE)

# Rule for building executable
$(EXECUTABLE): $(OBJECT)
	$(CXX) $(CXXFLAGS) $< -o $@

# Rule for building object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJECT) $(EXECUTABLE)

# Phony targets
.PHONY: all clean