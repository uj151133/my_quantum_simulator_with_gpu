SRC = myadd.cpp # source files
OBJ = $(SRC:.cpp=.o) # object files

CXX = g++ # compiler
PYTHON_INCLUDE_DIR := $(shell python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
PYTHON_LIB_DIR := $(shell python3 -c "from sysconfig import get_paths as gp; print(gp()['platlib'])")
PYBIND11_INCLUDE_DIR := $(PYTHON_LIB_DIR)/pybind11/include
CXXFLAGS = -Wall -std=c++11 -I$(PYTHON_INCLUDE_DIR) -I$(PYBIND11_INCLUDE_DIR) # compile options (flags)


main: $(OBJ)
	$(CXX) $(CXXFLAGS) -o main $(OBJ) # command to generate exe file

# How the object files are related
myadd.o: myadd.cpp

# clean
.PHONY: clean
clean:
	rm -f $(OBJ) main


.PHONY: install
install:
	pip install -r requirements.txt
