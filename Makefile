BASEDIR :=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
INC-DIR := "-I$(BASEDIR)"

CC := g++
CFLAGS := -c -std=c++11 -O3 -funroll-all-loops  -Wno-deprecated -D NO_VISUALIZATION
CUDA-FLAGS   = -ptx -std=c++11

MACHINE := rtx


ifeq ($(MACHINE),rtx)
  # anka-rtx
  TARGET := cuda-flow3d-rtx
  CUDA-TOP     = /home/ws/fe0968/local/cuda-10.0
  CUDA         = $(CUDA-TOP)/bin/nvcc 
  CUDA-INC-DIR = -I$(CUDA-TOP)/include 
  CUDA-LIB-DIR = -L$(CUDA-TOP)/lib64 -lcudart -lcuda  
else
  # ankaimage-concert
  TARGET := cuda-flow3d-concert
  CUDA         = /usr/bin/nvcc 
  CUDA-INC-DIR = -I/usr/include
  CUDA-LIB-DIR = -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda 
endif


SRCS := $(filter-out $(BASEDIR)/src/visualization.cpp, $(wildcard $(BASEDIR)/src/*.cpp \
			$(BASEDIR)/src/*/*/*.cpp \
			$(BASEDIR)/src/*/*.cpp))



#SRCS = $(filter-out src/visualization.cpp src/visualization.h, $(SRCS))

OBJS := $(patsubst %.cpp,%.o,$(SRCS))

CUDAOBJECTS 	= $(CUDASOURCES:.cu=.ptx)

#OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS) cuda
	$(CC) $(OBJS) $(CUDA-LIB-DIR) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR)  $< -o $@

cuda: 
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/add_3d.cu -o $(BASEDIR)/kernels/add_3d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/median_3d.cu -o $(BASEDIR)/kernels/median_3d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/convolution_3d.cu -o $(BASEDIR)/kernels/convolution_3d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/registration_3d.cu -o $(BASEDIR)/kernels/registration_3d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/registration_p_3d.cu -o $(BASEDIR)/kernels/registration_p_3d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/resample_3d.cu -o $(BASEDIR)/kernels/resample_3d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/resample_p_3d.cu -o $(BASEDIR)/kernels/resample_p_3d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/solve_3d.cu -o $(BASEDIR)/kernels/solve_3d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/solve_p_3d.cu -o $(BASEDIR)/kernels/solve_p_3d.ptx

clean:
	rm -rf $(TARGET) $(OBJS) $(CUDAOBJECTS)
	
.PHONY: all clean
