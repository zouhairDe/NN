CC = g++
NVCC = nvcc
# CFLAGS = -std=c++17 -O3 -Iinclude
NVCCFLAGS = -arch=sm_61 -O3 -Xcompiler -fPIC -Iinclude
# Add -lcurand for random number generation
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lz -lm

# Add zlib header path if needed
CFLAGS = -std=c++17 -O3 -Iinclude -I/usr/include

SRC = src/main.cpp src/data_loader.cpp src/utils.cpp
CU_SRC = src/convolution.cu
OBJ = $(SRC:.cpp=.o) $(CU_SRC:.cu=.o)
EXEC = mnist_cnn

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

src/convolution.o: src/convolution.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)

re:
	make clean
	make all


compile-test:
	$(CC) $(CFLAGS) src/test_model.cpp src/data_loader.cpp src/utils.cpp src/convolution.o -o test_model $(LDFLAGS)