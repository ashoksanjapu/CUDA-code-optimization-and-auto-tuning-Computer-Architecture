NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

sum_reduction_gpu: sum_reduction_gpu.cu
	$(NVCC) $(NVCC_FLAGS) sum_reduction_gpu.cu -o sum_reduction_gpu

clean:
	rm -f sum_reduction_gpu
