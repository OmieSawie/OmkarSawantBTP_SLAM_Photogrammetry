//#include <__clang_cuda_builtin_vars.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void vectorAdd(int *a,int *b,int *c )
{
	int i = threadIdx.x; 
	c[i] = a[i]+b[i];
	return;
}
int main()
{
	int a[3] = {5,2,3};
	int b[3] = {4,5,6};
	int c[sizeof(a)/sizeof(int)] = {0,0,0};
	//for (int i=0 ; i<sizeof(c); i++) {
	//	c[i] = a[i] + b[i];
	//}

	int *cudaA=0;
	int *cudaB=0;
	int *cudaC=0;

	cudaMalloc(&cudaA,sizeof(a));
	cudaMalloc(&cudaB,sizeof(b));
	cudaMalloc(&cudaC,sizeof(c));
	
	cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);
	//cudaMemcpy(cudaC, c, sizeof(c), cudaMemcpyHostToDevice);

	vectorAdd <<< 1,sizeof(a)/sizeof(int) >>> (cudaA,cudaB,cudaC);

	cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);
	for (int i : c) {
		std::cout << i << " " ;
	}
	
}
