#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <vector>
using namespace std;


__global__ void AddInts(int*a,int*b,int count){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id<count){
		a[id] += b[id];
	}
}


int main()
{
	int count = 100000;
	//vector<int> a(count,1);
	//vector<int> b(count,1);

	int a[count];
	int b[count];

	for (int i=0; i<count; i++) {
		a[i] = rand()%1000;

		b[i] = rand()%1000;
	}

	cout << "Prior to addition: " << endl;

	for (int i=0; i<5; i++) {
		cout << a[i] << " " << b[i] << endl;
	}

	int *d_a,*d_b;

	// Initialize the GPU memory
	if (cudaMalloc(&d_a,sizeof(int)*count)!=cudaSuccess) {
		cout << "Malloc failed" << endl;
		cudaFree(d_a);
		return -1;
	}
	if (cudaMalloc(&d_b,sizeof(int)*count)!=cudaSuccess) {
		cout << "Malloc failed" << endl;
		cudaFree(d_b);
		return -1;
	}

	// COpy to GPU
	if (cudaMemcpy(d_a,&a,sizeof(int)*count,cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not copy!" << endl;
		cudaFree(d_a);
		cudaFree(d_b);
		return -1;
	}
	if (cudaMemcpy(d_b,&b,sizeof(int)*count,cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not copy!" << endl;
		cudaFree(d_a);
		cudaFree(d_b);
		return -1;
	}

	AddInts<<<count/256+1,256>>>(d_a,d_b,count);

	if (cudaMemcpy(&a,d_a,sizeof(int)*count,cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout << "Could not fetch!" << endl;
		cudaFree(d_a);
		cudaFree(d_b);
		return -1;
	}
	if (cudaMemcpy(&b,d_b,sizeof(int)*count,cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout << "Could not fetch!" << endl;
		cudaFree(d_a);
		cudaFree(d_b);
		return -1;
	}


	cout << "After addition: " << endl;
	for (int i=0; i<5; i++) {
		cout << a[i] << " " << b[i] << endl;
	}

	cudaFree(d_a);
	cudaFree(d_b);


	




}
