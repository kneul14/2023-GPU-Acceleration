#include "Cuda.cuh"

__global__ void UpdateStateKernel(GrassState* currentStates, int* aliveCounts) {
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx < numCells) {
    //    UpdateState(currentStates[idx], aliveCounts[idx]);
    //}

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    GrassState currentState = currentStates[idx];
    int aliveCount = aliveCounts[idx];
    
    switch (currentState) {
    case startingStateAlive:
        if (!(aliveCount == 2 || aliveCount == 3)) {
            currentState = otherAliveCells;
        }
        else {
            currentState = aliveWith2or3Neighbors;
        }
        break;
    case startingStateDead:
        if (aliveCount < 3) {
            currentState = otherDeadCells;

        }
        else {
            currentState = deadWith3Neighbors;
        }
        break;
    default:
        break;
    }
    
    currentStates[idx] = currentState;    

}

cudaError_t Cuda::CudaSetup()
{
    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//Error:
//
    return cudaStatus;

}

int minimum(int a, int b) {
    return (a < b) ? a : b;
}

cudaError_t Cuda::UpdateStateCuda(GrassState* currentStates, int* aliveCounts, int sizeOfArray)
{
     // Allocates memory for currentStates and aliveCounts on the device
     GrassState* devCurrentStates;
     int* devAliveCounts;
    
     cudaError_t cudaStatus;
    
     cudaStatus = cudaMalloc((void**)&devCurrentStates, sizeOfArray * sizeof(GrassState));
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaMalloc failed!");
         return cudaStatus;
     }
    
     cudaStatus = cudaMalloc((void**)&devAliveCounts, sizeOfArray * sizeof(int));
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaMalloc failed!");
         cudaFree(devCurrentStates);
         return cudaStatus;
     }
    
     // Copy data from host to device
     cudaStatus = cudaMemcpy(devCurrentStates, currentStates, sizeOfArray * sizeof(GrassState), cudaMemcpyHostToDevice);
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaMemcpy failed!");
         cudaFree(devCurrentStates);
         cudaFree(devAliveCounts);
         return cudaStatus;
     }
    
     // Copy data from host to device for aliveCounts
     cudaStatus = cudaMemcpy(devAliveCounts, aliveCounts, sizeOfArray * sizeof(int), cudaMemcpyHostToDevice);
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaMemcpy failed!");
         cudaFree(devCurrentStates);
         cudaFree(devAliveCounts);
         return cudaStatus;
     }
    
     //int minimumGridSize;
     //int gridSize;
     //int blockSize;


     //cudaOccupancyMaxPotentialBlockSize(&minimumGridSize, &blockSize, UpdateStateKernel, 0, sizeOfArray);
     //gridSize = (sizeOfArray + blockSize - 1) / blockSize;


     // Get device properties to determine optimal block size
     cudaDeviceProp deviceProp;
     cudaGetDeviceProperties(&deviceProp, 0); // Assuming device 0
     int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

     // Choose an optimal block size
     int sizeOfBlocks = minimum(maxThreadsPerBlock, 256); // Use the minimum function

     // Calculate the number of blocks
     int numOfBlocks = (sizeOfArray + sizeOfBlocks - 1) / sizeOfBlocks;
    
     ///Launch kernel
     UpdateStateKernel << <numOfBlocks, sizeOfBlocks >> > (devCurrentStates, devAliveCounts);

     // Check for any errors launching the kernel
     cudaStatus = cudaGetLastError();
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
         cudaFree(devCurrentStates);
         cudaFree(devAliveCounts);
         return cudaStatus;
     }
    
     // cudaDeviceSynchronize waits for the kernel to finish
     // Returns any errors encountered during the launch.
     cudaStatus = cudaDeviceSynchronize();
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
         cudaFree(devCurrentStates);
         cudaFree(devAliveCounts);
         return cudaStatus;
     }

       // Copy data from host to device
     cudaStatus = cudaMemcpy(currentStates,devCurrentStates, sizeOfArray * sizeof(GrassState), cudaMemcpyDeviceToHost);
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaMemcpy failed!");
         cudaFree(devCurrentStates);
         cudaFree(devAliveCounts);
         return cudaStatus;
     }
    
     cudaFree(devCurrentStates);
     cudaFree(devAliveCounts);
    
     return cudaStatus;
}