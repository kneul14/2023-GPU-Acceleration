#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GameObject.h"

#include <stdio.h>
#include <iostream>


enum GrassState 
{
	startingStateAlive,
	startingStateDead,
	aliveWith2or3Neighbors,
	deadWith3Neighbors,
	otherAliveCells,
	otherDeadCells,
};

class Cuda 
{
public:
	cudaError_t CudaSetup();
	cudaError_t UpdateStateCuda(GrassState* currentStates, int* aliveCounts, int sizeOfArray);

};