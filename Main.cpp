/////////////////////////////////////////////////////////////////////////////////          
// Forward-compatible core GL 4.3 version 
//
// 
// 
//
//
#include <cmath>
#include <fstream>
#include <iostream>
#include "GameEngine.h"

using namespace std;
using namespace glm;

GameEngine engine;

void listDevices()
{
    int* count = new int;

    cudaError_t deviceStatus = cudaGetDeviceCount(count);
    if (*count == 0) {
        std::cout << "No CUDA devices found" << std::endl;
        delete count;
        return;
    }

    cudaDeviceProp* deviceProperties = new cudaDeviceProp;

    for (int i = 0; i < *count; i++) {
        deviceStatus = cudaGetDeviceProperties(deviceProperties, i);
        if (deviceStatus != cudaSuccess) {
            std::cout << "An error occurred in accessing properties of device " << i << std::endl;
            delete count;
            return;
        }

        std::cout << "The properties of device " << i << " are:" << std::endl;
        std::cout << "Name: " << deviceProperties->name << std::endl;
        std::cout << "Core Clock Rate: " << deviceProperties->clockRate / 1000 << " Mhz" << std::endl;
        std::cout << "Number of Compute Units: " << deviceProperties->multiProcessorCount << std::endl;
        std::cout << "Max Threads per Compute Unit: " << deviceProperties->maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Warp Size: " << deviceProperties->warpSize << std::endl;
        std::cout << "Total Memory: " << deviceProperties->totalGlobalMem / 1048576 << "MB" << std::endl;
        std::cout << "Level 2 Cache Size: " << deviceProperties->l2CacheSize / 1024 << "KB" << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    delete count;
    return;
}


// Routine to output interaction instructions to the C++ window.
void printInteraction(void)
{
    listDevices();
	engine.Initialise();
}

// Main routine.
int main(int argc, char** argv)
{
    //GlobalGozdillaPosition(); //MOVE TO UPDATE

	printInteraction();

	engine.InitEngine(argc, argv, "", 500, 500);
		
	engine.StartEngine();

    //cudaDeviceReset must be called before exiting in order for profiling and
    //tracing tools such as Nsight and Visual Profiler to show complete traces.
    GameEngine::cudaStatus = cudaDeviceReset();
    if (GameEngine::cudaStatus != cudaSuccess)
    {
	    fprintf(stderr,"cudaDeviceReset failed!");
	    return 1;
    }

	return 0;
}