
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PointsGenerator.h"
#include "Stopwatch.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <cmath>


const unsigned long N = 30000000;
#define DIM 3
#define THRESHOLD 0.01
#define K 5
#define MY_DEBUG false

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}


void displayPointsDevice(const std::vector<std::vector<float>>& points)
{
    for (size_t i = 0; i < points[0].size(); i++)
    {
        for (size_t j = 0; j < points.size(); j++)
        {
            std::cout << std::setw(9) << std::setprecision(6) << points[j][i];
        }
        std::cout << std::endl;
    }
}

void displayPointsHost(const std::vector<std::vector<float>>& points)
{
    for (size_t i = 0; i < points.size(); i++)
    {
        for (size_t j = 0; j < points[0].size(); j++)
        {
            std::cout << std::setw(9) << std::setprecision(6) << points[i][j];
        }
        std::cout << std::endl;
    }
}


// Calcuates the Euclidean distance between two points
float distance(const std::vector<float>& p1, const std::vector<float>& p2)
{
    double sum = 0;
    for (size_t i = 0; i < DIM; i++)
        sum += pow(double(p2[i] - p1[i]), 2.0);

    return sqrt(sum);
}


void lloyd_cpu(const std::vector<std::vector<float>>& points, PointsGenerator<DIM>& generator)
{
    long delta = N;
    short* membership = new short[N] {};
    long centroidsSizes[K] {};
    float newCentroids[K][DIM] {};

    auto centroids = generator.generateCentroidsHost(K);
    if (MY_DEBUG) displayPointsHost(centroids);

    while (delta / (float)N > THRESHOLD)
    {
        delta = 0;
        for (size_t i = 0; i < N; i++)
        {
            float minDist = INT_MAX, tempDist;
            short index = -1;
            for (size_t j = 0; j < K; j++)
            {
                tempDist = distance(points[i], centroids[j]);
                if (minDist > tempDist)
                {
                    minDist = tempDist;
                    index = j;
                }
            }
            centroidsSizes[index]++;
            for (size_t j = 0; j < DIM; j++)
                newCentroids[index][j] += points[i][j];

            if (membership[i] != index)
            {
                membership[i] = index;
                delta++;
            }
            if (membership[i] == -1)
                std::cout << "wtf";
        }
        //std::cout << "Membership:" << std::endl;
        //for (size_t i = 0; i < N; i++)
        //    std::cout << membership[i] << " ";
        //std::cout << std::endl;

        //std::cout << "centroidsSizes:" << std::endl;
        //for (size_t i = 0; i < K; i++)
        //    std::cout << centroidsSizes[i] << " ";
        //std::cout << std::endl;

        //std::cout << "newCentroids:" << std::endl;
        //for (size_t i = 0; i < K; i++)
        //{
        //    for (size_t j = 0; j < DIM; j++)
        //    {
        //        std::cout << newCentroids[i][j] << " ";
        //    }
        //    std::cout << std::endl;
        //}

        for (size_t i = 0; i < K; i++)
        {
            for (size_t j = 0; j < DIM; j++)
            {
                centroids[i][j] = newCentroids[i][j] / centroidsSizes[i];
                newCentroids[i][j] = 0;
            }
            centroidsSizes[i] = 0;
        }
        
        std::cout << "delta: ";
        std::cout << delta / (float)N << std::endl;
        if(MY_DEBUG) displayPointsHost(centroids);
    }
    if (MY_DEBUG)
    {
        std::cout << "Membership:" << std::endl;
        for (size_t i = 0; i < N; i++)
            std::cout << membership[i] << " ";
        std::cout << std::endl;
    }
    displayPointsHost(centroids);

    delete[] membership;
}

int main()
{
    Stopwatch stopwatch;
    PointsGenerator<DIM> gen;

    stopwatch.Start();
    auto points = gen.generatePointsDevice(N);
    stopwatch.Stop();
    if (MY_DEBUG) displayPointsDevice(points);


    auto h_points = gen.soaToAos(points);
    stopwatch.Start();
    lloyd_cpu(h_points, gen);
    stopwatch.Stop();
    return 0;
}
