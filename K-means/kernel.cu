
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PointsGenerator.h"
#include "Stopwatch.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <cmath>

#define DIM 3
#define N 100000
#define THRESHOLD 0.1
#define K 5

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


std::vector<std::vector<float>> SoaToAos(std::vector<std::vector<float>> points)
{
    std::vector<std::vector<float>> output;
    std::vector<float> point;

    for (size_t i = 0; i < N; i++)
    {
        point.clear();
        for (size_t j = 0; j < DIM; j++)
        {
            point.push_back(points[j][i]);
        }
        output.push_back(point);
    }

    return output;
}

float distance(std::vector<float> p1, std::vector<float> p2)
{
    double sum = 0;
    for (size_t i = 0; i < DIM; i++)
    {
        sum += pow(double(p2[i] - p1[i]), 2.0);
    }
    return sqrt(sum);
}


void lloyd_cpu(std::vector<std::vector<float>> points, PointsGenerator<DIM> generator)
{
    int delta = N;
    int membership[N];
    int centroidsSizes[K];
    std::vector<std::vector<float>> newCentroids;
    auto centroids = generator.generateCentroidsHost(K);

    while (delta / (float)N > THRESHOLD)
    {
        delta = 0;
        for (size_t i = 0; i < N; i++)
        {
            float minDist = INT_MAX, tempDist;
            int index = -1;
            for (size_t j = 0; j < DIM; j++)
            {
                tempDist = distance(points[i], centroids[j]);
                if (minDist > tempDist)
                {
                    minDist = tempDist;
                    index = j;
                }
            }
            if (membership[i] != index)
            {
                membership[i] = index;
                centroidsSizes[index]++;
                delta++;
                for (size_t j = 0; j < DIM; j++)
                {
                    newCentroids[index][j] += points[i][j];
                }
            }
        }

        for (size_t i = 0; i < K; i++)
        {
            for (size_t j = 0; j < DIM; j++)
            {
                centroids[i][j] = newCentroids[i][j] / centroidsSizes[i];
            }
        }
    }

}

int main()
{
    Stopwatch stopwatch;
    PointsGenerator<DIM> gen;

    stopwatch.Start();
    auto points = gen.generatePointsDevice(N);
    stopwatch.Stop();

    auto h_points = SoaToAos(points);

    //for (size_t i = 0; i < N; i++)
    //{
    //    for (size_t j = 0; j < DIM; j++)
    //    {
    //        std::cout << std::setw(9) << std::setprecision(6) << points[j][i];
    //    }
    //    std::cout << std::endl;
    //}
    return 0;
}
