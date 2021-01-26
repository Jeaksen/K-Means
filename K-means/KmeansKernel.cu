
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PointsGenerator.h"
#include "Stopwatch.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <cmath>
#include <thrust/count.h>
#include <thrust/fill.h>

const unsigned long N = 1000000UL;
//const unsigned int dim = 3;
const double THRESHOLD = 0.01;
const unsigned int K = 5;
const bool SHOULD_DEBUG = false;
const int SHARED_MEM_SIZE = 24576;
const int MEMBERSHIP_SHARED_MEM_SIZE = SHARED_MEM_SIZE / sizeof(short);

template <unsigned int dim>
void displayPointsDevice(const thrust::host_vector<float>& points)
{
    unsigned long pointsCount = points.size() / dim;
    for (size_t i = 0; i < pointsCount; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            std::cout << std::setw(9) << std::setprecision(6) << points[i + j * pointsCount];
        }
        std::cout << std::endl;
    }
}

template <unsigned int dim>
void displayPointsHost(const std::vector<std::vector<float>>& points)
{
    for (size_t i = 0; i < points.size(); i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            std::cout << std::setw(9) << std::setprecision(6) << points[i][j];
        }
        std::cout << std::endl;
    }
}

template <unsigned int dim>
struct Point
{
    const float* start_ptr;
    unsigned long stride;
    __device__ Point(const float* start, const unsigned long stride)
    {
        this->start_ptr = start;
        this->stride = stride;
    }
    __device__ float operator [] (unsigned long index) const
    {
        return start_ptr[index * stride];
    }
};

template<int dim>
__device__ float distance(const Point<dim>& p1, const Point<dim>& p2)
{
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++)
        sum += pow(p1[i] - p2[i], 2.0f);

    return sqrt(sum);
}

template<int dim>
__device__ short findNewCentroid(const Point<dim>& point, const float* centroids)
{
    float minDist = INT_MAX, tempDist;
    short centroidIndex = -1;
    for (short j = 0; j < K; j++)
    {
        tempDist = distance(point, Point<dim>(centroids + j, K));
        if (minDist > tempDist)
        {
            minDist = tempDist;
            centroidIndex = j;
        }
    }

    return centroidIndex;
}

template<int dim>
__global__ void assignPointToCentroid(const float *points, const float *centroids, short* membership, short* delta)
{
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIndex > N)
        return;

    Point<dim> point{ points + threadIndex, N };

    short centroidIndex = findNewCentroid(point, centroids);

    if (membership[threadIndex] != centroidIndex)
    {
        membership[threadIndex] = centroidIndex;
        delta[threadIndex] = 1;
    }
}

template<int dim>
__global__ void assignPointToCentroidWithSegmentation(const float* points, const float* centroids, short* membership, float *output, short* delta)
{
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIndex > N)
        return;

    Point<dim> point{ points + threadIndex, N };

    short prevCentroid = membership[threadIndex];
    short newCentroid = findNewCentroid(point, centroids);

    if (prevCentroid != newCentroid)
    {
        if (prevCentroid == -1)
            for (size_t i = 0; i < dim; i++)
                output[threadIndex + N * (dim * newCentroid + i)] = point[i];
        else
            for (size_t i = 0; i < dim; i++)
            {
                output[threadIndex + N * (dim * prevCentroid + i)] = 0.0f;
                output[threadIndex + N * (dim * newCentroid + i)] = point[i];
            }

        membership[threadIndex] = newCentroid;
        delta[threadIndex] = 1;
    }
}

template<int dim>
__global__ void findNewCentroids(const float* points, const short* membership, float * newCentroids, const unsigned long * centroidsSizes)
{
    __shared__ short local_membership[MEMBERSHIP_SHARED_MEM_SIZE];
    int threadIndex = threadIdx.x;
    int centroidIndex = threadIndex / dim;
    unsigned long stride = threadIndex - dim * centroidIndex;
    unsigned long size = centroidsSizes[centroidIndex];
    unsigned long sharedMemStride = blockDim.x;
    float center = 0;

    size_t index = 0;
    while (index < N)
    {
        for (size_t i = threadIndex; i < MEMBERSHIP_SHARED_MEM_SIZE && i + index < N; i += sharedMemStride)
            local_membership[i] = membership[index + i];
        __syncthreads();

        for (size_t i = 0; i < MEMBERSHIP_SHARED_MEM_SIZE && i + index < N; i++, index++)
            if (local_membership[i] == centroidIndex)
                center += points[index + N * stride] / size;
    }

    newCentroids[stride * K + centroidIndex] = center;
}

template<int dim>
void lloyd_gpu_kernel(const thrust::device_vector<float>& points, thrust::device_vector<float> centroids)
{
    thrust::device_vector<short> membership{ N, 0 };
    thrust::device_vector<short> delta{ N, 0 };
    thrust::device_vector<unsigned long> centroidsSizes{ K };

    unsigned long deltaCount = N;
    int threads = 1024;
    int blocks = ceil(N / (double)threads);
    float time;

    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    while (deltaCount / (double)N > THRESHOLD)
    {
        if (SHOULD_DEBUG) 
        {
            std::cout << "Centroids: " << std::endl;
            displayPointsDevice<dim>(centroids);
        }

        assignPointToCentroid<dim> <<< blocks, threads >>> (thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(membership.data()), thrust::raw_pointer_cast(delta.data()));
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));


        for (size_t i = 0; i < K; i++)
            centroidsSizes[i] = thrust::count(membership.begin(), membership.end(), i);

        findNewCentroids<dim> <<<1, dim*K>>> (thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(membership.data()), 
                                         thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(centroidsSizes.data()));
        

        deltaCount = thrust::count(delta.begin(), delta.end(), 1);
        thrust::fill(delta.begin(), delta.end(), 0);


        std::cout << "delta: " << deltaCount / (double)N << std::endl;
        if (SHOULD_DEBUG)
        {
            for (size_t i = 0; i < K ; i++)
                std::cout << centroidsSizes[i] << " ";
            std::cout << std::endl << std::endl;
            std::cout << "Membership:" << std::endl;
            for (size_t i = 0; i < N; i++)
                std::cout << membership[i] << " ";
            std::cout << std::endl;
        }
        
    }
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    displayPointsDevice<dim>(centroids);
    std::cout << "GPU-Kernel time: " << time << " ms" << std::endl;
}

template<int dim>
void lloyd_gpu_reduce(const thrust::device_vector<float>& points, thrust::device_vector<float> centroids)
{
    thrust::device_vector<short> membership{ N, -1 };
    thrust::device_vector<short> delta{ N, 0 };
    thrust::device_vector<unsigned long> centroidsSizes{ K };
    thrust::device_vector<float> output{N * dim * K, 0};

    int threads = 1024;
    int blocks = ceil(N / (double)threads);
    unsigned long deltaCount = N;

    float time;
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    while (deltaCount / (double)N > THRESHOLD)
    {
        if (SHOULD_DEBUG)
        {
            std::cout << "Centroids: " << std::endl;
            displayPointsDevice<dim>(centroids);
        }
        assignPointToCentroidWithSegmentation<dim> <<< blocks, threads >>> (thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(centroids.data()), 
                                                                       thrust::raw_pointer_cast(membership.data()), thrust::raw_pointer_cast(output.data()),
                                                                       thrust::raw_pointer_cast(delta.data()));
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));


        for (size_t i = 0; i < K; i++)
            centroidsSizes[i] = thrust::count(membership.begin(), membership.end(), i);

        for (size_t centroid = 0; centroid < K; centroid++)
            for (size_t dimension = 0; dimension < dim; dimension++)
                centroids[centroid + dimension * K] = thrust::reduce(output.begin() + N * (dim * centroid + dimension), output.begin() + N * (dim * centroid + dimension + 1)) / centroidsSizes[centroid];
        

        deltaCount = thrust::count(delta.begin(), delta.end(), 1);
        thrust::fill(delta.begin(), delta.end(), 0);


        std::cout << "delta: " << deltaCount / (double)N << std::endl;
        if (SHOULD_DEBUG)
        {
            for (size_t i = 0; i < K; i++)
                std::cout << centroidsSizes[i] << " ";
            std::cout << std::endl << std::endl;
            std::cout << "Membership:" << std::endl;
            for (size_t i = 0; i < N; i++)
                std::cout << membership[i] << " ";
            std::cout << std::endl;
        }
    }
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    displayPointsDevice<dim>(centroids);
    std::cout << "GPU-Reduce time: " << time << " ms" << std::endl;
}

// Calcuates the Euclidean distance between two points
template <unsigned int dim>
float distance(const std::vector<float>& p1, const std::vector<float>& p2)
{
    double sum = 0;
    for (size_t i = 0; i < dim; i++)
        sum += pow(double(p2[i] - p1[i]), 2.0);

    return sqrt(sum);
}

template <unsigned int dim>
void lloyd_cpu(const std::vector<std::vector<float>>& points, std::vector<std::vector<float>> centroids)
{
    unsigned long delta = N;
    short* membership = new short[N] {};
    unsigned long centroidsSizes[K] {};
    float newCentroids[K][dim] {};

    if (SHOULD_DEBUG) displayPointsHost<dim>(centroids);

    while (delta / (float)N > THRESHOLD)
    {
        delta = 0;
        for (size_t i = 0; i < N; i++)
        {
            float minDist = INT_MAX, tempDist;
            short index = -1;
            for (short j = 0; j < K; j++)
            {
                tempDist = distance<dim>(points[i], centroids[j]);
                if (minDist > tempDist)
                {
                    minDist = tempDist;
                    index = j;
                }
            }
            centroidsSizes[index]++;
            for (size_t j = 0; j < dim; j++)
                newCentroids[index][j] += points[i][j];

            if (membership[i] != index)
            {
                membership[i] = index;
                delta++;
            }
        }

        std::cout << "delta: " << delta / (double)N << std::endl;
        if (SHOULD_DEBUG) displayPointsHost<dim>(centroids);

        for (size_t i = 0; i < K; i++)
        {
            for (size_t j = 0; j < dim; j++)
            {
                centroids[i][j] = newCentroids[i][j] / centroidsSizes[i];
                newCentroids[i][j] = 0;
            }
            centroidsSizes[i] = 0;
        }
     }

    if (SHOULD_DEBUG)
    {
        std::cout << "Membership:" << std::endl;
        for (size_t i = 0; i < N; i++)
            std::cout << membership[i] << " ";
        std::cout << std::endl;
    }
    displayPointsHost<dim>(centroids);

    delete[] membership;
}


int main()
{
    unsigned const int dim = 3;
    Stopwatch stopwatch;
    PointsGenerator<dim> gen;
    
    std::cout << "Generating points ";
    stopwatch.Start();
    auto d_points = gen.generatePointsDevice(N);
    stopwatch.Stop();

    std::cout << "Generating centroids ";
    stopwatch.Start();
    auto d_centroids = gen.generateCentroidsDevice(K);
    stopwatch.Stop();

    std::cout << "Copying points ";
    stopwatch.Start();
    auto h_points = gen.deviceToHost(d_points);
    auto h_centroids = gen.deviceToHost(d_centroids);
    stopwatch.Stop();

    std::cout << std::endl << "LLoyd CPU ";
    stopwatch.Start();
    lloyd_cpu<dim>(h_points, h_centroids);
    stopwatch.Stop();

    std::cout << std::endl << "LLoyd GPU Kernel" << std::endl;
    lloyd_gpu_kernel<dim>(d_points, d_centroids);

    std::cout << std::endl << "LLoyd GPU Reduce" << std::endl;
    lloyd_gpu_reduce<dim>(d_points, d_centroids);

    return 0;
}
