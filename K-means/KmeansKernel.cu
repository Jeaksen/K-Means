
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PointsGenerator.h"
#include "Stopwatch.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <cmath>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>

const unsigned long N = 10000000UL;
const unsigned int DIM = 3;
const double THRESHOLD = 0.05;
const unsigned int K = 5;
const bool SHOULD_DEBUG = false;
const int SHARED_MEM_SIZE = 24576;
const int MEMBERSHIP_SHARED_MEM_SIZE = SHARED_MEM_SIZE / sizeof(short);

void displayPointsDevice(const thrust::host_vector<float>& points)
{
    unsigned long pointsCount = points.size() / DIM;
    for (size_t i = 0; i < pointsCount; i++)
    {
        for (size_t j = 0; j < DIM; j++)
        {
            std::cout << std::setw(9) << std::setprecision(6) << points[i + j * pointsCount];
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


__device__ float distance(const Point<DIM>& p1, const Point<DIM>& p2)
{
    float sum = 0.0f;
    for (size_t i = 0; i < DIM; i++)
        sum += pow(p1[i] - p2[i], 2.0f);

    return sqrt(sum);
}


__global__ void assignPointToCentroid(const float *points, const float *centroids, short* membership, short* delta)
{
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIndex > N)
        return;
    Point<DIM> point{ points + threadIndex, N };

    float minDist = INT_MAX, tempDist;
    short centroidIndex = -1;
    for (short j = 0; j < K; j++)
    {
        tempDist = distance(point, Point<DIM>(centroids + j, K));
        if (minDist > tempDist)
        {
            minDist = tempDist;
            centroidIndex = j;
        }
    }

    if (membership[threadIndex] != centroidIndex)
    {
        membership[threadIndex] = centroidIndex;
        delta[threadIndex] = 1;
    }
}

__global__ void assignPointToCentroidWithSegmentation(const float* points, const float* centroids, short* membership, float *output, short* delta)
{
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIndex > N)
        return;
    Point<DIM> point{ points + threadIndex, N };
    short prevCentroid = membership[threadIndex];
    float minDist = INT_MAX, tempDist;
    short centroidIndex = -1;
    for (short j = 0; j < K; j++)
    {
        tempDist = distance(point, Point<DIM>(centroids + j, K));
        if (minDist > tempDist)
        {
            minDist = tempDist;
            centroidIndex = j;
        }
    }

    if (prevCentroid != centroidIndex)
    {
        for (size_t i = 0; i < DIM; i++)
        {
            output[threadIndex + N * (DIM * prevCentroid + i)] = 0.0f;
            output[threadIndex + N * (DIM * centroidIndex + i)] = point[i];
        }

        membership[threadIndex] = centroidIndex;
        delta[threadIndex] = 1;
    }
}

__global__ void findNewCentroids(const float* points, const short* membership, float * newCentroids, const unsigned long * centroidsSizes)
{
    int threadIndex = threadIdx.x;
    int centroidIndex = threadIndex / DIM;
    unsigned long stride = threadIndex - DIM * centroidIndex;
    unsigned long size = centroidsSizes[centroidIndex];
    unsigned long sharedMemStride = blockDim.x;
    float center = 0;
    __shared__ short local_membership[MEMBERSHIP_SHARED_MEM_SIZE];

    size_t index = 0;
    while (index < N)
    {
        for (size_t i = threadIndex; i < MEMBERSHIP_SHARED_MEM_SIZE && i + index < N; i += sharedMemStride)
        {
            local_membership[i] = membership[index + i];
        }
        __syncthreads();
        for (size_t i = 0; i < MEMBERSHIP_SHARED_MEM_SIZE && i + index < N; i++, index++)
        {
            if (local_membership[i] == centroidIndex)
                center += points[index + N * stride] / size;
        }
    }
    //printf("index: %d  centroid: %d  stride: %d  size: %d  center: %f  output index: %d\n", threadIndex, centroidIndex, stride, size, center, stride * K + centroidIndex);

    newCentroids[stride * K + centroidIndex] = center;
}

__global__ void findNewCentroids2(const float* points, const short* membership, float* newCentroids, const unsigned long* centroidsSizes)
{
    int threadIndex = threadIdx.x;
    int centroidIndex = threadIndex / DIM;
    unsigned long stride = threadIndex - DIM * centroidIndex;
    unsigned long size = centroidsSizes[centroidIndex];
    float center = 0;

    for (size_t i = 0; i < N; i++)
    {
        if (membership[i] == centroidIndex)
            center += points[ i + N * stride] / size;
    }
    //printf("index: %d  centroid: %d  stride: %d  size: %d  center: %f  output index: %d\n", threadIndex, centroidIndex, stride, size, center, stride * K + centroidIndex);

    newCentroids[stride * K + centroidIndex] = center;
}


struct reduce_functor
{
    template <typename Tuple>
    __host__ __device__
        float operator()(Tuple t)
    {
        return thrust::get<1>(t) == thrust::get<2>(t) ? thrust::get<0>(t) : 0.0f;
    }
};

void lloyd_gpu(const thrust::device_vector<float>& points, thrust::device_vector<float> centroids)
{
    thrust::device_vector<short> membership{ N, 0 };
    thrust::device_vector<short> delta{ N, 0 };
    thrust::device_vector<unsigned long> centroidsSizes{ K };
    thrust::constant_iterator<unsigned long> ones(1);
    unsigned long deltaCount = N;
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t err;
    int threads = 1024;
    int blocks = ceil(N / (double)threads);
    cudaEventRecord(start, 0);

    while (deltaCount / (double)N > THRESHOLD)
    {
        if (SHOULD_DEBUG) 
        {
            std::cout << "Centroids: " << std::endl;
            displayPointsDevice(centroids);
        }

        assignPointToCentroid <<< blocks, threads >>> (thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(membership.data()), thrust::raw_pointer_cast(delta.data()));
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
        
        for (size_t i = 0; i < K; i++)
            centroidsSizes[i] = thrust::count(membership.begin(), membership.end(), i);

        findNewCentroids <<<1, DIM*K>>> (thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(membership.data()), thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(centroidsSizes.data()));

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
    displayPointsDevice(centroids);
    std::cout << "GPU time: " << time << " ms" << std::endl;

}

void lloyd_gpu2(const thrust::device_vector<float>& points, thrust::device_vector<float> centroids)
{
    thrust::device_vector<short> membership{ N, 0 };
    thrust::device_vector<short> delta{ N, 0 };
    thrust::device_vector<unsigned long> centroidsSizes{ K };
    thrust::constant_iterator<unsigned long> ones(1);
    unsigned long deltaCount = N;
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t err;
    int threads = 1024;
    int blocks = ceil(N / (double)threads);
    cudaEventRecord(start, 0);

    while (deltaCount / (double)N > THRESHOLD)
    {
        if (SHOULD_DEBUG)
        {
            std::cout << "Centroids: " << std::endl;
            displayPointsDevice(centroids);
        }

        assignPointToCentroid << < blocks, threads >> > (thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(membership.data()), thrust::raw_pointer_cast(delta.data()));
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

        for (size_t i = 0; i < K; i++)
            centroidsSizes[i] = thrust::count(membership.begin(), membership.end(), i);

        findNewCentroids2 << <1, DIM* K >> > (thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(membership.data()), thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(centroidsSizes.data()));


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
    displayPointsDevice(centroids);
    std::cout << "GPU time: " << time << " ms" << std::endl;

}

void lloyd_gpu3(const thrust::device_vector<float>& points, thrust::device_vector<float> centroids)
{
    thrust::device_vector<short> membership{ N, 0 };
    thrust::device_vector<short> delta{ N, 0 };
    thrust::device_vector<unsigned long> centroidsSizes{ K };
    thrust::constant_iterator<unsigned long> ones(1);
    unsigned long deltaCount = N;
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t err;
    int threads = 1024;
    int blocks = ceil(N / (double)threads);
    cudaEventRecord(start, 0);

    while (deltaCount / (double)N > THRESHOLD)
    {
        if (SHOULD_DEBUG)
        {
            std::cout << "Centroids: " << std::endl;
            displayPointsDevice(centroids);
        }

        assignPointToCentroid <<< blocks, threads >>> (thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(membership.data()), thrust::raw_pointer_cast(delta.data()));
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

        for (size_t i = 0; i < K; i++)
            centroidsSizes[i] = thrust::count(membership.begin(), membership.end(), i);

        for (size_t i = 0; i < K; i++)
        {
            auto centroidIdx = thrust::make_constant_iterator(i);
            for (size_t j = 0; j < DIM; j++)
            {
                auto first = thrust::make_zip_iterator(thrust::make_tuple(points.begin() + j * N, membership.begin(), centroidIdx));
                auto last = thrust::make_zip_iterator(thrust::make_tuple(points.begin() + (j + 1) * N, membership.end(), centroidIdx + N));
                centroids[i + j * K] = thrust::transform_reduce(first, last, reduce_functor(), 0.0f, thrust::plus<float>()) / centroidsSizes[i];
            }
        }

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
    displayPointsDevice(centroids);
    std::cout << "GPU3 time: " << time << " ms" << std::endl;

}

void lloyd_gpu4(const thrust::device_vector<float>& points, thrust::device_vector<float> centroids)
{
    thrust::device_vector<short> membership{ N, 0 };
    thrust::device_vector<short> delta{ N, 0 };
    thrust::device_vector<unsigned long> centroidsSizes{ K };
    thrust::constant_iterator<unsigned long> ones(1);
    thrust::device_vector<float> output{N * DIM * K, 0};
    unsigned long deltaCount = N;
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t err;
    int threads = 1024;
    int blocks = ceil(N / (double)threads);
    cudaEventRecord(start, 0);

    while (deltaCount / (double)N > THRESHOLD)
    {
        if (SHOULD_DEBUG)
        {
            std::cout << "Centroids: " << std::endl;
            displayPointsDevice(centroids);
        }

        assignPointToCentroidWithSegmentation <<< blocks, threads >>> (thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(centroids.data()), 
                                                                       thrust::raw_pointer_cast(membership.data()), thrust::raw_pointer_cast(output.data()),
                                                                       thrust::raw_pointer_cast(delta.data()));
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

        for (size_t i = 0; i < K; i++)
            centroidsSizes[i] = thrust::count(membership.begin(), membership.end(), i);

        for (size_t i = 0; i < K; i++)
        {
            for (size_t j = 0; j < DIM; j++)
            {
                centroids[i + j * K] = thrust::reduce(output.begin() + N * (DIM * i + j), output.begin() + N * (DIM * i + j + 1)) / centroidsSizes[i];
            }
        }

        findNewCentroids2 << <1, DIM* K >> > (thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(membership.data()), thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(centroidsSizes.data()));


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
    displayPointsDevice(centroids);
    std::cout << "GPU4 time: " << time << " ms" << std::endl;

}

// Calcuates the Euclidean distance between two points
float distance(const std::vector<float>& p1, const std::vector<float>& p2)
{
    double sum = 0;
    for (size_t i = 0; i < DIM; i++)
        sum += pow(double(p2[i] - p1[i]), 2.0);

    return sqrt(sum);
}

void lloyd_cpu(const std::vector<std::vector<float>>& points, std::vector<std::vector<float>> centroids)
{
    unsigned long delta = N;
    short* membership = new short[N] {};
    unsigned long centroidsSizes[K] {};
    float newCentroids[K][DIM] {};

    if (SHOULD_DEBUG) displayPointsHost(centroids);

    while (delta / (float)N > THRESHOLD)
    {
        delta = 0;
        for (size_t i = 0; i < N; i++)
        {
            float minDist = INT_MAX, tempDist;
            short index = -1;
            for (short j = 0; j < K; j++)
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
        }

        std::cout << "delta: " << delta / (double)N << std::endl;
        if (SHOULD_DEBUG) displayPointsHost(centroids);

        for (size_t i = 0; i < K; i++)
        {
            for (size_t j = 0; j < DIM; j++)
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
    displayPointsHost(centroids);

    delete[] membership;
}

struct prg
{
    float a, b;

    __host__ __device__
        prg(float _a = 0.f, float _b = 1.f) : a(_a), b(_b) {};

    __host__ __device__
        float operator()(const unsigned int n) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(n);

        return dist(rng);
    }
};


int main()
{
    Stopwatch stopwatch;
    PointsGenerator<DIM> gen;

    std::cout << "Generating points ";
    stopwatch.Start();
    auto d_points = gen.generatePointsDevice(N);
    stopwatch.Stop();
    std::cout << "Generating centroids ";
    stopwatch.Start();
    auto d_centroids = gen.generateCentroidsDevice(K);
    stopwatch.Stop();

    // Device number generator!
    //stopwatch.Start();
    //thrust::host_vector<float> h_points(N*DIM);
    //stopwatch.Stop();
    //stopwatch.Start();
    //thrust::device_vector<float> d_points = h_points;
    //stopwatch.Stop();
    ////thrust::device_vector<float> d_centroids(K*DIM);
    //stopwatch.Start();

    //thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    //stopwatch.Stop();

    //stopwatch.Start();
    //thrust::transform(index_sequence_begin,
    //    index_sequence_begin + DIM*N,
    //    d_points.begin(),
    //    prg(-100, 100));
    ////thrust::transform(index_sequence_begin,
    ////    index_sequence_begin + DIM * K,
    ////    d_centroids.begin(),
    ////    prg(-100, 100));
    //stopwatch.Stop();

    ////if (SHOULD_DEBUG) displayPointsHost(h_points);
    std::cout << "Copying points ";
    stopwatch.Start();
    auto h_points = gen.deviceToHost(d_points);
    auto h_centroids = gen.deviceToHost(d_centroids);
    stopwatch.Stop();

    //std::cout << std::endl << "LLoyd CPU ";
    //stopwatch.Start();
    //lloyd_cpu(h_points, h_centroids);
    //stopwatch.Stop();

    std::cout << std::endl << "LLoyd GPU1" << std::endl << std::endl;
    lloyd_gpu(d_points, d_centroids);

    std::cout << std::endl << "LLoyd GPU2" << std::endl << std::endl;
    lloyd_gpu2(d_points, d_centroids);

    //std::cout << std::endl << "LLoyd GPU3" << std::endl << std::endl;
    //lloyd_gpu3(d_points, d_centroids);

    std::cout << std::endl << "LLoyd GPU4" << std::endl << std::endl;
    lloyd_gpu4(d_points, d_centroids);


    return 0;
}
