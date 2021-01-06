
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PointsGenerator.h"
#include "Stopwatch.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <cmath>


const unsigned long N = 2000000;
#define DIM 3
#define THRESHOLD 0.01
#define K 5
#define MY_DEBUG false


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


        for (size_t i = 0; i < K; i++)
        {
            for (size_t j = 0; j < DIM; j++)
            {
                centroids[i][j] = newCentroids[i][j] / centroidsSizes[i];
                newCentroids[i][j] = 0;
            }
            centroidsSizes[i] = 0;
        }
        
        std::cout << "delta: " << delta / (double)N << std::endl;
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
    auto h_points = gen.generatePointsHost(N);
    stopwatch.Stop();

    stopwatch.Start();
    auto dd_points = gen.generatePointsDevice(N);
    stopwatch.Stop();

    stopwatch.Start();
    auto d_points = gen.hostToDevice(h_points);
    stopwatch.Stop();

    if (MY_DEBUG) displayPointsHost(h_points);


    //stopwatch.Start();
    //lloyd_cpu(h_points, gen);
    //stopwatch.Stop();
    return 0;
}
