#pragma once
#include <vector>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include "Stopwatch.h"

/*
* This class provides functionality for generating sets of points. The points come from a uniform distribution [-bound, bound].
* It supports arbitrary amount of dimensions.
* Two ways of generating are possible:
* - as a host vector of point vectors
* - as a device vector which is a 1D representation of structure of arrays. Coefficients from same dimension are 
* placed next to each other
* 
* The generator also provides functionality to convert those types of data representation
*/
template<unsigned int dim>
class PointsGenerator
{

	const float bound;
	std::default_random_engine generator{ std::chrono::system_clock::now().time_since_epoch().count() };
	std::uniform_real_distribution<float> distribution{-bound, bound};

	std::vector<float> generatePoint();
	
public:
	PointsGenerator(float bound = 100.0f): bound(bound) {};
	// Generate N points as a vector of points
	std::vector<std::vector<float>> generatePointsHost(unsigned long N);
	// Generate K centroids as a vector of points
	std::vector<std::vector<float>> generateCentroidsHost(int k);
	// Generate N points as an structure of arrays flatened to 1D host vector
	thrust::device_vector<float> generatePointsDevice(unsigned long N);
	// Generate K centroids as an structure of arrays flatened to 1D host vector
	thrust::device_vector<float> generateCentroidsDevice(int k);
	// Convert points from structure of arrays form to vector of points
	std::vector<std::vector<float>> deviceToHost(const thrust::device_vector<float>& points);
	// Convert points form vector of points to structure of arrays
	thrust::device_vector<float> hostToDevice(const std::vector<std::vector<float>> & points);
};


template<unsigned int dim>
std::vector<float> PointsGenerator<dim>::generatePoint()
{
	std::vector<float> point;
	for (size_t i = 0; i < dim; i++)
	{
		point.push_back(distribution(generator));
	}
	return point;
}

template<unsigned int dim>
thrust::device_vector<float> PointsGenerator<dim>::generatePointsDevice(unsigned long count)
{
	Stopwatch stopwatch;
	thrust::host_vector<float> points{ ((unsigned long)dim) * count };

	for (size_t i = 0; i < count; i++)
	{
		auto point = generatePoint();
		for (size_t j = 0; j < dim; j++)
			points[j * count + i] = point[j];
	}
	return points;
}

template<unsigned int dim>
std::vector<std::vector<float>>  PointsGenerator<dim>::generatePointsHost(unsigned long count)
{
	std::vector<std::vector<float>> points;

	for (size_t i = 0; i < count; i++)
		points.push_back(generatePoint());

	return points;
}

template<unsigned int dim>
std::vector<std::vector<float>> PointsGenerator<dim>::generateCentroidsHost(int k)
{
	return generatePointsHost(k);
}

template<unsigned int dim>
thrust::device_vector<float> PointsGenerator<dim>::generateCentroidsDevice(int k)
{
	return generatePointsDevice(k);
}

template<unsigned int dim>
std::vector<std::vector<float>> PointsGenerator<dim>::deviceToHost(const thrust::device_vector<float> & points)
{
	thrust::host_vector<float> h_points{ points };
	unsigned long size = h_points.size() / (unsigned long)dim;
	std::vector<std::vector<float>> output;
	auto point = std::vector<float> (dim);
	
	for (size_t i = 0; i < size; i++)
	{
		for (size_t j = 0; j < dim; j++)
			point[j] = h_points[i + j * size];

		output.push_back(point);
	}

	return output;
}

template<unsigned int dim>
thrust::device_vector<float> PointsGenerator<dim>::hostToDevice(const std::vector<std::vector<float>> & points)
{
	unsigned long size = points.size();
	thrust::host_vector<float> output{ ((unsigned long)dim) * size };

	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < dim; j++)
			output[j * size + i] = points[i][j];

	return output;
}
