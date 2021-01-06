#pragma once
#include <vector>
#include <random>

template<int dim>
class PointsGenerator
{
	const float bound = 100.0f;
	std::default_random_engine generator{ 1 };
	std::uniform_real_distribution<float> distribution{-bound, bound};

	std::vector<float> generatePoint();

public:
	PointsGenerator() {};
	std::vector<std::vector<float>> generatePointsDevice(int N);
	std::vector<std::vector<float>> generateCentroidsHost(int k);
	std::vector<std::vector<float>> soaToAos(const std::vector<std::vector<float>> & points);
	std::vector<std::vector<float>> aosToSoa(const std::vector<std::vector<float>> & points);
};


template<int dim>
std::vector<float> PointsGenerator<dim>::generatePoint()
{
	std::vector<float> point;
	for (size_t i = 0; i < dim; i++)
	{
		point.push_back(distribution(generator));
	}
	return point;
}

template<int dim>
std::vector<std::vector<float>> PointsGenerator<dim>::generatePointsDevice(int count)
{
	std::vector<std::vector<float>> points;
	for (size_t i = 0; i < dim; i++)
		points.push_back(std::vector<float>());

	for (size_t i = 0; i < count; i++)
	{
		auto point = generatePoint();
		for (size_t j = 0; j < dim; j++)
			points[j].push_back(point[j]);
	}
	return points;
}

template<int dim>
std::vector<std::vector<float>> PointsGenerator<dim>::generateCentroidsHost(int k)
{
	std::vector<std::vector<float>> points;
	for (size_t i = 0; i < k; i++)
	{
		points.push_back(generatePoint());
	}
	return points;
}

template<int dim>
std::vector<std::vector<float>> PointsGenerator<dim>::soaToAos(const std::vector<std::vector<float>>& points)
{
	std::vector<std::vector<float>> output;
	std::vector<float> point;

	for (size_t i = 0; i < points[0].size(); i++)
	{
		point.clear();
		for (size_t j = 0; j < dim; j++)
		{
			point.push_back(points[j][i]);
		}
		output.push_back(point);
	}

	return output;
}

template<int dim>
std::vector<std::vector<float>> PointsGenerator<dim>::aosToSoa(const std::vector<std::vector<float>>& points)
{
	std::vector<std::vector<float>> output;
	for (size_t i = 0; i < dim; i++)
		output.push_back(std::vector<float>());

	for (size_t i = 0; i < points.size(); i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			output[j].push_back(points[i][j]);
		}
	}

	return output;
}
