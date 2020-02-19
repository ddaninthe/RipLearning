#pragma once

struct RBF {
	double weight;
	double* coordinates;
	int dimensions;
};

extern "C" {
	__declspec(dllexport) double* createRBFModel(double* dataset, int datasetSize, int dimensions);
}