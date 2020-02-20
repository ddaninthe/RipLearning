#pragma once

extern "C" {
	typedef struct MLP {
		int* layers;
		int layersCount;
		double*** weights;
		double** deltas;
		double** x;
	} MLP;

	__declspec(dllexport) MLP* createMLPModel(int* layout, int layoutSize);
	__declspec(dllexport) void trainMLPClassification(MLP* model, double* dataset, double* expectedOutputs, int datasetSize, int iterations, double alpha);
	__declspec(dllexport) double* predictMLPClassification(MLP* model, double* inputs);
	__declspec(dllexport) double* predictMLPRegression(MLP* model, double* inputs);
}