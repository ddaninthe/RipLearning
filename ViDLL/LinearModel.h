#pragma once

extern "C" {
	typedef struct LinearModel {
		double* weights;
		double a;
		double b;
		bool isLinear;
	} LinearModel;

	__declspec(dllexport) LinearModel* createLinearModel(int nbInputs);
	__declspec(dllexport) void trainLinearClassification(double* dataset, int datasetSize, double* expectedOutputs, LinearModel* model, int modelSize, double nbIter, double learning);
	__declspec(dllexport) void trainLinearRegression(double* dataset, int datasetSize, double* expectedOutputs, LinearModel* model, int modelSize);
	__declspec(dllexport) int predictLinearClassification(LinearModel* ptr, int size, double* inputs);
	__declspec(dllexport) double predictLinearRegression(LinearModel* ptr, int size, double* inputs);
	__declspec(dllexport) void clear(void* ptr);
}

bool equals(double a, double b);
double* solve(double x1, double y1, double x2, double y2);