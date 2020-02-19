#pragma once

extern "C" {
	__declspec(dllexport) double* createLinearModel(int nbInputs);
	__declspec(dllexport) void trainLinearClassification(double* dataset, int datasetSize, double* expectedOutputs, double* model, int modelSize, double nbIter, double learning);
	__declspec(dllexport) void trainLinearRegression(double* dataset, int datasetSize, double* expectedOutputs, double* model, int modelSize);
	__declspec(dllexport) int predictLinearClassification(double* ptr, int size, double* inputs);
	__declspec(dllexport) double predictLinearRegression(double* ptr, int size, double* inputs);
	__declspec(dllexport) void clear(double* ptr);
}