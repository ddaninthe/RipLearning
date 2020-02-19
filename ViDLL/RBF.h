#pragma once

extern "C" {
	struct RBF {
		double* weights;
		double** coordinates;
	};

	__declspec(dllexport) RBF* createRBFModel(double* dataset, int datasetSize, int dimensions);
	__declspec(dllexport) void trainNaiveRBF(RBF* model, int datasetSize, double* expectedOutputs, int dimensions, int gamma);
	__declspec(dllexport) int predictRBFClassification(RBF* model, int gamma, double* inputs, int dimensions, int modelSize);
	__declspec(dllexport) double predictRBFRegression(RBF* model, int gamma, double* inputs, int dimensions, int modelSize);
}

double RBFExponent(int gamma, double* inputs, double* X, int dim);
double squareMagnitude(double* a, double* b, int dim);