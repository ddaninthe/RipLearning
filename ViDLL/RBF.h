#pragma once

extern "C" {
	struct RBF {
		double weight;
		double* coordinates;
	};

	__declspec(dllexport) RBF* createRBFModel(double* dataset, int datasetSize, int dimensions);
	__declspec(dllexport) int predictRBFClassification(RBF* model, int gamma, double* inputs, int dimensions, int modelSize);
	__declspec(dllexport) double predictRBFRegression(RBF* model, int gamma, double* inputs, int dimensions, int modelSize);
}

double squareMagnitude(double* a, double* b, int dim);