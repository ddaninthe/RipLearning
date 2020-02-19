#include "RBF.h"
#include "pch.h"


RBF* createRBFModel(double* dataset, int datasetSize, int dimensions) {
	RBF* model = new RBF[datasetSize];
	for (int i = 0; i < datasetSize; i++) {
		double* coor = new double[dimensions];
		for (int c = 0; c < dimensions; c++) {
			coor[c] = dataset[dimensions * i + c];
		}
		model[i].coordinates = coor;
		model[i].weight = ((rand() / (double) RAND_MAX) - 0.5) * 2;
	}

	return model;
}

void trainNaiveRBF() {

}

int predictRBFClassification(RBF* model, int gamma, double* inputs, int dimensions, int modelSize) {
	return predictRBFRegression(model, gamma, inputs, dimensions, modelSize) >= 0 ? 1 : -1;
}

double predictRBFRegression(RBF* model, int gamma, double* inputs, int dimensions, int modelSize) {
	double res = 0;
	for (int i = 0; i < modelSize; i++) {
		double exponent = -gamma * squareMagnitude(inputs, model[i].coordinates, dimensions);
		res += model[i].weight * exp(exponent);
	}
	return res;
}

double squareMagnitude(double* a, double* b, int dim) {
	double magn = 0;
	for (int i = 0; i < dim; i++) {
		magn += pow(a[i] - b[i], 2);
	}
	return magn;
}