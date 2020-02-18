#pragma once

#include "LinearModel.h"
#include "pch.h"


double* createLinearModel(int nbInputs) {
	auto model = new double[nbInputs + 1];
	for (int i = 0; i < nbInputs; i++) {
		model[i] = ((double)(rand() % 100 - 50)) / 100.f;
	}
	return model;
}

double* trainLinearClassification(double* dataset, int dataSize, double* model, int modelSize, double iterNumber, double learning) {
	int random;
	for (int i = 0; i < iterNumber; i++) {
		random = rand() % dataSize;
		double *data = dataset + random * (modelSize + 1);

		double g = predictLinearClassification(model, modelSize, data);

		double modif = learning * (data[modelSize] - g);

		for (int k = 0; k < modelSize; k++) {
			model[k] += modif * model[k];
		}
	}
	return model;
}

double* trainLinearRegression(double* dataset, int dataSize, double* model, int modelSize, double iterNumber, double learning) {
	int random;
	for (int i = 0; i < iterNumber; i++) {
		random = rand() % dataSize;
		double *data = dataset + random * (modelSize + 1);

		double g = predictLinearRegression(model, modelSize, data);

		double modif = learning * (data[modelSize] - g);

		for (int k = 0; k < modelSize; k++) {
			model[k] += modif * model[k];
		}
	}
	return model;
}


double predictLinearClassification(double* model, int size, double *inputs) {
	return predictLinearRegression(model, size, inputs) >= 0 ? 1 : -1;
}

double predictLinearRegression(double* model, int size, double* inputs) {
	double res = 0;
	for (int i = 0; i < size; i++) {
		res += model[i] * inputs[i];
	}
	return res;
}

void clear(double* ptr) {
	delete[] ptr;
}