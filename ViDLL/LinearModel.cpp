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

/**
 * @tparam dataset the array containing data
 * @tparam dataSize the size of the data
 * @tparam model the linear model created, with the bias
 * @tparam modelSize the size of the model, minus the bias/predict
 * @tparam nbIter the number of iteration
 */
double* trainLinearClassification(double* dataset, int dataSize, double* model, int modelSize, double nbIter, double learning) {
	int random;
	for (int i = 0; i < nbIter; i++) {
		random = rand() % dataSize;
		double *data = dataset + random * (modelSize + 1);

		double g = predictLinearClassification(model, modelSize, data);

		double modif = learning * (data[modelSize] - g);

		for (int k = 0; k < modelSize + 1; k++) {
			model[k] += modif * model[k];
		}
	}
	return model;
}


/**
 * @tparam dataset the array containing data
 * @tparam dataSize the size of the data
 * @tparam model the linear model created, with the bias
 * @tparam modelSize the size of the model, minus the bias/predict
 * @tparam nbIter the number of iteration
 */
double* trainLinearRegression(double* dataset, int dataSize, double* model, int modelSize, double nbIter, double learning) {
	int random;
	for (int i = 0; i < nbIter; i++) {
		random = rand() % dataSize;
		double *data = dataset + random * (modelSize + 1);

		double g = predictLinearRegression(model, modelSize, data);

		double modif = learning * (data[modelSize] - g);

		for (int k = 0; k < modelSize + 1; k++) {
			model[k] += modif * model[k];
		}
	}
	return model;
}


double predictLinearClassification(double* model, int size, double *inputs) {
	return predictLinearRegression(model, size, inputs) >= 0 ? 1 : -1;
}

double predictLinearRegression(double* model, int size, double* inputs) {
	double res = model[0];
	for (int i = 0; i < size; i++) {
		res += model[i+1] * inputs[i+1];
	}
	return res;
}

void clear(double* ptr) {
	delete[] ptr;
}